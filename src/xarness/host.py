"""The host process: runs sessions and serves the FastAPI app.

One host owns many :class:`~xarness.session.Session` objects. It serves the
API on a unix socket under the xarness data dir (mode 0600 in a 0700
directory), and ``--web`` adds a TCP listener for the browser UI.

**Lifecycle.** The host auto-starts (detached, survives the TUI) when the
first client needs it. It exits once no session is running, compacting, or
waiting on an ask — attached clients do not count — after a short grace
period. ``xarness --server`` runs it in the foreground in persistent mode
(never exits on its own); with a terminal it opens the live dashboard, and
piped it prints the stats.

**Hardening.** An exclusive startup lock keeps racing clients from
spawning two hosts (the loser retries connecting); a stale socket file
(connection refused) is removed and treated as "no host".
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__, protocol, session_store
from .config import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    SandboxSettings,
    SectionedConfig,
    ensure_config,
    load_sectioned,
    resolve_api_key,
    save_preferences,
)
from .gitwork import GitWorktreeError
from .protocol import ServerInfo, ServerStats, SessionSummary
from .sandbox import SandboxConfig, SandboxRuntime, SandboxUnavailable, ref_infos
from .errors import RefError, SessionError
from .session import Session, SessionParts
from .tools import report_tool_usage
from .workspace import (
    WorkspaceState,
    change_workspace_ref,
    has_workspace_state,
    load_workspace_refs,
    open_workspace,
)
from .util import parse_range, same_path, utcnow_iso as _utcnow

LOG = logging.getLogger(__name__)

HOST_DIR = Path("~/.local/share/xarness").expanduser()
SOCKET_PATH = HOST_DIR / "host.sock"
LOCK_PATH = HOST_DIR / "host.lock"

# How long the host lingers with no active session before exiting
# (auto-started hosts only; a foreground --server host is persistent).
GRACE_SECONDS = 5.0
# How long a fully idle session stays hosted before eviction. Evicting on
# every idle makes the next turn pay a full load (state read + tracking
# rebind + sandbox build) and kills the persistent shell; a short TTL keeps
# warm sessions cheap while still releasing resources. Override with
# XARNESS_IDLE_TTL_SECONDS (0 evicts immediately, as tests expect).
IDLE_TTL_SECONDS = 120.0


def _idle_ttl() -> float:
    return float(os.environ.get("XARNESS_IDLE_TTL_SECONDS", IDLE_TTL_SECONDS))

# How often the reaper wakes to evict inactive sessions and check for
# shutdown. Small enough that an idle session leaves promptly, cheap enough
# that the scan (one dict of sessions) never shows up in a profile.
REAP_INTERVAL_SECONDS = 0.25
# Trailing debounce for the cached session-list diff totals: a turn can run
# several mutating tools in a row, and one workspace snapshot/diff answers
# for every session in that repo.
STATS_DEBOUNCE_SECONDS = 0.4
# How long one pending diff answers for a saved session's own total. The
# listing never computes these; a hover fetches one session's row, and this
# cache keeps sweeping across several saved rows of one repo (or re-hovering)
# from running one worktree snapshot each time.
SAVED_STATS_TTL_SECONDS = 5.0

# Seconds a client waits for a freshly spawned host's socket to appear.
SPAWN_TIMEOUT = 15.0

# Defaults for the browser-facing listener (``xarness --web``). Loopback by
# default: reaching the agent's shell from the network is a deliberate choice.
DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8765


@dataclass(frozen=True)
class WebListen:
    """The TCP listener ``xarness --web`` adds beside the unix socket.

    ``token`` gates the ``/v1`` routes of that listener only; the unix-socket
    app the TUI uses never checks one."""

    bind: str = DEFAULT_WEB_HOST
    port: int = DEFAULT_WEB_PORT
    token: str | None = None

    @property
    def loopback(self) -> bool:
        return self.bind in ("127.0.0.1", "localhost", "::1")



class Host:
    """Owns sessions and the unix-socket listener."""

    def __init__(
        self,
        *,
        persistent: bool = False,
        config_path: Path = DEFAULT_CONFIG_PATH,
        unix_socket: Path = SOCKET_PATH,
    ) -> None:
        self.persistent = persistent
        self.config_path = config_path
        # A directly spawned host or --server must not fail on first run.
        try:
            ensure_config(config_path)
        except ConfigError:
            pass
        self.unix_socket = unix_socket
        self.sessions: dict[str, Session] = {}
        # One tracked state per repo root, shared by every session in it.
        self._workspace_states: dict[Path, WorkspaceState] = {}
        # One lock per session name: concurrent loads of the same saved
        # session (e.g. GET snapshot + GET events) must yield one Session.
        self._load_locks: dict[str, asyncio.Lock] = {}
        # Per-parent subagent counter, seeded from the stored names when the
        # parent first spawns in this host (so numbering survives replays).
        self._sub_counts: dict[str, int] = {}
        # Serializes the workspace map (opening a workspace takes awaits).
        self._workspace_lock = asyncio.Lock()
        self._started_at = time.monotonic()
        self._started_at_wall = _utcnow()
        self._shutdown_event = asyncio.Event()
        # The one background loop that evicts inactive sessions and decides
        # when a non-persistent host may exit (see _reap).
        self._reaper: asyncio.Task[None] | None = None
        # GET /v1/events subscribers: one queue per connected client. The
        # session list is pushed, not polled, so another session's state
        # reaches the UI as it changes.
        self._watchers: list[asyncio.Queue[Any]] = []
        # Workspace-row ids with a pending cached-stats refresh, and the one
        # debounce task per workspace (see _schedule_stats).
        self._stats_dirty: set[int] = set()
        self._stats_tasks: dict[int, asyncio.Task[None]] = {}
        # Repo root -> (monotonic time, diff) for the per-session summaries of
        # saved (unhosted) sessions. Short-lived on purpose: see
        # SAVED_STATS_TTL_SECONDS.
        self._saved_stats: dict[str, tuple[float, protocol.DiffInfo]] = {}

    # ------------------------------------------------------------------
    # session construction (config + workspace + sandbox)

    async def open_workspace(
        self, workspace: Path, *, allow_init: bool = True,
        persist: bool = True,
    ) -> WorkspaceState | None:
        """The tracked state of ``workspace``'s repo root, opened once and
        shared by every agent in it. None when it cannot be tracked."""
        root = workspace.resolve()
        async with self._workspace_lock:
            existing = self._workspace_states.get(root)
            if existing is None:
                opened = await open_workspace(
                    root, allow_init=allow_init, persist=persist,
                )
                if opened is None:
                    return None
                existing = self._workspace_states.setdefault(opened.root, opened)
            if persist:
                existing.enable_persistence()
            return existing

    def _workspace_last_used(self, workspace: Path) -> tuple[str | None, str | None]:
        """The most recently used (model, sandbox) among the saved sessions
        of this directory — the workspace's last choice. Reads only the
        small meta block, never the full conversation."""
        best = None
        for meta in session_store.load_metas().values():
            if not meta.workspace:
                continue
            try:
                same = Path(meta.workspace).resolve() == workspace.resolve()
            except OSError:
                continue
            if not same:
                continue
            if best is None or (meta.updated_at or "") > (best.updated_at or ""):
                best = meta
        return (best.model, best.sandbox) if best else (None, None)

    async def _build_parts(
        self,
        request: protocol.CreateSessionRequest,
        *,
        session_name: str,
        saved_sandbox_name: str | None = None,
        saved_sandbox_settings: "SandboxSettings | None" = None,
    ) -> SessionParts:
        """Everything Session needs, from the sectioned config.

        Model/sandbox precedence: explicit request > preset > the
        workspace's most recently used choice > the default preset."""
        try:
            sectioned = load_sectioned(self.config_path)
        except ConfigError as exc:
            raise SessionError(str(exc)) from exc
        ws = Path(request.workspace).expanduser().resolve() \
            if request.workspace else Path.cwd()
        if not ws.exists():
            # A new workspace is a fine thing to start a session on: create
            # it (mkdir -p) instead of refusing — a failed mkdir surfaces as
            # the session-start error, so permission problems still show.
            try:
                ws.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise SessionError(
                    f"workspace does not exist and could not be created "
                    f"({exc}): {ws}"
                ) from None

        preset_name = request.preset or sectioned.default_preset
        preset_settings = sectioned.presets.get(preset_name)
        last_model, last_sandbox = self._workspace_last_used(ws)
        model_name = request.model or (
            preset_settings.model if preset_settings else None
        ) or last_model
        sandbox_name = request.sandbox or (
            preset_settings.sandbox if preset_settings else None
        ) or last_sandbox
        try:
            profile_name, profile = sectioned.resolve_model(model_name)
            sandbox_key, settings = sectioned.resolve_sandbox(sandbox_name)
        except ConfigError as exc:
            raise SessionError(str(exc)) from exc
        # An existing session keeps the sandbox state it was saved with
        # (its /sandbox and /filter choices), not a fresh resolution.
        # Workspace refs are NEVER baked into the settings: they are per
        # workspace, passed separately, and merged when the sandbox config
        # is built (Session owns the merge).
        workspace_refs = load_workspace_refs(ws)
        if saved_sandbox_settings is not None:
            settings = SandboxSettings.model_validate(saved_sandbox_settings)
            sandbox_key = saved_sandbox_name or sandbox_key
        # Tri-state CLI overrides (--sandbox true/false, --network, --gpu,
        # --enable_tracking): a given flag wins over the sandbox profile.
        overrides = {
            "allow_network": request.network,
            "gpu_access": request.gpu,
            "use_sandbox": request.use_sandbox,
            "enable_tracking": request.enable_tracking,
        }
        overrides = {k: v for k, v in overrides.items() if v is not None}
        if overrides:
            settings = settings.model_copy(update=overrides)
        try:
            api_key = resolve_api_key(profile)
        except ConfigError as exc:
            raise SessionError(str(exc)) from exc
        # Which model this session's subagents run. "same" borrows the
        # spawning agent's model (an override may have picked another one);
        # unset falls back to the preset's own model, which is usually what
        # the parent resolved to anyway.
        subagent_model = preset_settings.subagent_model if preset_settings else None
        if subagent_model == "same":
            subagent_model = profile_name
        elif subagent_model is None:
            subagent_model = (
                preset_settings.model if preset_settings and preset_settings.model
                else profile_name
            )

        # Tracking off (profile or --enable_tracking false): no checkpoints,
        # so skip the workspace scan instead of doing it and discarding the
        # result. Otherwise open the workspace's state — the row already
        # carries the baseline, so only a first run pays for a snapshot.
        workspace_state: WorkspaceState | None = None
        if settings.enable_tracking:
            try:
                workspace_state = await self.open_workspace(
                    ws, allow_init=not request.no_init_repo,
                    persist=not request.no_session,
                )
            except (GitWorktreeError, OSError) as exc:
                # Tracking is best-effort: a workspace that cannot be
                # snapshotted (too big, git missing, index refused) must never
                # stop the session from starting — the chat works without
                # /diff and /undo.
                LOG.warning("change tracking off for %s: %s", ws, exc)
                workspace_state = None

        try:
            sandbox = SandboxRuntime.make(
                ws, settings, workspace_state, name=sandbox_key,
                extra_refs=workspace_refs,
            )
        except SandboxUnavailable:
            # Tools disabled, chat works.
            sandbox = SandboxRuntime(name=sandbox_key, settings=settings)

        return SessionParts(
            workspace=ws,
            profile=profile,
            profile_name=profile_name,
            api_key=api_key,
            config_path=self.config_path,
            sandbox=sandbox,
            workspace_state=workspace_state,
            tool_output_limit=sectioned.tool_output_limit,
            subagent_model=subagent_model,
            # Copy: a session toggles its own servers in memory, and the
            # sectioned config object is cached/shared across sessions.
            mcp=sectioned.mcp.model_copy(deep=True),
            workspace_refs=workspace_refs,
            notify_workspace=(
                (lambda reason=None: self.notify_workspace(
                    workspace_state, reason=reason))
                if workspace_state is not None else None
            ),
            notify_state=self._session_state_changed,
            spawn_subagent=self.spawn_subagent,
        )

    # ------------------------------------------------------------------
    # session management

    async def create_session(
        self, request: protocol.CreateSessionRequest,
    ) -> Session:
        name = session_store.new_session_name()
        parts = await self._build_parts(request, session_name=name)
        session = Session(
            **parts.session_kwargs(name),
            no_save=request.no_session,
            view=request.view,
        )
        self.sessions[name] = session
        # MCP may take its whole startup timeout on a broken server, so it
        # starts in the background: the create response and the first
        # snapshot must not wait for a handshake. A turn awaits it.
        session.ensure_mcp()
        if not request.no_session:
            # Durable from creation: the session shows up as saved in /sessions
            # and survives a host crash before the first turn.
            session._persist()
        self._broadcast_session(session)
        # A new row carries no totals until a diff runs; schedule one so a
        # watching client does not see it appear as +0 -0 while its workspace
        # has pending work.
        self._schedule_stats(session.workspace_state)
        if request.first_message:
            await session.execute(
                protocol.SendMessage(text=request.first_message),
            )
        return session

    def _next_agent_name(self, parent_name: str) -> str:
        """The next ``<parent>/agent-N`` name for a child of ``parent_name``.

        Numbering is per parent, so the ``agent-N`` short name the UI shows
        can repeat under different parents. The first spawn in this host
        scans the hosted and saved names so a restarted host continues the
        parent's sequence instead of overwriting ``agent-1``."""
        prefix = f"{parent_name}/agent-"
        count = self._sub_counts.get(parent_name)
        if count is None:
            known = set(self.sessions) | set(session_store.load_metas())
            count = max(
                (int(name[len(prefix):]) for name in known
                 if name.startswith(prefix)
                 and name[len(prefix):].isdigit()),
                default=0,
            )
        count += 1
        self._sub_counts[parent_name] = count
        return f"{prefix}{count}"

    async def spawn_subagent(
        self, parent: Session, prompt: str, mode: str,
    ) -> str:
        """Run one subagent turn and return its final message.

        The child is an ordinary session — its events are recorded like any
        other agent's and it shows up in /sessions under ``parent`` — but it
        has no subagent tool and no ask_user, and nobody streams it live.
        It shares the parent's workspace, container mounts and no-session
        flag. The caller's tool call awaits this; cancelling that call
        cancels and closes the child."""
        name = self._next_agent_name(parent.name)
        profile_name, profile, api_key = \
            parent.profile_name, parent.profile, parent.api_key
        if parent.subagent_model and parent.subagent_model != parent.profile_name:
            try:
                sectioned = load_sectioned(self.config_path)
                profile_name, profile = sectioned.resolve_model(parent.subagent_model)
                api_key = resolve_api_key(profile)
            except ConfigError as exc:
                raise SessionError(str(exc)) from exc
        state = parent.workspace_state
        try:
            sandbox = (
                SandboxRuntime.make(
                    parent.workspace, parent.sandbox.settings, state,
                    name=parent.sandbox.name,
                    extra_refs=parent.workspace_refs,
                ) if parent.sandbox.settings is not None else SandboxRuntime()
            )
        except SandboxUnavailable:
            # Tools disabled, chat works (same fallback as _build_parts).
            sandbox = SandboxRuntime(
                name=parent.sandbox.name, settings=parent.sandbox.settings)
        child = Session(
            name=name,
            workspace=parent.workspace,
            profile=profile,
            profile_name=profile_name,
            api_key=api_key,
            config_path=self.config_path,
            sandbox=sandbox,
            workspace_state=state,
            mode=mode,
            tool_output_limit=parent.tool_registry.max_output_chars,
            no_save=parent.no_save,
            parent_id=parent.name,
            workspace_refs=list(parent.workspace_refs),
            mcp_manager=parent.mcp,
            notify_workspace=(
                (lambda reason=None: self.notify_workspace(state, reason=reason))
                if state is not None else None
            ),
            notify_state=self._session_state_changed,
            report_usage=report_tool_usage,
        )
        self.sessions[name] = child
        child.ensure_mcp()
        if not parent.no_save:
            child._persist()
        self._broadcast_session(child)
        self._schedule_stats(child.workspace_state)
        parent._emit_notice(
            f"subagent {session_store.display_name(name)} ({mode}) started, "
            "/sessions to watch")
        try:
            await child.execute(protocol.SendMessage(text=prompt))
            await child.wait_idle()
            return next(
                (m.content for m in reversed(child.conversation.messages)
                 if m.role == "assistant" and m.content.strip()),
                "",
            )
        finally:
            # A cancelled tool call must not leave the child running: stop()
            # cancels its turn, flushes the save and closes its shell.
            self.sessions.pop(name, None)
            await child.shutdown()
            # Still on disk: the row falls back to its saved meta.
            self._broadcast_released(name)

    async def load_session(self, name: str) -> Session:
        """Host a saved session (idempotent, safe under concurrent loads)."""
        existing = self.sessions.get(name)
        if existing is not None:
            return existing
        async with self._load_locks.setdefault(name, asyncio.Lock()):
            # Re-check: another coroutine may have finished loading while
            # this one waited on the lock.
            existing = self.sessions.get(name)
            if existing is not None:
                return existing
            return await self._load_session_locked(name)

    async def _load_session_locked(self, name: str) -> Session:
        if session_store.load_meta(name) is None:
            raise SessionError(f"no saved session named {name!r}")
        try:
            state = await asyncio.to_thread(session_store.load_state, name)
        except (OSError, ValueError, KeyError) as exc:
            raise SessionError(f"could not read the saved session: {exc}") from exc
        if not state.workspace:
            raise SessionError(
                f"saved session {name!r} has no workspace recorded; cannot "
                "rebuild its sandbox and tracking"
            )
        # The baseline comes from the workspace row, not from the session:
        # every agent in a repo shares one tracked state, and it must be the
        # tree they measured against (a fresh snapshot at load would call
        # everything saved "drift" and cost a full scan on every resume).
        parts = await self._build_parts(
            protocol.CreateSessionRequest(
                workspace=state.workspace, model=state.profile,
            ),
            session_name=name,
            saved_sandbox_name=state.sandbox_name,
            saved_sandbox_settings=state.sandbox_settings,
        )
        session = Session.load(name, parts=parts, state=state)
        self.sessions[name] = session
        # Background: a slow/broken MCP server must not delay the attach
        # snapshot (the first turn awaits the same task).
        session.ensure_mcp()
        self._broadcast_session(session)
        # Opening a saved row must not drop its totals to zero: the row the
        # client already shows was computed from the saved baseline, this
        # session's own cache starts empty. One diff fixes the second frame.
        self._schedule_stats(session.workspace_state)
        return session

    async def delete_session(self, name: str) -> protocol.DeleteResult:
        session = self.sessions.get(name)
        if session is not None:
            if session.state != "idle":
                raise SessionError("wait for the current turn to finish before deleting")
            await self._evict(session)
        deleted = session_store.delete_session(name)
        if deleted:
            # Gone for good: the eviction above pushed the saved row, this
            # takes the row away entirely (saved-only sessions get this too).
            self._broadcast_removed([name])
        return protocol.DeleteResult(deleted=deleted)

    async def unload_session(self, name: str) -> bool:
        """Stop hosting a session ("stop and quit"): the turn is cancelled,
        the session leaves the host's list, and its subscribers are released.
        The saved session file is kept — /sessions reopens it."""
        session = self.sessions.get(name)
        if session is None:
            return False
        await self._evict(
            session,
            notice="session closed — it stays saved on disk; /sessions to reopen",
        )
        return True

    async def _evict(self, session: Session, notice: str | None = None) -> None:
        """Drop a session from the host: cancel its turn, release its
        subscribers, close it down. The saved file is kept."""
        if self.sessions.get(session.name) is not session:
            return
        self.sessions.pop(session.name, None)
        await session.shutdown(notice=notice)
        # The session is still saved, so its row does not leave the list: it
        # becomes the saved meta row (state ``saved``, ``hosted`` false).
        self._broadcast_released(session.name)

    def notify_workspace(
        self, workspace: WorkspaceState, *, reason: str | None = None,
    ) -> None:
        """Nudge every agent in a workspace to refresh its diff.

        The workspace (and its event log) is shared, so a change recorded
        by one agent is visible to all of them; ``reason`` also announces
        an accept/reject rebaseline."""
        for session in self.sessions.values():
            if session.workspace_state is not workspace:
                continue
            if reason is not None:
                session._emit(protocol.CheckpointsChanged(reason=reason))
            session._schedule_diff_changed()
        # The +/- totals are derived from the workspace event log, so one
        # snapshot+diff answers for every session here. The listing itself
        # always reads the cached result (see list_sessions).
        self._schedule_stats(workspace)

    async def list_sessions(
        self, *, active: bool | None = None, workspace: str | None = None,
        stats: bool = False, window: str | None = None,
    ) -> list[SessionSummary]:
        """One listing of hosted and saved sessions. Never runs git.

        ``stats`` is accepted for wire compatibility and ignored: the totals
        ride along on every summary, since they are cached on the session and
        including them is free. They are maintained by the host on workspace
        events — see :meth:`_refresh_stats` — never here: a listing request
        must not block on a worktree snapshot.

        Saved (not hosted) sessions get no totals: their workspace is
        usually not open, and guessing what is pending there would cost a
        full snapshot. ``window`` is a Python-style range over the
        recency-sorted list (see :func:`xarness.util.parse_range`)."""
        summaries: list[SessionSummary] = []
        for session in self.sessions.values():
            if not same_path(session.workspace, workspace):
                continue
            summaries.append(session.summary())
        if active is None or active is False:
            hosted_names = set(self.sessions)
            for name, meta in session_store.load_metas().items():
                if name in hosted_names:
                    continue
                if not same_path(meta.workspace, workspace):
                    continue
                summaries.append(meta)
        if active is True:
            summaries = [s for s in summaries if s.active]
        summaries.sort(key=lambda s: s.last_user_message_at or s.name, reverse=True)
        if window is not None:
            start, stop = parse_range(window, len(summaries))
            summaries = summaries[start:stop]
        return summaries

    async def session_summary(
        self, name: str,
    ) -> protocol.SessionSummary | None:
        """One listing row, with that session's own pending totals.

        A hosted session answers from memory: its totals are cached and kept
        current by :meth:`_refresh_stats`. A saved one is *not* hosted to
        answer — asking for a row (a hover, a status fetch) must never turn a
        saved session into an idle hosted one. Its workspace's saved baseline
        is opened instead and one diff is computed for this session's own
        subtree, subagents included, and cached for a few seconds so sweeping
        across several rows of one repo costs one snapshot."""
        session = self.sessions.get(name)
        if session is not None:
            return session.summary()
        meta = session_store.load_meta(name)
        if meta is None:
            return None
        info = await self._saved_workspace_diff(meta.workspace)
        if info is None:
            # Never tracked here: no baseline, nothing pending to report.
            return meta
        additions, deletions = self._subtree_totals(
            info, self._subtree(name, self._family_children()),
        )
        return meta.model_copy(update={
            "tracking": info.tracking,
            "additions": additions,
            "deletions": deletions,
        })

    # ------------------------------------------------------------------
    # pushed session list (GET /v1/events)

    def watch_sessions(self) -> asyncio.Queue[Any]:
        """Register one session-list subscriber (the API calls this).

        The caller sends the initial snapshot itself (``list_sessions``) and
        then drains this queue. Registering first and snapshotting second
        means an update cannot fall in the gap; the client may see one
        duplicate summary, which is an idempotent upsert."""
        queue: asyncio.Queue[Any] = asyncio.Queue()
        self._watchers.append(queue)
        # Seed the cached totals for every hosted workspace now that someone
        # is looking at the list (the first frame may carry zeros; the
        # refresh pushes the real numbers a moment later).
        for state in self._hosted_workspace_states():
            self._schedule_stats(state)
        return queue

    def unwatch_sessions(self, queue: asyncio.Queue[Any]) -> None:
        if queue in self._watchers:
            self._watchers.remove(queue)

    def _broadcast(self, event: Any) -> None:
        """Fan one host-level event out to every list subscriber."""
        for queue in list(self._watchers):
            queue.put_nowait(event)

    def _broadcast_session(self, session: Session) -> None:
        if self._watchers:
            self._broadcast(protocol.SessionSummaryChanged(
                summary=session.summary(),
            ))

    def _broadcast_removed(self, names: list[str]) -> None:
        if self._watchers and names:
            self._broadcast(protocol.SessionsRemoved(names=list(names)))

    def _broadcast_released(self, name: str) -> None:
        """Push the row a session falls back to when the host stops holding it.

        Eviction and unload keep the session file, so the listing still has
        it — as a saved meta row. A session that was never saved (a
        ``--no-session`` one) has no row left, so it is removed instead."""
        if not self._watchers:
            return
        meta = session_store.load_meta(name)
        if meta is None:
            self._broadcast_removed([name])
        else:
            self._broadcast(protocol.SessionSummaryChanged(summary=meta))

    def _session_state_changed(self, session: Session) -> None:
        """Session callback: push its listing row.

        Mutating tools already call :meth:`notify_workspace`, which schedules
        the diff-total refresh, so a state change only needs the push. A
        session the host has already dropped (it emits while shutting down)
        is skipped: its row was replaced or removed by the teardown path."""
        if self.sessions.get(session.name) is session:
            self._broadcast_session(session)

    # ------------------------------------------------------------------
    # cached pending-diff totals

    def _schedule_stats(self, state: WorkspaceState | None) -> None:
        """Coalesce one workspace diff refresh for the listing.

        Only worth doing when someone is watching the list (the totals are
        cosmetic and only live sessions have them). The debounce collapses
        a burst of tool writes into one snapshot."""
        if state is None or not self._watchers:
            return
        key = id(state)
        self._stats_dirty.add(key)
        task = self._stats_tasks.get(key)
        if task is not None and not task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop (a sync test): the next request can refresh
        self._stats_tasks[key] = loop.create_task(self._stats_loop(state))

    async def _stats_loop(self, state: WorkspaceState) -> None:
        try:
            while id(state) in self._stats_dirty:
                self._stats_dirty.discard(id(state))
                await asyncio.sleep(STATS_DEBOUNCE_SECONDS)
                await self._refresh_stats(state)
        finally:
            self._stats_tasks.pop(id(state), None)

    def _hosted_workspace_states(self) -> list[WorkspaceState]:
        """Every hosted workspace once (sessions in one repo share a state)."""
        return list({
            id(session.workspace_state): session.workspace_state
            for session in self.sessions.values()
            if session.workspace_state is not None
        }.values())

    @staticmethod
    async def _pending_diff(state: WorkspaceState) -> protocol.DiffInfo:
        """One worktree snapshot and diff: the pending rows this workspace
        contributes to the listing, with one row per author (``mine``)."""
        return await state.diff(
            None, current=await state.observe(), scope="mine",
        )

    async def refresh_stats(self) -> None:
        """Recompute the cached listing totals for every hosted workspace.

        Public for tests and for an explicit "refresh now"; the normal path
        is :meth:`_schedule_stats` from a workspace event. This is the only
        code that runs a worktree snapshot/diff for the session list."""
        for state in self._hosted_workspace_states():
            await self._refresh_stats(state)

    async def _refresh_stats(self, state: WorkspaceState) -> None:
        """One snapshot+diff for a workspace, fanned out to its sessions.

        One diff answers for every agent in the repo: ``scope="mine"``
        produces a row per author, so a session's total is the sum of the
        rows authored by itself or a hosted descendant (the same subtree
        semantics the old per-session listing computed, without the N
        git runs). Best-effort: a failing snapshot leaves the last cached
        values in place."""
        sessions = [
            session for session in self.sessions.values()
            if session.workspace_state is state
        ]
        if not sessions:
            return
        children = self._children_by_parent()
        try:
            info = await self._pending_diff(state)
        except Exception as exc:  # a git failure must not break the list
            LOG.warning("session stats refresh failed for %s: %s", state.root, exc)
            return
        # The same diff answers a saved session's hover in this repo.
        self._saved_stats[str(state.root)] = (time.monotonic(), info)
        for session in sessions:
            additions, deletions = self._subtree_totals(
                info, self._subtree(session.name, children),
            )
            if (additions, deletions) == (
                session.pending_additions, session.pending_deletions,
            ):
                continue
            session.pending_additions = additions
            session.pending_deletions = deletions
            self._broadcast_session(session)

    async def _saved_workspace_diff(
        self, workspace: str | None,
    ) -> protocol.DiffInfo | None:
        """A pending diff for a saved session's workspace, or None.

        Read-only about the *session*: the workspace's saved tracking state is
        opened, nothing is hosted. A workspace that was never tracked has no
        baseline, so it reports nothing rather than creating tracking as a
        side effect of a hover. The diff is cached for
        ``SAVED_STATS_TTL_SECONDS`` — it answers for every session in that
        repo, so a sweep across rows pays for one snapshot."""
        if not workspace:
            return None
        key = str(Path(workspace).expanduser().resolve())
        now = time.monotonic()
        cached = self._saved_stats.get(key)
        if cached is not None and now - cached[0] < SAVED_STATS_TTL_SECONDS:
            return cached[1]
        if not Path(key).is_dir() or not has_workspace_state(key):
            return None
        try:
            state = await self.open_workspace(Path(key))
            if state is None:
                return None
            info = await self._pending_diff(state)
        except Exception as exc:  # a git failure must not break a hover
            LOG.warning("saved-session stats failed for %s: %s", key, exc)
            return None
        self._saved_stats[key] = (now, info)
        return info

    def _family_children(self) -> dict[str, list[str]]:
        """Every known parent -> children link, hosted or only on disk, so a
        saved session's totals include the subagents it left behind."""
        children = self._children_by_parent()
        for name, meta in session_store.load_metas().items():
            if meta.parent_id and name not in self.sessions:
                children.setdefault(meta.parent_id, []).append(name)
        return children

    @staticmethod
    def _subtree_totals(
        info: protocol.DiffInfo, subtree: set[str],
    ) -> tuple[int, int]:
        """Sum the rows authored by ``subtree`` (``scope="mine"`` yields one
        row per author, so a file two agents wrote is never double-counted)."""
        additions = deletions = 0
        for row in info.files:
            authors = (row.author,) if row.author else row.authors
            if any(author in subtree for author in authors):
                additions += row.additions
                deletions += row.deletions
        return additions, deletions

    def _children_by_parent(self) -> dict[str, list[str]]:
        """Hosted sessions indexed by the parent that spawned them."""
        children: dict[str, list[str]] = {}
        for session in self.sessions.values():
            if session.parent_id:
                children.setdefault(session.parent_id, []).append(session.name)
        return children

    @staticmethod
    def _subtree(name: str, children: dict[str, list[str]]) -> set[str]:
        """``name`` plus every hosted descendant (a subagent's work)."""
        subtree = {name}
        pending = [name]
        while pending:
            for child in children.get(pending.pop(), ()):
                if child not in subtree:
                    subtree.add(child)
                    pending.append(child)
        return subtree

    @staticmethod
    def _workspace_ref_infos(
        workspace: Path, specs: list[str],
    ) -> list[protocol.RefInfo]:
        return ref_infos(SandboxConfig.resolve_auto_refs(workspace, specs))

    def workspace_refs(self, workspace: str) -> list[protocol.RefInfo]:
        """Canonical workspace refs, without hosting a session."""
        ws = Path(workspace).expanduser().resolve()
        return self._workspace_ref_infos(
            ws, load_workspace_refs(ws),
        )

    async def update_workspace_refs(
        self, update: protocol.WorkspaceRefUpdate,
    ) -> list[protocol.RefInfo]:
        """Add/remove one workspace ref and push the canonical list to every
        hosted session in that workspace. The add/remove logic is the
        workspace store's (:func:`change_workspace_ref`), shared with the
        session's /ref command."""
        ws = Path(update.workspace).expanduser().resolve()
        try:
            specs, _host = change_workspace_ref(ws, update.path, update.add)
        except RefError as exc:
            raise SessionError(str(exc)) from exc
        except OSError as exc:
            raise SessionError(f"could not save references: {exc}") from exc
        for session in self.sessions.values():
            if session.workspace.resolve() == ws:
                if session.state != "idle":
                    # Rebinding the sandbox mid-turn can close the shell
                    # under a running command; the session picks the new
                    # list up from the workspace store on its next reload,
                    # and the user can run /ref to rebind live.
                    session._emit_notice(
                        "references changed for this workspace while a turn "
                        "is running — run /refs to rebind them live",
                        "warn",
                    )
                    continue
                await session.apply_ref_specs(specs)
        return self._workspace_ref_infos(ws, specs)

    async def workspace_diff(
        self, workspace: str, *, path: str | None = None,
        author: str | None = None, scope: protocol.DiffScope = "all",
    ) -> protocol.DiffInfo:
        """A workspace's pending changes, hosting no session: tracking is
        per repo root, so a draft chat's /diff needs none. Rows are labelled
        for no viewer — a chat that has not started has written nothing."""
        ws = Path(workspace).expanduser().resolve()
        if not ws.is_dir():
            return protocol.DiffInfo(tracking=False)
        try:
            state = await self.open_workspace(ws)
            if state is None:
                return protocol.DiffInfo(tracking=False)
            return await state.diff_info(
                None, path=path, author=author, scope=scope,
            )
        except (GitWorktreeError, OSError):
            # Best-effort, as at session start: a workspace that cannot be
            # snapshotted just has no pending changes to show.
            return protocol.DiffInfo(tracking=False)

    def server_info(self) -> ServerInfo:
        return ServerInfo(
            version=__version__,
            protocol_version=protocol.PROTOCOL_VERSION,
            pid=os.getpid(),
            uptime_seconds=time.monotonic() - self._started_at,
            persistent=self.persistent,
            unix_socket=str(self.unix_socket),
        )

    def stats(self) -> ServerStats:
        sessions = [s.summary() for s in self.sessions.values()]
        return ServerStats(
            host=self.server_info(),
            started_at=self._started_at_wall,
            clients_attached=sum(s.clients_attached for s in sessions),
            total_input_tokens=sum(
                s.usage_total.input_tokens for s in self.sessions.values()
            ),
            total_output_tokens=sum(
                s.usage_total.output_tokens for s in self.sessions.values()
            ),
            sessions=sessions,
        )

    def set_default_profile(self, name: str | None) -> None:
        """Write the default model choice to the host-owned config file."""
        if name is None:
            return
        save_preferences(self.config_path, default_profile=name)

    def config_info(self) -> protocol.ConfigInfo:
        """Names and non-secret fields for the client's pickers (never API
        keys). A broken config falls back to an empty (default) config."""
        try:
            sectioned = load_sectioned(self.config_path)
        except ConfigError:
            sectioned = SectionedConfig()
        models = [
            protocol.ModelInfo(
                name=name,
                display_name=p.display_name or p.model_id,
                cot_strength=p.cot_strength.value,
                max_context=p.max_context,
                supports_vision=p.supports_vision,
                auto_compact=p.auto_compact,
            )
            for name, p in sectioned.models.items()
        ]
        sandboxes = [protocol.SandboxInfo(name=name) for name in sectioned.sandboxes]
        presets = list(sectioned.presets)
        default_model = sectioned.default_preset_model()
        return protocol.ConfigInfo(
            models=models, sandboxes=sandboxes, presets=presets,
            default_model=default_model,
            default_model_display_name=(
                sectioned.models[default_model].display_name
                or sectioned.models[default_model].model_id
                if default_model in sectioned.models else default_model
            ),
            default_sandbox=sectioned.default_preset_sandbox(),
        )

    def workspaces(self) -> list[protocol.WorkspaceInfo]:
        counts: dict[str, int] = {}
        active: dict[str, int] = {}
        for session in self.sessions.values():
            path = str(session.workspace)
            counts[path] = counts.get(path, 0) + 1
            if session.is_active:
                active[path] = active.get(path, 0) + 1
        hosted_names = set(self.sessions)
        for name, meta in session_store.load_metas().items():
            # Hosted sessions are persisted at creation; skip them so they
            # aren't double-counted (as list_sessions already does).
            if name in hosted_names:
                continue
            if meta.workspace:
                ws = meta.workspace
                counts[ws] = counts.get(ws, 0) + 1
        return [
            protocol.WorkspaceInfo(
                path=path, session_count=count, active_sessions=active.get(path, 0),
            )
            for path, count in sorted(counts.items())
        ]

    # ------------------------------------------------------------------
    # lifecycle

    async def serve(
        self, stats_interval: float | None = None, *,
        web: "WebListen | None" = None,
        on_ready=None,
    ) -> None:
        """Run until shutdown: bind the unix socket, arm the grace timer,
        run until the shutdown event (or print stats every
        ``stats_interval`` seconds in foreground mode), then tear everything
        down.

        ``web`` adds the browser-facing TCP listener (``xarness --web``);
        the unix socket is always bound, so a TUI can attach to the same
        host and its sessions. ``on_ready`` runs once both listeners are
        actually up — printing a URL before that would advertise something
        a failed bind never served."""
        HOST_DIR.mkdir(parents=True, mode=0o700, exist_ok=True)
        if web is not None:
            # Probe the TCP address before binding anything else: a bad
            # ``--host`` should cost one clear line, not a half-started host.
            _probe_bind(web.bind, web.port)
        server, server_task = await _start_unix_server(self)
        web_server = web_task = None
        if web is not None:
            web_server, web_task = await _start_web_server(
                self, web.bind, web.port, web.token,
            )
        self._reaper = asyncio.get_running_loop().create_task(self._reap())
        if on_ready is not None:
            on_ready()
        listeners = (server_task, web_task)
        try:
            while True:
                if stats_interval is not None:
                    stats = self.stats()
                    print(
                        f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
                        f"{_format_stats(stats)}",
                        flush=True,
                    )
                if await self._wait_for_stop(listeners, stats_interval):
                    break
        finally:
            if self._reaper is not None:
                self._reaper.cancel()
                self._reaper = None
            # A pending stats refresh must not outlive the host (and must
            # not touch a workspace whose session is stopping below).
            for task in self._stats_tasks.values():
                task.cancel()
            self._stats_tasks.clear()
            self._stats_dirty.clear()
            for one, task in ((server, server_task), (web_server, web_task)):
                if one is None or task is None:
                    continue
                one.should_exit = True
                try:
                    await asyncio.wait_for(task, timeout=5)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    task.cancel()
            for session in list(self.sessions.values()):
                await session.stop()
            if self.unix_socket.exists():
                self.unix_socket.unlink()

    async def _wait_for_stop(self, listeners, timeout: float | None) -> bool:
        """Whether the host should tear down: shutdown was requested, or a
        listener exited on its own.

        uvicorn turns SIGINT/SIGTERM into a clean exit of *its* server, and
        with two listeners the other one would keep the process alive (a
        Ctrl-C that closes the port but never returns). Waiting on the
        listener tasks as well fixes that for ``--server`` and ``--web``
        alike."""
        shutdown = asyncio.ensure_future(self._shutdown_event.wait())
        waiters = [shutdown, *(t for t in listeners if t is not None)]
        try:
            done, _ = await asyncio.wait(
                waiters, timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            shutdown.cancel()
        if self._shutdown_event.is_set():
            return True
        if done:
            self._shutdown_event.set()
            return True
        return False

    async def _reap(self) -> None:
        """The host's one lifecycle loop: evict inactive sessions and, for a
        non-persistent host, exit after GRACE_SECONDS with nothing to keep it
        alive.

        Polling instead of per-session timers keeps the whole policy in one
        place: there is no idle callback for a session to fire, no handle to
        re-arm, and an eviction can never cancel the task that scheduled it.

        A session is evicted once it has been inactive (no turn, no ask, no
        client) for ``_idle_ttl()`` seconds. A warm idle session is why the
        host lingers: the next turn pays no state reload, no sandbox rebuild,
        and keeps the persistent shell's cwd."""
        inactive: dict[str, float] = {}
        quiet_since: float | None = None
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=REAP_INTERVAL_SECONDS,
                )
                return
            except asyncio.TimeoutError:
                pass
            now = time.monotonic()
            ttl = _idle_ttl()
            warm = False
            for session in list(self.sessions.values()):
                if session.is_active:
                    inactive.pop(session.name, None)
                    continue
                since = inactive.setdefault(session.name, now)
                if now - since < ttl:
                    warm = True
                    continue
                inactive.pop(session.name, None)
                session._persist()
                await self._evict(session)
            # Sessions that left the host by another path (unload, delete)
            # must not keep their clock around.
            for name in [n for n in inactive if n not in self.sessions]:
                del inactive[name]
            if warm or any(s.is_active for s in self.sessions.values()):
                quiet_since = None
            elif self.persistent:
                quiet_since = None
            elif quiet_since is None:
                quiet_since = now
            elif now - quiet_since >= GRACE_SECONDS:
                self._shutdown_event.set()

    async def request_shutdown(self) -> None:
        self._shutdown_event.set()


# ---------------------------------------------------------------------------
# uvicorn server


async def _start_unix_server(host: Host):
    import uvicorn

    from .api import create_app
    app = create_app(host)
    if host.unix_socket.exists():
        if _socket_alive_at(host.unix_socket):
            raise RuntimeError(
                f"a live host is already listening on {host.unix_socket} "
                "— never unbinding a live socket"
            )
        host.unix_socket.unlink()  # stale socket from a dead host
    config = uvicorn.Config(app, log_level="warning", uds=str(host.unix_socket))
    return await _serve_uvicorn(config, wait_for=lambda: host.unix_socket)


async def _start_web_server(
    host: Host, bind: str, port: int, token: str | None,
):
    """Second listener: the same app (API + the web UI's static files) on
    TCP, so a browser can reach it. The token gates ``/v1/*`` only when one
    was given."""
    import uvicorn

    from .api import create_app
    app = create_app(host, token=token)
    _probe_bind(bind, port)
    config = uvicorn.Config(
        app, host=bind, port=port, log_level="warning",
    )
    try:
        return await _serve_uvicorn(config, wait_for=lambda: None)
    except (OSError, SystemExit) as exc:
        # A bind that went wrong after the probe (another process took the
        # port in between): uvicorn answers with a raw ``sys.exit`` from
        # inside its task, so say it in one line instead of a traceback.
        raise RuntimeError(
            f"cannot listen on {bind}:{port} ({_bind_reason(exc)}) — check "
            "--host/--port, or that the port is free",
        ) from None


def _probe_bind(bind: str, port: int) -> None:
    """Bind and release the address before uvicorn does.

    uvicorn turns a failing bind into ``sys.exit(3)`` from inside its server
    task: the process dies with a traceback from the event loop, before the
    URLs above it were ever true. A wrong ``--host`` (an address this
    machine no longer has, say after DHCP moved it) is an easy mistake, so
    it gets one clear line instead."""
    try:
        infos = socket.getaddrinfo(
            bind or None, port, type=socket.SOCK_STREAM,
            flags=socket.AI_PASSIVE,
        )
    except socket.gaierror as exc:
        raise RuntimeError(
            f"cannot listen on {bind}:{port} ({exc.strerror or exc}) — check "
            "--host",
        ) from None
    reason = "no usable address"
    for family, kind, proto, _canon, address in infos:
        probe = socket.socket(family, kind, proto)
        try:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(address)
            return
        except OSError as exc:
            reason = exc.strerror or str(exc)
        finally:
            probe.close()
    raise RuntimeError(
        f"cannot listen on {bind}:{port} ({reason}) — check --host/--port, "
        "or whether the port is already taken",
    )


def _bind_reason(exc: BaseException) -> str:
    """The short human reason a bind failed."""
    if isinstance(exc, OSError) and exc.strerror:
        return exc.strerror
    return "address unavailable or port in use"


async def _serve_uvicorn(config, *, wait_for):
    """Start one uvicorn server on the running loop and wait until it is up.

    ``wait_for`` returns a socket path to chmod 0600 once bound (the unix
    listener), or None for TCP."""
    import uvicorn

    server = uvicorn.Server(config)
    task = asyncio.get_running_loop().create_task(server.serve())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 10.0
    while not server.started:
        if task.done():
            task.result()
            raise RuntimeError("the host server exited before it started")
        if server.should_exit:
            raise RuntimeError("the host server was asked to exit before it started")
        if loop.time() >= deadline:
            task.cancel()
            raise RuntimeError("timed out waiting for the host server to start")
        await asyncio.sleep(0.05)
    path = wait_for()
    if path is not None and path.exists():
        os.chmod(path, 0o600)
    return server, task


# ---------------------------------------------------------------------------
# client-side helpers: auto-start and connect


def connect_or_spawn(*, timeout: float = SPAWN_TIMEOUT) -> Path:
    """Make sure a host is running; return the socket path.

    Connection refused on a stale socket counts as "no host": the stale
    file is removed and a host is spawned. An exclusive startup lock keeps
    racing clients from spawning two hosts — the loser retries connecting."""
    HOST_DIR.mkdir(parents=True, mode=0o700, exist_ok=True)
    if _socket_alive():
        return SOCKET_PATH
    lock_handle = open(LOCK_PATH, "a")
    try:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Another process is spawning right now: wait for its socket.
            _wait_for_socket(timeout)
            return SOCKET_PATH
        if _socket_alive():
            return SOCKET_PATH
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()  # stale
        # Spawned detached: it survives this client. Its output goes to the
        # system journal-style devnull — debugging happens in the foreground
        # (--server) mode.
        subprocess.Popen(
            [sys.executable, "-m", "xarness.host"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True, cwd="/",
        )
        _wait_for_socket(timeout)
        return SOCKET_PATH
    finally:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()


def _socket_alive_at(path: Path) -> bool:
    """Whether something is listening at ``path``."""
    if not path.exists():
        return False
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        sock.connect(str(path))
        sock.close()
        return True
    except (OSError, socket.timeout):
        return False


def _socket_alive() -> bool:
    """Whether a live host is on the default socket; a stale socket file
    (nothing listening behind it) is unlinked."""
    if not SOCKET_PATH.exists():
        return False
    if _socket_alive_at(SOCKET_PATH):
        return True
    # Refused / stale: no host behind the file.
    try:
        SOCKET_PATH.unlink()
    except OSError:
        pass
    return False


def _wait_for_socket(timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _socket_alive_at(SOCKET_PATH):
            return
        time.sleep(0.1)
    raise TimeoutError(f"host socket did not appear within {timeout}s: {SOCKET_PATH}")


def stop_host(
    pid: int, socket_path: Path = SOCKET_PATH, timeout: float = SPAWN_TIMEOUT,
) -> None:
    """Terminate a stale host and wait for its socket to disappear.

    Used when a client finds a host speaking an older protocol: with no
    active sessions, the old process is replaced so the next connect gets
    the current command set."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _socket_alive_at(socket_path):
            socket_path.unlink(missing_ok=True)
            return
        time.sleep(0.1)
    raise TimeoutError(f"stale host (pid {pid}) did not exit within {timeout}s")


def run_foreground(*, web: "WebListen | None" = None) -> None:
    """``xarness --server``: run the host in the foreground, persistent.

    With a terminal attached the host's dashboard is the display: sessions,
    clients and per-session token spend, plus the browser listener's URLs
    and token when ``web`` is given. Piped or under a service manager it
    keeps the old behaviour — one stats line per interval — because escape
    sequences in a log file are worse than lines.

    ``web`` is what ``xarness --web`` passes; ``--server --host/--port/
    --token`` builds the same listener. A live host is never shared: a bare
    ``--server`` prints the running host's stats, while asking for the web
    listener fails (that host has no listener to add one to)."""
    HOST_DIR.mkdir(parents=True, mode=0o700, exist_ok=True)
    if _socket_alive():
        if web is not None:
            print(
                f"error: a host is already running (socket {SOCKET_PATH}).\n"
                "  It has no web listener. Stop it first (`xarness --kill`), then "
                "run the command again.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        _print_live_stats()
        return
    lock_handle = open(LOCK_PATH, "a")
    try:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(
                "error: another host is starting up; retry in a moment",
                file=sys.stderr,
            )
            raise SystemExit(2) from None
        host = Host(persistent=True, unix_socket=SOCKET_PATH, config_path=DEFAULT_CONFIG_PATH)
        if _interactive():
            _run_dashboard(host, web=web)
            return
        try:
            # Piped: lines, not a screen. The web listener still gets its
            # banner, once it is really up.
            print("\n".join(_host_intro(host.server_info())), flush=True)
            asyncio.run(host.serve(
                stats_interval=1.0 if web is None else 10.0,
                web=web, on_ready=_banner(web) if web is not None else None,
            ))
        except KeyboardInterrupt:
            pass
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise SystemExit(2) from None
    finally:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()


def _interactive() -> bool:
    """Whether the dashboard can be drawn and driven (both ends are a
    terminal): a redirected stdin would make the app exit immediately and
    take the host down with it."""
    try:
        return sys.stdout.isatty() and sys.stdin.isatty()
    except (AttributeError, ValueError):  # replaced or closed streams
        return False


def _run_dashboard(host: Host, *, web: "WebListen | None") -> None:
    """Run the foreground host inside the dashboard, then report its exit.

    The app starts ``host.serve`` on its own event loop; a startup failure
    (a live socket, a bad bind) surfaces through ``error`` instead of a
    traceback from a task."""
    from .tui.server_view import ServerDashboard

    view = ServerDashboard(
        host,
        web=web,
        urls=_web_urls(web) if web is not None else (),
        token=web.token if web is not None else None,
    )
    try:
        view.run()
    except KeyboardInterrupt:
        # A signal (not the ctrl+c key, which is a binding) already unwound
        # the app; the host's teardown ran with it.
        return
    if view.error is not None:
        reason = (
            f"exited with status {view.error.code}"
            if isinstance(view.error, SystemExit) else str(view.error)
        )
        print(f"error: {reason}", file=sys.stderr)
        raise SystemExit(2)
    if view.stats is not None:
        print(f"host stopped · {_format_stats(view.stats)}", flush=True)


def _format_stats(stats: ServerStats) -> str:
    """The one-line session/client/token summary shared by the foreground
    stats printer and the periodic live-stats loop."""
    states = ", ".join(
        f"{state}={count}" for state, count in sorted(stats.by_state().items())
    ) or "none"
    return (
        f"sessions: {len(stats.sessions)} ({states}), "
        f"clients: {stats.clients_attached}, "
        f"tokens: {stats.total_input_tokens} in / "
        f"{stats.total_output_tokens} out"
    )


def _print_live_stats() -> None:
    """Print the stats of the host already listening on the socket."""
    import httpx
    try:
        with httpx.Client(
            transport=httpx.HTTPTransport(uds=str(SOCKET_PATH)),
            base_url="http://localhost", timeout=5.0,
        ) as http:
            stats = ServerStats.model_validate(http.get("/v1/server/stats").json())
    except Exception as exc:
        print(f"error: could not read the live host's stats: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    print(
        f"xarness host already running (pid {stats.host.pid}):\n"
        f"  socket: {stats.host.unix_socket}\n"
        f"  {_format_stats(stats)}",
        flush=True,
    )


def _host_intro(info: ServerInfo) -> list[str]:
    """The first lines a piped foreground host prints: who it is and where."""
    return [
        f"xarness host: pid {info.pid}, protocol v{info.protocol_version}",
        f"  socket: {info.unix_socket}",
        "  docs:   (unix socket) GET /v1/health, /v1/server/stats, /docs",
    ]


# ---------------------------------------------------------------------------
# web UI (``xarness --web``)


def _lan_address() -> str | None:
    """The machine's primary LAN address, for the printed URL (best effort:
    a UDP connect sends no packet, but an offline host has no route)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 9))  # TEST-NET-1: never actually reached
        address = sock.getsockname()[0]
        return None if address.startswith("127.") else address
    except OSError:
        return None
    finally:
        sock.close()


def _web_urls(web: WebListen) -> list[str]:
    """The URLs to print: the bind address, plus the LAN address when the
    listener is reachable from other machines."""
    if web.bind in ("0.0.0.0", "::"):
        urls = [f"http://127.0.0.1:{web.port}/"]
        lan = _lan_address()
        if lan:
            urls.append(f"http://{lan}:{web.port}/")
        return urls
    return [f"http://{web.bind}:{web.port}/"]


def _banner(web: WebListen):
    """A callable that prints where the UI is, once it is really there."""

    def announce() -> None:
        print("xarness web UI:", flush=True)
        for url in _web_urls(web):
            print(f"  {url}", flush=True)
        print(f"  workspace: {Path.cwd()}", flush=True)
        if web.token:
            print(
                f"  token:     {web.token}\n"
                "  (enter it once in the UI; it is not stored on the host)",
                flush=True,
            )
        else:
            print(
                "  (loopback only, no token — a phone needs "
                "`--host 0.0.0.0`)",
                flush=True,
            )

    return announce


def web_listen(
    *,
    bind: str = DEFAULT_WEB_HOST,
    port: int = DEFAULT_WEB_PORT,
    token: str | None = None,
) -> WebListen:
    """The browser-facing listener, with a token when it is not loopback.

    The API can run shell commands in the workspace, so an address reachable
    from the LAN never goes unauthenticated."""
    web = WebListen(bind=bind, port=port, token=token or None)
    if not web.loopback and not web.token:
        web = WebListen(
            bind=web.bind, port=web.port, token=secrets.token_urlsafe(18),
        )
    return web


def main(argv: list[str] | None = None) -> None:
    """Host process entrypoint: ``python -m xarness.host`` (auto-start mode)."""
    host = Host(persistent=False)
    try:
        asyncio.run(host.serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
