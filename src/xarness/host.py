"""The host process: runs sessions and serves the FastAPI app.

One host owns many :class:`~xarness.session.Session` objects. It serves the
API on a unix socket under the xarness data dir (mode 0600 in a 0700
directory) — the only listener; the web UI (a later task) will decide on
TCP.

**Lifecycle.** The host auto-starts (detached, survives the TUI) when the
first client needs it. It exits once no session is running, compacting, or
waiting on an ask/drift prompt — attached clients do not count — after a
short grace period. ``xarness --server`` runs it in the foreground in
persistent mode (never exits on its own) and prints the stats.

**Hardening.** An exclusive startup lock keeps racing clients from
spawning two hosts (the loser retries connecting); a stale socket file
(connection refused) is removed and treated as "no host".
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, fields, replace as dataclasses_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__, protocol, session_store
from .config import (
    DEFAULT_CONFIG_PATH, ConfigError, ProviderProfile, SandboxSettings,
    SectionedConfig, load_sectioned, resolve_api_key,
)
from . import gitwork
from .gitwork import GitInfo, GitWorktreeError
from .protocol import CommandResult, SessionSummary, ServerInfo, ServerStats
from .sandbox import (
    SandboxConfig, SandboxSession, make_sandbox, SandboxUnavailable,
)
from .session import Session, SessionError
from .util import utcnow_iso as _utcnow

HOST_DIR = Path("~/.local/share/xarness").expanduser()
SOCKET_PATH = HOST_DIR / "host.sock"
LOCK_PATH = HOST_DIR / "host.lock"

# How long the host lingers with no active session before exiting
# (auto-started hosts only; a foreground --server host is persistent).
GRACE_SECONDS = 5.0

# Seconds a client waits for a freshly spawned host's socket to appear.
SPAWN_TIMEOUT = 15.0


@dataclass(slots=True)
class SessionParts:
    """Everything :class:`Session` needs, resolved by ``_build_parts``.

    Typed instead of a ``**``-splatted dict, so a mismatch with the
    ``Session`` constructor is a type error, not a runtime surprise. Field
    names match Session's keyword parameters, so ``Session(name=...,
    **parts.session_kwargs(name))`` constructs it."""

    workspace: Path
    profile: ProviderProfile
    profile_name: str
    api_key: str | None
    config_path: Path | None
    sandbox: SandboxConfig | None
    sandbox_session: SandboxSession | None
    git_info: GitInfo | None
    auto_compact: bool
    tool_output_limit: int | None
    sandbox_name: str
    sandbox_settings: "SandboxSettings"

    def session_kwargs(self, name: str) -> dict[str, Any]:
        """Keyword arguments for ``Session(name=..., **kwargs)`` (shallow:
        nested dataclasses like the sandbox config must stay objects)."""
        return {"name": name, **{f.name: getattr(self, f.name) for f in fields(self)}}



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
        self.unix_socket = unix_socket
        self.sessions: dict[str, Session] = {}
        self._started_at = time.monotonic()
        self._started_at_wall = _utcnow()
        self._shutdown_event = asyncio.Event()
        # Pending call_later handle for the non-persistent grace shutdown.
        self._grace_handle: asyncio.TimerHandle | None = None

    # ------------------------------------------------------------------
    # session construction (config + workspace + git + sandbox)

    def _workspace_last_used(self, workspace: Path) -> tuple[str | None, str | None]:
        """The most recently used (model, sandbox) among the saved sessions
        of this directory — the workspace's last choice. Reads only the
        small meta block, never the full conversation."""
        best: tuple[str | None, str | None, str] | None = None
        for name in session_store.list_sessions():
            meta = session_store.load_meta(name)
            if not meta:
                continue
            saved_ws = meta.get("workspace")
            if not saved_ws:
                continue
            try:
                same = Path(saved_ws).resolve() == workspace.resolve()
            except OSError:
                continue
            if not same:
                continue
            stamp = meta.get("updated_at") or ""
            if best is None or stamp > best[2]:
                best = (meta.get("profile"), meta.get("sandbox_name"), stamp)
        return (best[0], best[1]) if best else (None, None)

    async def _build_parts(
        self,
        *,
        workspace: str | None,
        model: str | None,
        sandbox: str | None,
        preset: str | None,
        no_init_repo: bool,
        network: bool | None = None,
        gpu: bool | None = None,
        session_name: str,
        saved_sandbox_name: str | None = None,
        saved_sandbox_settings: dict | None = None,
    ) -> SessionParts:
        """Everything Session needs, from the sectioned config.

        Model/sandbox precedence: explicit request > preset > the
        workspace's most recently used choice > the default preset."""
        try:
            sectioned = load_sectioned(self.config_path)
        except ConfigError as exc:
            raise SessionError(str(exc)) from exc
        ws = Path(workspace).expanduser().resolve() if workspace else Path.cwd()
        if not ws.exists():
            raise SessionError(f"workspace does not exist: {ws}")

        preset_name = preset or sectioned.default_preset
        preset_settings = sectioned.presets.get(preset_name)
        last_model, last_sandbox = self._workspace_last_used(ws)
        model_name = model or (preset_settings.model if preset_settings else None) \
            or last_model
        sandbox_name = sandbox or (preset_settings.sandbox if preset_settings else None) \
            or last_sandbox
        try:
            profile_name, profile = sectioned.resolve_model(model_name)
            sandbox_key, settings = sectioned.resolve_sandbox(sandbox_name)
        except ConfigError as exc:
            raise SessionError(str(exc)) from exc
        # An existing session keeps the sandbox state it was saved with
        # (its /sandbox and /filter choices), not a fresh resolution.
        if saved_sandbox_settings is not None:
            settings = SandboxSettings.model_validate(saved_sandbox_settings)
            sandbox_key = saved_sandbox_name or sandbox_key
        # Tri-state CLI overrides (--network/--gpu): the flag wins over the
        # sandbox profile when given.
        if network is not None or gpu is not None:
            settings = settings.model_copy(update={
                **({"allow_network": network} if network is not None else {}),
                **({"gpu_access": gpu} if gpu is not None else {}),
            })
        try:
            api_key = resolve_api_key(profile)
        except ConfigError as exc:
            raise SessionError(str(exc)) from exc

        git_info, _notes = await self._setup_tracking(ws, session_name, allow_init=not no_init_repo)
        if not settings.enable_tracking:
            git_info = None

        try:
            sandbox_config, sandbox_session = make_sandbox(ws, settings, git_info)
        except SandboxUnavailable:
            # Tools disabled, chat works.
            sandbox_config = sandbox_session = None

        return SessionParts(
            workspace=ws,
            profile=profile,
            profile_name=profile_name,
            api_key=api_key,
            config_path=self.config_path,
            sandbox=sandbox_config,
            sandbox_session=sandbox_session,
            git_info=git_info,
            auto_compact=profile.auto_compact,
            tool_output_limit=settings.tool_output_limit,
            sandbox_name=sandbox_key,
            sandbox_settings=settings,
        )

    @staticmethod
    async def _setup_tracking(
        workspace: Path, session_name: str, allow_init: bool,
    ) -> tuple[GitInfo | None, list[str]]:
        try:
            return await gitwork.setup_tracking(workspace, session_name, allow_init=allow_init)
        except GitWorktreeError:
            return None, []

    # ------------------------------------------------------------------
    # session management

    async def create_session(
        self,
        *,
        workspace: str | None = None,
        model: str | None = None,
        sandbox: str | None = None,
        preset: str | None = None,
        no_init_repo: bool = False,
        network: bool | None = None,
        gpu: bool | None = None,
        no_session: bool = False,
        first_message: str | None = None,
    ) -> Session:
        name = session_store.new_session_name()
        parts = await self._build_parts(
            workspace=workspace, model=model, sandbox=sandbox, preset=preset,
            no_init_repo=no_init_repo, network=network, gpu=gpu,
            session_name=name,
        )
        session = Session(**parts.session_kwargs(name), idle_callback=self._on_session_idle)
        self.sessions[name] = session
        if no_session:
            # The client asked for an unsaved conversation: nothing is ever
            # written to the session store (Session._persist no-ops).
            session.no_save = True
        else:
            # Durable from creation: the session shows up as saved in /sessions
            # and survives a host crash before the first turn.
            session._persist()
        if first_message:
            await session.execute(f"boot-{name}", protocol.SendMessage(text=first_message))
        return session

    async def load_session(self, name: str) -> Session:
        """Host a saved session (idempotent)."""
        existing = self.sessions.get(name)
        if existing is not None:
            return existing
        if not session_store.session_path(name).exists():
            raise SessionError(f"no saved session named {name!r}")
        try:
            state = session_store.load_state(name)
        except (OSError, ValueError, KeyError) as exc:
            raise SessionError(f"could not read the saved session: {exc}") from exc
        if not state.workspace:
            raise SessionError(
                f"saved session {name!r} has no workspace recorded; cannot "
                "rebuild its sandbox and tracking"
            )
        parts = await self._build_parts(
            workspace=state.workspace, model=state.profile, sandbox=None,
            preset=None, no_init_repo=False, session_name=name,
            saved_sandbox_name=state.sandbox_name,
            saved_sandbox_settings=state.sandbox_settings,
        )
        # Rebind tracking from the SAVED git block: the baseline must be the
        # one the session measured against when it was last open, not the
        # workspace's current state (otherwise every saved change would look
        # like drift and /undo would lose its turn trees).
        saved_block = state.git if isinstance(state.git, dict) else None
        if saved_block:
            try:
                parts = dataclasses_replace(
                    parts, git_info=GitInfo.from_block(saved_block)
                )
            except (KeyError, TypeError, ValueError):
                pass  # keep the freshly built tracking as the fallback
        session = Session.load(**parts.session_kwargs(name), state=state,
                               idle_callback=self._on_session_idle)
        self.sessions[name] = session
        # A turn left open by a host crash or a quit mid-turn never got its
        # after-tree (finish_turn never ran). Record the workspace's current
        # state as that turn's after-tree — what finish_turn would have
        # written had the process survived — so a plain reopen doesn't
        # surface the session's own unaccepted edits as "drift". The turn
        # must be the last one (nothing streamed after it).
        if session.git_info is not None:
            last = session.conversation.last_turn_message()
            if last is not None and not last.after_tree:
                try:
                    last.after_tree = await gitwork.snapshot_state(session.git_info)
                    session._persist()
                except GitWorktreeError:
                    pass
        # Drift is NOT computed here: it is detected lazily when the first
        # client attaches (Session.prompt_resume_drift). A drifted workspace
        # with nobody attached must not report "waiting" nor keep the host
        # alive.
        return session

    async def delete_session(self, name: str) -> CommandResult:
        session = self.sessions.get(name)
        if session is not None:
            if session.state != "idle":
                raise SessionError("wait for the current turn to finish before deleting")
            await self._evict(session)
        deleted = session_store.delete_session(name)
        return CommandResult(cmd_id="", ok=True, payload={"deleted": deleted})

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

    def list_sessions(
        self, *, active: bool | None = None, workspace: str | None = None,
    ) -> list[SessionSummary]:
        summaries: list[SessionSummary] = []
        for session in self.sessions.values():
            if workspace is not None and str(session.workspace) != workspace:
                continue
            summaries.append(self.summary_for(session))
        if active is None or active is False:
            hosted_names = set(self.sessions)
            for name in session_store.list_sessions():
                if name in hosted_names:
                    continue
                if workspace is not None:
                    meta = session_store.load_meta(name)
                    if not meta or meta.get("workspace") != workspace:
                        continue
                summaries.append(self.saved_summary(name))
        if active is True:
            summaries = [s for s in summaries if s.active]
        summaries.sort(key=lambda s: s.last_user_message_at or s.name, reverse=True)
        return summaries

    def summary_for(self, session: Session) -> SessionSummary:
        state = session.state
        if session.pending_ask is not None or (
            session.pending_drift is not None and session.clients_attached
        ):
            state = "waiting"
        # Active derives from the same predicate the grace loop uses — a
        # hosted but fully idle session (no stream, no ask, no running
        # turn) is not active; it is released and listed as saved.
        active = session.is_active
        return SessionSummary(
            name=session.name, workspace=str(session.workspace),
            state=state if active else "saved",
            active=active, clients_attached=session.clients_attached,
            model=session.profile_name, sandbox=session.sandbox_name,
            last_user_message_at=session.last_user_message_at(),
            message_count=len(session.conversation.messages),
            tracking=session.git_info is not None,
        )

    def saved_summary(self, name: str) -> SessionSummary:
        meta = session_store.load_meta(name)
        if meta is None:
            return SessionSummary(name=name, state="saved")
        return SessionSummary(
            name=name, workspace=meta.get("workspace"), state="saved",
            active=False, model=meta.get("profile"),
            sandbox=meta.get("sandbox_name"),
            last_user_message_at=meta.get("last_user_message_at"),
            message_count=meta.get("message_count", 0),
            tracking=bool(meta.get("tracking")),
        )

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
        sessions = [self.summary_for(s) for s in self.sessions.values()]
        by_state: dict[str, int] = {}
        for summary in sessions:
            by_state[summary.state] = by_state.get(summary.state, 0) + 1
        return ServerStats(
            host=self.server_info(),
            started_at=self._started_at_wall,
            sessions_by_state=by_state,
            clients_attached=sum(s.clients_attached for s in sessions),
            total_input_tokens=sum(
                s.controller.usage_total.input_tokens for s in self.sessions.values()
            ),
            total_output_tokens=sum(
                s.controller.usage_total.output_tokens for s in self.sessions.values()
            ),
            sessions=sessions,
        )

    def config_info(self) -> protocol.ConfigInfo:
        """Names and non-secret fields for the client's pickers (never API
        keys). A broken config falls back to an empty (default) config."""
        try:
            sectioned = load_sectioned(self.config_path)
        except ConfigError:
            sectioned = SectionedConfig()
        models = [
            {
                "name": name,
                "display_name": p.display_name,
                "cot_strength": p.cot_strength.value,
                "max_context": p.max_context,
                "supports_vision": p.supports_vision,
                "auto_compact": p.auto_compact,
            }
            for name, p in sectioned.models.items()
        ]
        sandboxes = [{"name": name} for name in sectioned.sandboxes]
        presets = list(sectioned.presets)
        from . import theme
        default_model = sectioned.default_preset_model()
        return protocol.ConfigInfo(
            models=models, sandboxes=sandboxes, presets=presets,
            themes=list(theme.THEMES),
            default_model=default_model,
            default_model_display_name=(
                sectioned.models[default_model].display_name
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
            active[path] = active.get(path, 0) + 1
        hosted_names = set(self.sessions)
        for name in session_store.list_sessions():
            # Hosted sessions are persisted at creation; skip them so they
            # aren't double-counted (as list_sessions already does).
            if name in hosted_names:
                continue
            meta = session_store.load_meta(name)
            if meta and meta.get("workspace"):
                ws = meta["workspace"]
                counts[ws] = counts.get(ws, 0) + 1
        return [
            protocol.WorkspaceInfo(
                path=path, session_count=count, active_sessions=active.get(path, 0),
            )
            for path, count in sorted(counts.items())
        ]

    # ------------------------------------------------------------------
    # lifecycle

    async def serve(self, stats_interval: float | None = None) -> None:
        """Run until shutdown: bind the unix socket, arm the grace timer,
        run until the shutdown event (or print stats every
        ``stats_interval`` seconds in foreground mode), then tear everything
        down."""
        HOST_DIR.mkdir(parents=True, mode=0o700, exist_ok=True)
        server, server_task = await _start_unix_server(self)
        self._arm_grace()
        try:
            if stats_interval is None:
                await self._shutdown_event.wait()
            else:
                while not self._shutdown_event.is_set():
                    stats = self.stats()
                    print(
                        f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
                        f"{_format_stats(stats)}",
                        flush=True,
                    )
                    await asyncio.sleep(stats_interval)
        finally:
            if self._grace_handle is not None:
                self._grace_handle.cancel()
            server.should_exit = True
            try:
                await asyncio.wait_for(server_task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                server_task.cancel()
            for session in list(self.sessions.values()):
                await session.stop()
            if self.unix_socket.exists():
                self.unix_socket.unlink()

    def _on_session_idle(self, session: Session) -> None:
        if self.sessions.get(session.name) is not session:
            return
        if session.is_active:
            return
        session._persist()
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._release_session(session))
            self._arm_grace()
        except RuntimeError:
            pass

    def _arm_grace(self) -> None:
        """(Re)schedule the shutdown check GRACE_SECONDS out. A
        non-persistent host shuts down once the grace passes with nothing
        active — a client between connections finds the host alive because
        every idle notification re-arms the timer."""
        if self.persistent:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._grace_handle is not None:
            self._grace_handle.cancel()
        self._grace_handle = loop.call_later(GRACE_SECONDS, self._grace_expired)

    def _grace_expired(self) -> None:
        if self._shutdown_event.is_set():
            return
        if any(s.is_active for s in self.sessions.values()):
            # Still active (or active again): re-check after another grace
            # period — the next idle notification re-arms the timer too.
            self._arm_grace()
            return
        self._shutdown_event.set()

    async def _release_session(self, session: Session) -> None:
        """Evict an inactive session: stop its loops and sandbox shell and
        drop it from the host. The saved file is kept (it was just
        persisted); /sessions reopens it on demand."""
        if self.sessions.get(session.name) is not session:
            return
        if session.is_active:
            return  # a client re-attached or a turn started in the meantime
        await self._evict(session)
        if session.sandbox_session is not None:
            await session.sandbox_session.close()
            session.sandbox_session = None

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
    server = uvicorn.Server(config)
    task = asyncio.get_running_loop().create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    os.chmod(host.unix_socket, 0o600)
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


def run_foreground() -> None:
    """``xarness --server``: run the host in the foreground, persistent,
    and print the stats once a second. If a live host already exists, print
    its stats instead of binding (a live socket is never unlinked)."""
    HOST_DIR.mkdir(parents=True, mode=0o700, exist_ok=True)
    if _socket_alive():
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
        asyncio.run(_serve_forever(host))
    finally:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()


def _format_stats(stats: ServerStats) -> str:
    """The one-line session/client/token summary shared by the foreground
    stats printer and the periodic live-stats loop."""
    states = ", ".join(
        f"{k}={v}" for k, v in sorted(stats.sessions_by_state.items())
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
    except Exception as exc:  # noqa: BLE001
        print(f"error: could not read the live host's stats: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    print(
        f"xarness host already running (pid {stats.host.pid}):\n"
        f"  socket: {stats.host.unix_socket}\n"
        f"  {_format_stats(stats)}",
        flush=True,
    )


async def _serve_forever(host: Host) -> None:
    """``xarness --server``: run the host in the foreground, persistent,
    printing the stats once a second."""
    info = host.server_info()
    print(
        f"xarness host: pid {info.pid}, protocol v{info.protocol_version}\n"
        f"  socket: {info.unix_socket}\n"
        f"  docs:   (unix socket) GET /v1/health, /v1/server/stats, /docs",
        flush=True,
    )
    await host.serve(stats_interval=1.0)


def main(argv: list[str] | None = None) -> None:
    """Host process entrypoint: ``python -m xarness.host`` (auto-start mode)."""
    host = Host(persistent=False)
    try:
        asyncio.run(host.serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
