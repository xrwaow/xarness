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
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, protocol, session_store
from .config import (
    DEFAULT_CONFIG_PATH, ConfigError, ProviderProfile, load_sectioned,
    resolve_api_key,
)
from .gitwork import GitInfo, GitWorktreeError
from .protocol import CommandResult, SessionSummary, ServerInfo, ServerStats
from .sandbox import SandboxConfig, SandboxSession, SandboxUnavailable
from .session import Session, SessionError

import hashlib

HOST_DIR = Path("~/.local/share/xarness").expanduser()
SOCKET_PATH = HOST_DIR / "host.sock"
LOCK_PATH = HOST_DIR / "host.lock"
BASELINES_DIR = HOST_DIR / "baselines"

# How long the host lingers with no active session before exiting
# (auto-started hosts only; a foreground --server host is persistent).
GRACE_SECONDS = 5.0

# Seconds a client waits for a freshly spawned host's socket to appear.
SPAWN_TIMEOUT = 15.0


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    ) -> dict:
        """Everything Session.create/load needs, from the sectioned config.

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
        # Old sessions persisted profile.model_id instead of the profile
        # key; a last-used value that isn't a known name must fall through
        # to the preset/default instead of failing the request.
        if last_model is not None and last_model not in sectioned.models:
            last_model = None
        if last_sandbox is not None and last_sandbox not in sectioned.sandboxes:
            last_sandbox = None
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
            from .config import SandboxSettings
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

        sandbox_config: SandboxConfig | None = None
        sandbox_session: SandboxSession | None = None
        try:
            sandbox_config = SandboxConfig(
                workspace=ws,
                subtree="",
                allow_network=settings.allow_network,
                gpu_access=settings.gpu_access,
                use_container=settings.use_container,
                filter_list=list(settings.filter_list),
                timeout_seconds=settings.default_timeout,
                external_refs=SandboxConfig.resolve_auto_refs(
                    ws, settings.auto_include_refs,
                ),
                git_dir=git_info.git_dir if git_info is not None else None,
            )
            sandbox_session = SandboxSession(sandbox_config)
        except SandboxUnavailable as exc:
            # Same policy as the old CLI path: tools disabled, chat works.
            sandbox_config = sandbox_session = None

        return {
            "workspace": ws,
            "profile": profile,
            "profile_name": profile_name,
            "api_key": api_key,
            "config_path": self.config_path,
            "sandbox": sandbox_config,
            "sandbox_session": sandbox_session,
            "git_info": git_info,
            "auto_compact": profile.auto_compact,
            "tool_output_limit": settings.tool_output_limit,
            "sandbox_name": sandbox_key,
            "sandbox_settings": settings,
        }

    @staticmethod
    async def _setup_tracking(
        workspace: Path, session_name: str, allow_init: bool,
    ) -> tuple[GitInfo | None, list[str]]:
        from . import gitwork
        try:
            return await gitwork.setup_tracking(workspace, session_name, allow_init=allow_init)
        except GitWorktreeError:
            return None, []

    # ------------------------------------------------------------------
    # shared per-workspace baseline

    def baseline_record_path(self, workspace: Path) -> Path:
        """One baseline record per workspace directory (hash-keyed, so
        every session on that directory shares it)."""
        digest = hashlib.sha256(str(workspace).encode()).hexdigest()[:16]
        return BASELINES_DIR / f"{digest}.json"

    def read_baseline(self, workspace: Path) -> tuple[str | None, int]:
        try:
            data = json.loads(self.baseline_record_path(workspace).read_text())
            return data.get("baseline_tree"), int(data.get("epoch", 0))
        except (OSError, ValueError, TypeError):
            return None, 0

    def write_baseline(self, workspace: Path, tree: str) -> int:
        """Record the workspace's accepted baseline; returns the new epoch."""
        _, epoch = self.read_baseline(workspace)
        epoch += 1
        path = self.baseline_record_path(workspace)
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        path.write_text(json.dumps({
            "workspace": str(workspace), "baseline_tree": tree, "epoch": epoch,
        }))
        return epoch

    def share_baseline(self, changed: Session, info: GitInfo, sha: str) -> None:
        """After /accept in one session: move the shared baseline record and
        update every other live session on the same workspace immediately."""
        self.write_baseline(info.workspace, sha)
        for session in list(self.sessions.values()):
            if session is changed or session.git_info is None:
                continue
            if session.git_info.workspace != info.workspace:
                continue
            session.accept_shared_baseline(sha)

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
        first_message: str | None = None,
    ) -> Session:
        name = session_store.new_session_name()
        parts = await self._build_parts(
            workspace=workspace, model=model, sandbox=sandbox, preset=preset,
            no_init_repo=no_init_repo, network=network, gpu=gpu,
            session_name=name,
        )
        session = Session.create(name=name, **parts)
        self.sessions[name] = session
        session.start()
        session.baseline_callback = self.share_baseline
        self.watch_session(session)
        if session.git_info is not None:
            tree, _epoch = self.read_baseline(session.git_info.workspace)
            if tree is None:
                self.write_baseline(session.git_info.workspace, session.git_info.baseline_tree)
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
        session = Session.load(name, **parts)
        # Old session files may carry a bogus profile key (model_id instead
        # of the profile name); keep the freshly resolved one in that case.
        try:
            sectioned = load_sectioned(self.config_path)
        except ConfigError:
            sectioned = None
        if sectioned is not None and session.profile_name not in sectioned.models:
            session.profile_name = parts["profile_name"]
            session.profile = parts["profile"]
        self.sessions[name] = session
        session.start()
        session.baseline_callback = self.share_baseline
        self.watch_session(session)
        # Saved sessions re-sync with the shared per-workspace baseline
        # record: another session may have accepted changes while this one
        # was closed.
        if session.git_info is not None:
            tree, _epoch = self.read_baseline(session.git_info.workspace)
            if tree is not None and tree != session.git_info.baseline_tree:
                session.git_info.baseline_tree = tree
                session._rewrite_checkpoints(tree)
                session._persist()
        # Drift is NOT computed here: it is detected lazily when the first
        # client attaches (Session.prompt_resume_drift). A drifted workspace
        # with nobody attached must not report "waiting" nor keep the host
        # alive.
        return session

    async def delete_session(self, name: str) -> CommandResult:
        session = self.sessions.get(name)
        if session is not None:
            if session._state != "idle":
                raise SessionError("wait for the current turn to finish before deleting")
            await session.stop()
            self.sessions.pop(name, None)
            session._emit(protocol.SessionDeleted(name=name))
            for queue in list(session._subscribers):
                session.unsubscribe(queue)
        deleted = session_store.delete_session(name)
        return CommandResult(cmd_id="", ok=True, payload={"deleted": deleted})

    async def unload_session(self, name: str) -> bool:
        """Stop hosting a session ("stop and quit"): the turn is cancelled,
        the session leaves the host's list, and its subscribers are released.
        The saved session file is kept — /sessions reopens it."""
        session = self.sessions.get(name)
        if session is None:
            return False
        if session._turn_task is not None and not session._turn_task.done():
            session._turn_task.cancel()
        await session.stop()
        self.sessions.pop(name, None)
        session._emit(protocol.Notice(
            text="session closed — it stays saved on disk; /sessions to reopen",
            level="info",
        ))
        session._emit(protocol.SessionDeleted(name=name))
        for queue in list(session._subscribers):
            session.unsubscribe(queue)
        return True

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
        row = session.stats_row()
        state = row["state"]
        if session._pending_ask is not None or (
            session._pending_drift is not None and session.clients_attached
        ):
            state = "waiting"
        # Active derives from the same predicate the grace loop uses — a
        # hosted but fully idle session (no stream, no ask, no running
        # turn) is not active; it is released and listed as saved.
        active = self.is_active_session(session)
        return SessionSummary(
            name=row["name"], workspace=row["workspace"],
            state=state if active else "saved",
            active=active, clients_attached=row["clients_attached"],
            model=row["model"], sandbox=row["sandbox"],
            last_user_message_at=row["last_user_message_at"],
            message_count=row["message_count"], tracking=row["tracking"],
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

    async def diff_info(self, session: Session, path: str | None) -> protocol.DiffInfo:
        stat = await session.diff_summary()
        info = protocol.DiffInfo()
        if stat is None:
            return info
        files = stat.files
        if path is not None:
            files = [f for f in files if f.path == path]
        info.files = [
            protocol.DiffFileInfo(
                path=f.path, additions=f.additions, deletions=f.deletions,
                is_new=f.is_new, source=f.source,
            )
            for f in files
        ]
        info.additions = sum(f.additions for f in files)
        info.deletions = sum(f.deletions for f in files)
        if path is not None:
            # The per-file click-through needs the unified diff text; the
            # summary poll (no path) must stay cheap, so only fetch it here.
            info.text = await session.diff_text(path)
        return info

    # ------------------------------------------------------------------
    # server info / stats

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
        keys)."""
        try:
            sectioned = load_sectioned(self.config_path)
        except ConfigError:
            sectioned = None
        models = []
        if sectioned is not None:
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
        sandboxes = (
            [{"name": name} for name in sectioned.sandboxes]
            if sectioned is not None else [{"name": "default"}]
        )
        presets = (
            list(sectioned.presets) if sectioned is not None else ["default"]
        )
        from . import theme
        default_model = (
            sectioned.default_preset_model() if sectioned is not None else None
        )
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
        for name in session_store.list_sessions():
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

    async def serve(self) -> None:
        """Run until shutdown: the unix socket + the grace timer."""
        HOST_DIR.mkdir(parents=True, mode=0o700, exist_ok=True)
        server = await _start_unix_server(self)
        grace = asyncio.get_running_loop().create_task(self._grace_loop())
        try:
            await self._shutdown_event.wait()
        finally:
            grace.cancel()
            server.close()
            await server.wait_closed()
            for session in list(self.sessions.values()):
                await session.stop()
            if self.unix_socket.exists():
                self.unix_socket.unlink()

    def is_active_session(self, session: Session) -> bool:
        """A session is active while it is running, compacting, waiting on
        an ask, or has an open event stream.

        A pending drift WITHOUT an attached client does not count: nothing
        can resolve it, so it must neither report "waiting" nor keep the
        host alive (the prompt is (re)computed when a client attaches)."""
        return (
            session._state != "idle"
            or session._pending_ask is not None
            or bool(session._subscribers)
        )

    def watch_session(self, session: Session) -> None:
        """Install the idle hook so a fully inactive session is persisted
        and released (its sandbox shell closed) instead of lingering in
        host.sessions; /sessions then lists it as saved."""
        session.idle_callback = self._on_session_idle

    def _on_session_idle(self, session: Session) -> None:
        if self.sessions.get(session.name) is not session:
            return
        if self.is_active_session(session):
            return
        session._persist()
        try:
            asyncio.get_running_loop().create_task(self._release_session(session))
        except RuntimeError:
            pass

    async def _release_session(self, session: Session) -> None:
        """Evict an inactive session: stop its loops and sandbox shell and
        drop it from the host. The saved file is kept (it was just
        persisted); /sessions reopens it on demand."""
        if self.sessions.get(session.name) is not session:
            return
        if self.is_active_session(session):
            return  # a client re-attached or a turn started in the meantime
        self.sessions.pop(session.name, None)
        await session.stop()
        if session.sandbox_session is not None:
            await session.sandbox_session.close()
            session.sandbox_session = None

    async def _grace_loop(self) -> None:
        while not self._shutdown_event.is_set():
            if not self.persistent and not any(
                self.is_active_session(s) for s in self.sessions.values()
            ):
                # Grace: linger briefly so a client that is between
                # connections finds the host alive.
                deadline = time.monotonic() + GRACE_SECONDS
                while time.monotonic() < deadline:
                    await asyncio.sleep(0.25)
                    if any(
                        self.is_active_session(s) for s in self.sessions.values()
                    ):
                        break
                else:
                    await self.request_shutdown()
                    break
            await asyncio.sleep(0.5)

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
    return _ServerHandle(server, task)


class _ServerHandle:
    """Minimal close/wait surface over a uvicorn.Server run as a task."""

    def __init__(self, server, task: asyncio.Task) -> None:
        self._server = server
        self._task = task

    def close(self) -> None:
        self._server.should_exit = True

    async def wait_closed(self) -> None:
        try:
            await asyncio.wait_for(self._task, timeout=5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._task.cancel()


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


def _socket_alive() -> bool:
    if not SOCKET_PATH.exists():
        return False
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        sock.connect(str(SOCKET_PATH))
        sock.close()
        return True
    except (OSError, socket.timeout):
        # Refused / stale: no host behind the file.
        try:
            SOCKET_PATH.unlink()
        except OSError:
            pass
        return False


def _socket_alive_at(path: Path) -> bool:
    """Whether something is listening at ``path`` (no unlinking)."""
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


def _wait_for_socket(timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if SOCKET_PATH.exists():
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                sock.connect(str(SOCKET_PATH))
                sock.close()
                return
            except OSError:
                pass
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
    states = ", ".join(
        f"{k}={v}" for k, v in sorted(stats.sessions_by_state.items())
    ) or "none"
    print(
        f"xarness host already running (pid {stats.host.pid}):\n"
        f"  socket: {stats.host.unix_socket}\n"
        f"  sessions: {len(stats.sessions)} ({states}), "
        f"clients: {stats.clients_attached}, "
        f"tokens: {stats.total_input_tokens} in / "
        f"{stats.total_output_tokens} out",
        flush=True,
    )


async def _serve_forever(host: Host) -> None:
    HOST_DIR.mkdir(parents=True, mode=0o700, exist_ok=True)
    server = await _start_unix_server(host)
    info = host.server_info()
    print(
        f"xarness host: pid {info.pid}, protocol v{info.protocol_version}\n"
        f"  socket: {info.unix_socket}\n"
        f"  docs:   (unix socket) GET /v1/health, /v1/server/stats, /docs",
        flush=True,
    )
    try:
        while True:
            stats = host.stats()
            states = ", ".join(
                f"{k}={v}" for k, v in sorted(stats.sessions_by_state.items())
            ) or "none"
            print(
                f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
                f"sessions: {len(stats.sessions)} ({states}), "
                f"clients: {stats.clients_attached}, "
                f"tokens: {stats.total_input_tokens} in / "
                f"{stats.total_output_tokens} out",
                flush=True,
            )
            await asyncio.sleep(1.0)
    finally:
        server.close()
        await server.wait_closed()
        for session in list(host.sessions.values()):
            await session.stop()
        if host.unix_socket.exists():
            host.unix_socket.unlink()


def main(argv: list[str] | None = None) -> None:
    """Host process entrypoint: ``python -m xarness.host`` (auto-start mode)."""
    host = Host(persistent=False)
    try:
        asyncio.run(host.serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
