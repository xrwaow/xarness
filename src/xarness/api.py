"""FastAPI app exposing sessions over the host's unix socket.

Every route is typed with the pydantic models from :mod:`xarness.protocol`,
so FastAPI generates the OpenAPI docs (``/docs``, ``/redoc``,
``/openapi.json``) straight from the same shapes the clients use.

Handlers run commands through the session's serialized ``execute`` and
await the result — they never touch session internals. Pure read-only queries
(snapshots, diffs, lists) bypass that lock; they are safe because only the
session mutates that state.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import mimetypes
import secrets
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, AsyncIterator

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, protocol, session_store
from .config import ConfigError
from .host import DEFAULT_WEB_HOST, DEFAULT_WEB_PORT, Host, WebListen
from .images import sniff_mime
from .protocol import (
    CommandResult,
    ServerInfo,
    ServerStats,
    SessionSummary,
    Snapshot,
    ToolsInfo,
)
from .session import Session, SessionError
from .util import RangeError

# Cap on one uploaded image blob (base64 inflation happens client-side; this
# bounds the raw bytes the host buffers).
MAX_IMAGE_UPLOAD = 20 * 1024 * 1024

# Shared OpenAPI text for the list-window parameter.
# A Python-style slice into a list; negative bounds count from the end.
Range = Annotated[
    str | None,
    Query(alias="range", description=(
        "Python-style slice into the list, e.g. [-40:] for the newest 40, "
        "[10:20], or [:5]; negative bounds count from the end."
    )),
]

LOG = logging.getLogger(__name__)

# The web UI's static files. The repo layout (``<root>/webui``) is what an
# editable install sees; the packaged copy lives next to this module.
WEBUI_DIRS = (
    Path(__file__).resolve().parent / "webui",
    Path(__file__).resolve().parents[2] / "webui",
)


def webui_dir() -> Path | None:
    """The first existing web UI directory, or None (no static mount)."""
    return next((path for path in WEBUI_DIRS if (path / "index.html").is_file()), None)


def create_app(
    host: Host, *, auth: Callable[[], str | None] | None = None,
    admin: bool = True,
) -> FastAPI:
    """Build the FastAPI app.

    ``auth`` is only given to the TCP listener ``xarness --web`` starts: the
    web UI is reachable from the network, so it must prove it was told the
    token the host printed. It returns the token *at request time*, which is
    what lets the listener retune it without a rebind. The unix-socket app
    (the TUI's, this CLI's) passes none and is unauthenticated exactly as
    before — its directory is 0700 and the socket 0600.

    ``admin`` mounts the host-control routes (the browser listener's address
    and token). Only the unix-socket app has them: a client holding the web
    token must not be able to read, rotate or drop the gate that token is.
    """
    app = FastAPI(
        title="xarness",
        version=__version__,
        description=(
            "Headless chat sessions for the xarness agent. Clients attach "
            "to a session's event stream (SSE) and drive it with commands."
        ),
    )
    app.state.host = host

    @app.exception_handler(SessionError)
    async def _session_error_handler(request: Request, exc: SessionError):
        """Every SessionError (bad request, unknown session, idle-only
        violations) maps to one 400 with the message — handlers don't need
        their own try/except."""
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(RangeError)
    async def _range_error_handler(request: Request, exc: RangeError):
        """A malformed ``range`` query value is the client's mistake: one
        400 naming the expected syntax, like every other bad request."""
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def _unexpected_handler(request: Request, exc: Exception):
        """An unexpected failure still names itself to the client, and the log
    keeps the traceback: an auto-spawned host's stderr is /dev/null."""
        LOG.error(
            "unhandled error on %s %s", request.method, request.url.path,
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": f"{type(exc).__name__}: {exc}"},
        )

    app.include_router(sessions_router(host))
    app.include_router(meta_router(host))
    if admin:
        # Host control serves the browser listener's token: unix socket only.
        app.include_router(host_router(host))
    if auth is not None:
        _require_token(app, auth)
    # Mounts must come last: a mount at "/" matches every remaining path, so
    # it would shadow the API routes registered above it.
    _mount_webui(app)
    return app


def _require_token(app: FastAPI, token_of: Callable[[], str | None]) -> None:
    """Bearer-token gate for the API routes of a network-reachable app.

    Only ``/v1/*`` is gated: the static UI itself is public (it is what asks
    for the token), and the OpenAPI docs carry no data. A listener without a
    token is loopback-only (see :func:`web_listen`) and stays open exactly
    as it always was."""

    @app.middleware("http")
    async def _check_token(request: Request, call_next):
        if request.url.path.startswith("/v1"):
            token = token_of()
            header = request.headers.get("authorization", "")
            scheme, _, value = header.partition(" ")
            if token is not None and (
                scheme.lower() != "bearer"
                or not secrets.compare_digest(value.strip(), token)
            ):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "missing or invalid host token"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
        return await call_next(request)


def _mount_webui(app: FastAPI) -> None:
    """Serve ``webui/`` at ``/`` when it exists (after every API route)."""
    directory = webui_dir()
    if directory is None:
        LOG.debug("no webui directory found; serving the API only")
        return
    # Python's mimetypes does not know the manifest on every platform, and a
    # PWA manifest served as octet-stream is ignored by browsers.
    mimetypes.add_type("application/manifest+json", ".webmanifest")

    class _WebUIStaticFiles(StaticFiles):
        """Static files that must be revalidated on every load.

        Without a ``Cache-Control`` header browsers apply *heuristic*
        freshness (a slice of the file's age), so a phone that reloads the
        page can keep running yesterday's ``style.css`` while a desktop — or
        a localhost page, where a service worker is allowed and revalidates —
        shows the current one. ``no-cache`` still lets the ETag answer a
        cheap 304, so the cost on a LAN is one round trip.
        """

        def file_response(self, *args: Any, **kwargs: Any) -> Response:
            response = super().file_response(*args, **kwargs)
            response.headers.setdefault("cache-control", "no-cache")
            return response

    app.mount("/", _WebUIStaticFiles(directory=directory, html=True), name="webui")


async def _session_or_load(host: Host, name: str) -> Session:
    """The hosted session, or (saved but released — an idle session is
    evicted from the host) freshly re-hosted from its saved file."""
    session = host.sessions.get(name)
    if session is not None:
        return session
    try:
        return await host.load_session(name)
    except SessionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def sessions_router(host: Host) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["sessions"])

    @router.get(
        "/sessions",
        response_model=list[SessionSummary],
        summary="List active and saved sessions",
    )
    async def list_sessions(
        active: bool | None = None, workspace: str | None = None,
        stats: bool = False,
        window: Range = None,
    ) -> list[SessionSummary]:
        """``stats`` is accepted for compatibility and ignored: the pending
        diff totals are cached on the host and ride along on every row (this
        route never runs git). ``range`` slices the recency-sorted list
        (newest first), so a large history can be read a page at a time."""
        return await host.list_sessions(
            active=active, workspace=workspace, stats=stats, window=window,
        )

    @router.post(
        "/sessions",
        response_model=SessionSummary,
        status_code=201,
        summary="Create a session",
    )
    async def create_session(body: protocol.CreateSessionRequest) -> SessionSummary:
        session = await host.create_session(body)
        return session.summary()

    @router.post(
        "/sessions/{name:path}/load",
        response_model=SessionSummary,
        summary="Host a saved session (idempotent)",
    )
    async def load_session(name: str) -> SessionSummary:
        session = await host.load_session(name)
        return session.summary()

    @router.post(
        "/sessions/{name:path}/unload",
        response_model=protocol.DeleteResult,
        summary="Stop hosting a session (it stays saved on disk)",
    )
    async def unload_session(name: str) -> protocol.DeleteResult:
        return protocol.DeleteResult(deleted=await host.unload_session(name))

    @router.get(
        "/sessions/{name:path}/events",
        summary="SSE event stream: one snapshot event, then live events",
        response_class=StreamingResponse,
        responses={
            200: {
                "description": (
                    "text/event-stream. The first frame is `snapshot` (a "
                    "Snapshot); then `event` frames carrying Event models "
                    "(discriminated on `type`). Reconnects always receive "
                    "a fresh snapshot (`range` windows its history; `slim` "
                    "drops tool bodies and truncates reasoning — hydrate a "
                    "row from GET …/messages/{index})."
                ),
            },
        },
    )
    async def events(
        name: str, window: Range = None, slim: bool = False,
    ) -> StreamingResponse:
        session = await _session_or_load(host, name)
        return StreamingResponse(
            _event_stream(session, window, slim),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get(
        "/sessions/{name:path}/messages",
        response_model=protocol.MessagePage,
        summary="One page of older history, oldest-first and turn-aligned",
    )
    async def messages(
        name: str, window: Range = None, slim: bool = False,
    ) -> protocol.MessagePage:
        """The window ``range`` selects, oldest-first and turn-aligned; a
        missing ``range`` is the newest page. Walk back with the page's
        ``start`` (``[start-N:start]``) while ``has_more``. ``slim`` drops
        reasoning, arguments and output (expand to fetch)."""
        return (await _session_or_load(host, name)).message_page(
            window=window, slim=slim,
        )

    @router.get(
        "/sessions/{name:path}/messages/{index}",
        response_model=protocol.MessageBody,
        summary="Full text of one history round (lazy page expansion)",
    )
    async def message_body(
        name: str, index: int, content: bool = True,
    ) -> protocol.MessageBody:
        """``content=False`` is the round's headers: reasoning and tool
        arguments, no tool output."""
        return (await _session_or_load(host, name)).message_body(
            index, content=content,
        )

    @router.post(
        "/sessions/{name:path}/commands",
        response_model=CommandResult,
        summary="Send one command; the reply goes only to the issuer",
    )
    async def post_command(name: str, command: protocol.Command) -> CommandResult:
        session = await _session_or_load(host, name)
        return await session.execute(command)

    @router.get(
        "/sessions/{name:path}/images/{image_id}",
        summary="Fetch one stored image blob by content hash",
        response_class=Response,
    )
    async def get_image(name: str, image_id: str) -> Response:
        session = host.sessions.get(name)
        # The session's own store first: a no_save session keeps blobs in
        # memory only, where the disk lookup below would 404.
        data = session.images.load(image_id) if session is not None else None
        if data is None:
            data = session_store.load_blob(name, image_id)
        if data is None:
            raise HTTPException(status_code=404, detail="image not found")
        return Response(content=data, media_type=sniff_mime(data))

    @router.put(
        "/sessions/{name:path}/images/{image_id}",
        status_code=201,
        summary="Upload one image blob (content-addressed); reference it by hash in SendMessage",
    )
    async def put_image(name: str, image_id: str, request: Request) -> Response:
        session = await _session_or_load(host, name)
        # Cap the body BEFORE reading it: an unbounded upload would let one
        # request balloon the host's memory.
        length = request.headers.get("content-length") or ""
        if length.isdigit() and int(length) > MAX_IMAGE_UPLOAD:
            raise HTTPException(
                status_code=413,
                detail=f"image too large: {length} bytes (limit {MAX_IMAGE_UPLOAD})",
            )
        data = await request.body()
        if not data:
            raise HTTPException(status_code=400, detail="empty image body")
        if len(data) > MAX_IMAGE_UPLOAD:
            raise HTTPException(
                status_code=413,
                detail=f"image too large: {len(data)} bytes (limit {MAX_IMAGE_UPLOAD})",
            )
        # Hash first, store second: a mismatched body must not write a blob
        # under an id nothing will ever reference.
        digest = hashlib.sha256(data).hexdigest()
        if digest != image_id:
            raise HTTPException(
                status_code=400,
                detail=f"body hash {digest} does not match the requested id {image_id}",
            )
        # Through the session's store: a no_save session keeps the blob in
        # memory (and visible to the provider round) instead of on disk.
        # The write (with its up-to-20 MB fsync) runs off the event loop.
        await asyncio.to_thread(session.images.store, data)
        return Response(status_code=201)

    @router.get(
        "/sessions/{name:path}/quit",
        response_model=protocol.QuitVerdict,
        summary="Ask whether closing now needs the continue-headless choice",
    )
    async def quit_verdict(name: str) -> protocol.QuitVerdict:
        """What a disconnecting client should do.

        A read-only query (no command lock, no mutation): the client asks
        before closing instead of trusting a cached status, and the policy
        lives with the state machine it is derived from."""
        return (await _session_or_load(host, name)).quit_verdict()

    @router.get(
        "/sessions/{name:path}/tools",
        response_model=ToolsInfo,
        summary="Effective tools plus configured MCP servers and availability",
    )
    async def tools(name: str) -> ToolsInfo:
        """The /tools modal's data. Lazily starts configured MCP servers,
        so opening the modal also refreshes a failed one."""
        return await (await _session_or_load(host, name)).tools_info()

    @router.get(
        "/sessions/{name:path}/diff",
        response_model=protocol.DiffInfo,
        summary="Pending changes: summary and per-file diffs",
    )
    async def diff(
        name: str, path: str | None = None, author: str | None = None,
    ) -> protocol.DiffInfo:
        session = await _session_or_load(host, name)
        return await session.diff_info(path, author)

    @router.get(
        "/sessions/{name:path}/summary",
        response_model=protocol.SessionSummary,
        summary="One session's listing row (saved sessions are not loaded)",
    )
    async def session_summary(name: str) -> protocol.SessionSummary:
        """The row the list would show, with this session's own pending
        totals — the read a hovering client makes.

        A saved session is answered without hosting it: the totals come from
        the workspace's saved baseline, scoped to that session and its
        subagents (never the whole repo's pending change), so peeking at a
        row does not turn it into an idle hosted session."""
        summary = await host.session_summary(name)
        if summary is None:
            raise HTTPException(
                status_code=404, detail=f"unknown session: {name}",
            )
        return summary

    @router.get(
        "/sessions/{name:path}",
        response_model=Snapshot,
        summary="Snapshot of a session (messages, status, live round)",
    )
    async def get_session(name: str, window: Range = None) -> Snapshot:
        """The session's snapshot. A saved-but-released session is re-hosted
        on demand (cheap: constructing a Session starts no shell); the
        reaper evicts it again once nothing is attached.

        ``range`` windows the history like ``GET …/events``; omit it for
        the whole transcript. Fixed suffixes (``/events``, ``/images/..``,
        ``/quit``, ``/diff``, ``/messages``) are registered first: their
        ``:path`` also matches here, and first match wins."""
        return (await _session_or_load(host, name)).snapshot(window=window)

    @router.delete(
        "/sessions/{name:path}",
        response_model=protocol.DeleteResult,
        summary="Delete a saved session (idle only; the workspace is untouched)",
    )
    async def delete_session(name: str) -> protocol.DeleteResult:
        return await host.delete_session(name)

    return router


def meta_router(host: Host) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["server"])

    @router.get(
        "/workspaces/refs",
        response_model=list[protocol.RefInfo],
        summary="Workspace external references (no session needed)",
    )
    async def workspace_refs(workspace: str) -> list[protocol.RefInfo]:
        return host.workspace_refs(workspace)

    @router.post(
        "/workspaces/refs",
        response_model=list[protocol.RefInfo],
        summary="Add/remove one workspace external reference",
    )
    async def update_workspace_refs(
        body: protocol.WorkspaceRefUpdate,
    ) -> list[protocol.RefInfo]:
        return await host.update_workspace_refs(body)

    @router.get(
        "/workspaces/diff",
        response_model=protocol.DiffInfo,
        summary="Pending changes of a workspace (no session needed)",
    )
    async def workspace_diff(
        workspace: str, path: str | None = None, author: str | None = None,
        scope: protocol.DiffScope = "all",
    ) -> protocol.DiffInfo:
        """A draft chat's /diff: tracking lives with the repo root, so the
        panel works before a session exists."""
        return await host.workspace_diff(
            workspace, path=path, author=author, scope=scope,
        )

    @router.get("/health", response_model=ServerInfo, summary="Host health and protocol version")
    async def health() -> ServerInfo:
        return host.server_info()

    @router.get("/server/stats", response_model=ServerStats, summary="Live server stats")
    async def stats() -> ServerStats:
        return host.stats()

    @router.get("/config", response_model=protocol.ConfigInfo, summary="Model/sandbox/preset names for pickers (never API keys)")
    async def config_info() -> protocol.ConfigInfo:
        return host.config_info()

    @router.put(
        "/config/defaults",
        response_model=protocol.ConfigInfo,
        summary="Persist the default model for future sessions",
    )
    async def update_defaults(
        body: protocol.ConfigDefaults,
    ) -> protocol.ConfigInfo:
        try:
            host.set_default_profile(body.default_profile)
        except (ConfigError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return host.config_info()

    @router.get(
        "/events",
        summary="SSE stream of the session list: one snapshot, then deltas",
        response_class=StreamingResponse,
        responses={
            200: {
                "description": (
                    "text/event-stream. The first frame is `snapshot` (a "
                    "SessionsSnapshot: every listed session, cached diff "
                    "totals included); then `event` frames carrying "
                    "SessionSummaryChanged / SessionsRemoved, discriminated "
                    "on `type`. Never runs git: totals are maintained by the "
                    "host from workspace events. Reconnects receive a fresh "
                    "snapshot."
                ),
            },
        },
    )
    async def events() -> StreamingResponse:
        return StreamingResponse(
            _sessions_stream(host),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/workspaces", response_model=list[protocol.WorkspaceInfo], summary="Known workspace directories")
    async def workspaces() -> list[protocol.WorkspaceInfo]:
        return host.workspaces()

    @router.get(
        "/workspaces/files",
        response_model=list[str],
        summary="File search inside a workspace (the @ popup; runs server-side)",
    )
    async def workspace_files(workspace: str, q: str = "") -> list[str]:
        from .file_search import search_files
        try:
            return await search_files(Path(workspace), q)
        except OSError:
            return []

    return router


def host_router(host: Host) -> APIRouter:
    """Host control: the browser listener and stopping the host.

    Unix socket only (see :func:`create_app`): these can read the web token,
    rotate it, or drop the listener that token guards."""
    router = APIRouter(prefix="/v1/server", tags=["server"])

    @router.get(
        "/web", response_model=protocol.WebInfo,
        summary="The browser listener: address, URLs and token",
    )
    async def web_info() -> protocol.WebInfo:
        return host.web_info()

    @router.put(
        "/web", response_model=protocol.WebInfo,
        summary="Add the browser listener, or change its token",
    )
    async def update_web(body: protocol.WebUpdate) -> protocol.WebInfo:
        web = WebListen(
            bind=body.bind or DEFAULT_WEB_HOST,
            port=DEFAULT_WEB_PORT if body.port is None else body.port,
            token=body.token,
        )
        try:
            return await host.enable_web(web, replace=body.replace)
        except RuntimeError as exc:
            # A bad address, a taken port, or a move that was not asked for:
            # the host is unchanged, and the message says why.
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @router.post(
        "/stop", response_model=protocol.StopResult,
        summary="Stop the host; sessions stay saved on disk",
    )
    async def stop_host() -> protocol.StopResult:
        # Delayed so this response goes out before the listeners close.
        await host.request_shutdown(delay=0.1)
        return protocol.StopResult()

    return router


async def _sessions_stream(host: Host) -> AsyncIterator[str]:
    """SSE frames for the session list: one full snapshot, then deltas.

    Registering the queue before building the snapshot means neither can
    miss an update in between; the overlap is an idempotent upsert on the
    client. The snapshot is a plain read (cached totals), so the first frame
    is immediate even on a big repo."""
    queue = host.watch_sessions()
    try:
        yield _sse("snapshot", protocol.SessionsSnapshot(
            sessions=await host.list_sessions(),
        ))
        async for frame in _queue_frames(queue):
            yield frame
    finally:
        host.unwatch_sessions(queue)


async def _event_stream(
    session: Session, window: str | None = None, slim: bool = False,
) -> AsyncIterator[str]:
    """SSE frames: a fresh snapshot first, then live events. Reconnects
    always re-snapshot: on a local unix socket the only real disconnect is
    host death, and replaying deltas across a history rewrite would corrupt
    the transcript. ``window`` selects that snapshot's history slice and
    ``slim`` its bodies."""
    snapshot, queue = session.subscribe(window=window, slim=slim)
    yield _sse("snapshot", snapshot)
    try:
        async for frame in _queue_frames(queue, (protocol.SessionDeleted,)):
            yield frame
    finally:
        session.unsubscribe(queue)


async def _queue_frames(
    queue: asyncio.Queue[Any], stop_on: tuple[type, ...] = (),
) -> AsyncIterator[str]:
    """The `event` half of an SSE stream: one frame per subscriber queue
    item, ending when a ``stop_on`` event has been sent (a deleted session
    has nothing left to stream)."""
    while True:
        event = await queue.get()
        yield _sse("event", event)
        if stop_on and isinstance(event, stop_on):
            return


def _sse(event_name: str, payload: Any) -> str:
    data = (
        payload.model_dump_json()
        if hasattr(payload, "model_dump_json") else json.dumps(payload)
    )
    lines = [f"event: {event_name}", f"data: {data}"]
    return "\n".join(lines) + "\n\n"
