"""FastAPI app exposing sessions over the host's unix socket.

Every route is typed with the pydantic models from :mod:`xarness.protocol`,
so FastAPI generates the OpenAPI docs (``/docs``, ``/redoc``,
``/openapi.json``) straight from the same shapes the clients use.

Handlers run commands through the session's serialized ``execute`` and
await the result — they never touch the controller. Pure read-only queries
(snapshots, diffs, lists) bypass that lock; they are safe because only the
session mutates that state.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import __version__, protocol, session_store
from .config import ConfigError
from .host import Host
from .images import sniff_mime
from .protocol import (
    CommandEnvelope,
    CommandResult,
    ServerInfo,
    ServerStats,
    SessionSummary,
    Snapshot,
)
from .session import Session, SessionError

# Cap on one uploaded image blob (base64 inflation happens client-side; this
# bounds the raw bytes the host buffers).
MAX_IMAGE_UPLOAD = 20 * 1024 * 1024


def create_app(host: Host) -> FastAPI:
    """Build the FastAPI app (served on the unix socket only)."""
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

    app.include_router(sessions_router(host))
    app.include_router(meta_router(host))
    return app


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
    ) -> list[SessionSummary]:
        """``stats`` adds pending-diff totals per hosted session — one git
        snapshot each, so it is opt-in (the CLI listing leaves it off)."""
        return await host.list_sessions(
            active=active, workspace=workspace, stats=stats,
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
        "/sessions/{name}/load",
        response_model=SessionSummary,
        summary="Host a saved session (idempotent)",
    )
    async def load_session(name: str) -> SessionSummary:
        session = await host.load_session(name)
        return session.summary()

    @router.post(
        "/sessions/{name}/unload",
        response_model=protocol.DeleteResult,
        summary="Stop hosting a session (it stays saved on disk)",
    )
    async def unload_session(name: str) -> protocol.DeleteResult:
        return protocol.DeleteResult(deleted=await host.unload_session(name))

    @router.get(
        "/sessions/{name}",
        response_model=Snapshot,
        summary="Snapshot of a session (messages, status, live round)",
    )
    async def get_session(name: str) -> Snapshot:
        """The session's snapshot. A saved-but-released session is re-hosted
        on demand (cheap: constructing a Session starts no shell); the
        reaper evicts it again once nothing is attached."""
        return (await _session_or_load(host, name)).snapshot()

    @router.get(
        "/sessions/{name}/events",
        summary="SSE event stream: one snapshot event, then live events",
        response_class=StreamingResponse,
        responses={
            200: {
                "description": (
                    "text/event-stream. The first frame is `snapshot` (a "
                    "Snapshot); then `event` frames carrying Event models "
                    "(discriminated on `type`). The snapshot's "
                    "`partial_round` renders the in-flight round. "
                    "Reconnects always receive a fresh snapshot."
                ),
            },
        },
    )
    async def events(name: str) -> StreamingResponse:
        session = await _session_or_load(host, name)
        return StreamingResponse(
            _event_stream(session),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post(
        "/sessions/{name}/commands",
        response_model=CommandResult,
        summary="Send one command (CommandEnvelope); the reply goes only to the issuer",
    )
    async def post_command(name: str, envelope: CommandEnvelope) -> CommandResult:
        session = await _session_or_load(host, name)
        return await session.execute(envelope.cmd_id, envelope.command)

    @router.get(
        "/sessions/{name}/images/{image_id}",
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
        "/sessions/{name}/images/{image_id}",
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

    @router.patch(
        "/sessions/{name}",
        response_model=protocol.CommandResult,
        summary="Update session settings (mode, model, sandbox, filters, refs)",
    )
    async def patch_session(name: str, body: protocol.SessionUpdate) -> protocol.CommandResult:
        session = await _session_or_load(host, name)
        return await session.update_settings(body)

    @router.get(
        "/sessions/{name}/diff",
        response_model=protocol.DiffInfo,
        summary="Pending changes: summary and per-file diffs",
    )
    async def diff(name: str, path: str | None = None) -> protocol.DiffInfo:
        session = await _session_or_load(host, name)
        return await session.diff_info(path)

    @router.delete(
        "/sessions/{name}",
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


async def _event_stream(session: Session) -> AsyncIterator[str]:
    """SSE frames: a fresh snapshot first, then live events. Reconnects
    always re-snapshot: on a local unix socket the only real disconnect is
    host death, and replaying deltas across a history rewrite would corrupt
    the transcript."""
    # (Re)attach time is when resume drift is detected (never at load time —
    # a drifted workspace with nobody attached must not report "waiting" nor
    # keep the host alive). Run it AFTER the snapshot: the check snapshots the
    # workspace's git tree, which can take a while on a large repo; blocking
    # the first frame on it made resume feel slow. It emits StatusChanged when
    # it finds drift, so the client still gets the prompt.
    snapshot, queue = session.subscribe()
    yield _sse("snapshot", snapshot)
    drift_task = asyncio.create_task(session.prompt_resume_drift())
    try:
        while True:
            event = await queue.get()
            yield _sse("event", event)
            if isinstance(event, protocol.SessionDeleted):
                break
    finally:
        drift_task.cancel()
        session.unsubscribe(queue)


def _sse(event_name: str, payload: Any) -> str:
    data = (
        payload.model_dump_json()
        if hasattr(payload, "model_dump_json") else json.dumps(payload)
    )
    lines = [f"event: {event_name}", f"data: {data}"]
    return "\n".join(lines) + "\n\n"
