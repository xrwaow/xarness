"""Typed async Python client for the xarness host API.

One method per endpoint, httpx over the host's unix socket, an SSE reader
with reconnect support, and the protocol models as the only wire shapes.
The TUI is a pure client of this package; it is also the reference for the
later web client.

Typical use::

    client = await XarnessClient.connect_unix()      # auto-starts the host
    async for item in client.attach("my-session"):   # Snapshot, then Events
        ...
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from . import protocol
from .protocol import (
    Command, CommandEnvelope, CommandResult, Event, Snapshot, SessionSummary,
    ServerInfo,
)

async def ensure_host(timeout: float = 5.0) -> None:
    """Make sure a host process is running (spawning one if needed).

    Used by the TUI after a connection loss: the host may have exited while
    nobody was attached."""
    from .host import connect_or_spawn
    await asyncio.to_thread(connect_or_spawn, timeout=timeout)


class ClientError(Exception):
    """The API returned an error (bad request, unknown session, …).

    Transport failures (``httpx.HTTPError``) are wrapped into this too, so
    callers catch one exception type; the SSE reader's reconnect loop is the
    one place that still needs the raw httpx error and reads it directly."""


async def _wrap_http(coro):
    """Await an HTTP request, folding transport errors into ClientError."""
    try:
        return await coro
    except httpx.HTTPError as exc:
        raise ClientError(str(exc) or type(exc).__name__) from exc


class XarnessClient:
    """Typed client for one host."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def _request(
        self, method: str, path: str, *, stream: bool = False, **kwargs: Any,
    ) -> httpx.Response:
        """Send one request; on a dead/unreachable socket, respawn the host
        and retry once.

        The host exits when nothing is active (grace) — a client sitting on
        a draft screen can outlive it, and the next request must
        transparently bring the host back instead of surfacing a raw
        ConnectError. ``stream=True`` connects here (unlike httpx.stream(),
        whose request only goes out at __aenter__), so the retry covers SSE
        too."""
        request = self._http.build_request(method, path, **kwargs)
        try:
            return await self._http.send(request, stream=stream)
        except httpx.ConnectError:
            await ensure_host()
            return await self._http.send(request, stream=stream)

    @classmethod
    async def connect_unix(
        cls, socket_path: Path | None = None, *, spawn: bool = True,
    ) -> "XarnessClient":
        """Connect over the host's unix socket, auto-starting the host.

        The protocol version is checked against /health; on a mismatch with
        no active sessions the stale host is shut down and respawned."""
        from .host import SOCKET_PATH, connect_or_spawn
        if spawn:
            # connect_or_spawn forks/waits on subprocesses — blocking work
            # that must not stall the event loop this coroutine serves.
            await asyncio.to_thread(connect_or_spawn)
        path = socket_path or SOCKET_PATH
        http = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=str(path)),
            base_url="http://localhost",
            timeout=httpx.Timeout(10.0, read=None),
        )
        client = cls(http)
        info = await client.health()
        if info.protocol_version != protocol.PROTOCOL_VERSION:
            # An old host is still running from a previous build. With no
            # active sessions it is safe to replace it; otherwise tell the
            # user rather than killing their work.
            active = []
            try:
                active = await client.sessions(active=True)
            except ClientError:
                pass
            await http.aclose()
            if active:
                raise ClientError(
                    f"host speaks v{info.protocol_version}, this client speaks "
                    f"v{protocol.PROTOCOL_VERSION}; finish its {len(active)} active "
                    "session(s) and restart the host"
                )
            from .host import stop_host
            await asyncio.to_thread(stop_host, info.pid, path)
            return await cls.connect_unix(socket_path, spawn=spawn)
        return client

    async def aclose(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------
    # meta

    async def health(self) -> ServerInfo:
        return ServerInfo.model_validate(await self._get("/v1/health"))

    async def config(self) -> protocol.ConfigInfo:
        return protocol.ConfigInfo.model_validate(await self._get("/v1/config"))

    async def set_default_profile(self, name: str) -> protocol.ConfigInfo:
        """Ask the host to persist the default model for future sessions."""
        return protocol.ConfigInfo.model_validate(
            await self._put("/v1/config/defaults", {"default_profile": name})
        )

    async def search_files(self, workspace: str, q: str) -> list[str]:
        return list(await self._get(
            "/v1/workspaces/files", params={"workspace": workspace, "q": q},
        ))

    async def workspace_refs(self, workspace: str) -> list[protocol.RefInfo]:
        return [
            protocol.RefInfo.model_validate(item)
            for item in await self._get(
                "/v1/workspaces/refs", params={"workspace": workspace},
            )
        ]

    async def update_workspace_refs(
        self, workspace: str, path: str, add: bool,
    ) -> list[protocol.RefInfo]:
        return [
            protocol.RefInfo.model_validate(item)
            for item in await self._post("/v1/workspaces/refs", {
                "workspace": workspace, "path": path, "add": add,
            })
        ]

    # ------------------------------------------------------------------
    # sessions

    async def sessions(
        self, *, active: bool | None = None, workspace: str | None = None,
    ) -> list[SessionSummary]:
        params: dict[str, Any] = {}
        if active is not None:
            params["active"] = str(active).lower()
        if workspace is not None:
            params["workspace"] = workspace
        return [
            SessionSummary.model_validate(s)
            for s in await self._get("/v1/sessions", params=params)
        ]

    async def create_session(
        self, request: protocol.CreateSessionRequest,
    ) -> SessionSummary:
        return SessionSummary.model_validate(
            await self._post(
                "/v1/sessions", request.model_dump(exclude_none=True),
            )
        )

    async def load_session(self, name: str) -> SessionSummary:
        return SessionSummary.model_validate(
            await self._post(f"/v1/sessions/{name}/load", {})
        )

    async def diff(self, name: str, path: str | None = None) -> protocol.DiffInfo:
        params = {"path": path} if path else None
        return protocol.DiffInfo.model_validate(
            await self._get(f"/v1/sessions/{name}/diff", params=params)
        )

    async def delete_session(self, name: str) -> protocol.DeleteResult:
        return protocol.DeleteResult.model_validate(
            self._json(
                await _wrap_http(self._request("DELETE", f"/v1/sessions/{name}"))
            )
        )

    async def close_session(self, name: str) -> protocol.CloseResult:
        """Stop hosting the session: cancel + unload; the saved file stays."""
        return protocol.CloseResult.model_validate(
            await self._post(f"/v1/sessions/{name}/close", {})
        )

    # ------------------------------------------------------------------
    # commands

    async def command(self, name: str, command: Command) -> CommandResult:
        """Send one command; the reply goes only to this client."""
        envelope = CommandEnvelope(cmd_id=uuid.uuid4().hex, command=command)
        result = CommandResult.model_validate(self._json(await self._request(
            "POST", f"/v1/sessions/{name}/commands",
            json=envelope.model_dump(),
        )))
        if not result.ok:
            raise ClientError(result.error or "command failed")
        return result

    # ------------------------------------------------------------------
    # attach: SSE with reconnect

    async def attach(self, name: str) -> AsyncIterator[Snapshot | Event]:
        """Yield a Snapshot, then live Events from one stream connection.

        Every attach starts from a fresh Snapshot. Transport and API errors
        propagate to the caller, which owns reconnection (and can respawn the
        host): keeping one recovery layer avoids two loops racing to
        reattach."""
        async for item in self._stream_once(name):
            yield item

    async def _stream_once(self, name: str) -> AsyncIterator[Snapshot | Event]:
        headers = {"Accept": "text/event-stream"}

        @asynccontextmanager
        async def _response_cm() -> AsyncIterator[httpx.Response]:
            response = await self._request(
                "GET", f"/v1/sessions/{name}/events",
                headers=headers, stream=True,
            )
            try:
                yield response
            finally:
                await response.aclose()

        async with _response_cm() as response:
            if response.status_code != 200:
                raise ClientError(f"attach failed: HTTP {response.status_code}")
            async for kind, data in _read_sse(response):
                if kind == "snapshot":
                    yield Snapshot.model_validate(data)
                elif kind == "event":
                    try:
                        yield protocol.EVENT_ADAPTER.validate_python(data)
                    except ValidationError as exc:
                        raise ClientError(f"invalid event frame: {exc}") from None
                # Unknown frame kinds are ignored (forward compatibility).

    # ------------------------------------------------------------------
    # plumbing

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._json(await _wrap_http(self._request("GET", path, params=params)))

    async def _post(self, path: str, body: dict[str, Any]) -> Any:
        return self._json(await _wrap_http(self._request("POST", path, json=body)))

    async def _put(self, path: str, body: dict[str, Any]) -> Any:
        return self._json(await _wrap_http(self._request("PUT", path, json=body)))

    @staticmethod
    def _json(response: httpx.Response) -> Any:
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:  # noqa: BLE001
                detail = response.text
            raise ClientError(f"HTTP {response.status_code}: {detail}")
        return response.json()


async def _read_sse(response: httpx.Response) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Minimal SSE parser: yields (event_kind, json_data) pairs."""
    event_kind = "event"
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if data_lines:
                yield event_kind, json.loads("\n".join(data_lines))
                event_kind = "event"
                data_lines = []
            continue
        if line.startswith("event:"):
            event_kind = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
