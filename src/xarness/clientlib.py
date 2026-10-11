"""Typed async Python client for the xarness host API.

One method per endpoint, httpx over the host's unix socket, an SSE reader
with reconnect support, and the protocol models as the only wire shapes.
The TUI is a pure client of this package; it is also the reference for the
later web client.

Typical use::

    client = await XarnessClient.connect_unix()      # auto-starts the host
    snapshot = await client.snapshot("my-session")
    async for item in client.attach("my-session"):   # Snapshot, then Events
        ...
"""

from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

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


# Reconnect backoff for the SSE reader (seconds).
RECONNECT_MIN = 0.25
RECONNECT_MAX = 4.0


class ClientError(Exception):
    """The API returned an error (bad request, unknown session, …)."""


class XarnessClient:
    """Typed client for one host."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def _with_host_retry(self, send):
        """Run one request; on a dead/unreachable socket, respawn the host
        and retry once.

        The host exits when nothing is active (grace) — a client sitting on
        a draft screen can outlive it, and the next request must transparently
        bring the host back instead of surfacing a raw ConnectError."""
        try:
            result = send()
            # send() may return a coroutine (awaitable request) or an async
            # context manager (httpx.stream) — handle both.
            if inspect.isawaitable(result):
                return await result
            return result
        except httpx.ConnectError:
            await ensure_host()
            result = send()
            if inspect.isawaitable(result):
                return await result
            return result

    @classmethod
    async def connect_unix(
        cls, socket_path: Path | None = None, *, spawn: bool = True,
    ) -> "XarnessClient":
        """Connect over the host's unix socket, auto-starting the host.

        The protocol version is checked against /health; on a mismatch with
        no active sessions the stale host is shut down and respawned."""
        from .host import SOCKET_PATH, connect_or_spawn
        if spawn:
            connect_or_spawn()
        path = socket_path or SOCKET_PATH
        http = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=str(path)),
            base_url="http://localhost",
            timeout=httpx.Timeout(10.0, read=None),
        )
        client = cls(http)
        info = await client.health()
        if info.protocol_version != protocol.PROTOCOL_VERSION:
            await http.aclose()
            raise ClientError(
                f"protocol version mismatch: host speaks v{info.protocol_version}, "
                f"this client speaks v{protocol.PROTOCOL_VERSION}"
            )
        return client

    async def aclose(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------
    # meta

    async def health(self) -> ServerInfo:
        return ServerInfo.model_validate(await self._get("/v1/health"))

    async def config(self) -> protocol.ConfigInfo:
        return protocol.ConfigInfo.model_validate(await self._get("/v1/config"))

    async def search_files(self, workspace: str, q: str) -> list[str]:
        return list(await self._get(
            "/v1/workspaces/files", params={"workspace": workspace, "q": q},
        ))

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
        self,
        *,
        workspace: str | None = None,
        model: str | None = None,
        sandbox: str | None = None,
        preset: str | None = None,
        network: bool | None = None,
        gpu: bool | None = None,
        no_init_repo: bool = False,
        first_message: str | None = None,
    ) -> SessionSummary:
        body: dict[str, Any] = {"no_init_repo": no_init_repo}
        for key, value in [
            ("workspace", workspace), ("model", model), ("sandbox", sandbox),
            ("preset", preset), ("network", network), ("gpu", gpu),
            ("first_message", first_message),
        ]:
            if value is not None:
                body[key] = value
        return SessionSummary.model_validate(await self._post("/v1/sessions", body))

    async def load_session(self, name: str) -> SessionSummary:
        return SessionSummary.model_validate(
            await self._post(f"/v1/sessions/{name}/load", {})
        )

    async def snapshot(self, name: str) -> Snapshot:
        return Snapshot.model_validate(await self._get(f"/v1/sessions/{name}"))

    async def diff(self, name: str, path: str | None = None) -> protocol.DiffInfo:
        params = {"path": path} if path else None
        return protocol.DiffInfo.model_validate(
            await self._get(f"/v1/sessions/{name}/diff", params=params)
        )

    async def delete_session(self, name: str) -> CommandResult:
        response = await self._http.delete(f"/v1/sessions/{name}")
        return CommandResult.model_validate(self._json(response))

    async def close_session(self, name: str) -> CommandResult:
        """Stop hosting the session: cancel + unload; the saved file stays."""
        return CommandResult.model_validate(
            await self._post(f"/v1/sessions/{name}/close", {})
        )

    # ------------------------------------------------------------------
    # commands

    async def command(self, name: str, command: Command) -> CommandResult:
        """Send one command; the reply goes only to this client."""
        envelope = CommandEnvelope(cmd_id=uuid.uuid4().hex, command=command)
        response = await self._http.post(
            f"/v1/sessions/{name}/commands", json=envelope.model_dump(),
        )
        result = CommandResult.model_validate(self._json(response))
        if not result.ok:
            raise ClientError(result.error or "command failed")
        return result

    # ------------------------------------------------------------------
    # attach: SSE with reconnect

    async def attach(
        self, name: str, last_seq: int | None = None,
    ) -> AsyncIterator[Snapshot | Event]:
        """Yield a Snapshot, then live Events, reconnecting automatically.

        Pass ``last_seq`` (e.g. the seq of an already-rendered snapshot) to
        resume instead of receiving a fresh full snapshot: the host replays
        only the missed events, or sends a new Snapshot on a gap.

        On a dropped connection the reader retries with backoff, sending
        Last-Event-ID; if the host replayed, the stream continues seamlessly,
        otherwise a fresh Snapshot arrives. Transport errors retry forever;
        API errors (unknown session) raise :class:`ClientError`."""
        backoff = RECONNECT_MIN
        while True:
            try:
                async for item in self._stream_once(name, last_seq):
                    if isinstance(item, Snapshot):
                        last_seq = item.seq
                    elif isinstance(item, protocol.EventBase):
                        last_seq = max(last_seq or 0, item.seq)
                    backoff = RECONNECT_MIN
                    yield item
                # Server closed the stream cleanly (e.g. session deleted).
                return
            except httpx.HTTPError:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_MAX)

    async def _stream_once(
        self, name: str, last_seq: int | None,
    ) -> AsyncIterator[Snapshot | Event]:
        headers = {"Accept": "text/event-stream"}
        if last_seq is not None:
            headers["Last-Event-ID"] = str(last_seq)
        async with await self._with_host_retry(
            lambda: self._http.stream(
                "GET", f"/v1/sessions/{name}/events", headers=headers,
            )
        ) as response:
            if response.status_code != 200:
                raise ClientError(f"attach failed: HTTP {response.status_code}")
            async for kind, data in _read_sse(response):
                if kind == "snapshot":
                    yield Snapshot.model_validate(data)
                elif kind == "event":
                    yield _parse_event(data)
                # Unknown frame kinds are ignored (forward compatibility).

    # ------------------------------------------------------------------
    # plumbing

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        response = await self._with_host_retry(
            lambda: self._http.get(path, params=params)
        )
        return self._json(response)

    async def _post(self, path: str, body: dict[str, Any]) -> Any:
        response = await self._with_host_retry(
            lambda: self._http.post(path, json=body)
        )
        return self._json(response)

    @staticmethod
    def _json(response: httpx.Response) -> Any:
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:  # noqa: BLE001
                detail = response.text
            raise ClientError(f"HTTP {response.status_code}: {detail}")
        return response.json()


def _parse_event(data: dict[str, Any]) -> Event:
    event_type = data.get("type")
    model = protocol.EVENT_TYPES.get(event_type)
    if model is None:
        raise ClientError(f"unknown event type: {event_type!r}")
    return model.model_validate(data)


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
