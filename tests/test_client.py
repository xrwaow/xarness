"""Tests for the ChatClient stream: transient connection failures are
retried before any tokens arrive; server responses and mid-stream drops are
not. Uses a custom httpx transport — no network."""

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest

from xarness.client import ChatClient
from xarness.config import CotStrength, ProviderProfile
from xarness.events import ContentDelta, ProcessingStarted, StreamError, TurnComplete, Usage

PROFILE = ProviderProfile(
    base_url="https://api.example.test/v1",
    model_id="test-model",
    max_context=1000,
    cot_strength=CotStrength.MEDIUM,
)

SSE_BODY = (
    b'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n'
    b'data: {"usage": {"prompt_tokens": 3, "completion_tokens": 2}}\n\n'
    b"data: [DONE]\n\n"
)


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    import xarness.client as client_module

    monkeypatch.setattr(client_module, "_RETRY_BACKOFF", 0.0)


class FlakyTransport(httpx.AsyncBaseTransport):
    """Drops the first ``failures`` connection attempts, then answers with a
    normal SSE stream."""

    def __init__(self, failures: int = 1) -> None:
        self.failures = failures
        self.calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls <= self.failures:
            raise httpx.ConnectError("connection reset by peer", request=request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=SSE_BODY
        )


class MidStreamDropTransport(httpx.AsyncBaseTransport):
    """Streams one delta, then the connection dies mid-response."""

    def __init__(self) -> None:
        self.calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1

        async def body() -> AsyncIterator[bytes]:
            yield b'data: {"choices": [{"delta": {"content": "partial"}}]}\n\n'
            raise httpx.ReadError("connection dropped mid-stream", request=request)

        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body()
        )


class StatusTransport(httpx.AsyncBaseTransport):
    """Always answers with a real HTTP error status."""

    def __init__(self, status: int = 500) -> None:
        self.status = status
        self.calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return httpx.Response(self.status, text="server exploded")


def _client(transport: httpx.AsyncBaseTransport) -> ChatClient:
    return ChatClient(PROFILE, "key", transport=transport)


async def _drain(stream) -> list:
    return [event async for event in stream]


def test_transient_connect_error_is_retried() -> None:
    transport = FlakyTransport(failures=1)
    events = asyncio.run(_drain(_client(transport).stream([{"role": "user", "content": "hi"}])))

    assert transport.calls == 2  # one drop, one retry
    assert events[0] == ProcessingStarted()
    assert isinstance(events[-1], TurnComplete)
    assert events[-1].usage == Usage(input_tokens=3, output_tokens=2)
    assert any(isinstance(e, ContentDelta) and e.text == "hello" for e in events)


def test_persistent_failures_end_in_stream_error() -> None:
    transport = FlakyTransport(failures=99)
    events = asyncio.run(_drain(_client(transport).stream([{"role": "user", "content": "hi"}])))

    assert transport.calls == 3  # the configured attempt count, then give up
    assert isinstance(events[-1], StreamError)
    assert "connection reset" in events[-1].message


def test_http_error_response_is_not_retried() -> None:
    transport = StatusTransport(500)
    events = asyncio.run(_drain(_client(transport).stream([{"role": "user", "content": "hi"}])))

    assert transport.calls == 1  # a real server response is never retried
    assert isinstance(events[-1], StreamError)
    assert "HTTP 500" in events[-1].message


def test_client_error_response_is_not_retried() -> None:
    transport = StatusTransport(401)
    events = asyncio.run(_drain(_client(transport).stream([{"role": "user", "content": "hi"}])))

    assert transport.calls == 1
    assert isinstance(events[-1], StreamError)
    assert "HTTP 401" in events[-1].message


def test_midstream_drop_is_not_retried() -> None:
    transport = MidStreamDropTransport()
    events = asyncio.run(_drain(_client(transport).stream([{"role": "user", "content": "hi"}])))

    assert transport.calls == 1  # a partial answer can never be replayed
    assert isinstance(events[-1], StreamError)
    assert any(isinstance(e, ContentDelta) for e in events)  # tokens did arrive
