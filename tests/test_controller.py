"""Tests for the controller: event enrichment and conversation state."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

from agentcli.config import CotStrength, ProviderProfile
from agentcli.controller import ChatController
from agentcli.events import (
    ContentDelta,
    ProcessingStarted,
    ReasoningDelta,
    StreamError,
    TurnComplete,
    Usage,
)

PROFILE = ProviderProfile(
    base_url="https://api.example.test/v1",
    model_id="test-model",
    max_context=1000,
    cot_strength=CotStrength.MEDIUM,
)


class FakeClient:
    """Replays a scripted list of events instead of talking HTTP."""

    def __init__(self, script: list[Any], delay: float = 0.0) -> None:
        self.script = script
        self.delay = delay
        self.received_wire: list[list[dict[str, Any]]] = []

    async def stream(self, wire_messages: Sequence[dict[str, Any]]) -> AsyncIterator[Any]:
        self.received_wire.append(list(wire_messages))
        yield ProcessingStarted()
        for event in self.script:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield event


async def collect(controller: ChatController, text: str) -> list[Any]:
    return [event async for event in controller.send(text)]


def test_turn_builds_conversation_and_wire_messages() -> None:
    client = FakeClient(
        [
            ReasoningDelta("let me think"),
            ContentDelta("Hello"),
            ContentDelta(" world"),
            TurnComplete(usage=Usage(input_tokens=12, output_tokens=3)),
        ]
    )
    controller = ChatController(PROFILE, "key", client=client)

    events = asyncio.run(collect(controller, "Hi there"))

    assert events[0] == ProcessingStarted()
    assert isinstance(events[-1], TurnComplete)
    assert events[-1].usage == Usage(input_tokens=12, output_tokens=3)
    assert events[-1].reasoning_seconds is not None  # reasoning was present

    # Conversation holds user + assistant, with reasoning stored locally.
    assert [m.role for m in controller.conversation.messages] == ["user", "assistant"]
    assistant = controller.conversation.messages[1]
    assert assistant.content == "Hello world"
    assert assistant.reasoning == "let me think"

    # Wire messages sent to the client: user message only (first request).
    assert client.received_wire[0] == [{"role": "user", "content": "Hi there"}]
    # Reasoning must never be sent back over the wire.
    assert all("reasoning" not in msg for msg in client.received_wire[0])


def test_usage_fallback_is_approximate_when_absent() -> None:
    client = FakeClient([ContentDelta("12345678"), TurnComplete(usage=None)])
    controller = ChatController(PROFILE, "key", client=client)

    events = asyncio.run(collect(controller, "abcd"))

    usage = events[-1].usage
    assert usage.approximate is True
    assert usage.output_tokens == 2  # 8 chars / 4
    assert usage.input_tokens > 0


def test_stream_error_leaves_history_consistent() -> None:
    client = FakeClient([StreamError("HTTP 500: boom")])
    controller = ChatController(PROFILE, "key", client=client)

    events = asyncio.run(collect(controller, "Hi"))

    assert isinstance(events[-1], StreamError)
    # The user message stays (visible in UI); no assistant message was added.
    assert [m.role for m in controller.conversation.messages] == ["user"]


def test_multi_turn_history_accumulates() -> None:
    client = FakeClient([ContentDelta("one"), TurnComplete(usage=Usage(5, 3))])
    controller = ChatController(PROFILE, "key", client=client)
    asyncio.run(collect(controller, "first"))

    client.script = [ContentDelta("two"), TurnComplete(usage=Usage(9, 4))]
    asyncio.run(collect(controller, "second"))

    # Second request carried the full running conversation.
    assert client.received_wire[-1] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": "second"},
    ]
    assert len(controller.conversation.messages) == 4
