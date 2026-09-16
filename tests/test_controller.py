"""Tests for the controller: event enrichment and conversation state."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

from xarness.config import CotStrength, ProviderProfile
from xarness.controller import ChatController
from xarness.conversation import Message
from xarness.events import (
    ContentDelta,
    ProcessingStarted,
    ReasoningDelta,
    StreamError,
    ToolCallArgumentsDelta,
    ToolCallArgumentsDone,
    ToolCallStarted,
    TurnComplete,
    Usage,
)
from xarness.tools import ToolResult, build_registry

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
        self.received_tools: list[dict[str, Any]] | None = None

    async def stream(
        self,
        wire_messages: Sequence[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[Any]:
        self.received_wire.append(list(wire_messages))
        self.received_tools = tools
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


def test_tool_call_round_records_wire_shaped_tool_calls() -> None:
    client = FakeClient(
        [
            ToolCallStarted("call_1", "fake_tool"),
            ToolCallArgumentsDelta("call_1", '{"a"'),
            ToolCallArgumentsDelta("call_1", ": 1}"),
            ToolCallArgumentsDone("call_1", "fake_tool", '{"a": 1}'),
            TurnComplete(has_tool_calls=True),
        ]
    )
    controller = ChatController(PROFILE, "key", client=client)

    events = asyncio.run(collect(controller, "use the tool"))

    assert events[-1].has_tool_calls is True
    assistant = controller.conversation.messages[1]
    assert assistant.role == "assistant"
    assert assistant.tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "fake_tool", "arguments": '{"a": 1}'},
        }
    ]


def test_continue_after_tools_sends_tool_result_and_omits_empty_content() -> None:
    client = FakeClient(
        [
            ToolCallStarted("call_1", "fake_tool"),
            ToolCallArgumentsDone("call_1", "fake_tool", "{}"),
            TurnComplete(has_tool_calls=True),
        ]
    )
    controller = ChatController(PROFILE, "key", client=client)
    asyncio.run(collect(controller, "use the tool"))

    controller.record_tool_result("call_1", ToolResult(ok=True, output="success!"))
    client.script = [ContentDelta("all done"), TurnComplete(usage=Usage(4, 2))]
    events = asyncio.run(_drain(controller.continue_after_tools()))

    assert events[-1].has_tool_calls is False
    # The follow-up request replays: user, assistant tool-call turn (content
    # omitted — tool-call-only assistant message), tool result, and the model
    # answers in plain content.
    assert client.received_wire[-1] == [
        {"role": "user", "content": "use the tool"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "fake_tool", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "success!"},
    ]
    assert [m.role for m in controller.conversation.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


def test_tools_schema_is_passed_to_client() -> None:
    client = FakeClient([ContentDelta("hi"), TurnComplete()])
    controller = ChatController(
        PROFILE, "key", client=client, tool_registry=build_registry(None, None)
    )

    asyncio.run(collect(controller, "hello"))

    assert client.received_tools == build_registry(None, None).schema()


def test_no_registry_sends_no_tools() -> None:
    client = FakeClient([ContentDelta("hi"), TurnComplete()])
    controller = ChatController(PROFILE, "key", client=client)

    asyncio.run(collect(controller, "hello"))

    assert client.received_tools is None


def test_compact_replaces_history_with_summary() -> None:
    client = FakeClient([ContentDelta("one"), TurnComplete(usage=Usage(5, 3))])
    controller = ChatController(PROFILE, "key", client=client)
    asyncio.run(collect(controller, "first question"))
    client.script = [ContentDelta("two"), TurnComplete(usage=Usage(9, 4))]
    asyncio.run(collect(controller, "second question"))

    client.script = [ContentDelta("summary text"), TurnComplete()]
    summary = asyncio.run(controller.compact())

    assert summary == "summary text"
    # Everything except the first message was folded into a summary user message.
    assert [m.role for m in controller.conversation.messages] == ["user", "user"]
    assert "summary text" in controller.conversation.messages[1].content
    # The summarizer round saw the rendered transcript (the first message is
    # kept verbatim, so only the rest is summarized), with no tools attached.
    summarizer_prompt = client.received_wire[-1][0]["content"]
    assert "second question" in summarizer_prompt
    assert "assistant: one" in summarizer_prompt
    # The summarizer round ran without tools.
    assert client.received_tools is None


def test_compact_preserves_trailing_tool_call_round() -> None:
    client = FakeClient([ContentDelta("one"), TurnComplete(usage=Usage(5, 3))])
    controller = ChatController(PROFILE, "key", client=client)
    asyncio.run(collect(controller, "first question"))

    client.script = [
        ToolCallStarted("call_1", "fake_tool"),
        ToolCallArgumentsDone("call_1", "fake_tool", "{}"),
        TurnComplete(has_tool_calls=True),
    ]
    asyncio.run(collect(controller, "use the tool"))

    client.script = [ContentDelta("summary text"), TurnComplete()]
    asyncio.run(controller.compact())

    # The tool result recorded after compaction must pair with the preserved
    # assistant tool-call round on the wire.
    controller.record_tool_result("call_1", ToolResult(ok=True, output="ok"))
    assert [m.role for m in controller.conversation.messages] == [
        "user", "user", "assistant", "tool",
    ]
    wire = controller.conversation.to_wire()
    assert wire[2]["tool_calls"][0]["id"] == "call_1"
    assert wire[3]["tool_call_id"] == "call_1"


def test_compact_without_history_is_a_noop() -> None:
    client = FakeClient([])
    controller = ChatController(PROFILE, "key", client=client)

    summary = asyncio.run(controller.compact())

    assert summary == "nothing to compact yet"
    assert controller.conversation.messages == []


# ---------------------------------------------------------------------------
# usage accounting (per-message + cumulative)


def test_usage_attaches_to_messages_and_totals() -> None:
    client = FakeClient([ContentDelta("one"), TurnComplete(usage=Usage(5, 3))])
    controller = ChatController(PROFILE, "key", client=client)
    asyncio.run(collect(controller, "first"))

    client.script = [ContentDelta("two"), TurnComplete(usage=Usage(9, 4))]
    asyncio.run(collect(controller, "second"))

    messages = controller.conversation.messages
    assert messages[1].usage == Usage(5, 3)
    assert messages[3].usage == Usage(9, 4)
    # Usage is local-only: never sent over the wire.
    assert all("usage" not in m for m in client.received_wire[-1])
    assert controller.usage_total == Usage(input_tokens=14, output_tokens=7)


def test_send_snapshots_conversation_for_undo() -> None:
    client = FakeClient([ContentDelta("hi"), TurnComplete()])
    controller = ChatController(PROFILE, "key", client=client)
    asyncio.run(collect(controller, "hello"))

    # Snapshot taken when the turn began: just the user message.
    assert [m.role for m in controller.conversation.undo_snapshot] == ["user"]
    assert controller.conversation.undo_snapshot[0].content == "hello"
    assert controller.conversation.messages[0].checkpoint_sha is None  # no git


def test_compact_reports_token_counts_and_folds_summarizer_usage() -> None:
    client = FakeClient([ContentDelta("one"), TurnComplete(usage=Usage(5, 40))])
    controller = ChatController(PROFILE, "key", client=client)
    asyncio.run(collect(controller, "first question"))

    client.script = [ContentDelta("summary text"), TurnComplete(usage=Usage(100, 7))]
    summary = asyncio.run(controller.compact())

    assert summary == "summary text"
    before, after = controller.last_compaction
    assert before == 40 and after == 7
    # The summarization round's own spend is part of the session total.
    assert controller.usage_total == Usage(input_tokens=105, output_tokens=47)
    # The summary message carries the summarizer's usage (for /undo).
    assert controller.conversation.messages[1].usage == Usage(100, 7)


def test_compact_then_undo_restores_messages_and_usage() -> None:
    """Undoing a turn that compacted mid-flight restores the pre-compact
    message list AND un-does the summarizer's usage adjustment."""
    client = FakeClient([ContentDelta("one"), TurnComplete(usage=Usage(5, 3))])
    controller = ChatController(PROFILE, "key", client=client)
    asyncio.run(collect(controller, "first"))

    # Turn 2, round 1: the model calls compact.
    client.script = [
        ToolCallStarted("call_1", "compact"),
        ToolCallArgumentsDone("call_1", "compact", "{}"),
        TurnComplete(usage=Usage(4, 1), has_tool_calls=True),
    ]
    asyncio.run(collect(controller, "second"))
    client.script = [ContentDelta("summary text"), TurnComplete(usage=Usage(100, 7))]
    asyncio.run(controller.compact())
    controller.record_tool_result("call_1", ToolResult(ok=True, output="compacted"))
    client.script = [ContentDelta("final"), TurnComplete(usage=Usage(6, 2))]
    asyncio.run(_drain(controller.continue_after_tools()))

    assert controller.usage_total == Usage(input_tokens=115, output_tokens=13)
    assert [m.role for m in controller.conversation.messages] == [
        "user", "user", "assistant", "tool", "assistant",
    ]

    plan = controller.rollback_plan()
    assert plan is not None and plan.user_text == "second"
    controller.apply_rollback(plan)

    # Pre-compact history restored (the turn's own user message removed).
    assert [m.role for m in controller.conversation.messages] == ["user", "assistant"]
    assert controller.conversation.messages[1].content == "one"
    # Usage back to exactly the pre-turn total: the compacted rounds', the
    # summarizer's, and the final round's spend all subtracted.
    assert controller.usage_total == Usage(input_tokens=5, output_tokens=3)
    assert controller.conversation.undo_snapshot is None

    # A further /undo falls back to the previous turn (resumed-session
    # semantics): it rolls back to an empty conversation.
    plan2 = controller.rollback_plan()
    assert plan2 is not None and plan2.user_text == "first"
    controller.apply_rollback(plan2)
    assert controller.conversation.messages == []
    assert controller.usage_total == Usage(input_tokens=0, output_tokens=0)
    assert controller.rollback_plan() is None


def test_rollback_plan_drops_the_turn_for_undo_and_retry() -> None:
    """Both /undo and /retry truncate to before the turn's user message;
    /retry re-sends the text (with a fresh checkpoint) via send()."""
    client = FakeClient([ContentDelta("one"), TurnComplete(usage=Usage(5, 3))])
    controller = ChatController(PROFILE, "key", client=client)
    asyncio.run(collect(controller, "first"))
    client.script = [ContentDelta("two"), TurnComplete(usage=Usage(9, 4))]
    asyncio.run(collect(controller, "second"))

    plan = controller.rollback_plan()
    assert plan is not None
    assert plan.user_text == "second"
    assert [m.content for m in plan.keep] == ["first", "one"]
    assert [m.content for m in plan.dropped] == ["second", "two"]

    controller.apply_rollback(plan)
    assert [m.content for m in controller.conversation.messages] == [
        "first", "one",
    ]
    assert controller.usage_total == Usage(input_tokens=5, output_tokens=3)


def test_rollback_without_snapshot_truncates_at_last_user_message() -> None:
    """Resumed sessions have no snapshot: fall back to the last user message
    (its persisted checkpoint sha is used for the worktree revert)."""
    controller = ChatController(PROFILE, "key", client=FakeClient([]))
    controller.conversation.add(Message(role="user", content="old"))
    controller.conversation.add(Message(role="assistant", content="a", usage=Usage(5, 3)))
    controller.conversation.add(
        Message(role="user", content="last", checkpoint_sha="abc123")
    )
    controller.conversation.add(Message(role="assistant", content="b", usage=Usage(9, 4)))
    controller.usage_total = Usage(input_tokens=14, output_tokens=7)

    plan = controller.rollback_plan()
    assert plan is not None
    assert plan.user_text == "last"
    assert plan.checkpoint_sha == "abc123"
    controller.apply_rollback(plan)
    assert [m.content for m in controller.conversation.messages] == ["old", "a"]
    assert controller.usage_total == Usage(input_tokens=5, output_tokens=3)


def test_rollback_is_a_noop_without_a_finished_turn() -> None:
    controller = ChatController(PROFILE, "key", client=FakeClient([]))
    assert controller.rollback_plan() is None  # empty conversation
    asyncio.run(collect(controller, "only a user message"))
    assert controller.rollback_plan() is None  # no response yet


async def _drain(stream: AsyncIterator[Any]) -> list[Any]:
    return [event async for event in stream]
