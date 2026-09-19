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


def test_compact_drops_trailing_tool_call_round() -> None:
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

    # The compact call itself is summarized away, and its result is never
    # recorded: the history ends on the summary, with no compaction
    # mechanics left on the wire.
    controller.record_tool_result("call_1", ToolResult(ok=True, output="ok"))
    assert [m.role for m in controller.conversation.messages] == ["user", "user"]
    assert "summary text" in controller.conversation.messages[1].content
    # The suppression is one-shot: the next real tool result is recorded.
    controller.record_tool_result("call_2", ToolResult(ok=True, output="fine"))
    assert [m.role for m in controller.conversation.messages] == [
        "user", "user", "tool",
    ]


def test_compact_keeps_round_with_multiple_tool_calls() -> None:
    client = FakeClient([ContentDelta("one"), TurnComplete(usage=Usage(5, 3))])
    controller = ChatController(PROFILE, "key", client=client)
    asyncio.run(collect(controller, "first question"))

    client.script = [
        ToolCallStarted("call_1", "fake_tool"),
        ToolCallArgumentsDone("call_1", "fake_tool", "{}"),
        ToolCallStarted("call_2", "fake_tool"),
        ToolCallArgumentsDone("call_2", "fake_tool", "{}"),
        TurnComplete(has_tool_calls=True),
    ]
    asyncio.run(collect(controller, "use the tools"))

    client.script = [ContentDelta("summary text"), TurnComplete()]
    asyncio.run(controller.compact())

    # With other pending calls in the same round, the assistant message is
    # kept so their results stay paired on the wire (old behavior).
    controller.record_tool_result("call_1", ToolResult(ok=True, output="ok"))
    controller.record_tool_result("call_2", ToolResult(ok=True, output="ok"))
    assert [m.role for m in controller.conversation.messages] == [
        "user", "user", "assistant", "tool", "tool",
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


def test_compact_with_no_history_is_a_no_op() -> None:
    """Compacting a fresh conversation changes nothing — in particular it
    must not touch the compact snapshot or suppress a tool result."""
    client = FakeClient([ContentDelta("summary text"), TurnComplete()])
    controller = ChatController(PROFILE, "key", client=client)
    controller.conversation.add(Message(role="system", content="system prompt"))

    assert asyncio.run(controller.compact()) == "nothing to compact yet"
    assert controller.conversation.compact_snapshot is None
    assert controller.conversation.messages == [
        Message(role="system", content="system prompt")
    ]
    assert controller.last_compaction is None
    assert controller._drop_next_tool_result is False


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


def test_undo_across_compaction_after_later_turn() -> None:
    """Compaction in turn 2, another turn after it: /undo first removes the
    later turn, then a second /undo restores the pre-compaction history."""
    client = FakeClient([ContentDelta("one"), TurnComplete(usage=Usage(5, 3))])
    controller = ChatController(PROFILE, "key", client=client)
    asyncio.run(collect(controller, "first"))

    # Turn 2: model calls compact, then answers.
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

    # Turn 3 happens after the compaction.
    client.script = [ContentDelta("third answer"), TurnComplete(usage=Usage(7, 2))]
    asyncio.run(collect(controller, "third"))
    assert [m.role for m in controller.conversation.messages] == [
        "user", "user", "assistant", "user", "assistant",
    ]

    # First /undo: removes turn 3, back to the end of turn 2 (post-compaction).
    plan = controller.rollback_plan()
    assert plan is not None and plan.user_text == "third"
    assert not plan.clears_compaction
    controller.apply_rollback(plan)
    assert [m.role for m in controller.conversation.messages] == [
        "user", "user", "assistant",
    ]
    assert controller.conversation.compact_snapshot is not None

    # Second /undo: undoes the compaction itself — the full pre-compaction
    # history comes back, turn 2 still intact.
    plan = controller.rollback_plan()
    assert plan is not None and plan.compaction_only
    assert plan.clears_compaction
    controller.apply_rollback(plan)
    assert [m.role for m in controller.conversation.messages] == [
        "user", "assistant", "user", "assistant",
    ]
    assert controller.conversation.messages[2].content == "second"
    assert controller.conversation.compact_snapshot is None
    # Usage back to the end of turn 2's compact round: the summarizer's and
    # the final round's spend subtracted; turn 2's own spend (including the
    # compact call, restored verbatim) stays until the turn is undone.
    assert controller.usage_total == Usage(input_tokens=9, output_tokens=4)

    # Third /undo: removes turn 2 normally.
    plan = controller.rollback_plan()
    assert plan is not None and plan.user_text == "second"
    assert not plan.compaction_only
    controller.apply_rollback(plan)
    assert [m.content for m in controller.conversation.messages] == ["first", "one"]
    assert controller.conversation.messages[1].role == "assistant"
    # Turn 2's spend (including the compact call) subtracted.
    assert controller.usage_total == Usage(input_tokens=5, output_tokens=3)

    # A further /undo walks back the pre-compaction turns normally.
    plan = controller.rollback_plan()
    assert plan is not None and plan.user_text == "first"
    controller.apply_rollback(plan)
    assert controller.conversation.messages == []
    assert controller.usage_total == Usage(input_tokens=0, output_tokens=0)


def test_resumed_compacted_session_undoes_across_compaction(tmp_path) -> None:
    """Save a compacted session, resume it, and undo across the compaction."""
    import json

    from xarness import session_store
    from xarness.conversation import SUMMARY_PREFIX

    session_store.SESSIONS_DIR = tmp_path
    client = FakeClient([ContentDelta("one"), TurnComplete(usage=Usage(5, 3))])
    controller = ChatController(PROFILE, "key", client=client)
    asyncio.run(collect(controller, "first"))

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

    session_store.save_session("s", "test-model", controller.conversation)
    resumed_conversation = session_store.load_session("s")
    assert resumed_conversation.compact_snapshot is not None
    assert any(
        m.content.startswith(SUMMARY_PREFIX) for m in resumed_conversation.messages
    )

    controller2 = ChatController(PROFILE, "key", client=client)
    controller2.conversation = resumed_conversation
    # First /undo: undoes the compaction itself — full pre-compaction
    # history restored, turn 2 intact.
    plan = controller2.rollback_plan()
    assert plan is not None and plan.compaction_only
    assert plan.clears_compaction
    controller2.apply_rollback(plan)
    assert [m.role for m in controller2.conversation.messages] == [
        "user", "assistant", "user", "assistant",
    ]
    assert controller2.conversation.messages[2].content == "second"
    assert controller2.conversation.compact_snapshot is None
    # Second /undo removes turn 2 normally.
    plan = controller2.rollback_plan()
    assert plan is not None and plan.user_text == "second"
    controller2.apply_rollback(plan)
    assert [m.role for m in controller2.conversation.messages] == ["user", "assistant"]
    assert controller2.conversation.messages[1].content == "one"


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
    # The compact call and its result were dropped: the history ends on the
    # summary, followed only by the final round's reply.
    assert [m.role for m in controller.conversation.messages] == [
        "user", "user", "assistant",
    ]

    # First /undo: undoes the compaction itself — the full pre-compaction
    # history (including the compact-call round) is restored, turn 2 intact.
    plan = controller.rollback_plan()
    assert plan is not None and plan.compaction_only
    assert plan.user_text == ""
    assert plan.checkpoint_sha is None
    controller.apply_rollback(plan)
    assert [m.role for m in controller.conversation.messages] == [
        "user", "assistant", "user", "assistant",
    ]
    assert controller.conversation.messages[2].content == "second"
    assert controller.conversation.compact_snapshot is None
    # The summarizer's and the final round's spend subtracted; turn 2's own
    # spend (the compact call, restored verbatim) stays until the turn is
    # undone.
    assert controller.usage_total == Usage(input_tokens=9, output_tokens=4)
    assert controller.conversation.undo_snapshot is None

    # Second /undo: removes turn 2 normally.
    plan2 = controller.rollback_plan()
    assert plan2 is not None and plan2.user_text == "second"
    assert not plan2.compaction_only
    controller.apply_rollback(plan2)
    assert [m.role for m in controller.conversation.messages] == ["user", "assistant"]
    assert controller.conversation.messages[1].content == "one"
    # Usage back to exactly the pre-turn total: turn 2's spend (including
    # the compact call) now subtracted.
    assert controller.usage_total == Usage(input_tokens=5, output_tokens=3)

    # A further /undo falls back to the previous turn (resumed-session
    # semantics): it rolls back to an empty conversation.
    plan3 = controller.rollback_plan()
    assert plan3 is not None and plan3.user_text == "first"
    controller.apply_rollback(plan3)
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
