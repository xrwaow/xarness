"""Tests for the controller: event enrichment and conversation state."""

import asyncio
import base64
from collections.abc import AsyncIterator, Sequence
from typing import Any

from xarness.config import CotStrength, ProviderProfile
from xarness.controller import ChatController, PartialRound
from xarness.conversation import Message, ToolCall, build_parts
from xarness.gitwork import TurnCheckpoint
from xarness.images import ImageAttachment
from xarness.protocol import (
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
            ReasoningDelta(text="let me think"),
            ContentDelta(text="Hello"),
            ContentDelta(text=" world"),
            TurnComplete(usage=Usage(input_tokens=12, output_tokens=3)),
        ]
    )
    controller = ChatController(PROFILE, "key", client=client)

    events = asyncio.run(collect(controller, "Hi there"))

    assert isinstance(events[0], ProcessingStarted)
    assert isinstance(events[-1], TurnComplete)
    assert events[-1].usage == Usage(input_tokens=12, output_tokens=3)
    # Conversation holds user + assistant, with reasoning stored locally.
    assert [m.role for m in controller.conversation.messages] == ["user", "assistant"]
    assistant = controller.conversation.messages[1]
    assert assistant.content == "Hello world"
    assert assistant.reasoning == "let me think"
    assert assistant.timing.reasoning_seconds is not None

    # Wire messages sent to the client: user message only (first request).
    assert client.received_wire[0] == [{"role": "user", "content": "Hi there"}]
    # Reasoning must never be sent back over the wire.
    assert all("reasoning" not in msg for msg in client.received_wire[0])


def test_usage_fallback_is_approximate_when_absent() -> None:
    client = FakeClient([ContentDelta(text="12345678"), TurnComplete(usage=None)])
    controller = ChatController(PROFILE, "key", client=client)

    events = asyncio.run(collect(controller, "abcd"))

    usage = events[-1].usage
    assert usage.approximate is True
    assert usage.output_tokens == 2  # 8 chars / 4
    assert usage.input_tokens > 0


def test_stream_error_leaves_history_consistent() -> None:
    client = FakeClient([StreamError(message="HTTP 500: boom")])
    controller = ChatController(PROFILE, "key", client=client)

    events = asyncio.run(collect(controller, "Hi"))

    assert isinstance(events[-1], StreamError)
    # The user message stays (visible in UI); no assistant message was added.
    assert [m.role for m in controller.conversation.messages] == ["user"]


def test_multi_turn_history_accumulates() -> None:
    client = FakeClient([ContentDelta(text="one"), TurnComplete(usage=Usage(input_tokens=5, output_tokens=3))])
    controller = ChatController(PROFILE, "key", client=client)
    asyncio.run(collect(controller, "first"))

    client.script = [ContentDelta(text="two"), TurnComplete(usage=Usage(input_tokens=9, output_tokens=4))]
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
            ToolCallStarted(call_id="call_1", name="fake_tool"),
            ToolCallArgumentsDelta(call_id="call_1", text='{"a"'),
            ToolCallArgumentsDelta(call_id="call_1", text=": 1}"),
            ToolCallArgumentsDone(call_id="call_1", name="fake_tool", arguments_json='{"a": 1}'),
            TurnComplete(has_tool_calls=True),
        ]
    )
    controller = ChatController(PROFILE, "key", client=client)

    events = asyncio.run(collect(controller, "use the tool"))

    assert events[-1].has_tool_calls is True
    assistant = controller.conversation.messages[1]
    assert assistant.role == "assistant"
    assert assistant.tool_calls == [ToolCall(
        call_id="call_1", name="fake_tool", arguments_json='{"a": 1}',
    )]


def test_continue_after_tools_sends_tool_result_and_omits_empty_content() -> None:
    client = FakeClient(
        [
            ToolCallStarted(call_id="call_1", name="fake_tool"),
            ToolCallArgumentsDone(call_id="call_1", name="fake_tool", arguments_json="{}"),
            TurnComplete(has_tool_calls=True),
        ]
    )
    controller = ChatController(PROFILE, "key", client=client)
    asyncio.run(collect(controller, "use the tool"))

    controller.record_tool_result("call_1", ToolResult(ok=True, output="success!"))
    client.script = [ContentDelta(text="all done"), TurnComplete(usage=Usage(input_tokens=4, output_tokens=2))]
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
    client = FakeClient([ContentDelta(text="hi"), TurnComplete()])
    controller = ChatController(
        PROFILE, "key", client=client, tool_registry=build_registry(None, None)
    )

    asyncio.run(collect(controller, "hello"))

    assert client.received_tools == build_registry(None, None).schema()


def test_no_registry_sends_no_tools() -> None:
    client = FakeClient([ContentDelta(text="hi"), TurnComplete()])
    controller = ChatController(PROFILE, "key", client=client)

    asyncio.run(collect(controller, "hello"))

    assert client.received_tools is None


def test_compact_replaces_history_with_summary() -> None:
    client = FakeClient([ContentDelta(text="one"), TurnComplete(usage=Usage(input_tokens=5, output_tokens=3))])
    controller = ChatController(PROFILE, "key", client=client)
    asyncio.run(collect(controller, "first question"))
    client.script = [ContentDelta(text="two"), TurnComplete(usage=Usage(input_tokens=9, output_tokens=4))]
    asyncio.run(collect(controller, "second question"))

    client.script = [ContentDelta(text="summary text"), TurnComplete()]
    summary = asyncio.run(controller.compact())

    assert summary == "summary text"
    # The whole visible history folds into one summary message.
    visible = controller.conversation.visible_messages()
    assert [m.role for m in visible] == ["user"]
    assert visible[0].kind == "summary"
    assert "summary text" in visible[0].content
    # The summarizer round saw the rendered transcript, with no tools attached.
    summarizer_prompt = client.received_wire[-1][0]["content"]
    assert "first question" in summarizer_prompt
    assert "second question" in summarizer_prompt
    assert "assistant: one" in summarizer_prompt
    # The summarizer round ran without tools.
    assert client.received_tools is None


def test_compact_without_history_is_a_noop() -> None:
    client = FakeClient([])
    controller = ChatController(PROFILE, "key", client=client)

    summary = asyncio.run(controller.compact())

    assert summary == "nothing to compact yet"
    assert controller.conversation.messages == []


# ---------------------------------------------------------------------------
# usage accounting (per-message + cumulative)


def test_usage_attaches_to_messages_and_totals() -> None:
    client = FakeClient([ContentDelta(text="one"), TurnComplete(usage=Usage(input_tokens=5, output_tokens=3))])
    controller = ChatController(PROFILE, "key", client=client)
    asyncio.run(collect(controller, "first"))

    client.script = [ContentDelta(text="two"), TurnComplete(usage=Usage(input_tokens=9, output_tokens=4))]
    asyncio.run(collect(controller, "second"))

    messages = controller.conversation.messages
    assert messages[1].usage == Usage(input_tokens=5, output_tokens=3)
    assert messages[3].usage == Usage(input_tokens=9, output_tokens=4)
    # Usage is local-only: never sent over the wire.
    assert all("usage" not in m for m in client.received_wire[-1])
    assert controller.usage_total == Usage(input_tokens=14, output_tokens=7)


def test_send_snapshots_conversation_for_undo() -> None:
    client = FakeClient([ContentDelta(text="hi"), TurnComplete()])
    controller = ChatController(PROFILE, "key", client=client)
    asyncio.run(collect(controller, "hello"))

    # A Turn owns the checkpoint taken when the turn began.
    turn = controller.conversation.turns[0]
    assert turn.user_message_id == controller.conversation.messages[0].id
    assert turn.before_tree is None  # no git
    assert turn.after_tree is None
    assert controller.conversation.messages[0].content == "hello"


def test_compact_with_no_history_is_a_no_op() -> None:
    """Compacting a fresh conversation changes nothing."""
    client = FakeClient([ContentDelta(text="summary text"), TurnComplete()])
    controller = ChatController(PROFILE, "key", client=client)

    assert asyncio.run(controller.compact()) == "nothing to compact yet"
    assert controller.conversation.messages == []
    assert controller.conversation.epoch == 0
    assert controller.last_compaction is None


def test_compact_reports_token_counts_and_folds_summarizer_usage() -> None:
    client = FakeClient([ContentDelta(text="one"), TurnComplete(usage=Usage(input_tokens=5, output_tokens=40))])
    controller = ChatController(PROFILE, "key", client=client)
    asyncio.run(collect(controller, "first question"))

    client.script = [ContentDelta(text="summary text"), TurnComplete(usage=Usage(input_tokens=100, output_tokens=7))]
    summary = asyncio.run(controller.compact())

    assert summary == "summary text"
    before, after = controller.last_compaction
    assert before == 43 and after == 7
    # The summarization round's own spend is part of the session total.
    assert controller.usage_total == Usage(input_tokens=105, output_tokens=47)
    # The summary message carries the summarizer's usage (for /undo).
    assert controller.conversation.visible_messages()[0].usage == Usage(input_tokens=100, output_tokens=7)


def test_undo_across_compaction_after_later_turn() -> None:
    """Auto-compaction at a turn boundary, later turns after it: /undo
    first removes the later turn, then a second /undo restores the
    pre-compaction history."""
    client = FakeClient([ContentDelta(text="one"), TurnComplete(usage=Usage(input_tokens=5, output_tokens=3))])
    controller = ChatController(PROFILE, "key", client=client)
    asyncio.run(collect(controller, "first"))

    # Turn 2: a normal answer; compaction then runs at the turn boundary.
    client.script = [ContentDelta(text="two"), TurnComplete(usage=Usage(input_tokens=4, output_tokens=1))]
    asyncio.run(collect(controller, "second"))
    client.script = [ContentDelta(text="summary text"), TurnComplete(usage=Usage(input_tokens=100, output_tokens=7))]
    asyncio.run(controller.compact())

    # Later turns happen after the compaction.
    client.script = [ContentDelta(text="final"), TurnComplete(usage=Usage(input_tokens=6, output_tokens=2))]
    asyncio.run(collect(controller, "third"))
    client.script = [ContentDelta(text="fourth answer"), TurnComplete(usage=Usage(input_tokens=7, output_tokens=2))]
    asyncio.run(collect(controller, "fourth"))
    assert [m.role for m in controller.conversation.visible_messages()] == [
        "user", "user", "assistant", "user", "assistant",
    ]

    # First /undo: removes turn 4, back to the end of turn 3 (post-compaction).
    plan = controller.rollback_plan()
    assert plan is not None and plan.user_text == "fourth"
    assert not plan.compaction_only
    controller.apply_rollback(plan)
    assert [m.role for m in controller.conversation.visible_messages()] == [
        "user", "user", "assistant",
    ]
    assert controller.conversation.epoch == 1

    # Second /undo: removes turn 3 — the first turn after the compaction.
    plan = controller.rollback_plan()
    assert plan is not None and plan.user_text == "third"
    assert not plan.compaction_only
    controller.apply_rollback(plan)
    assert [m.role for m in controller.conversation.visible_messages()] == ["user"]

    # Third /undo: undoes the compaction itself — the full pre-compaction
    # history comes back, turns 1 and 2 still intact.
    plan = controller.rollback_plan()
    assert plan is not None and plan.compaction_only
    assert plan.user_text == ""
    assert plan.revert_turns == []  # compaction only: no files reverted
    controller.apply_rollback(plan)
    assert [m.role for m in controller.conversation.visible_messages()] == [
        "user", "assistant", "user", "assistant",
    ]
    assert controller.conversation.messages[2].content == "second"
    assert controller.conversation.epoch == 0
    # Totals are monotonic: everything ever spent (turns 1-2, the
    # summarizer, turns 3-4) stays counted.
    assert controller.usage_total == Usage(input_tokens=122, output_tokens=15)

    # Fourth /undo: removes turn 2 normally.
    plan = controller.rollback_plan()
    assert plan is not None and plan.user_text == "second"
    assert not plan.compaction_only
    controller.apply_rollback(plan)
    assert [m.content for m in controller.conversation.visible_messages()] == ["first", "one"]
    assert controller.conversation.messages[1].role == "assistant"
    # Totals unchanged: nothing new was spent, and undo never subtracts.
    assert controller.usage_total == Usage(input_tokens=122, output_tokens=15)

    # A further /undo walks back the pre-compaction turns normally.
    plan = controller.rollback_plan()
    assert plan is not None and plan.user_text == "first"
    controller.apply_rollback(plan)
    assert controller.conversation.visible_messages() == []
    # Totals are monotonic: everything ever spent stays counted.
    assert controller.usage_total == Usage(input_tokens=122, output_tokens=15)


def test_resumed_compacted_session_undoes_across_compaction(tmp_path) -> None:
    """Save a compacted session, resume it, and undo across the compaction."""

    from xarness import session_store
    from xarness.conversation import SUMMARY_PREFIX

    session_store.SESSIONS_DIR = tmp_path
    client = FakeClient([ContentDelta(text="one"), TurnComplete(usage=Usage(input_tokens=5, output_tokens=3))])
    controller = ChatController(PROFILE, "key", client=client)
    asyncio.run(collect(controller, "first"))

    # Turn 2: a normal answer; compaction then runs at the turn boundary,
    # and one more turn happens after it.
    client.script = [ContentDelta(text="two"), TurnComplete(usage=Usage(input_tokens=4, output_tokens=1))]
    asyncio.run(collect(controller, "second"))
    client.script = [ContentDelta(text="summary text"), TurnComplete(usage=Usage(input_tokens=100, output_tokens=7))]
    asyncio.run(controller.compact())
    client.script = [ContentDelta(text="final"), TurnComplete(usage=Usage(input_tokens=6, output_tokens=2))]
    asyncio.run(collect(controller, "third"))

    session_store.save_state("s", session_store.SessionState(controller.conversation, "test-model"))
    resumed_conversation = session_store.load_state("s").conversation
    assert resumed_conversation.epoch == 1
    assert any(
        m.content.startswith(SUMMARY_PREFIX)
        for m in resumed_conversation.visible_messages()
    )

    controller2 = ChatController(PROFILE, "key", client=client)
    controller2.conversation = resumed_conversation
    # First /undo: removes turn 3 — the first turn after the compaction.
    plan = controller2.rollback_plan()
    assert plan is not None and plan.user_text == "third"
    controller2.apply_rollback(plan)
    # Second /undo: undoes the compaction itself — full pre-compaction
    # history restored, turns 1 and 2 intact.
    plan = controller2.rollback_plan()
    assert plan is not None and plan.compaction_only
    controller2.apply_rollback(plan)
    assert [m.role for m in controller2.conversation.visible_messages()] == [
        "user", "assistant", "user", "assistant",
    ]
    assert controller2.conversation.messages[2].content == "second"
    assert controller2.conversation.epoch == 0
    # Third /undo removes turn 2 normally.
    plan = controller2.rollback_plan()
    assert plan is not None and plan.user_text == "second"
    controller2.apply_rollback(plan)
    assert [m.role for m in controller2.conversation.visible_messages()] == ["user", "assistant"]
    assert controller2.conversation.messages[1].content == "one"


def test_rollback_plan_drops_the_turn_for_undo_and_retry() -> None:
    """Both /undo and /retry truncate to before the turn's user message;
    /retry re-sends the text (with a fresh checkpoint) via send()."""
    client = FakeClient([ContentDelta(text="one"), TurnComplete(usage=Usage(input_tokens=5, output_tokens=3))])
    controller = ChatController(PROFILE, "key", client=client)
    asyncio.run(collect(controller, "first"))
    client.script = [ContentDelta(text="two"), TurnComplete(usage=Usage(input_tokens=9, output_tokens=4))]
    asyncio.run(collect(controller, "second"))

    plan = controller.rollback_plan()
    assert plan is not None
    assert plan.user_text == "second"
    assert [m.content for m in plan.keep] == ["first", "one"]

    controller.apply_rollback(plan)
    assert [m.content for m in controller.conversation.messages] == [
        "first", "one",
    ]
    # Totals are monotonic: the undone turn's spend stays counted.
    assert controller.usage_total == Usage(input_tokens=14, output_tokens=7)


def test_rollback_without_snapshot_truncates_at_last_user_message() -> None:
    """Resumed sessions have no snapshot: fall back to the last user message.
    Its persisted before/after trees drive the file revert."""
    controller = ChatController(PROFILE, "key", client=FakeClient([]))
    controller.conversation.add(Message(role="user", content="old"))
    controller.conversation.add(Message(role="assistant", content="a", usage=Usage(input_tokens=5, output_tokens=3)))
    last = controller.conversation.add(Message(role="user", content="last"))
    controller.conversation.begin_turn(last, "abc123")
    controller.conversation.turns[-1].after_tree = "def456"
    controller.conversation.add(Message(role="assistant", content="b", usage=Usage(input_tokens=9, output_tokens=4)))
    controller.usage_total = Usage(input_tokens=14, output_tokens=7)

    plan = controller.rollback_plan()
    assert plan is not None
    assert plan.user_text == "last"
    assert plan.revert_turns == [TurnCheckpoint("abc123", "def456")]
    controller.apply_rollback(plan)
    assert [m.content for m in controller.conversation.visible_messages()] == ["old", "a"]
    # Totals are monotonic: the undone turn's spend stays counted.
    assert controller.usage_total == Usage(input_tokens=14, output_tokens=7)


def test_rollback_is_a_noop_without_a_finished_turn() -> None:
    controller = ChatController(PROFILE, "key", client=FakeClient([]))
    assert controller.rollback_plan() is None  # empty conversation
    asyncio.run(collect(controller, "only a user message"))
    assert controller.rollback_plan() is None  # no response yet


def test_rollback_plan_to_an_arbitrary_earlier_turn() -> None:
    """Click-to-undo truncates at any earlier user message, restoring its
    checkpoint and subtracting every dropped turn's usage."""
    client = FakeClient([ContentDelta(text="one"), TurnComplete(usage=Usage(input_tokens=5, output_tokens=3))])
    controller = ChatController(PROFILE, "key", client=client)
    asyncio.run(collect(controller, "first"))
    client.script = [ContentDelta(text="two"), TurnComplete(usage=Usage(input_tokens=9, output_tokens=4))]
    asyncio.run(collect(controller, "second"))
    client.script = [ContentDelta(text="three"), TurnComplete(usage=Usage(input_tokens=7, output_tokens=2))]
    asyncio.run(collect(controller, "third"))

    first_user = controller.conversation.messages[0]
    plan = controller.rollback_plan_to(first_user)
    assert plan is not None
    assert plan.user_text == "first"
    assert plan.keep == []

    controller.apply_rollback(plan)
    assert controller.conversation.messages == []
    # Totals are monotonic: all three turns' spend stays counted.
    assert controller.usage_total == Usage(input_tokens=21, output_tokens=9)


def test_rollback_plan_to_rejects_non_turns_allows_the_latest() -> None:
    controller = ChatController(PROFILE, "key", client=FakeClient([]))
    controller.conversation.add(Message(role="user", content="first"))
    controller.conversation.add(Message(role="assistant", content="one"))
    controller.conversation.add(Message(role="user", content="pending"))
    messages = controller.conversation.messages
    assert controller.rollback_plan_to(messages[1]) is None  # not a user turn
    # The latest user message is still undoable: nothing to drop or revert,
    # but the plan puts its text back in the input.
    plan = controller.rollback_plan_to(messages[2])
    assert plan is not None
    assert plan.user_text == "pending"
    assert plan.keep == messages[:2]
    assert plan.revert_turns == []


def test_save_interrupted_round_keeps_partial_answer() -> None:
    """An interrupted round is saved with the text and reasoning it managed
    to stream before the cancellation."""
    controller = ChatController(PROFILE, "key", client=FakeClient([]))
    controller._partial_round = PartialRound(["thinking "], ["partial answer"])
    assert controller.save_interrupted_round() is True
    message = controller.conversation.messages[-1]
    assert message.role == "assistant"
    assert message.content == "partial answer"
    assert message.reasoning == "thinking "
    # Cleared, so a second call is a no-op (the round is only saved once).
    assert controller.save_interrupted_round() is False


def test_save_interrupted_round_keeps_reasoning_only() -> None:
    """A round interrupted while still thinking (no answer text) saves the
    partial CoT: it belongs in the history and is sent with it (the same
    keep_reasoning rule as any assistant message's reasoning)."""
    controller = ChatController(PROFILE, "key", client=FakeClient([]))
    controller._partial_round = PartialRound(["just thinking"], [])
    assert controller.save_interrupted_round() is True
    (message,) = controller.conversation.messages
    assert message.role == "assistant"
    assert message.reasoning == "just thinking"
    assert message.content == ""
    # The reasoning rides along on the wire (keep_reasoning on by default).
    wire = message.to_wire(include_reasoning=PROFILE.keep_reasoning)
    assert wire["reasoning_content"] == "just thinking"
    # Cleared, so a second call is a no-op (the round is only saved once).
    assert controller.save_interrupted_round() is False


def test_save_interrupted_round_drops_empty_round() -> None:
    """A round with neither text nor reasoning saves nothing."""


def test_save_interrupted_round_drops_unpaired_tool_calls() -> None:
    """Half-streamed tool calls are dropped: they have no result to pair with,
    and an orphaned tool call would be rejected on the next request."""
    controller = ChatController(PROFILE, "key", client=FakeClient([]))
    controller._partial_round = PartialRound([], [])
    assert controller.save_interrupted_round() is False
    assert controller.conversation.messages == []


async def _drain(stream: AsyncIterator[Any]) -> list[Any]:
    return [event async for event in stream]


def _image(name: str = "cat.png") -> ImageAttachment:
    return ImageAttachment(
        name=name, mime="image/png",
        data_b64=base64.b64encode(name.encode()).decode(),
        width=1, height=1,
    )


def test_rollback_plan_keeps_images() -> None:
    controller = ChatController(PROFILE, "key", client=FakeClient([]))
    image = _image()
    controller.conversation.add(Message(
        role="user", content="look", parts=build_parts("look", [image]),
    ))
    controller.conversation.add(Message(role="assistant", content="ok"))
    plan = controller.rollback_plan()
    assert plan is not None and plan.images == [image.to_ref()]


def test_image_token_maps_by_text_not_ordinal() -> None:
    msg = Message(
        role="user", content="[🖼 stale 1×1] mid [🖼 cat.png 1×1] end",
        parts=build_parts(
            "[🖼 stale 1×1] mid [🖼 cat.png 1×1] end", [_image()],
        ),
    )
    parts = msg.to_wire()["content"]
    image_at = next(i for i, p in enumerate(parts) if p["type"] == "image_url")
    before = "".join(p["text"] for p in parts[:image_at] if p["type"] == "text")
    assert "stale" in before and "cat.png" not in before
