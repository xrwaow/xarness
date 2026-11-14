"""Tests for the session's round runner: event enrichment and history."""

import asyncio
import base64
import hashlib
import tempfile
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from xarness import session_store
from xarness.config import CotStrength, ProviderProfile
from xarness.conversation import (
    AssistantMessage, ToolCall, Usage,
    user_message,
)
from xarness.images import ImageAttachment, token_text
from xarness.session import Session
from xarness.session_store import SessionImageStore
from xarness.protocol import (
    ContentDelta,
    PartialRound,
    ProcessingStarted,
    ReasoningDelta,
    StreamError,
    ToolCallArgumentsDelta,
    ToolCallArgumentsDone,
    ToolCallStarted,
    RoundComplete,
)
from xarness.tools import ToolRegistry, ToolResult, build_registry

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


def _save(name: str, stored) -> None:
    payload, meta = session_store.serialize_state(stored)
    session_store.write_state(name, payload, meta)


# A throwaway workspace: these sessions never track git or save to disk.
WORKSPACE = Path(tempfile.mkdtemp())


def make_session(client: FakeClient, tools: "ToolRegistry | None" = None) -> Session:
    """A session wired to a scripted client — the round runner under test."""
    session = Session(
        name="test", workspace=WORKSPACE, profile=PROFILE, profile_name="test",
        api_key="key", client=client, no_save=True,
    )
    if tools is not None:
        session.tool_registry = tools
    # No prompt on the wire: these tests assert on the conversation itself.
    session.system_prompt = None
    return session


async def collect(session: Session, text: str) -> list[Any]:
    """One turn: add the user message, stream its first round."""
    session._begin_turn(text, None, None)
    return [event async for event in session._stream_round()]


def test_turn_builds_conversation_and_wire_messages() -> None:
    client = FakeClient(
        [
            ReasoningDelta(text="let me think"),
            ContentDelta(text="Hello"),
            ContentDelta(text=" world"),
            RoundComplete(usage=Usage(input_tokens=12, output_tokens=3)),
        ]
    )
    session = make_session(client)

    events = asyncio.run(collect(session, "Hi there"))

    assert isinstance(events[0], ProcessingStarted)
    assert isinstance(events[-1], RoundComplete)
    assert events[-1].usage == Usage(input_tokens=12, output_tokens=3)
    # Conversation holds user + assistant, with reasoning stored locally.
    assert [m.role for m in session.conversation.messages] == ["user", "assistant"]
    assistant = session.conversation.messages[1]
    assert assistant.content == "Hello world"
    assert assistant.reasoning == "let me think"
    assert assistant.timing.reasoning_seconds is not None

    # Wire messages sent to the client: user message only (first request).
    assert client.received_wire[0] == [{"role": "user", "content": "Hi there"}]
    # Reasoning must never be sent back over the wire.
    assert all("reasoning" not in msg for msg in client.received_wire[0])


def test_reasoning_seconds_cover_the_cot_block_only() -> None:
    """``reasoning_seconds`` ends at the first answer token: the answer's own
    streaming time is not thinking time (and the live TUI pill derives the
    same number from the same two event stamps)."""
    client = FakeClient(
        [
            ReasoningDelta(text="let me think"),
            ContentDelta(text="Hello"),
            ContentDelta(text=" world"),
            RoundComplete(),
        ],
        delay=0.05,
    )
    session = make_session(client)

    asyncio.run(collect(session, "Hi there"))

    assistant = session.conversation.messages[1]
    # One delay (reasoning → first content token), not three.
    assert 0.03 < assistant.timing.reasoning_seconds < 0.09


def test_reasoning_seconds_exclude_tool_argument_streaming() -> None:
    """A tool call ends the cot block too: streaming its arguments is call
    time, so it must not inflate the thinking number."""
    client = FakeClient(
        [
            ReasoningDelta(text="read the file"),
            ToolCallStarted(call_id="call_1", name="fake_tool"),
            ToolCallArgumentsDelta(call_id="call_1", text='{"a"'),
            ToolCallArgumentsDelta(call_id="call_1", text=": 1}"),
            ToolCallArgumentsDone(
                call_id="call_1", name="fake_tool", arguments_json='{"a": 1}',
            ),
            RoundComplete(has_tool_calls=True),
        ],
        delay=0.05,
    )
    session = make_session(client)

    asyncio.run(collect(session, "use the tool"))

    assistant = session.conversation.messages[1]
    # One delay (reasoning → first tool call), not four.
    assert 0.03 < assistant.timing.reasoning_seconds < 0.09


def test_usage_fallback_is_approximate_when_absent() -> None:
    client = FakeClient([ContentDelta(text="12345678"), RoundComplete(usage=None)])
    session = make_session(client)

    events = asyncio.run(collect(session, "abcd"))

    usage = events[-1].usage
    assert usage.approximate is True
    assert usage.output_tokens == 2  # 8 chars / 4
    assert usage.input_tokens > 0


def test_stream_error_leaves_history_consistent() -> None:
    client = FakeClient([StreamError(message="HTTP 500: boom")])
    session = make_session(client)

    events = asyncio.run(collect(session, "Hi"))

    assert isinstance(events[-1], StreamError)
    # The user message stays (visible in UI); no assistant message was added.
    assert [m.role for m in session.conversation.messages] == ["user"]


def test_multi_turn_history_accumulates() -> None:
    client = FakeClient([ContentDelta(text="one"), RoundComplete(usage=Usage(input_tokens=5, output_tokens=3))])
    session = make_session(client)
    asyncio.run(collect(session, "first"))

    client.script = [ContentDelta(text="two"), RoundComplete(usage=Usage(input_tokens=9, output_tokens=4))]
    asyncio.run(collect(session, "second"))

    # Second request carried the full running conversation.
    assert client.received_wire[-1] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": "second"},
    ]
    assert len(session.conversation.messages) == 4


def test_tool_call_round_records_wire_shaped_tool_calls() -> None:
    client = FakeClient(
        [
            ToolCallStarted(call_id="call_1", name="fake_tool"),
            ToolCallArgumentsDelta(call_id="call_1", text='{"a"'),
            ToolCallArgumentsDelta(call_id="call_1", text=": 1}"),
            ToolCallArgumentsDone(call_id="call_1", name="fake_tool", arguments_json='{"a": 1}'),
            RoundComplete(has_tool_calls=True),
        ]
    )
    session = make_session(client)

    events = asyncio.run(collect(session, "use the tool"))

    assert events[-1].has_tool_calls is True
    assistant = session.conversation.messages[1]
    assert assistant.role == "assistant"
    assert assistant.tool_calls == [ToolCall(
        call_id="call_1", name="fake_tool", arguments_json='{"a": 1}',
    )]


def test_continue_after_tools_sends_tool_result_and_omits_empty_content() -> None:
    client = FakeClient(
        [
            ToolCallStarted(call_id="call_1", name="fake_tool"),
            ToolCallArgumentsDone(call_id="call_1", name="fake_tool", arguments_json="{}"),
            RoundComplete(has_tool_calls=True),
        ]
    )
    session = make_session(client)
    asyncio.run(collect(session, "use the tool"))

    session.record_tool_result("call_1", ToolResult(content="success!"))
    client.script = [ContentDelta(text="all done"), RoundComplete(usage=Usage(input_tokens=4, output_tokens=2))]
    events = asyncio.run(_drain(session._stream_round()))

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
    assert [m.role for m in session.conversation.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


def test_tools_schema_is_passed_to_client() -> None:
    client = FakeClient([ContentDelta(text="hi"), RoundComplete()])
    session = make_session(client, build_registry(None, None))

    asyncio.run(collect(session, "hello"))

    assert client.received_tools == build_registry(None, None).schema()


def test_empty_registry_sends_no_tools() -> None:
    client = FakeClient([ContentDelta(text="hi"), RoundComplete()])
    session = make_session(client, ToolRegistry())

    asyncio.run(collect(session, "hello"))

    assert client.received_tools == []


def test_compact_replaces_history_with_summary() -> None:
    client = FakeClient([ContentDelta(text="one"), RoundComplete(usage=Usage(input_tokens=5, output_tokens=3))])
    session = make_session(client)
    session.system_prompt = "system prompt"
    asyncio.run(collect(session, "first question"))
    client.script = [ContentDelta(text="two"), RoundComplete(usage=Usage(input_tokens=9, output_tokens=4))]
    asyncio.run(collect(session, "second question"))

    client.script = [ContentDelta(text="summary text"), RoundComplete()]
    summary = asyncio.run(session.compact())

    from xarness.compaction import CompactionResult
    assert isinstance(summary, CompactionResult) and summary.summary == "summary text"
    # The whole visible history folds into one summary message.
    visible = session.conversation.visible_messages()
    assert [m.role for m in visible] == ["user"]
    assert visible[0].is_summary is True
    assert "summary text" in visible[0].content
    # The summarizer round continues the live conversation: the exact same
    # system prompt and history the model just saw (cacheable prefix), plus
    # one handoff instruction as a new final user message.
    summarizer_wire = client.received_wire[-1]
    assert summarizer_wire[0] == {"role": "system", "content": "system prompt"}
    assert summarizer_wire[:5] == client.received_wire[1] + [
        {"role": "assistant", "content": "two"},
    ]
    assert summarizer_wire[-1]["role"] == "user"
    instruction = summarizer_wire[-1]["content"]
    assert "handoff" in instruction.lower()
    assert "without tools" in instruction
    # The history is sent as-is, not flattened into the prompt.
    assert "first question" not in instruction
    assert "assistant: one" not in instruction
    # The summarizer round ran without tools.
    assert client.received_tools is None


# ---------------------------------------------------------------------------
# usage accounting (per-message + cumulative)


def test_usage_attaches_to_messages_and_totals() -> None:
    client = FakeClient([ContentDelta(text="one"), RoundComplete(usage=Usage(input_tokens=5, output_tokens=3))])
    session = make_session(client)
    asyncio.run(collect(session, "first"))

    client.script = [ContentDelta(text="two"), RoundComplete(usage=Usage(input_tokens=9, output_tokens=4))]
    asyncio.run(collect(session, "second"))

    messages = session.conversation.messages
    assert messages[1].usage == Usage(input_tokens=5, output_tokens=3)
    assert messages[3].usage == Usage(input_tokens=9, output_tokens=4)
    # Usage is local-only: never sent over the wire.
    assert all("usage" not in m for m in client.received_wire[-1])
    assert session.usage_total == Usage(input_tokens=14, output_tokens=7)


def test_send_snapshots_conversation_for_undo() -> None:
    client = FakeClient([ContentDelta(text="hi"), RoundComplete()])
    session = make_session(client)
    asyncio.run(collect(session, "hello"))

    # The user message owns its turn's checkpoint — even without git
    # tracking (an untracked turn simply has no trees and closes at turn
    # end, so a rollback never has to guess its scope).
    message = session.conversation.messages[0]
    turn = session.conversation.turns.get(message.id)
    assert turn is not None
    assert turn.before_tree is None  # no git tracking: no checkpoint
    assert message.content == "hello"


def test_compact_with_no_history_is_a_no_op() -> None:
    """Compacting a fresh conversation changes nothing."""
    client = FakeClient([ContentDelta(text="summary text"), RoundComplete()])
    session = make_session(client)

    assert asyncio.run(session.compact()) is None  # nothing to compact
    assert session.conversation.messages == []


def test_compact_reports_token_counts_and_folds_summarizer_usage() -> None:
    client = FakeClient([ContentDelta(text="one"), RoundComplete(usage=Usage(input_tokens=5, output_tokens=40))])
    session = make_session(client)
    asyncio.run(collect(session, "first question"))

    client.script = [ContentDelta(text="summary text"), RoundComplete(usage=Usage(input_tokens=100, output_tokens=7))]
    summary = asyncio.run(session.compact())

    assert summary.summary == "summary text"
    # The last real usage (input + output) is the context size; the
    # summary's own cost is estimated from its characters.
    assert (summary.before_tokens, summary.after_tokens) == (45, 3)
    # The summarization round's own spend is part of the session total.
    assert session.usage_total == Usage(input_tokens=105, output_tokens=47)
    # The summary message carries no usage of its own: right after a
    # compaction no visible message has usage, so the auto-compactor
    # cannot loop on it.
    assert all(
        getattr(m, "usage", None) is None
        for m in session.conversation.visible_messages()
    )


def test_undo_across_compaction_after_later_turn() -> None:
    """Auto-compaction at a turn boundary, later turns after it: /undo
    first removes the later turn, then a second /undo restores the
    pre-compaction history."""
    client = FakeClient([ContentDelta(text="one"), RoundComplete(usage=Usage(input_tokens=5, output_tokens=3))])
    session = make_session(client)
    asyncio.run(collect(session, "first"))

    # Turn 2: a normal answer; compaction then runs at the turn boundary.
    client.script = [ContentDelta(text="two"), RoundComplete(usage=Usage(input_tokens=4, output_tokens=1))]
    asyncio.run(collect(session, "second"))
    client.script = [ContentDelta(text="summary text"), RoundComplete(usage=Usage(input_tokens=100, output_tokens=7))]
    asyncio.run(session.compact())

    # Later turns happen after the compaction.
    client.script = [ContentDelta(text="final"), RoundComplete(usage=Usage(input_tokens=6, output_tokens=2))]
    asyncio.run(collect(session, "third"))
    client.script = [ContentDelta(text="fourth answer"), RoundComplete(usage=Usage(input_tokens=7, output_tokens=2))]
    asyncio.run(collect(session, "fourth"))
    assert [m.role for m in session.conversation.visible_messages()] == [
        "user", "user", "assistant", "user", "assistant",
    ]

    # First /undo: removes turn 4, back to the end of turn 3 (post-compaction).
    plan = session.conversation.rollback_plan()
    assert plan is not None and plan.user_text == "fourth"
    assert plan.compaction is False
    session.conversation.apply_rollback(plan)
    assert [m.role for m in session.conversation.visible_messages()] == [
        "user", "user", "assistant",
    ]
    
    # Second /undo: removes turn 3 — the first turn after the compaction.
    plan = session.conversation.rollback_plan()
    assert plan is not None and plan.user_text == "third"
    assert plan.compaction is False
    session.conversation.apply_rollback(plan)
    assert [m.role for m in session.conversation.visible_messages()] == ["user"]

    # Third /undo: undoes the compaction itself — the full pre-compaction
    # history comes back, turns 1 and 2 still intact.
    plan = session.conversation.rollback_plan()
    assert plan is not None and plan.compaction
    assert plan.user_text == ""
    assert plan.revert_turns == []  # compaction only: no files reverted
    session.conversation.apply_rollback(plan)
    assert [m.role for m in session.conversation.visible_messages()] == [
        "user", "assistant", "user", "assistant",
    ]
    assert session.conversation.messages[2].content == "second"
        # Totals are monotonic: everything ever spent (turns 1-2, the
    # summarizer, turns 3-4) stays counted.
    assert session.usage_total == Usage(input_tokens=122, output_tokens=15)

    # Fourth /undo: removes turn 2 normally.
    plan = session.conversation.rollback_plan()
    assert plan is not None and plan.user_text == "second"
    assert plan.compaction is False
    session.conversation.apply_rollback(plan)
    assert [m.content for m in session.conversation.visible_messages()] == ["first", "one"]
    assert session.conversation.messages[1].role == "assistant"
    # Totals unchanged: nothing new was spent, and undo never subtracts.
    assert session.usage_total == Usage(input_tokens=122, output_tokens=15)

    # A further /undo walks back the pre-compaction turns normally.
    plan = session.conversation.rollback_plan()
    assert plan is not None and plan.user_text == "first"
    session.conversation.apply_rollback(plan)
    assert session.conversation.visible_messages() == []
    # Totals are monotonic: everything ever spent stays counted.
    assert session.usage_total == Usage(input_tokens=122, output_tokens=15)


def test_resumed_compacted_session_undoes_across_compaction(tmp_path) -> None:
    """Save a compacted session, resume it, and undo across the compaction."""

    from xarness import session_store
    from xarness.conversation import SUMMARY_PREFIX

    session_store.SESSIONS_DIR = tmp_path
    client = FakeClient([ContentDelta(text="one"), RoundComplete(usage=Usage(input_tokens=5, output_tokens=3))])
    session = make_session(client)
    asyncio.run(collect(session, "first"))

    # Turn 2: a normal answer; compaction then runs at the turn boundary,
    # and one more turn happens after it.
    client.script = [ContentDelta(text="two"), RoundComplete(usage=Usage(input_tokens=4, output_tokens=1))]
    asyncio.run(collect(session, "second"))
    client.script = [ContentDelta(text="summary text"), RoundComplete(usage=Usage(input_tokens=100, output_tokens=7))]
    asyncio.run(session.compact())
    client.script = [ContentDelta(text="final"), RoundComplete(usage=Usage(input_tokens=6, output_tokens=2))]
    asyncio.run(collect(session, "third"))

    _save("s", session_store.StoredSession(
        profile="test-model", conversation=session.conversation,
    ))
    resumed_conversation = session_store.load_state("s").conversation
    assert any(
        m.content.startswith(SUMMARY_PREFIX)
        for m in resumed_conversation.visible_messages()
    )

    session2 = make_session(client)
    session2.conversation = resumed_conversation
    # First /undo: removes turn 3 — the first turn after the compaction.
    plan = session2.conversation.rollback_plan()
    assert plan is not None and plan.user_text == "third"
    session2.conversation.apply_rollback(plan)
    # Second /undo: undoes the compaction itself — full pre-compaction
    # history restored, turns 1 and 2 intact.
    plan = session2.conversation.rollback_plan()
    assert plan is not None and plan.compaction
    session2.conversation.apply_rollback(plan)
    assert [m.role for m in session2.conversation.visible_messages()] == [
        "user", "assistant", "user", "assistant",
    ]
    assert session2.conversation.messages[2].content == "second"
    # Third /undo removes turn 2 normally.
    plan = session2.conversation.rollback_plan()
    assert plan is not None and plan.user_text == "second"
    session2.conversation.apply_rollback(plan)
    assert [m.role for m in session2.conversation.visible_messages()] == ["user", "assistant"]
    assert session2.conversation.messages[1].content == "one"


def test_rollback_plan_drops_the_turn_for_undo_and_retry() -> None:
    """Both /undo and /retry truncate to before the turn's user message;
    /retry re-sends the text (with a fresh checkpoint) via send()."""
    client = FakeClient([ContentDelta(text="one"), RoundComplete(usage=Usage(input_tokens=5, output_tokens=3))])
    session = make_session(client)
    asyncio.run(collect(session, "first"))
    client.script = [ContentDelta(text="two"), RoundComplete(usage=Usage(input_tokens=9, output_tokens=4))]
    asyncio.run(collect(session, "second"))

    plan = session.conversation.rollback_plan()
    assert plan is not None
    assert plan.user_text == "second"
    assert [m.content for m in [m for m in session.conversation.messages[:plan.drop_from]]] == ["first", "one"]

    session.conversation.apply_rollback(plan)
    assert [m.content for m in session.conversation.messages] == [
        "first", "one",
    ]
    # Totals are monotonic: the undone turn's spend stays counted.
    assert session.usage_total == Usage(input_tokens=14, output_tokens=7)


def test_rollback_without_snapshot_truncates_at_last_user_message() -> None:
    """Resumed sessions have no snapshot: fall back to the last user message.
    Its persisted before/after trees drive the file revert."""
    session = make_session(FakeClient([]))
    session.conversation.add(user_message("old"))
    session.conversation.add(AssistantMessage(content="a", usage=Usage(input_tokens=5, output_tokens=3)))
    last = session.conversation.add(user_message("last"))
    session.conversation.begin_turn(last, "abc123")
    conversation_turn = session.conversation.turns[last.id]
    conversation_turn.after_tree = "def456"
    session.conversation.add(AssistantMessage(content="b", usage=Usage(input_tokens=9, output_tokens=4)))
    session.usage_total = Usage(input_tokens=14, output_tokens=7)

    plan = session.conversation.rollback_plan()
    assert plan is not None
    assert plan.user_text == "last"
    assert plan.revert_turns == [last.id]
    session.conversation.apply_rollback(plan)
    assert [m.content for m in session.conversation.visible_messages()] == ["old", "a"]
    # Totals are monotonic: the undone turn's spend stays counted.
    assert session.usage_total == Usage(input_tokens=14, output_tokens=7)


def test_rollback_is_a_noop_without_a_finished_turn() -> None:
    session = make_session(FakeClient([]))
    assert session.conversation.rollback_plan() is None  # empty conversation
    asyncio.run(collect(session, "only a user message"))
    assert session.conversation.rollback_plan() is None  # no response yet


def test_rollback_plan_to_an_arbitrary_earlier_turn() -> None:
    """Click-to-undo truncates at any earlier user message, restoring its
    checkpoint and subtracting every dropped turn's usage."""
    client = FakeClient([ContentDelta(text="one"), RoundComplete(usage=Usage(input_tokens=5, output_tokens=3))])
    session = make_session(client)
    asyncio.run(collect(session, "first"))
    client.script = [ContentDelta(text="two"), RoundComplete(usage=Usage(input_tokens=9, output_tokens=4))]
    asyncio.run(collect(session, "second"))
    client.script = [ContentDelta(text="three"), RoundComplete(usage=Usage(input_tokens=7, output_tokens=2))]
    asyncio.run(collect(session, "third"))

    first_user = session.conversation.messages[0]
    plan = session.conversation.rollback_plan_to(first_user)
    assert plan is not None
    assert plan.user_text == "first"
    assert [m for m in session.conversation.messages[:plan.drop_from]] == []

    session.conversation.apply_rollback(plan)
    assert session.conversation.messages == []
    # Totals are monotonic: all three turns' spend stays counted.
    assert session.usage_total == Usage(input_tokens=21, output_tokens=9)


def test_rollback_plan_to_rejects_non_turns_allows_the_latest() -> None:
    session = make_session(FakeClient([]))
    session.conversation.add(user_message("first"))
    session.conversation.add(AssistantMessage(content="one"))
    session.conversation.add(user_message("pending"))
    messages = session.conversation.messages
    assert session.conversation.rollback_plan_to(messages[1]) is None  # not a user turn
    # The latest user message is still undoable: nothing to drop or revert,
    # but the plan puts its text back in the input.
    plan = session.conversation.rollback_plan_to(messages[2])
    assert plan is not None
    assert plan.user_text == "pending"
    assert [m for m in session.conversation.messages[:plan.drop_from]] == messages[:2]
    assert plan.revert_turns == []


def test_save_interrupted_round_keeps_partial_answer() -> None:
    """An interrupted round is saved with the text and reasoning it managed
    to stream before the cancellation."""
    session = make_session(FakeClient([]))
    session.partial_round = PartialRound(
        reasoning="thinking ", content="partial answer")
    assert session._save_interrupted_round() is True
    message = session.conversation.messages[-1]
    assert message.role == "assistant"
    assert message.content == "partial answer"
    assert message.reasoning == "thinking "
    # Cleared, so a second call is a no-op (the round is only saved once).
    assert session._save_interrupted_round() is False


def test_save_interrupted_round_keeps_reasoning_only() -> None:
    """A round interrupted while still thinking (no answer text) saves the
    partial CoT: it belongs in the history and is sent with it (the same
    keep_reasoning rule as any assistant message's reasoning)."""
    session = make_session(FakeClient([]))
    session.partial_round = PartialRound(reasoning="just thinking")
    assert session._save_interrupted_round() is True
    (message,) = session.conversation.messages
    assert message.role == "assistant"
    assert message.reasoning == "just thinking"
    assert message.content == ""
    # The reasoning rides along on the wire (keep_reasoning on by default).
    wire = message.to_wire(include_reasoning=PROFILE.keep_reasoning)
    assert wire["reasoning_content"] == "just thinking"
    # Cleared, so a second call is a no-op (the round is only saved once).
    assert session._save_interrupted_round() is False


def test_save_interrupted_round_drops_unpaired_tool_calls() -> None:
    """Half-streamed tool calls are dropped: they have no result to pair with,
    and an orphaned tool call would be rejected on the next request."""
    session = make_session(FakeClient([]))
    session.partial_round = PartialRound()
    assert session._save_interrupted_round() is False
    assert session.conversation.messages == []


async def _drain(stream: AsyncIterator[Any]) -> list[Any]:
    return [event async for event in stream]


def _image(name: str = "cat.png") -> ImageAttachment:
    data_b64 = base64.b64encode(name.encode()).decode()
    return ImageAttachment(
        id=hashlib.sha256(name.encode()).hexdigest(), name=name,
        mime="image/png", data_b64=data_b64, width=1, height=1,
    )


STORE = SessionImageStore("test", persist=False)


def test_rollback_plan_keeps_images() -> None:
    session = make_session(FakeClient([]))
    image = _image()
    session.conversation.add(user_message("look", [image], STORE))
    session.conversation.add(AssistantMessage(content="ok"))
    plan = session.conversation.rollback_plan()
    assert plan is not None and [i.id for i in plan.images] == [image.id]


def test_image_token_maps_by_text_not_ordinal() -> None:
    image = _image()
    text = f"[🖼 stale 1×1 #0000] mid {token_text(image)} end"
    msg = user_message(text, [image], STORE)
    parts = msg.to_wire(image_loader=STORE.data_url)["content"]
    image_at = next(i for i, p in enumerate(parts) if p["type"] == "image_url")
    before = "".join(p["text"] for p in parts[:image_at] if p["type"] == "text")
    assert "stale" in before and "cat.png" not in before
