"""Session tests: single-writer, fan-out, atomic save, tool repair, undo."""

import asyncio
import base64
import json
import time
from typing import Any
from unittest import mock

import pytest

from xarness import gitwork, protocol, session_store
from xarness.config import ProviderProfile
from xarness.conversation import (
    SUMMARY_PREFIX, AssistantMessage, Timing, ToolCall, ToolMessage, user_message,
    Conversation,
)
from xarness.images import ImageAttachment
from xarness.protocol import (
    Accept,
    AnswerAsk,
    Cancel,
    Compact,
    ContentDelta,
    Filter,
    ProcessingStarted,
    ReasoningDelta,
    Reject,
    SendMessage,
    SetMode,
    StreamError,
    ToolCallArgumentsDone,
    ToolCallStarted,
    RoundComplete,
    Rollback,
    Usage,
)
from xarness.errors import SessionError
from xarness.session import Session
from xarness.sandbox import SandboxConfig, SandboxRuntime
from xarness.workspace import WorkspaceState, open_workspace
from pydantic import BaseModel as _PydanticBase

from xarness.tools import ToolRegistry, ToolResult, tool

PROFILE = ProviderProfile(
    base_url="https://api.example.test/v1", model_id="test-model", max_context=1000,
)


class FakeClient:
    """Replays scripted rounds; each stream() call consumes the next round."""

    def __init__(self, script: list[list[Any]]) -> None:
        self.script = list(script)
        self.received_wire: list[list[dict[str, Any]]] = []

    async def stream(self, wire_messages, tools=None):
        self.received_wire.append(list(wire_messages))
        yield ProcessingStarted()
        for event in (self.script.pop(0) if self.script else []):
            yield event


@pytest.fixture(autouse=True)
def _sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path / "sessions")


def make_session(script, tmp_path, tools=None) -> Session:
    session = Session(
        name="test", workspace=tmp_path, profile=PROFILE, profile_name="test",
        api_key="key", client=FakeClient(script),
    )
    if tools is not None:
        session.tool_registry = tools
        session.tool_registry = tools
    return session


async def wait_idle(session: Session, timeout: float = 10.0) -> None:
    """Session.wait_idle with a test timeout, so a stuck turn fails loudly."""
    await asyncio.wait_for(session.wait_idle(), timeout=timeout)


def mark_running(session: Session) -> asyncio.Task[None]:
    """Make ``session`` look mid-turn without running one (then cancel)."""
    task = asyncio.get_running_loop().create_task(asyncio.sleep(3600))
    session._turn_task = task
    return task


def _image() -> ImageAttachment:
    import hashlib
    return ImageAttachment(
        id=hashlib.sha256(b"cat").hexdigest(), name="cat.png", mime="image/png",
        data_b64=base64.b64encode(b"cat").decode(), width=1, height=1,
    )


def _round(text: str = "ok") -> list[Any]:
    return [ContentDelta(text=text), RoundComplete(usage=Usage(input_tokens=1, output_tokens=1))]


def test_undo_and_retry_keep_images(tmp_path) -> None:
    session = make_session([_round("one"), _round("two")], tmp_path)

    async def go():
        session.images.save(_image())
        await session.execute(SendMessage(
            text="look [x]", images=[_image()],
        ))
        await wait_idle(session)
        await session.execute(Rollback(resend=True))
        await wait_idle(session)
        return await session.execute(Rollback())

    undone = asyncio.run(go())
    assert undone.images[0].name == "cat.png"
    wire = session._client.received_wire[-1]
    assert any(
        isinstance(m["content"], list)
        and any(p.get("type") == "image_url" for p in m["content"])
        for m in wire if m["role"] == "user"
    )


def test_rebuilds_preserve_output_limit_and_reject_bad_filters(tmp_path) -> None:
    session = Session(
        name="test", workspace=tmp_path, profile=PROFILE, profile_name="test",
        api_key="key", client=FakeClient([]), tool_output_limit=4096,
    )

    async def go():
        await session.execute(SetMode(mode="plan"))
        for bad in (".", "..", "/etc"):
            with pytest.raises(SessionError):
                await session.execute(Filter(path=bad, add=True))
        assert session.tool_registry.max_output_chars == 4096

    asyncio.run(go())


def test_send_runs_turn_and_fans_out_no_gap(tmp_path) -> None:
    session = make_session(
        [[ReasoningDelta(text="hmm"), ContentDelta(text="Hello"), RoundComplete(usage=Usage(input_tokens=10, output_tokens=2))]],
        tmp_path,
    )

    async def go():
        snapshot, queue = session.subscribe()
        await session.execute(SendMessage(text="hi"))
        await wait_idle(session)
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        return snapshot, events

    snapshot, events = asyncio.run(go())
    types = [type(e).__name__ for e in events]
    # No local echo: the user message arrives via the stream (after the
    # running status), then the round's events.
    assert types[0] == "StatusChanged"
    assert types[1] == "MessageAdded"
    assert "TurnEnded" in types
    # Snapshot + subscribe has no gap and no duplicate.
    assert snapshot.messages == []


class _NoArgs(_PydanticBase):
    pass


def test_status_refresh_per_round_during_streaming(tmp_path) -> None:
    """Each round folds its usage; a StatusChanged must follow so clients
    update in/out/ctx per round, not only when the turn ends."""
    noop = tool("noop", "does nothing", _NoArgs)(lambda: "ok")
    session = make_session(
        [
            [
                ContentDelta(text="one"),
                ToolCallStarted(call_id="c1", name="noop"),
                ToolCallArgumentsDone(call_id="c1", name="noop", arguments_json="{}"),
                RoundComplete(usage=Usage(input_tokens=100, output_tokens=10)),
            ],
            [ContentDelta(text="two"), RoundComplete(usage=Usage(input_tokens=200, output_tokens=20))],
        ],
        tmp_path, tools=ToolRegistry([noop]),
    )

    async def go():
        snapshot, queue = session.subscribe()
        await session.execute(SendMessage(text="hi"))
        await wait_idle(session)
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        return snapshot, events

    snapshot, events = asyncio.run(go())
    statuses = [e.status for e in events if isinstance(e, protocol.StatusChanged)]
    totals = [(s.usage_total.input_tokens, s.usage_total.output_tokens) for s in statuses]
    # The per-round refreshes appear in order: after round 1 and after round 2.
    assert (100, 10) in totals and (300, 30) in totals, totals
    assert totals.index((100, 10)) < totals.index((300, 30))
    assert statuses[-1].last_usage.output_tokens > 0


def test_history_replaced_keeps_live_partial_round(tmp_path) -> None:
    """A rewrite mid-turn must not lose the partial answer: the snapshot
    carries it alongside the fresh history, so attach renders once."""
    session = make_session([[ContentDelta(text="x"), RoundComplete()]], tmp_path)

    async def go():
        session.subscribe()
        session.partial_round.content = "before"
        assert session.snapshot().partial_round.content == "before"
        session._commit_history('rollback')
        assert session.snapshot().partial_round.content == "before"

    asyncio.run(go())


def test_send_while_running_becomes_steering(tmp_path) -> None:
    release = asyncio.Event()

    @tool("noop", "does nothing", _NoArgs)
    async def noop(args) -> ToolResult:
        await release.wait()
        return ToolResult()

    registry = ToolRegistry()
    registry.register(noop)
    session = make_session(
        [
            [
                ToolCallStarted(call_id="c1", name="noop"),
                ToolCallArgumentsDone(call_id="c1", name="noop", arguments_json="{}"),
                RoundComplete(has_tool_calls=True),
            ],
            [ContentDelta(text="second round"), RoundComplete()],
        ],
        tmp_path, tools=registry,
    )

    async def go():
        session.subscribe()
        await session.execute(SendMessage(text="go"))
        await session.execute(SendMessage(text="steer me"))
        assert session._queued, "message should be queued while running"
        release.set()
        await wait_idle(session)
        # Injected at the round boundary as a proper user message.
        user_texts = [m.content for m in session.conversation.messages if m.role == "user"]
        assert "steer me" in user_texts
        assert session.last_user_message_at() is not None

    asyncio.run(go())


def test_several_steering_messages_merge_into_one(tmp_path) -> None:
    """Steers queued during the same round reach the history as one user
    message, not one message per send."""
    release = asyncio.Event()

    @tool("noop", "does nothing", _NoArgs)
    async def noop(args) -> ToolResult:
        await release.wait()
        return ToolResult()

    registry = ToolRegistry()
    registry.register(noop)
    session = make_session(
        [
            [
                ToolCallStarted(call_id="c1", name="noop"),
                ToolCallArgumentsDone(call_id="c1", name="noop", arguments_json="{}"),
                RoundComplete(has_tool_calls=True),
            ],
            [ContentDelta(text="second round"), RoundComplete()],
        ],
        tmp_path, tools=registry,
    )

    async def go():
        session.subscribe()
        await session.execute(SendMessage(text="go"))
        await session.execute(SendMessage(text="steer one"))
        await session.execute(SendMessage(text="steer two"))
        assert len(session._queued) == 2
        release.set()
        await wait_idle(session)

    asyncio.run(go())
    user_texts = [
        m.content for m in session.conversation.messages if m.role == "user"
    ]
    # Both steers landed in the same message, in send order.
    assert user_texts[1:] == ["steer one\n\nsteer two"]


def test_rewrite_commands_rejected_while_busy(tmp_path) -> None:
    release = asyncio.Event()

    @tool("blocker", "blocks", _NoArgs)
    async def blocker(args) -> ToolResult:
        await release.wait()
        return ToolResult()

    registry = ToolRegistry()
    registry.register(blocker)
    session = make_session(
        [
            [
                ToolCallStarted(call_id="c1", name="blocker"),
                ToolCallArgumentsDone(call_id="c1", name="blocker", arguments_json="{}"),
                RoundComplete(has_tool_calls=True),
            ],
            [ContentDelta(text="x"), RoundComplete()],
        ],
        tmp_path, tools=registry,
    )

    async def go():
        session.subscribe()
        await session.execute(SendMessage(text="hi"))
        with pytest.raises(SessionError, match="wait for"):
            await session.execute(Rollback(message_id="whatever"))
        release.set()
        await wait_idle(session)

    asyncio.run(go())


def test_cancel_interrupts_turn(tmp_path) -> None:
    session = make_session([[ContentDelta(text="partial"), RoundComplete()]], tmp_path)

    async def go():
        session.subscribe()
        await session.execute(SendMessage(text="hi"))
        await session.execute(Cancel())
        await wait_idle(session)
        assert session.state == "idle"

    asyncio.run(go())


def _blocking_session(tmp_path, started: asyncio.Event) -> Session:
    """A session whose provider starts a round and then hangs, so the test
    controls the moment the turn is still running."""
    class BlockingClient:
        async def stream(self, wire_messages, tools=None):
            yield ProcessingStarted()
            started.set()
            await asyncio.sleep(30)

    return Session(
        name="test", workspace=tmp_path, profile=PROFILE, profile_name="test",
        api_key="key", client=BlockingClient(),
    )


def _tracked_workspace(tmp_path) -> WorkspaceState:
    """A workspace state whose snapshots are faked by the caller."""
    return WorkspaceState(
        root=tmp_path, git_dir=tmp_path,
        baseline_tree="a" * 40, head_tree="a" * 40,
    )


def _slow_after_tree(monkeypatch, session, gate: asyncio.Event):
    """First snapshot (turn start) is instant; the after-tree snapshot
    blocks on ``gate``, widening the teardown window a Cancel can land in."""
    calls = 0

    async def snapshot():
        nonlocal calls
        calls += 1
        if calls == 1:
            return "a" * 40
        gate.set()
        await asyncio.sleep(0.2)
        return "b" * 40

    session.workspace_state._snapshot = snapshot

    async def noop():
        pass

    monkeypatch.setattr(session, "_emit_diff_changed", noop)


def test_cancel_while_turn_is_finishing_does_not_strand_running(
    tmp_path, monkeypatch,
) -> None:
    """The footer still says "running" during the end-of-turn snapshot, so a
    Cancel can land in the teardown of an *already finished* turn. It used to
    abort the state flip and leave a phantom "running" session with no task;
    every later attach then rendered a live turn that did not exist."""
    session = make_session([_round("done")], tmp_path)
    session.workspace_state = _tracked_workspace(tmp_path)
    gate = asyncio.Event()
    _slow_after_tree(monkeypatch, session, gate)

    async def go():
        queue = session.subscribe()[1]
        await session.execute(SendMessage(text="hi"))
        await asyncio.wait_for(gate.wait(), 5)
        assert session.state == "running"  # teardown has not flipped yet
        await session.execute(Cancel())
        await wait_idle(session)
        assert session._turn_task is None
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        assert any(isinstance(e, protocol.TurnEnded) for e in events)

    asyncio.run(go())


def test_repeated_cancel_during_unwind_does_not_strand_running(
    tmp_path, monkeypatch,
) -> None:
    """A second Cancel while the interrupted turn is still unwinding must not
    abort its teardown (the classic phantom-running state)."""
    started = asyncio.Event()
    session = _blocking_session(tmp_path, started)
    session.workspace_state = _tracked_workspace(tmp_path)
    gate = asyncio.Event()
    _slow_after_tree(monkeypatch, session, gate)

    async def go():
        session.subscribe()
        await session.execute(SendMessage(text="hi"))
        await asyncio.wait_for(started.wait(), 5)
        await session.execute(Cancel())
        await asyncio.wait_for(gate.wait(), 5)
        # The user, still seeing "running", cancels again — and again.
        for i in range(3):
            await session.execute(Cancel())
        await wait_idle(session)
        assert session._turn_task is None

    asyncio.run(go())


def test_teardown_failure_still_flips_idle(tmp_path, monkeypatch) -> None:
    """A failing after-tree snapshot/persist must not skip the state flip:
    the session reports idle and the turn end is still announced."""
    session = make_session([_round("done")], tmp_path)
    session.workspace_state = _tracked_workspace(tmp_path)

    async def boom() -> None:
        raise RuntimeError("snapshot store exploded")

    async def noop():
        pass

    monkeypatch.setattr(session, "_finish_turn", boom)
    monkeypatch.setattr(session, "_emit_diff_changed", noop)

    async def go():
        queue = session.subscribe()[1]
        await session.execute(SendMessage(text="hi"))
        await wait_idle(session)
        assert session._turn_task is None
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        assert any(isinstance(e, protocol.TurnEnded) for e in events)

    asyncio.run(go())


def test_a_turn_that_dies_before_teardown_still_reports_its_end(
    tmp_path, monkeypatch,
) -> None:
    """A task that dies in its own teardown still reports the turn ended."""
    session = make_session([_round("done")], tmp_path)

    async def boom(outcome, turn_start) -> None:
        raise RuntimeError("teardown skipped")

    monkeypatch.setattr(session, "_close_out_turn", boom)

    async def go():
        queue = session.subscribe()[1]
        await session.execute(SendMessage(text="hi"))
        await asyncio.sleep(0.05)
        await wait_idle(session)
        assert session._turn_task is None
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        assert any(isinstance(e, protocol.TurnEnded) for e in events)
        # Nothing to heal: a dead task simply is not running any more.
        assert session.state == "idle"
        await session.execute(Cancel())
        session.unsubscribe(queue)
        assert not session.is_active

    asyncio.run(go())


def test_attach_does_not_touch_a_real_running_turn(tmp_path) -> None:
    """The attach-time check must not mistake a live turn for a stranded
    one: the snapshot shows it running and it finishes normally."""
    started = asyncio.Event()
    session = _blocking_session(tmp_path, started)

    async def go():
        session.subscribe()
        await session.execute(SendMessage(text="hi"))
        await asyncio.wait_for(started.wait(), 5)
        snapshot, _queue = session.subscribe()
        assert snapshot.status.state == "running"
        await session.execute(Cancel())
        await wait_idle(session)

    asyncio.run(go())


def test_quit_verdict_asks_only_for_the_last_viewer_of_work_in_flight(tmp_path) -> None:
    """The quit menu is the host's call, not the client's: work in flight
    AND nobody else streaming is the only case that needs an answer."""
    session = make_session([], tmp_path)

    async def go():
        assert session.quit_verdict().ask is False      # idle, nobody watching
        first = session.subscribe()[1]
        second = session.subscribe()[1]
        # Work in flight, but another client is still streaming it: quitting
        # is just one viewer leaving.
        task = mark_running(session)
        verdict = session.quit_verdict()
        assert verdict.ask is False and verdict.other_clients == 1
        assert verdict.state == "running"
        # The other client leaves: closing would now end the turn unseen.
        session.unsubscribe(second)
        verdict = session.quit_verdict()
        assert verdict.ask is True and verdict.other_clients == 0
        # A pending question counts as work in flight, even when idle.
        task.cancel()
        session._pending_ask = protocol.PendingAsk(ask_id="a1", questions=["q?"])
        verdict = session.quit_verdict()
        assert verdict.ask is True and verdict.pending_ask is True
        session.unsubscribe(first)

    asyncio.run(go())


def test_ask_first_answer_wins(tmp_path) -> None:
    session = make_session([[ContentDelta(text="x"), RoundComplete()]], tmp_path)

    async def go():
        queue = session.subscribe()[1]
        ask_task = asyncio.get_running_loop().create_task(
            session._ask_user([protocol.AskQuestion(question="what next?")]),
        )
        await asyncio.sleep(0)
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        asked = [e for e in events if isinstance(e, protocol.StatusChanged)
                 and e.status.pending_ask is not None]
        assert len(asked) == 1
        ask_id = asked[0].status.pending_ask.ask_id
        await session.execute(AnswerAsk(ask_id=ask_id, answers=["do it"]))
        with pytest.raises(SessionError):  # first answer wins
            await session.execute(AnswerAsk(ask_id=ask_id, answers=["no"]))
        assert await ask_task == ["do it"]

    asyncio.run(go())


def test_pending_ask_still_parses_plain_string_questions() -> None:
    """Bare-string questions from an older client parse as free text."""
    ask = protocol.PendingAsk(ask_id="a1", questions=["q?", {"question": "r?"}])
    assert [(q.question, q.options) for q in ask.questions] == [
        ("q?", []), ("r?", []),
    ]


def test_tool_call_repair_after_cancel(tmp_path) -> None:
    @tool("slow", "slow tool", _NoArgs)
    async def slow(args) -> ToolResult:
        await asyncio.sleep(30)
        return ToolResult()

    registry = ToolRegistry()
    registry.register(slow)
    session = make_session(
        [[
            ToolCallStarted(call_id="c1", name="slow"),
            ToolCallArgumentsDone(call_id="c1", name="slow", arguments_json="{}"),
            RoundComplete(has_tool_calls=True),
        ]],
        tmp_path, tools=registry,
    )

    async def go():
        session.subscribe()
        await session.execute(SendMessage(text="hi"))
        await session.execute(Cancel())
        await wait_idle(session, timeout=15)

    asyncio.run(go())
    # The interrupted call got a synthetic error result, so the next API
    # request is valid (every tool call is answered).
    tool_messages = [m for m in session.conversation.messages if m.role == "tool"]
    assert [m.content for m in tool_messages] == ["interrupted"]
    assert all(m.status != 'ok' for m in tool_messages)
    wire = session.conversation.to_wire()
    answered = {m["tool_call_id"] for m in wire if m["role"] == "tool"}
    for m in wire:
        for call in m.get("tool_calls") or []:
            assert call["id"] in answered


def test_undo_to_by_message_id(tmp_path) -> None:
    session = make_session(
        [[ContentDelta(text="answer"), RoundComplete(usage=Usage(input_tokens=5, output_tokens=2))]], tmp_path,
    )

    async def go():
        await session.execute(SendMessage(text="first question"))
        await wait_idle(session)
        user = next(m for m in session.conversation.messages if m.role == "user")
        return await session.execute(Rollback(message_id=user.id))

    result = asyncio.run(go())
    assert result.user_text == "first question"
    # The turn was dropped: nothing remains.
    assert session.conversation.messages == []


def test_atomic_save_and_timestamps(tmp_path, monkeypatch) -> None:
    session = make_session(
        [[ContentDelta(text="answer"), RoundComplete(usage=Usage(input_tokens=5, output_tokens=2))]], tmp_path,
    )

    async def go():
        await session.execute(SendMessage(text="hello"))
        await wait_idle(session)

    asyncio.run(go())
    # Atomic write: no temp file left behind, and the saved file parses.
    leftovers = list((tmp_path / "sessions").glob("*.tmp"))
    assert leftovers == []
    state = session_store.load_state("test")
    assert state.profile == "test"  # the profile key, not model_id
    user = state.conversation.messages[0]
    assistant = state.conversation.messages[1]
    assert user.timing.send is not None
    assert assistant.timing.stats is not None  # the turn's measured numbers
    # Timestamps are never sent to the provider.
    wire = session._client.received_wire[-1]
    for message in wire:
        assert set(message) <= {
            "role", "content", "name", "tool_call_id", "tool_calls", "reasoning_content",
        }


def test_user_message_is_on_disk_while_the_turn_runs(tmp_path) -> None:
    """A host killed mid-turn must not lose what the user asked: the
    message is persisted when it lands, not only at turn end."""
    started = asyncio.Event()
    session = _blocking_session(tmp_path, started)

    async def go():
        session.subscribe()
        await session.execute(SendMessage(text="hello there"))
        await asyncio.wait_for(started.wait(), 5)
        await session._flush_persist()
        assert session.state == "running"  # the turn has not finished
        stored = session_store.load_state("test").conversation
        assert [m.role for m in stored.messages][:1] == ["user"]
        assert stored.messages[0].content == "hello there"

    asyncio.run(go())


def test_undo_and_retry_commands_happy_path(tmp_path) -> None:
    """The /undo and /retry commands actually run their handlers: undo
    drops the last turn and reports the user text; retry drops the turn and
    resends it."""
    script = [
        [ContentDelta(text="first answer"), RoundComplete(usage=Usage(input_tokens=5, output_tokens=2))],
        [ContentDelta(text="second answer"), RoundComplete(usage=Usage(input_tokens=5, output_tokens=2))],
        [ContentDelta(text="second answer"), RoundComplete(usage=Usage(input_tokens=5, output_tokens=2))],
    ]

    async def go():
        # /undo: drops the last turn, hands the user text back.
        session = make_session([list(s) for s in script[:2]], tmp_path)
        await session.execute(SendMessage(text="first question"))
        await wait_idle(session)
        await session.execute(SendMessage(text="second question"))
        await wait_idle(session)
        undo = await session.execute(Rollback())
        assert undo.user_text == "second question"
        return session

    undone = asyncio.run(go())
    assert [
        m.content for m in undone.conversation.messages if m.role == "assistant"
    ] == ["first answer"]

    async def go2():
        # /retry: drops the last turn and resends its user message.
        session = make_session(script, tmp_path)
        await session.execute(SendMessage(text="first question"))
        await wait_idle(session)
        await session.execute(SendMessage(text="second question"))
        await wait_idle(session)
        await session.execute(Rollback(resend=True))
        async def _wait_answers():
            # The retry replaces the turn: the resent question gets a fresh
            # answer (assistant messages after the resend are stamped anew).
            while session._turn_task is not None or session.state != "idle":
                await asyncio.sleep(0.005)
        await asyncio.wait_for(_wait_answers(), timeout=5)
        return session

    retried = asyncio.run(go2())
    assert [
        m.content for m in retried.conversation.messages if m.role == "assistant"
    ] == ["first answer", "second answer"]
    user = [m.content for m in retried.conversation.messages if m.role == "user"]
    assert user == ["first question", "second question"]
    # The resent turn is a fresh exchange: a new user message was added.
    assert len(retried.conversation.messages) >= 4


def test_send_during_finish_turn_is_queued_not_started(tmp_path) -> None:
    """Auto-compact runs inside the turn's non-idle window: a send that
    arrives while finish_turn is still unwinding is queued, and only the
    turn's finally sets idle (a send must never start a second turn while
    the first is finishing)."""
    finish_started = asyncio.Event()
    finish_gate = asyncio.Event()

    session = make_session(
        [[ContentDelta(text="answer"), RoundComplete(usage=Usage(input_tokens=900, output_tokens=100))]],
        tmp_path,
    )
    session.auto_compact = True

    async def fake_compact():
        return None  # nothing to compact

    async def fake_finish_turn() -> None:
        finish_started.set()
        await finish_gate.wait()

    session.compact = fake_compact
    session._finish_turn = fake_finish_turn

    async def go():
        session.subscribe()
        first = asyncio.create_task(session.execute(SendMessage(text="hi")))
        await asyncio.wait_for(finish_started.wait(), timeout=5)
        # Auto-compact ran; the session is NOT idle while finish_turn blocks.
        assert session.state != "idle"
        await session.execute(SendMessage(text="go again"))
        assert session._queued and session._queued[0].text == "go again"
        finish_gate.set()
        await first
        await wait_idle(session)

    asyncio.run(go())
    user_texts = [m.content for m in session.conversation.messages if m.role == "user"]
    assert "go again" in user_texts
    assert session._turn_task is None


def test_manual_compact_does_not_block_inbox_and_drains_queue(tmp_path) -> None:
    """A manual compaction runs as a task (the inbox keeps serving commands)
    and the steering queue is drained when it finishes."""
    gate = asyncio.Event()
    session = make_session([[ContentDelta(text="x"), RoundComplete()]], tmp_path)

    async def fake_compact():
        await gate.wait()
        return None  # nothing to compact

    session.compact = fake_compact

    async def go():
        session.subscribe()
        # Returns immediately instead of awaiting the compaction.
        await asyncio.wait_for(session.execute(Compact()), timeout=2)
        assert session.state == "compacting"
        assert session._compact_task is not None
        # The inbox is still responsive while compaction runs.
        await asyncio.wait_for(
            session.execute(SendMessage(text="after compact")), timeout=2,
        )
        assert session._queued
        gate.set()
        await wait_idle(session)

    asyncio.run(go())
    user_texts = [m.content for m in session.conversation.messages if m.role == "user"]
    assert "after compact" in user_texts


def test_compaction_emits_typed_outcome_event(tmp_path) -> None:
    """Compaction reports its outcome as a HistoryReplaced, not prose."""
    session = make_session([[ContentDelta(text="x"), RoundComplete()]], tmp_path)

    async def fake_compact():
        from xarness.compaction import CompactionResult
        return CompactionResult(summary="handoff summary", usage=None,
                                before_tokens=100, after_tokens=30)

    session.compact = fake_compact

    async def go():
        queue = session.subscribe()[1]
        await session.execute(Compact())
        await wait_idle(session)
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        done = [e for e in events if isinstance(e, protocol.HistoryReplaced)]
        assert len(done) == 1, done
        assert done[0].reason == "compaction"
        assert (done[0].before_tokens, done[0].after_tokens) == (100, 30)
        assert done[0].automatic is False

    asyncio.run(go())


def test_undo_emits_typed_rollback_event(tmp_path) -> None:
    """Undo reports its outcome as a HistoryReplaced, not prose."""
    session = make_session(
        [[ContentDelta(text="answer"), RoundComplete()]], tmp_path,
    )

    async def go():
        await session.execute(SendMessage(text="question"))
        await wait_idle(session)
        queue = session.subscribe()[1]
        await session.execute(Rollback())
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        applied = [e for e in events if isinstance(e, protocol.HistoryReplaced)]
        assert len(applied) == 1, applied
        assert applied[0].resend is False
        assert applied[0].scope == "last turn"
        assert applied[0].note == "no_tracking"

    asyncio.run(go())


def test_status_fills_sandbox_name_and_filter_list(tmp_path) -> None:
    from xarness.config import SandboxSettings

    session = make_session([], tmp_path)
    session.sandbox.name = "netless"
    session.sandbox.settings = SandboxSettings(filter_list=["secrets"])
    status = session.status()
    assert status.sandbox_name == "netless"
    assert status.filter_list == ["secrets"]


def test_summary_reports_a_pending_ask_as_ask(tmp_path) -> None:
    """A pending ask blocks the turn and wants the user's answer, so the
    /sessions row reads "ask" (not "waiting") in every activity state."""
    async def go():
        session = make_session([], tmp_path)
        _, queue = session.subscribe()
        session._pending_ask = protocol.PendingAsk(ask_id="ask-1", questions=["q?"])

        task = mark_running(session)
        assert session.summary().state == "ask"
        session._compacting = True
        assert session.summary().state == "ask"
        task.cancel()
        session._compacting = False
        assert session.summary().state == "ask"
        session.unsubscribe(queue)

    asyncio.run(go())


def test_summary_reports_a_running_tool_as_waiting(tmp_path) -> None:
    """A tool in flight (a subagent can run for minutes) is not generation:
    the /sessions row reads waiting, then running again when it returns."""

    async def go():
        session = make_session([], tmp_path)
        task = mark_running(session)
        assert session.summary().state == "running"
        session._executing_tool = "subagent"
        assert session.summary().state == "waiting"
        session._executing_tool = None
        assert session.summary().state == "running"
        task.cancel()

    asyncio.run(go())


def test_snapshot_carries_hidden_epochs_as_human_history(tmp_path) -> None:
    """Compaction hides old messages from the model, but the snapshot still
    carries them so a resumed/attached client can show the full transcript."""
    session = make_session([], tmp_path)
    old = session.conversation.add(user_message("before"))
    summary = session.conversation.add(user_message(
        f"{SUMMARY_PREFIX}\n\nthe handoff", is_summary=True,
    ))

    snap = session.snapshot()
    assert [m.id for m in snap.messages] == [old.id, summary.id]


def test_windowed_snapshot_is_the_newest_whole_turns(tmp_path) -> None:
    """Attaching ships only the tail, turn-aligned; an untrimmed snapshot
    (the TUI's) still carries every message."""
    session = make_session([], tmp_path)
    for index in range(60):
        session.conversation.add(user_message(f"q{index}"))
        session.conversation.add(AssistantMessage(content=f"a{index}"))
    full = list(session.conversation.messages)

    snap = session.snapshot(window="[-10:]")
    assert snap.message_total == 120 and snap.has_more
    assert 0 < snap.message_offset < 120 and snap.message_offset % 2 == 0
    assert snap.messages == full[snap.message_offset:] and len(snap.messages) >= 10

    # No limit: the whole history, with the window fields saying so.
    for whole in (session.snapshot(), session.subscribe()[0]):
        assert whole.messages == full and not whole.has_more
        assert (whole.message_offset, whole.message_total) == (0, 120)


def test_message_pages_walk_the_history_once_in_whole_turns(tmp_path) -> None:
    """Paging back from the snapshot covers every message exactly once, and
    never cuts a turn in two."""
    session = make_session([], tmp_path)
    for index in range(25):
        session.conversation.add(user_message(f"q{index}"))
        session.conversation.add(AssistantMessage(content=f"a{index}"))
    full = list(session.conversation.messages)

    page, collected = session.message_page(window="[-8:]"), []
    while True:
        assert page.start % 2 == 0, "a page must end on a turn boundary"
        collected = [*page.messages, *collected]
        if not page.has_more:
            break
        page = session.message_page(window=f"[{max(0, page.start - 8)}:{page.start}]")
    assert collected == full and page.start == 0


def test_slim_pages_leave_the_bodies_for_the_body_endpoint(tmp_path) -> None:
    """A history page carries shapes and headers only; one round's text comes
    from GET …/messages/{index} (headers alone with content=false)."""
    session = make_session([], tmp_path)
    session.conversation.add(user_message("q0"))
    call = ToolCall(call_id="c1", name="bash", arguments_json='{"command": "ls"}')
    session.conversation.add(AssistantMessage(
        content="answer", reasoning="thought " * 50, tool_calls=[call],
    ))
    session.conversation.add(ToolMessage(tool_call_id="c1", content="output"))
    full = list(session.conversation.messages)

    page = session.message_page(window="[-10:]", slim=True)
    assert page.slim and len(page.messages) == 3
    assert page.messages[0] == full[0]
    assistant, result = page.messages[1], page.messages[2]
    assert assistant.content == "answer"
    assert assistant.reasoning == "thought " * 20  # SLIM_REASONING chars
    assert assistant.tool_calls[0].arguments_json == ""
    assert result.content == ""

    headers = session.message_body(1, content=False)
    assert [m.role for m in headers.messages] == ["assistant", "tool"]
    assert headers.messages[0].tool_calls[0].arguments_json == '{"command": "ls"}'
    assert headers.messages[1].content == ""

    body = session.message_body(1)
    assert body.messages[-1].content == "output"
    with pytest.raises(SessionError):
        session.message_body(99)


def test_attach_snapshot_takes_the_same_slim_mode_as_a_page(tmp_path) -> None:
    """The first frame of an attach can be slim: headers and prose, with
    tool bodies and call arguments left for the hydration requests."""
    session = make_session([], tmp_path)
    session.conversation.add(user_message("q0"))
    call = ToolCall(call_id="c1", name="bash", arguments_json='{"command": "ls"}')
    session.conversation.add(AssistantMessage(content="answer", tool_calls=[call]))
    session.conversation.add(ToolMessage(tool_call_id="c1", content="output"))

    snapshot, _queue = session.subscribe(window="[-10:]", slim=True)
    assert snapshot.slim is True
    assert [m.role for m in snapshot.messages] == ["user", "assistant", "tool"]
    assert snapshot.messages[1].content == "answer"
    assert snapshot.messages[1].tool_calls[0].arguments_json == ""
    assert snapshot.messages[2].content == ""

    # The default attach is still complete: the TUI takes the whole history
    # and renders it without a second round trip.
    full = session.snapshot()
    assert full.slim is False
    assert full.messages[2].content == "output"
    assert full.messages[1].tool_calls[0].arguments_json == '{"command": "ls"}'


def test_history_page_keeps_a_tool_call_with_its_result(tmp_path) -> None:
    """Cutting off the tool result would render the call as interrupted, so
    the page takes the whole turn."""
    session = make_session([], tmp_path)
    session.conversation.add(user_message("q0"))
    session.conversation.add(AssistantMessage(content="a0"))
    call = ToolCall(call_id="c1", name="read_file", arguments_json="{}")
    session.conversation.add(user_message("q1"))
    session.conversation.add(AssistantMessage(content="", tool_calls=[call]))
    session.conversation.add(ToolMessage(tool_call_id="c1", content="ok"))
    messages = list(session.conversation.messages)

    page = session.message_page(window="[-2:]")  # two from the end would split it
    assert (page.start, page.messages) == (2, messages[2:])


def test_stream_events_are_protocol_models() -> None:
    """Internal stream events cross the wire as-is: they are the protocol
    models themselves, with the ``type`` discriminator baked in."""
    cases = [
        (ProcessingStarted(), "processing_started"),
        (ReasoningDelta(text="x"), "reasoning_delta"),
        (ContentDelta(text="y"), "content_delta"),
        (StreamError(message="bad"), "stream_error"),
        (RoundComplete(usage=Usage(input_tokens=1, output_tokens=2), has_tool_calls=True), "round_complete"),
    ]
    for internal, expected_type in cases:
        assert internal.model_dump()["type"] == expected_type


def test_back_to_back_sends_one_turn_one_steering(tmp_path) -> None:
    """Two send_message commands queued back-to-back must produce exactly
    ONE turn with the second message steered into it — the turn state is
    set synchronously in _start_turn, before create_task, so the inbox's
    next dispatch cannot still see "idle" and start a second turn."""
    release = asyncio.Event()

    @tool("noop", "does nothing", _NoArgs)
    async def noop(args) -> ToolResult:
        await release.wait()
        return ToolResult()

    registry = ToolRegistry()
    registry.register(noop)
    session = make_session(
        [
            [
                ToolCallStarted(call_id="c1", name="noop"),
                ToolCallArgumentsDone(call_id="c1", name="noop", arguments_json="{}"),
                RoundComplete(has_tool_calls=True),
            ],
            [ContentDelta(text="second round"), RoundComplete()],
        ],
        tmp_path, tools=registry,
    )

    async def go():
        snapshot, queue = session.subscribe()
        # Back-to-back: no awaiting of turn progress between the two sends.
        await session.execute(SendMessage(text="one"))
        await session.execute(SendMessage(text="two"))
        assert session._queued, (
            "back-to-back send must steer, not start a second turn"
        )
        release.set()
        await wait_idle(session)
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        finished = [e for e in events if type(e).__name__ == "TurnEnded"]
        assert len(finished) == 1, f"expected one turn, got {len(finished)}"
        # The steered text was injected inside the SAME turn (the second LLM
        # round saw it), not replayed as a fresh turn afterwards.
        assert session._client.received_wire[1] and any(
            m.get("role") == "user" and m.get("content") == "two"
            for m in session._client.received_wire[1]
        )

    asyncio.run(go())


def test_partial_round_is_none_when_idle(tmp_path) -> None:
    session = make_session([[ContentDelta(text="x"), RoundComplete()]], tmp_path)

    async def go():
        session.subscribe()
        await session.execute(SendMessage(text="hi"))
        await wait_idle(session)
        assert session.snapshot().partial_round is None

    asyncio.run(go())


def test_only_round_content_updates_partial_round(tmp_path) -> None:
    """Status/notice events are snapshot state, not replay log."""
    session = make_session([], tmp_path)

    async def go():
        mark_running(session)
        session._emit(protocol.Notice(text="stale notice"))
        session._emit(protocol.StatusChanged(status=session.status()))
        assert session.snapshot().partial_round is None
        # The round runner fills the model-stream half (status/notice
        # events never create a partial round).
        session.partial_round.content = "live"
        partial = session.snapshot().partial_round
        assert partial is not None and partial.content == "live"

    asyncio.run(go())


def test_partial_round_tracks_a_running_tool_call(tmp_path) -> None:
    session = make_session([], tmp_path)

    async def go():
        mark_running(session)
        # What _stream_round accumulates: answer text, then the call it
        # started with its parsed arguments.
        partial = session.partial_round
        partial.content = "answer"
        partial.tool_calls.append(protocol.PartialToolCall(
            call_id="c1", name="read_file", arguments='{"path": "a"}',
        ))
        session._emit(protocol.ToolOutputDelta(call_id="c1", text="body"))
        partial.content = ""  # what _stream_round does on RoundComplete
        session._emit(protocol.RoundComplete())
        partial = session.snapshot().partial_round
        assert partial is not None
        assert partial.content == ""  # persisted in the assistant message
        assert len(partial.tool_calls) == 1
        call = partial.tool_calls[0]
        assert call.arguments == '{"path": "a"}'
        assert call.content == "body"
        assert call.status is None  # still running
        session._emit(protocol.ToolFinished(
            call_id="c1", status="ok", content="full body",
        ))
        call = session.snapshot().partial_round.tool_calls[0]
        assert call.status == "ok"

    asyncio.run(go())


def test_a_running_tool_stays_in_the_live_round(tmp_path) -> None:
    """While a tool executes its call has no result yet, so an attaching
    client must get it from the snapshot's live round — not from the
    history, where the bare call reads as interrupted and the round's
    stamp as a finished turn."""
    registry, started, release = _blocking_edit(tmp_path)
    session = make_session([_tool_round()], tmp_path, tools=registry)

    async def go():
        session.subscribe()
        await session.execute(SendMessage(text="go"))
        await asyncio.wait_for(started.wait(), 5)
        snap = session.snapshot()
        assert snap.status.state == "running"
        assert snap.partial_round is not None
        assert [(c.call_id, c.status) for c in snap.partial_round.tool_calls] \
            == [("c1", None)]
        # Nothing to pair the call with yet: the history alone would lie.
        assert [m.role for m in snap.messages] == ["user", "assistant"]
        release.set()
        await wait_idle(session)

    asyncio.run(go())


def test_partial_round_never_truncates_a_long_answer(tmp_path) -> None:
    session = make_session([], tmp_path)

    async def go():
        mark_running(session)
        # The live content is the round runner's, verbatim: the snapshot
        # must never truncate or re-copy it.
        session.partial_round.content = "x" * 600
        partial = session.snapshot().partial_round
        assert partial is not None and len(partial.content) == 600

    asyncio.run(go())


def test_filter_add_rejects_missing_path(tmp_path) -> None:
    """/filter add on a path that does not exist is rejected — nothing is
    created in the worktree and the filter list is unchanged."""
    from xarness.config import SandboxSettings

    session = make_session([[ContentDelta(text="x"), RoundComplete()]], tmp_path)
    session.sandbox.settings = SandboxSettings()
    (tmp_path / "real.txt").write_text("x")

    async def go():
        with pytest.raises(SessionError, match="no such path"):
            await session.execute(protocol.Filter(path="ghost.txt", add=True))
        assert not (tmp_path / "ghost.txt").exists()
        await session.execute(protocol.Filter(path="real.txt", add=True))
        assert session.sandbox.settings.filter_list == ["real.txt"]

    asyncio.run(go())


def test_ref_add_remove_is_per_workspace(tmp_path, monkeypatch) -> None:
    """Refs added via the command bind read-only under /tmp/refs/<alias> and
    are persisted per workspace; a missing path is rejected and nothing is
    created in the worktree."""
    from xarness import session_store, workspace
    from xarness.config import REFS_ANCHOR, SandboxSettings

    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path / "sessions")
    session = make_session([[ContentDelta(text="x"), RoundComplete()]], tmp_path)
    session.sandbox.settings = SandboxSettings()
    (tmp_path / "spec.md").write_text("x")
    expected = str((tmp_path / "spec.md").resolve())

    async def go():
        with pytest.raises(SessionError, match="no such path"):
            await session.execute(protocol.Ref(path="ghost.md", add=True))
        # Never mkdir a reference (and never create the container anchor).
        assert not (tmp_path / "ghost.md").exists()
        assert not (tmp_path / ".refs").exists()

        await session.execute(protocol.Ref(path="spec.md", add=True))
        assert session.workspace_refs == [expected]
        refs = session.status().refs
        assert [(r.path, r.mount) for r in refs] == [
            (expected, f"{REFS_ANCHOR}/spec.md")]
        assert workspace.load_workspace_refs(tmp_path) == [expected]

        await session.execute(protocol.Ref(path="spec.md", add=False))
        assert session.workspace_refs == []
        assert workspace.load_workspace_refs(tmp_path) == []

    asyncio.run(go())


def test_undo_history_replaced_renders_once_on_attach(tmp_path) -> None:
    """After an /undo (a rewrite) an attaching client's snapshot has no
    live round: the history is rendered exactly once."""
    session = make_session([[ContentDelta(text="x"), RoundComplete()]], tmp_path)

    async def go():
        session.subscribe()
        await session.execute(SendMessage(text="hi"))
        await wait_idle(session)
        await session.execute(Rollback())
        assert session.snapshot().partial_round is None
        # A client attaching now sees the rewritten history exactly once.
        snapshot, _queue = session.subscribe()
        assert snapshot.partial_round is None
        # The undo dropped the turn: the rewritten history (which the client
        # renders exactly once) holds no leftover messages.
        assert snapshot.messages == []

    asyncio.run(go())


def test_cancel_saves_partial_cot_reasoning_only(tmp_path) -> None:
    """Cancelling mid-thinking (no answer text yet, no tool call) now saves
    the partial chain of thought instead of discarding it — and it is
    persisted, so a reload/replay shows it too."""
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingClient:
        async def stream(self, wire_messages, tools=None):
            yield ProcessingStarted()
            yield ReasoningDelta(text="thinking half ")
            yield ReasoningDelta(text="a thought")
            started.set()
            await release.wait()
            yield ContentDelta(text="never arrives")

    session = Session(
        name="test", workspace=tmp_path, profile=PROFILE, profile_name="test",
        api_key="key", client=BlockingClient(),
    )

    async def go():
        session.subscribe()
        task = asyncio.create_task(session.execute(SendMessage(text="hi")))
        await asyncio.wait_for(started.wait(), timeout=5)
        await session.execute(Cancel())
        release.set()
        await task
        await wait_idle(session)

    asyncio.run(go())

    assistant = [m for m in session.conversation.messages if m.role == "assistant"]
    assert len(assistant) == 1
    assert assistant[0].reasoning == "thinking half a thought"
    assert assistant[0].content == ""
    # Persisted: a fresh load sees the partial CoT.
    loaded = session_store.load_state("test").conversation
    loaded_assistant = [m for m in loaded.messages if m.role == "assistant"]
    assert loaded_assistant and loaded_assistant[0].reasoning == "thinking half a thought"


def test_forced_steer_saves_partial_cot(tmp_path) -> None:
    """Flush-queued (forced steer) cancels the running turn: its partial CoT
    is saved the same way, and the queued message starts as a new turn."""
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingClient:
        async def stream(self, wire_messages, tools=None):
            yield ProcessingStarted()
            yield ReasoningDelta(text="partial thoughts")
            started.set()
            await release.wait()
            yield ContentDelta(text="never arrives")

    session = Session(
        name="test", workspace=tmp_path, profile=PROFILE, profile_name="test",
        api_key="key", client=BlockingClient(),
    )

    async def go():
        session.subscribe()
        await session.execute(SendMessage(text="first"))
        await asyncio.wait_for(started.wait(), timeout=5)
        await session.execute(SendMessage(text="second"))  # queued
        await session.execute(protocol.FlushQueued())
        release.set()
        await wait_idle(session)

    asyncio.run(go())

    assistant = [m for m in session.conversation.messages if m.role == "assistant"]
    assert assistant and assistant[0].reasoning == "partial thoughts"
    user_texts = [m.content for m in session.conversation.messages if m.role == "user"]
    assert "second" in user_texts


def test_stop_cancels_a_running_compaction(tmp_path) -> None:
    """session.stop() cancels a detached compaction task too: closing a
    session mid-compact must not leave a task that starts a turn on a
    stopped session afterwards."""
    gate = asyncio.Event()
    session = make_session([], tmp_path)

    async def fake_compact():
        await gate.wait()
        return None  # nothing to compact

    session.compact = fake_compact

    async def go():
        session.subscribe()
        await asyncio.wait_for(session.execute(Compact()), timeout=2)
        assert session.state == "compacting"
        await session.stop()
        assert session._compact_task is None

    asyncio.run(go())
    # The compact task was cancelled (stop() cleared the reference); the
    # session state never came back to idle via the cancelled task's except
    # path — stop() owns the teardown.
    assert session.state == "compacting" or session.state == "idle"


def test_no_save_disables_persistence(tmp_path) -> None:
    """no_session: _persist never writes a session file."""
    session = Session(
        name="nosave", workspace=tmp_path, profile=PROFILE,
        profile_name="test", api_key="key", client=FakeClient([]),
        no_save=True,
    )
    session.conversation.add(user_message("q"))
    session._persist()
    assert not session_store.session_dir("nosave").exists()


def test_stamp_turn_end_records_the_turn_stats(tmp_path) -> None:
    """The host writes the numbers it measured onto the turn's last answer:
    total | busy | idle, busy being the model's generation time."""
    session = make_session([], tmp_path)
    opening = session.conversation.add(user_message("hi"))
    session.conversation.begin_turn(opening, before_tree=None)
    session.conversation.add(AssistantMessage(content="thinking about it"))
    session.conversation.add(ToolMessage(tool_call_id="c1"))
    last = session.conversation.add(AssistantMessage(content="done"))
    stats = session.conversation.stamp_turn_end(total=8.0, busy=7.7)
    assert stats is not None
    assert (stats.total, stats.busy, round(stats.idle, 1)) == (8.0, 7.7, 0.3)
    # Stored where a reload (and any other client) reads it back from.
    assert last.timing.stats == stats


def test_stamp_turn_end_sums_round_usage(tmp_path) -> None:
    """The stamped spend is the turn's whole: every round it sent."""
    session = make_session([], tmp_path)
    session.conversation.add(
        user_message("hi", timing=Timing(send="2025-01-01T00:00:00+00:00")))
    session.conversation.add(AssistantMessage(
        content="a", usage=Usage(input_tokens=1_234, output_tokens=56),
    ))
    session.conversation.add(AssistantMessage(
        content="b", usage=Usage(input_tokens=2_000, output_tokens=400),
    ))
    stats = session.conversation.stamp_turn_end(total=5.0, busy=4.0)
    assert stats is not None and stats.usage is not None
    assert (stats.usage.input_tokens, stats.usage.output_tokens) == (3_234, 456)
    # Local estimates are flagged, never passed off as exact counts.
    estimated = Usage(input_tokens=10, output_tokens=2, approximate=True)
    assert (estimated + Usage(input_tokens=1, output_tokens=3)).approximate


def test_stamp_turn_end_needs_an_answer(tmp_path) -> None:
    """A turn that never produced an answer leaves no stamp — and so no
    footer."""
    session = make_session([], tmp_path)
    session.conversation.add(user_message("hi"))
    assert session.conversation.stamp_turn_end(1.0, 1.0) is None


def test_turn_ended_carries_the_stamped_breakdown(tmp_path) -> None:
    """TurnEnded carries the numbers just stamped on the turn's last answer,
    and they are on disk: a live footer and a reloaded one agree."""
    session = make_session(
        [[ReasoningDelta(text="hmm"), ContentDelta(text="Hello"),
          RoundComplete(usage=Usage(input_tokens=10, output_tokens=2))]],
        tmp_path,
    )

    async def go():
        _, queue = session.subscribe()
        await session.execute(SendMessage(text="hi"))
        await wait_idle(session)
        finished = None
        while not queue.empty():
            event = queue.get_nowait()
            if isinstance(event, protocol.TurnEnded):
                finished = event
        return finished

    finished = asyncio.run(go())
    last_assistant = session.conversation.messages[-1]
    assert last_assistant.role == "assistant"
    stats = last_assistant.timing.stats
    assert finished is not None and stats is not None
    assert finished.stats == stats
    # The split is measured, not reconstructed: it closes on the total.
    assert stats.busy is not None and stats.idle is not None
    assert stats.busy + stats.idle == stats.total
    assert stats.usage is not None and stats.usage.input_tokens == 10

    # Persisted, so a reload renders the same footer instead of deriving one.
    reloaded = session_store.load_state("test")
    assert reloaded.conversation.messages[-1].timing.stats == stats


def test_busy_is_generation_only_so_tool_time_is_idle(tmp_path) -> None:
    """The split is generating vs waiting: the model's streaming rounds are
    busy, but the time a tool runs is the client waiting, so it is idle."""

    @tool("edit_file", "edit a file", _NoArgs)
    async def edit(args) -> ToolResult:
        await asyncio.sleep(0.1)
        return ToolResult(content="ok")

    registry = ToolRegistry()
    registry.register(edit)
    session = make_session(
        [_tool_round(), _round("done")], tmp_path, tools=registry,
    )

    async def go():
        session.subscribe()
        await session.execute(SendMessage(text="hi"))
        await wait_idle(session)

    asyncio.run(go())
    stats = session.conversation.messages[-1].timing.stats
    assert stats is not None
    assert stats.busy is not None and stats.idle is not None
    assert stats.idle >= 0.08  # the tool's sleep is the client waiting
    assert stats.busy < stats.idle


def test_cancelled_turn_is_stamped_without_a_split(tmp_path) -> None:
    """A turn cancelled while its tool ran ended too: it carries how long it
    actually ran, but no busy/idle split (nothing was measured to an end)."""
    started = asyncio.Event()
    release = asyncio.Event()

    @tool("edit_file", "edit a file", _NoArgs)
    async def edit(args) -> ToolResult:
        started.set()
        await release.wait()
        return ToolResult(content="ok")

    registry = ToolRegistry()
    registry.register(edit)
    session = make_session([_tool_round()], tmp_path, tools=registry)

    async def go():
        session.subscribe()
        await session.execute(SendMessage(text="hi"))
        await asyncio.wait_for(started.wait(), timeout=5)
        await session.execute(Cancel())
        await wait_idle(session)

    asyncio.run(go())
    assistants = [
        m for m in session.conversation.messages if m.role == "assistant"
    ]
    assert assistants  # the round that reached the tool exists
    stats = assistants[-1].timing.stats
    assert stats is not None and stats.total >= 0
    assert stats.busy is None and stats.idle is None


def test_a_steered_turn_is_one_turn_for_the_stamp(tmp_path) -> None:
    """A steering message injected mid-turn opens no turn of its own, so the
    stamp covers the whole turn: pre-steer rounds included."""
    session = make_session([], tmp_path)
    conversation = session.conversation
    opening = conversation.add(
        user_message("hi", timing=Timing(send="2025-01-01T00:00:00+00:00")))
    conversation.begin_turn(opening, before_tree=None)
    conversation.add(AssistantMessage(
        content="a", usage=Usage(input_tokens=10, output_tokens=1),
    ))
    conversation.add(user_message("also this"))
    last = conversation.add(AssistantMessage(
        content="b", usage=Usage(input_tokens=20, output_tokens=2),
    ))
    stats = conversation.stamp_turn_end(total=4.0, busy=3.0)
    assert stats is not None and stats.usage is not None
    assert (stats.usage.input_tokens, stats.usage.output_tokens) == (30, 3)
    assert stats.total == 4.0
    assert last.timing.stats == stats


# ----------------------------------------------------------------------
# mid-turn change attribution (/diff source tags)


async def _tracked_session(tmp_path, script, registry=None, files=None):
    """A session with real workspace tracking.

    The tracked workspace is a subdirectory so the (test-local) session
    files, which live directly under ``tmp_path``, are not themselves
    tracked as drift. ``files`` are written into the workspace before the
    baseline snapshot, so they start out tracked rather than pending."""
    ws = tmp_path / "ws"
    ws.mkdir()
    for name, content in (files or {}).items():
        (ws / name).write_text(content)
    state = await open_workspace(ws, persist=False)
    assert state is not None
    session = Session(
        name="test", workspace=ws, profile=PROFILE, profile_name="test",
        api_key="key", client=FakeClient(script),
    )
    if registry is not None:
        session.tool_registry = registry
        session.tool_registry = registry
    session.workspace_state = state
    return session, state


def _blocking_edit(tmp_path, name="a.txt"):
    """An edit tool that writes ``name`` and then blocks, so tests can act
    while the turn is still generating."""
    started = asyncio.Event()
    release = asyncio.Event()

    @tool("edit_file", "edit a file", _NoArgs)
    async def edit(args) -> ToolResult:
        (tmp_path / name).write_text("agent\n")
        started.set()
        await release.wait()
        return ToolResult(content="ok")

    registry = ToolRegistry()
    registry.register(edit)
    return registry, started, release


def _tool_round(call_id="c1", path="a.txt"):
    return [
        ToolCallStarted(call_id=call_id, name="edit_file"),
        ToolCallArgumentsDone(
            call_id=call_id, name="edit_file",
            arguments_json=json.dumps({"path": path}),
        ),
        RoundComplete(has_tool_calls=True),
    ]


@pytest.mark.parametrize(
    "path", ["a.txt", "/workspace/a.txt"], ids=["relative", "absolute"],
)
def test_an_explicit_write_records_its_own_provenance(
    tmp_path, monkeypatch, path,
) -> None:
    """edit_file's event carries the before/after blob shas, so /diff reports
    that call's delta instead of the whole pending file.

    An absolute path the tool accepts names one file on both sides, so it is
    attributed the same way — the container view's /workspace included."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    async def go():
        ws = tmp_path / "ws"

        @tool("edit_file", "edit a file", _NoArgs)
        async def edit(args) -> ToolResult:
            (ws / "a.txt").write_text("after\n")
            return ToolResult(content="ok")

        registry = ToolRegistry()
        registry.register(edit)
        session, state = await _tracked_session(
            tmp_path, [_tool_round(path=path)], registry,
            files={"a.txt": "before\n"},
        )
        if path.startswith("/"):
            # In a container the tool is handed /workspace/a.txt; without a
            # config that names the same file, the write is unattributable.
            with mock.patch("shutil.which", return_value="/usr/bin/bwrap"):
                session.sandbox = SandboxRuntime(
                    config=SandboxConfig(workspace=ws))
        session.subscribe()
        await session.execute(SendMessage(text="hi"))
        await wait_idle(session)

        event = state.events[0]
        assert event.paths == ("a.txt",)
        written, before, after = event.writes[0]
        assert written == "a.txt"
        assert before is not None and after is not None
        assert await gitwork.read_blob(state, before) == b"before\n"
        assert await gitwork.read_blob(state, after) == b"after\n"

        info = await session.diff_info(None)
        assert [
            (row.path, row.kind, row.additions, row.deletions)
            for row in info.files
        ] == [("a.txt", "mine", 1, 1)]
        assert info.text == ""  # the summary poll stays cheap
        assert await session.diff_info("a.txt") is not None

    asyncio.run(go())


def test_mutating_tool_and_turn_end_emit_diff_changed(tmp_path, monkeypatch) -> None:
    """The host pushes the pending-changes summary after a mutating tool and
    at turn end, so clients stop polling /diff on every tool result."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    async def go():
        @tool("edit_file", "edit a file", _NoArgs)
        async def edit(args) -> ToolResult:
            (tmp_path / "ws" / "a.txt").write_text("agent\n")
            return ToolResult(content="ok")

        registry = ToolRegistry()
        registry.register(edit)
        session, _info = await _tracked_session(tmp_path, [
            [
                ToolCallStarted(call_id="c1", name="edit_file"),
                ToolCallArgumentsDone(
                    call_id="c1", name="edit_file", arguments_json="{}",
                ),
                RoundComplete(has_tool_calls=True),
            ],
            [ContentDelta(text="done"), RoundComplete()],
        ], registry)
        queue = session.subscribe()[1]
        await session.execute(SendMessage(text="hi"))
        await wait_idle(session)
        # The turn-end DiffChanged is emitted after the state flips to idle
        # (the emit itself snapshots the workspace), so give it a moment.
        changed: list = []
        for _ in range(100):
            while not queue.empty():
                event = queue.get_nowait()
                if isinstance(event, protocol.DiffChanged):
                    changed.append(event)
            if changed:
                break
            await asyncio.sleep(0.02)
        assert changed, "expected at least one DiffChanged"
        paths = {f.path for event in changed for f in event.summary.files}
        assert paths == {"a.txt"}, paths

    asyncio.run(go())


def test_diff_info_path_snapshots_workspace_once(tmp_path, monkeypatch) -> None:
    """`/diff <path>` needs the stat rows and the text; both must share a
    single workspace snapshot instead of running two full snapshots."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    async def go():
        session, state = await _tracked_session(tmp_path, [])
        (tmp_path / "ws" / "a.txt").write_text("changed\n")
        calls = 0
        real = state.observe

        async def counting():
            nonlocal calls
            calls += 1
            return await real()

        state.observe = counting
        payload = await session.diff_info("a.txt")
        assert [f.path for f in payload.files] == ["a.txt"]
        assert payload.text
        assert calls == 1, calls

    asyncio.run(go())


def test_accept_rebaselines_mid_turn_edits(tmp_path, monkeypatch) -> None:
    """A mid-turn /accept absorbs the in-flight write into the new baseline."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    async def go():
        registry, started, release = _blocking_edit(tmp_path / "ws")
        session, _info = await _tracked_session(tmp_path, [
            _tool_round(),
            [ContentDelta(text="done"), RoundComplete()],
        ], registry)
        session.subscribe()
        await session.execute(SendMessage(text="hi"))
        await asyncio.wait_for(started.wait(), 5)

        stat = await session.diff_info(None)
        # The blocked tool has not returned yet, so no event exists for its
        # write: an in-flight edit is indistinguishable from drift.
        assert {f.path: f.kind for f in stat.files} == {"a.txt": "drift"}

        await session.execute(Accept())
        assert session.state == "running"

        # Accept absorbs the in-flight a.txt write into the new baseline.
        # The raw b.txt write made after it is drift.
        (tmp_path / "ws" / "b.txt").write_text("later edit\n")
        stat = await session.diff_info(None)
        by_path = {f.path: f.kind for f in stat.files}
        assert by_path == {"b.txt": "drift"}, by_path

        release.set()
        await wait_idle(session)

        # The tool's event lands after the accept, but its a.txt write is
        # already in the baseline, so it adds nothing pending.
        stat = await session.diff_info(None)
        by_path = {f.path: f.kind for f in stat.files}
        assert by_path == {"b.txt": "drift"}, by_path
        # The turn spans the accept, so it is no longer revertible.
        assert session.turn_checkpoints() == []

    asyncio.run(go())


def test_event_recording_is_atomic_with_accept(tmp_path, monkeypatch) -> None:
    """A tool's event is appended under the workspace lock, so /accept either
    sweeps it or runs entirely before it — never between the after-snapshot
    and the append, which left a pre-accept change pending forever."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    async def go():
        started, release = asyncio.Event(), asyncio.Event()

        @tool("run_bash", "run a command", _NoArgs)
        async def bash(args) -> ToolResult:
            (tmp_path / "ws" / "bash.txt").write_text("written\n")
            started.set()
            await release.wait()
            return ToolResult(content="done")

        registry = ToolRegistry()
        registry.register(bash)
        session, state = await _tracked_session(tmp_path, [
            [
                ToolCallStarted(call_id="c1", name="run_bash"),
                ToolCallArgumentsDone(
                    call_id="c1", name="run_bash", arguments_json="{}",
                ),
                RoundComplete(has_tool_calls=True),
            ],
            [ContentDelta(text="done"), RoundComplete()],
        ], registry)

        # Gate the after-snapshot: the test then knows the recorder holds the
        # workspace lock, between its snapshot and its event append.
        inside = asyncio.Event()
        resume = asyncio.Event()
        real_snapshot = state.snapshot

        async def gated():
            tree = await real_snapshot()
            inside.set()
            await resume.wait()
            return tree

        state.snapshot = gated
        await session.execute(SendMessage(text="go"))
        await asyncio.wait_for(started.wait(), 5)
        release.set()
        await asyncio.wait_for(inside.wait(), 5)

        accept = asyncio.create_task(state.accept())
        await asyncio.sleep(0.05)
        assert not accept.done(), "accept ran inside the event-recording step"
        resume.set()
        await accept
        await wait_idle(session)

        # The event landed before accept could run, so accept swept it and
        # folded the file into the new baseline (no pending row, no drift).
        assert state.events == []
        assert await gitwork.tree_entry(
            state, state.baseline_tree, "bash.txt",
        ) is not None
        stat = await session.diff_info(None)
        assert stat.files == []

    asyncio.run(go())


def test_a_failing_bash_command_owns_what_it_wrote(
    tmp_path, monkeypatch,
) -> None:
    """A command that writes a file and then fails is still credited with it."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    async def go():
        @tool("run_bash", "run a command", _NoArgs)
        async def bash(args) -> ToolResult:
            (tmp_path / "ws" / "new.txt").write_text("written\n")
            return ToolResult(status="error", content="boom\n[exit 1]")

        registry = ToolRegistry()
        registry.register(bash)
        session, state = await _tracked_session(tmp_path, [
            [
                ToolCallStarted(call_id="c1", name="run_bash"),
                ToolCallArgumentsDone(
                    call_id="c1", name="run_bash", arguments_json="{}",
                ),
                RoundComplete(has_tool_calls=True),
            ],
            [ContentDelta(text="done"), RoundComplete()],
        ], registry)
        await session.execute(SendMessage(text="go"))
        await wait_idle(session)

        stat = await session.diff_info(None)
        assert {f.path: f.kind for f in stat.files} == {"new.txt": "agent"}
        assert [(e.tool, e.changed) for e in state.events] == [
            ("run_bash", ("new.txt",)),
        ]

    asyncio.run(go())


def test_a_failed_explicit_write_is_not_credited(
    tmp_path, monkeypatch,
) -> None:
    """A failed ``edit_file`` wrote nothing through the tool: the path it
    named stays drift, so a dead call cannot claim the user's own change."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    async def go():
        @tool("edit_file", "edit a file", _NoArgs)
        async def edit(args) -> ToolResult:
            return ToolResult(status="error", content="path is not a file")

        registry = ToolRegistry()
        registry.register(edit)
        session, state = await _tracked_session(tmp_path, [
            _tool_round(),
            [ContentDelta(text="done"), RoundComplete()],
        ], registry)
        # The user's own write while the tool runs is drift, unclaimed.
        (tmp_path / "ws" / "a.txt").write_text("mine\n")
        await session.execute(SendMessage(text="go"))
        await wait_idle(session)

        stat = await session.diff_info(None)
        assert {f.path: f.kind for f in stat.files} == {"a.txt": "drift"}
        assert state.events == []

    asyncio.run(go())


def test_reject_restores_the_baseline_mid_turn(tmp_path, monkeypatch) -> None:
    """A mid-turn /reject restores the baseline; a later raw write is drift
    while the turn's tool edits were already reverted."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    async def go():
        registry, started, release = _blocking_edit(tmp_path / "ws")
        session, _info = await _tracked_session(tmp_path, [
            _tool_round(),
            [ContentDelta(text="done"), RoundComplete()],
        ], registry)
        session.subscribe()
        await session.execute(SendMessage(text="hi"))
        await asyncio.wait_for(started.wait(), 5)

        await session.execute(Reject())
        assert not (tmp_path / "ws" / "a.txt").exists()

        (tmp_path / "ws" / "b.txt").write_text("later edit\n")
        stat = await session.diff_info(None)
        by_path = {f.path: f.kind for f in stat.files}
        assert by_path == {"b.txt": "drift"}, by_path

        release.set()
        await wait_idle(session)
        # The turn spans the reject, so it is no longer revertible.
        assert session.turn_checkpoints() == []

    asyncio.run(go())


def test_compaction_keeps_open_turn_attribution_and_undo(tmp_path, monkeypatch) -> None:
    """Compaction moves the open turn's message into compact_snapshot. Its
    before tree must survive so the turn's events stay attributable, and
    finish_turn must still give it an after tree (so /undo can reverse it)."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    async def go():
        registry, started, release = _blocking_edit(tmp_path / "ws")
        session, _info = await _tracked_session(tmp_path, [
            _tool_round(),
            [ContentDelta(text="done"), RoundComplete()],
        ], registry)
        session.subscribe()
        await session.execute(SendMessage(text="hi"))
        await asyncio.wait_for(started.wait(), 5)

        from unittest import mock
        from xarness import compaction as compaction_mod

        async def fake_summarize(_client, _wire_messages):
            return "handoff summary", None

        with mock.patch.object(compaction_mod, "summarize", fake_summarize):
            await session.compact()
        assert session._open_turn_before() is not None

        (tmp_path / "ws" / "b.txt").write_text("later edit\n")
        stat = await session.diff_info(None)
        by_path = {f.path: f.kind for f in stat.files}
        # The blocked tool has not returned, so its a.txt write is drift.
        assert by_path["b.txt"] == "drift"
        assert by_path["a.txt"] == "drift"

        release.set()
        await wait_idle(session)

        # The open turn survived compaction and now has a reversible diff;
        # its recorded tool event labels a.txt as this viewer's edit.
        turns = session.turn_checkpoints()
        assert len(turns) == 1 and turns[0].after_tree
        stat = await session.diff_info(None)
        by_path = {f.path: f.kind for f in stat.files}
        assert by_path["a.txt"] == "mine"
        assert by_path["b.txt"] == "drift"

    asyncio.run(go())


def test_checkpoint_failure_does_not_restamp_previous_turn(tmp_path, monkeypatch) -> None:
    """If a turn's before-snapshot fails, finish_turn must not write that
    turn's after tree onto the previous (already closed) turn."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    async def go():
        registry, started, release = _blocking_edit(tmp_path / "ws", "b.txt")
        session, _info = await _tracked_session(tmp_path, [
            [ContentDelta(text="first"), RoundComplete()],
            _tool_round(),
            [ContentDelta(text="done"), RoundComplete()],
        ], registry)
        session.subscribe()

        await session.execute(SendMessage(text="first"))
        await wait_idle(session)
        user_messages = [
            m for m in session.conversation.messages if m.role == "user"
        ]
        first_turn = session.conversation.turns[user_messages[0].id]
        first_after = first_turn.after_tree
        assert first_after

        async def fail_checkpoint():
            return None

        session._take_checkpoint = fail_checkpoint
        await session.execute(SendMessage(text="second"))
        await asyncio.wait_for(started.wait(), 5)

        # The failing checkpoint means this turn cannot be attributed yet,
        # but the previous, closed turn must not be touched.
        assert first_turn.after_tree == first_after

        release.set()
        await wait_idle(session)

        assert first_turn.after_tree == first_after
        user_messages = [
            m for m in session.conversation.messages if m.role == "user"
        ]
        second_turn = session.conversation.turns[user_messages[1].id]
        # The failed checkpoint left the second turn unattributed.
        assert second_turn.before_tree is None
        # The first turn made no file edits (the blocked edit belongs to the
        # second, unattributed turn), so nothing is reversible.
        assert session.turn_checkpoints() == []

    asyncio.run(go())


def test_compaction_undo_does_not_resend_an_empty_turn(tmp_path) -> None:
    """/retry right after a compaction pops the summary; it must NOT start a
    turn with an empty user message."""
    async def go():
        session = make_session([], tmp_path)
        conversation = session.conversation
        user = conversation.add(user_message("real question"))
        conversation.begin_turn(user, None)
        conversation.add(AssistantMessage(content="answer"))
        conversation.add(user_message("summary", is_summary=True))
        await session.execute(Rollback(resend=True))
        # The summary was popped, but no turn started (the state went idle
        # and no empty user message landed in the history).
        assert session.state == "idle"
        assert session._turn_task is None
        assert [m.content for m in conversation.visible_messages()] == [
            "real question", "answer",
        ]
        return session

    asyncio.run(go())


def test_send_rejects_image_refs_missing_from_the_store(tmp_path) -> None:
    """A referenced blob that was never uploaded is rejected up front — it
    must not degrade to a text token on the wire."""
    from xarness.images import ImageRef

    async def go():
        session = make_session([], tmp_path)
        ghost = ImageRef(
            id="f" * 64, name="ghost.png", mime="image/png", width=1, height=1,
        )
        with pytest.raises(SessionError, match="never uploaded"):
            await session.execute(SendMessage(text="look", images=[ghost]))
        # The known-good path still works.
        session.images.store(b"cat")
        real = ImageRef(
            id=session.images.store(b"cat"), name="cat.png",
            mime="image/png", width=1, height=1,
        )
        await session.execute(SendMessage(text="look", images=[real]))

    asyncio.run(go())


def test_set_mode_is_allowed_mid_turn(tmp_path) -> None:
    """Mode is a flag (no history/sandbox rewrite), so it flips mid-turn;
    the model/sandbox commands stay idle-only."""
    async def go():
        session = make_session([], tmp_path)  # no config path known
        mark_running(session)
        await session.execute(protocol.SetMode(mode="plan"))
        assert session.mode == "plan"
        with pytest.raises(SessionError, match="wait for"):
            await session.execute(protocol.SwitchModel(name="other-model"))

    asyncio.run(go())


def test_accept_emits_checkpoints_changed_not_history_replaced(tmp_path, monkeypatch) -> None:
    """/accept rebaselines the checkpoints: clients get a lightweight
    CheckpointsChanged, not a full HistoryReplaced snapshot."""
    session = make_session([_round("one")], tmp_path)
    state = _tracked_workspace(tmp_path)

    async def fake_accept():
        return "f" * 40

    async def fake_diff(viewer=None, current=None):
        return protocol.DiffInfo()

    state.accept = fake_accept
    state.diff = fake_diff
    session.workspace_state = state

    async def go():
        queue = session.subscribe()[1]
        await session.execute(protocol.Accept())
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        assert any(
            isinstance(e, protocol.CheckpointsChanged) and e.reason == "accept"
            for e in events
        ), [type(e).__name__ for e in events]
        assert not any(isinstance(e, protocol.HistoryReplaced) for e in events)

    asyncio.run(go())


def test_rollback_watermark_survives_inflight_write(tmp_path, monkeypatch) -> None:
    """A rollback that races the background write moves the watermark back,
    and the in-flight write's completion must not advance it past the cut —
    the next write rewrites the truncated rows so nothing is lost."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    session = make_session([], tmp_path)
    real = session_store.write_state
    started, release = asyncio.Event(), asyncio.Event()

    def slow(name, payload, meta):
        started.set()
        deadline = time.monotonic() + 5
        while not release.is_set() and time.monotonic() < deadline:
            time.sleep(0.005)
        return real(name, payload, meta)

    monkeypatch.setattr(session_store, "write_state", slow)

    async def go():
        session.conversation.add(user_message("hi"))
        session.conversation.add(AssistantMessage(content="hello"))
        session.conversation._dirty_from = 0
        session._persist()
        await asyncio.wait_for(started.wait(), 5)
        # The rollback lands while the write thread is running.
        session.conversation._dirty_from = 1  # plan.drop_from
        release.set()
        await session._flush_persist()
        assert session.conversation._dirty_from == 1, session.conversation._dirty_from

        # A follow-up write covers the cut onward; the reload keeps both
        # messages (the truncated row was rewritten, not skipped).
        session._persist()
        await session._flush_persist()
        stored = session_store.load_state("test")
        assert [m.role for m in stored.conversation.messages] == ["user", "assistant"]

    asyncio.run(go())


def test_no_message_is_mutated_after_first_persist(tmp_path, monkeypatch) -> None:
    """Tail-only persistence is sound because messages are immutable: once
    a message has been serialized for a write, nothing may ever change it
    (turn stamping happens before the first persist of the message)."""
    session = make_session([_round("one"), _round("two")], tmp_path)
    real = session_store.serialize_state
    seen: dict[str, Any] = {}

    def spy(stored, from_seq):
        for message in stored.conversation.messages:
            snapshot = message.model_copy(deep=True)
            if message.id in seen:
                assert seen[message.id] == snapshot, (
                    f"message {message.id} mutated after being persisted"
                )
            seen[message.id] = snapshot
        return real(stored, from_seq)

    monkeypatch.setattr(session_store, "serialize_state", spy)

    async def go():
        await session.execute(protocol.SendMessage(text="hi"))
        await wait_idle(session)
        # Accept rewrites checkpoints (not messages); a second turn then a
        # rollback exercises truncation too. (No tracking here, so Accept
        # would refuse; the rollback alone still exercises the tail.)
        if session.workspace_state is not None:
            await session.execute(protocol.Accept())
        await session.execute(protocol.SendMessage(text="again"))
        await wait_idle(session)
        await session.execute(protocol.Rollback())

    asyncio.run(go())
    assert seen, "expected at least one persist"


def test_load_starts_watermark_at_end_and_closes_crashed_turns(tmp_path) -> None:
    """A loaded session must not rewrite its whole history on the first
    persist (the watermark starts at the end), and turns left open by a
    crash are closed — including untracked ones (before_tree None), which
    the tree-based repair never saw."""
    async def go():
        session = make_session([], tmp_path)
        user = session.conversation.add(user_message("hi"))
        session.conversation.begin_turn(user, None)  # untracked: never closed
        stored = session_store.StoredSession(
            conversation=session.conversation,
            profile="test", workspace=str(tmp_path),
        )
        loaded = Session.load(
            "test", parts=_parts(tmp_path), state=stored,
        )
        # The watermark starts "everything persisted": the first write must
        # not rewrite a loaded history.
        assert loaded.conversation._dirty_from is None
        assert loaded.conversation.open_turn is None  # closed on load
        assert all(t.closed for t in loaded.conversation.turns.values())
        # Untracked: closing a crashed turn leaves its trees unset.
        assert all(t.after_tree is None
                   for t in loaded.conversation.turns.values())

    asyncio.run(go())


def test_view_state_is_saved_with_the_session(tmp_path) -> None:
    """The /diff scope is session state: it goes out in the status, is
    written to the session row, and comes back on load."""
    async def go():
        session = make_session([], tmp_path)
        assert session.status().view.diff_scope == "all"

        await session.execute(protocol.SetViewState(
            view=protocol.ViewState(diff_scope="mine")))
        assert session.view.diff_scope == "mine"
        assert session.status().view.diff_scope == "mine"

        await session._flush_persist()
        stored = session_store.load_state("test")
        assert stored.view.diff_scope == "mine"

        loaded = Session.load("test", parts=_parts(tmp_path), state=stored)
        assert loaded.status().view.diff_scope == "mine"

    asyncio.run(go())


def _parts(tmp_path):
    from xarness.sandbox import SandboxRuntime
    from xarness.session import SessionParts
    return SessionParts(
        workspace=tmp_path, profile=PROFILE, profile_name="test",
        api_key="key", config_path=None,
        sandbox=SandboxRuntime(), workspace_state=None, tool_output_limit=None,
    )


def test_repair_inserts_synthetic_result_next_to_its_call() -> None:
    """A session saved mid-tool-loop and then steered has the orphaned tool
    call in the middle of the history: the synthetic result must be
    inserted right after the owning assistant message (providers require
    adjacency), not appended at the end."""
    from xarness.conversation import AssistantMessage, ToolCall

    conversation = Conversation()
    conversation.add(user_message("go"))
    call = ToolCall(call_id="c1", name="run_bash", arguments_json="{}")
    conversation.add(AssistantMessage(content="", tool_calls=[call]))
    conversation.add(user_message("steer"))  # appended mid-loop
    changed = conversation.repair_orphaned_tool_calls()
    assert changed
    roles = [m.role for m in conversation.messages]
    assert roles == ["user", "assistant", "tool", "user"], roles
    assert conversation.messages[2].tool_call_id == "c1"
    assert conversation.messages[2].status == "interrupted"


def test_persist_watermark_advances_across_writes(tmp_path, monkeypatch) -> None:
    """The optimistic watermark must advance: after the first write, later
    persists serialize only the tail (O(tail), not O(history))."""
    session = make_session([_round("one"), _round("two")], tmp_path)
    from_seqs: list[int] = []
    real = session_store.serialize_state

    def spy(stored, from_seq):
        from_seqs.append(from_seq)
        return real(stored, from_seq)

    monkeypatch.setattr(session_store, "serialize_state", spy)

    async def go():
        await session.execute(protocol.SendMessage(text="hi"))
        await wait_idle(session)
        await session.execute(protocol.SendMessage(text="again"))
        await wait_idle(session)

    asyncio.run(go())
    assert from_seqs, "no persist happened"
    assert from_seqs[-1] >= 1, from_seqs
    # Fully written: nothing left to rewrite next time.
    assert session.conversation._dirty_from is None


def test_persist_failure_does_not_stall_the_turn(tmp_path, monkeypatch) -> None:
    """A failing write (SQLite busy, disk full) is logged and swallowed: the
    turn still ends idle with TurnEnded, and the watermark rewinds so the
    next persist rewrites from the last known-good point."""
    session = make_session([_round("one")], tmp_path)

    def boom(name, payload, meta):
        raise OSError("disk full")

    monkeypatch.setattr(session_store, "write_state", boom)

    async def go():
        queue = session.subscribe()[1]
        await session.execute(protocol.SendMessage(text="hi"))
        await wait_idle(session)
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        assert session.state == "idle"
        assert any(isinstance(e, protocol.TurnEnded) for e in events)
        assert session.conversation._dirty_from == 0  # rewound

    asyncio.run(go())
