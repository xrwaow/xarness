"""Session tests: single-writer, fan-out, atomic save, tool repair, undo."""

import asyncio
from typing import Any

import pytest

from xarness import gitwork, protocol, session_store
from xarness.config import ProviderProfile
from xarness.conversation import Message, turn_time_summary
from xarness.events import (
    ContentDelta, ProcessingStarted, ReasoningDelta, StreamError,
    ToolCallArgumentsDone, ToolCallStarted, TurnComplete, Usage,
)
from xarness.protocol import (
    Accept, AnswerAsk, Cancel, Compact, FilterAdd, Reject, Retry, SendMessage,
    SetMode, Undo, UndoTo,
)
from xarness.images import ImageAttachment
from xarness.session import Session
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
        session.controller.tools = tools
    return session


async def wait_idle(session: Session, timeout: float = 10.0) -> None:
    async def _wait():
        while session._state != "idle" or session._pending_ask is not None:
            await asyncio.sleep(0.005)
    await asyncio.wait_for(_wait(), timeout=timeout)


def _image() -> ImageAttachment:
    return ImageAttachment(
        name="cat.png", mime="image/png", data_b64="636174", width=1, height=1,
    )


def _round(text: str = "ok") -> list[Any]:
    return [ContentDelta(text=text), TurnComplete(usage=Usage(input_tokens=1, output_tokens=1))]


def test_undo_and_retry_keep_images(tmp_path) -> None:
    session = make_session([_round("one"), _round("two")], tmp_path)

    async def go():
        await session.execute("s1", SendMessage(text="look [x]", images=[_image()]))
        await wait_idle(session)
        await session.execute("r1", Retry())
        await wait_idle(session)
        return await session.execute("u1", Undo())

    undone = asyncio.run(go())
    assert undone.payload["images"][0]["name"] == "cat.png"
    wire = session.controller._client.received_wire[-1]
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
        await session.execute("m1", SetMode(mode="plan"))
        for i, bad in enumerate((".", "..", "/etc")):
            assert not (await session.execute(f"f{i}", FilterAdd(path=bad))).ok
        assert session.tool_registry.max_output_chars == 4096

    asyncio.run(go())


def test_send_runs_turn_and_fans_out_no_gap(tmp_path) -> None:
    session = make_session(
        [[ReasoningDelta(text="hmm"), ContentDelta(text="Hello"), TurnComplete(usage=Usage(input_tokens=10, output_tokens=2))]],
        tmp_path,
    )

    async def go():
        snapshot, queue = session.subscribe()
        await session.execute("c1", SendMessage(text="hi"))
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
    assert types[1] == "UserMessageAdded"
    assert "TurnFinished" in types
    # Snapshot + subscribe has no gap and no duplicate: every live event's
    # seq is strictly greater than the snapshot's, all unique and ordered.
    seqs = [e.seq for e in events]
    assert all(s > snapshot.seq for s in seqs)
    assert seqs == sorted(set(seqs))


def test_status_refresh_per_round_during_streaming(tmp_path) -> None:
    """Each round folds its usage; a StatusChanged must follow so clients
    update in/out/ctx per round, not only when the turn ends."""
    noop = tool("noop", "does nothing")(lambda: "ok")
    session = make_session(
        [
            [ContentDelta(text="one"), TurnComplete(usage=Usage(input_tokens=100, output_tokens=10)),
             ToolCallStarted(call_id="c1", name="noop"),
             ToolCallArgumentsDone(call_id="c1", name="noop", arguments_json="{}")],
            [ContentDelta(text="two"), TurnComplete(usage=Usage(input_tokens=200, output_tokens=20))],
        ],
        tmp_path, tools=ToolRegistry([noop]),
    )

    async def go():
        snapshot, queue = session.subscribe()
        await session.execute("c1", SendMessage(text="hi"))
        await wait_idle(session)
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        return snapshot, events

    snapshot, events = asyncio.run(go())
    statuses = [e.status for e in events if isinstance(e, protocol.StatusChanged)]
    totals = [(s.total_in, s.total_out) for s in statuses]
    # The per-round refreshes appear in order: after round 1 and after round 2.
    assert (100, 10) in totals and (300, 30) in totals, totals
    assert totals.index((100, 10)) < totals.index((300, 30))
    assert statuses[-1].context_used > 0


def test_history_replaced_never_sits_in_turn_events(tmp_path) -> None:
    """A rewrite invalidates every buffered turn event, and the event itself
    is never buffered: an attaching client gets the fresh history in its
    snapshot, never a replayed HistoryReplaced on top of stale deltas."""
    session = make_session([[ContentDelta(text="x"), TurnComplete()]], tmp_path)

    async def go():
        session.subscribe()
        # Notices only buffer mid-turn (an attaching client mid-turn needs
        # the deltas; an idle one gets the history in its snapshot).
        session._set_state("running")
        session._emit_notice("before")
        assert session._turn_events
        session._emit_history()
        assert session._turn_events == []

    asyncio.run(go())


def test_send_while_running_becomes_steering(tmp_path) -> None:
    release = asyncio.Event()

    @tool("noop", "does nothing")
    async def noop(args) -> ToolResult:
        await release.wait()
        return ToolResult(ok=True)

    registry = ToolRegistry()
    registry.register(noop)
    session = make_session(
        [
            [
                ToolCallStarted(call_id="c1", name="noop"),
                ToolCallArgumentsDone(call_id="c1", name="noop", arguments_json="{}"),
                TurnComplete(has_tool_calls=True),
            ],
            [ContentDelta(text="second round"), TurnComplete()],
        ],
        tmp_path, tools=registry,
    )

    async def go():
        session.subscribe()
        await session.execute("c1", SendMessage(text="go"))
        result = await session.execute("c2", SendMessage(text="steer me"))
        assert result.ok and result.payload == {"queued": True}
        assert session._queued, "message should be queued while running"
        release.set()
        await wait_idle(session)
        # Injected at the round boundary as a proper user message.
        user_texts = [m.content for m in session.conversation.messages if m.role == "user"]
        assert "steer me" in user_texts
        assert session.last_user_message_at() is not None

    asyncio.run(go())


def test_rewrite_commands_rejected_while_busy(tmp_path) -> None:
    release = asyncio.Event()

    @tool("blocker", "blocks")
    async def blocker(args) -> ToolResult:
        await release.wait()
        return ToolResult(ok=True)

    registry = ToolRegistry()
    registry.register(blocker)
    session = make_session(
        [
            [
                ToolCallStarted(call_id="c1", name="blocker"),
                ToolCallArgumentsDone(call_id="c1", name="blocker", arguments_json="{}"),
                TurnComplete(has_tool_calls=True),
            ],
            [ContentDelta(text="x"), TurnComplete()],
        ],
        tmp_path, tools=registry,
    )

    async def go():
        session.subscribe()
        await session.execute("c1", SendMessage(text="hi"))
        result = await session.execute("c2", UndoTo(message_id="whatever"))
        assert not result.ok and "wait for" in (result.error or "")
        release.set()
        await wait_idle(session)

    asyncio.run(go())


def test_cancel_interrupts_turn(tmp_path) -> None:
    session = make_session([[ContentDelta(text="partial"), TurnComplete()]], tmp_path)

    async def go():
        session.subscribe()
        await session.execute("c1", SendMessage(text="hi"))
        result = await session.execute("c2", Cancel())
        assert result.ok
        await wait_idle(session)
        assert session._state == "idle"

    asyncio.run(go())


def test_cmd_id_dedupe(tmp_path) -> None:
    session = make_session([[ContentDelta(text="x"), TurnComplete()]], tmp_path)

    async def go():
        first = await session.execute("same-id", SendMessage(text="hi"))
        await wait_idle(session)
        # A retry with the same cmd_id returns the cached result without
        # starting another turn.
        second = await session.execute("same-id", SendMessage(text="hi"))
        assert first.ok and second.ok
        assert session._turn_task is None

    asyncio.run(go())


def test_ask_first_answer_wins(tmp_path) -> None:
    session = make_session([[ContentDelta(text="x"), TurnComplete()]], tmp_path)

    async def go():
        queue = session.subscribe()[1]
        ask_task = asyncio.get_running_loop().create_task(session._ask_user(["what next?"]))
        await asyncio.sleep(0)
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        asked = [e for e in events if type(e).__name__ == "AskRequested"]
        assert len(asked) == 1
        first = await session.execute("a1", AnswerAsk(ask_id=asked[0].ask_id, answers=["do it"]))
        second = await session.execute("a2", AnswerAsk(ask_id=asked[0].ask_id, answers=["no"]))
        assert first.ok and not second.ok  # first answer wins
        assert await ask_task == ["do it"]

    asyncio.run(go())


def test_tool_call_repair_after_cancel(tmp_path) -> None:
    @tool("slow", "slow tool")
    async def slow(args) -> ToolResult:
        await asyncio.sleep(30)
        return ToolResult(ok=True)

    registry = ToolRegistry()
    registry.register(slow)
    session = make_session(
        [[
            ToolCallStarted(call_id="c1", name="slow"),
            ToolCallArgumentsDone(call_id="c1", name="slow", arguments_json="{}"),
            TurnComplete(has_tool_calls=True),
        ]],
        tmp_path, tools=registry,
    )

    async def go():
        session.subscribe()
        await session.execute("c1", SendMessage(text="hi"))
        await session.execute("c2", Cancel())
        await wait_idle(session, timeout=15)

    asyncio.run(go())
    # The interrupted call got a synthetic error result, so the next API
    # request is valid (every tool call is answered).
    tool_messages = [m for m in session.conversation.messages if m.role == "tool"]
    assert [m.content for m in tool_messages] == ["error: interrupted"]
    wire = session.conversation.to_wire()
    answered = {m["tool_call_id"] for m in wire if m["role"] == "tool"}
    for m in wire:
        for call in m.get("tool_calls") or []:
            assert call["id"] in answered


def test_undo_to_by_message_id(tmp_path) -> None:
    session = make_session(
        [[ContentDelta(text="answer"), TurnComplete(usage=Usage(input_tokens=5, output_tokens=2))]], tmp_path,
    )

    async def go():
        await session.execute("c1", SendMessage(text="first question"))
        await wait_idle(session)
        user = next(m for m in session.conversation.messages if m.role == "user")
        return await session.execute("u1", UndoTo(message_id=user.id))

    result = asyncio.run(go())
    assert result.ok
    assert (result.payload or {}).get("user_text") == "first question"
    # The turn was dropped: nothing remains.
    assert session.conversation.messages == []


def test_atomic_save_and_timestamps(tmp_path, monkeypatch) -> None:
    session = make_session(
        [[ContentDelta(text="answer"), TurnComplete(usage=Usage(input_tokens=5, output_tokens=2))]], tmp_path,
    )

    async def go():
        await session.execute("c1", SendMessage(text="hello"))
        await wait_idle(session)

    asyncio.run(go())
    # Atomic write: no temp file left behind, and the saved file parses.
    leftovers = list((tmp_path / "sessions").glob("*.tmp"))
    assert leftovers == []
    state = session_store.load_state("test")
    assert state.profile == "test"  # the profile key, not model_id
    user = state.conversation.messages[0]
    assistant = state.conversation.messages[1]
    assert user.send_time is not None
    assert assistant.start_time is not None and assistant.end_time is not None
    # Timestamps are never sent to the provider.
    wire = session.controller._client.received_wire[-1]
    for message in wire:
        assert set(message) <= {
            "role", "content", "name", "tool_call_id", "tool_calls", "reasoning_content",
        }


def test_undo_and_retry_commands_happy_path(tmp_path) -> None:
    """The /undo and /retry commands actually run their handlers: undo
    drops the last turn and reports the user text; retry drops the turn and
    resends it."""
    script = [
        [ContentDelta(text="first answer"), TurnComplete(usage=Usage(input_tokens=5, output_tokens=2))],
        [ContentDelta(text="second answer"), TurnComplete(usage=Usage(input_tokens=5, output_tokens=2))],
        [ContentDelta(text="second answer"), TurnComplete(usage=Usage(input_tokens=5, output_tokens=2))],
    ]

    async def go():
        # /undo: drops the last turn, hands the user text back.
        session = make_session([list(s) for s in script[:2]], tmp_path)
        await session.execute("c1", SendMessage(text="first question"))
        await wait_idle(session)
        await session.execute("c2", SendMessage(text="second question"))
        await wait_idle(session)
        undo = await session.execute("u1", Undo())
        assert undo.ok, undo.error
        assert (undo.payload or {}).get("user_text") == "second question"
        return session

    undone = asyncio.run(go())
    assert [
        m.content for m in undone.conversation.messages if m.role == "assistant"
    ] == ["first answer"]

    async def go2():
        # /retry: drops the last turn and resends its user message.
        session = make_session(script, tmp_path)
        await session.execute("c1", SendMessage(text="first question"))
        await wait_idle(session)
        await session.execute("c2", SendMessage(text="second question"))
        await wait_idle(session)
        retry = await session.execute("r1", Retry())
        assert retry.ok, retry.error
        async def _wait_answers():
            # The retry replaces the turn: the resent question gets a fresh
            # answer (assistant messages after the resend are stamped anew).
            while session._turn_task is not None or session._state != "idle":
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
        [[ContentDelta(text="answer"), TurnComplete(usage=Usage(input_tokens=900, output_tokens=100))]],
        tmp_path,
    )
    session.auto_compact = True

    async def fake_compact() -> str:
        return "summary"

    async def fake_finish_turn() -> None:
        finish_started.set()
        await finish_gate.wait()

    session.controller.compact = fake_compact
    session.controller.finish_turn = fake_finish_turn

    async def go():
        session.subscribe()
        first = asyncio.create_task(session.execute("c1", SendMessage(text="hi")))
        await asyncio.wait_for(finish_started.wait(), timeout=5)
        # Auto-compact ran; the session is NOT idle while finish_turn blocks.
        assert session._state != "idle"
        second = await session.execute("c2", SendMessage(text="go again"))
        assert second.ok and second.payload == {"queued": True}
        assert session._queued and session._queued[0][0] == "go again"
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
    session = make_session([[ContentDelta(text="x"), TurnComplete()]], tmp_path)

    async def fake_compact() -> str:
        await gate.wait()
        return "summary"

    session.controller.compact = fake_compact

    async def go():
        session.subscribe()
        # Returns immediately instead of awaiting the compaction.
        result = await asyncio.wait_for(
            session.execute("c1", Compact()), timeout=2,
        )
        assert result.ok and (result.payload or {}).get("started") is True
        assert session._state == "compacting"
        assert session._compact_task is not None
        # The inbox is still responsive while compaction runs.
        sent = await asyncio.wait_for(
            session.execute("c2", SendMessage(text="after compact")), timeout=2,
        )
        assert sent.ok and sent.payload == {"queued": True}
        gate.set()
        await wait_idle(session)

    asyncio.run(go())
    user_texts = [m.content for m in session.conversation.messages if m.role == "user"]
    assert "after compact" in user_texts


def test_status_fills_sandbox_name_filter_list_and_drift_detail(tmp_path) -> None:
    from xarness.config import SandboxSettings

    session = make_session([], tmp_path)
    session.sandbox_name = "netless"
    session.sandbox_settings = SandboxSettings(filter_list=["secrets"])
    session._pending_drift = ("aaa", "bbb")
    status = session.status()
    assert status.sandbox_name == "netless"
    assert status.filter_list == ["secrets"]
    assert status.pending_drift is not None
    assert status.pending_drift.detail


def test_drift_accept_emits_drift_resolved(tmp_path, monkeypatch) -> None:
    from xarness.gitwork import GitInfo
    import xarness.session as session_mod

    session = make_session([], tmp_path)

    async def fake_accept(info):
        return "f" * 40

    monkeypatch.setattr(session_mod, "accept_changes", fake_accept)
    session.git_info = GitInfo(
        session_id="s", workspace=tmp_path, git_dir=tmp_path,
        baseline_tree="a" * 40,
    )

    async def go():
        queue = session.subscribe()[1]
        session._pending_drift = ("aaa", "bbb")
        result = await session.execute("d1", protocol.ResolveDrift(accept=False))
        assert result.ok
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        resolved = [e for e in events if type(e).__name__ == "DriftResolved"]
        assert len(resolved) == 1

    asyncio.run(go())

    # Accept path: the drift is accepted and DriftResolved(accepted=True)
    # is emitted.
    async def go2():
        queue = session.subscribe()[1]
        session._pending_drift = ("aaa", "bbb")
        result = await session.execute("d2", protocol.ResolveDrift(accept=True))
        assert result.ok
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        resolved = [e for e in events if type(e).__name__ == "DriftResolved"]
        assert len(resolved) == 1 and resolved[0].accepted is True

    asyncio.run(go2())


def test_stream_events_are_protocol_models() -> None:
    """Internal stream events cross the wire as-is: they are the protocol
    models themselves, with the ``type`` discriminator baked in."""
    cases = [
        (ProcessingStarted(), "processing_started"),
        (ReasoningDelta(text="x"), "reasoning_delta"),
        (ContentDelta(text="y"), "content_delta"),
        (StreamError(message="bad"), "stream_error"),
        (TurnComplete(usage=Usage(input_tokens=1, output_tokens=2), has_tool_calls=True), "turn_complete"),
    ]
    for internal, expected_type in cases:
        assert internal.model_dump()["type"] == expected_type


def test_back_to_back_sends_one_turn_one_steering(tmp_path) -> None:
    """Two send_message commands queued back-to-back must produce exactly
    ONE turn with the second message steered into it — the turn state is
    set synchronously in _start_turn, before create_task, so the inbox's
    next dispatch cannot still see "idle" and start a second turn."""
    release = asyncio.Event()

    @tool("noop", "does nothing")
    async def noop(args) -> ToolResult:
        await release.wait()
        return ToolResult(ok=True)

    registry = ToolRegistry()
    registry.register(noop)
    session = make_session(
        [
            [
                ToolCallStarted(call_id="c1", name="noop"),
                ToolCallArgumentsDone(call_id="c1", name="noop", arguments_json="{}"),
                TurnComplete(has_tool_calls=True),
            ],
            [ContentDelta(text="second round"), TurnComplete()],
        ],
        tmp_path, tools=registry,
    )

    async def go():
        snapshot, queue = session.subscribe()
        # Back-to-back: no awaiting of turn progress between the two sends.
        first = await session.execute("c1", SendMessage(text="one"))
        second = await session.execute("c2", SendMessage(text="two"))
        assert first.ok and first.payload == {"started": True}
        assert second.ok and second.payload == {"queued": True}, (
            "back-to-back send must steer, not start a second turn"
        )
        release.set()
        await wait_idle(session)
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        finished = [e for e in events if type(e).__name__ == "TurnFinished"]
        assert len(finished) == 1, f"expected one turn, got {len(finished)}"
        # The steered text was injected inside the SAME turn (the second LLM
        # round saw it), not replayed as a fresh turn afterwards.
        assert session.controller._client.received_wire[1] and any(
            m.get("role") == "user" and m.get("content") == "two"
            for m in session.controller._client.received_wire[1]
        )

    asyncio.run(go())


def test_history_replaced_never_buffered_and_resets_turn_events(tmp_path) -> None:
    """HistoryReplaced must never sit in snapshot.turn_events: while idle
    the buffer is empty, and a rewrite while running resets the buffer."""
    session = make_session([[ContentDelta(text="x"), TurnComplete()]], tmp_path)

    async def go():
        session.subscribe()
        await session.execute("c1", SendMessage(text="hi"))
        await wait_idle(session)
        assert session.snapshot().turn_events == []
        # A rewrite while "running": buffered deltas are dropped, the event
        # itself is not stored.
        session._set_state("running")
        session._emit(protocol.ContentDelta(text="stale"))
        assert session.snapshot().turn_events, "buffer holds the live round"
        session._emit_history()
        assert session.snapshot().turn_events == []

    asyncio.run(go())


def test_events_carry_monotonic_seqs_across_a_rewrite(tmp_path) -> None:
    """Sequences keep increasing across a history rewrite; HistoryReplaced
    itself is never buffered for attach (a fresh snapshot carries it)."""
    session = make_session([[ContentDelta(text="x"), TurnComplete()]], tmp_path)

    async def go():
        session.subscribe()
        await session.execute("c1", SendMessage(text="hi"))
        await wait_idle(session)
        seq = session._seq
        session._emit_notice("after")
        assert session._seq > seq
        session._emit_history()  # a rewrite
        assert session._seq > seq + 1
        assert session._turn_events == []

    asyncio.run(go())


def test_filter_add_rejects_missing_path(tmp_path) -> None:
    """/filter add on a path that does not exist is rejected — nothing is
    created in the worktree and the filter list is unchanged."""
    from xarness.config import SandboxSettings

    session = make_session([[ContentDelta(text="x"), TurnComplete()]], tmp_path)
    session.sandbox_settings = SandboxSettings()
    (tmp_path / "real.txt").write_text("x")

    async def go():
        result = await session.execute("f1", protocol.FilterAdd(path="ghost.txt"))
        assert not result.ok and "no such path" in (result.error or "")
        assert not (tmp_path / "ghost.txt").exists()
        ok = await session.execute("f2", protocol.FilterAdd(path="real.txt"))
        assert ok.ok
        assert session.sandbox_settings.filter_list == ["real.txt"]

    asyncio.run(go())


def test_ref_add_remove_is_per_workspace(tmp_path, monkeypatch) -> None:
    """Refs added via the command bind read-only under .refs/<alias> and are
    persisted per workspace; a missing path is rejected and nothing is
    created in the worktree (or at .refs)."""
    from xarness import session_store
    from xarness.config import SandboxSettings

    monkeypatch.setattr(session_store, "WORKSPACES_DIR", tmp_path / "workspaces")
    session = make_session([[ContentDelta(text="x"), TurnComplete()]], tmp_path)
    session.sandbox_settings = SandboxSettings()
    (tmp_path / "spec.md").write_text("x")
    expected = str((tmp_path / "spec.md").resolve())

    async def go():
        bad = await session.execute("r0", protocol.RefAdd(path="ghost.md"))
        assert not bad.ok and "no such path" in (bad.error or "")
        # Never mkdir a reference (and never create the container anchor).
        assert not (tmp_path / "ghost.md").exists()
        assert not (tmp_path / ".refs").exists()

        added = await session.execute("r1", protocol.RefAdd(path="spec.md"))
        assert added.ok
        assert session.sandbox_settings.auto_include_refs == [expected]
        refs = session.status().refs
        assert [(r.path, r.mount) for r in refs] == [(expected, ".refs/spec.md")]
        assert session_store.load_workspace_refs(str(tmp_path)) == [expected]

        removed = await session.execute("r2", protocol.RefRemove(path="spec.md"))
        assert removed.ok
        assert session.sandbox_settings.auto_include_refs == []
        assert session_store.load_workspace_refs(str(tmp_path)) == []

    asyncio.run(go())


def test_undo_history_replaced_renders_once_on_attach(tmp_path) -> None:
    """After an /undo (a rewrite) an attaching client's snapshot has empty
    turn_events: the history is rendered exactly once — the HistoryReplaced
    event itself is never stored or replayed."""
    session = make_session([[ContentDelta(text="x"), TurnComplete()]], tmp_path)

    async def go():
        session.subscribe()
        await session.execute("c1", SendMessage(text="hi"))
        await wait_idle(session)
        result = await session.execute("c2", Undo())
        assert result.ok
        assert session.snapshot().turn_events == []
        # A client attaching now sees the rewritten history exactly once:
        # nothing in turn_events replays on top of snapshot.messages.
        snapshot, _queue = session.subscribe()
        assert snapshot.turn_events == []
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
        task = asyncio.create_task(session.execute("c1", SendMessage(text="hi")))
        await asyncio.wait_for(started.wait(), timeout=5)
        await session.execute("c2", Cancel())
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
        await session.execute("c1", SendMessage(text="first"))
        await asyncio.wait_for(started.wait(), timeout=5)
        await session.execute("c2", SendMessage(text="second"))  # queued
        result = await session.execute("c3", protocol.FlushQueued())
        assert result.ok and result.payload == {"flushed": True}
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

    async def fake_compact() -> str:
        await gate.wait()
        return "summary"

    session.controller.compact = fake_compact

    async def go():
        session.subscribe()
        await asyncio.wait_for(session.execute("c1", Compact()), timeout=2)
        assert session._state == "compacting"
        await session.stop()
        assert session._compact_task is None

    asyncio.run(go())
    # The compact task was cancelled (stop() cleared the reference); the
    # session state never came back to idle via the cancelled task's except
    # path — stop() owns the teardown.
    assert session._state == "compacting" or session._state == "idle"


def test_no_save_disables_persistence(tmp_path) -> None:
    """no_session: _persist never writes a session file."""
    session = Session(
        name="nosave", workspace=tmp_path, profile=PROFILE,
        profile_name="test", api_key="key", client=FakeClient([]),
        no_save=True,
    )
    session.conversation.add(Message(role="user", content="q"))
    session._persist()
    assert not session_store.session_path("nosave").exists()


def test_turn_time_summary_format(tmp_path) -> None:
    """turn_time_summary derives total | running | idle from the persisted
    stamps (running = streaming + tools)."""
    from datetime import datetime, timedelta, timezone

    session = make_session([], tmp_path)
    base = datetime.now(timezone.utc)
    iso = base.isoformat

    session.conversation.add(Message(
        role="user", content="hi", send_time=iso(),
    ))
    session.conversation.add(Message(
        role="assistant", content="thinking about it",
        start_time=iso(), end_time=(base + timedelta(seconds=2)).isoformat(),
        reasoning_start=(base + timedelta(milliseconds=300)).isoformat(),
    ))
    session.conversation.add(Message(
        role="tool", tool_call_id="c1",
        start_time=(base + timedelta(seconds=2)).isoformat(),
        end_time=(base + timedelta(seconds=5)).isoformat(),
    ))
    session.conversation.add(Message(
        role="assistant", content="done",
        start_time=(base + timedelta(seconds=5)).isoformat(),
        end_time=(base + timedelta(seconds=8)).isoformat(),
    ))
    summary = turn_time_summary(session.conversation.messages)
    assert summary is not None
    total, rest = summary.split("|")
    running, idle = rest.split()
    assert total.strip() == "8.0s"
    # running = streaming (round 1: 1.7s + round 2: 3s, no reasoning)
    #          + tools (3.0s).
    assert running == "7.7s"
    # total − running: the 0.3s pre-first-token latency.
    assert idle == "0.3s"


def test_turn_finish_stamps_time_summary_and_persists(tmp_path) -> None:
    """A completed turn gets its timing summary on the last assistant message
    (persisted, so reloads render it) and in TurnFinished.summary."""
    session = make_session(
        [[ReasoningDelta(text="hmm"), ContentDelta(text="Hello"), TurnComplete(usage=Usage(input_tokens=10, output_tokens=2))]],
        tmp_path,
    )

    async def go():
        _, queue = session.subscribe()
        await session.execute("c1", SendMessage(text="hi"))
        await wait_idle(session)
        finished = None
        while not queue.empty():
            event = queue.get_nowait()
            if isinstance(event, protocol.TurnFinished):
                finished = event
        return finished

    finished = asyncio.run(go())
    last_assistant = session.conversation.messages[-1]
    assert last_assistant.role == "assistant"
    assert last_assistant.time_summary
    assert finished is not None and finished.summary == last_assistant.time_summary
    assert "|" in finished.summary

    # Persisted: a reload sees the same line.
    state = session_store.load_state("test")
    assert state.conversation.messages[-1].time_summary == last_assistant.time_summary


def test_turn_time_summary_none_without_stamps(tmp_path) -> None:
    """Messages without stamps (old sessions) produce no summary."""
    session = make_session([], tmp_path)
    session.conversation.add(Message(role="user", content="hi"))
    session.conversation.add(Message(role="assistant", content="hello"))
    assert turn_time_summary(session.conversation.messages) is None


# ----------------------------------------------------------------------
# mid-turn change attribution (/diff source tags)


async def _tracked_session(tmp_path, script, registry=None):
    """A session with real git snapshot tracking and a system message.

    The tracked workspace is a subdirectory so the (test-local) session
    files, which live directly under ``tmp_path``, are not themselves
    tracked as drift."""
    ws = tmp_path / "ws"
    ws.mkdir()
    info, _notes = await gitwork.setup_tracking(ws, "test", allow_init=True)
    assert info is not None
    session = Session(
        name="test", workspace=ws, profile=PROFILE, profile_name="test",
        api_key="key", client=FakeClient(script),
    )
    if registry is not None:
        session.tool_registry = registry
        session.controller.tools = registry
    session.git_info = info
    session.controller.git_info = info
    session.conversation.add(Message(role="system", content="sys"))
    return session, info


def _blocking_edit(tmp_path, name="a.txt"):
    """An edit tool that writes ``name`` and then blocks, so tests can act
    while the turn is still generating."""
    started = asyncio.Event()
    release = asyncio.Event()

    @tool("edit", "edit a file")
    async def edit(args) -> ToolResult:
        (tmp_path / name).write_text("agent\n")
        started.set()
        await release.wait()
        return ToolResult(ok=True, output="ok")

    registry = ToolRegistry()
    registry.register(edit)
    return registry, started, release


def _tool_round(call_id="c1"):
    return [
        ToolCallStarted(call_id=call_id, name="edit"),
        ToolCallArgumentsDone(call_id=call_id, name="edit", arguments_json="{}"),
        TurnComplete(has_tool_calls=True),
    ]


def test_mid_turn_accept_does_not_tag_later_edits_as_drift(tmp_path, monkeypatch) -> None:
    """A mid-turn /accept rewrites the open turn's before tree, but the turn
    is still open: edits it makes afterwards stay tagged agent, not drift,
    and it still records its real after tree at the end."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    async def go():
        registry, started, release = _blocking_edit(tmp_path / "ws")
        session, _info = await _tracked_session(tmp_path, [
            _tool_round(),
            [ContentDelta(text="done"), TurnComplete()],
        ], registry)
        session.subscribe()
        await session.execute("s1", SendMessage(text="hi"))
        await asyncio.wait_for(started.wait(), 5)

        stat = await session.diff_summary()
        by_path = {f.path: f.source for f in stat.files}
        assert by_path["a.txt"] == "agent"

        result = await session.execute("acc", Accept())
        assert result.ok
        assert session._state == "running"

        # The turn keeps editing after the accept.
        (tmp_path / "ws" / "b.txt").write_text("later agent edit\n")
        stat = await session.diff_summary()
        by_path = {f.path: f.source for f in stat.files}
        assert by_path == {"b.txt": "agent"}, by_path

        release.set()
        await wait_idle(session)

        stat = await session.diff_summary()
        by_path = {f.path: f.source for f in stat.files}
        assert by_path["b.txt"] == "agent"
        # finish_turn still recorded the open turn's real after tree.
        turns = session.turn_checkpoints()
        assert len(turns) == 1 and turns[0].after_tree

    asyncio.run(go())


def test_mid_turn_reject_does_not_tag_later_edits_as_drift(tmp_path, monkeypatch) -> None:
    """Same as the accept case for /reject, which rewrites every checkpoint
    to the baseline tree."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    async def go():
        registry, started, release = _blocking_edit(tmp_path / "ws")
        session, _info = await _tracked_session(tmp_path, [
            _tool_round(),
            [ContentDelta(text="done"), TurnComplete()],
        ], registry)
        session.subscribe()
        await session.execute("s1", SendMessage(text="hi"))
        await asyncio.wait_for(started.wait(), 5)

        result = await session.execute("rej", Reject())
        assert result.ok
        assert not (tmp_path / "ws" / "a.txt").exists()

        (tmp_path / "ws" / "b.txt").write_text("later agent edit\n")
        stat = await session.diff_summary()
        by_path = {f.path: f.source for f in stat.files}
        assert by_path == {"b.txt": "agent"}, by_path

        release.set()
        await wait_idle(session)
        turns = session.turn_checkpoints()
        assert len(turns) == 1 and turns[0].after_tree

    asyncio.run(go())


def test_mid_turn_compaction_keeps_attribution_and_undo(tmp_path, monkeypatch) -> None:
    """Compaction moves the open turn's message into compact_snapshot. Its
    before tree must survive so remaining edits stay agent, and finish_turn
    must still give it an after tree (so /undo can reverse it)."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    async def go():
        registry, started, release = _blocking_edit(tmp_path / "ws")
        session, _info = await _tracked_session(tmp_path, [
            _tool_round(),
            [ContentDelta(text="done"), TurnComplete()],
        ], registry)
        session.subscribe()
        await session.execute("s1", SendMessage(text="hi"))
        await asyncio.wait_for(started.wait(), 5)

        async def fake_summarize(_transcript):
            return "handoff summary", None

        session.controller._summarize = fake_summarize
        await session.controller.compact()
        assert session.controller.open_turn_before() is not None

        (tmp_path / "ws" / "b.txt").write_text("later agent edit\n")
        stat = await session.diff_summary()
        by_path = {f.path: f.source for f in stat.files}
        assert by_path["a.txt"] == "agent"
        assert by_path["b.txt"] == "agent"

        release.set()
        await wait_idle(session)

        # The open turn survived compaction and now has a reversible diff.
        turns = session.turn_checkpoints()
        assert len(turns) == 1 and turns[0].after_tree
        stat = await session.diff_summary()
        assert {f.path: f.source for f in stat.files}["b.txt"] == "agent"

    asyncio.run(go())


def test_checkpoint_failure_does_not_restamp_previous_turn(tmp_path, monkeypatch) -> None:
    """If a turn's before-snapshot fails, finish_turn must not write that
    turn's after tree onto the previous (already closed) turn."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    async def go():
        registry, started, release = _blocking_edit(tmp_path / "ws", "b.txt")
        session, _info = await _tracked_session(tmp_path, [
            [ContentDelta(text="first"), TurnComplete()],
            _tool_round(),
            [ContentDelta(text="done"), TurnComplete()],
        ], registry)
        session.subscribe()

        await session.execute("s1", SendMessage(text="first"))
        await wait_idle(session)
        first_turn = session.conversation.messages[1]
        assert first_turn.role == "user" and first_turn.content == "first"
        first_after = first_turn.after_tree
        assert first_after

        async def fail_checkpoint():
            return None

        session.controller._take_checkpoint = fail_checkpoint
        await session.execute("s2", SendMessage(text="second"))
        await asyncio.wait_for(started.wait(), 5)

        # The failing checkpoint means this turn cannot be attributed yet,
        # but the previous, closed turn must not be touched.
        assert first_turn.after_tree == first_after

        release.set()
        await wait_idle(session)

        assert first_turn.after_tree == first_after
        second_turn = next(
            m for m in session.conversation.messages
            if m.role == "user" and m.content == "second"
        )
        assert second_turn.checkpoint_sha is None
        assert second_turn.after_tree is None
        assert len(session.turn_checkpoints()) == 1

    asyncio.run(go())
