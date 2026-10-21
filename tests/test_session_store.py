"""Tests for session persistence: round-trip of local-only message fields."""

import asyncio

from xarness import session_store
from xarness.config import CotStrength, ProviderProfile
from xarness.controller import ChatController
from xarness.conversation import Conversation, Message, Timing, ToolCall
from xarness.protocol import ContentDelta, TurnComplete, Usage

PROFILE = ProviderProfile(
    base_url="https://api.example.test/v1",
    model_id="test-model",
    max_context=1000,
    cot_strength=CotStrength.MEDIUM,
)


def test_message_fields_round_trip(tmp_path, monkeypatch) -> None:
    """usage / turn checkpoints survive save + load, and stay out of the
    wire format."""
    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)

    conversation = Conversation()
    user = Message(role="user", content="hi")
    assistant = Message(
        role="assistant", content="hello", reasoning="thoughts", usage=Usage(input_tokens=12, output_tokens=3)
    )
    conversation.add(user)
    conversation.add(assistant)
    conversation.begin_turn(user, "abc123")
    conversation.turns[-1].after_tree = "def456"
    session_store.save_state("s", session_store.SessionState(conversation, "m"))

    loaded = session_store.load_state("s").conversation
    assert [m.role for m in loaded.messages] == ["user", "assistant"]
    assert loaded.messages[1].usage == Usage(input_tokens=12, output_tokens=3)
    assert loaded.messages[1].reasoning == "thoughts"
    turn = loaded.turns[0]
    assert turn.user_message_id == loaded.messages[0].id
    assert turn.before_tree == "abc123"
    assert turn.after_tree == "def456"

    # Local-only fields never reach the wire.
    assert all(
        "usage" not in m and "checkpoint" not in m
        for m in loaded.to_wire()
    )


def test_usage_totals_round_trip(tmp_path, monkeypatch) -> None:
    """Cumulative and last-round token counters persist, so a reloaded
    session's footer shows real in/out/ctx instead of 0."""
    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)

    conversation = Conversation()
    conversation.add(Message(role="user", content="hi"))
    conversation.add(Message(role="assistant", content="hello"))
    session_store.save_state("s", session_store.SessionState(
        conversation, "m", workspace="/w",
        total_in=1200, total_out=340, last_in=1100, last_out=300,
    ))
    loaded = session_store.load_state("s")
    # The model and context accounting a resumed session needs are all here.
    assert loaded.profile == "m" and loaded.workspace == "/w"
    assert (loaded.total_in, loaded.total_out) == (1200, 340)
    assert (loaded.last_in, loaded.last_out) == (1100, 300)

    # A session saved with no usage yet loads as zeros.
    session_store.save_state("empty", session_store.SessionState(conversation, "m"))
    empty = session_store.load_state("empty")
    assert (empty.total_in, empty.total_out, empty.last_in, empty.last_out) == (0, 0, 0, 0)


def test_mode_round_trips(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)
    conversation = Conversation()
    session_store.save_state("s", session_store.SessionState(
        conversation, "m", mode="plan",
    ))
    assert session_store.load_state("s").mode == "plan"


def test_to_wire_drops_unpaired_tool_calls() -> None:
    """A tool call without a recorded result (turn interrupted mid-execution,
    session saved) must not reach the wire as an unpaired call: providers
    reject unpaired calls. Repair adds the synthetic result; to_wire then
    sends every message as-is."""

    def make_conversation(content, calls, results):
        conversation = Conversation()
        conversation.add(Message(role="user", content="go"))
        conversation.add(Message(role="assistant", content=content, tool_calls=calls))
        for call_id, output in results:
            conversation.add(Message(role="tool", content=output, tool_call_id=call_id))
        conversation.repair_orphaned_tool_calls()
        return conversation

    # Two calls, one answered: the orphaned call gets a synthetic result.
    conversation = make_conversation(
        "",
        [ToolCall(call_id="call_1", name="run_bash", arguments_json="{}"), ToolCall(call_id="call_2", name="read_file", arguments_json="{}")],
        [("call_1", "ok")],
    )
    wire = conversation.to_wire()
    assert [m["role"] for m in wire] == ["user", "assistant", "tool", "tool"]
    assert [c["id"] for c in wire[1]["tool_calls"]] == ["call_1", "call_2"]
    assert wire[3]["tool_call_id"] == "call_2"
    assert wire[3]["content"] == "interrupted"

    # Text plus both calls survive; the synthetic result keeps the round valid.
    wire = make_conversation(
        "let me check",
        [ToolCall(call_id="call_1", name="run_bash", arguments_json="{}"), ToolCall(call_id="call_2", name="read_file", arguments_json="{}")],
        [("call_1", "ok")],
    ).to_wire()
    assert wire[1]["content"] == "let me check"
    assert [c["id"] for c in wire[1]["tool_calls"]] == ["call_1", "call_2"]

    # No answered calls and no content: repair adds a synthetic result for
    # the unanswered call so the round stays valid.
    wire = make_conversation(
        "", [ToolCall(call_id="call_1", name="run_bash", arguments_json="{}")], []
    ).to_wire()
    assert [m["role"] for m in wire] == ["user", "assistant", "tool"]

    # Fully answered calls are untouched.
    wire = make_conversation(
        "",
        [ToolCall(call_id="call_1", name="run_bash", arguments_json="{}"), ToolCall(call_id="call_2", name="read_file", arguments_json="{}")],
        [("call_1", "ok"), ("call_2", "ok")],
    ).to_wire()
    assert [c["id"] for c in wire[1]["tool_calls"]] == ["call_1", "call_2"]


def test_resumed_session_undo_rolls_back(tmp_path, monkeypatch) -> None:
    """/undo works after a resume: the snapshot and checkpoint sha loaded
    from the session file drive the same rollback as in-session."""
    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)

    client_script = [ContentDelta(text="answer"), TurnComplete(usage=Usage(input_tokens=10, output_tokens=4))]

    class FakeClient:
        async def stream(self, wire_messages, tools=None):
            for event in client_script:
                yield event

    controller = ChatController(PROFILE, "k", client=FakeClient())
    asyncio.run(_send(controller, "hi"))
    session_store.save_state("s", session_store.SessionState(controller.conversation, "m"))

    resumed = session_store.load_state("s").conversation
    controller2 = ChatController(PROFILE, "k", client=FakeClient())
    controller2.conversation = resumed

    plan = controller2.rollback_plan()
    assert plan is not None and plan.user_text == "hi"
    controller2.apply_rollback(plan)
    assert controller2.conversation.messages == []
    assert controller2.usage_total == Usage(input_tokens=0, output_tokens=0)


async def _send(controller: ChatController, text: str) -> None:
    async for _ in controller.send(text):
        pass


def test_meta_block_written_and_read_without_parsing_conversation(tmp_path, monkeypatch) -> None:
    import json

    from xarness import session_store as store

    monkeypatch.setattr(store, "SESSIONS_DIR", tmp_path / "sessions")
    conversation = Conversation()
    conversation.add(Message(role="user", content="q", timing=Timing(send="2026-01-01T00:00:00+00:00")))
    conversation.add(Message(role="assistant", content="a"))
    store.save_state("s", store.SessionState(
        conversation, profile="p", workspace="/w", git={"session_id": "s"},
        sandbox_name="sb",
    ))
    data = json.loads((tmp_path / "sessions" / "s.json").read_text())
    # The meta block lives only in the shared index, never in the session file.
    assert "meta" not in data
    index = json.loads((tmp_path / "sessions" / "index.json").read_text())
    meta_file = index["sessions"]["s"]
    assert meta_file["workspace"] == "/w"
    assert meta_file["message_count"] == 2
    assert meta_file["last_user_message_at"] == "2026-01-01T00:00:00+00:00"
    assert meta_file["tracking"] is True
    assert meta_file["profile"] == "p" and meta_file["sandbox_name"] == "sb"

    # Reading the meta must not build Conversation objects (the whole point:
    # listing and last-used lookups stay cheap).
    def explode(*args, **kwargs):
        raise AssertionError("full conversation parsed")
    monkeypatch.setattr(store, "load_state", explode)
    meta = store.load_meta("s")
    assert meta["workspace"] == "/w" and meta["message_count"] == 2
    assert meta["last_user_message_at"] == "2026-01-01T00:00:00+00:00"
    assert store.load_meta("s")["workspace"] == "/w"




def _write_fixture(tmp_path, name: str, payload: dict) -> None:
    import json
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_saved_files_carry_a_schema_version(tmp_path, monkeypatch) -> None:
    import json

    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)
    conversation = Conversation()
    conversation.add(Message(role="user", content="hi"))
    session_store.save_state("s", session_store.SessionState(conversation))
    data = json.loads((tmp_path / "s.json").read_text())
    assert data["schema"] == session_store.SCHEMA_VERSION


def test_tool_calls_persist_in_internal_shape(tmp_path, monkeypatch) -> None:
    import json

    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)
    conversation = Conversation()
    conversation.add(Message(
        role="assistant", content="",
        tool_calls=[ToolCall(call_id="c1", name="run_bash", arguments_json="{}")],
    ))
    session_store.save_state("s", session_store.SessionState(conversation))
    raw = json.loads((tmp_path / "s.json").read_text())
    assert raw["messages"][0]["tool_calls"] == [
        {"call_id": "c1", "name": "run_bash", "arguments_json": "{}"},
    ]


def test_unsupported_schemas_are_rejected(tmp_path, monkeypatch) -> None:
    import pytest

    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)
    _write_fixture(tmp_path, "future", {
        "schema": session_store.SCHEMA_VERSION + 1,
        "messages": [],
        "turns": [],
    })
    with pytest.raises(ValueError, match="newer than supported"):
        session_store.load_state("future")

    _write_fixture(tmp_path, "old", {
        "schema": session_store.SCHEMA_VERSION - 1,
        "messages": [],
        "turns": [],
    })
    with pytest.raises(ValueError, match="no longer supported"):
        session_store.load_state("old")


def test_meta_index_written_and_removed(tmp_path, monkeypatch) -> None:
    """save_state folds metadata into one index; deleting the session drops
    its entry and the session file."""
    import json

    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)
    conversation = Conversation()
    conversation.add(Message(role="user", content="q", timing=Timing(send="2026-01-01T00:00:00+00:00")))
    session_store.save_state("s", session_store.SessionState(conversation, "m", workspace="/w"))

    index_file = tmp_path / "index.json"
    assert index_file.exists()
    assert not (tmp_path / "s.meta.json").exists()
    meta = json.loads(index_file.read_text())["sessions"]["s"]
    assert meta["workspace"] == "/w" and meta["message_count"] == 1
    # The index is not picked up as a session by list_sessions.
    assert session_store.list_sessions() == ["s"]
    assert session_store.load_meta("s")["workspace"] == "/w"

    assert session_store.delete_session("s") is True
    assert not (tmp_path / "s.json").exists()
    assert session_store.load_metas() == {}
