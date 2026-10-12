"""Tests for session persistence: round-trip of local-only message fields."""

import asyncio

from xarness import session_store
from xarness.conversation import Conversation, Message, ToolCall
from xarness.events import ContentDelta, TurnComplete, Usage
from xarness.controller import ChatController
from xarness.config import CotStrength, ProviderProfile

PROFILE = ProviderProfile(
    base_url="https://api.example.test/v1",
    model_id="test-model",
    max_context=1000,
    cot_strength=CotStrength.MEDIUM,
)


def test_message_fields_round_trip(tmp_path, monkeypatch) -> None:
    """usage / checkpoint_sha / after_tree / undo_snapshot survive save +
    load, and stay out of the wire format."""
    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)

    conversation = Conversation()
    user = Message(role="user", content="hi", checkpoint_sha="abc123", after_tree="def456")
    assistant = Message(
        role="assistant", content="hello", reasoning="thoughts", usage=Usage(12, 3)
    )
    conversation.add(user)
    conversation.add(assistant)
    conversation.turn_start = 0
    session_store.save_state("s", session_store.SessionState(conversation, "m"))

    loaded = session_store.load_state("s").conversation
    assert [m.role for m in loaded.messages] == ["user", "assistant"]
    assert loaded.messages[0].checkpoint_sha == "abc123"
    assert loaded.messages[0].after_tree == "def456"
    assert loaded.messages[1].usage == Usage(12, 3)
    assert loaded.messages[1].reasoning == "thoughts"
    assert loaded.turn_start == 0
    assert loaded.messages[loaded.turn_start].checkpoint_sha == "abc123"
    assert loaded.messages[loaded.turn_start].after_tree == "def456"

    # Local-only fields never reach the wire.
    assert all(
        "usage" not in m and "checkpoint_sha" not in m and "after_tree" not in m
        for m in loaded.to_wire()
    )


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
        [ToolCall("call_1", "run_bash", "{}"), ToolCall("call_2", "read_file", "{}")],
        [("call_1", "ok")],
    )
    wire = conversation.to_wire()
    assert [m["role"] for m in wire] == ["user", "assistant", "tool", "tool"]
    assert [c["id"] for c in wire[1]["tool_calls"]] == ["call_1", "call_2"]
    assert wire[3]["tool_call_id"] == "call_2"
    assert wire[3]["content"] == "error: interrupted"

    # Text plus both calls survive; the synthetic result keeps the round valid.
    wire = make_conversation(
        "let me check",
        [ToolCall("call_1", "run_bash", "{}"), ToolCall("call_2", "read_file", "{}")],
        [("call_1", "ok")],
    ).to_wire()
    assert wire[1]["content"] == "let me check"
    assert [c["id"] for c in wire[1]["tool_calls"]] == ["call_1", "call_2"]

    # No answered calls and no content: repair adds a synthetic result for
    # the unanswered call so the round stays valid.
    wire = make_conversation(
        "", [ToolCall("call_1", "run_bash", "{}")], []
    ).to_wire()
    assert [m["role"] for m in wire] == ["user", "assistant", "tool"]

    # Fully answered calls are untouched.
    wire = make_conversation(
        "",
        [ToolCall("call_1", "run_bash", "{}"), ToolCall("call_2", "read_file", "{}")],
        [("call_1", "ok"), ("call_2", "ok")],
    ).to_wire()
    assert [c["id"] for c in wire[1]["tool_calls"]] == ["call_1", "call_2"]


def test_resumed_session_undo_rolls_back(tmp_path, monkeypatch) -> None:
    """/undo works after a resume: the snapshot and checkpoint sha loaded
    from the session file drive the same rollback as in-session."""
    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)

    client_script = [ContentDelta("answer"), TurnComplete(usage=Usage(10, 4))]

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
    conversation.add(Message(role="user", content="q", send_time="2026-01-01T00:00:00+00:00"))
    conversation.add(Message(role="assistant", content="a"))
    store.save_state("s", store.SessionState(
        conversation, profile="p", workspace="/w", git={"session_id": "s"},
        sandbox_name="sb",
    ))
    data = json.loads((tmp_path / "sessions" / "s.json").read_text())
    assert data["meta"]["workspace"] == "/w"
    assert data["meta"]["message_count"] == 2
    assert data["meta"]["last_user_message_at"] == "2026-01-01T00:00:00+00:00"
    assert data["meta"]["tracking"] is True
    assert data["meta"]["profile"] == "p" and data["meta"]["sandbox_name"] == "sb"

    # Reading the meta must not build Conversation objects (the whole point:
    # listing and last-used lookups stay cheap).
    def explode(*args, **kwargs):
        raise AssertionError("full conversation parsed")
    monkeypatch.setattr(store, "_conversation_from", explode)
    meta = store.load_meta("s")
    assert meta["workspace"] == "/w" and meta["message_count"] == 2
    assert meta["last_user_message_at"] == "2026-01-01T00:00:00+00:00"
    assert store.load_meta("s")["workspace"] == "/w"




def test_meta_sidecar_written_and_removed(tmp_path, monkeypatch) -> None:
    """save_state writes a meta sidecar so listing lookups never read the
    full session file; deleting the session removes both files."""
    import json

    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)
    conversation = Conversation()
    conversation.add(Message(role="user", content="q", send_time="2026-01-01T00:00:00+00:00"))
    session_store.save_state("s", session_store.SessionState(conversation, "m", workspace="/w"))

    sidecar = tmp_path / "s.meta.json"
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["workspace"] == "/w" and meta["message_count"] == 1
    # The sidecar is not picked up as a session by list_sessions.
    assert session_store.list_sessions() == ["s"]
    assert session_store.load_meta("s")["workspace"] == "/w"

    assert session_store.delete_session("s") is True
    assert not sidecar.exists()
    assert not (tmp_path / "s.json").exists()
