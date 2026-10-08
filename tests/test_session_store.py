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
    session_store.save_session("s", "m", conversation)

    loaded = session_store.load_session("s")
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
    session saved) must not reach the wire: providers reject unpaired calls."""

    def make_conversation(content, calls, results):
        conversation = Conversation()
        conversation.add(Message(role="user", content="go"))
        conversation.add(Message(role="assistant", content=content, tool_calls=calls))
        for call_id, output in results:
            conversation.add(Message(role="tool", content=output, tool_call_id=call_id))
        return conversation

    # Two calls, one answered: the round stays valid with only the answered
    # call; the orphaned call is gone.
    wire = make_conversation(
        "",
        [ToolCall("call_1", "run_bash", "{}"), ToolCall("call_2", "read_file", "{}")],
        [("call_1", "ok")],
    ).to_wire()
    assert [m["role"] for m in wire] == ["user", "assistant", "tool"]
    assert [c["id"] for c in wire[1]["tool_calls"]] == ["call_1"]

    # Text plus the answered call both survive.
    wire = make_conversation(
        "let me check",
        [ToolCall("call_1", "run_bash", "{}"), ToolCall("call_2", "read_file", "{}")],
        [("call_1", "ok")],
    ).to_wire()
    assert wire[1]["content"] == "let me check"
    assert [c["id"] for c in wire[1]["tool_calls"]] == ["call_1"]

    # No answered calls and no content: the round is dropped entirely.
    wire = make_conversation(
        "", [ToolCall("call_1", "run_bash", "{}")], []
    ).to_wire()
    assert [m["role"] for m in wire] == ["user"]

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
    session_store.save_session("s", "m", controller.conversation)

    resumed = session_store.load_session("s")
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


def test_load_drops_unknown_message_fields(tmp_path, monkeypatch) -> None:
    """Stray keys in saved messages (older builds wrote e.g. a message-level
    'kind') are dropped on load instead of raising TypeError in /sessions."""
    import json

    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)
    (tmp_path / "old-session.json").write_text(json.dumps({
        "messages": [
            {"role": "user", "content": "hi", "kind": "job_notification",
             "some_future_field": 1},
            {"role": "assistant", "content": "ho",
             "usage": {"input_tokens": 1, "output_tokens": 2}},
        ],
        "turn_start": 0,
    }), encoding="utf-8")

    loaded = session_store.load_session("old-session")
    assert [m.content for m in loaded.messages] == ["hi", "ho"]
    assert loaded.messages[1].usage == Usage(1, 2)
