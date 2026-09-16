"""Tests for session persistence: round-trip of local-only message fields."""

import asyncio

from xarness import session_store
from xarness.conversation import Conversation, Message
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
    """usage / checkpoint_sha / undo_snapshot survive save + load, and stay
    out of the wire format."""
    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)

    conversation = Conversation()
    user = Message(role="user", content="hi", checkpoint_sha="abc123")
    assistant = Message(
        role="assistant", content="hello", reasoning="thoughts", usage=Usage(12, 3)
    )
    conversation.add(user)
    conversation.add(assistant)
    conversation.undo_snapshot = [user]
    session_store.save_session("s", "m", conversation)

    loaded = session_store.load_session("s")
    assert [m.role for m in loaded.messages] == ["user", "assistant"]
    assert loaded.messages[0].checkpoint_sha == "abc123"
    assert loaded.messages[1].usage == Usage(12, 3)
    assert loaded.messages[1].reasoning == "thoughts"
    assert loaded.undo_snapshot is not None
    assert loaded.undo_snapshot[0].checkpoint_sha == "abc123"

    # Local-only fields never reach the wire.
    assert all("usage" not in m and "checkpoint_sha" not in m for m in loaded.to_wire())


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
