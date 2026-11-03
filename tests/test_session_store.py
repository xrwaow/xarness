"""Tests for session persistence: round-trip of local-only message fields."""

import asyncio

import pytest

import json

from xarness import session_store
from xarness.config import CotStrength, ProviderProfile
from xarness.controller import ChatController
from xarness.conversation import (
    SUMMARY_PREFIX,
    AssistantMessage,
    Conversation,
    Timing,
    ToolCall,
    Usage,
    user_message,
)
from xarness.protocol import ContentDelta, RoundComplete

PROFILE = ProviderProfile(
    base_url="https://api.example.test/v1",
    model_id="test-model",
    max_context=1000,
    cot_strength=CotStrength.MEDIUM,
)


def _save(name: str, stored: session_store.StoredSession) -> None:
    # Turn checkpoints live on the conversation (the one home) and land in
    # the state row at serialize time.
    payload, meta = session_store.serialize_state(stored)
    session_store.write_state(name, payload, meta)


def test_message_fields_round_trip(tmp_path, monkeypatch) -> None:
    """usage / turn checkpoints survive save + load, and stay out of the
    wire format."""
    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)

    conversation = Conversation()
    user = user_message("hi")
    assistant = AssistantMessage(
        content="hello", reasoning="thoughts",
        usage=Usage(input_tokens=12, output_tokens=3),
    )
    conversation.add(user)
    conversation.add(assistant)
    conversation.begin_turn(user, "abc123")
    conversation.turns[user.id].after_tree = "def456"
    _save("s", session_store.StoredSession(profile="m", conversation=conversation))

    loaded = session_store.load_state("s").conversation
    assert [m.role for m in loaded.messages] == ["user", "assistant"]
    assert loaded.messages[1].usage == Usage(input_tokens=12, output_tokens=3)
    assert loaded.messages[1].reasoning == "thoughts"
    turn = loaded.turns[loaded.messages[0].id]
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
    conversation.add(user_message("hi"))
    conversation.add(AssistantMessage(content="hello"))
    _save("s", session_store.StoredSession(
        profile="m", workspace="/w", conversation=conversation,
        usage_total=Usage(input_tokens=1200, output_tokens=340),
        last_usage=Usage(input_tokens=1100, output_tokens=300),
    ))
    loaded = session_store.load_state("s")
    # The model and context accounting a resumed session needs are all here.
    assert loaded.profile == "m" and loaded.workspace == "/w"
    assert loaded.usage_total == Usage(input_tokens=1200, output_tokens=340)
    assert loaded.last_usage == Usage(input_tokens=1100, output_tokens=300)

    # A session saved with no usage yet loads as zeros.
    _save("empty", session_store.StoredSession(profile="m", conversation=conversation))
    empty = session_store.load_state("empty")
    assert (empty.usage_total.input_tokens, empty.usage_total.output_tokens) == (0, 0)
    assert (empty.last_usage.input_tokens, empty.last_usage.output_tokens) == (0, 0)


def test_mode_round_trips(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)
    _save("s", session_store.StoredSession(
        profile="m", conversation=Conversation(), mode="plan",
    ))
    assert session_store.load_state("s").mode == "plan"


def test_to_wire_drops_unpaired_tool_calls() -> None:
    """A tool call without a recorded result (turn interrupted mid-execution,
    session saved) must not reach the wire as an unpaired call: providers
    reject unpaired calls. Repair adds the synthetic result; to_wire then
    sends every message as-is."""

    def make_conversation(content, calls, results):
        conversation = Conversation()
        conversation.add(user_message("go"))
        conversation.add(AssistantMessage(content=content, tool_calls=calls))
        for call_id, output in results:
            conversation.add(
                __import__("xarness.conversation", fromlist=["ToolMessage"]).ToolMessage(
                    tool_call_id=call_id, content=output,
                )
            )
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
    # The synthetic result sits directly after the assistant message that
    # owns the call (adjacency), before the recorded result.
    assert wire[2]["tool_call_id"] == "call_2"
    assert wire[2]["content"] == "interrupted"
    assert wire[3]["tool_call_id"] == "call_1"

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

    client_script = [ContentDelta(text="answer"), RoundComplete(usage=Usage(input_tokens=10, output_tokens=4))]

    class FakeClient:
        async def stream(self, wire_messages, tools=None):
            for event in client_script:
                yield event

    controller = ChatController(PROFILE, "k", client=FakeClient())
    asyncio.run(_send(controller, "hi"))
    _save("s", session_store.StoredSession(profile="m", conversation=controller.conversation))

    resumed = session_store.load_state("s").conversation
    controller2 = ChatController(PROFILE, "k", client=FakeClient())
    controller2.conversation = resumed

    plan = controller2.conversation.rollback_plan()
    assert plan is not None and plan.user_text == "hi"
    controller2.conversation.apply_rollback(plan)
    assert controller2.conversation.messages == []
    assert controller2.usage_total == Usage(input_tokens=0, output_tokens=0)


async def _send(controller: ChatController, text: str) -> None:
    async for _ in controller.begin_turn(text, None, None):
        pass


def test_meta_block_written_and_read_without_parsing_conversation(tmp_path, monkeypatch) -> None:
    from xarness import session_store as store

    monkeypatch.setattr(store, "SESSIONS_DIR", tmp_path / "sessions")
    conversation = Conversation()
    conversation.add(user_message("q", timing=Timing(send="2026-01-01T00:00:00+00:00")))
    conversation.add(AssistantMessage(content="a"))
    _save("s", store.StoredSession(
        profile="p", workspace="/w",
        git=__import__("xarness.gitwork", fromlist=["GitInfo"]).GitInfo(
            session_id="s", repo_root="/w", git_dir="/w/.git",
        ),
        sandbox_name="sb", conversation=conversation,
    ))
    meta = store.load_meta("s")
    assert meta["workspace"] == "/w"
    assert meta["message_count"] == 2
    assert meta["last_user_message_at"] == "2026-01-01T00:00:00+00:00"
    assert meta["tracking"] is True
    assert meta["profile"] == "p" and meta["sandbox_name"] == "sb"

    # Reading the meta must not build Conversation objects (the whole point:
    # listing and last-used lookups stay cheap).
    def explode(*args, **kwargs):
        raise AssertionError("full conversation parsed")
    monkeypatch.setattr(store, "load_state", explode)
    meta = store.load_meta("s")
    assert meta["workspace"] == "/w" and meta["message_count"] == 2
    assert meta["last_user_message_at"] == "2026-01-01T00:00:00+00:00"
    assert store.load_meta("s")["workspace"] == "/w"


def test_saved_sessions_carry_a_schema_version(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)
    conversation = Conversation()
    conversation.add(user_message("hi"))
    _save("s", session_store.StoredSession(conversation=conversation))
    # The version lives in its own column (not duplicated in meta).
    with session_store.connection() as db:
        (schema,) = db.execute(
            "SELECT schema FROM sessions WHERE name = 's'").fetchone()
    assert schema == session_store.SCHEMA_VERSION
    assert session_store.load_state("s").schema_version == session_store.SCHEMA_VERSION


def test_tool_calls_persist_in_internal_shape(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)
    conversation = Conversation()
    conversation.add(AssistantMessage(
        content="",
        tool_calls=[ToolCall(call_id="c1", name="run_bash", arguments_json="{}")],
    ))
    _save("s", session_store.StoredSession(conversation=conversation))
    loaded = session_store.load_state("s").conversation.messages[0]
    assert loaded.tool_calls == [
        ToolCall(call_id="c1", name="run_bash", arguments_json="{}"),
    ]


def test_unsupported_schemas_are_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)

    def plant(name: str, schema: int) -> None:
        import sqlite3
        connection = sqlite3.connect(tmp_path / "sessions.db")
        connection.executescript(
            "CREATE TABLE IF NOT EXISTS sessions ("
            "name TEXT PRIMARY KEY, schema INTEGER NOT NULL, state TEXT NOT NULL,"
            " meta TEXT NOT NULL, updated_at TEXT NOT NULL);"
        )
        with connection:
            connection.execute(
                "INSERT INTO sessions (name, schema, state, meta, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (name, schema, "{}", "{}", ""),
            )
        connection.close()

    plant("future", session_store.SCHEMA_VERSION + 1)
    with pytest.raises(ValueError, match="not supported"):
        session_store.load_state("future")

    plant("old", session_store.SCHEMA_VERSION - 1)
    with pytest.raises(ValueError, match="not supported"):
        session_store.load_state("old")


def test_meta_written_and_removed(tmp_path, monkeypatch) -> None:
    """Writing folds metadata into the per-session meta file; deleting the
    session drops its directory."""
    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)
    conversation = Conversation()
    conversation.add(user_message("q", timing=Timing(send="2026-01-01T00:00:00+00:00")))
    _save("s", session_store.StoredSession(
        profile="m", workspace="/w", conversation=conversation,
    ))

    meta = session_store.load_meta("s")
    assert meta["workspace"] == "/w" and meta["message_count"] == 1
    assert list(session_store.load_metas()) == ["s"]

    assert session_store.delete_session("s") is True
    assert session_store.load_metas() == {}


def test_summary_message_round_trips(tmp_path, monkeypatch) -> None:
    """A compacted session keeps its summary marker and its hidden history."""
    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)
    conversation = Conversation()
    conversation.add(user_message("before"))
    conversation.add(AssistantMessage(content="answer"))
    conversation.add(user_message(f"{SUMMARY_PREFIX}\n\nhandoff", is_summary=True))
    _save("s", session_store.StoredSession(profile="m", conversation=conversation))
    loaded = session_store.load_state("s").conversation
    assert [m.role for m in loaded.history_messages()] == ["user", "assistant"]
    assert [m.role for m in loaded.visible_messages()] == ["user"]
    assert loaded.visible_messages()[0].is_summary is True


def test_watermark_write_replaces_truncated_rows(tmp_path, monkeypatch) -> None:
    """A rollback moves the persist watermark back: the rows past the cut
    are replaced on disk, not left behind by an append."""
    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)
    conversation = Conversation()
    first = conversation.add(user_message("one"))
    conversation.begin_turn(first, None)
    conversation.add(AssistantMessage(content="a"))
    second = conversation.add(user_message("two"))
    conversation.begin_turn(second, None)
    conversation.add(AssistantMessage(content="b"))
    stored = session_store.StoredSession(profile="m", conversation=conversation)
    session_store.write_state("s", *session_store.serialize_state(stored))
    assert [m.content for m in session_store.load_state("s").conversation.messages] == [
        "one", "a", "two", "b",
    ]

    # Truncate to the first turn and rewrite from the watermark.
    plan = conversation.rollback_plan_to(second)
    conversation.apply_rollback(plan)
    payload, meta = session_store.serialize_state(stored, from_seq=plan.drop_from)
    assert payload["from_seq"] == plan.drop_from
    session_store.write_state("s", payload, meta)
    assert [m.content for m in session_store.load_state("s").conversation.messages] == [
        "one", "a",
    ]
    # And the pruned turn is gone from the state row.
    assert second.id not in session_store.load_state("s").conversation.turns


def test_turns_persist_in_the_state_row_not_the_messages(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)
    conversation = Conversation()
    user = conversation.add(user_message("hi"))
    conversation.begin_turn(user, "abc")
    stored = session_store.StoredSession(profile="m", conversation=conversation)
    session_store.write_state("s", *session_store.serialize_state(stored))
    # The message row carries no checkpoint...
    with session_store.connection() as db:
        (row,) = db.execute(
            "SELECT data FROM messages WHERE name = 's'").fetchone()
    assert "turn" not in json.loads(row)
    # ...the state row does, and a reload restores it.
    loaded = session_store.load_state("s")
    turn = loaded.conversation.turns[user.id]
    assert turn.before_tree == "abc"
    assert turn.closed is False


def test_change_workspace_ref_adds_and_removes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(session_store, "WORKSPACES_DIR", tmp_path / "ws")
    notes = tmp_path / "notes"
    notes.mkdir()
    specs, host = session_store.change_workspace_ref(tmp_path, str(notes), add=True)
    assert specs == [str(notes.resolve())]
    # Adding twice is idempotent (no duplicate).
    specs, _ = session_store.change_workspace_ref(tmp_path, "notes", add=True)
    assert specs == [str(notes.resolve())]
    specs, _ = session_store.change_workspace_ref(tmp_path, "notes", add=False)
    assert specs == []
    with pytest.raises(session_store.RefError):
        session_store.change_workspace_ref(tmp_path, "missing", add=True)
    with pytest.raises(session_store.RefError):
        session_store.change_workspace_ref(tmp_path, "notes", add=False)
