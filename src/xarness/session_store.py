"""Save/load Conversation objects as JSON, keyed by session name.

:class:`SessionState` is the whole persisted shape (profile, workspace,
messages, turn marker, git block) in memory; :func:`load_state` parses the
file once and returns it, so callers never re-read per field.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .conversation import Conversation, Message, ToolCall
from .events import Usage

SESSIONS_DIR = Path("~/.local/share/xarness/sessions").expanduser()


def session_path(name: str) -> Path:
    return SESSIONS_DIR / f"{name}.json"


def new_session_name() -> str:
    """Timestamped unique name for a freshly started session."""
    return f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


@dataclass(slots=True)
class SessionState:
    """Everything persisted for one session."""

    conversation: Conversation
    profile: str | None = None
    workspace: str | None = None
    # Change-tracking block (see gitwork.GitInfo.to_block); None drops it.
    git: dict | None = None
    updated_at: str | None = None


# Message's persisted fields; anything else in a saved file (stray keys
# written by older builds, e.g. a message-level "kind") is dropped on load
# instead of raising TypeError deep inside /sessions.
_MESSAGE_FIELDS = {f.name for f in fields(Message)}


def _message_from(raw: dict) -> Message:
    raw = {k: v for k, v in raw.items() if k in _MESSAGE_FIELDS}
    if isinstance(raw.get("usage"), dict):
        raw["usage"] = Usage(**raw["usage"])
    if isinstance(raw.get("tool_calls"), list):
        raw["tool_calls"] = [c if isinstance(c, ToolCall) else ToolCall.from_wire(c) for c in raw["tool_calls"]]
    return Message(**raw)


def _turn_start_from(data: dict, conversation: Conversation) -> int | None:
    """The persisted turn snapshot marker as an index into ``messages``.

    Current sessions store ``turn_start`` directly; older files kept the
    snapshot as a message list (``undo_snapshot``) that is a value-prefix of
    the message list — recover its length and convert."""
    if isinstance(data.get("turn_start"), int):
        return data["turn_start"]
    snapshot = data.get("undo_snapshot")
    if not isinstance(snapshot, list) or len(snapshot) > len(conversation.messages):
        return None
    return len(snapshot) - 1 if all(a == b for a, b in zip(snapshot, conversation.messages)) else None


def _message_dump(message: Message) -> dict:
    dump = {k: v for k, v in asdict(message).items() if v is not None}
    if message.tool_calls is not None:
        dump["tool_calls"] = [call.to_wire() for call in message.tool_calls]
    return dump


def _conversation_from(data: dict[str, Any]) -> Conversation:
    conversation = Conversation()
    for raw in data["messages"]:
        conversation.add(_message_from(raw))
    conversation.turn_start = _turn_start_from(data, conversation)
    if isinstance(data.get("compact_snapshot"), list):
        conversation.compact_snapshot = [_message_from(raw) for raw in data["compact_snapshot"]]
    return conversation


def save_state(name: str, state: SessionState) -> None:
    """Persist a session state as one JSON file."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    conversation = state.conversation
    data: dict[str, Any] = {
        "profile": state.profile,
        "workspace": state.workspace,
        "updated_at": state.updated_at or datetime.now(timezone.utc).isoformat(),
        "messages": [_message_dump(m) for m in conversation.messages],
    }
    if conversation.turn_start is not None:
        data["turn_start"] = conversation.turn_start
    if conversation.compact_snapshot is not None:
        data["compact_snapshot"] = [_message_dump(m) for m in conversation.compact_snapshot]
    if state.git is not None:
        data["git"] = state.git
    session_path(name).write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_session(
    name: str,
    profile_name: str,
    conversation: Conversation,
    git: dict | None = None,
    workspace: str | None = None,
) -> None:
    """Persist a session (convenience wrapper around :func:`save_state`)."""
    save_state(name, SessionState(conversation, profile_name, workspace, git))


def load_state(name: str) -> SessionState:
    """Parse the session file once and return everything in it."""
    data = json.loads(session_path(name).read_text(encoding="utf-8"))
    return SessionState(
        conversation=_conversation_from(data),
        profile=data.get("profile"),
        workspace=data.get("workspace"),
        git=data.get("git") if isinstance(data.get("git"), dict) else None,
        updated_at=data.get("updated_at"),
    )


def load_session(name: str) -> Conversation:
    return load_state(name).conversation


def load_git_block(name: str) -> dict | None:
    """The persisted change-tracking block for a session, or None."""
    try:
        return load_state(name).git
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def list_sessions() -> list[str]:
    if not SESSIONS_DIR.exists():
        return []
    return sorted(p.stem for p in SESSIONS_DIR.glob("*.json"))


def session_meta(name: str) -> dict:
    state = load_state(name)
    return {
        "updated_at": state.updated_at,
        "message_count": len(state.conversation.messages),
        "workspace": state.workspace,
    }


def delete_session(name: str) -> bool:
    p = session_path(name)
    if not p.exists():
        return False
    p.unlink()
    return True
