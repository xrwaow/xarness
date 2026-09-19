"""Save/load Conversation objects as JSON, keyed by session name."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .conversation import Conversation, Message
from .events import Usage

SESSIONS_DIR = Path("~/.local/share/xarness/sessions").expanduser()


def session_path(name: str) -> Path:
    return SESSIONS_DIR / f"{name}.json"


def new_session_name() -> str:
    """Timestamped unique name for a freshly started session."""
    return f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _message_from(raw: dict) -> Message:
    usage = raw.get("usage")
    if isinstance(usage, dict):
        raw["usage"] = Usage(**usage)
    return Message(**raw)


def save_session(
    name: str,
    profile_name: str,
    conversation: Conversation,
    git: dict | None = None,
    workspace: str | None = None,
) -> None:
    """Persist a session. ``git`` is the change-tracking block
    (see gitwork.GitInfo.to_block); passing None drops any existing block."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "profile": profile_name,
        "workspace": workspace,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "messages": [asdict(m) for m in conversation.messages],
    }
    if conversation.undo_snapshot:
        # /undo //retry need the pre-turn conversation even after a resume.
        data["undo_snapshot"] = [asdict(m) for m in conversation.undo_snapshot]
    if conversation.compact_snapshot:
        # /undo needs the pre-compaction history to undo across a compaction.
        data["compact_snapshot"] = [asdict(m) for m in conversation.compact_snapshot]
    if git is not None:
        data["git"] = git
    session_path(name).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_session(name: str) -> Conversation:
    data = json.loads(session_path(name).read_text(encoding="utf-8"))
    conversation = Conversation()
    for raw in data["messages"]:
        conversation.add(_message_from(raw))
    snapshot = data.get("undo_snapshot")
    if isinstance(snapshot, list):
        loaded = [_message_from(raw) for raw in snapshot]
        # rollback_plan matches messages by identity. On disk the snapshot is
        # a value-prefix of the message list (the conversation as it stood
        # when the turn began); re-tie it to the loaded message objects so
        # the identity checks keep working after a resume.
        if len(loaded) <= len(conversation.messages) and all(
            a == b for a, b in zip(loaded, conversation.messages)
        ):
            loaded = conversation.messages[: len(loaded)]
        conversation.undo_snapshot = loaded
    compact = data.get("compact_snapshot")
    if isinstance(compact, list):
        conversation.compact_snapshot = [_message_from(raw) for raw in compact]
    return conversation


def load_git_block(name: str) -> dict | None:
    """The persisted change-tracking block for a session, or None."""
    path = session_path(name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    block = data.get("git")
    return block if isinstance(block, dict) else None


def list_sessions() -> list[str]:
    if not SESSIONS_DIR.exists():
        return []
    return sorted(p.stem for p in SESSIONS_DIR.glob("*.json"))


def session_meta(name: str) -> dict:
    data = json.loads(session_path(name).read_text(encoding="utf-8"))
    return {
        "updated_at": data.get("updated_at"),
        "message_count": len(data.get("messages", [])),
        "workspace": data.get("workspace"),
    }


def delete_session(name: str) -> bool:
    p = session_path(name)
    if not p.exists():
        return False
    p.unlink()
    return True
