"""Save/load Conversation objects as JSON, keyed by session name."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .conversation import Conversation, Message

SESSIONS_DIR = Path("~/.local/share/xarness/sessions").expanduser()


def session_path(name: str) -> Path:
    return SESSIONS_DIR / f"{name}.json"


def new_session_name() -> str:
    """Timestamped unique name for a freshly started session."""
    return f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def save_session(
    name: str,
    profile_name: str,
    conversation: Conversation,
    git: dict | None = None,
) -> None:
    """Persist a session. ``git`` is the harness-managed worktree block
    (see gitwork.GitInfo.to_block); passing None drops any existing block —
    which is exactly what accept/reject want after resolving a session."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "profile": profile_name,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "messages": [asdict(m) for m in conversation.messages],
    }
    if git is not None:
        data["git"] = git
    session_path(name).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_session(name: str) -> Conversation:
    data = json.loads(session_path(name).read_text(encoding="utf-8"))
    conversation = Conversation()
    for raw in data["messages"]:
        conversation.add(Message(**raw))
    return conversation


def load_git_block(name: str) -> dict | None:
    """The persisted worktree block for a session, or None."""
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
    return {"updated_at": data.get("updated_at"), "message_count": len(data.get("messages", []))}


def delete_session(name: str) -> bool:
    p = session_path(name)
    if not p.exists():
        return False
    p.unlink()
    return True
