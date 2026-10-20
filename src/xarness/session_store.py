"""Save/load Conversation objects as JSON, keyed by session name.

:class:`SessionState` is the whole persisted shape (profile, workspace,
messages, turn marker, git block) in memory; :func:`load_state` parses the
file once and returns it, so callers never re-read per field.

Per-workspace state (the external reference paths managed with ``/refs``)
lives beside the sessions, keyed by the resolved workspace path, so every
session opened in the same directory shares it.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .conversation import Conversation, Message
from .util import utcnow_iso as _utcnow_iso

SESSIONS_DIR = Path("~/.local/share/xarness/sessions").expanduser()
WORKSPACES_DIR = Path("~/.local/share/xarness/workspaces").expanduser()


def _workspace_path(workspace: str) -> Path:
    """Settings file for a workspace, keyed by its resolved path."""
    key = hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()[:16]
    return WORKSPACES_DIR / f"{key}.json"


def load_workspace_refs(workspace: str) -> list[str]:
    """Reference paths saved for this workspace (shared by its sessions)."""
    try:
        data = json.loads(_workspace_path(workspace).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    refs = data.get("refs")
    if not isinstance(refs, list):
        return []
    return [ref for ref in refs if isinstance(ref, str) and ref.strip()]


def save_workspace_refs(workspace: str, refs: list[str]) -> None:
    """Persist the workspace's refs; an empty list drops the file."""
    refs = [ref for ref in refs if isinstance(ref, str) and ref.strip()]
    path = _workspace_path(workspace)
    if not refs:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"workspace": str(workspace), "refs": refs}, indent=2) + "\n",
        encoding="utf-8",
    )


def session_path(name: str) -> Path:
    return SESSIONS_DIR / f"{name}.json"


def meta_path(name: str) -> Path:
    """Sidecar holding the small meta block, so listing / last-used lookups
    never parse (or even read) the full session file with its base64
    images."""
    return SESSIONS_DIR / f"{name}.meta.json"


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
    # Per-session sandbox settings (resolved at creation from the config's
    # sandbox profiles; fixed except via /sandbox or /filter).
    sandbox_name: str | None = None
    sandbox_settings: dict | None = None
    # Cumulative (total_in/total_out) and last-round (last_in/last_out) token
    # usage, so a reloaded session's footer shows real numbers immediately
    # instead of 0 until the next round completes.
    total_in: int = 0
    total_out: int = 0
    last_in: int = 0
    last_out: int = 0


def _message_from(raw: dict) -> Message:
    # model_validate handles nested Usage/ToolCall/ImageAttachment models
    # (Message's tool_calls validator accepts the persisted wire shape).
    return Message.model_validate(raw)


def _message_dump(message: Message) -> dict:
    dump = message.model_dump(exclude_none=True)
    if message.tool_calls is not None:
        dump["tool_calls"] = [call.to_wire() for call in message.tool_calls]
    return dump


def _conversation_from(data: dict[str, Any]) -> Conversation:
    conversation = Conversation()
    for raw in data["messages"]:
        conversation.add(_message_from(raw))
    conversation.turn_start = data.get("turn_start")
    if isinstance(data.get("compact_snapshot"), list):
        conversation.compact_snapshot = [_message_from(raw) for raw in data["compact_snapshot"]]
    return conversation


def _meta_block(
    conversation: Conversation,
    *, profile: str | None, workspace: str | None, updated_at: str,
    sandbox_name: str | None, tracking: bool,
) -> dict[str, Any]:
    """The small summary block listing and last-used lookups read: no
    conversation parsing needed on those paths."""
    last_user = max(
        (m.send_time for m in conversation.messages
         if m.role == "user" and m.send_time),
        default=None,
    )
    return {
        "profile": profile,
        "workspace": workspace,
        "updated_at": updated_at,
        "sandbox_name": sandbox_name,
        "message_count": len(conversation.messages),
        "last_user_message_at": last_user,
        "tracking": tracking,
    }


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` as JSON atomically: unique tmp file in the same
    directory, fsync, then ``os.replace`` over the target, so a crash
    mid-write never leaves a truncated file behind (a reader sees either
    the old or the new state)."""
    # Unique tmp name: concurrent writers must not clobber each other
    # before the os.replace.
    tmp = path.with_suffix(f".json.tmp-{uuid.uuid4().hex}")
    with open(tmp, "w", encoding="utf-8") as handle:
        # No indent: the file can hold base64 images; compact JSON halves
        # the serialization and write cost of every persist.
        handle.write(json.dumps(data))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def save_state(name: str, state: SessionState) -> None:
    """Persist a session state as one JSON file, atomically.

    Written to a temp file in the same directory, fsynced, then ``os.replace``d
    over the target, so a crash mid-write never leaves a truncated session
    file behind (a reader would see either the old or the new state)."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    conversation = state.conversation
    data: dict[str, Any] = {
        "profile": state.profile,
        "workspace": state.workspace,
        "updated_at": state.updated_at or _utcnow_iso(),
        "messages": [_message_dump(m) for m in conversation.messages],
    }
    if conversation.turn_start is not None:
        data["turn_start"] = conversation.turn_start
    if conversation.compact_snapshot is not None:
        data["compact_snapshot"] = [_message_dump(m) for m in conversation.compact_snapshot]
    if state.git is not None:
        data["git"] = state.git
    if state.sandbox_name is not None:
        data["sandbox_name"] = state.sandbox_name
    if state.sandbox_settings is not None:
        data["sandbox_settings"] = state.sandbox_settings
    if state.total_in or state.total_out:
        data["total_in"] = state.total_in
        data["total_out"] = state.total_out
    if state.last_in or state.last_out:
        data["last_in"] = state.last_in
        data["last_out"] = state.last_out
    _atomic_write_json(session_path(name), data)
    _atomic_write_json(meta_path(name), _meta_block(
        conversation,
        profile=state.profile,
        workspace=state.workspace,
        updated_at=data["updated_at"],
        sandbox_name=state.sandbox_name,
        tracking=state.git is not None,
    ))


def load_state(name: str) -> SessionState:
    """Parse the session file once and return everything in it."""
    data = json.loads(session_path(name).read_text(encoding="utf-8"))
    return SessionState(
        conversation=_conversation_from(data),
        profile=data.get("profile"),
        workspace=data.get("workspace"),
        git=data.get("git") if isinstance(data.get("git"), dict) else None,
        updated_at=data.get("updated_at"),
        sandbox_name=data.get("sandbox_name"),
        sandbox_settings=data.get("sandbox_settings")
        if isinstance(data.get("sandbox_settings"), dict) else None,
        total_in=data.get("total_in") or 0,
        total_out=data.get("total_out") or 0,
        last_in=data.get("last_in") or 0,
        last_out=data.get("last_out") or 0,
    )


def list_sessions() -> list[str]:
    if not SESSIONS_DIR.exists():
        return []
    return sorted(
        p.stem for p in SESSIONS_DIR.glob("*.json")
        if not p.stem.endswith(".meta")
    )


def load_meta(name: str) -> dict[str, Any] | None:
    """The saved session's small meta block (no conversation parsing).

    The sidecar is the only live copy; when it is missing (older save, or an
    interrupted write) the meta block is rebuilt once from the full session
    file."""
    try:
        meta = json.loads(meta_path(name).read_text(encoding="utf-8"))
        if isinstance(meta, dict):
            return meta
    except (OSError, json.JSONDecodeError):
        pass
    try:
        state = load_state(name)
    except (OSError, json.JSONDecodeError, KeyError, ValidationError):
        return None
    return _meta_block(
        state.conversation,
        profile=state.profile,
        workspace=state.workspace,
        updated_at=state.updated_at or "",
        sandbox_name=state.sandbox_name,
        tracking=state.git is not None,
    )


def delete_session(name: str) -> bool:
    p = session_path(name)
    if not p.exists():
        return False
    p.unlink()
    meta_path(name).unlink(missing_ok=True)
    return True
