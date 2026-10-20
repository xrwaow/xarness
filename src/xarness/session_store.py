"""Save/load Conversation objects as JSON, keyed by session name.

:class:`SessionState` is the whole persisted shape (profile, workspace,
messages, turn marker, git block) in memory; :func:`load_state` parses the
file once and returns it, so callers never re-read per field.

Per-workspace state (the external reference paths managed with ``/refs``)
lives beside the sessions, keyed by the resolved workspace path, so every
session opened in the same directory shares it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .conversation import Conversation, Message, ToolCall, Turn, build_parts
from .images import ImageAttachment, ImageRef
from .protocol import SessionSummary
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
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    path.write_text(
        json.dumps({"workspace": str(workspace), "refs": refs}, indent=2) + "\n",
        encoding="utf-8",
    )


_SAFE_SESSION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")


def _validate_session_name(name: str) -> str:
    """Reject names that could escape the sessions directory."""
    if not isinstance(name, str) or not _SAFE_SESSION_NAME.match(name) or ".." in name:
        raise ValueError(f"invalid session name: {name!r}")
    return name


def session_path(name: str) -> Path:
    return SESSIONS_DIR / f"{_validate_session_name(name)}.json"


def session_dir(name: str) -> Path:
    """Per-session directory holding image blobs."""
    return SESSIONS_DIR / _validate_session_name(name)


def _blob_path(name: str, image_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", image_id or ""):
        raise ValueError(f"invalid image id: {image_id!r}")
    return session_dir(name) / "blobs" / image_id


class SessionImageStore:
    """Content-addressed image bytes for one session.

    ``persist=False`` (a ``no_session`` conversation) keeps the blobs in
    memory: images still work for the provider, nothing touches disk."""

    def __init__(self, name: str, persist: bool = True) -> None:
        self.name = name
        self.persist = persist
        self._memory: dict[str, bytes] = {}

    def save(self, attachment: ImageAttachment) -> ImageRef:
        ref = attachment.to_ref()
        data = attachment.data()
        if self.persist:
            path = _blob_path(self.name, ref.id)
            if not path.exists():
                path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
                tmp = path.with_suffix(f".tmp-{uuid.uuid4().hex}")
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, path)
        else:
            self._memory[ref.id] = data
        return ref

    def load(self, image_id: str) -> bytes | None:
        if not self.persist:
            return self._memory.get(image_id)
        try:
            return _blob_path(self.name, image_id).read_bytes()
        except (OSError, ValueError):
            return None

    def attachment(self, ref: ImageRef) -> ImageAttachment | None:
        data = self.load(ref.id)
        if data is None:
            return None
        return ImageAttachment(
            name=ref.name, mime=ref.mime,
            data_b64=base64.b64encode(data).decode("ascii"),
            width=ref.width, height=ref.height,
        )


def load_blob(name: str, image_id: str) -> bytes | None:
    """Read one stored image blob (saved sessions included)."""
    try:
        return _blob_path(name, image_id).read_bytes()
    except (OSError, ValueError):
        return None


def meta_path(name: str) -> Path:
    """Sidecar holding the small meta block, so listing / last-used lookups
    never parse (or even read) the full session file with its base64
    images."""
    return SESSIONS_DIR / f"{_validate_session_name(name)}.meta.json"


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
    mode: str | None = None
    # Message id the last auto-compaction fired on (loop guard).
    compacted_on: str | None = None


# Persisted-payload schema. Bump when a migration is added to MIGRATIONS.
SCHEMA_VERSION = 3

_TIMING_FIELDS = (
    ("send_time", "send"), ("start_time", "start"), ("end_time", "end"),
)


def _migrate_message(
    raw: dict, store: SessionImageStore | None = None,
) -> dict:
    """Normalize one persisted message to the current shape.

    Idempotent: current-shape messages pass through unchanged. Legacy inline
    ``images`` become ordered parts (and blobs when a store is given); tool
    errors stored as an ``"error: "`` content prefix gain the explicit
    ``ok=False`` field and lose the prefix.
    """
    raw = dict(raw)
    timing = dict(raw.get("timing") or {})
    # The old preformatted timing line is derived client-side now.
    timing.pop("summary", None)
    raw.pop("time_summary", None)
    for old, new in _TIMING_FIELDS:
        if old in raw:
            value = raw.pop(old)
            if value is not None:
                timing[new] = value
    # v2 stored reasoning bounds as ISO stamps; v3 keeps the duration.
    r_start, r_end = raw.pop("reasoning_start", None), raw.pop("reasoning_end", None)
    if r_start and r_end:
        try:
            timing["reasoning_seconds"] = max(0.0, (
                datetime.fromisoformat(r_end) - datetime.fromisoformat(r_start)
            ).total_seconds())
        except ValueError:
            pass
    if timing:
        raw["timing"] = timing
    if "images" in raw:
        images = [
            ImageAttachment.model_validate(item) for item in raw.pop("images") or ()
        ]
        if images and not raw.get("parts"):
            raw["parts"] = [
                part.model_dump()
                for part in build_parts(raw.get("content") or "", images, store)
            ]
    if raw.get("role") == "tool" and isinstance(raw.get("content"), str):
        if raw["content"].startswith("error: "):
            raw["content"] = raw["content"][len("error: "):]
            raw.setdefault("ok", False)
    calls = raw.get("tool_calls")
    if isinstance(calls, list):
        raw["tool_calls"] = [
            ToolCall.from_wire(call).model_dump()
            if isinstance(call, dict) and "function" in call
            else call
            for call in calls
        ]
    return raw


def _migrate_v0_to_v1(
    data: dict[str, Any], store: SessionImageStore | None,
) -> dict:
    """Unversioned legacy files: inline images, tool error prefixes and the
    persisted system prompt (now derived per request)."""
    data = dict(data)
    messages = []
    for raw in data.get("messages", []):
        raw = _migrate_message(raw, store)
        if raw.get("role") == "system":
            continue
        messages.append(raw)
    data["messages"] = messages
    data["schema"] = 1
    return data


def _migrate_v1_to_v2(
    data: dict[str, Any], store: SessionImageStore | None,
) -> dict:
    """Tool calls persist in the internal shape (``call_id``/``name``/
    ``arguments_json``), not the provider wire shape."""
    data = dict(data)
    data["messages"] = [
        _migrate_message(raw, store) for raw in data.get("messages", [])
    ]
    data["schema"] = 2
    return data


def _migrate_v2_to_v3(
    data: dict[str, Any], store: SessionImageStore | None,
) -> dict:
    """Per-message ISO stamps collapse into one ``timing`` sub-model."""
    data = dict(data)
    data["messages"] = [
        _migrate_message(raw, store) for raw in data.get("messages", [])
    ]
    data["schema"] = 3
    return data


# version -> migration to the next version (each one bumps "schema").
MIGRATIONS: dict[int, Any] = {
    0: _migrate_v0_to_v1, 1: _migrate_v1_to_v2, 2: _migrate_v2_to_v3,
}


def _migrate(
    data: dict[str, Any], store: SessionImageStore | None,
) -> dict[str, Any]:
    """Apply the migration chain; reject files from a newer build."""
    try:
        version = int(data.get("schema") or 0)
    except (TypeError, ValueError):
        raise ValueError(f"invalid session schema: {data.get('schema')!r}") from None
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"session schema {version} is newer than supported {SCHEMA_VERSION}"
        )
    while version < SCHEMA_VERSION:
        migration = MIGRATIONS.get(version)
        if migration is None:
            raise ValueError(f"no migration from session schema {version}")
        data = migration(data, store)
        version += 1
    return data


def _message_from(
    raw: dict, store: SessionImageStore | None = None,
) -> Message:
    """Parse one persisted message (migrating it if necessary)."""
    # model_validate handles nested Usage/ToolCall/ImageRef models
    # (Message's tool_calls validator accepts the persisted wire shape).
    return Message.model_validate(_migrate_message(raw, store))


def _message_dump(message: Message) -> dict:
    return message.model_dump(exclude_none=True)


def _migrate_legacy_turns(
    raw_messages: list[dict[str, Any]], parsed: list[Message],
) -> list[Turn]:
    """Reconstruct turns from a pre-``Turn`` session's message fields."""
    turns: list[Turn] = []
    for raw, message in zip(raw_messages, parsed):
        before = raw.get("checkpoint_sha")
        if message.role == "user" and before is not None:
            turns.append(Turn(
                user_message_id=message.id, before_tree=before,
                after_tree=raw.get("after_tree"),
            ))
    return turns


def _migrate_legacy_compaction(
    conversation: Conversation, data: dict[str, Any],
    store: SessionImageStore | None,
) -> None:
    """Merge a legacy ``compact_snapshot`` into the append-only message list,
    splitting it into epochs (the snapshot is the pre-compaction history)."""
    snapshot = data.get("compact_snapshot")
    if not isinstance(snapshot, list):
        return
    pre = [
        _message_from(raw, store) for raw in snapshot
        if raw.get("role") != "system"
    ]
    pre_raw = [raw for raw in snapshot if raw.get("role") != "system"]
    parsed = conversation.messages
    summary_index = next(
        (i for i, m in enumerate(parsed) if m.kind == "summary"), 0,
    )
    post = parsed[summary_index:]
    for message in pre:
        message.epoch = 0
    for message in post:
        message.epoch = 1
    conversation.messages = pre + post
    conversation.epoch = 1
    conversation.turns = _migrate_legacy_turns(pre_raw, pre) + conversation.turns


def _conversation_from(
    data: dict[str, Any], name: str | None = None,
) -> Conversation:
    """Parse a saved conversation, migrating legacy shapes (persisted system
    prompts, message-level checkpoint fields, ``compact_snapshot``, inline
    base64 images) into the append-only epochs + turns model."""
    conversation = Conversation()
    store = SessionImageStore(name) if name is not None else None
    data = _migrate(data, store)
    raw_messages = data["messages"]
    parsed = [_message_from(raw, store) for raw in raw_messages]

    turns_raw = data.get("turns")
    if isinstance(turns_raw, list):
        conversation.turns = [Turn.model_validate(t) for t in turns_raw]
        conversation.epoch = int(data.get("epoch") or 0)
        conversation.messages = parsed
        return conversation

    conversation.turns = _migrate_legacy_turns(raw_messages, parsed)
    conversation.messages = parsed
    _migrate_legacy_compaction(conversation, data, store)
    return conversation


def _meta_block(
    conversation: Conversation,
    *, profile: str | None, workspace: str | None, updated_at: str,
    sandbox_name: str | None, tracking: bool,
) -> dict[str, Any]:
    """The small summary block listing and last-used lookups read: no
    conversation parsing needed on those paths."""
    visible = conversation.visible_messages()
    last_user = max(
        (m.timing.send for m in visible
         if m.role == "user" and m.timing.send),
        default=None,
    )
    return {
        "profile": profile,
        "workspace": workspace,
        "updated_at": updated_at,
        "sandbox_name": sandbox_name,
        "message_count": len(visible),
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
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        # No indent: the file can hold base64 images; compact JSON halves
        # the serialization and write cost of every persist.
        handle.write(json.dumps(data))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def serialize_state(state: SessionState) -> tuple[dict[str, Any], dict[str, Any]]:
    """Snapshot the live state into JSON-ready dicts (call on the event loop).

    Runs without a deepcopy: pydantic dumps copy the data, so the writer
    thread only ever touches plain dicts while the session keeps mutating."""
    conversation = state.conversation
    data: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "profile": state.profile,
        "workspace": state.workspace,
        "updated_at": state.updated_at or _utcnow_iso(),
        "messages": [_message_dump(m) for m in conversation.messages],
        "turns": [turn.model_dump() for turn in conversation.turns],
    }
    if conversation.epoch:
        data["epoch"] = conversation.epoch
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
    if state.mode is not None:
        data["mode"] = state.mode
    if state.compacted_on is not None:
        data["compacted_on"] = state.compacted_on
    meta = _meta_block(
        conversation,
        profile=state.profile,
        workspace=state.workspace,
        updated_at=data["updated_at"],
        sandbox_name=state.sandbox_name,
        tracking=state.git is not None,
    )
    return data, meta


def write_state(name: str, data: dict[str, Any], meta: dict[str, Any]) -> None:
    """Write pre-serialized dicts atomically (safe in a worker thread)."""
    SESSIONS_DIR.mkdir(parents=True, mode=0o700, exist_ok=True)
    _atomic_write_json(session_path(name), data)
    _atomic_write_json(meta_path(name), meta)


def save_state(name: str, state: SessionState) -> None:
    """Serialize and persist a session state atomically."""
    write_state(name, *serialize_state(state))


def load_state(name: str) -> SessionState:
    """Parse the session file once and return everything in it."""
    data = json.loads(session_path(name).read_text(encoding="utf-8"))
    return SessionState(
        conversation=_conversation_from(data, name),
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
        mode=data.get("mode") if data.get("mode") in ("plan", "write") else None,
        compacted_on=data.get("compacted_on"),
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


def saved_summary(name: str) -> SessionSummary:
    """The list row for a session that is only on disk."""
    meta = load_meta(name)
    if meta is None:
        return SessionSummary(name=name, state="saved")
    return SessionSummary(
        name=name, workspace=meta.get("workspace"), state="saved",
        active=False, model=meta.get("profile"),
        sandbox=meta.get("sandbox_name"),
        last_user_message_at=meta.get("last_user_message_at"),
        message_count=meta.get("message_count", 0),
        tracking=bool(meta.get("tracking")),
    )


def delete_session(name: str) -> bool:
    p = session_path(name)
    if not p.exists():
        return False
    p.unlink()
    meta_path(name).unlink(missing_ok=True)
    directory = session_dir(name)
    if directory.is_dir():
        shutil.rmtree(directory)
    return True
