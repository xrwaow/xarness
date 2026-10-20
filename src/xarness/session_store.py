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
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .conversation import Conversation, Message, Turn
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


def index_path() -> Path:
    """One small file holding every session's listing metadata."""
    return SESSIONS_DIR / "index.json"


_INDEX_VERSION = 1
_INDEX_LOCK = threading.Lock()


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


# Persisted-payload schema. Sessions from any other version are rejected:
# this build reads and writes exactly one shape.
SCHEMA_VERSION = 3


def _check_schema(data: dict[str, Any]) -> None:
    """Reject session files this build cannot read."""
    try:
        version = int(data.get("schema"))
    except (TypeError, ValueError):
        raise ValueError(
            f"invalid session schema: {data.get('schema')!r}"
        ) from None
    if version < SCHEMA_VERSION:
        raise ValueError(
            f"session schema {version} is no longer supported "
            f"(current {SCHEMA_VERSION}); start a new session"
        )
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"session schema {version} is newer than supported {SCHEMA_VERSION}"
        )


class StoredSession(BaseModel):
    """The on-disk session shape: the one place its field names live.

    ``schema_version`` is serialized as ``schema``. Optional keys are omitted
    when empty, so the file stays as small as the old hand-built dict."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    schema_version: int = Field(alias="schema")
    updated_at: str
    messages: list[Message]
    turns: list[Turn]
    profile: str | None = None
    workspace: str | None = None
    epoch: int = 0
    git: Any = None
    sandbox_name: str | None = None
    sandbox_settings: Any = None
    total_in: int = 0
    total_out: int = 0
    last_in: int = 0
    last_out: int = 0
    mode: str | None = None
    compacted_on: str | None = None

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> "StoredSession":
        _check_schema(data)
        return cls.model_validate(data)

    @classmethod
    def from_state(cls, state: SessionState) -> "StoredSession":
        conversation = state.conversation
        return cls(
            schema_version=SCHEMA_VERSION,
            updated_at=state.updated_at or _utcnow_iso(),
            messages=conversation.messages,
            turns=conversation.turns,
            epoch=conversation.epoch,
            profile=state.profile,
            workspace=state.workspace,
            git=state.git,
            sandbox_name=state.sandbox_name,
            sandbox_settings=state.sandbox_settings,
            total_in=state.total_in,
            total_out=state.total_out,
            last_in=state.last_in,
            last_out=state.last_out,
            mode=state.mode,
            compacted_on=state.compacted_on,
        )

    def to_state(self) -> SessionState:
        return SessionState(
            conversation=self.conversation,
            profile=self.profile,
            workspace=self.workspace,
            git=self.git if isinstance(self.git, dict) else None,
            updated_at=self.updated_at,
            sandbox_name=self.sandbox_name,
            sandbox_settings=self.sandbox_settings
            if isinstance(self.sandbox_settings, dict) else None,
            total_in=self.total_in,
            total_out=self.total_out,
            last_in=self.last_in,
            last_out=self.last_out,
            mode=self.mode if self.mode in ("plan", "write") else None,
            compacted_on=self.compacted_on,
        )

    def to_data(self) -> dict[str, Any]:
        data = self.model_dump(
            by_alias=True, exclude_none=True, exclude_defaults=True,
            exclude={"messages", "turns"},
        )
        data["messages"] = [
            message.model_dump(exclude_none=True) for message in self.messages
        ]
        data["turns"] = [turn.model_dump() for turn in self.turns]
        return data

    @property
    def conversation(self) -> Conversation:
        return Conversation(
            messages=self.messages, turns=self.turns, epoch=self.epoch,
        )


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
    stored = StoredSession.from_state(state)
    meta = _meta_block(
        state.conversation,
        profile=state.profile,
        workspace=state.workspace,
        updated_at=stored.updated_at,
        sandbox_name=state.sandbox_name,
        tracking=state.git is not None,
    )
    return stored.to_data(), meta


def _seal_meta(name: str, meta: dict[str, Any]) -> dict[str, Any]:
    """Attach the session file's size/mtime so a stale index entry is
    detectable without parsing the whole file."""
    sealed = dict(meta)
    try:
        stat = session_path(name).stat()
    except OSError:
        return sealed
    sealed["_file"] = [stat.st_size, stat.st_mtime_ns]
    return sealed


def _entry_fresh(name: str, meta: dict[str, Any]) -> bool:
    stamp = meta.get("_file")
    if not (isinstance(stamp, list) and len(stamp) == 2):
        return False
    try:
        stat = session_path(name).stat()
    except OSError:
        return False
    return [stat.st_size, stat.st_mtime_ns] == stamp


def _read_index_unlocked() -> dict[str, dict[str, Any]]:
    """The index's session → meta map, rebuilding it once when absent."""
    try:
        raw = json.loads(index_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = None
    if isinstance(raw, dict) and isinstance(raw.get("sessions"), dict):
        return raw["sessions"]
    return _rebuild_index_unlocked()


def _write_index_unlocked(sessions: dict[str, dict[str, Any]]) -> None:
    SESSIONS_DIR.mkdir(parents=True, mode=0o700, exist_ok=True)
    _atomic_write_json(
        index_path(), {"version": _INDEX_VERSION, "sessions": sessions},
    )


def _meta_from_file(name: str) -> dict[str, Any] | None:
    """Rebuild one session's meta from its file."""
    try:
        state = load_state(name)
    except (OSError, ValueError, json.JSONDecodeError, KeyError, ValidationError):
        return None
    return _meta_block(
        state.conversation,
        profile=state.profile,
        workspace=state.workspace,
        updated_at=state.updated_at or "",
        sandbox_name=state.sandbox_name,
        tracking=state.git is not None,
    )


def _rebuild_index_unlocked() -> dict[str, dict[str, Any]]:
    """Rebuild the index from the session files (once)."""
    sessions: dict[str, dict[str, Any]] = {}
    if SESSIONS_DIR.exists():
        for path in SESSIONS_DIR.glob("*.json"):
            if path.name == "index.json" or path.name.endswith(".meta.json"):
                continue
            meta = _meta_from_file(path.stem)
            if meta is not None:
                sessions[path.stem] = _seal_meta(path.stem, meta)
    _write_index_unlocked(sessions)
    return sessions


def load_metas() -> dict[str, dict[str, Any]]:
    """All listing metadata from the single index (one file read), healing
    any entry whose session file changed since the index was written."""
    with _INDEX_LOCK:
        sessions = _read_index_unlocked()
        if any(not _entry_fresh(name, meta) for name, meta in sessions.items()):
            sessions = _rebuild_index_unlocked()
        return dict(sessions)


def write_state(name: str, data: dict[str, Any], meta: dict[str, Any]) -> None:
    """Write the session and fold its meta into the single index."""
    SESSIONS_DIR.mkdir(parents=True, mode=0o700, exist_ok=True)
    _atomic_write_json(session_path(name), data)
    sealed = _seal_meta(name, meta)
    with _INDEX_LOCK:
        sessions = _read_index_unlocked()
        sessions[name] = sealed
        _write_index_unlocked(sessions)


def save_state(name: str, state: SessionState) -> None:
    """Serialize and persist a session state atomically."""
    write_state(name, *serialize_state(state))


def load_state(name: str) -> SessionState:
    """Parse the session file once and return everything in it."""
    data = json.loads(session_path(name).read_text(encoding="utf-8"))
    return StoredSession.from_data(data).to_state()


def list_sessions() -> list[str]:
    return sorted(load_metas())


def load_meta(name: str) -> dict[str, Any] | None:
    """One saved session's small meta block from the shared index."""
    with _INDEX_LOCK:
        sessions = _read_index_unlocked()
        meta = sessions.get(name)
        # An index that predates the file, or whose file changed since the
        # index was sealed, heals from the session itself.
        if meta is None or not _entry_fresh(name, meta):
            sessions = _rebuild_index_unlocked()
            meta = sessions.get(name)
        return meta


def saved_summary(
    name: str, meta: dict[str, Any] | None = None,
) -> SessionSummary:
    """The list row for a session that is only on disk."""
    if meta is None:
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
    with _INDEX_LOCK:
        sessions = _read_index_unlocked()
        sessions.pop(name, None)
        _write_index_unlocked(sessions)
    p.unlink(missing_ok=True)
    directory = session_dir(name)
    if directory.is_dir():
        shutil.rmtree(directory)
    return True
