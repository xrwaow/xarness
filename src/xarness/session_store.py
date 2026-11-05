"""Save/load sessions, keyed by session name.

One :class:`StoredSession` (pydantic) is the whole persisted shape — the
single place its field names live. Sessions live in one SQLite database
(``sessions.db`` inside :data:`SESSIONS_DIR`): the session row carries the
state (including the turn checkpoints) and the cheap listing metadata, and
the conversation is a ``messages`` table — messages are immutable, so a
write only ever has to replace rows from a watermark onward
(``DELETE ... WHERE seq >= from_seq``, then insert). Nothing
re-serializes O(history) on every tool result. Image blobs stay
content-addressed files under the session directory.

Per-workspace state (the external reference paths managed with ``/refs``)
lives beside the sessions as small JSON files, keyed by the resolved
workspace path, so every session opened in the same directory shares it.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .config import SandboxSettings
from .errors import RefError
from .conversation import MESSAGE_ADAPTER, Conversation, Message, Usage
from .gitwork import GitInfo, TurnCheckpoint
from .images import ImageAttachment, ImageRef
from .util import utcnow_iso as _utcnow_iso

SESSIONS_DIR = Path("~/.local/share/xarness/sessions").expanduser()
WORKSPACES_DIR = Path("~/.local/share/xarness/workspaces").expanduser()

# Persisted-payload schema. Sessions from any other version are rejected:
# this build reads and writes exactly one shape.
SCHEMA_VERSION = 4


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


def change_workspace_ref(
    workspace: Path, raw: str, add: bool, *, save: bool = True,
) -> tuple[list[str], Path]:
    """Add/remove one workspace reference path — the ONE implementation,
    used by the session's /ref command and the host's workspace endpoint.

    Returns the canonical (saved) list and the resolved host path. With
    ``save=False`` the caller persists the returned list itself (the
    session tolerates a read-only data dir; the host does not).

    Raises :class:`RefError` when the path does not exist (a typo must
    never mkdir a reference) or is not a known reference."""
    workspace = Path(workspace).expanduser().resolve()
    raw = raw.strip()
    if not raw:
        raise RefError("references need a path")
    path = Path(raw).expanduser()
    host = (path if path.is_absolute() else workspace / path).resolve()
    specs = list(load_workspace_refs(str(workspace)))
    if add:
        if not host.exists():
            raise RefError(f"no such path: {raw}")
        if str(host) not in specs:
            specs.append(str(host))
    else:
        kept = [
            ref for ref in specs
            if Path(ref).expanduser().resolve() != host
        ]
        if len(kept) == len(specs):
            raise RefError(f"no such reference: {raw}")
        specs = kept
    if save:
        save_workspace_refs(str(workspace), specs)
    return specs, host


_SAFE_SESSION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")


def _validate_session_name(name: str) -> str:
    """Reject names that could escape the sessions directory."""
    if not isinstance(name, str) or not _SAFE_SESSION_NAME.match(name) or ".." in name:
        raise ValueError(f"invalid session name: {name!r}")
    return name


def session_dir(name: str) -> Path:
    """Per-session directory: the image blobs."""
    return SESSIONS_DIR / _validate_session_name(name)


def _blob_path(name: str, image_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", image_id or ""):
        raise ValueError(f"invalid image id: {image_id!r}")
    return session_dir(name) / "blobs" / image_id


class SessionImageStore:
    """Content-addressed image bytes for one session.

    ``persist=False`` (a ``no_session`` conversation) keeps the blobs in
    memory: images still work for the provider, nothing touches disk."""

    # data-URL cache bound: to_wire re-encodes the same blobs every round
    # otherwise; the bytes never change once stored.
    URL_CACHE_LIMIT = 64

    def __init__(self, name: str, persist: bool = True) -> None:
        self.name = name
        self.persist = persist
        self._memory: dict[str, bytes] = {}
        self._url_cache: "OrderedDict[str, str]" = OrderedDict()

    def save(self, attachment: ImageAttachment) -> ImageRef:
        """Store one attachment's bytes; returns its content-addressed
        ref (the attachment already carries the id — the hash was computed
        once, on load)."""
        self._write(attachment.id, attachment.data())
        return ImageRef(
            id=attachment.id, name=attachment.name, mime=attachment.mime,
            width=attachment.width, height=attachment.height,
        )

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

    def store(self, data: bytes) -> str:
        """Store raw uploaded bytes under their content hash; returns the id.

        The one write path for blobs — client uploads use it (and a
        ``no_save`` session keeps them in memory, where the provider round
        can still reach them)."""
        image_id = hashlib.sha256(data).hexdigest()
        self._write(image_id, data)
        return image_id

    def _write(self, image_id: str, data: bytes) -> None:
        if self.persist:
            path = _blob_path(self.name, image_id)
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
            self._memory[image_id] = data

    def data_url(self, ref: ImageRef) -> str | None:
        """``data:<mime>;base64,...`` for one stored image, cached by id."""
        url = self._url_cache.get(ref.id)
        if url is None:
            data = self.load(ref.id)
            if data is None:
                return None
            url = f"data:{ref.mime};base64,{base64.b64encode(data).decode('ascii')}"
            self._url_cache[ref.id] = url
            while len(self._url_cache) > self.URL_CACHE_LIMIT:
                self._url_cache.popitem(last=False)
        else:
            self._url_cache.move_to_end(ref.id)
        return url


def load_blob(name: str, image_id: str) -> bytes | None:
    """Read one stored image blob straight from disk (saved sessions included,
    hosted or not)."""
    try:
        return _blob_path(name, image_id).read_bytes()
    except (OSError, ValueError):
        return None


def new_session_name() -> str:
    """Timestamped unique name for a freshly started session."""
    return f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


class StoredSession(BaseModel):
    """The persisted session shape: the one place its field names live.

    ``schema_version`` is serialized as ``schema`` (the sessions table's
    schema column). Optional keys are omitted when empty, so the row stays
    small."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = Field(alias="schema", default=SCHEMA_VERSION)
    updated_at: str = Field(default_factory=_utcnow_iso)
    conversation: Conversation = Field(default_factory=Conversation)
    # Turn checkpoints, keyed by the opening user message's id. ONE home at
    # runtime (Conversation.turns); this is the persisted shape of it, a
    # real field so neither serialize nor load has to splice the dict in by
    # hand. They live in the state row (not on the messages): messages are
    # immutable, and these mutate at every turn end / accept / rollback.
    turns: dict[str, TurnCheckpoint] = Field(default_factory=dict)
    profile: str | None = None
    workspace: str | None = None
    git: GitInfo | None = None
    # Per-session sandbox settings (resolved at creation from the config's
    # sandbox profiles; fixed except via /sandbox or /filter).
    sandbox_name: str | None = None
    sandbox_settings: SandboxSettings | None = None
    usage_total: Usage = Field(default_factory=Usage)
    last_usage: Usage = Field(default_factory=Usage)
    mode: str | None = None


def _meta_block(stored: StoredSession, tracking: bool) -> dict[str, Any]:
    """The small summary block listing and last-used lookups read: no
    conversation parsing needed on those paths.

    ``updated_at``/``schema`` are NOT duplicated here: the sessions table
    has a column for each, and the loaders inject them back."""
    visible = stored.conversation.visible_messages()
    return {
        "profile": stored.profile,
        "workspace": stored.workspace,
        "sandbox_name": stored.sandbox_name,
        "message_count": len(visible),
        "last_user_message_at": stored.conversation.last_user_message_at(),
        "tracking": tracking,
    }


def _db_path() -> Path:
    return SESSIONS_DIR / "sessions.db"


_local = threading.local()


def _connect() -> sqlite3.Connection:
    """A per-thread cached connection (keyed by the database path, so tests
    that redirect SESSIONS_DIR get a fresh one).

    The persist writes come from worker threads (to_thread reuses its pool
    threads), listing reads from the event loop; one connection per thread
    avoids cross-thread check_same_thread issues entirely and skips the
    per-operation PRAGMAs/DDL. WAL keeps a reader and the writer from
    blocking each other."""
    path = str(_db_path())
    cached = getattr(_local, "connection", None)
    if cached is not None and _local.path == path:
        return cached
    if cached is not None:
        cached.close()
    SESSIONS_DIR.mkdir(parents=True, mode=0o700, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=10000")
    _ensure_schema(connection)
    _local.connection = connection
    _local.path = path
    return connection


@contextlib.contextmanager
def connection() -> "sqlite3.Connection":
    """The thread's cached connection. An open transaction is committed on a
    clean exit (writers that do not manage their own ``with db`` block) and
    rolled back on error, so the next use starts clean."""
    db = _connect()
    try:
        yield db
    except BaseException:
        if db.in_transaction:
            db.rollback()
        raise
    else:
        if db.in_transaction:
            db.commit()


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            name TEXT PRIMARY KEY,
            schema INTEGER NOT NULL,
            state TEXT NOT NULL,
            meta TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            name TEXT NOT NULL,
            seq INTEGER NOT NULL,
            data TEXT NOT NULL,
            PRIMARY KEY (name, seq)
        );
        """
    )


def _state_block(stored: StoredSession) -> dict[str, Any]:
    """Everything but the messages (and the redundant schema/version key).

    Dumping only the small fields — never the whole model — keeps a persist
    O(state), not O(history). ``turns`` comes from the conversation, the one
    runtime home (``StoredSession.turns`` is its persisted shape)."""
    state = stored.model_dump(
        mode="json", by_alias=True, exclude_none=True,
        exclude={"conversation", "schema_version", "turns"},
    )
    state["turns"] = {
        message_id: turn.model_dump(mode="json", exclude_none=True)
        for message_id, turn in stored.conversation.turns.items()
    }
    return state


def serialize_state(
    stored: StoredSession, from_seq: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Snapshot the live state into write instructions (call on the event
    loop; pydantic dumps copy, so the writer thread only ever touches plain
    data while the session keeps mutating).

    ``from_seq`` is the watermark: the first message index not known to be
    already on disk. Messages are immutable, so rows before the watermark
    stay; ``from_seq=0`` (first write, or after a truncation moved the
    watermark back) rewrites everything."""
    tracking = stored.git is not None
    meta = _meta_block(stored, tracking)
    messages = stored.conversation.messages
    from_seq = max(0, min(from_seq, len(messages)))
    state = _state_block(stored)
    payload = {
        "state": state,
        "messages": [
            m.model_dump(mode="json", exclude_none=True)
            for m in messages[from_seq:]
        ],
        "from_seq": from_seq,
    }
    return payload, meta


def write_state(
    name: str, payload: dict[str, Any], meta: dict[str, Any],
) -> None:
    """Apply one serialized write (runs in a worker thread).

    One transaction per write: readers never see a half-applied session,
    and a crash mid-write leaves the previous state intact. Rows from the
    watermark on are replaced — that covers appends AND the (rare)
    truncations, with no separate full-snapshot mode to keep in step."""
    _validate_session_name(name)
    state_json = json.dumps(payload["state"], separators=(",", ":"))
    from_seq = payload["from_seq"]
    rows = [
        (name, from_seq + index,
         json.dumps(message, separators=(",", ":")))
        for index, message in enumerate(payload["messages"])
    ]
    with connection() as db:
        db.execute(
            "DELETE FROM messages WHERE name = ? AND seq >= ?",
            (name, from_seq),
        )
        db.executemany(
            "INSERT OR REPLACE INTO messages (name, seq, data) VALUES (?, ?, ?)",
            rows,
        )
        db.execute(
            "INSERT OR REPLACE INTO sessions "
            "(name, schema, state, meta, updated_at) VALUES (?, ?, ?, ?, ?)",
            (name, SCHEMA_VERSION, state_json,
             json.dumps(meta, separators=(",", ":")),
             payload["state"]["updated_at"]),
        )


def load_state(name: str) -> StoredSession:
    """Load one saved session: the state row plus its replayed messages."""
    _validate_session_name(name)
    if not _db_path().exists():
        raise OSError(f"no saved session named {name!r}")
    with connection() as db:
        row = db.execute(
            "SELECT schema, state FROM sessions WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            raise OSError(f"no saved session named {name!r}")
        schema, state_json = row
        if schema != SCHEMA_VERSION:
            raise ValueError(
                f"session schema {schema} is not supported "
                f"(current {SCHEMA_VERSION}); start a new session"
            )
        rows = db.execute(
            "SELECT data FROM messages WHERE name = ? ORDER BY seq", (name,)
        ).fetchall()
    raw_state = json.loads(state_json)
    # A state saved by an older build may carry a since-renamed key. Config
    # files reject it with a pointed error (SandboxSettings), but a saved
    # session cannot be hand-edited — migrate it or it stays unloadable.
    settings = raw_state.get("sandbox_settings")
    if isinstance(settings, dict) and "use_container" in settings:
        settings["use_sandbox"] = settings.pop("use_container")
    stored = StoredSession(schema=SCHEMA_VERSION, **raw_state)
    # Conversation.turns is the ONE runtime home; the state row is its
    # persisted shape (a real StoredSession field).
    stored.conversation.turns = stored.turns
    messages: list[Message] = []
    for (data,) in rows:
        messages.append(MESSAGE_ADAPTER.validate_python(json.loads(data)))
    stored.conversation.messages = messages
    # The list was replaced wholesale: recompute the derived visible index.
    stored.conversation._recompute_visible()
    return stored


def load_meta(name: str) -> dict[str, Any] | None:
    """One saved session's small meta block."""
    if not _db_path().exists():
        return None
    try:
        with connection() as db:
            row = db.execute(
                "SELECT meta, updated_at FROM sessions WHERE name = ?", (name,)
            ).fetchone()
    except sqlite3.Error:
        return None
    return _meta_row(row)


def load_metas() -> dict[str, dict[str, Any]]:
    """Every saved session's meta block, in one query (listing hot path:
    never parses conversations)."""
    _warn_legacy_json_sessions()
    if not _db_path().exists():
        return {}
    try:
        with connection() as db:
            rows = db.execute("SELECT name, meta, updated_at FROM sessions").fetchall()
    except sqlite3.Error:
        return {}
    metas: dict[str, dict[str, Any]] = {}
    for name, meta_json, updated_at in rows:
        try:
            meta = json.loads(meta_json)
        except json.JSONDecodeError:
            continue
        meta["updated_at"] = updated_at
        metas[name] = meta
    return metas


def _meta_row(row: "tuple[str, str] | None") -> dict[str, Any] | None:
    """Decode one (meta JSON, updated_at column) row, injecting the column
    value the meta block deliberately does not duplicate."""
    if row is None:
        return None
    meta_json, updated_at = row
    try:
        meta = json.loads(meta_json)
    except json.JSONDecodeError:
        return None
    meta["updated_at"] = updated_at
    return meta


def delete_session(name: str) -> bool:
    existed = False
    if _db_path().exists():
        with connection() as db:
            cursor = db.execute(
                "DELETE FROM messages WHERE name = ?", (name,))
            existed = cursor.rowcount > 0
            cursor = db.execute(
                "DELETE FROM sessions WHERE name = ?", (name,))
            existed = existed or cursor.rowcount > 0
    directory = session_dir(name)
    if directory.is_dir():
        shutil.rmtree(directory)
        existed = True
    return existed


_legacy_warned = False


def _warn_legacy_json_sessions() -> None:
    """Log, once, that pre-SQLite (schema 3 JSON) sessions exist on disk but
    are not readable by this build — they are silently invisible in
    listings otherwise."""
    global _legacy_warned
    if _legacy_warned:
        return
    _legacy_warned = True
    try:
        legacy = [
            path.parent.name for path in SESSIONS_DIR.glob("*/meta.json")
        ]
    except OSError:
        return
    if legacy:
        logging.getLogger(__name__).warning(
            "skipping %d session(s) saved in the pre-SQLite JSON format "
            "(no importer in this build): %s",
            len(legacy), ", ".join(sorted(legacy)),
        )
