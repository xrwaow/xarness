"""Per-workspace change tracking: the state every agent in a repo shares.

A session is an agent; a workspace is a repo root. Any number of agents can
work in one workspace — they share one container and one filesystem — so the
tracked state is per workspace, not per session:

  - ``baseline_tree``: the tree /diff measures against, moved by accept
  - ``head_tree``: the last state a turn ended on (advanced by ``refresh``),
    and the incremental base the next snapshot is built on
  - ``epoch``: bumped on accept/reject, so turn checkpoints taken before a
    rebaseline stop being revertible
  - ``auto``: the collapsed ignored-path listing as of the baseline
  - ``oversized``: paths over the size limit, which never enter a tree

It also owns the one persistent snapshot index and the lock that serializes
snapshots, accept, reject and undo (a second ``git add`` on the same index
would fail on ``index.lock``).

Rows live in the session store's sqlite database: the schema and the
connection belong to :mod:`xarness.session_store`, the rows to this module.
A workspace opened by a ``--no-session`` run keeps its state in memory and
never opens the database (the read side is ``mode=ro``, so even reading
cannot create it); if a saved session later joins it, persistence is
switched on and the state and its events are flushed in one transaction.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import gitwork, session_store
from .errors import RefError
from .protocol.views import DiffFileInfo, DiffGroup, DiffInfo
from .util import utcnow_iso

# Tools whose successful call is recorded as a workspace event.
_EXPLICIT_WRITES = frozenset({"edit_file", "write_file"})


@dataclass(slots=True)
class Event:
    """One recorded tool call: what it changed, and for which turn.

    ``paths`` are repo-root-relative tracked paths; ``auto`` the collapsed
    ignored paths the call created. There is intentionally no timestamp and
    no stored label — labels are derived per viewer."""

    seq: int
    session: str
    tool: str
    turn: str | None
    paths: tuple[str, ...]
    auto: tuple[str, ...]


@dataclass
class WorkspaceState:
    """The tracked state of one repo root, shared by every agent in it."""

    root: Path
    git_dir: Path
    external: bool = False
    # False for a --no-session run: nothing is read from or written to the
    # database.
    persist: bool = True
    baseline_tree: str = ""
    head_tree: str = ""
    epoch: int = 0
    auto: tuple[str, ...] = ()
    oversized: tuple[str, ...] = ()
    # Append-only per-workspace event log (pending until accept/reject).
    events: list[Event] = field(default_factory=list)
    next_seq: int = 1
    # One lock per workspace: snapshots, accept, reject and undo must never
    # run ``git add -A`` on the same index file at once.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def index_path(self) -> Path:
        return gitwork.index_path(self.root)

    def git_env(self) -> dict[str, str]:
        """Env for a git subprocess: external mode addresses the store via
        GIT_DIR/GIT_WORK_TREE; otherwise cwd picks up the worktree's .git."""
        return gitwork.git_env(self.root, self.git_dir, self.external)

    # -- snapshots ------------------------------------------------------

    async def refresh(self) -> str:
        """Observe the worktree now: snapshot it and make that the head
        tree."""
        async with self.lock:
            tree = await self._snapshot()
            self.head_tree = tree
            self.save()
            return tree

    async def observe(self) -> str:
        """Snapshot the worktree now without moving the head tree. /diff is
        always computed against the live worktree, not the head."""
        async with self.lock:
            return await self._snapshot()

    async def snapshot(self) -> str:
        """Snapshot the worktree (see :meth:`observe`); no locking, so a
        caller that already holds the lock can snapshot."""
        return await self._snapshot()

    async def _snapshot(self) -> str:
        """Snapshot the worktree (the caller holds the lock).

        Files that grew past the size limit are dropped from the snapshot
        AND added to the exclusion set, or the next ``git add -A`` would
        re-hash the same huge blob every time."""
        grown: list[str] = []
        tree = await gitwork.snapshot_tree(
            self.root, exclude=self.oversized, env_base=self.git_env(),
            index=self.index_path, previous=self.head_tree or None,
            oversized_out=grown,
        )
        if grown:
            known = set(self.oversized)
            self.oversized = (*self.oversized,
                              *(p for p in grown if p not in known))
        return tree

    async def ignored(self) -> tuple[str, ...]:
        """The workspace's ignored paths right now, collapsed per directory."""
        return await gitwork.ignored_listing(self.root, self.git_env())

    # -- event log ------------------------------------------------------

    def record(
        self, *, session: str, tool: str, turn: str | None,
        paths: tuple[str, ...], auto: tuple[str, ...],
    ) -> None:
        """Append one event (a no-op row for an unchanged call)."""
        if not paths and not auto:
            return
        event = Event(
            seq=self.next_seq, session=session, tool=tool, turn=turn,
            paths=paths, auto=auto,
        )
        self.next_seq += 1
        self.events.append(event)
        if self.persist:
            with session_store.connection() as db:
                _write_event(db, self.root, event)

    def clear_events(self) -> None:
        """Accept/reject acknowledged every pending event."""
        self.events = []
        if self.persist:
            with session_store.connection() as db:
                _delete_events(db, self.root)

    def drop_events(self, session: str, turns: set[str]) -> None:
        """Forget one session's events for the turns an undo removed."""
        kept = [
            event for event in self.events
            if not (event.session == session and event.turn in turns)
        ]
        if len(kept) == len(self.events):
            return
        self.events = kept
        if self.persist:
            with session_store.connection() as db:
                _delete_events(db, self.root, session=session, turns=turns)

    # -- diff, labelled per viewer --------------------------------------

    async def diff(
        self,
        viewer: str | Collection[str] | None = None,
        current: str | None = None,
    ) -> DiffInfo:
        """Every pending change since the baseline, labelled for ``viewer``.

        Rows and groups are identical for every session; only the label of
        paths the viewer wrote itself differs (``mine``). A viewer may be a
        set of session names (a session plus its subagents), so /sessions
        can report the work a parent delegated. Explicit writes win over
        bash, another session's events are never drift, and any path no
        event covers is drift."""
        current = current or await self.snapshot()
        stat = await gitwork.tree_diff_stat(self, self.baseline_tree, current)
        # Pending explicitly written path -> the sessions that wrote it. Only
        # write_file/edit_file count: bash changes and drift have no author,
        # so they never appear here.
        explicit: dict[str, set[str]] = {}
        bash: set[str] = set()
        event_auto: set[str] = set()
        for event in self.events:
            event_auto.update(event.auto)
            if event.tool in _EXPLICIT_WRITES:
                for path in event.paths:
                    explicit.setdefault(path, set()).add(event.session)
            else:
                bash.update(event.paths)

        def is_mine(editors: set[str]) -> bool:
            if viewer is None:
                return False
            if isinstance(viewer, str):
                return viewer in editors
            return any(name in editors for name in viewer)

        def label(path: str) -> str:
            editors = explicit.get(path)
            if editors:
                return "mine" if is_mine(editors) else "agent"
            return "agent" if path in bash else "drift"

        files = [
            DiffFileInfo(
                path=f.path, additions=f.additions, deletions=f.deletions,
                is_new=f.is_new, kind=label(f.path),
            )
            for f in stat.files
        ]
        current_ignored = await self.ignored()
        tracked = {row.path for row in files}
        for path, editors in explicit.items():
            # An explicit write to an ignored path never enters a tree (so it
            # has no stat row) and has no baseline copy, but it is still that
            # agent's pending edit. Accept folds the path into `auto` and
            # clears the event, so it stops showing there; reject lists it as
            # unrestorable.
            if path in tracked or not _is_ignored(path, current_ignored):
                continue
            files.append(DiffFileInfo(
                path=path, is_new=True,
                kind="mine" if viewer in editors else "agent",
            ))
        known = set(self.auto)
        fresh = [group for group in current_ignored if group not in known]
        groups: list[DiffGroup] = []
        ignored_files = 0
        if fresh:
            # One uncollapsed listing buys every group's file count.
            uncollapsed = await gitwork.ignored_listing(
                self.root, self.git_env(), collapsed=False,
            )
            for group in fresh:
                under = [
                    path for path in uncollapsed
                    if path == group or path.startswith(group + "/")
                ]
                editors = [explicit[p] for p in under if p in explicit]
                if editors:
                    kind = "mine" if any(is_mine(e) for e in editors) else "agent"
                elif group in event_auto:
                    kind = "auto"
                else:
                    kind = "drift"
                count = max(len(under), 1)
                ignored_files += count
                prefix = group + "/" if (self.root / group).is_dir() else group
                groups.append(DiffGroup(prefix=prefix, files=count, kind=kind))
        return DiffInfo(
            files=files, groups=groups,
            additions=sum(row.additions for row in files),
            deletions=sum(row.deletions for row in files),
            ignored_files=ignored_files,
        )

    # -- rebaselining ---------------------------------------------------

    async def accept(self) -> str:
        """Lock in every non-drift change: the worktree becomes the new
        baseline, but paths no event covers keep their old baseline entry.
        Returns the new baseline tree."""
        async with self.lock:
            current = await self._snapshot()
            summary = await self.diff(current=current)
            paths = [row.path for row in summary.files if row.kind != "drift"]
            baseline = await gitwork.splice_tree(
                self, self.baseline_tree, current, paths,
            )
            self.auto = await self.ignored()
            self.reset_to(baseline, current)
            self.clear_events()
            self.save()
            return baseline

    async def reject(self) -> tuple[str, ...]:
        """Discard every change since the baseline (drift included), remove
        the ignored groups events created, and reset. Returns the ignored
        files agents edited, which cannot be restored."""
        async with self.lock:
            ignored = await self.ignored()
            unrestorable = sorted({
                path for event in self.events
                if event.tool in _EXPLICIT_WRITES
                for path in event.paths if _is_ignored(path, ignored)
            })
            current = await self._snapshot()
            await gitwork.revert_to_tree(self, self.baseline_tree, current)
            known = set(self.auto)
            for group in sorted(
                {g for event in self.events for g in event.auto} - known
            ):
                _remove_path(self.root / group)
            self.reset_to(self.baseline_tree, self.baseline_tree)
            self.clear_events()
            self.save()
            return tuple(unrestorable)

    async def undo(
        self, session: str, turns: list[tuple[str, "TurnCheckpoint"]],
    ) -> list[str]:
        """Path-scoped undo of one session's turns (newest first).

        Returns the conflicting paths and touches nothing when any path
        changed since its turn; otherwise restores each path from the
        turn's before tree, unlinks the ones it created, and drops the
        turns' events."""
        async with self.lock:
            desired: dict[str, str | None] = {}
            target: dict[str, str] = {}
            conflicts: list[str] = []
            for turn_id, turn in reversed(turns):
                paths = {
                    path for event in self.events
                    if event.session == session and event.turn == turn_id
                    for path in event.paths
                }
                for path in sorted(paths):
                    after = await gitwork.tree_entry(
                        self, turn.after_tree, path,
                    )
                    if path in desired:
                        current = desired[path]
                    else:
                        current = gitwork.worktree_entry(self.root, path)
                    if current != after:
                        conflicts.append(path)
                        continue
                    before = await gitwork.tree_entry(
                        self, turn.before_tree, path,
                    )
                    if before is None:
                        desired[path] = None
                    else:
                        desired[path] = path
                        target[path] = turn.before_tree
            if conflicts:
                return sorted(set(conflicts))
            by_tree: dict[str, list[str]] = {}
            for path, keep in desired.items():
                if keep is not None:
                    by_tree.setdefault(target[path], []).append(path)
                    continue
                target_path = self.root / path
                _remove_path(target_path)
                # Drop the directories the deletion left empty, up to (not
                # including) the repo root.
                parent = target_path.parent
                while parent != self.root and self.root in parent.parents:
                    try:
                        parent.rmdir()
                    except OSError:
                        break
                    parent = parent.parent
            for tree, paths in by_tree.items():
                await gitwork.restore_paths(self, tree, paths)
            self.drop_events(session, {turn_id for turn_id, _ in turns})
            return []

    def reset_to(self, baseline_tree: str, head_tree: str) -> None:
        """Rebaseline after accept/reject: bump the epoch (turn checkpoints
        from before it stop being revertible) and move both trees."""
        self.baseline_tree = baseline_tree
        self.head_tree = head_tree
        self.epoch += 1

    # -- persistence ----------------------------------------------------

    def save(self) -> None:
        """Write the tracking columns of this workspace's row."""
        if not self.persist:
            return
        with session_store.connection() as db:
            _write(db, self.root, state=self._state_json())

    def _state_json(self) -> str:
        return json.dumps({
            "baseline_tree": self.baseline_tree,
            "head_tree": self.head_tree,
            "epoch": self.epoch,
            "auto": list(self.auto),
            "oversized": list(self.oversized),
        }, separators=(",", ":"))

    def enable_persistence(self) -> None:
        """A saved session joined a workspace that a --no-session run created:
        turn persistence on and flush the in-memory state and events to the
        database in one transaction (never a row without its events)."""
        if self.persist:
            return
        with session_store.connection() as db:
            _write(db, self.root, state=self._state_json())
            for event in self.events:
                _write_event(db, self.root, event)
        # Only once the row is there: a failed flush stays in-memory-only (and
        # is retried by the next saved session that joins this workspace).
        self.persist = True


def _write(
    db: sqlite3.Connection, root: Path, *,
    state: str | None = None, refs: str | None = None,
) -> None:
    """Upsert one row's columns on ``db``. The tracking state and the refs
    share a row but are written independently, so neither can clobber the
    other."""
    # COALESCE keeps the column this call does not set (the INSERT's own
    # value only matters for the row it creates).
    db.execute(
        "INSERT INTO workspaces (root, state, refs, updated_at) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(root) DO UPDATE SET state = COALESCE(?, state), "
        "refs = COALESCE(?, refs), updated_at = excluded.updated_at",
        (str(root), state or "", refs or "", utcnow_iso(), state, refs),
    )


def _is_ignored(path: str, listing: tuple[str, ...]) -> bool:
    """Whether ``path`` is covered by an ignored-path listing.

    The listing is the collapsed one (``git ls-files --directory``), so a
    directory entry covers everything beneath it; a file inside a mixed
    directory appears verbatim."""
    return any(
        path == entry or path.startswith(entry + "/") for entry in listing
    )


def _row(root: Path) -> tuple[str, str] | None:
    """The (state, refs) JSON columns for ``root``, or None (also when there
    is no database yet: opening a workspace for reading must not create it)."""
    try:
        with session_store.read_connection() as db:
            if db is None:
                return None
            return db.execute(
                "SELECT state, refs FROM workspaces WHERE root = ?", (str(root),),
            ).fetchone()
    except sqlite3.Error:
        return None


def _write_event(db: sqlite3.Connection, root: Path, event: Event) -> None:
    """Append one event row."""
    db.execute(
        "INSERT OR REPLACE INTO workspace_events (root, seq, data) "
        "VALUES (?, ?, ?)",
        (str(root), event.seq, json.dumps({
            "session": event.session, "tool": event.tool, "turn": event.turn,
            "paths": list(event.paths), "auto": list(event.auto),
        }, separators=(",", ":"))),
    )


def _delete_events(
    db: sqlite3.Connection, root: Path, *,
    session: str | None = None, turns: set[str] | None = None,
) -> None:
    """Clear a workspace's events, or one session's events for ``turns``.

    The session and turn live inside the row's JSON, so they are matched
    with sqlite's JSON functions rather than read back into Python."""
    sql = "DELETE FROM workspace_events WHERE root = ?"
    params: list[Any] = [str(root)]
    if session is not None:
        sql += (
            " AND json_extract(data, '$.session') = ?"
            " AND json_extract(data, '$.turn')"
            " IN (SELECT value FROM json_each(?))"
        )
        params += [session, json.dumps(sorted(turns or ()))]
    db.execute(sql, params)


def _remove_path(target: Path) -> None:
    """Delete a file or a whole directory, ignoring absence."""
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target, ignore_errors=True)
        return
    try:
        target.unlink()
    except OSError:
        pass


async def open_workspace(
    root: Path, *, persist: bool = True, allow_init: bool = True,
) -> "WorkspaceState | None":
    """Open the tracked state for ``root``'s repo (detecting or, when
    allowed, initializing one). None when the directory cannot be tracked.

    An existing row is used as-is: the baseline belongs to the workspace, so
    reopening an agent must never re-baseline it (everything saved would look
    like drift and every turn checkpoint would lose its trees). A row that is
    missing (first run, or an upgrade) is created from a fresh snapshot."""
    root = Path(root).resolve()
    repo = await gitwork.detect_repo(root)
    if repo is None:
        if not allow_init:
            return None
        repo = await gitwork.init_external_repo(root)
    state = WorkspaceState(
        root=repo.root, git_dir=repo.git_common_dir,
        external=repo.external, persist=persist,
    )
    row = _row(state.root) if persist else None
    if persist:
        # The persisted event log, oldest first (unreadable rows are dropped).
        try:
            with session_store.read_connection() as db:
                rows = [] if db is None else db.execute(
                    "SELECT seq, data FROM workspace_events WHERE root = ? "
                    "ORDER BY seq", (str(state.root),),
                ).fetchall()
        except sqlite3.Error:
            rows = []
        for seq, data in rows:
            try:
                raw = json.loads(data)
                state.events.append(Event(
                    seq=seq, session=raw["session"], tool=raw["tool"],
                    turn=raw.get("turn"), paths=tuple(raw.get("paths", ())),
                    auto=tuple(raw.get("auto", ())),
                ))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        state.next_seq = max((e.seq for e in state.events), default=0) + 1
    saved = {}
    if row is not None:
        try:
            saved = json.loads(row[0] or "{}")
        except json.JSONDecodeError:
            saved = {}
    if saved.get("baseline_tree"):
        state.baseline_tree = saved["baseline_tree"]
        state.head_tree = saved.get("head_tree") or state.baseline_tree
        state.epoch = int(saved.get("epoch", 0))
        state.auto = tuple(saved.get("auto", ()))
        state.oversized = tuple(saved.get("oversized", ()))
        return state
    state.oversized = gitwork.oversized_paths(state.root)
    async with state.lock:
        state.baseline_tree = state.head_tree = await state._snapshot()
    state.auto = await state.ignored()
    state.save()
    return state


# ----------------------------------------------------------------------
# workspace refs (/refs): per workspace, in the same row as its state


def load_workspace_refs(workspace: Path | str) -> list[str]:
    """Reference paths saved for this workspace (shared by its sessions)."""
    row = _row(Path(workspace).expanduser().resolve())
    if row is None:
        return []
    try:
        refs = json.loads(row[1] or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(refs, list):
        return []
    return [ref for ref in refs if isinstance(ref, str) and ref.strip()]


def save_workspace_refs(workspace: Path | str, refs: list[str]) -> None:
    """Persist the workspace's refs (an empty list clears the column)."""
    specs = [ref for ref in refs if isinstance(ref, str) and ref.strip()]
    root = Path(workspace).expanduser().resolve()
    with session_store.connection() as db:
        _write(db, root, refs=json.dumps(specs))


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
    specs = load_workspace_refs(workspace)
    if add:
        if not host.exists():
            raise RefError(f"no such path: {raw}")
        if str(host) not in specs:
            specs.append(str(host))
    else:
        kept = [ref for ref in specs if Path(ref).expanduser().resolve() != host]
        if len(kept) == len(specs):
            raise RefError(f"no such reference: {raw}")
        specs = kept
    if save:
        save_workspace_refs(workspace, specs)
    return specs, host
