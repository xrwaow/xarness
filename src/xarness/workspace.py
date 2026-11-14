"""Per-workspace change tracking: the state every agent in a repo shares.

A session is an agent; a workspace is a repo root. Any number of agents can
work in one workspace — they share one container and one filesystem — so the
tracked state is per workspace, not per session:

  - ``baseline_tree``: the tree /diff measures against, moved by accept
  - ``head_tree``: the last state a turn ended on, and the incremental base
    the next snapshot is built on
  - ``epoch``: bumped on accept/reject, so turn checkpoints taken before a
    rebaseline stop being revertible
  - ``auto``: the collapsed ignored-path listing as of the baseline
  - ``oversized``: paths over the size limit, which never enter a tree
  - ``binary``: content-detected binaries (path -> size, mtime_ns), which
    never enter a tree either and are not re-probed while unchanged

It also owns the one persistent snapshot index and the lock that serializes
snapshots, accept, reject and undo (a second ``git add`` on the same index
would fail on ``index.lock``).

Rows live in the session store's sqlite database (schema and connection in
:mod:`xarness.session_store`). A ``--no-session`` run keeps its state in
memory and never opens the database; if a saved session later joins it,
persistence switches on and state + events are flushed in one transaction.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import gitwork, session_store
from .errors import RefError
from .protocol.views import DiffFileInfo, DiffGroup, DiffInfo, DiffScope
from .util import utcnow_iso

# Tools whose successful call is recorded as a workspace event.
_EXPLICIT_WRITES = frozenset({"edit_file", "write_file"})


@dataclass(slots=True)
class Event:
    """One recorded tool call: what it changed, and for which turn.

    ``auto`` are the collapsed ignored paths the call created; ``changed`` the
    tracked paths it changed for the calls known only from a before/after
    snapshot (``run_bash``). There is intentionally no timestamp and no stored
    label — labels are derived per viewer, and :attr:`paths` is the one answer
    to "which paths did this touch".

    ``writes`` is the provenance behind per-agent attribution: for an explicit
    ``write_file``/``edit_file``, ``(path, before blob sha, after blob sha)``
    per path (``None`` where it did not exist). Content-addressed, so the
    delta between the shas is that call's own patch — the baseline→worktree
    diff would credit it with earlier agents' hunks in the same file. A
    ``run_bash`` event records none: its changes are never one agent's."""

    session: str
    tool: str
    turn: str | None
    changed: tuple[str, ...]
    auto: tuple[str, ...]
    writes: tuple[tuple[str, str | None, str | None], ...] = ()

    @property
    def paths(self) -> tuple[str, ...]:
        """Every path the call touched. A write's own provenance names its
        path, so ``changed`` stays empty for it: one fact, one field."""
        return tuple(path for path, _, _ in self.writes) or self.changed


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
    # Content-detected binaries: path -> (size, mtime_ns), a stat cache so an
    # unchanged binary is never probed again. Persisted with the rest of the
    # workspace state (an excluded binary stays untracked forever).
    binary: dict[str, tuple[int, int]] = field(default_factory=dict)
    # Append-only per-workspace event log (pending until accept/reject).
    events: list[Event] = field(default_factory=list)
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

        A changed file that has grown past the size limit is dropped from the
        snapshot AND added to the exclusion set, or the next ``git add -A``
        would re-hash the same huge blob every time."""
        grown: list[str] = []
        tree = await gitwork.snapshot_tree(
            self.root, exclude=self.oversized, env_base=self.git_env(),
            index=self.index_path, oversized_out=grown,
            binary_cache=self.binary,
        )
        if grown:
            known = set(self.oversized)
            self.oversized = (*self.oversized,
                              *(p for p in grown if p not in known))
        return tree

    async def ignored(self) -> tuple[str, ...]:
        """The workspace's ignored paths right now, collapsed per directory:
        git's pattern rules plus the binaries content sniffing found."""
        listing = await gitwork.ignored_listing(self.root, self.git_env())
        return _collapse_ignored((*listing, *self.binary))

    # -- event log ------------------------------------------------------

    def record(
        self, *, session: str, tool: str, turn: str | None,
        auto: tuple[str, ...], paths: tuple[str, ...] = (),
        writes: tuple[tuple[str, str | None, str | None], ...] = (),
    ) -> None:
        """Append one event (a no-op row for an unchanged call).

        ``paths`` are the paths the call touched; when ``writes`` carries
        provenance it already names them, so they are not stored twice (a
        write that changed nothing has none and keeps its path here)."""
        if not (paths or auto or writes):
            return
        if writes:
            paths = ()
        event = Event(
            session=session, tool=tool, turn=turn,
            changed=paths, auto=auto, writes=writes,
        )
        if self.persist:
            with session_store.connection() as db:
                _write_event(db, self.root, event)
        self.events.append(event)

    def _sync_events(self) -> None:
        """Re-read the persisted log before using it: it belongs to the
        workspace, not to this process, so a second window (or a ``xarness
        -p`` run) appends to the same table and a stale copy would report an
        older author's delta. A ``--no-session`` run has no table to read."""
        if not self.persist:
            return
        events = _read_events(self.root)
        if events is None:
            return
        self.events = events

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

    def provenance(
        self,
    ) -> tuple[
        dict[str, dict[str, tuple[str | None, str | None]]],
        dict[str, str],
    ]:
        """``path -> {session: (before sha, after sha)}``, in event order, and
        ``path -> the session that wrote it last``.

        Each session's entry spans its own writes to the path — the blob it
        found before its first write and the blob it left after its last — so
        its diff is that session's delta. A session that writes twice with
        another agent in between absorbs the interleaved change."""
        chains: dict[str, dict[str, tuple[str | None, str | None]]] = {}
        last_writer: dict[str, str] = {}
        for event in self.events:
            for path, before, after in event.writes:
                by_session = chains.setdefault(path, {})
                first = by_session.get(event.session)
                by_session[event.session] = (
                    first[0] if first is not None else before, after,
                )
                last_writer[path] = event.session
        return chains, last_writer

    async def diff(
        self,
        viewer: str | Collection[str] | None = None,
        current: str | None = None,
        path: str | None = None,
        scope: DiffScope = "all",
    ) -> DiffInfo:
        """Every pending change since the baseline, labelled for ``viewer``.

        Rows and groups are identical for every session; only the label of
        paths the viewer wrote itself differs (``mine``). A viewer may be a
        set of session names (a session plus its subagents), so /sessions can
        report the work a parent delegated. Explicit writes win over bash,
        another session's events are never drift, and any path no event
        covers is drift.

        ``scope`` picks the granularity a shared file is reported in:
        ``"all"`` is one row per path counting its whole pending change,
        ``"mine"`` one row per author (the viewer's own delta leading, every
        other session's in the foldable ``agent`` section). Either way a
        row's counts are exactly the patch ``file_diff(path, author)``
        returns for it; ``additions``/``deletions`` stay the workspace's
        whole pending change.

        With ``path``, only that path's rows are built and no groups are
        listed: /diff <path> wants one row next to one diff, not the tree."""
        current = current or await self.snapshot()
        self._sync_events()
        stat = await gitwork.tree_diff_stat(self, self.baseline_tree, current)
        # Pending path -> the sessions that wrote it explicitly. Only
        # write_file/edit_file count: bash changes and drift have no author.
        explicit: dict[str, set[str]] = {}
        bash: set[str] = set()
        event_auto: set[str] = set()
        for event in self.events:
            event_auto.update(event.auto)
            if event.tool in _EXPLICIT_WRITES:
                for written in event.paths:
                    explicit.setdefault(written, set()).add(event.session)
            else:
                bash.update(event.paths)

        def is_mine(editors: Collection[str]) -> bool:
            if viewer is None:
                return False
            names = (viewer,) if isinstance(viewer, str) else viewer
            return any(name in editors for name in names)

        chains, last_writer = self.provenance()

        def kind_of(editors: Collection[str] | None, bash_touched: bool) -> str:
            if editors:
                return "mine" if is_mine(editors) else "agent"
            return "agent" if bash_touched else "drift"

        async def path_rows(
            path: str, is_new: bool, pending: tuple[int, int] | None,
        ) -> tuple[list[DiffFileInfo], tuple[int, int]]:
            """The rows one path contributes, and its whole pending delta.

            ``pending`` is its change against the baseline, None for a path no
            tree holds (an ignored one): then the span of the recorded writes
            is the whole change."""
            by_session = chains.get(path)
            if by_session:
                names = tuple(by_session)
                whole = pending
                if whole is None:
                    whole = await gitwork.blob_diff_stat(
                        self, by_session[names[0]][0], by_session[names[-1]][1],
                    )
                if scope == "all":
                    # Every agent's edits (and anything else pending in the
                    # file) in one row: what git reports for the path. The
                    # badge names no single delta, so it follows the last
                    # writer — the one the file is now as they left it.
                    return [DiffFileInfo(
                        path=path, additions=whole[0], deletions=whole[1],
                        is_new=is_new,
                        kind=kind_of((last_writer[path],), path in bash),
                        authors=names,
                    )], whole
                rows = []
                for name in names:
                    before, after = by_session[name]
                    additions, deletions = await gitwork.blob_diff_stat(
                        self, before, after,
                    )
                    rows.append(DiffFileInfo(
                        path=path, additions=additions, deletions=deletions,
                        # Only the write that created the path is "new".
                        is_new=is_new and before is None,
                        kind="mine" if is_mine((name,)) else "agent",
                        author=name, authors=names,
                    ))
                return rows, whole
            editors = explicit.get(path)
            additions, deletions = pending or (0, 0)
            return [DiffFileInfo(
                path=path, additions=additions, deletions=deletions,
                is_new=is_new, kind=kind_of(editors, path in bash),
                authors=tuple(sorted(editors or ())),
            )], (additions, deletions)

        built = await asyncio.gather(*(
            path_rows(f.path, f.is_new, (f.additions, f.deletions))
            for f in stat.files if path is None or f.path == path
        ))
        files = [row for rows, _ in built for row in rows]
        # The header totals count the workspace, not the rows: per-author
        # spans can overlap when agents interleave their writes.
        additions = sum(whole[0] for _, whole in built)
        deletions = sum(whole[1] for _, whole in built)
        current_ignored = await self.ignored()
        tracked = {f.path for f in files}
        for candidate in explicit:
            # An explicit write to an ignored path never enters a tree (so it
            # has no stat row) and has no baseline copy, but it is still that
            # agent's pending edit. Accept folds the path into `auto` and
            # clears the event, so it stops showing there; reject lists it as
            # unrestorable.
            if candidate in tracked or (path is not None and candidate != path):
                continue
            if _is_ignored(candidate, current_ignored):
                rows, (adds, dels) = await path_rows(candidate, True, None)
                files.extend(rows)
                additions += adds
                deletions += dels
        groups: list[DiffGroup] = []
        ignored_files = 0
        known = set(self.auto)
        fresh = [
            group for group in current_ignored
            if path is None and group not in known
        ]
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
            additions=additions, deletions=deletions,
            ignored_files=ignored_files,
        )

    async def file_diff(
        self,
        path: str,
        current: str | None = None,
        author: str | None = None,
    ) -> str:
        """One file's unified diff: the patch one row counts.

        ``author`` names the session whose own delta the row shows (the blob
        pair of its first-to-last write), so a shared file never displays an
        earlier agent's lines as this one's. None means the path's whole
        pending change; an ignored path has no baseline copy, so it reads as
        added."""
        self._sync_events()
        by_session = self.provenance()[0].get(path) or {}
        span = by_session.get(author) if author is not None else None
        if span is not None:
            return await gitwork.blob_diff_text(self, path, *span)
        if _is_ignored(path.rstrip("/"), await self.ignored()):
            if by_session and author is None:
                # An ignored path has no baseline copy, so its whole pending
                # change is the log's own span (what its combined row counts)
                # rather than "everything, added from nothing".
                names = tuple(by_session)
                return await gitwork.blob_diff_text(
                    self, path, by_session[names[0]][0], by_session[names[-1]][1],
                )
            return await gitwork.ignored_diff(self, path)
        return await gitwork.git_diff(self, path, current=current)

    # -- rebaselining ---------------------------------------------------

    async def accept(self) -> str:
        """Lock in the worktree as it stands: every pending change, drift
        included, becomes the new baseline. Returns the new baseline tree."""
        async with self.lock:
            current = await self._snapshot()
            self.auto = await self.ignored()
            self.reset_to(current, current)
            self.clear_events()
            self.save()
            return current

    async def reject(self) -> tuple[str, ...]:
        """Discard every change since the baseline (drift included), remove
        the ignored groups events created, and reset. Returns the ignored
        files agents edited, which cannot be restored."""
        async with self.lock:
            self._sync_events()
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
                target = self.root / group
                _remove_path(target)
                _prune_empty_parents(self.root, target)
            self.reset_to(self.baseline_tree, self.baseline_tree)
            self.clear_events()
            self.save()
            return tuple(unrestorable)

    async def undo(
        self, session: str, turns: list[tuple[str, "TurnCheckpoint"]],
    ) -> list[str]:
        """Path-scoped undo of one session's turns (newest first).

        A turn is reversed by merging its *own* change out of the live file:
        three-way merge with the turn's after state as the base, the worktree
        as ours and the turn's before state as theirs. A file another agent
        later edited elsewhere merges cleanly and keeps their hunks; only
        overlapping hunks (or a path the turn created/deleted, which has no
        content to merge into) conflict. Returns the conflicting paths, and
        touches nothing when any path conflicts."""
        async with self.lock:
            self._sync_events()
            # path -> ("bytes", content) | ("absent", None) | ("tree", sha).
            # A tree is the untouched-file fast path: restoring it through
            # git keeps the mode (the exec bit) without reading it back.
            desired: dict[str, tuple[str, Any]] = {}
            conflicts: list[str] = []
            for turn_id, turn in reversed(turns):
                paths = {
                    path for event in self.events
                    if event.session == session and event.turn == turn_id
                    for path in event.paths
                }
                for path in sorted(paths):
                    base = await _tree_content(self, turn.after_tree, path)
                    if path in desired:
                        kind, value = desired[path]
                        if kind == "absent":
                            current = None
                        elif kind == "tree":
                            current = await _tree_content(self, value, path)
                        else:
                            current = value
                    else:
                        current = _read_path(self.root, path)
                    before = await _tree_content(self, turn.before_tree, path)
                    if current == base:
                        # Untouched since the turn: reverse it whole.
                        desired[path] = (
                            ("absent", None) if before is None
                            else ("tree", turn.before_tree)
                        )
                        continue
                    if base is None or before is None or current is None:
                        # The turn created, deleted or would delete the whole
                        # path and it changed since: nothing to merge into.
                        conflicts.append(path)
                        continue
                    if (self.root / path).is_symlink():
                        # A later change to a link is never merged: writing
                        # through it would hit whatever it points at.
                        conflicts.append(path)
                        continue
                    merged = await gitwork.merge_texts(
                        self, base=base, ours=current, theirs=before,
                    )
                    if merged is None:
                        conflicts.append(path)
                    else:
                        desired[path] = ("bytes", merged)
            if conflicts:
                return sorted(set(conflicts))
            by_tree: dict[str, list[str]] = {}
            for path, (kind, value) in desired.items():
                if kind == "tree":
                    by_tree.setdefault(value, []).append(path)
                    continue
                target_path = self.root / path
                if kind == "bytes":
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_bytes(value)
                    continue
                _remove_path(target_path)
                _prune_empty_parents(self.root, target_path)
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
            "binary": {path: list(key) for path, key in self.binary.items()},
        }, separators=(",", ":"))

    def enable_persistence(self) -> None:
        """A saved session joined a workspace a --no-session run created: turn
        persistence on and flush state + events in one transaction (never a
        row without its events)."""
        if self.persist:
            return
        with session_store.connection() as db:
            _write(db, self.root, state=self._state_json())
            for event in self.events:
                _write_event(db, self.root, event)
        # Only once the row is there: a failed flush stays in-memory-only (and
        # is retried by the next saved session that joins this workspace).
        self.persist = True


def _parse_writes(
    raw: Any,
) -> tuple[tuple[str, str | None, str | None], ...]:
    """The ``(path, before, after)`` provenance of a persisted event.

    Rows written before provenance existed have none, and an unreadable row is
    dropped rather than turning every /diff into an error — its path still
    shows, counted as the whole file's pending diff."""
    writes = []
    for entry in raw if isinstance(raw, list) else ():
        if isinstance(entry, (list, tuple)) and len(entry) == 3:
            path, before, after = entry
            if isinstance(path, str) and path:
                writes.append((path, before or None, after or None))
    return tuple(writes)


def _parse_binary(raw: Any) -> dict[str, tuple[int, int]]:
    """The persisted binary cache (path -> size, mtime_ns); bad entries drop."""
    if not isinstance(raw, dict):
        return {}
    return {
        path: (key[0], key[1]) for path, key in raw.items()
        if isinstance(path, str) and path
        and isinstance(key, (list, tuple)) and len(key) == 2
        and all(isinstance(value, int) for value in key)
    }


def _collapse_ignored(paths: Iterable[str]) -> tuple[str, ...]:
    """Drop entries already covered by a parent entry. Walks each path's
    ancestors against the kept set: a sibling (``a/b!x``) can sort between a
    parent (``a/b``) and its child (``a/b/c``), so sorting is not enough."""
    kept: set[str] = set()
    out: list[str] = []
    for path in sorted(set(paths)):
        parts = path.split("/")
        if not any("/".join(parts[:i]) in kept for i in range(1, len(parts))):
            kept.add(path)
            out.append(path)
    return tuple(out)


def _write(
    db: sqlite3.Connection, root: Path, *,
    state: str | None = None, refs: str | None = None,
) -> None:
    """Upsert one row's columns on ``db``. State and refs share a row but are
    written independently, so neither can clobber the other."""
    # COALESCE keeps the column this call does not set (the INSERT's own
    # value only matters for the row it creates).
    db.execute(
        "INSERT INTO workspaces (root, state, refs, updated_at) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(root) DO UPDATE SET state = COALESCE(?, state), "
        "refs = COALESCE(?, refs), updated_at = excluded.updated_at",
        (str(root), state or "", refs or "", utcnow_iso(), state, refs),
    )


async def _tree_content(
    store: gitwork.RepoStore, tree: str, path: str,
) -> bytes | None:
    """One path's bytes in a snapshot tree (None when absent there)."""
    entry = await gitwork.tree_entry(store, tree, path)
    if entry is None:
        return None
    return await gitwork.read_blob(store, entry[1])


def _read_path(root: Path, path: str) -> bytes | None:
    """One worktree path's bytes (None when it does not exist). A symlink
    reads as its target path, exactly what git stores as its blob, so undo's
    content comparison sees a link the way the tree does."""
    target = root / path
    try:
        if target.is_symlink():
            return os.readlink(target).encode()
        return target.read_bytes()
    except OSError:
        return None


def _is_ignored(path: str, listing: tuple[str, ...]) -> bool:
    """Whether ``path`` is covered by the collapsed ignored-path listing (a
    directory entry covers everything beneath it; a file inside a mixed
    directory appears verbatim)."""
    return any(
        path == entry or path.startswith(entry + "/") for entry in listing
    )


def _row(root: Path) -> tuple[str, str] | None:
    """The (state, refs) JSON columns for ``root``, or None — also when there
    is no database yet, since reading must not create one."""
    try:
        with session_store.read_connection() as db:
            if db is None:
                return None
            return db.execute(
                "SELECT state, refs FROM workspaces WHERE root = ?", (str(root),),
            ).fetchone()
    except sqlite3.Error:
        return None


def _read_events(root: Path) -> list[Event] | None:
    """The workspace's persisted event log, oldest first (None when the
    database cannot be read, so a transient failure never looks like an empty
    log). An unreadable row is dropped, not fatal: its path still shows,
    counted as the file's whole pending change."""
    try:
        with session_store.read_connection() as db:
            rows = [] if db is None else db.execute(
                "SELECT data FROM workspace_events WHERE root = ? "
                "ORDER BY seq", (str(root),),
            ).fetchall()
    except sqlite3.Error:
        return None
    events = []
    for (data,) in rows:
        try:
            raw = json.loads(data)
            events.append(Event(
                session=raw["session"], tool=raw["tool"],
                turn=raw.get("turn"),
                changed=tuple(raw.get("paths", ())),
                auto=tuple(raw.get("auto", ())),
                writes=_parse_writes(raw.get("writes")),
            ))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return events


def _write_event(db: sqlite3.Connection, root: Path, event: Event) -> None:
    """Append one event row. The sequence number comes from the table, not
    from this process: two hosts would both claim ``seq=1`` and the primary
    key would let the second write replace the first."""
    db.execute(
        "INSERT INTO workspace_events (root, seq, data) VALUES (?, "
        "(SELECT COALESCE(MAX(seq), 0) + 1 FROM workspace_events "
        "WHERE root = ?), ?)",
        (str(root), str(root), json.dumps({
            "session": event.session, "tool": event.tool, "turn": event.turn,
            "paths": list(event.changed), "auto": list(event.auto),
            "writes": [list(write) for write in event.writes],
        }, separators=(",", ":"))),
    )


def _delete_events(
    db: sqlite3.Connection, root: Path, *,
    session: str | None = None, turns: set[str] | None = None,
) -> None:
    """Clear a workspace's events, or one session's events for ``turns``. Both
    live inside the row's JSON, so sqlite's JSON functions match them."""
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


def _prune_empty_parents(root: Path, target: Path) -> None:
    """Drop the directories a deletion left empty, up to the workspace root."""
    for parent in target.parents:
        if parent == root or root not in parent.parents:
            break
        try:
            parent.rmdir()
        except OSError:
            break


async def open_workspace(
    root: Path, *, persist: bool = True, allow_init: bool = True,
) -> "WorkspaceState | None":
    """Open the tracked state for ``root``'s repo (detecting, or when allowed
    initializing one). None when the directory cannot be tracked.

    An existing row is used as-is: the baseline belongs to the workspace, so
    reopening an agent must never re-baseline it (everything saved would look
    like drift and every turn checkpoint would lose its trees). A missing row
    (first run, or an upgrade) is created from a fresh snapshot."""
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
        state._sync_events()
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
        state.binary = _parse_binary(saved.get("binary"))
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
    """Add/remove one workspace reference path — the ONE implementation, used
    by the session's /ref command and the host's workspace endpoint.

    Returns the canonical (saved) list and the resolved host path. With
    ``save=False`` the caller persists the returned list itself (the session
    tolerates a read-only data dir; the host does not). Raises
    :class:`RefError` when the path does not exist (a typo must never mkdir a
    reference) or is not a known reference."""
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
