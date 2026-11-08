"""Direct-write change tracking for agent edits.

The agent's sandbox is bound to the user's real workspace: file edits and
shell commands land in the user's files immediately — there is no worktree
and no accept/reject copy. Git is used purely as a change record:

  - opening a workspace snapshots it as a tree object (the workspace's
    ``baseline_tree``) — the base /diff measures against
  - every successful tool call is recorded as an event in
    :mod:`xarness.workspace`; labels (mine/agent/auto/drift) are derived
    from those events, never stored
  - /undo and /retry restore only the paths a turn's events touched, from
    the turn's before tree, and refuse (all-or-nothing) when a path changed
    since the turn

Snapshots are built with a throwaway index (``GIT_INDEX_FILE`` pointing at a
temp file): the user's index, HEAD, and refs are never touched, and no
commits are created. What *is* written is unreachable tree/blob
objects — and which object store they land in depends on the workspace
(see :func:`open_workspace`):

  - **the workspace is a repo**: the user's own object store, i.e. their
    ``.git``. Nothing else in it changes, and because objects are
    content-addressed a blob git already holds is shared rather than
    duplicated (files over 1 MB are never tracked at all). The trade-off is
    that the harness's data then lives in a store whose pruning schedule is
    the user's, not ours: no ref points at these objects, so ``git gc
    --prune=now``, ``git repack -ad`` or any aggressive prune discards
    them — taking an open session's checkpoints with it. ``/diff`` and
    ``/undo`` then fail against the missing baseline until the workspace
    re-baselines (``/accept``, a fresh session, or the row being rebuilt).
    ``git fsck`` reports the objects as dangling and they never travel
    with clone/fetch/push.
  - **no repo** (``--no-init-repo`` not given): a private store under
    xarness's state directory (:func:`_managed_store_dir`), addressed with
    an explicit ``GIT_DIR`` so no ``.git`` is created in the workspace.
    That store is xarness's own: nothing else gc's it, so its objects stay
    put until someone prunes them deliberately.

Keeping the snapshots in the user's store is a deliberate trade-off (no
second copy of the user's files, and diffing stays a single-repo
operation), not an oversight; making every session use a private store
would separate the lifetimes, at the cost of a second object store per
repo. ``docs/future.md`` has the note.

All git operations here run as plain subprocesses against host paths in the
harness process; only the agent's *tools* run sandboxed inside the
workspace.

Also home to the generated ``git`` shim (:func:`git_shim_script`), the
best-effort filter that keeps the agent's shell tool from mutating the repo
(only read-only inspection subcommands and ``git config --get/--list``
pass; add, commit, checkout, branch, tag, reset, switch, rebase, push,
config writes, … are blocked). It is a UX safety net, not a security
boundary: the user's history is theirs; the agent edits files, not refs.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import tempfile
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

_GIT_TIMEOUT = 30.0

class GitWorktreeError(Exception):
    """A git operation the harness manages failed; surfaced to the user."""


class RevertConflict(GitWorktreeError):
    """A turn's files changed after the turn ended, so it cannot be undone.

    Raised by the caller (``WorkspaceState.undo``) with the conflicting
    paths, so the UI can report them instead of silently picking a side;
    the workspace is left untouched."""


@dataclass(slots=True)
class RepoInfo:
    """What we need to know about the user's repo at session start."""

    root: Path
    git_common_dir: Path  # shared .git dir — bound into the sandbox read-only
    # Tracked-with-no-.git mode: the git dir lives outside the workspace and
    # every git call sets GIT_DIR/GIT_WORK_TREE explicitly.
    external: bool = False


class RepoStore(Protocol):
    """What the tree helpers here need from a tracked workspace: the repo
    root, the env that reaches its object store, its persistent snapshot
    index, and the oversized exclusion set.

    :class:`xarness.workspace.WorkspaceState` is the only implementation;
    the protocol keeps this module free of a circular import."""

    root: Path
    oversized: tuple[str, ...]

    @property
    def index_path(self) -> Path: ...

    def git_env(self) -> dict[str, str]: ...

    async def snapshot(self) -> str:
        """Snapshot the worktree now, without locking (the caller holds the
        workspace lock) and without moving the head tree."""


def git_env(root: Path, git_dir: Path, external: bool) -> dict[str, str]:
    """Env for a git subprocess in a workspace: external mode addresses the
    snapshot store via GIT_DIR/GIT_WORK_TREE; otherwise cwd picks up the
    worktree's own .git."""
    if not external:
        return {**os.environ}
    return {**os.environ, "GIT_DIR": str(git_dir), "GIT_WORK_TREE": str(root)}


class TurnCheckpoint(BaseModel):
    """One turn's before/after workspace trees.

    ``before_tree`` is the state when the turn's user message was sent,
    ``after_tree`` the state once the turn's edits finished. Reversing the
    turn means applying ``before_tree - after_tree`` to the live workspace.

    A pydantic model (it persists inside the session's state row), keyed by
    the opening user message's id — the messages themselves stay immutable.
    """

    before_tree: str | None = None
    after_tree: str | None = None
    # The workspace epoch the turn opened in. An accept/reject bumps the
    # epoch, making the turn (and any checkpoint taken earlier) no longer
    # revertible.
    epoch: int = 0
    # False until the turn's finally ran ``finish_turn``. Untracked turns
    # (no git info, or the snapshot failed) close with no trees, so they
    # never look open forever.
    closed: bool = False


# ----------------------------------------------------------------------
# subprocess plumbing


async def _run_git(
    args: list[str], cwd: Path | None = None, env: dict[str, str] | None = None,
    input_bytes: bytes | None = None,
) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdin=asyncio.subprocess.PIPE if input_bytes is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input_bytes), timeout=_GIT_TIMEOUT,
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise GitWorktreeError(f"git {' '.join(args)} timed out") from None
    return (
        proc.returncode or 0,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


# ----------------------------------------------------------------------
# repo detection


async def detect_repo(path: Path) -> RepoInfo | None:
    """Repo info for ``path``, or None if it isn't a git repo.

    """
    path = Path(path).resolve()
    rc, out, _ = await _run_git(
        ["rev-parse", "--is-inside-work-tree", "--show-toplevel",
         "--git-common-dir"],
        cwd=path,
    )
    if rc != 0 or out.splitlines()[0].strip() != "true":
        return None
    _, top, common = out.splitlines()
    common_dir = Path(common.strip())
    if not common_dir.is_absolute():
        common_dir = (path / common_dir).resolve()
    return RepoInfo(
        root=Path(top.strip()).resolve(),
        git_common_dir=common_dir,
    )


def _managed_store_dir(workspace: Path) -> Path:
    """Xarness-owned git dir for ``workspace``, outside the workspace so
    tracking a gitless directory never drops a ``.git`` into it. Keyed on
    the resolved path (snapshots are content-addressed trees, no refs, so
    reuse is harmless).

    Only used when the workspace has no repo: a repo's sessions snapshot
    into that repo's own object store instead (see the module docstring).
    """
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    key = hashlib.sha1(str(workspace).encode()).hexdigest()[:16]
    return Path(state) / "xarness" / "repos" / key


async def init_external_repo(workspace: Path) -> RepoInfo:
    """Track a workspace that has no repo WITHOUT creating ``.git`` in it:
    init a store under xarness's state dir; later calls address it via
    :meth:`WorkspaceState.git_env`. The store only ever holds unreachable
    trees."""
    workspace = Path(workspace).resolve()
    store = _managed_store_dir(workspace)
    store.mkdir(parents=True, exist_ok=True)
    rc, _, err = await _run_git(["init", "-q", "-b", "main"], cwd=store)
    if rc != 0:  # older git without -b
        rc, _, err = await _run_git(["init", "-q"], cwd=store)
    if rc != 0:
        raise GitWorktreeError(
            f"could not initialize the snapshot store for {workspace}: {err.strip()}"
        )
    git_dir = store / ".git"
    await _run_git(
        ["config", "xarness.managed", "true"],
        cwd=store, env={**os.environ, "GIT_DIR": str(git_dir)},
    )
    return RepoInfo(root=workspace, git_common_dir=git_dir, external=True)


# ----------------------------------------------------------------------
# tree snapshots (checkpoints + diff base)


@contextmanager
def _temp_index():
    """Yield the path of a throwaway index file for GIT_INDEX_FILE,
    removing it afterwards.

    The path must NOT exist while git runs — git refuses to read an empty
    file as an index and wants to create it itself."""
    fd, name = tempfile.mkstemp(prefix="xarness-index-")
    os.close(fd)
    os.unlink(name)
    try:
        yield name
    finally:
        try:
            os.unlink(name)
        except OSError:
            pass


def index_path(workspace: Path) -> Path:
    """The workspace's persistent snapshot index.

    A fresh index per snapshot would re-hash every file in the worktree; git's
    stat cache makes a persistent one incremental."""
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    key = hashlib.sha1(str(workspace).encode()).hexdigest()[:16]
    directory = Path(state) / "xarness" / "indexes"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{key}.index"


# ----------------------------------------------------------------------
# ignore rules

# Ignore rules the harness applies on top of the repo's own, in gitignore
# syntax. ``target/``/``target-sandbox/`` are cargo's build trees and are
# gated on Cargo.toml (see :func:`harness_ignores`); everything listed here
# is ignored everywhere. Media/binary types are never tracked: their diffs
# are meaningless and their blobs bloat the object store. Ignored paths never
# enter a snapshot tree (see /diff's auto groups for how they are reported).
HARNESS_IGNORES: tuple[str, ...] = (
    "node_modules/", "dist/", "build/", "__pycache__/", ".venv/", ".next/",
    "*.pyc", "*.o", "*.class",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.bmp", "*.tiff",
    "*.ico", "*.icns", "*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm",
    "*.mp3", "*.wav", "*.flac", "*.ogg", "*.pdf", "*.zip", "*.tar",
    "*.gz", "*.tgz", "*.bz2", "*.xz", "*.7z", "*.rar", "*.iso", "*.exe",
    "*.dll", "*.so", "*.dylib", "*.bin", "*.jar", "*.a", "*.woff",
    "*.woff2", "*.ttf", "*.otf", "*.eot", "*.db", "*.sqlite", "*.sqlite3",
    "*.psd", "*.blend",
    ".xarness/",
)

# Cargo's own ``target/`` and the sandbox's redirected ``target-sandbox/``
# (the sandbox sets CARGO_TARGET_DIR): both are build output, not source.
_CARGO_IGNORES: tuple[str, ...] = ("target/", "target-sandbox/")

# Files larger than this are excluded from tracking (snapshots/diffs/undo).
_MAX_TRACKED_SIZE = 1024 * 1024

# The all-zero sha diff-tree reports for a path deleted between two trees.
_DELETED_SHA = "0" * 40

# The user's own ignore file, in gitignore syntax (ignore-only: no
# re-include), resolved on every snapshot.
_XARNESS_IGNORE = ".xarness/ignore"


def harness_ignores(root: Path) -> tuple[str, ...]:
    """The harness's ignore rules for ``root``, in gitignore syntax.

    The cargo build trees are gated: a directory that merely happens to be
    called ``target`` in a non-Cargo project is still tracked."""
    if (root / "Cargo.toml").is_file():
        return (*_CARGO_IGNORES, *HARNESS_IGNORES)
    return HARNESS_IGNORES


_harness_ignore_file: Path | None = None


def harness_ignore_file() -> Path:
    """Path of the process-wide file holding :data:`HARNESS_IGNORES`, written
    once.

    ``git add`` has no ``--exclude-from`` (its rules go in as pathspecs, see
    :func:`_harness_pathspecs`), but the ignored-path listings do need a
    pattern file: ``git check-ignore`` cannot read an arbitrary file either.
    The Cargo-gated entries are added per call with ``--exclude``, since the
    gate depends on the workspace, not the process."""
    global _harness_ignore_file
    if _harness_ignore_file is None or not _harness_ignore_file.is_file():
        fd, name = tempfile.mkstemp(prefix="xarness-ignore-", suffix=".gitignore")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("\n".join(HARNESS_IGNORES) + "\n")
        _harness_ignore_file = Path(name)
    return _harness_ignore_file


_GLOB_META = re.compile(r"([*?\[])")


def _escape_glob(path: str) -> str:
    """Escape glob metacharacters so a path is matched literally by git's
    glob pathspec magic (``x[1].dat`` must not become a character class)."""
    return _GLOB_META.sub(r"\\\1", path)


def _exclude_pathspecs(path: str) -> list[str]:
    """``:(exclude,glob)`` pathspecs matching one concrete path at any depth,
    as a file and as a directory's contents (the caller does not know
    which one it is: an ignore listing collapses directories to a bare
    name)."""
    escaped = _escape_glob(path)
    return [f":(exclude,glob)**/{escaped}", f":(exclude,glob)**/{escaped}/**"]


def _harness_pathspecs(root: Path) -> list[str]:
    """The harness ignore rules as excluded ``git add`` pathspecs.

    ``:(exclude,literal)`` is not usable here: naming a path in a format git
    *also* considers a pathspec makes ``git add`` exit 1 with "paths are
    ignored — use -f" for exactly the paths being excluded. The ``**/``
    glob form is path write-protected instead of path-matching, and works
    for a directory that does not exist yet (the agent's first build)."""
    specs: list[str] = []
    for rule in harness_ignores(root):
        if rule.endswith("/"):
            specs.append(f":(exclude,glob)**/{_escape_glob(rule[:-1])}/**")
        elif rule.startswith("*."):
            specs.append(f":(exclude,glob)**/{rule}")
        else:
            specs.extend(_exclude_pathspecs(rule))
    return specs


async def xarness_ignored_paths(
    root: Path, env: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """Repo-root-relative paths ``.xarness/ignore`` ignores right now
    (directories collapsed to one entry, no trailing slash).

    ``git check-ignore`` cannot read an arbitrary pattern file, so the file is
    resolved through ``ls-files --exclude-from``. Called on every snapshot, so
    editing it mid-session takes effect on the next one."""
    if not (root / _XARNESS_IGNORE).is_file():
        return ()
    rc, out, _ = await _run_git(
        ["ls-files", "-o", "-i", "--exclude-from", _XARNESS_IGNORE,
         "--directory", "-z"],
        cwd=root, env=env,
    )
    if rc != 0:
        return ()
    return tuple(p.rstrip("/") for p in out.split("\0") if p)


async def ignored_listing(
    root: Path, env: dict[str, str] | None = None, *, collapsed: bool = True,
) -> tuple[str, ...]:
    """The workspace's ignored paths right now.

    Collapsed per directory by default (about a millisecond on a large
    tree); ``collapsed=False`` returns every file, for the per-group counts
    the diff payload reports. Harness defaults and ``.xarness/ignore`` are
    applied as extra exclude files/patterns; repo rules (``.gitignore``
    including nested ones, ``.git/info/exclude``, global excludes) come from
    ``--exclude-standard``."""
    args = [
        "ls-files", "-o", "-i", "--exclude-standard",
        "--exclude-from", str(harness_ignore_file()),
    ]
    if (root / "Cargo.toml").is_file():
        args += [f"--exclude={rule}" for rule in _CARGO_IGNORES]
    if (root / _XARNESS_IGNORE).is_file():
        args += ["--exclude-from", _XARNESS_IGNORE]
    if collapsed:
        args.append("--directory")
    args.append("-z")
    rc, out, _ = await _run_git(args, cwd=root, env=env)
    if rc != 0:
        return ()
    return tuple(p.rstrip("/") for p in out.split("\0") if p)


def oversized_paths(root: Path, limit: int = _MAX_TRACKED_SIZE) -> tuple[str, ...]:
    """Repo-root-relative paths of files over ``limit`` bytes, walked once
    when the workspace is opened. Best-effort; skips the harness-ignored
    directories (cargo's build tree, node_modules, …) and symlinks."""
    skip = {rule.rstrip("/") for rule in harness_ignores(root) if rule.endswith("/")}
    skip.add(".git")
    oversized: list[str] = []
    stack = [root]
    while stack:
        try:
            entries = list(os.scandir(stack.pop()))
        except OSError:
            continue
        for entry in entries:
            rel = Path(entry.path).relative_to(root).as_posix()
            if entry.is_symlink():
                continue
            if entry.is_dir(follow_symlinks=False):
                if rel not in skip and entry.name not in skip:
                    stack.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                try:
                    if entry.stat(follow_symlinks=False).st_size > limit:
                        oversized.append(rel)
                except OSError:
                    pass
    return tuple(sorted(oversized))


async def _oversized_in_tree(
    root: Path, tree: str, env: dict[str, str],
    previous: str | None = None, limit: int = _MAX_TRACKED_SIZE,
) -> list[tuple[str, str, str | None]]:
    """Oversized entries in ``tree``, over ``limit`` bytes.

    Each entry is ``(path, mode, previous_blob_sha)``: the mode and blob of
    the same path in ``previous`` (None when the path is new there), which
    is what lets the caller put a grown file *back* to the state it had
    before it ballooned — see :func:`snapshot_tree`.

    Sizes come from the tree itself, so this costs no file reads. With
    ``previous`` (the prior snapshot's tree) only the changed blobs are
    stat'ed — ``diff-tree`` finds them, ``cat-file --batch-check`` sizes
    them — so the per-snapshot cost scales with the edit, not the tree.
    Without it the whole tree is listed once (workspace open). The walk in
    :func:`oversized_paths` still catches files that were already big (so
    they are never written to the object store); this catches the ones that
    grew past the limit mid-session."""
    by_sha: dict[str, list[tuple[str, str, str | None]]] = {}
    if previous == tree:
        # Nothing changed since the previous snapshot, so nothing can have
        # grown past the limit: skip the whole ls-tree/cat-file pass (this
        # is the common case for the diff loop and read-only run_bash).
        return []
    if previous is not None:
        # --raw carries both blob shas per entry (NUL-separated, one meta
        # record plus one path record), so a path containing a newline cannot
        # corrupt the stream: no path ever feeds the batch-check input.
        rc, out, _ = await _run_git(
            ["diff-tree", "-r", "--raw", "--no-renames", "-z", previous, tree],
            cwd=root, env=env,
        )
        if rc != 0:
            return []
        fields = out.split("\0")
        for i in range(0, len(fields) - 1, 2):
            meta = fields[i]
            path = fields[i + 1]
            if not meta.startswith(":"):
                continue
            parts = meta[1:].split()
            if len(parts) >= 4 and parts[3] != _DELETED_SHA:
                old_sha = None if parts[2] == _DELETED_SHA else parts[2]
                by_sha.setdefault(parts[3], []).append(
                    (path, parts[0], old_sha)
                )
    else:
        rc, out, _ = await _run_git(
            ["ls-tree", "-r", "-z", tree], cwd=root, env=env,
        )
        if rc != 0:
            return []
        for entry in out.split("\0"):
            parts = entry.split() if entry else []
            if len(parts) >= 3 and parts[1] == "blob":
                # ls-tree -z separates path from metadata with a TAB; the
                # split() above keeps the metadata fields intact.
                by_sha.setdefault(parts[2], []).append(
                    (entry.split("\t", 1)[1], parts[0], None)
                )
    if not by_sha:
        return []
    # batch-check prints newline-separated lines keyed by the input sha.
    rc, out, _ = await _run_git(
        ["cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
         "--buffer"],
        cwd=root, env=env,
        input_bytes="".join(f"{sha}\n" for sha in by_sha).encode(),
    )
    if rc != 0:
        return []
    oversized: list[tuple[str, str, str | None]] = []
    seen: set[str] = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[1] != "blob":
            continue
        size = _safe_size(parts[2])
        if size is not None and size > limit:
            for path, mode, old_sha in by_sha.get(parts[0], ()):
                if path not in seen:
                    seen.add(path)
                    oversized.append((path, mode, old_sha))
    return sorted(oversized)


def _safe_size(text: str) -> int | None:
    try:
        return int(text)
    except ValueError:
        return None


async def snapshot_tree(
    root: Path, exclude: Iterable[str] = (),
    env_base: dict[str, str] | None = None, index: Path | None = None,
    previous: str | None = None,
    oversized_out: list[str] | None = None,
) -> str:
    """Tree sha capturing the workspace's *current* state.

    Built in a scratch index (``index`` when given, else a temp file):
    ``git add -A`` stages tracked changes, uncommitted changes, and
    untracked-not-ignored files (the user's real index is untouched), then
    ``write-tree`` records it as a tree object. Ignored files are excluded —
    they are not reverted by /undo and don't show in /diff.

    Exclusion is resolved on every snapshot (nothing is frozen per
    session): git's own ignore rules (``git add -A`` already honours
    ``.gitignore``, ``.git/info/exclude`` and global excludes), the harness
    defaults as :func:`_harness_pathspecs`, ``.xarness/ignore`` resolved to
    concrete paths, and ``exclude`` (paths the caller must keep out —
    oversized files, and whatever grew past the limit mid-session).

    ``env_base`` carries GIT_DIR/GIT_WORK_TREE for external snapshot stores.

    ``oversized_out``, when given, collects the repo-root-relative paths
    that grew past the size limit and were left out of the snapshot — the
    caller adds them to the workspace's exclusion set so the next ``git
    add`` does not re-hash the same huge blob all over again.
    """
    root = Path(root).resolve()

    @contextmanager
    def _reusable_index():
        """A persistent index keeps git's stat cache (and thus stops
        re-hashing unchanged files); the caller owns it, so it is left in
        place and pruned by git itself."""
        if index is None:
            with _temp_index() as temp:
                yield temp
        else:
            yield str(index)

    async def _capture(index_file: str) -> str:
        env = {**(env_base if env_base is not None else os.environ), "GIT_INDEX_FILE": index_file}
        # Harness rules and .xarness/ignore are resolved now, not pinned: an
        # edit to either takes effect on this snapshot.
        pathspecs = [
            *_harness_pathspecs(root),
            *(
                spec
                for p in (*exclude, *await xarness_ignored_paths(root, env_base))
                for spec in _exclude_pathspecs(p)
            ),
        ]
        args = ["add", "-A", "--", *pathspecs]
        rc, _, err = await _run_git(args, cwd=root, env=env)
        if rc != 0:
            raise GitWorktreeError(f"could not snapshot the workspace: {err.strip()}")
        rc, out, err = await _run_git(["write-tree"], cwd=root, env=env)
        if rc != 0:
            raise GitWorktreeError(f"could not write the snapshot tree: {err.strip()}")
        tree = out.strip()
        oversized = await _oversized_in_tree(root, tree, env, previous)
        if oversized and oversized_out is not None:
            oversized_out.extend(path for path, _, _ in oversized)
        if oversized:
            # Files that grew past the size limit mid-session: keep them out
            # of the snapshot rather than tracking (and diffing, and
            # reverting) a multi-hundred-megabyte artifact every turn.
            #
            # They are *rewound* to their previous snapshot entry, not
            # deleted: plain ``git rm --cached`` would make /diff report a
            # deletion the agent never made, and pin the path as ignored with
            # the baseline still holding it. A path that is new since
            # ``previous`` is un-indexed instead (mode 0).
            #
            # ``update-index --index-info`` takes NUL-terminated
            # ``<mode> <sha> <stage>\t<path>`` records, so paths are never
            # parsed as pathspecs — no globbing (``x[1].txt`` must not match
            # ``x1.txt``), no quoting, and newlines in a name are harmless.
            # (``git rm`` has no ``-z``/``--stdin`` at all; only
            # ``checkout-index`` does.)
            records = [
                (f"{mode} {old_sha} 0\t{path}" if old_sha is not None
                 else f"0 {_DELETED_SHA} 0\t{path}")
                for path, mode, old_sha in oversized
            ]
            payload = b"".join(r.encode() + b"\0" for r in records)
            rc, _, err = await _run_git(
                ["update-index", "--add", "-z", "--index-info"],
                cwd=root, env=env, input_bytes=payload,
            )
            if rc != 0:
                raise GitWorktreeError(
                    f"could not drop oversized files from the snapshot: {err.strip()}"
                )
            rc, out, err = await _run_git(["write-tree"], cwd=root, env=env)
            if rc != 0:
                raise GitWorktreeError(f"could not write the snapshot tree: {err.strip()}")
            tree = out.strip()
        return tree

    with _reusable_index() as index_file:
        # A persistent index is shared by every snapshot of the workspace:
        # the caller's workspace lock (WorkspaceState.lock) serializes the
        # add/write-tree sequence, or two snapshots race on index.lock.
        return await _capture(index_file)


async def tree_entry(
    repo: RepoStore, tree: str, path: str,
) -> tuple[str, str] | None:
    """``(mode, blob sha)`` for one exact path in a tree, or None."""
    rc, out, err = await _run_git(
        ["ls-tree", "-z", tree, "--", f":(literal){path}"],
        cwd=repo.root, env=repo.git_env(),
    )
    if rc != 0:
        raise GitWorktreeError(f"could not read {tree[:8]}: {err.strip()}")
    record = out.split("\0", 1)[0]
    if not record:
        return None
    meta, _, _ = record.partition("\t")
    mode, kind, sha = meta.split()
    return (mode, sha) if kind == "blob" else None


def worktree_entry(root: Path, path: str) -> tuple[str, str] | None:
    """``(mode, blob sha)`` of one worktree path, or None when it is absent.

    The worktree twin of :func:`tree_entry`: git hashes a file's content the
    same way whether it is read from a tree or from disk, so what a turn
    left behind can be compared with the entry its after tree recorded
    without touching the index. Symlinks hash their target path."""
    target = root / path
    try:
        if target.is_symlink():
            data, mode = os.readlink(target).encode(), "120000"
        else:
            data = target.read_bytes()
            mode = "100755" if os.access(target, os.X_OK) else "100644"
    except OSError:
        return None
    return mode, hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


async def restore_paths(repo: RepoStore, tree: str, paths: list[str]) -> None:
    """Write ``paths`` from ``tree`` over the worktree via a scratch index."""
    if not paths:
        return
    payload = b"\0".join(p.encode() for p in paths) + b"\0"
    with _temp_index() as index:
        env = repo.git_env()
        env["GIT_INDEX_FILE"] = index
        rc, _, err = await _run_git(["read-tree", tree], cwd=repo.root, env=env)
        if rc != 0:
            raise GitWorktreeError(
                f"could not restore checkpoint {tree[:8]}: {err.strip()}"
            )
        rc, _, err = await _run_git(
            ["checkout-index", "-f", "-z", "--stdin"],
            cwd=repo.root, env=env, input_bytes=payload,
        )
        if rc != 0:
            raise GitWorktreeError(
                f"could not restore checkpoint {tree[:8]}: {err.strip()}"
            )


async def splice_tree(
    repo: RepoStore, base_tree: str, other_tree: str, paths: list[str],
) -> str:
    """``base_tree`` with ``paths`` replaced by their ``other_tree`` content.

    One tree sha, no per-path override structure: read the baseline into a
    scratch index, take the accepted paths from ``other_tree`` in batches,
    drop the ones it no longer has, and write the tree back. Paths not in
    ``paths`` (drift) keep their baseline entry."""
    if not paths:
        return base_tree
    env = repo.git_env()
    with _temp_index() as index:
        env = {**env, "GIT_INDEX_FILE": index}
        rc, _, err = await _run_git(
            ["read-tree", base_tree], cwd=repo.root, env=env,
        )
        if rc != 0:
            raise GitWorktreeError(
                f"could not read the baseline {base_tree[:8]}: {err.strip()}"
            )
        found: dict[str, tuple[str, str]] = {}
        in_base: set[str] = set()
        batch = 100
        for start in range(0, len(paths), batch):
            specs = [f":(literal){p}" for p in paths[start:start + batch]]
            for tree, into in ((other_tree, found), (base_tree, None)):
                rc, out, err = await _run_git(
                    ["ls-tree", "-z", tree, "--", *specs],
                    cwd=repo.root, env=env,
                )
                if rc != 0:
                    raise GitWorktreeError(
                        f"could not read tree {tree[:8]}: {err.strip()}"
                    )
                for record in out.split("\0"):
                    if not record:
                        continue
                    meta, _, path = record.partition("\t")
                    mode, kind, sha = meta.split()
                    if kind != "blob":
                        continue
                    if into is None:
                        in_base.add(path)
                    else:
                        into[path] = (mode, sha)
        records = []
        for path in paths:
            entry = found.get(path)
            if entry is None:
                if path in in_base:
                    records.append(f"0 {_DELETED_SHA} 0\t{path}")
            else:
                records.append(f"{entry[0]} {entry[1]} 0\t{path}")
        payload = b"".join(r.encode() + b"\0" for r in records)
        rc, _, err = await _run_git(
            ["update-index", "--add", "-z", "--index-info"],
            cwd=repo.root, env=env, input_bytes=payload,
        )
        if rc != 0:
            raise GitWorktreeError(
                f"could not splice the accepted tree: {err.strip()}"
            )
        rc, out, err = await _run_git(["write-tree"], cwd=repo.root, env=env)
        if rc != 0:
            raise GitWorktreeError(f"could not write the tree: {err.strip()}")
        return out.strip()


async def revert_to_tree(repo: RepoStore, tree: str, current: str | None = None) -> None:
    """Restore the workspace to a snapshot tree, discarding every file change
    made since it: modifications, deletions, and newly created files.

    One ``diff-tree -z --name-status`` lists what differs; only paths that
    actually differ are touched (rewriting the whole tree would bump every
    mtime and wake every watcher): files created since the checkpoint are
    deleted, the rest are written back from the tree through a scratch index.

    ``current`` is the workspace's current snapshot tree, when the caller
    already has it (e.g. :func:`revert_turns`); passing it skips a redundant
    re-snapshot. Only reverts files — anything run_bash did outside the
    workspace (installs, background jobs, network calls) is not undone."""
    root = repo.root
    if current is None:
        current = await snapshot_tree(
            root, exclude=repo.oversized, env_base=repo.git_env(),
            index=repo.index_path,
        )
    if current == tree:
        return

    rc, out, err = await _run_git(
        ["diff-tree", "-r", "-z", "--name-status", current, tree],
        cwd=root, env=repo.git_env(),
    )
    if rc != 0:
        raise GitWorktreeError(f"could not compare against the checkpoint: {err.strip()}")

    delete: list[str] = []
    restore: list[str] = []
    for status, path in _parse_status_z(out):
        # current→tree: D = exists now, gone in the target (created since the
        # checkpoint); anything else has to be written back.
        (delete if status.startswith("D") else restore).append(path)

    for rel in delete:
        try:
            (root / rel).unlink()
        except OSError:
            pass
    await restore_paths(repo, tree, restore)

def _parse_status_z(out: str) -> list[tuple[str, str]]:
    """Parse ``git diff/diff-tree -z --name-status`` into (status, path) pairs.

    NUL-separated, so paths with spaces, quotes or non-ASCII bytes survive
    verbatim (without ``-z`` git quotes them, and the path then does not
    exist on disk). Renames/copies carry two paths; the destination is the
    one callers mean."""
    fields = out.split("\0")
    changes: list[tuple[str, str]] = []
    index = 0
    while index < len(fields):
        status = fields[index].strip()
        index += 1
        if not status:
            continue
        if status[0] in "RC":
            if index + 1 >= len(fields):
                break
            changes.append((status, fields[index + 1]))
            index += 2
        else:
            if index >= len(fields):
                break
            changes.append((status, fields[index]))
            index += 1
    return changes


def _parse_numstat_z(out: str) -> list[DiffFile]:
    """Parse ``git diff --numstat -z`` into :class:`DiffFile` rows.

    Renames are ``add\\tdel\\t\\0old\\0new\\0``; the destination path is
    reported (a rename shows up as one row, like the UI expects)."""
    fields = out.split("\0")
    files: list[DiffFile] = []
    index = 0
    while index < len(fields):
        entry = fields[index]
        index += 1
        if not entry:
            continue
        parts = entry.split("\t")
        if len(parts) < 3:
            continue
        adds, dels, path = parts[0], parts[1], "\t".join(parts[2:])
        if not path:
            if index + 1 >= len(fields):
                break
            path = fields[index + 1]
            index += 2
        files.append(DiffFile(
            path=path,
            additions=0 if adds == "-" else int(adds or 0),
            deletions=0 if dels == "-" else int(dels or 0),
        ))
    return files


async def _diff_tree_pair(repo: RepoStore, base_tree: str, other_tree: str, *extra: str) -> str:
    """Raw ``git diff`` between two snapshot trees.

    Both sides are tree objects, so the user's index is never touched. ``-M``
    is passed explicitly so rename detection works regardless of the user's
    diff.renames config. Paths are repo-root relative; nothing is filtered."""
    root = repo.root
    rc, out, err = await _run_git(
        ["diff", "-M", base_tree, other_tree, *extra],
        cwd=root, env=repo.git_env(),
    )
    if rc != 0:
        raise GitWorktreeError(f"git diff failed: {err.strip()}")
    return out


async def _diff_trees(repo: RepoStore, *extra: str, current: str | None = None) -> str:
    """Diff the workspace's current state (tracked, uncommitted, and
    untracked-not-ignored files) against the session's baseline tree."""
    if current is None:
        current = await repo.snapshot()
    return await _diff_tree_pair(repo, repo.baseline_tree, current, *extra)


# Snapshot trees are immutable, so a tree pair's diff never changes once the
# objects exist. Cache it so /diff stays cheap on long sessions. The cached
# strings are immutable, so callers can't corrupt the cache by mutating what
# they get back. Bounded (oldest entry evicted) so a long-lived process
# doesn't grow without end.
_CACHE_LIMIT = 512
_numstat_cache: dict[tuple[str, str], tuple[str, str]] = {}


def _cache_put(cache: dict, key, value):
    if len(cache) >= _CACHE_LIMIT:
        cache.pop(next(iter(cache)))
    cache[key] = value


async def _diff_between(repo: RepoStore, base_tree: str, other_tree: str) -> tuple[str, str]:
    """Cached ``(--numstat, --name-status)`` NUL-separated output between two
    snapshot trees. ``--name-status`` is where ``is_new`` comes from (the
    ``A`` rows), so no second ``ls-tree`` pass per tree is needed."""
    key = (base_tree, other_tree)
    cached = _numstat_cache.get(key)
    if cached is None:
        numstat = await _diff_tree_pair(repo, base_tree, other_tree, "--numstat", "-z")
        status = await _diff_tree_pair(repo, base_tree, other_tree, "--name-status", "-z")
        cached = (numstat, status)
        _cache_put(_numstat_cache, key, cached)
    return cached


async def tree_diff_stat(repo: RepoStore, base_tree: str, other_tree: str) -> DiffStat:
    """Per-file +/- counts between two snapshot trees, with ``is_new`` set
    relative to ``base_tree``."""
    numstat, status = await _diff_between(repo, base_tree, other_tree)
    files = _parse_numstat_z(numstat)
    created = {path for change, path in _parse_status_z(status) if change.startswith("A")}
    for f in files:
        f.is_new = f.path in created
    return _diff_info(files)


async def git_diff(
    repo: RepoStore, path: str | None = None, current: str | None = None,
) -> str:
    """Unified diff of the workspace's current state vs the session's
    baseline tree, optionally limited to one path. Pass ``current`` (a
    snapshot sha) when the caller already has one, so /diff <path> snapshots
    the workspace only once."""
    extra: tuple[str, ...] = ("--", path) if path else ()
    return await _diff_trees(repo, *extra, current=current)


# ----------------------------------------------------------------------
# diff summary data — plain dataclasses, deliberately independent of the
# wire protocol. The API edge maps these to protocol.DiffInfo.


@dataclass(slots=True)
class DiffFile:
    path: str
    additions: int
    deletions: int
    is_new: bool = False


@dataclass(slots=True)
class DiffStat:
    files: list[DiffFile] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0


def _diff_info(files: list[DiffFile]) -> DiffStat:
    """A DiffStat with the totals computed from ``files``."""
    return DiffStat(
        files=files,
        additions=sum(f.additions for f in files),
        deletions=sum(f.deletions for f in files),
    )


# ----------------------------------------------------------------------
# shell-command filter — best-effort UX safety net


# Deny by default: only these inspection subcommands pass. Everything else
# (add, commit, checkout, reset, switch, rebase, clean, push, update-ref, …)
# is blocked. ``reflog expire``/``delete`` mutate, so reflog is _READ_FORMS.
_ALLOWED_SUBCOMMANDS = {
    "status", "log", "diff", "show", "blame", "grep", "shortlog",
    "whatchanged", "rev-parse", "rev-list", "ls-files", "ls-tree",
    "ls-remote", "cat-file", "describe", "merge-base", "name-rev",
    "var", "version", "check-ignore", "check-attr", "for-each-ref",
    "show-ref", "count-objects", "diff-tree",
}
# Subcommands that are read-only only in some forms: the first token must be
# one of these (or there must be no token at all, e.g. bare `git branch`
# lists). A positional argument (`git branch feature`) is a mutation.
_READ_FORMS: dict[str, frozenset[str]] = {
    "branch": frozenset({
        "--list", "-l", "--show-current", "-a", "--all", "-r", "--remotes",
        "-v", "-vv", "--verbose", "-i", "--ignore-case", "--omit-empty",
    }),
    "tag": frozenset({
        "--list", "-l", "-n", "--contains", "--no-contains", "--merged",
        "--no-merged", "--points-at", "--sort", "-i", "--ignore-case",
        "-v", "--verify",
    }),
    "stash": frozenset({"list", "show"}),
    "remote": frozenset({"-v", "--verbose", "show", "get-url"}),
    "reflog": frozenset({"show"}),
}
# Subcommands whose no-argument form just lists (bare `git stash` *creates* a
# stash, so it is not here).
_BARE_IS_READ = {"branch", "tag", "remote", "reflog"}
_CONFIG_READ_FLAGS = {"--get", "--get-all", "--get-regexp", "--list", "-l"}
# Global git flags that consume a following value.
_GIT_VALUE_FLAGS = {
    "-C", "-c", "--git-dir", "--work-tree", "--namespace",
    "--super-prefix", "--exec-path",
}


def git_shim_script() -> str:
    """A POSIX-sh ``git`` wrapper that enforces the read-only policy at
    execution time.

    Installed first on PATH (in the container, and for unsandboxed sessions
    on the host), it lets statement-level git through: ``cmd1; git commit;
    cmd2`` runs cmd1, prints the block message in place, and still runs
    cmd2, while ``cmd1 && git commit && cmd2`` stops at the block as ``&&``
    should. Commands that merely mention the word git (``grep "git is
    read" …``) never invoke it. It does not *rely* on being first: the real
    git is found by skipping this file wherever it sits on PATH (see the
    lookup in the generated script).

    The allow-lists below are the only description of the policy: the shim
    is generated from the same constants the block message names.

    This is an accident guardrail, not a security boundary: a determined
    bypass (``/usr/bin/git``, ``command git``) is out of scope by design —
    the read-only ``.git`` mount is the real boundary.
    """
    value_flags = "|".join(sorted(_GIT_VALUE_FLAGS))
    always = "|".join(sorted(_ALLOWED_SUBCOMMANDS))
    config_flags = "|".join(sorted(_CONFIG_READ_FLAGS))

    def form_branch(sub: str, forms: frozenset[str], bare_ok: bool) -> str:
        pat = "|".join(sorted(forms))
        bare = '\n        if [ -z "$rest" ]; then exec "$real" "$@"; fi' if bare_ok else ""
        return (
            f"        {sub}){bare}\n"
            f"        case $rest in {pat}) exec \"$real\" \"$@\" ;; *) blocked ;; esac ;;"
        )

    form_branches = "\n".join(
        form_branch(sub, forms, sub in _BARE_IS_READ)
        for sub, forms in sorted(_READ_FORMS.items())
    )
    return f'''#!/bin/sh
# Generated by xarness.gitwork.git_shim_script(); do not edit.
# An accident guardrail, not a security boundary: PATH can be bypassed
# (/usr/bin/git, `command git`); the read-only .git mount is the real
# boundary.
real=
# Find the real git: the first executable `git` on PATH that is not this
# script. Skip by file identity, not by dropping PATH's first entry: the
# shim's own directory is not ours to assume — a session PATH is inherited
# (another prepended bin dir, or a user PATH that already carries the
# shim dir), and relaunching through PATH then resolves straight back to
# this script, which execs itself forever: a `git` pinning a core with no
# output.
if [ -n "${{XARNESS_GIT_SHIM:-}}" ]; then
    # Second entry: an earlier run already resolved the real git, and the
    # PATH lookup handed this script back. Reuse its answer (this is also
    # the fallback when `test -ef` below is unsupported) instead of
    # looping.
    if [ -x "${{XARNESS_GIT_SHIM}}" ] && ! [ "${{XARNESS_GIT_SHIM}}" -ef "$0" ] 2>/dev/null; then
        real=${{XARNESS_GIT_SHIM}}
    fi
else
    set -f  # PATH entries are literal text, never globs
    old_ifs=$IFS
    IFS=:
    for dir in ${{PATH:-}}; do
        [ -n "$dir" ] || dir=.
        candidate=$dir/git
        if [ -f "$candidate" ] && [ -x "$candidate" ] && ! [ "$candidate" -ef "$0" ] 2>/dev/null; then
            real=$candidate
            break
        fi
    done
    IFS=$old_ifs
    set +f
fi
if [ -z "$real" ] || [ ! -x "$real" ]; then
    echo "git: shim found no real git binary on PATH" >&2
    exit 127
fi
# Tells a re-entered shim which git to defer to (see the resolution above).
# Harmless to the real git, which ignores it.
XARNESS_GIT_SHIM=$real
export XARNESS_GIT_SHIM

# The first non-global-flag token is the subcommand; global flags that
# consume a value skip their value too.
sub=
pending=0
for a in "$@"; do
    if [ "$pending" = 1 ]; then pending=0; continue; fi
    case $a in
        {value_flags}) pending=1 ;;
        -*) ;;
        *) sub=$a; break ;;
    esac
done
if [ -z "$sub" ]; then
    # No subcommand: git prints its usage; nothing to guard.
    exec "$real" "$@"
fi

# First token after the subcommand, for the allow-forms check.
rest=
seen=0
for a in "$@"; do
    if [ $seen = 1 ]; then rest=$a; break; fi
    if [ "$a" = "$sub" ]; then seen=1; fi
done

blocked() {{
    echo "git: '$sub' blocked, read-only here (this command did not run)." >&2
    echo "Inspection commands (status, log, diff, show, ...) work; continue editing files normally." >&2
    exit 1
}}

case $sub in
    {always}) exec "$real" "$@" ;;
    config)
        for a in "$@"; do
            case $a in
                {config_flags}) exec "$real" "$@" ;;
            esac
        done
        blocked ;;
{form_branches}
    *) blocked ;;
esac
'''
