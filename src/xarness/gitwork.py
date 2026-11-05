"""Direct-write change tracking for agent edits.

The agent's sandbox is bound to the user's real workspace: file edits and
shell commands land in the user's files immediately — there is no worktree
and no accept/reject step. Git is used purely as an undo log:

  - at session start the workspace state is snapshotted as a tree object
    (the session's ``baseline_tree``) — the base /diff measures against
  - each turn records two snapshots: ``before_tree`` (before the turn's
    edits) and ``after_tree`` (once they finish)
  - /undo and /retry reverse the turn's own diff (before → after) onto the
    live workspace with a 3-way merge, so edits made outside the session
    since the turn are preserved rather than clobbered

Snapshots are built with a throwaway index (``GIT_INDEX_FILE`` pointing at a
temp file): the user's index, HEAD, and refs are never touched, and no
commits are created (the one ``commit-tree`` wrapper exists only to feed
``merge-tree`` during an undo). What *is* written is unreachable tree/blob
objects — and which object store they land in depends on the workspace
(see :func:`setup_tracking`):

  - **the workspace is a repo**: the user's own object store, i.e. their
    ``.git``. Nothing else in it changes, and because objects are
    content-addressed a blob git already holds is shared rather than
    duplicated (files over 1 MB are never tracked at all). The trade-off is
    that the harness's data then lives in a store whose pruning schedule is
    the user's, not ours: no ref points at these objects, so ``git gc
    --prune=now``, ``git repack -ad`` or any aggressive prune discards
    them — taking an open session's checkpoints with it. ``/diff`` and
    ``/undo`` then fail against the missing baseline until the session
    re-baselines (``/accept``, a fresh session, or the resume-drift
    prompt). ``git fsck`` reports the objects as dangling and they never
    travel with clone/fetch/push.
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

from pydantic import BaseModel, ConfigDict

_GIT_TIMEOUT = 30.0

# One lock per persistent snapshot index: concurrent snapshots (diff loop,
# finish_turn, drift detection, /accept) must not run `git add -A` on the
# same index file at once, or git fails with index.lock. Locks are created
# lazily per running loop (tests spin up many short-lived loops).
_index_locks: dict[tuple[int, str], "tuple[asyncio.AbstractEventLoop, asyncio.Lock]"] = {}


def _index_lock(index: Path) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    key = (id(loop), str(index))
    entry = _index_locks.get(key)
    if entry is None or entry[0].is_closed():
        # Rebuild (and prune) entries whose loop is gone — tests spin up
        # many short-lived loops, and a dict keyed by loop id would only
        # ever grow.
        if entry is not None and entry[0].is_closed():
            for stale in [k for k, v in _index_locks.items() if v[0].is_closed()]:
                del _index_locks[stale]
        entry = _index_locks[key] = (loop, asyncio.Lock())
    return entry[1]


class GitWorktreeError(Exception):
    """A git operation the harness manages failed; surfaced to the user."""


class RevertConflict(GitWorktreeError):
    """A turn's changes overlap unresolvably with edits made since the turn.

    Raised by the caller (controller.revert_changes) so the UI can surface
    the conflict instead of silently picking a side; the workspace is left
    untouched."""


@dataclass(slots=True)
class RepoInfo:
    """What we need to know about the user's repo at session start."""

    root: Path
    git_common_dir: Path  # shared .git dir — bound into the sandbox read-only
    # Tracked-with-no-.git mode: the git dir lives outside the workspace and
    # every git call sets GIT_DIR/GIT_WORK_TREE explicitly.
    external: bool = False


class GitInfo(BaseModel):
    """Per-session git state; persisted in the session JSON as the ``git``
    block. A pydantic model, so it round-trips through the session file and
    snapshots without a hand-written block converter."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    # Repo root the session tracks (renamed from ``workspace``: a Session's
    # ``workspace`` is the agent dir, and the two got confused). The agent
    # works in ``agent_workspace`` (the subtree when scoped).
    repo_root: Path
    git_dir: Path
    # Repo-root-relative posix subdirectory the session is scoped to ("" =
    # the whole repo). Set when the user's --workspace sits inside a bigger
    # repo: snapshots, diffs, and reverts only ever cover this subtree.
    subtree: str = ""
    # Tree sha of the workspace state at session start — the base /diff
    # measures against. A plain object-sha, not a ref or commit.
    baseline_tree: str = ""
    # The workspace state the session has last accounted for: updated at
    # turn end, on accept/keep-tracking, and after a rollback. Resume drift
    # compares against this, so changes the user already acknowledged (or
    # the session itself made) are not re-detected as external drift.
    last_seen_tree: str = ""
    # Repo-root-relative paths the repo's ignore rules flagged at session
    # start. Snapshots exclude them as explicit pathspecs (not by re-reading
    # the worktree's .gitignore), so what /diff and /undo track is pinned for
    # the whole session.
    ignored_paths: tuple[str, ...] = ()
    external: bool = False

    def mark_seen(self, tree: str) -> None:
        """Record ``tree`` as the workspace state the session has accounted
        for (turn end, accept, keep-tracking, rollback). The one writer of
        ``last_seen_tree``."""
        self.last_seen_tree = tree

    def reset_to(self, tree: str) -> None:
        """Rebaseline: the workspace is back at ``tree`` (accept/reject), so
        that is now both the accounted state and the diff base."""
        self.baseline_tree = tree
        self.last_seen_tree = tree

    @property
    def index_path(self) -> Path:
        """Persistent snapshot index for this session.

        A fresh index per snapshot would re-hash every file in the worktree,
        twice per turn; git's stat cache makes a persistent one incremental
        (and, because previously snapshotted paths are then *tracked* in it,
        it also makes the session-start ignore pinning real: a mid-session
        .gitignore edit no longer changes what is snapshotted)."""
        return _index_path(self.repo_root, self.session_id)

    def git_env(self, **extra: str) -> dict[str, str]:
        """Env for a git subprocess: external mode addresses the store via
        GIT_DIR/GIT_WORK_TREE; otherwise cwd picks up the worktree's .git."""
        env = {**os.environ}
        if self.external:
            env["GIT_DIR"] = str(self.git_dir)
            env["GIT_WORK_TREE"] = str(self.repo_root)
        env.update(extra)
        return env

    @property
    def agent_workspace(self) -> Path:
        """Directory the agent works in: the scoped subtree of the repo, or
        the repo root for a whole-repo session."""
        return self.repo_root / self.subtree if self.subtree else self.repo_root

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
    :meth:`GitInfo.git_env`. The store only ever holds unreachable trees."""
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


def _index_path(workspace: Path, session_id: str) -> Path:
    """Persistent snapshot index for one session (see GitInfo.index_path)."""
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    key = hashlib.sha1(f"{workspace}\0{session_id}".encode()).hexdigest()[:16]
    directory = Path(state) / "xarness" / "indexes"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{key}.index"


async def _git_ignored_paths(
    root: Path, env: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """Repo-root-relative paths the repo's ignore rules flag at a moment in
    time (directories collapsed to single entries), for pinning snapshot
    semantics in :class:`GitInfo`. Empty when git can't answer."""
    rc, out, _ = await _run_git(
        ["ls-files", "--others", "--ignored", "--exclude-standard", "--directory"],
        cwd=root, env=env,
    )
    if rc != 0:
        return ()
    return tuple(line.rstrip("/") for line in out.splitlines() if line)


# File types snapshots never track (noise diffs, store bloat) — applied as
# exclude pathspecs on every snapshot, no pinning needed.
_UNTRACKED_EXTS: tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".ico", ".icns",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".mp3", ".wav", ".flac", ".ogg",
    ".pdf", ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".iso",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".class", ".jar", ".o", ".a",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".db", ".sqlite", ".sqlite3", ".psd", ".blend", ".pyc",
)

# Files larger than this are excluded from tracking (snapshots/diffs/undo).
_MAX_TRACKED_SIZE = 1024 * 1024

# The all-zero sha diff-tree reports for a path deleted between two trees.
_DELETED_SHA = "0" * 40

# Directory names whose exclusion must survive the directory not existing yet:
# a pinned build tree appears on the first build, while the exclude pathspecs
# are built at snapshot time, before it exists.
_BUILD_DIR_NAMES: tuple[str, ...] = ("target", "target-sandbox")


def _cargo_build_dirs(workspace: Path, subtree: str = "") -> tuple[str, ...]:
    """Repo-root-relative build trees of a Cargo project: cargo's own
    ``target/`` and the sandbox's redirected ``target-sandbox/``.

    Both are build output, not source. ``target-sandbox`` is created by the
    agent's *own* builds (the sandbox sets CARGO_TARGET_DIR), so without this
    the first ``cargo build`` would bury /diff in object files and make
    /undo revert a tree full of them. Gated on ``Cargo.toml``: a directory
    that merely happens to be called ``target`` elsewhere is still tracked.
    """
    if not (workspace / "Cargo.toml").is_file():
        return ()
    return tuple(f"{subtree}/{name}" if subtree else name
                 for name in _BUILD_DIR_NAMES)


def _oversized_paths(
    root: Path, subtree: str, ignored: Iterable[str],
    limit: int = _MAX_TRACKED_SIZE,
) -> tuple[str, ...]:
    """Repo-root-relative paths of files over ``limit`` bytes, pinned into
    the session's exclusion set (so mid-session growth past the limit keeps
    the semantics of one that started large). Best-effort walk of the scoped
    subtree, skipping pinned dirs and symlinks."""
    base = root / subtree if subtree else root
    skip_dirs = {p for p in ignored if (root / p).is_dir()}
    oversized: list[str] = []
    stack = [base]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            rel = Path(entry.path).relative_to(root).as_posix()
            if entry.is_symlink():
                continue
            if entry.is_dir(follow_symlinks=False):
                if rel not in skip_dirs and rel != ".git":
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
    Without it the whole tree is listed once (session start). The
    session-start pinning in :func:`_oversized_paths` still catches files
    that were already big (so they are never written to the object store);
    this catches the ones that grew past the limit mid-session."""
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
    root: Path, subtree: str = "", exclude: Iterable[str] = (),
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

    Exclusion is three-layered: git's own ignore rules; ``exclude``
    (session-start paths, pinned as explicit pathspecs so mid-session
    .gitignore edits can't change what's tracked); and the always-on
    :data:`_UNTRACKED_EXTS` media/binary excludes.

    With ``subtree`` (repo-root relative), only that subtree is captured;
    paths stay repo-root relative. ``env_base`` carries GIT_DIR/
    GIT_WORK_TREE for external snapshot stores.

    ``oversized_out``, when given, collects the repo-root-relative paths
    that grew past the size limit and were left out of the snapshot — the
    caller adds them to the session's ignore list so the next ``git add``
    does not re-hash the same huge blob all over again.
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
        args = ["add", "-A"]
        pathspecs: list[str] = []
        if subtree:
            pathspecs.append(subtree)
        # Exclusions are ``**/``-anchored glob pathspecs rather than plain
        # ``:(exclude)<path>``: a literal pathspec that names an ignored path
        # makes ``git add`` fail with "paths are ignored … use -f" (the very
        # paths we are excluding). The ``**/`` form avoids that and matches
        # the path at any depth.
        for p in exclude:
            target_path = root / p
            # A pinned build tree may not exist yet (the session creates it by
            # building), so match it as a directory either way; anything else
            # is classified by what is on disk.
            as_dir = ((target_path.is_dir() and not target_path.is_symlink())
                      or Path(p).name in _BUILD_DIR_NAMES)
            kind = "/**" if as_dir else ""
            pathspecs.append(f":(exclude,glob)**/{p}{kind}")
        for ext in _UNTRACKED_EXTS:
            pathspecs.append(f":(exclude,glob)**/*{ext}")
        if pathspecs:
            args += ["--", *pathspecs]
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
        if index is None:
            # A throwaway temp index cannot collide with anyone.
            return await _capture(index_file)
        # A persistent index is shared by every snapshot of the session:
        # serialize the add/write-tree sequence, or concurrent callers
        # (diff loop, finish_turn, drift check, /accept) race on index.lock.
        async with _index_lock(Path(index_file)):
            return await _capture(index_file)


async def setup_tracking(
    workspace: Path,
    session_id: str,
    allow_init: bool = True,
) -> tuple[GitInfo | None, list[str]]:
    """One-stop session-start setup: detect the repo (or initialize one, if
    allowed, so tracking works with zero setup) and snapshot the workspace
    state as the session's baseline tree.

    Returns ``(git_info, notes)`` — notes are human-readable lines for the
    chat log / stderr. git_info is None when the workspace can't be tracked
    (no repo and init not allowed, or setup failed); the agent still edits
    the directory directly, there is just no /diff or /undo for files.

    Where the snapshot objects go follows the workspace: a repo's own
    object store when it has one (``external=False``), otherwise the
    private managed store (``external=True``, see the module docstring).
    """
    notes: list[str] = []
    workspace = Path(workspace).resolve()
    repo = await detect_repo(workspace)
    if repo is None:
        if not allow_init:
            return None, notes
        repo = await init_external_repo(workspace)
        notes.append(
            "no git repo here — tracking changes with a private snapshot "
            "store (no .git created) so the agent's changes can be "
            "tracked and undone"
        )
    subtree = ""
    if repo.root != workspace:
        # The workspace is a subdirectory of the repo: scope the session to
        # it, so snapshots/diffs/reverts (and the agent's sandbox) only ever
        # cover the directory the user actually pointed at.
        subtree = workspace.relative_to(repo.root).as_posix()
        notes.append(
            f"workspace is inside an existing repo — tracking is scoped to "
            f"{subtree}/; the agent's workspace is exactly that directory "
            "(the rest of the repo isn't mounted in the sandbox and is "
            "never touched by diffs or reverts)"
        )
    # External store: explicit GIT_DIR/GIT_WORK_TREE for the setup-time git
    # calls (later calls go through GitInfo.git_env).
    env = (
        {**os.environ, "GIT_DIR": str(repo.git_common_dir),
         "GIT_WORK_TREE": str(repo.root)}
        if repo.external else None
    )
    # Pin what snapshots track up front so ignore rules are not re-read
    # per snapshot (mid-session .gitignore edits stay out of the picture).
    # A Cargo project's build trees are pinned alongside the repo's own
    # ignore rules: they are build output the session itself creates.
    ignored = await _git_ignored_paths(repo.root, env=env)
    agent_dir = repo.root / subtree if subtree else repo.root
    ignored = tuple(dict.fromkeys((*ignored, *_cargo_build_dirs(agent_dir, subtree))))
    oversized = _oversized_paths(repo.root, subtree, ignored)
    ignored = (*ignored, *oversized)
    if oversized:
        notes.append(
            f"{len(oversized)} file(s) over 1 MB are not tracked "
            "(excluded from /diff and /undo)"
        )
    baseline = await snapshot_tree(
        repo.root, subtree, exclude=ignored, env_base=env,
        index=_index_path(repo.root, session_id),
    )
    info = GitInfo(
        session_id=session_id,
        repo_root=repo.root,
        git_dir=repo.git_common_dir,
        subtree=subtree,
        baseline_tree=baseline,
        last_seen_tree=baseline,
        ignored_paths=ignored,
        external=repo.external,
    )
    notes.append(f"tracking edits to {info.agent_workspace} with git snapshots")
    return info, notes


# ----------------------------------------------------------------------
# per-turn checkpoints + revert


async def snapshot_state(info: GitInfo) -> str:
    """:func:`snapshot_tree` for a session's workspace, with its pinned
    ignore exclusions applied.

    Files that grew past the size limit are dropped from the snapshot AND
    added to the session's ignore list: otherwise the next ``git add -A``
    re-adds them, re-hashing (and re-writing) the same huge blob every
    snapshot."""
    oversized: list[str] = []
    tree = await snapshot_tree(
        info.repo_root, info.subtree, exclude=info.ignored_paths,
        env_base=info.git_env(), index=info.index_path,
        previous=info.last_seen_tree or None, oversized_out=oversized,
    )
    if oversized:
        # ignored_paths is an immutable tuple (it feeds argv pathspecs and
        # round-trips through pydantic); rebuild it. A plain AttributeError
        # here would kill finish_turn in _run_turn's finally and skip
        # TurnEnded, so the caller's own error handling never sees it.
        known = set(info.ignored_paths)
        info.ignored_paths = (*info.ignored_paths,
                              *(p for p in oversized if p not in known))
    return tree


async def accept_changes(info: GitInfo) -> str:
    """Lock in every change made so far: snapshot the current state and make
    it the session's new baseline.

    After this, /diff is empty again (it only ever shows changes since the
    last accept), and /undo //retry can no longer revert file state past
    this point — earlier turns' checkpoints are rewritten to the accepted
    tree by the caller. Returns the accepted tree sha."""
    sha = await snapshot_state(info)
    info.baseline_tree = sha
    info.last_seen_tree = sha
    return sha


async def revert_to_tree(info: GitInfo, tree: str, current: str | None = None) -> None:
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
    root = info.repo_root
    if current is None:
        current = await snapshot_state(info)
    if current == tree:
        return

    rc, out, err = await _run_git(
        ["diff-tree", "-r", "-z", "--name-status", current, tree],
        cwd=root, env=info.git_env(),
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
    if not restore:
        return

    payload = b"\0".join(p.encode() for p in restore) + b"\0"
    with _temp_index() as index:
        env = info.git_env(GIT_INDEX_FILE=index)
        rc, _, err = await _run_git(["read-tree", tree], cwd=root, env=env)
        if rc != 0:
            raise GitWorktreeError(
                f"could not restore checkpoint {tree[:8]}: {err.strip()}"
            )
        rc, _, err = await _run_git(
            ["checkout-index", "-f", "-z", "--stdin"], cwd=root, env=env,
            input_bytes=payload,
        )
        if rc != 0:
            raise GitWorktreeError(
                f"could not restore checkpoint {tree[:8]}: {err.strip()}"
            )


async def _wrap_commit(info: GitInfo, tree: str, parent: str | None = None) -> str:
    """Wrap a bare tree sha in a throwaway, unreachable commit object.

    ``git merge-tree`` operates on commit-ish, not bare trees, so a snapshot
    tree needs a commit wrapper. No ref is created and nothing is reachable
    from HEAD or any branch — the commit is an unreferenced object in the
    object store, exactly like the snapshot trees themselves. Identity is
    supplied via the environment so this works in a repo with no
    ``user.name``/``user.email`` configured.
    """
    env = info.git_env(
        GIT_AUTHOR_NAME="xarness",
        GIT_AUTHOR_EMAIL="xarness@localhost",
        GIT_COMMITTER_NAME="xarness",
        GIT_COMMITTER_EMAIL="xarness@localhost",
    )
    args = ["commit-tree", tree, "-m", "checkpoint"]
    if parent:
        args += ["-p", parent]
    rc, out, err = await _run_git(args, cwd=info.repo_root, env=env)
    if rc != 0:
        raise GitWorktreeError(f"could not wrap checkpoint: {err.strip()}")
    return out.strip()


async def _require_merge_tree() -> None:
    """``merge-tree --write-tree`` (used to reverse a turn) needs git ≥ 2.38.

    Checked before doing anything, so an old git fails with a clear message
    instead of an obscure usage error halfway through an undo."""
    rc, out, _ = await _run_git(["version"])
    match = re.search(r"(\d+)\.(\d+)", out) if rc == 0 else None
    if match is None:
        return  # unknown format: let git report its own error
    version = (int(match.group(1)), int(match.group(2)))
    if version < (2, 38):
        raise GitWorktreeError(
            f"undoing a turn needs git >= 2.38 (merge-tree --write-tree); "
            f"this git is {version[0]}.{version[1]}"
        )


_CONFLICT_STAGE_RE = re.compile(r"^\d+ [0-9a-f]+ [123]\t(.+)$")


def _conflict_summary(out: str) -> str:
    """A short, human-readable summary of a ``merge-tree`` conflict.

    merge-tree's raw output is a tree oid, per-stage blob lines, and git's
    internal CONFLICT notices — not something to show verbatim in a toast.
    Extract just the conflicting paths."""
    paths: list[str] = []
    for line in out.splitlines():
        match = _CONFLICT_STAGE_RE.match(line)
        if match:
            path = match.group(1)
            if path not in paths:
                paths.append(path)
    if not paths:
        notices = [line.strip() for line in out.splitlines() if line.startswith("CONFLICT")]
        return "; ".join(notices)
    listed = ", ".join(paths[:5])
    if len(paths) > 5:
        listed += f", … (+{len(paths) - 5} more)"
    return f"conflicting files: {listed}"


async def _revert_turn_tree(
    info: GitInfo, before_tree: str, after_tree: str, current_tree: str,
) -> tuple[str | None, str]:
    """Compute the merge result of reversing one turn without applying it.

    Returns ``(conflict, result_tree)``; ``result_tree`` is meaningless (an
    empty string) when ``conflict`` is set."""
    if before_tree == after_tree:
        return None, current_tree
    root = info.repo_root
    base = await _wrap_commit(info, after_tree)     # state right after the turn
    theirs = await _wrap_commit(info, before_tree)  # target: state before the turn
    ours = await _wrap_commit(info, current_tree)   # live now, incl. manual edits
    rc, out, err = await _run_git(
        ["merge-tree", "--write-tree", f"--merge-base={base}", ours, theirs],
        cwd=root, env=info.git_env(),
    )
    if rc != 0:
        # Conflict: merge-tree still prints a (conflict-marked) tree, but we
        # must not apply it. Surface a readable summary and leave files alone.
        return _conflict_summary(out) or err.strip() or "merge conflict", ""
    lines = out.strip().splitlines()
    if not lines:
        raise GitWorktreeError(f"merge produced no tree: {err.strip()}")
    return None, lines[0]


async def revert_turns(info: GitInfo, turns: list[TurnCheckpoint]) -> str | None:
    """Reverse a range of turns, newest first, all-or-nothing.

    Every turn's merge is preflighted against the tree produced by the
    previous step; only when all are clean is the final tree applied to the
    workspace once. A conflict on any turn therefore leaves every file (and
    the caller's history) untouched, instead of partially reverting the
    newer turns and then reporting failure. An empty list is a no-op.
    """
    if not turns:
        return None
    await _require_merge_tree()
    current_tree = await snapshot_state(info)
    original_tree = current_tree
    for turn in reversed(turns):
        conflict, result_tree = await _revert_turn_tree(
            info, turn.before_tree, turn.after_tree, current_tree
        )
        if conflict is not None:
            return conflict
        current_tree = result_tree
    if current_tree != original_tree:
        await revert_to_tree(info, current_tree, original_tree)
    return None


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


async def _diff_tree_pair(info: GitInfo, base_tree: str, other_tree: str, *extra: str) -> str:
    """Raw ``git diff`` between two snapshot trees.

    Both sides are tree objects, so the user's index is never touched. ``-M``
    is passed explicitly so rename detection works regardless of the user's
    diff.renames config. With a scoped subtree the diff is limited to it and
    its paths are reported relative to it (git --relative)."""
    root = info.repo_root
    args = ["diff", "-M"]
    if info.subtree:
        args.append("--relative")
    args += [base_tree, other_tree]
    rc, out, err = await _run_git(
        [*args, *extra],
        cwd=root / info.subtree if info.subtree else root,
        env=info.git_env(),
    )
    if rc != 0:
        raise GitWorktreeError(f"git diff failed: {err.strip()}")
    return out


async def _diff_trees(info: GitInfo, *extra: str, current: str | None = None) -> str:
    """Diff the workspace's current state (tracked, uncommitted, and
    untracked-not-ignored files) against the session's baseline tree."""
    if current is None:
        current = await snapshot_state(info)
    return await _diff_tree_pair(info, info.baseline_tree, current, *extra)


# Snapshot trees are immutable, so a tree pair's diff never changes once the
# objects exist. Cache it (keyed on the shas plus the subtree, since paths
# are reported relative to it) so /diff stays cheap on long sessions. The
# cached strings are immutable, so callers can't corrupt the cache by
# mutating what they get back. Bounded (oldest entry evicted) so a
# long-lived process doesn't grow without end.
_CACHE_LIMIT = 512
_numstat_cache: dict[tuple[str, str, str], tuple[str, str]] = {}


def _cache_put(cache: dict, key, value):
    if len(cache) >= _CACHE_LIMIT:
        cache.pop(next(iter(cache)))
    cache[key] = value


async def _diff_between(info: GitInfo, base_tree: str, other_tree: str) -> tuple[str, str]:
    """Cached ``(--numstat, --name-status)`` NUL-separated output between two
    snapshot trees. ``--name-status`` is where ``is_new`` comes from (the
    ``A`` rows), so no second ``ls-tree`` pass per tree is needed."""
    key = (info.subtree, base_tree, other_tree)
    cached = _numstat_cache.get(key)
    if cached is None:
        numstat = await _diff_tree_pair(info, base_tree, other_tree, "--numstat", "-z")
        status = await _diff_tree_pair(info, base_tree, other_tree, "--name-status", "-z")
        cached = (numstat, status)
        _cache_put(_numstat_cache, key, cached)
    return cached


async def tree_diff_stat(info: GitInfo, base_tree: str, other_tree: str) -> DiffStat:
    """Per-file +/- counts between two snapshot trees, with ``is_new`` set
    relative to ``base_tree``."""
    numstat, status = await _diff_between(info, base_tree, other_tree)
    files = _parse_numstat_z(numstat)
    created = {path for change, path in _parse_status_z(status) if change.startswith("A")}
    for f in files:
        f.is_new = f.path in created
    return _diff_info(files)


async def diff_stat(info: GitInfo) -> DiffStat:
    """Per-file +/- line counts of everything that changed since the
    session's baseline. With a scoped subtree, limited to it, paths
    relative to it."""
    return await tree_diff_stat(info, info.baseline_tree, await snapshot_state(info))


async def attributed_diff_stat(
    info: GitInfo, turns: list[TurnCheckpoint],
    open_before: str | None = None, current: str | None = None,
) -> DiffStat:
    """Pending changes since ``baseline_tree``, tagged by source.

    ``baseline_tree`` only ever moves via :func:`accept_changes`, so this is
    always the full, correct picture of "what's changed since the user last
    accepted" — regardless of how many turns or drift-detection gaps happened
    in between. Turn diffs are used only to *tag* paths within that total as
    agent-caused; anything else in the total is drift (made outside the
    session — manual edits, or edits from a prior harness run).

    ``open_before`` is the still-open turn's before tree: without it the
    in-flight turn's edits would show as drift until end of streaming.
    """
    current = current or await snapshot_state(info)
    total = await tree_diff_stat(info, info.baseline_tree, current)

    agent_touched: set[str] = set()
    for turn in turns:
        for f in (await tree_diff_stat(info, turn.before_tree, turn.after_tree)).files:
            agent_touched.add(f.path)
    if open_before:
        for f in (await tree_diff_stat(info, open_before, current)).files:
            agent_touched.add(f.path)

    for f in total.files:
        f.source = "agent" if f.path in agent_touched else "drift"
    return total


async def git_diff(
    info: GitInfo, path: str | None = None, current: str | None = None,
) -> str:
    """Unified diff of the workspace's current state vs the session's
    baseline tree, optionally limited to one path (subtree-relative when
    the session is scoped). Pass ``current`` (a snapshot sha) when the caller
    already has one, so /diff <path> snapshots the workspace only once."""
    extra: tuple[str, ...] = ("--", path) if path else ()
    return await _diff_trees(info, *extra, current=current)


# ----------------------------------------------------------------------
# diff summary data — plain dataclasses, deliberately independent of the
# wire protocol. The API edge maps these to protocol.DiffInfo.


@dataclass(slots=True)
class DiffFile:
    path: str
    additions: int
    deletions: int
    is_new: bool = False
    source: str = ""


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
