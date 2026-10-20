"""Direct-write change tracking for agent edits.

The agent's sandbox is bound to the user's real workspace: file edits and
shell commands land in the user's files immediately — there is no worktree
and no accept/reject step. Git is used purely as an undo log:

  - at session start the workspace state is snapshotted as a tree object
    (the session's ``baseline_tree``) — the base /diff measures against
  - each turn records two snapshots on its user message: ``checkpoint_sha``
    (before the turn's edits) and ``after_tree`` (once they finish)
  - /undo and /retry reverse the turn's own diff (before → after) onto the
    live workspace with a 3-way merge, so edits made outside the session
    since the turn are preserved rather than clobbered

Snapshots are built with a throwaway index (``GIT_INDEX_FILE`` pointing at a
temp file): the user's index, HEAD, and refs are never touched, and no
commits are created. The only git state written is unreachable tree/blob
objects in the object store.

All git operations here run as plain subprocesses against host paths in the
harness process; only the agent's *tools* run sandboxed inside the
workspace.

Also home to ``check_blocked_git`` — the best-effort filter that keeps the
agent's shell tool from mutating the repo (only read-only inspection
subcommands and ``git config --get/--list`` pass; add, commit, checkout,
branch, tag, reset, switch, rebase, push, config writes, … are blocked).
It is a UX safety net, not
a security boundary: the user's history is theirs; the agent edits files,
not refs.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shlex
import tempfile
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from .protocol import DiffFileInfo, DiffInfo


_GIT_TIMEOUT = 30.0


class GitWorktreeError(Exception):
    """A git operation the harness manages failed; surfaced to the user."""


class RevertConflict(GitWorktreeError):
    """A turn's changes overlap unresolvably with edits made since the turn.

    Raised by the revert path so the caller can surface the conflict instead
    of silently picking a side; the workspace is left untouched."""


@dataclass(slots=True)
class RepoInfo:
    """What we need to know about the user's repo at session start."""

    root: Path
    git_common_dir: Path  # shared .git dir — bound into the sandbox read-write
    # Tracked-with-no-.git mode: the git dir lives outside the workspace and
    # every git call sets GIT_DIR/GIT_WORK_TREE explicitly.
    external: bool = False


@dataclass(slots=True)
class GitInfo:
    """Per-session git state; persisted in the session JSON as the ``git`` block."""

    session_id: str
    # Repo root the session tracks. The agent works in ``agent_workspace``
    # (the subtree when scoped) — the user's real directory.
    workspace: Path
    git_dir: Path
    # Repo-root-relative posix subdirectory the session is scoped to ("" =
    # the whole repo). Set when the user's --workspace sits inside a bigger
    # repo: snapshots, diffs, and reverts only ever cover this subtree.
    subtree: str = ""
    # Tree sha of the workspace state at session start — the base /diff
    # measures against. A plain object-sha, not a ref or commit.
    baseline_tree: str = ""
    # Repo-root-relative paths the repo's ignore rules flagged at session
    # start. Snapshots exclude them as explicit pathspecs (not by re-reading
    # the worktree's .gitignore), so what /diff and /undo track is pinned for
    # the whole session.
    ignored_paths: tuple[str, ...] = ()
    external: bool = False

    def git_env(self, **extra: str) -> dict[str, str]:
        """Env for a git subprocess: external mode addresses the store via
        GIT_DIR/GIT_WORK_TREE; otherwise cwd picks up the worktree's .git."""
        env = {**os.environ}
        if self.external:
            env["GIT_DIR"] = str(self.git_dir)
            env["GIT_WORK_TREE"] = str(self.workspace)
        env.update(extra)
        return env

    @property
    def agent_workspace(self) -> Path:
        """Directory the agent works in: the scoped subtree of the repo, or
        the repo root for a whole-repo session."""
        return self.workspace / self.subtree if self.subtree else self.workspace

    def to_block(self) -> dict:
        """JSON-safe dict for the session file (paths as strings)."""
        return {
            k: list(v) if isinstance(v, tuple) else (str(v) if isinstance(v, Path) else v)
            for k, v in asdict(self).items()
        }

    @classmethod
    def from_block(cls, block: dict) -> GitInfo:
        return cls(
            session_id=block["session_id"],
            workspace=Path(block["workspace"]),
            git_dir=Path(block["git_dir"]),
            subtree=block.get("subtree", ""),
            baseline_tree=block.get("baseline_tree", ""),
            ignored_paths=tuple(block.get("ignored_paths", ())),
            external=block.get("external", False),
        )


@dataclass(slots=True)
class TurnCheckpoint:
    """One turn's before/after workspace trees.

    ``before_tree`` is the state when the turn's user message was sent,
    ``after_tree`` the state once the turn's edits finished. Reversing the
    turn means applying ``before_tree - after_tree`` to the live workspace.
    """

    before_tree: str
    after_tree: str


# ----------------------------------------------------------------------
# subprocess plumbing


async def _run_git(
    args: list[str], cwd: Path | None = None, env: dict[str, str] | None = None
) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_GIT_TIMEOUT)
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
    reuse is harmless)."""
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


async def snapshot_tree(
    root: Path, subtree: str = "", exclude: Iterable[str] = (),
    env_base: dict[str, str] | None = None,
) -> str:
    """Tree sha capturing the workspace's *current* state.

    Built with a throwaway index: ``git add -A`` stages tracked changes,
    uncommitted changes, and untracked-not-ignored files into the temp index
    (the user's real index is untouched), then ``write-tree`` records it as
    a tree object. Ignored files are excluded — they are not reverted by
    /undo and don't show in /diff, matching classic git semantics.

    Exclusion is three-layered: git's own ignore rules; ``exclude``
    (session-start paths, pinned as explicit pathspecs so mid-session
    .gitignore edits can't change what's tracked); and the always-on
    :data:`_UNTRACKED_EXTS` media/binary excludes.

    With ``subtree`` (repo-root relative), only that subtree is captured;
    paths stay repo-root relative. ``env_base`` carries GIT_DIR/
    GIT_WORK_TREE for external snapshot stores.
    """
    root = Path(root).resolve()
    with _temp_index() as index:
        env = {**(env_base if env_base is not None else os.environ), "GIT_INDEX_FILE": index}
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
            kind = "/**" if (root / p).is_dir() and not (root / p).is_symlink() else ""
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
        return out.strip()


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
    ignored = await _git_ignored_paths(repo.root, env=env)
    oversized = _oversized_paths(repo.root, subtree, ignored)
    ignored = (*ignored, *oversized)
    if oversized:
        notes.append(
            f"{len(oversized)} file(s) over 1 MB are not tracked "
            "(excluded from /diff and /undo)"
        )
    baseline = await snapshot_tree(repo.root, subtree, exclude=ignored, env_base=env)
    info = GitInfo(
        session_id=session_id,
        workspace=repo.root,
        git_dir=repo.git_common_dir,
        subtree=subtree,
        baseline_tree=baseline,
        ignored_paths=ignored,
        external=repo.external,
    )
    notes.append(f"tracking edits to {info.agent_workspace} with git snapshots")
    return info, notes


# ----------------------------------------------------------------------
# per-turn checkpoints + revert


async def snapshot_state(info: GitInfo) -> str:
    """:func:`snapshot_tree` for a session's workspace, with its pinned
    ignore exclusions applied."""
    return await snapshot_tree(
        info.workspace, info.subtree, exclude=info.ignored_paths,
        env_base=info.git_env(),
    )


async def accept_changes(info: GitInfo) -> str:
    """Lock in every change made so far: snapshot the current state and make
    it the session's new baseline.

    After this, /diff is empty again (it only ever shows changes since the
    last accept), and /undo //retry can no longer revert file state past
    this point — earlier turns' checkpoints are rewritten to the accepted
    tree by the caller. Returns the accepted tree sha."""
    sha = await snapshot_state(info)
    info.baseline_tree = sha
    return sha


async def revert_to_tree(info: GitInfo, tree: str, current: str | None = None) -> None:
    """Restore the workspace to a snapshot tree, discarding every file change
    made since it: modifications, deletions, and newly created files.

    Two steps, both through throwaway indexes so the user's index and HEAD
    are never touched:

      1. files that exist now but not in the target tree (created since the
         checkpoint) are deleted;
      2. ``read-tree`` + ``checkout-index -f`` write every target-tree file
         back over the working tree (restoring modified and deleted ones).

    ``current`` is the workspace's current snapshot tree, when the caller
    already has it (e.g. :func:`revert_turn`); passing it skips a redundant
    re-snapshot. Only reverts files — anything run_bash did outside the
    workspace (installs, background jobs, network calls) is not undone."""
    root = info.workspace
    env = info.git_env()
    if current is None:
        current = await snapshot_state(info)

    # Paths present now but absent in the target: created since the
    # checkpoint. diff-tree current→tree reports them as deletions (D).
    rc, out, err = await _run_git(
        ["diff-tree", "-r", "--name-only", "--diff-filter=D", current, tree],
        cwd=root, env=env,
    )
    if rc != 0:
        raise GitWorktreeError(f"could not compare against the checkpoint: {err.strip()}")
    for rel in out.splitlines():
        if not rel.strip():
            continue
        path = root / rel
        try:
            path.unlink()
        except OSError:
            pass

    # Write the target tree's files back over the working tree.
    with _temp_index() as index:
        env = info.git_env(GIT_INDEX_FILE=index)
        rc, _, err = await _run_git(["read-tree", tree], cwd=root, env=env)
        if rc != 0:
            raise GitWorktreeError(
                f"could not restore checkpoint {tree[:8]}: {err.strip()}"
            )
        rc, _, err = await _run_git(["checkout-index", "-a", "-f"], cwd=root, env=env)
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
    rc, out, err = await _run_git(args, cwd=info.workspace, env=env)
    if rc != 0:
        raise GitWorktreeError(f"could not wrap checkpoint: {err.strip()}")
    return out.strip()


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
            path = _clean_numstat_path(match.group(1))
            if path not in paths:
                paths.append(path)
    if not paths:
        notices = [line.strip() for line in out.splitlines() if line.startswith("CONFLICT")]
        return "; ".join(notices)
    listed = ", ".join(paths[:5])
    if len(paths) > 5:
        listed += f", … (+{len(paths) - 5} more)"
    return f"conflicting files: {listed}"


async def revert_turn(info: GitInfo, before_tree: str, after_tree: str) -> str | None:
    """Reverse one turn's edits onto the current live state via 3-way merge.

    ``before_tree`` is the workspace before the turn, ``after_tree`` the
    state the turn produced. The merge base is ``after_tree`` (what the turn
    left behind), "ours" is the live workspace now (including any manual
    edits made since), and "theirs" is ``before_tree`` (the target). The
    merge therefore applies the *reverse* of the turn's own diff on top of
    the live state, leaving unrelated changes made since the turn untouched.

    Returns a short conflict description if the turn's changes overlap
    unresolvably with edits made since; None on a clean revert. On conflict
    the workspace is left exactly as it was — nothing is partially applied.
    """
    if before_tree == after_tree:
        return None  # the turn made no edits — nothing to reverse

    current_tree = await snapshot_state(info)
    conflict, result_tree = await _revert_turn_tree(
        info, before_tree, after_tree, current_tree
    )
    if conflict is not None:
        return conflict
    # Safe: reverting to the merge RESULT, not a raw historical snapshot. The
    # current tree is passed through so revert_to_tree needn't re-snapshot.
    await revert_to_tree(info, result_tree, current_tree)
    return None


async def _revert_turn_tree(
    info: GitInfo, before_tree: str, after_tree: str, current_tree: str,
) -> tuple[str | None, str]:
    """Compute the merge result of reversing one turn without applying it.

    Returns ``(conflict, result_tree)``; ``result_tree`` is meaningless (an
    empty string) when ``conflict`` is set."""
    if before_tree == after_tree:
        return None, current_tree
    root = info.workspace
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


def _clean_numstat_path(raw: str) -> str:
    path = raw.strip()
    if path.startswith('"') and path.endswith('"') and len(path) >= 2:
        path = path[1:-1]
    if " => " in path:  # rename: "a/{old => new}/b" or "old => new"
        path = path.rsplit(" => ", 1)[1].replace("{", "").replace("}", "")
    return path


async def _diff_tree_pair(info: GitInfo, base_tree: str, other_tree: str, *extra: str) -> str:
    """Raw ``git diff`` between two snapshot trees.

    Both sides are tree objects, so the user's index is never touched. ``-M``
    is passed explicitly so rename detection works regardless of the user's
    diff.renames config. With a scoped subtree the diff is limited to it and
    its paths are reported relative to it (git --relative)."""
    root = info.workspace
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


async def _diff_trees(info: GitInfo, *extra: str) -> str:
    """Diff the workspace's current state (tracked, uncommitted, and
    untracked-not-ignored files) against the session's baseline tree."""
    current = await snapshot_state(info)
    return await _diff_tree_pair(info, info.baseline_tree, current, *extra)


def _parse_numstat(out: str) -> list[DiffFileInfo]:
    """Parse ``git diff --numstat`` output into DiffFileInfo rows."""
    files: list[DiffFileInfo] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue
        adds, dels, raw = parts
        files.append(DiffFileInfo(
            path=_clean_numstat_path(raw),
            additions=0 if adds == "-" else int(adds or 0),
            deletions=0 if dels == "-" else int(dels or 0),
        ))
    return files


# Snapshot trees are immutable, so a tree pair's diff never changes once the
# objects exist. Cache it (keyed on the shas plus the subtree, since paths
# are reported relative to it) so /diff stays cheap on long sessions. The
# cached string is immutable, so callers can't corrupt the cache by mutating
# what they get back. Bounded (oldest entry evicted) so a long-lived process
# doesn't grow without end.
_CACHE_LIMIT = 512
_numstat_cache: dict[tuple[str, str, str], str] = {}


def _cache_put(cache: dict, key, value):
    if len(cache) >= _CACHE_LIMIT:
        cache.pop(next(iter(cache)))
    cache[key] = value


async def _numstat_between(info: GitInfo, base_tree: str, other_tree: str) -> str:
    """Cached ``git diff --numstat --summary`` between two snapshot trees.

    The ``--summary`` block marks created (and deleted/renamed) files, which
    is where ``is_new`` comes from — no second ``ls-tree -r`` per tree."""
    key = (info.subtree, base_tree, other_tree)
    cached = _numstat_cache.get(key)
    if cached is None:
        cached = await _diff_tree_pair(
            info, base_tree, other_tree, "--numstat", "--summary"
        )
        _cache_put(_numstat_cache, key, cached)
    return cached


def _created_paths(out: str, subtree: str = "") -> frozenset[str]:
    """Paths the diff's ``--summary`` block reports as newly created.

    Summary lines are always repo-root relative (``--relative`` does not
    apply to them), so with a scoped subtree the prefix is stripped to make
    them comparable to the numstat paths."""
    created = set()
    for line in out.splitlines():
        if line.startswith(" create mode "):
            path = line.rsplit(" ", 1)[-1]
            if subtree:
                path = path.removeprefix(subtree + "/")
            created.add(path)
    return frozenset(created)


async def tree_diff_stat(info: GitInfo, base_tree: str, other_tree: str) -> DiffInfo:
    """Per-file +/- counts between two snapshot trees, with ``is_new`` set
    relative to ``base_tree``."""
    out = await _numstat_between(info, base_tree, other_tree)
    files = _parse_numstat(out)
    created = _created_paths(out, info.subtree)
    for f in files:
        f.is_new = f.path in created
    return _diff_info(files)


async def diff_stat(info: GitInfo) -> DiffInfo:
    """Per-file +/- line counts of everything that changed since the
    session's baseline. With a scoped subtree, limited to it, paths
    relative to it."""
    return await tree_diff_stat(info, info.baseline_tree, await snapshot_state(info))


async def attributed_diff_stat(
    info: GitInfo, turns: list[TurnCheckpoint],
    open_before: str | None = None,
) -> DiffInfo:
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
    current = await snapshot_state(info)
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


async def git_diff(info: GitInfo, path: str | None = None) -> str:
    """Unified diff of the workspace's current state vs the session's
    baseline tree, optionally limited to one path (subtree-relative when
    the session is scoped)."""
    extra: tuple[str, ...] = ("--", path) if path else ()
    return await _diff_trees(info, *extra)


# ----------------------------------------------------------------------
# diff summary data — the protocol's DiffInfo/DiffFileInfo models, so
# session.diff_info can hand gitwork's result straight to clients with no
# field-by-field copy.


def _diff_info(files: list[DiffFileInfo]) -> DiffInfo:
    """A DiffInfo with the totals computed from ``files``."""
    return DiffInfo(
        files=files,
        additions=sum(f.additions for f in files),
        deletions=sum(f.deletions for f in files),
    )


# ----------------------------------------------------------------------
# shell-command filter — best-effort UX safety net


_BLOCK_REASON = (
    "blocked: git in this shell is read-only — only inspection commands "
    "(status, log, diff, show, blame, grep, ls-files, rev-parse, …) and "
    "git config --get/--list are available. Anything that mutates the repo, "
    "refs, or config is not. Continue editing files normally."
)

# Deny by default: git in the shell is strictly read-only — only these
# inspection subcommands pass. Everything else (add, commit, checkout,
# branch, tag, reset, switch, rebase, stash, clean, push, config writes,
# update-ref, …) is blocked.
_ALLOWED_SUBCOMMANDS = {
    "status", "log", "diff", "show", "blame", "grep", "shortlog",
    "whatchanged", "rev-parse", "rev-list", "ls-files", "ls-tree",
    "ls-remote", "cat-file", "describe", "merge-base", "name-rev",
    "reflog", "var", "version",
}
_CONFIG_READ_FLAGS = {"--get", "--get-all", "--get-regexp", "--list", "-l"}
# Global git flags that consume a following value.
_GIT_VALUE_FLAGS = {
    "-C", "-c", "--git-dir", "--work-tree", "--namespace",
    "--super-prefix", "--exec-path",
}
_SHELL_SPLIT_RE = re.compile(r"&&|\|\||[;|&\n]")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\+?=")


def _check_git_args(args: list[str]) -> str | None:
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in _GIT_VALUE_FLAGS else 1
    if i >= len(args):
        return None
    sub, rest = args[i], args[i + 1:]

    if sub in _ALLOWED_SUBCOMMANDS:
        return None
    if sub == "config" and any(a in _CONFIG_READ_FLAGS for a in rest):
        return None
    return _BLOCK_REASON


def check_blocked_git(command: str) -> str | None:
    """Return a block reason if the shell command runs a git subcommand
    outside the read-only allow-list, else None.

    Best effort: splits on shell operators and inspects each segment's
    command word (env assignments, subshell parens, sudo/env prefixes, and
    ``sh -c`` recursion are handled). Anything more adversarial
    (command substitution indirection, aliases, a git binary earlier in
    PATH…) can route around it by design — see the module docstring.
    """
    for segment in _SHELL_SPLIT_RE.split(command):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()
        idx = 0
        while idx < len(tokens) and (
            _ENV_ASSIGN_RE.match(tokens[idx]) or tokens[idx] in ("(", "{", "sudo", "env", "nohup")
        ):
            idx += 1
        if idx >= len(tokens):
            continue
        head = tokens[idx]
        if os.path.basename(head) == "git":
            reason = _check_git_args(tokens[idx + 1:])
            if reason:
                return reason
        elif os.path.basename(head) in ("sh", "bash", "dash", "zsh") and "-c" in tokens[idx:]:
            inner = tokens[idx:].index("-c")
            if idx + inner + 1 < len(tokens):
                reason = check_blocked_git(tokens[idx + inner + 1])
                if reason:
                    return reason
    return None
