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
agent's shell tool from rewriting the user's branch/refs (checkout/switch,
reset --hard, branch deletion/renaming, tag deletion/forcing, update-ref,
worktree/rebase). It is a UX safety net, not
a security boundary: the user's history is theirs; the agent edits files,
not refs.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


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
    head_branch: str | None  # None when detached
    base_ref: str | None  # commit sha; None = unborn HEAD (fine — snapshots
    # are trees, not commits, so no baseline commit is needed anymore)


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

    An unborn HEAD (``git init`` with no commits) is still a repo — it comes
    back with ``base_ref=None``, which direct-write tracking handles fine
    (snapshots are trees, so no baseline commit is needed).
    """
    path = Path(path).resolve()
    rc, out, _ = await _run_git(["rev-parse", "--is-inside-work-tree"], cwd=path)
    if rc != 0 or out.strip() != "true":
        return None
    rc, top, err = await _run_git(["rev-parse", "--show-toplevel"], cwd=path)
    if rc != 0:
        raise GitWorktreeError(f"could not find repo root: {err.strip()}")
    rc, common, _ = await _run_git(["rev-parse", "--git-common-dir"], cwd=path)
    if rc != 0:
        raise GitWorktreeError("could not resolve the repo's git dir")
    common_dir = Path(common.strip())
    if not common_dir.is_absolute():
        common_dir = (path / common_dir).resolve()
    rc, sha, _ = await _run_git(["rev-parse", "HEAD"], cwd=path)
    base_ref = sha.strip() if rc == 0 else None  # unborn HEAD: no commits yet
    rc, ref, _ = await _run_git(["branch", "--show-current"], cwd=path)
    head_branch = ref.strip() or None if rc == 0 else None
    return RepoInfo(
        root=Path(top.strip()).resolve(),
        git_common_dir=common_dir,
        head_branch=head_branch,
        base_ref=base_ref,
    )


async def init_managed_repo(workspace: Path) -> RepoInfo:
    """``git init`` a workspace that has no repo, so change tracking works
    with zero setup from the user.

    This only creates ``.git/`` — the user's files are not touched. The repo
    is marked ``xarness.managed`` so it's identifiable later. If the user
    runs their own ``git init`` afterwards it's a no-op; if they already had
    a repo we never get here.
    """
    workspace = Path(workspace).resolve()
    rc, _, err = await _run_git(["init", "-q", "-b", "main"], cwd=workspace)
    if rc != 0:  # older git without -b
        rc, _, err = await _run_git(["init", "-q"], cwd=workspace)
    if rc != 0:
        raise GitWorktreeError(f"could not initialize a git repo in {workspace}: {err.strip()}")
    await _run_git(["config", "xarness.managed", "true"], cwd=workspace)
    repo = await detect_repo(workspace)
    if repo is None:  # pragma: no cover — init just succeeded
        raise GitWorktreeError(f"repo detection failed right after git init in {workspace}")
    return repo


# ----------------------------------------------------------------------
# tree snapshots (checkpoints + diff base)


def _temp_index() -> str:
    """Path of a throwaway index file for GIT_INDEX_FILE.

    The path must NOT exist — git refuses to read an empty file as an index
    and wants to create it itself."""
    fd, name = tempfile.mkstemp(prefix="xarness-index-")
    os.close(fd)
    os.unlink(name)
    return name


async def _git_ignored_paths(root: Path) -> tuple[str, ...]:
    """Repo-root-relative paths the repo's ignore rules flag at a moment in
    time (directories collapsed to single entries), for pinning snapshot
    semantics in :class:`GitInfo`. Empty when git can't answer."""
    rc, out, _ = await _run_git(
        ["ls-files", "--others", "--ignored", "--exclude-standard", "--directory"],
        cwd=root,
    )
    if rc != 0:
        return ()
    return tuple(line.rstrip("/") for line in out.splitlines() if line)


async def snapshot_tree(
    root: Path, subtree: str = "", exclude: Iterable[str] = ()
) -> str:
    """Tree sha capturing the workspace's *current* state.

    Built with a throwaway index: ``git add -A`` stages tracked changes,
    uncommitted changes, and untracked-not-ignored files into the temp index
    (the user's real index is untouched), then ``write-tree`` records it as
    a tree object. Ignored files are excluded — they are not reverted by
    /undo and don't show in /diff, matching classic git semantics.

    Exclusion is two-layered: git's own ignore rules apply as always, and
    ``exclude`` (repo-root-relative paths captured at session start) is
    passed as explicit negative pathspecs — so mid-session edits to
    .gitignore files cannot change what snapshots track.

    With ``subtree`` (repo-root relative), only that subtree is captured;
    the resulting tree's paths are still repo-root relative.
    """
    root = Path(root).resolve()
    index = _temp_index()
    try:
        env = {**os.environ, "GIT_INDEX_FILE": index}
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
        if pathspecs:
            args += ["--", *pathspecs]
        rc, _, err = await _run_git(args, cwd=root, env=env)
        if rc != 0:
            raise GitWorktreeError(f"could not snapshot the workspace: {err.strip()}")
        rc, out, err = await _run_git(["write-tree"], cwd=root, env=env)
        if rc != 0:
            raise GitWorktreeError(f"could not write the snapshot tree: {err.strip()}")
        return out.strip()
    finally:
        try:
            os.unlink(index)
        except OSError:
            pass


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
        repo = await init_managed_repo(workspace)
        notes.append(
            "no git repo here — initialized one (files untouched) so the "
            "agent's changes can be tracked and undone"
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
    # Pin what snapshots track up front so ignore rules are not re-read
    # per snapshot (mid-session .gitignore edits stay out of the picture).
    ignored = await _git_ignored_paths(repo.root)
    baseline = await snapshot_tree(repo.root, subtree, exclude=ignored)
    info = GitInfo(
        session_id=session_id,
        workspace=repo.root,
        git_dir=repo.git_common_dir,
        subtree=subtree,
        baseline_tree=baseline,
        ignored_paths=ignored,
    )
    notes.append(f"tracking edits to {info.agent_workspace} with git snapshots")
    return info, notes


# ----------------------------------------------------------------------
# per-turn checkpoints + revert


async def snapshot_state(info: GitInfo) -> str:
    """:func:`snapshot_tree` for a session's workspace, with its pinned
    ignore exclusions applied."""
    return await snapshot_tree(info.workspace, info.subtree, exclude=info.ignored_paths)


async def checkpoint(info: GitInfo) -> str:
    """Snapshot the workspace's current state as a tree sha.

    Used for both halves of a turn's checkpoint: the before-tree taken when
    its user message is sent, and the after-tree recorded once its edits
    finish. :func:`revert_turn` reverses the difference."""
    return await snapshot_state(info)


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
    if current is None:
        current = await snapshot_state(info)

    # Paths present now but absent in the target: created since the
    # checkpoint. diff-tree current→tree reports them as deletions (D).
    rc, out, err = await _run_git(
        ["diff-tree", "-r", "--name-only", "--diff-filter=D", current, tree], cwd=root
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
    index = _temp_index()
    try:
        env = {**os.environ, "GIT_INDEX_FILE": index}
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
    finally:
        try:
            os.unlink(index)
        except OSError:
            pass


async def _wrap_commit(root: Path, tree: str, parent: str | None = None) -> str:
    """Wrap a bare tree sha in a throwaway, unreachable commit object.

    ``git merge-tree`` operates on commit-ish, not bare trees, so a snapshot
    tree needs a commit wrapper. No ref is created and nothing is reachable
    from HEAD or any branch — the commit is an unreferenced object in the
    object store, exactly like the snapshot trees themselves. Identity is
    supplied via the environment so this works in a repo with no
    ``user.name``/``user.email`` configured.
    """
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "xarness",
        "GIT_AUTHOR_EMAIL": "xarness@localhost",
        "GIT_COMMITTER_NAME": "xarness",
        "GIT_COMMITTER_EMAIL": "xarness@localhost",
    }
    args = ["commit-tree", tree, "-m", "checkpoint"]
    if parent:
        args += ["-p", parent]
    rc, out, err = await _run_git(args, cwd=root, env=env)
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

    root = info.workspace
    current_tree = await snapshot_state(info)

    base = await _wrap_commit(root, after_tree)     # state right after the turn
    theirs = await _wrap_commit(root, before_tree)  # target: state before the turn
    ours = await _wrap_commit(root, current_tree)   # live now, incl. manual edits

    rc, out, err = await _run_git(
        ["merge-tree", "--write-tree", f"--merge-base={base}", ours, theirs],
        cwd=root,
    )
    if rc != 0:
        # Conflict: merge-tree still prints a (conflict-marked) tree, but we
        # must not apply it. Surface a readable summary and leave files alone.
        return _conflict_summary(out) or err.strip() or "merge conflict"
    lines = out.strip().splitlines()
    if not lines:
        raise GitWorktreeError(f"merge produced no tree: {err.strip()}")
    result_tree = lines[0]
    # Safe: reverting to the merge RESULT, not a raw historical snapshot. The
    # current tree is passed through so revert_to_tree needn't re-snapshot.
    await revert_to_tree(info, result_tree, current_tree)
    return None


async def revert_turns(info: GitInfo, turns: list[TurnCheckpoint]) -> str | None:
    """Reverse a range of turns, newest first.

    Each turn is reversed against the live state left by the previous call,
    so a multi-turn drop undoes the most recent turn's diff first. Returns
    conflict text on the first unresolvable overlap (leaving the workspace
    as it stood before that turn's revert), or None when all reverted
    cleanly. An empty list is a no-op.
    """
    for turn in reversed(turns):
        conflict = await revert_turn(info, turn.before_tree, turn.after_tree)
        if conflict is not None:
            return conflict
    return None


async def undo_last_turn(info: GitInfo, turns: list[TurnCheckpoint]) -> str | None:
    """Reverse the most recent turn, or do nothing when there is none.

    The empty case is deliberate: with no turn since the baseline there is
    nothing to reverse, and falling back to ``baseline_tree`` would destroy
    every change made outside the session."""
    if not turns:
        return None
    return await revert_turns(info, turns[-1:])


# ----------------------------------------------------------------------
# diff computation


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
    )
    if rc != 0:
        raise GitWorktreeError(f"git diff failed: {err.strip()}")
    return out


async def _diff_trees(info: GitInfo, *extra: str) -> str:
    """Diff the workspace's current state (tracked, uncommitted, and
    untracked-not-ignored files) against the session's baseline tree."""
    current = await snapshot_state(info)
    return await _diff_tree_pair(info, info.baseline_tree, current, *extra)


def _parse_numstat(out: str) -> list[FileDiff]:
    """Parse ``git diff --numstat`` output into FileDiff rows."""
    files: list[FileDiff] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue
        adds, dels, raw = parts
        files.append(FileDiff(
            path=_clean_numstat_path(raw),
            additions=0 if adds == "-" else int(adds or 0),
            deletions=0 if dels == "-" else int(dels or 0),
        ))
    return files


# Snapshot trees are immutable, so a tree pair's diff and a tree's file list
# never change once the objects exist. Cache them (keyed on the shas plus the
# subtree, since paths are reported relative to it) so /diff stays cheap on
# long sessions. Values are immutable, so callers can't corrupt the cache by
# mutating what they get back. Bounded (oldest entry evicted) so a long-lived
# process doesn't grow without end.
_CACHE_LIMIT = 512
_numstat_cache: dict[tuple[str, str, str], str] = {}
_tree_files_cache: dict[tuple[str, str], frozenset[str]] = {}


def _cache_put(cache: dict, key, value):
    if len(cache) >= _CACHE_LIMIT:
        cache.pop(next(iter(cache)))
    cache[key] = value


async def _numstat_between(info: GitInfo, base_tree: str, other_tree: str) -> str:
    """Cached ``git diff --numstat`` between two snapshot trees."""
    key = (info.subtree, base_tree, other_tree)
    cached = _numstat_cache.get(key)
    if cached is None:
        cached = await _diff_tree_pair(info, base_tree, other_tree, "--numstat")
        _cache_put(_numstat_cache, key, cached)
    return cached


async def _tree_files(info: GitInfo, tree: str) -> set[str]:
    """Paths in a snapshot tree, subtree-relative when the session is scoped
    (matching the paths ``--relative`` diffs report). Cached per tree."""
    key = (info.subtree, tree)
    cached = _tree_files_cache.get(key)
    if cached is None:
        rc, out, _ = await _run_git(
            ["ls-tree", "--name-only", "-r", tree], cwd=info.workspace
        )
        names = set(out.splitlines()) if rc == 0 else set()
        if info.subtree:
            prefix = info.subtree + "/"
            names = {n[len(prefix):] for n in names if n.startswith(prefix)}
        cached = frozenset(names)
        _cache_put(_tree_files_cache, key, cached)
    return set(cached)


async def tree_diff_stat(info: GitInfo, base_tree: str, other_tree: str) -> DiffStat:
    """Per-file +/- counts between two snapshot trees, with ``is_new`` set
    relative to ``base_tree``."""
    files = _parse_numstat(await _numstat_between(info, base_tree, other_tree))
    base_files = await _tree_files(info, base_tree)
    for f in files:
        f.is_new = f.path not in base_files
    return DiffStat(files=files)


async def diff_stat(info: GitInfo) -> DiffStat:
    """Per-file +/- line counts of everything that changed since the
    session's baseline. With a scoped subtree, limited to it, paths
    relative to it."""
    return await tree_diff_stat(info, info.baseline_tree, await snapshot_state(info))


async def attributed_diff_stat(info: GitInfo, turns: list[TurnCheckpoint]) -> DiffStat:
    """Pending changes since ``baseline_tree``, tagged by source.

    ``baseline_tree`` only ever moves via :func:`accept_changes`, so this is
    always the full, correct picture of "what's changed since the user last
    accepted" — regardless of how many turns or drift-detection gaps happened
    in between. Turn diffs are used only to *tag* paths within that total as
    agent-caused; anything else in the total is drift (made outside the
    session — manual edits, or edits from a prior harness run).
    """
    current = await snapshot_state(info)
    total = await tree_diff_stat(info, info.baseline_tree, current)

    agent_touched: set[str] = set()
    for turn in turns:
        for f in (await tree_diff_stat(info, turn.before_tree, turn.after_tree)).files:
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
# diff summary data


@dataclass(slots=True)
class FileDiff:
    path: str
    additions: int
    deletions: int
    is_new: bool = False
    # Who caused the change: "agent" (some turn's diff touched the path) or
    # "drift" (made outside the session). A path is one or the other, never
    # both — attribution is per-file, so there is no combined value.
    source: Literal["agent", "drift"] = "agent"


@dataclass(slots=True)
class DiffStat:
    files: list[FileDiff]

    @property
    def additions(self) -> int:
        return sum(f.additions for f in self.files)

    @property
    def deletions(self) -> int:
        return sum(f.deletions for f in self.files)


# ----------------------------------------------------------------------
# shell-command filter — best-effort UX safety net


_BLOCK_REASON = (
    "blocked: your branch and refs are the user's — git checkout <ref>, git "
    "switch, git worktree, git branch -d/-m, git tag -d/-f, git update-ref, "
    "git reset --hard, and git rebase are not available in this shell. "
    "Continue editing files normally; read-only git commands (status, diff, "
    "log, show, blame) and git add/commit still work."
)

# Subcommands that always change identity — blocked outright.
_BLOCKED_SUBCOMMANDS = {"switch", "worktree", "rebase", "update-ref", "symbolic-ref"}
# Global git flags that consume a following value.
_GIT_VALUE_FLAGS = {
    "-C", "-c", "--git-dir", "--work-tree", "--namespace",
    "--super-prefix", "--exec-path",
}
_SHELL_SPLIT_RE = re.compile(r"&&|\|\||[;|&\n]")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\+?=")


def _check_git_args(
    args: list[str], current_branch: str | None, worktree_root: Path | None
) -> str | None:
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in _GIT_VALUE_FLAGS else 1
    if i >= len(args):
        return None
    sub, rest = args[i], args[i + 1:]

    if sub in _BLOCKED_SUBCOMMANDS:
        return _BLOCK_REASON
    # Renames/copies move or duplicate refs just like deletions do.
    if sub == "branch" and any(
        a in ("-d", "-D", "--delete", "-m", "-M", "--move", "-c", "-C", "--copy")
        for a in rest
    ):
        return _BLOCK_REASON
    # Forcing or deleting a tag rewrites a ref the user owns.
    if sub == "tag" and any(a in ("-d", "-D", "--delete", "-f", "--force") for a in rest):
        return _BLOCK_REASON
    if sub == "reset" and "--hard" in rest:
        return _BLOCK_REASON
    if sub == "checkout":
        # `git checkout -- file` (or any pathspec after --) restores files
        # within the current branch: always allowed.
        target: str | None = None
        for arg in rest:
            if arg == "--":
                return None
            if arg.startswith("-"):
                # These flags create/switch/detach branches.
                if arg in ("-b", "-B", "-t", "--track", "--orphan", "--detach", "-d"):
                    return _BLOCK_REASON
                continue
            target = arg
            break
        if target is None:
            return None
        if target == current_branch or target == "HEAD":
            return None
        # A bare word that names an existing path is a file restore, not a
        # ref switch (don't false-positive on files named like branches).
        if worktree_root is not None and (worktree_root / target).exists():
            return None
        return _BLOCK_REASON
    return None


def check_blocked_git(
    command: str, current_branch: str | None, worktree_root: Path | None
) -> str | None:
    """Return a block reason if the shell command tries an identity-changing
    git operation, else None.

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
            reason = _check_git_args(tokens[idx + 1:], current_branch, worktree_root)
            if reason:
                return reason
        elif os.path.basename(head) in ("sh", "bash", "dash", "zsh") and "-c" in tokens[idx:]:
            inner = tokens[idx:].index("-c")
            if idx + inner + 1 < len(tokens):
                reason = check_blocked_git(
                    tokens[idx + inner + 1], current_branch, worktree_root
                )
                if reason:
                    return reason
    return None
