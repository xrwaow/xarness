"""Harness-managed git worktree isolation for agent edits.

The agent's sandbox is bound to a ``git worktree`` — a second working
directory linked to the user's repository, checked out on its own branch —
never to the user's actual checkout. The user's working directory, index,
staged files, and uncommitted work are therefore untouched by anything the
agent does; the whole session's output lives on ``agent/<session-id>`` until
the user explicitly accepts (merge into their branch) or rejects (delete)
it.

All git operations here run as plain subprocesses against host paths in the
harness process — the worktree is a real host directory; only the agent's
*tools* run sandboxed inside it.

Lifecycle:
  - ``create_worktree`` at session start (fresh sessions in a git repo)
  - worktrees persist across quit/ resume — they are only removed by an
    explicit ``accept``/``reject`` (see app.py), never on session end
  - ``recreate_worktree`` rebuilds a worktree that was removed externally
    (``git worktree prune``, manual deletion) when a session is resumed

Also home to ``check_blocked_git`` — the best-effort filter that keeps the
agent's shell tool from interfering with this lifecycle. It is a UX safety
net, not a security boundary: the real protection is that branch/worktree
management happens in harness code before the agent's first turn, and the
user's checkout is never bound into the sandbox at all.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

# Overridable in tests via monkeypatch — must stay a module-level global that
# functions read at call time.
WORKTREES_DIR = Path("~/.local/share/xarness/worktrees").expanduser()

_GIT_TIMEOUT = 30.0


class GitWorktreeError(Exception):
    """A git operation the harness manages failed; surfaced to the user."""


@dataclass(slots=True)
class RepoInfo:
    """What we need to know about the user's repo at session start."""

    root: Path
    git_common_dir: Path  # shared .git dir (objects/refs) — bind into the sandbox
    head_branch: str | None  # None when detached
    base_ref: str | None  # commit sha to diff/merge against; None = unborn HEAD


@dataclass(slots=True)
class GitInfo:
    """Per-session git state; persisted in the session JSON as the ``git`` block."""

    session_id: str
    base_ref: str
    branch: str
    worktree: Path
    original_workspace: Path
    git_common_dir: Path
    # Repo-root-relative posix subdirectory the session is scoped to ("" =
    # the whole repo). Set when the user's --workspace sits inside a bigger
    # repo: the sandbox binds only this subtree read-write, and diffs cover
    # only it, so the rest of the user's repo stays out of the session.
    subtree: str = ""
    # Untracked files copied from the user's checkout at worktree creation so
    # the agent can see them. Tracked separately so accept can avoid committing
    # the ones the agent never touched (they stay untracked in the user's repo).
    copied_untracked: list[str] = field(default_factory=list)

    @property
    def agent_workspace(self) -> Path:
        """Directory the agent works in: the scoped subtree of the worktree,
        or the worktree root for a whole-repo session."""
        return self.worktree / self.subtree if self.subtree else self.worktree

    def to_block(self) -> dict:
        """JSON-safe dict for the session file (paths as strings)."""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in asdict(self).items()
        }

    @classmethod
    def from_block(cls, block: dict) -> GitInfo:
        return cls(
            session_id=block["session_id"],
            base_ref=block["base_ref"],
            branch=block["branch"],
            worktree=Path(block["worktree"]),
            original_workspace=Path(block["original_workspace"]),
            git_common_dir=Path(block["git_common_dir"]),
            subtree=block.get("subtree", ""),
            copied_untracked=list(block.get("copied_untracked", [])),
        )


# ----------------------------------------------------------------------
# Diff summary data (Part 3 UI contract)


@dataclass(slots=True)
class FileDiff:
    path: str
    additions: int
    deletions: int
    is_new: bool = False


@dataclass(slots=True)
class DiffStat:
    files: list[FileDiff]

    @property
    def additions(self) -> int:
        return sum(f.additions for f in self.files)

    @property
    def deletions(self) -> int:
        return sum(f.deletions for f in self.files)


class MergeOutcome(Enum):
    MERGED = "merged"
    CONFLICT = "conflict"
    IN_PROGRESS = "in_progress"  # a previous merge is unresolved in the repo
    ERROR = "error"


@dataclass(slots=True)
class MergeResult:
    outcome: MergeOutcome
    detail: str = ""


# ----------------------------------------------------------------------
# subprocess plumbing


async def _run_git(args: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        cwd=str(cwd) if cwd else None,
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
# repo detection + worktree lifecycle (Part 1)


async def detect_repo(path: Path) -> RepoInfo | None:
    """Repo info for ``path``, or None if it isn't a git repo.

    An unborn HEAD (``git init`` with no commits) is still a repo — it comes
    back with ``base_ref=None`` and gets a baseline snapshot when the
    worktree is created. Runs against the real host path, before any
    sandboxing decision.
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


async def _commit_identity(cwd: Path) -> list[str]:
    """``-c user.name/email`` overrides for repos with no configured identity
    (common right after the harness auto-initialized one)."""
    rc, email, _ = await _run_git(["config", "user.email"], cwd=cwd)
    if rc == 0 and email.strip():
        return []
    return ["-c", "user.name=xarness", "-c", "user.email=xarness@localhost"]


async def create_baseline_commit(repo: RepoInfo) -> str:
    """Snapshot the workspace's current state as the repo's first commit.

    Needed because a worktree can only be created from a commit. Only ever
    runs on an unborn HEAD — the user's existing history is never touched.
    ``--allow-empty`` covers a completely empty workspace.
    """
    await _run_git(["add", "-A"], cwd=repo.root)
    rc, _, err = await _run_git(
        [*await _commit_identity(repo.root), "commit", "-q", "--allow-empty",
         "-m", "xarness: baseline snapshot of your workspace"],
        cwd=repo.root,
    )
    if rc != 0:
        raise GitWorktreeError(f"could not create the baseline snapshot: {err.strip()}")
    rc, sha, _ = await _run_git(["rev-parse", "HEAD"], cwd=repo.root)
    if rc != 0:
        raise GitWorktreeError("baseline commit created but HEAD unreadable")
    return sha.strip()


def _untracked_from_status(status_z: str) -> list[str]:
    """Paths from ``git status --porcelain -z -uall`` output (?? entries)."""
    return [
        entry[3:]
        for entry in status_z.split("\0")
        if entry.startswith("?? ")
    ]


async def copy_untracked_files(
    original: Path, worktree: Path, subtree: str = ""
) -> list[str]:
    """Copy the user's untracked files (and only those — never a blanket
    directory copy, which would drag in gitignored secrets/huge files) into
    the fresh worktree. With ``subtree`` (repo-root relative), only files
    under it are copied. Returns the relative paths copied."""
    rc, out, _ = await _run_git(["status", "--porcelain", "-uall", "-z"], cwd=original)
    if rc != 0:
        return []
    prefix = f"{subtree}/" if subtree else ""
    copied: list[str] = []
    for rel in _untracked_from_status(out):
        if subtree and not rel.startswith(prefix):
            continue
        src = original / rel
        if not src.is_file():
            continue
        dest = worktree / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(rel)
    return copied


async def _free_branch_name(root: Path, base: str) -> str:
    """``base``, or ``base-2``, ``base-3``, … — first name without a branch.

    Needed because resuming a resolved session reuses its session id, and
    /accept keeps the agent branch as history: a fresh worktree for the same
    session must never clobber that kept branch.
    """
    candidate, n = base, 2
    while True:
        rc, _, _ = await _run_git(
            ["rev-parse", "-q", "--verify", f"refs/heads/{candidate}"], cwd=root
        )
        if rc != 0:
            return candidate
        candidate = f"{base}-{n}"
        n += 1


async def create_worktree(
    repo: RepoInfo, session_id: str, copy_untracked: bool = True, subtree: str = ""
) -> GitInfo:
    """Create ``agent/<session-id>`` + its worktree outside the repo tree.

    Never touches the user's checkout — ``git worktree add`` only writes to
    the shared git dir and the new worktree path. On an unborn HEAD a
    baseline snapshot commit is created first (a worktree needs a commit).

    With ``subtree`` (repo-root-relative posix path), the session is scoped
    to that subdirectory: it is created in the worktree if missing (git only
    checks out tracked files), untracked files are copied only from under
    it, and GitInfo.subtree scopes the sandbox + diffs to it.
    """
    if repo.base_ref is None:
        repo.base_ref = await create_baseline_commit(repo)
    worktree_path = WORKTREES_DIR / session_id
    branch = await _free_branch_name(repo.root, f"agent/{session_id}")
    WORKTREES_DIR.mkdir(parents=True, exist_ok=True)
    if worktree_path.exists():
        # Harness-owned leftover (e.g. a crashed prior run with the same id).
        shutil.rmtree(worktree_path)
        await _run_git(["worktree", "prune"], cwd=repo.root)
    rc, out, err = await _run_git(
        ["worktree", "add", "-b", branch, str(worktree_path), repo.base_ref],
        cwd=repo.root,
    )
    if rc != 0:
        raise GitWorktreeError(
            f"git worktree add failed: {err.strip() or out.strip()}"
        )
    if subtree:
        (worktree_path / subtree).mkdir(parents=True, exist_ok=True)
    info = GitInfo(
        session_id=session_id,
        base_ref=repo.base_ref,
        branch=branch,
        worktree=worktree_path,
        original_workspace=repo.root,
        git_common_dir=repo.git_common_dir,
        subtree=subtree,
    )
    if copy_untracked:
        info.copied_untracked = await copy_untracked_files(
            repo.root, worktree_path, subtree=subtree
        )
    return info


async def setup_isolation(
    workspace: Path,
    session_id: str,
    copy_untracked: bool = True,
    allow_init: bool = True,
) -> tuple[GitInfo | None, list[str]]:
    """One-stop session-start setup: detect the repo (or initialize one, if
    allowed, so tracking works with zero setup), snapshot a baseline on an
    unborn HEAD, and create the agent worktree.

    Returns ``(git_info, notes)`` — notes are human-readable lines for the
    chat log / stderr. git_info is None when the workspace can't be isolated
    (no repo and init not allowed, or setup failed); the caller then falls
    back to direct editing exactly as before this feature existed.
    """
    notes: list[str] = []
    workspace = Path(workspace).resolve()
    repo = await detect_repo(workspace)
    if repo is None:
        if not allow_init:
            return None, notes
        repo = await init_managed_repo(workspace)
        notes.append(
            "no git repo here — initialized one (files untouched) and will "
            "snapshot your current files as the baseline, so the agent's "
            "changes can be tracked, accepted, or rejected"
        )
    elif repo.base_ref is None:
        notes.append(
            "your repo has no commits yet — snapshotting the current files as "
            "the baseline commit so the agent's changes can be tracked"
        )
    subtree = ""
    if repo.root != workspace:
        # The workspace is a subdirectory of the repo: scope the session to
        # it, so the diff/accept surface (and the agent's sandbox) only ever
        # cover the directory the user actually pointed at.
        subtree = workspace.relative_to(repo.root).as_posix()
        notes.append(
            f"workspace is inside an existing repo — tracking is scoped to "
            f"{subtree}/; the rest of the repo is visible to the agent but "
            "read-only"
        )
    info = await create_worktree(
        repo, session_id, copy_untracked=copy_untracked, subtree=subtree
    )
    notes.append(
        f"git isolation: agent edits go to worktree {info.worktree} "
        f"(branch {info.branch}, base {info.base_ref[:8]}); "
        "/diff shows pending changes, /accept merges them, /reject discards them"
    )
    return info, notes


async def worktree_is_valid(info: GitInfo) -> bool:
    """True if the worktree still exists and git still tracks it."""
    if not info.worktree.is_dir():
        return False
    rc, out, _ = await _run_git(
        ["worktree", "list", "--porcelain"], cwd=info.original_workspace
    )
    if rc != 0:
        return False
    return any(
        Path(line[len("worktree "):]).resolve() == info.worktree.resolve()
        for line in out.splitlines() if line.startswith("worktree ")
    )


async def recreate_worktree(info: GitInfo) -> GitInfo:
    """Rebuild a worktree that was removed externally.

    Prefers checking the agent branch back out (preserves committed agent
    work); falls back to a new branch at base_ref if the branch is gone.
    """
    if info.worktree.exists():
        shutil.rmtree(info.worktree)
        await _run_git(["worktree", "prune"], cwd=info.original_workspace)
    rc, out, err = await _run_git(
        ["worktree", "add", str(info.worktree), info.branch],
        cwd=info.original_workspace,
    )
    if rc != 0:
        rc, out, err = await _run_git(
            ["worktree", "add", "-b", info.branch, str(info.worktree), info.base_ref],
            cwd=info.original_workspace,
        )
    if rc != 0:
        raise GitWorktreeError(
            f"could not recreate worktree: {err.strip() or out.strip()}"
        )
    info.copied_untracked = await copy_untracked_files(
        info.original_workspace, info.worktree, subtree=info.subtree
    )
    if info.subtree:
        (info.worktree / info.subtree).mkdir(parents=True, exist_ok=True)
    return info


# ----------------------------------------------------------------------
# diff computation (Part 3)


async def stage_untracked(worktree: Path) -> None:
    """``git add -N -A`` every path so intent-to-add makes plain ``git diff``
    the single source of truth for new files too. Idempotent: untracked paths
    only, safe to call repeatedly."""
    await _run_git(["add", "-N", "-A"], cwd=worktree)


def _clean_numstat_path(raw: str) -> str:
    path = raw.strip()
    if path.startswith('"') and path.endswith('"') and len(path) >= 2:
        path = path[1:-1]
    if " => " in path:  # rename: "a/{old => new}/b" or "old => new"
        path = path.rsplit(" => ", 1)[1].replace("{", "").replace("}", "")
    return path


async def _run_diff(
    worktree: Path, base_ref: str, *extra: str, subtree: str = ""
) -> str:
    """Shared plumbing for diff_stat/git_diff: stage untracked paths first,
    then diff the worktree against base_ref. ``-M`` is passed explicitly so
    rename detection works regardless of the user's diff.renames config.
    With ``subtree``, the diff is limited to that worktree subdirectory and
    its paths are reported relative to it (git --relative), so a session
    scoped to a workspace inside a bigger repo only sees its own files."""
    await stage_untracked(worktree)
    args = ["diff", "-M"]
    if subtree:
        args.append("--relative")
    args.append(base_ref)
    rc, out, err = await _run_git(
        [*args, *extra], cwd=worktree / subtree if subtree else worktree
    )
    if rc != 0:
        raise GitWorktreeError(f"git diff failed: {err.strip()}")
    return out


async def diff_stat(worktree: Path, base_ref: str, subtree: str = "") -> DiffStat:
    """Per-file +/- line counts of everything in the worktree vs base_ref
    (staged, unstaged, intent-to-add, and committed-but-unmerged). With
    ``subtree``, limited to that subdirectory, paths relative to it."""
    out = await _run_diff(worktree, base_ref, "--numstat", subtree=subtree)
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
    rc, tree, _ = await _run_git(
        ["ls-tree", "--name-only", "-r", base_ref],
        cwd=worktree / subtree if subtree else worktree,
    )
    base_files = set(tree.splitlines()) if rc == 0 else set()
    for f in files:
        f.is_new = f.path not in base_files
    return DiffStat(files=files)


async def git_diff(
    worktree: Path, base_ref: str, path: str | None = None, subtree: str = ""
) -> str:
    """Unified diff of the worktree vs base_ref (staged, unstaged,
    intent-to-add, and committed-but-unmerged), optionally limited to one
    path (subtree-relative when ``subtree`` is set)."""
    extra: tuple[str, ...] = ("--", path) if path else ()
    return await _run_diff(worktree, base_ref, *extra, subtree=subtree)


# ----------------------------------------------------------------------
# accept / reject (Part 4)


async def commit_worktree_changes(info: GitInfo) -> str:
    """Commit all worktree changes onto the agent branch so a merge can
    carry them. Returns "clean", "nothing-staged", or "committed".

    Untracked files copied from the user's checkout that the agent never
    touched are unstaged again (byte-identical check): they stay untracked
    in the user's repo instead of silently becoming tracked by the merge.
    """
    rc, out, err = await _run_git(["status", "--porcelain", "-uall", "-z"], cwd=info.worktree)
    if rc != 0:
        raise GitWorktreeError(f"could not read worktree status: {err.strip()}")
    if not out.replace("\0", "").strip():
        return "clean"
    await _run_git(["add", "-A"], cwd=info.worktree)
    for rel in info.copied_untracked:
        wt_file = info.worktree / rel
        orig = info.original_workspace / rel
        if wt_file.is_file() and orig.is_file():
            try:
                if wt_file.read_bytes() == orig.read_bytes():
                    await _run_git(["restore", "--staged", "--", rel], cwd=info.worktree)
            except OSError:
                pass
    rc, _, _ = await _run_git(["diff", "--cached", "--quiet"], cwd=info.worktree)
    if rc == 0:
        return "nothing-staged"
    rc, out, err = await _run_git(
        [*await _commit_identity(info.worktree), "commit", "-q",
         "-m", f"xarness: session {info.session_id} changes"],
        cwd=info.worktree,
    )
    if rc != 0:
        raise GitWorktreeError(f"could not commit worktree changes: {err.strip() or out.strip()}")
    return "committed"


async def merge_branch(
    original_root: Path, branch: str, no_ff: bool = True
) -> MergeResult:
    """Merge the agent branch into the user's current branch, in their real
    checkout. Conflicts are reported, never auto-resolved."""
    rc, _, _ = await _run_git(
        ["rev-parse", "-q", "--verify", "MERGE_HEAD"], cwd=original_root
    )
    if rc == 0:
        return MergeResult(
            MergeOutcome.IN_PROGRESS,
            "a previous merge is still unresolved in the original directory",
        )
    args = ["merge"]
    if no_ff:
        args.append("--no-ff")
    args.append(branch)
    rc, out, err = await _run_git(args, cwd=original_root)
    combined = out + err
    if rc == 0:
        return MergeResult(MergeOutcome.MERGED, combined.strip())
    if "CONFLICT" in combined or "Automatic merge failed" in combined:
        return MergeResult(MergeOutcome.CONFLICT, combined.strip())
    return MergeResult(MergeOutcome.ERROR, combined.strip() or f"git merge exited {rc}")


async def remove_worktree(original_root: Path, worktree: Path, force: bool = False) -> None:
    args = ["worktree", "remove"]
    if force:
        args.append("--force")
    args.append(str(worktree))
    rc, _, _ = await _run_git(args, cwd=original_root)
    if rc != 0:
        # Entry may be stale (dir already gone) — prune, then force the
        # harness-owned directory away as a last resort.
        await _run_git(["worktree", "prune"], cwd=original_root)
        if worktree.exists():
            shutil.rmtree(worktree, ignore_errors=True)


async def delete_branch(original_root: Path, branch: str) -> None:
    rc, out, err = await _run_git(["branch", "-D", branch], cwd=original_root)
    if rc != 0:
        raise GitWorktreeError(
            f"could not delete branch {branch}: {err.strip() or out.strip()}"
        )


# ----------------------------------------------------------------------
# shell-command filter (Part 2) — best-effort UX safety net


_BLOCK_REASON = (
    "blocked: branch and worktree management is handled automatically by the "
    "harness — git checkout <ref>, git switch, git worktree, git branch -d/-D, "
    "git reset --hard, and git rebase are not available in this shell. "
    "Continue editing files normally; read-only git commands (status, diff, "
    "log, show, blame) and git add/commit still work."
)

# Subcommands that always change identity — blocked outright.
_BLOCKED_SUBCOMMANDS = {"switch", "worktree", "rebase"}
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
    if sub == "branch" and any(a in ("-d", "-D", "--delete") for a in rest):
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
