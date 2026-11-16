"""Direct-write change tracking for agent edits.

The agent's sandbox is bound to the user's real workspace: file edits and
shell commands land in the user's files immediately — there is no worktree
and no accept/reject copy. Git is used purely as a change record:

  - opening a workspace snapshots it as a tree object (the workspace's
    ``baseline_tree``) — the base /diff measures against
  - every mutating tool call is recorded as an event in
    :mod:`xarness.workspace` (a command's writes are credited whatever its
    exit status); labels (mine/agent/auto/drift) are derived from those
    events, never stored, and an explicit write's before/after blob shas are
    recorded with it so each agent's own delta can be diffed
  - /undo and /retry reverse only the paths a turn's events touched: an
    untouched path is restored from the turn's before tree, and a path that
    changed since is three-way merged (see :func:`merge_texts`) so another
    agent's later edit elsewhere in the file survives. Any conflict makes
    the whole undo a no-op

Snapshots are built with a throwaway index (``GIT_INDEX_FILE`` pointing at a
temp file): the user's index, HEAD, and refs are never touched, and no
commits are created. What *is* written is unreachable tree/blob
objects — and which object store they land in depends on the workspace
(see :func:`open_workspace`):

  - **the workspace is a repo**: the user's own object store, i.e. their
    ``.git``. Nothing else in it changes, and because objects are
    content-addressed a blob git already holds is shared rather than
    duplicated (files over 1 MB and files whose content sniffs as binary are
    never tracked at all). The trade-off is that the harness's data then
    lives in a store whose pruning schedule is
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
import stat
import tempfile
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

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
    index, and the oversized/binary exclusion sets.

    :class:`xarness.workspace.WorkspaceState` is the only implementation;
    the protocol keeps this module free of a circular import."""

    root: Path
    oversized: tuple[str, ...]
    # Content-detected binaries (path -> size, mtime_ns), maintained by the
    # workspace and shared by every snapshot of it.
    binary: dict[str, tuple[int, int]]

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
    input_bytes: bytes | None = None, *, text: bool = True,
) -> tuple[int, Any, Any]:
    """One git subprocess; output decoded unless ``text=False`` (blob
    contents and merge output are bytes, not text).

    A failed spawn (git missing, or an argv the kernel refuses with E2BIG) is
    a git failure like any other: callers that can degrade (tracking) catch
    :class:`GitWorktreeError`, so it must never surface as a raw ``OSError``.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdin=asyncio.subprocess.PIPE if input_bytes is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise GitWorktreeError(f"could not run git {args[0]}: {exc}") from exc
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input_bytes), timeout=_GIT_TIMEOUT,
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise GitWorktreeError(f"git {' '.join(args)} timed out") from None
    if not text:
        return proc.returncode or 0, stdout, stderr
    return proc.returncode or 0, _text(stdout), _text(stderr)


def _text(data: bytes) -> str:
    """git output as text; undecodable bytes are replaced, never raised."""
    return data.decode(errors="replace")


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

    A fresh index per snapshot would re-hash every file in the worktree;
    git's stat cache makes a persistent one incremental."""
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    key = hashlib.sha1(str(workspace).encode()).hexdigest()[:16]
    directory = Path(state) / "xarness" / "indexes"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{key}.index"


# ----------------------------------------------------------------------
# ignore rules

# Ignore rules the harness applies on top of the repo's own, in gitignore
# syntax. Only *directories* are listed: a build tree is pruned before
# anything is read, which a content check cannot do cheaply (tracking build
# output adds hundreds of MB of objects and makes /diff about a thousand
# times slower). Binaries are detected by content instead (see
# :func:`probe_binary_path`), so no extension belongs here: an extension is
# only a guess, and a wrong guess either bloats the store or hides a file.
# Ignored paths never enter a snapshot tree (see /diff's auto groups).
HARNESS_IGNORES: tuple[str, ...] = (
    "node_modules/", "dist/", "build/", "__pycache__/", ".venv/", ".next/",
    ".xarness/",
)

# Cargo's own ``target/`` and the sandbox's redirected ``target-sandbox/``
# (it sets CARGO_TARGET_DIR) are build output, not source; so is
# ``Cargo.lock`` — text, so content sniffing cannot catch it.
_CARGO_IGNORES: tuple[str, ...] = ("target/", "target-sandbox/", "Cargo.lock")

# Files larger than this are excluded from tracking (snapshots/diffs/undo).
_MAX_TRACKED_SIZE = 1024 * 1024


# The user's own ignore file, in gitignore syntax (ignore-only: no
# re-include), resolved on every snapshot.
_XARNESS_IGNORE = ".xarness/ignore"


def harness_ignores(root: Path) -> tuple[str, ...]:
    """The harness's ignore rules for ``root``, in gitignore syntax.

    The cargo entries are gated: a directory that merely happens to be
    called ``target`` in a non-Cargo project is still tracked, and so is a
    ``Cargo.lock`` left over outside a crate."""
    if (root / "Cargo.toml").is_file():
        return (*_CARGO_IGNORES, *HARNESS_IGNORES)
    return HARNESS_IGNORES


# ----------------------------------------------------------------------
# binary detection

# How much of a file's head is inspected: git's own ``buffer_is_binary``
# window (see :func:`looks_binary`).
TEXT_PROBE_BYTES = 8000
_BINARY_MAGIC = (
    b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF8", b"%PDF-", b"PK\x03\x04",
    b"PK\x05\x06", b"\x1f\x8b", b"BZh", b"\xfd7zXZ", b"7z\xbc\xaf", b"Rar!",
    b"\x28\xb5\x2f\xfd", b"\x7fELF", b"MZ", b"\xfe\xed\xfa", b"\xcf\xfa\xed",
    b"\xca\xfe\xba\xbe", b"SQLite format 3\0", b"OggS", b"fLaC", b"ID3",
    b"II*\0", b"MM\0*", b"8BPS", b"wOFF", b"wOF2", b"OTTO", b"\x1aE\xdf\xa3",
)
# A UTF-16 BOM's NULs are structural, so such a file is not binary (git
# tracks it as text) — but a text tool cannot carry it either, so the tools
# check it separately.
UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")


def looks_binary(head: bytes) -> bool:
    """Whether a file head is binary: a magic prefix or a NUL byte, unless
    it is a UTF-16 BOM file (its NULs are structural) or empty.

    The NUL scan catches everything the format list misses; the list covers
    the formats whose head carries no NUL (PDF, images, ID3…)."""
    return bool(head) and not head.startswith(UTF16_BOMS) and (
        head.startswith(_BINARY_MAGIC)
        or (head[:4] == b"RIFF" and head[8:12] in (b"WEBP", b"AVI ", b"WAVE"))
        or head[4:8] == b"ftyp"
        or b"\0" in head
    )


def probe_binary_path(target: Path) -> bool:
    """Whether the file at ``target`` is binary (see :func:`looks_binary`
    for the test itself). Symlinks and non-files are not: git stores a link
    as text."""
    try:
        if not stat.S_ISREG(os.lstat(target).st_mode):
            return False
        with open(target, "rb") as fh:
            return looks_binary(fh.read(TEXT_PROBE_BYTES))
    except OSError:
        return False


def _classify_binary(
    root: Path, candidates: Iterable[str], cache: dict[str, tuple[int, int]],
) -> tuple[str, ...]:
    """Probe ``candidates``, reusing ``cache`` (path -> size, mtime_ns) so
    an unchanged path costs one lstat and no read. Fills ``cache`` in place."""
    binary: list[str] = []
    candidates = tuple(candidates)
    for rel in candidates:
        try:
            info = os.lstat(root / rel)
            key = (info.st_size, info.st_mtime_ns)
            is_binary = stat.S_ISREG(info.st_mode) and (
                cache.get(rel) == key or probe_binary_path(root / rel)
            )
        except OSError:
            is_binary = False
        if is_binary:
            cache[rel] = key
            binary.append(rel)
        else:
            cache.pop(rel, None)
    # An unchanged binary may not be a candidate (a frozen tracked file is
    # skipped by ``ls-files -m``), so only a path that is gone is dropped.
    for rel in set(cache) - set(candidates):
        if not (root / rel).exists():
            del cache[rel]
    return tuple(binary)


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


async def _changed_paths(root: Path, env: dict[str, str]) -> list[str]:
    """Changed/untracked, non-ignored paths (git's stat cache does it)."""
    rc, out, _ = await _run_git(
        ["ls-files", "-m", "-o", "--exclude-standard", "-z"],
        cwd=root, env=env,
    )
    return [rel for rel in out.split("\0") if rel] if rc == 0 else []


def _is_regular(path: Path) -> bool:
    """A regular file itself — a symlink is a link, not its target."""
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


def _grown_paths(
    root: Path, paths: Iterable[str], limit: int = _MAX_TRACKED_SIZE,
) -> tuple[str, ...]:
    """The ``paths`` that are regular files over ``limit`` bytes; keeping
    them out of ``git add`` stops a huge artifact entering the snapshot."""
    grown = []
    for rel in paths:
        try:
            info = os.lstat(root / rel)
        except OSError:
            continue
        if stat.S_ISREG(info.st_mode) and info.st_size > limit:
            grown.append(rel)
    return tuple(grown)


async def _git_run(
    root: Path, env: dict[str, str], args: list[str], payload: bytes = b"",
) -> str:
    """One git command, ``payload`` on stdin; a failure raises."""
    rc, out, err = await _run_git(
        args, cwd=root, env=env, input_bytes=payload or None,
    )
    if rc != 0:
        raise GitWorktreeError(f"git {args[0]} failed: {err.strip()}")
    return out


async def _index_state(
    root: Path, env: dict[str, str],
) -> tuple[set[str], list[str]]:
    """The snapshot index's paths and those marked assume-unchanged: one
    listing serves both recognising a new path and lifting last snapshot's
    freeze before looking for changes."""
    out = await _git_run(root, env, ["ls-files", "-v", "-z"])
    indexed: set[str] = set()
    frozen: list[str] = []
    for entry in out.split("\0"):
        if not entry:
            continue
        indexed.add(entry[2:])
        if entry[0].islower():  # a lowercase tag means assume-unchanged
            frozen.append(entry[2:])
    return indexed, frozen


async def _unfreeze(
    root: Path, env: dict[str, str], paths: Iterable[str],
) -> None:
    """Lift last snapshot's assume-unchanged flags before scanning."""
    paths = list(paths)
    if paths:
        await _git_run(
            root, env,
            ["update-index", "--no-assume-unchanged", "-z", "--stdin"],
            "".join(f"{path}\0" for path in paths).encode(),
        )


async def _freeze(
    root: Path, env: dict[str, str], paths: Iterable[str], indexed: set[str],
) -> tuple[tuple[str, ...], str]:
    """Pin ``paths`` in the index so ``git add`` neither reads nor hashes them.

    Pathspecs cannot do this: git matches each one against every file in the
    worktree (minutes for thousands) and thousands do not fit in argv
    (``execve`` fails with E2BIG). The index can. An indexed path — a tracked
    file that grew past the limit — is marked assume-unchanged and keeps its
    previous blob, which is what /undo needs; a new one is seeded with the
    empty blob and dropped again by the caller. Returns the seeded paths and
    the null sha to drop them with."""
    live = [p for p in dict.fromkeys(paths) if _is_regular(root / p)]
    if not live:
        return (), ""
    seeded = [p for p in live if p not in indexed]
    empty = ""
    if seeded:
        empty = (await _git_run(root, env, ["hash-object", "-w", "--stdin"])).strip()
        await _git_run(
            root, env, ["update-index", "-z", "--add", "--index-info"],
            b"".join(f"100644 {empty} 0\t{p}\0".encode() for p in seeded),
        )
    await _git_run(
        root, env, ["update-index", "--assume-unchanged", "-z", "--stdin"],
        "".join(f"{p}\0" for p in live).encode(),
    )
    return tuple(seeded), "0" * len(empty)


async def _add_all(
    root: Path, env: dict[str, str], pathspecs: list[str],
) -> None:
    """``git add -A`` with the pathspecs on stdin, never in argv."""
    payload = b"".join(spec.encode() + b"\0" for spec in pathspecs)
    rc, _, err = await _run_git(
        ["add", "-A", "--pathspec-from-file=-", "--pathspec-file-nul"],
        cwd=root, env=env, input_bytes=payload,
    )
    if rc != 0 and "unknown option" in err:  # git < 2.25
        rc, _, err = await _run_git(["add", "-A", "--", *pathspecs], cwd=root, env=env)
    if rc != 0:
        raise GitWorktreeError(f"could not snapshot the workspace: {err.strip()}")


async def snapshot_tree(
    root: Path, exclude: Iterable[str] = (),
    env_base: dict[str, str] | None = None, index: Path | None = None,
    oversized_out: list[str] | None = None,
    binary_cache: dict[str, tuple[int, int]] | None = None,
) -> str:
    """Tree sha capturing the workspace's *current* state.

    Built in a scratch index (``index`` when given, else a temp file):
    ``git add -A`` stages tracked changes, uncommitted changes, and
    untracked-not-ignored files (the user's real index is untouched), then
    ``write-tree`` records it as a tree object. Ignored files are excluded —
    they are not reverted by /undo and don't show in /diff.

    Exclusion is resolved on every snapshot (nothing is frozen per session):
    git's own ignore rules, the harness defaults, ``.xarness/ignore``
    resolved to concrete paths, ``exclude``, and any changed/untracked file
    that has grown past the size limit or sniffs as binary (see
    :func:`probe_binary_path`). Content matches no pathspec, so those last
    two groups are pinned in the index instead (see :func:`_freeze`).

    ``env_base`` carries GIT_DIR/GIT_WORK_TREE for external snapshot stores.
    ``oversized_out`` collects the paths newly found over the limit.
    ``binary_cache`` is filled in place (path -> size, mtime_ns) so the
    caller can persist it and skip re-probing unchanged binaries.
    """
    root = Path(root).resolve()
    if binary_cache is None:
        # Callers that do not persist tracking state (tests, one-off
        # snapshots) still get content-based exclusion; they just re-probe.
        binary_cache = {}

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
        indexed, frozen = await _index_state(root, env)
        await _unfreeze(root, env, frozen)
        # One stat-only scan serves both exclusions below (oversized and
        # binary): cost follows the number of *changed* files, not the tree.
        changed = await _changed_paths(root, env)
        grown = _grown_paths(root, changed)
        if grown and oversized_out is not None:
            oversized_out.extend(grown)
        # .xarness/ignore is resolved now, not pinned: an edit to it takes
        # effect on this snapshot. Its file entries are pinned with the
        # oversized files; only directories stay pathspecs (a directory
        # cannot be an index entry, and each one costs a single pathspec).
        ignored = set(await xarness_ignored_paths(root, env_base))
        files = {p for p in ignored if _is_regular(root / p)}
        binary = await asyncio.to_thread(
            _classify_binary, root, changed, binary_cache,
        )
        seeded, null_sha = await _freeze(
            root, env, (*exclude, *grown, *files, *binary), indexed,
        )
        try:
            await _add_all(root, env, [
                *_harness_pathspecs(root),
                *(s for p in ignored - files for s in _exclude_pathspecs(p)),
            ])
        finally:
            # A placeholder left behind would look like an empty file in
            # every later snapshot.
            if seeded:
                await _git_run(
                    root, env, ["update-index", "-z", "--index-info"],
                    "".join(f"0 {null_sha} 0\t{p}\0" for p in seeded).encode(),
                )
        return (await _git_run(root, env, ["write-tree"])).strip()

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


async def write_blob(repo: RepoStore, data: bytes) -> str:
    """Store ``data`` in the object store and return its blob sha.

    Explicit writes record the shas of the file before and after the tool
    call (see ``WorkspaceState.record``), so /diff can show each agent's own
    delta without snapshotting the workspace on every edit."""
    rc, out, err = await _run_git(
        ["hash-object", "-w", "--stdin"],
        cwd=repo.root, env=repo.git_env(), input_bytes=data,
    )
    if rc != 0:
        raise GitWorktreeError(f"could not store a blob: {err.strip()}")
    return out.strip()


async def read_blob(repo: RepoStore, sha: str) -> bytes:
    """The bytes of one blob (raises when the object is gone: a pruned
    store takes pending provenance with it, exactly like the trees)."""
    rc, out, err = await _run_git(
        ["cat-file", "blob", sha], cwd=repo.root, env=repo.git_env(), text=False,
    )
    if rc != 0:
        raise GitWorktreeError(
            f"could not read blob {sha[:8]}: {_text(err).strip()}"
        )
    return out


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
            index=repo.index_path, binary_cache=repo.binary,
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


# Snapshot trees and recorded blobs are immutable, so their diffs never
# change once the objects exist: /diff reads the same few tree pairs on every
# poll and a recorded provenance pair never moves at all. Remember those
# results for the life of the process, keyed (kind, …) so they share one
# bounded dict (oldest entry evicted); the values are immutable, so no caller
# can corrupt them.
_MEMO_LIMIT = 512
_memo: dict[tuple, Any] = {}


def _memo_put(key: tuple, value: Any) -> Any:
    """Remember one entry (evicting the oldest when full) and return it."""
    if len(_memo) >= _MEMO_LIMIT:
        _memo.pop(next(iter(_memo)))
    _memo[key] = value
    return value


async def _diff_between(repo: RepoStore, base_tree: str, other_tree: str) -> tuple[str, str]:
    """Cached ``(--numstat, --name-status)`` NUL-separated output between two
    snapshot trees. ``--name-status`` is where ``is_new`` comes from (the
    ``A`` rows), so no second ``ls-tree`` pass per tree is needed."""
    key = ("numstat", base_tree, other_tree)
    cached = _memo.get(key)
    if cached is None:
        numstat = await _diff_tree_pair(repo, base_tree, other_tree, "--numstat", "-z")
        status = await _diff_tree_pair(repo, base_tree, other_tree, "--name-status", "-z")
        cached = _memo_put(key, (numstat, status))
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


# Patches one ignored group's /diff lists before it says "… N more files".
_IGNORED_PATCHES = 50


async def ignored_diff(repo: RepoStore, prefix: str) -> str:
    """An "added" patch for an ignored path: one file, or the files under a
    directory group (``prefix`` is the row the summary shows, trailing slash
    included).

    Ignored paths never enter a snapshot tree, so the baseline→worktree diff
    has nothing to compare them against. Diffing each against ``/dev/null``
    is the only view of their contents; a directory's patches are headed with
    the group so the pane keeps naming the row that was clicked. Symlinks are
    skipped (reading one would print whatever it points at) and a big group is
    cut off, since the pane shows one scrollback, not a directory browser."""
    root, env = repo.root, repo.git_env()
    target = root / prefix
    if target.is_dir():
        listing = await ignored_listing(root, env, collapsed=False)
        paths = [path for path in listing if path.startswith(prefix)]
        header = prefix
    elif target.is_file():
        paths, header = [prefix], prefix
    else:
        return ""
    shown = paths[:_IGNORED_PATCHES]
    patches = [
        await _added_patch(root, env, path, header) for path in shown
        if not (root / path).is_symlink()
    ]
    if len(paths) > len(shown):
        patches.append(f"… {len(paths) - len(shown)} more files\n")
    return "".join(patches)


async def _added_patch(
    root: Path, env: dict[str, str], path: str, header: str,
) -> str:
    """One path's ``/dev/null`` diff, its header naming ``header``."""
    rc, out, err = await _run_git(
        ["diff", "--no-index", "--no-color", "--", os.devnull, path],
        cwd=root, env=env,
    )
    if rc > 1:
        raise GitWorktreeError(f"git diff failed: {err.strip()}")
    lines = out.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("diff --git "):
            lines[index] = f"diff --git a/{header} b/{header}"
        elif not line.startswith(("index ", "--- ", "+++ ")):
            break
    return "\n".join(lines) + ("\n" if lines else "")


# The empty blob has to exist in the store before it can be diffed against:
# one write per store, remembered in the same memo (its sha is the same
# everywhere).


async def _blob_pair(
    repo: RepoStore, base: str | None, other: str | None,
) -> tuple[str, str]:
    """The pair to diff, with an absent side mapped to the empty blob."""
    if base and other:
        return base, other
    empty = _memo.get(("empty", repo.root))
    if empty is None:
        empty = _memo_put(("empty", repo.root), await write_blob(repo, b""))
    return base or empty, other or empty


async def blob_diff_stat(
    repo: RepoStore, base: str | None, other: str | None,
) -> tuple[int, int]:
    """``(additions, deletions)`` of one path between two blobs, where
    ``None`` means the path was absent.

    This is the per-author delta /diff reports: a patch between the content
    just before an agent's first write and just after its last one, so an
    earlier agent's hunks in the same file are not credited to it."""
    key = ("blobstat", *await _blob_pair(repo, base, other))
    if key[1] == key[2]:
        return 0, 0
    cached = _memo.get(key)
    if cached is not None:
        return cached
    rc, out, err = await _run_git(
        ["diff", "--numstat", "-z", key[1], key[2]],
        cwd=repo.root, env=repo.git_env(),
    )
    if rc != 0:
        raise GitWorktreeError(f"git diff failed: {err.strip()}")
    files = _parse_numstat_z(out)
    stat = (files[0].additions, files[0].deletions) if files else (0, 0)
    return _memo_put(key, stat)


async def blob_diff_text(
    repo: RepoStore, path: str, base: str | None, other: str | None,
) -> str:
    """Unified diff of one path between two blobs, with the sha-named header
    lines rewritten to the real path so the pane renders it like any other
    file diff."""
    first, second = await _blob_pair(repo, base, other)
    if first == second:
        return ""
    rc, out, err = await _run_git(
        ["diff", "--no-color", first, second],
        cwd=repo.root, env=repo.git_env(),
    )
    if rc != 0:
        raise GitWorktreeError(f"git diff failed: {err.strip()}")
    lines = out.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("@@"):
            break
        if line.startswith("diff --git "):
            lines[index] = f"diff --git a/{path} b/{path}"
        elif line.startswith("--- ") or line.startswith("+++ "):
            side = "a" if line.startswith("--- ") else "b"
            lines[index] = f"{line[:4]}{side}/{path}"
    return "\n".join(lines) + ("\n" if lines else "")


async def merge_texts(
    repo: RepoStore, *, base: bytes, ours: bytes, theirs: bytes,
) -> bytes | None:
    """Three-way merge of one file's contents; None when the changes
    conflict.

    ``WorkspaceState.undo`` reverses a turn by merging that turn's *own*
    edit away: ``base`` is what the turn left, ``ours`` what is on disk now
    (another agent's later edits included) and ``theirs`` what the path held
    before the turn. Overlapping hunks conflict; disjoint ones merge, so a
    later agent's edit elsewhere in the file no longer blocks the undo."""
    with tempfile.TemporaryDirectory(prefix="xarness-merge-") as tmp:
        sides = []
        for name, data in (("ours", ours), ("base", base), ("theirs", theirs)):
            side = Path(tmp) / name
            side.write_bytes(data)
            sides.append(str(side))
        rc, out, err = await _run_git(
            ["merge-file", "-p", "-L", "ours", "-L", "base", "-L", "theirs",
             *sides],
            cwd=repo.root, env=repo.git_env(), text=False,
        )
        if rc < 0:
            raise GitWorktreeError(
                f"git merge-file failed: {_text(err).strip()}"
            )
        if rc > 0:
            return None  # rc is the number of conflicted hunks
        return out


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
