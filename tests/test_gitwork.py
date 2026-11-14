"""Tests for gitwork: direct-write change tracking (tree snapshots,
checkpoints, revert, diff) and the run_bash git-command filter.

Uses a real temp git repo (git is a hard dependency of the feature anyway).
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xarness import gitwork
from xarness.gitwork import (
    detect_repo, git_diff, revert_to_tree, snapshot_tree,
    tree_diff_stat,
)
from xarness.workspace import WorkspaceState, open_workspace


async def open_state(path: Path, **kwargs) -> WorkspaceState:
    """The tracked state of ``path``'s repo, without persisting anything."""
    kwargs.setdefault("persist", False)
    state = await open_workspace(path, **kwargs)
    assert state is not None
    return state


async def diff(state: WorkspaceState) -> gitwork.DiffStat:
    """The pending diff: baseline vs the live worktree."""
    return await tree_diff_stat(
        state, state.baseline_tree, await state.observe(),
    )


def _git(cwd: Path, *args: str, check: bool = True, env: dict | None = None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
        env=env,
    )
    if check:
        assert result.returncode == 0, f"git {' '.join(args)}: {result.stderr}"
    return result.stdout


class GitworkTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)

    def make_repo(self) -> Path:
        repo = self.base / "repo"
        repo.mkdir()
        _git(repo, "init", "-q", "-b", "main", ".")
        _git(repo, "config", "user.email", "t@t")
        _git(repo, "config", "user.name", "t")
        (repo / "app.py").write_text("line1\nline2\n")
        (repo / "pkg").mkdir()
        (repo / "pkg" / "mod.py").write_text("a = 1\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "init")
        return repo

    # ------------------------------------------------------------------
    # detection + setup

    async def test_detect_repo_reads_root(self):
        assert await detect_repo(self.base) is None  # plain directory
        repo = self.make_repo()
        info = await detect_repo(repo)
        assert info is not None
        assert info.root == repo.resolve()

    async def test_open_auto_initializes_non_repo(self):
        """No git init from the user: the harness tracks via an external
        snapshot store — and never drops a .git into the workspace."""
        workspace = self.base / "plain"
        workspace.mkdir()
        (workspace / "notes.txt").write_text("user's file\n")

        with mock.patch.dict(os.environ, {"XDG_STATE_HOME": str(self.base / "state")}):
            state = await open_state(workspace)

        assert state.root == workspace.resolve()
        assert state.external
        with self.assertRaises(ValueError):
            state.git_dir.relative_to(workspace.resolve())  # store is outside
        # No .git appears in the workspace; the user's files are untouched.
        assert not (workspace / ".git").exists()
        assert (workspace / "notes.txt").read_text() == "user's file\n"
        env = {**os.environ, "GIT_DIR": str(state.git_dir)}
        out = _git(workspace, "config", "xarness.managed", env=env)
        assert out.strip() == "true"
        # Snapshots/diffs work against the external store.
        (workspace / "notes.txt").write_text("agent edited\n")
        stat = await diff(state)
        assert [f.path for f in stat.files] == ["notes.txt"]

    async def test_external_store_revert_works(self):
        """In gitless (external-store) mode the full cycle works: snapshot,
        edit, and revert — with no .git ever created in the workspace."""
        workspace = self.base / "plain2"
        workspace.mkdir()
        (workspace / "doc.txt").write_text("base\n")
        with mock.patch.dict(os.environ, {"XDG_STATE_HOME": str(self.base / "state")}):
            state = await open_state(workspace)

        before = await state.refresh()
        (workspace / "doc.txt").write_text("agent edit\n")
        (workspace / "new.txt").write_text("created\n")
        after = await state.refresh()

        await revert_to_tree(state, before, after)

        assert (workspace / "doc.txt").read_text() == "base\n"
        assert not (workspace / "new.txt").exists()
        assert not (workspace / ".git").exists()

    async def test_snapshots_skip_binary_and_oversized_files(self):
        """Binaries (detected by content, not by extension) and files over
        1 MB never enter snapshots: not in /diff, not reverted by /undo."""
        repo = self.make_repo()
        png = b"\x89PNG\r\n\x1a\n" + b"\0" * 16
        (repo / "pic.png").write_bytes(png)
        (repo / "clip.mp4").write_bytes(b"\0\0\0\x18ftypmp42" + b"\0" * 8)
        # A text file with a media extension is text: the extension is not
        # consulted, only the content.
        (repo / "notes.png").write_text("not an image\n")
        # .dat with text content: this one tests the *size* rule, which only
        # applies to paths the content check leaves tracked.
        (repo / "big.dat").write_bytes(b"0" * (1024 * 1024 + 1))
        (repo / "small.txt").write_text("tracked\n")

        state = await open_state(repo)
        (repo / "small.txt").write_text("edited\n")
        (repo / "notes.png").write_text("still text\n")
        (repo / "pic.png").write_bytes(png + b"changed\0")
        (repo / "big.dat").write_bytes(b"1" * (1024 * 1024 + 1))

        stat = await diff(state)
        assert [f.path for f in stat.files] == ["notes.png", "small.txt"]
        # The oversized file was found under the size limit at workspace open.
        assert "big.dat" in state.oversized
        # Binaries are detected by content whether or not they are large.
        assert {"pic.png", "clip.mp4"} <= set(state.binary)
        assert "pic.png" not in state.oversized
        assert "clip.mp4" not in state.oversized

    async def test_content_decides_what_is_binary(self):
        """The head decides, never the extension: a NUL or magic number is binary."""
        repo = self.make_repo()
        (repo / "diagram.png").write_text("<svg>one</svg>\n")
        state = await open_state(repo)

        (repo / "blob.dat").write_bytes(b"head\0tail")
        (repo / "blob.unknown").write_bytes(b"head\0tail")
        (repo / "text.dat").write_text("plain text\n")
        # An archive header with no NUL in it depends on the magic check.
        (repo / "archive.zip").write_bytes(b"PK\x03\x04" + b"x" * 8)
        (repo / "wide.txt").write_bytes("hello\n".encode("utf-16"))
        (repo / "diagram.png").write_text("<svg>two</svg>\n")

        stat = await diff(state)
        assert [f.path for f in stat.files] == [
            "diagram.png", "text.dat", "wide.txt",
        ]
        assert set(state.binary) == {"blob.dat", "blob.unknown", "archive.zip"}

    async def test_binary_probe_is_cached_until_the_file_changes(self):
        """An excluded binary stays untracked forever, so it must not be
        re-read on every snapshot; a change reclassifies it."""
        repo = self.make_repo()
        state = await open_state(repo)
        (repo / "blob.bin").write_bytes(b"\0binary")
        await state.observe()
        assert "blob.bin" in state.binary

        with mock.patch.object(
            gitwork, "probe_binary_path", wraps=gitwork.probe_binary_path,
        ) as probe:
            await state.observe()
            assert probe.call_count == 0  # cache hit: only stat'ed
            (repo / "blob.bin").write_text("now text\n")
            await state.observe()

        assert "blob.bin" not in state.binary
        stat = await diff(state)
        assert "blob.bin" in {f.path for f in stat.files}

    async def test_binary_and_oversized_edge_cases(self):
        """A tracked file frozen as binary is unfrozen once text again, and a
        symlink is stored as its tiny target path, never as the target's size."""
        repo = self.make_repo()
        state = await open_state(repo)

        (repo / "app.py").write_bytes(b"\0binary")
        await state.observe()
        assert "app.py" in state.binary
        (repo / "app.py").write_text("line1\nline2\nchanged\n")
        stat = await diff(state)
        assert [f.path for f in stat.files] == ["app.py"]
        assert "app.py" not in state.binary

        (repo / "big.dat").write_bytes(b"x" * (2 * 1024 * 1024))
        (repo / "link.dat").symlink_to("big.dat")
        await state.observe()
        assert "link.dat" not in state.oversized
        stat = await diff(state)
        assert {f.path for f in stat.files} == {"app.py", "link.dat"}

    async def test_sniffed_binary_is_a_diff_group(self):
        """Detected binaries are a group per file: git has no pattern for them."""
        repo = self.make_repo()
        state = await open_state(repo)

        (repo / "out").mkdir()
        (repo / "out" / "app.bin").write_bytes(b"\0bin")

        info = await state.diff("A")
        assert info.files == []
        groups = {group.prefix: group for group in info.groups}
        assert groups["out/app.bin"].kind == "drift"

        # A recorded command that created it labels it `auto`.
        state.record(
            session="A", tool="run_bash", turn="t1", paths=(),
            auto=("out/app.bin",),
        )
        info = await state.diff("A")
        groups = {group.prefix: group for group in info.groups}
        assert groups["out/app.bin"].kind == "auto"

    async def test_mid_session_oversized_file_is_dropped(self):
        """A file that grows past 1 MB mid-session is dropped from the
        snapshot instead of being diffed/reverted forever."""
        repo = self.make_repo()
        (repo / "data.dat").write_text("small\n")
        (repo / "odd[1].dat").write_text("small\n")
        (repo / "odd1.dat").write_text("keep me\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "bins")

        state = await open_state(repo)
        assert "data.dat" not in state.oversized

        big = b"0" * (gitwork._MAX_TRACKED_SIZE + 1)
        (repo / "data.dat").write_bytes(big)
        (repo / "odd[1].dat").write_bytes(big)
        (repo / "app.py").write_text("edited\n")

        stat = await diff(state)
        paths = [f.path for f in stat.files]
        # Neither the grown files nor a phantom "deleted" row for them: the
        # file is ignored, so /diff says nothing about it.
        assert paths == ["app.py"]
        # Grown files are pinned now, so the next snapshot does not re-hash them.
        assert "data.dat" in state.oversized
        assert "odd[1].dat" in state.oversized

        # The snapshot keeps the pre-growth blob for each of them...
        tree = await state.refresh()
        tracked = {
            path: meta.split()[2]
            for meta, path in (
                line.split("\t", 1)
                for line in _git(repo, "ls-tree", "-r", tree).splitlines()
            )
        }
        # ...and the glob-looking name did not take its lookalike with it.
        assert "odd1.dat" in tracked
        assert tracked["odd1.dat"] == _git(
            repo, "rev-parse", "HEAD:odd1.dat").strip()
        assert tracked["odd[1].dat"] == _git(
            repo, "rev-parse", "HEAD:odd[1].dat").strip()
        assert tracked["data.dat"] == _git(repo, "rev-parse", "HEAD:data.dat").strip()

    async def test_oversized_file_created_mid_session_is_not_tracked(self):
        """A file that appears mid-session already over the limit is left out
        of the snapshot (mode-0 index record), so /diff never shows it."""
        repo = self.make_repo()
        state = await open_state(repo)

        big = b"0" * (gitwork._MAX_TRACKED_SIZE + 1)
        (repo / "dump.dat").write_bytes(big)
        (repo / "app.py").write_text("edited\n")

        stat = await diff(state)
        assert [f.path for f in stat.files] == ["app.py"]
        assert "dump.dat" in state.oversized
        tree = await state.refresh()
        assert "dump.dat" not in _git(repo, "ls-tree", "-r", "--name-only", tree).split()

    async def test_cargo_build_dirs_are_not_tracked(self):
        """A Cargo project's build trees never enter snapshots: the sandbox
        sets CARGO_TARGET_DIR=target-sandbox, so the agent's own first build
        would otherwise bury /diff (and /undo) in object files."""
        repo = self.make_repo()
        (repo / "Cargo.toml").write_text("[package]\nname = 'x'\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "cargo")

        state = await open_state(repo)

        # Both trees appear *after* the session started (first build).
        for tree in ("target", "target-sandbox"):
            (repo / tree / "debug").mkdir(parents=True)
            (repo / tree / "debug" / "app.o").write_bytes(b"\x7fELF")
        (repo / "src").mkdir()
        (repo / "src" / "main.rs").write_text("fn main() {}\n")

        stat = await diff(state)
        assert [f.path for f in stat.files] == ["src/main.rs"]
        listing = await gitwork.ignored_listing(repo)
        assert "target" in listing and "target-sandbox" in listing

    async def test_cargo_lock_is_not_tracked(self):
        """The first build writes Cargo.lock; it is resolved output, so it
        stays out of snapshots (only a Cargo project ignores it)."""
        repo = self.make_repo()
        (repo / "Cargo.toml").write_text("[package]\nname = 'x'\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "cargo")

        state = await open_state(repo)

        (repo / "Cargo.lock").write_text("version = 3\n")
        (repo / "app.py").write_text("edited\n")

        stat = await diff(state)
        assert [f.path for f in stat.files] == ["app.py"]
        assert "Cargo.lock" in await gitwork.ignored_listing(repo)

    async def test_cargo_lock_tracked_without_cargo_toml(self):
        """A stray `Cargo.lock` outside a crate is just a file."""
        repo = self.make_repo()
        (repo / "Cargo.lock").write_text("v1\n")

        state = await open_state(repo)

        (repo / "Cargo.lock").write_text("v2\n")
        stat = await diff(state)
        assert [f.path for f in stat.files] == ["Cargo.lock"]

    async def test_target_dir_tracked_without_cargo_toml(self):
        """Not every `target/` is cargo's: without Cargo.toml it stays a
        normal tracked directory."""
        repo = self.make_repo()
        (repo / "target").mkdir()
        (repo / "target" / "notes.txt").write_text("user data\n")

        state = await open_state(repo)

        (repo / "target" / "notes.txt").write_text("edited\n")
        stat = await diff(state)
        assert [f.path for f in stat.files] == ["target/notes.txt"]

    async def test_xarness_ignore_is_resolved_per_snapshot(self):
        """`.xarness/ignore` (gitignore syntax) takes effect on the next
        snapshot — writing it mid-session is enough."""
        repo = self.make_repo()
        (repo / "out").mkdir()
        (repo / "out" / "artifact.txt").write_text("v1\n")
        (repo / "scratch").mkdir()
        (repo / "scratch" / "note.txt").write_text("v1\n")

        state = await open_state(repo)
        before = await state.refresh()
        names = _git(repo, "ls-tree", "-r", "--name-only", before).splitlines()
        assert "out/artifact.txt" in names

        # Now the user tells the harness to ignore both directories.
        (repo / ".xarness").mkdir()
        (repo / ".xarness" / "ignore").write_text("out/\nscratch/\n")
        (repo / "out" / "artifact.txt").write_text("v2\n")
        (repo / "scratch" / "note.txt").write_text("v2\n")
        (repo / "app.py").write_text("edited\n")

        stat = await diff(state)
        assert [f.path for f in stat.files] == ["app.py"]

    async def test_ignore_paths_are_not_globs(self):
        """A resolved ignore path is excluded literally: a name like
        `x[1].dat` must not become a character class that takes `x1.dat`
        with it."""
        repo = self.make_repo()
        (repo / ".xarness").mkdir()
        # gitignore syntax, so the brackets are escaped there too.
        (repo / ".xarness" / "ignore").write_text("x\\[1\\].dat\n")
        (repo / "x[1].dat").write_text("v1\n")
        (repo / "x1.dat").write_text("v1\n")

        state = await open_state(repo)
        (repo / "x[1].dat").write_text("v2\n")
        (repo / "x1.dat").write_text("v2\n")

        stat = await diff(state)
        assert [f.path for f in stat.files] == ["x1.dat"]

    async def test_harness_build_output_is_excluded(self):
        """node_modules/dist/build etc. never enter a snapshot: their blobs
        would bloat the object store and make /diff unusable."""
        repo = self.make_repo()
        state = await open_state(repo)

        for name in ("node_modules", "dist", "build"):
            (repo / name).mkdir()
            (repo / name / "out.js").write_text("x\n")
        (repo / "app.py").write_text("edited\n")

        stat = await diff(state)
        assert [f.path for f in stat.files] == ["app.py"]
        listing = await gitwork.ignored_listing(repo)
        assert set(listing) >= {"node_modules", "dist", "build"}

    async def test_open_snapshots_uncommitted_state(self):
        """The baseline tree captures uncommitted + untracked files, so /diff
        starts empty even in a dirty repo."""
        repo = self.make_repo()
        (repo / "app.py").write_text("line1\nedited\n")  # uncommitted
        (repo / "draft.txt").write_text("untracked\n")

        state = await open_state(repo)
        assert (await diff(state)).files == []

    async def test_open_respects_allow_init_false(self):
        workspace = self.base / "plain"
        workspace.mkdir()
        assert await open_workspace(workspace, allow_init=False) is None
        assert not (workspace / ".git").exists()

    async def test_diffs_cover_the_whole_repo(self):
        """--workspace on a subdirectory only sets the sandbox's working
        directory: tracking covers the repo root, with repo-root-relative
        paths."""
        repo = self.make_repo()
        (repo / "f").mkdir()
        (repo / "f" / "notes.txt").write_text("tracked\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "f")

        state = await open_state(repo / "f")
        assert state.root == repo.resolve()

        (repo / "f" / "notes.txt").write_text("tracked\nagent edit\n")
        (repo / "stray.txt").write_text("untracked\n")
        stat = await diff(state)
        by_path = {f.path: f for f in stat.files}
        assert set(by_path) == {"f/notes.txt", "stray.txt"}
        assert by_path["f/notes.txt"].additions == 1
        assert by_path["stray.txt"].is_new

    # ------------------------------------------------------------------
    # snapshots never touch the user's index / HEAD

    async def test_snapshot_leaves_index_and_head_alone(self):
        repo = self.make_repo()
        (repo / "app.py").write_text("line1\nstaged\n")
        _git(repo, "add", "app.py")  # user staged something
        head_before = _git(repo, "rev-parse", "HEAD").strip()

        sha = await snapshot_tree(repo)

        assert sha
        assert _git(repo, "rev-parse", "HEAD").strip() == head_before
        status = _git(repo, "status", "--porcelain")
        assert "M  app.py" in status  # still staged, index untouched

    async def test_snapshot_captures_untracked_not_ignored(self):
        repo = self.make_repo()
        (repo / ".gitignore").write_text("ignored*\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "gitignore")
        (repo / "ignored.bin").write_text("x\n")
        (repo / "kept.txt").write_text("y\n")

        sha = await snapshot_tree(repo)
        names = _git(repo, "ls-tree", "-r", "--name-only", sha).splitlines()
        assert "kept.txt" in names
        assert "ignored.bin" not in names

    # ------------------------------------------------------------------
    # checkpoint + revert

    async def test_checkpoint_and_revert_restore_workspace_state(self):
        """Edits, deletions, and creations since the checkpoint are all
        undone by revert_to_tree."""
        repo = self.make_repo()
        state = await open_state(repo)

        sha = await state.refresh()

        # Agent turn: modify, delete, create.
        (repo / "app.py").write_text("line1\nagent edit\n")
        (repo / "pkg" / "mod.py").unlink()
        (repo / "new_file.py").write_text("created\n")

        await revert_to_tree(state, sha)

        assert (repo / "app.py").read_text() == "line1\nline2\n"
        assert (repo / "pkg" / "mod.py").read_text() == "a = 1\n"
        assert not (repo / "new_file.py").exists()
        # Nothing staged, HEAD unmoved.
        assert _git(repo, "status", "--porcelain").strip() == ""

    async def test_revert_restores_untracked_baseline_files(self):
        """Files untracked at checkpoint time are restored too — the snapshot
        tree holds their content, not just tracked files'."""
        repo = self.make_repo()
        (repo / "draft.txt").write_text("untracked\n")
        state = await open_state(repo)
        sha = await state.refresh()

        (repo / "draft.txt").unlink()
        await revert_to_tree(state, sha)

        assert (repo / "draft.txt").read_text() == "untracked\n"
        # Still untracked — the user's index was never involved.
        assert "??" in _git(repo, "status", "--porcelain")

    async def test_revert_keeps_ignored_files(self):
        repo = self.make_repo()
        (repo / ".gitignore").write_text("ignored*\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "gitignore")
        state = await open_state(repo)
        sha = await state.refresh()

        (repo / "ignored.dir").mkdir()
        (repo / "ignored.dir" / "artifact.bin").write_text("x\n")
        await revert_to_tree(state, sha)

        assert (repo / "ignored.dir" / "artifact.bin").exists()

    async def test_revert_to_missing_tree_fails_loudly(self):
        repo = self.make_repo()
        state = await open_state(repo)
        with self.assertRaises(gitwork.GitWorktreeError):
            await revert_to_tree(state, "0" * 40)

    async def test_revert_deletes_created_non_ascii_paths(self):
        """Without -z, git quotes non-ASCII paths and the file created since
        the checkpoint silently survived /undo."""
        repo = self.make_repo()
        state = await open_state(repo)
        sha = await state.refresh()
        (repo / "café.txt").write_text("new\n")
        await revert_to_tree(state, sha)
        assert not (repo / "café.txt").exists()

    async def test_rename_is_reported_at_its_new_path(self):
        """Brace-style renames ('a/{old => new}/f') used to lose the prefix."""
        repo = self.make_repo()
        (repo / "a" / "old").mkdir(parents=True)
        (repo / "a" / "old" / "f.txt").write_text("x\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "add")
        state = await open_state(repo)
        (repo / "a" / "new").mkdir()
        (repo / "a" / "old" / "f.txt").rename(repo / "a" / "new" / "f.txt")
        (repo / "a" / "old").rmdir()

        stat = await diff(state)
        assert {f.path for f in stat.files} == {"a/new/f.txt"}

    async def test_grown_files_drop_out_of_the_snapshot(self):
        """The 1 MB rule holds mid-session, not just at setup."""
        repo = self.make_repo()
        state = await open_state(repo)
        (repo / "big.bin").write_text("x" * (gitwork._MAX_TRACKED_SIZE + 1))
        sha = await state.refresh()
        names = _git(repo, "ls-tree", "-r", "--name-only", sha).splitlines()
        assert "big.bin" not in names
        assert "app.py" in names

    async def test_snapshot_index_is_reused_across_snapshots(self):
        """A persistent index keeps git's stat cache; a fresh one per
        snapshot would re-hash the whole worktree twice per turn."""
        repo = self.make_repo()
        state = await open_state(repo)
        await state.refresh()
        assert state.index_path.is_file()
        assert repo not in state.index_path.parents  # never inside the repo

    # ------------------------------------------------------------------
    # keeping a workspace's big files out of argv

    async def test_many_exclusions_never_reach_the_command_line(self):
        """Oversized paths are pinned in the index: thousands blow ARG_MAX."""
        repo = self.make_repo()
        (repo / "data").mkdir()
        paths = []
        for i in range(3000):
            paths.append(f"data/f{i}.dat")
            (repo / paths[-1]).write_text("x")

        seen: list[list[str]] = []
        real = gitwork._run_git

        async def recording(args, *a, **kw):
            seen.append(list(args))
            return await real(args, *a, **kw)

        with mock.patch.object(gitwork, "_run_git", recording):
            tree = await snapshot_tree(repo, exclude=paths)

        assert seen
        # No command line carries the exclusion list...
        assert max(sum(len(arg) + 1 for arg in args) for args in seen) < 64 * 1024
        assert not any(
            "f2999.dat" in arg for args in seen for arg in args
        )
        # ...git reads the pathspecs from stdin instead...
        assert any("--pathspec-from-file=-" in args for args in seen)
        # ...and none of the excluded paths entered the snapshot.
        names = _git(repo, "ls-tree", "-r", "--name-only", tree).splitlines()
        assert "app.py" in names
        assert not [n for n in names if n.startswith("data/")]

    async def test_add_falls_back_to_argv_on_old_git(self):
        """git < 2.25 has no --pathspec-from-file: hand the (short) pathspec
        list over as argv instead."""
        seen: list[list[str]] = []

        async def fake(args, *a, **kw):
            seen.append(list(args))
            if "--pathspec-from-file=-" in args:
                return 129, "", "error: unknown option `pathspec-from-file'"
            return 0, "", ""

        specs = [f":(exclude,glob)**/d{i}/**" for i in range(3)]
        with mock.patch.object(gitwork, "_run_git", fake):
            await gitwork._add_all(Path("/does-not-matter"), {}, specs)

        adds = [args for args in seen if args[0] == "add"]
        assert [args[0] for args in adds] == ["add", "add"]
        assert adds[1] == ["add", "-A", "--", *specs]

    async def test_a_git_spawn_failure_is_a_git_error(self):
        """A refused exec must be a GitWorktreeError, so tracking can degrade."""
        with mock.patch.object(
            gitwork.asyncio, "create_subprocess_exec",
            side_effect=OSError(7, "Argument list too long"),
        ), self.assertRaises(gitwork.GitWorktreeError) as caught:
            await gitwork._run_git(["add", "-A"])
        assert "Argument list too long" in str(caught.exception)

    # ------------------------------------------------------------------
    # diff

    async def test_diff_stat_tracks_edits_and_new_files(self):
        repo = self.make_repo()
        state = await open_state(repo)

        (repo / "app.py").write_text("line1\nline2\nline3\n")
        (repo / "added.txt").write_text("new\n")
        (repo / "pkg" / "mod.py").unlink()

        stat = await diff(state)
        by_path = {f.path: f for f in stat.files}
        assert set(by_path) == {"app.py", "added.txt", "pkg/mod.py"}
        assert by_path["app.py"].additions == 1
        assert by_path["added.txt"].is_new
        assert by_path["pkg/mod.py"].deletions == 1
        assert stat.additions == 2 and stat.deletions == 1

    async def test_full_and_file_diff(self):
        repo = self.make_repo()
        state = await open_state(repo)
        (repo / "app.py").write_text("line1\nagent\n")

        full = await git_diff(state)
        assert "app.py" in full and "agent" in full
        one = await git_diff(state, "app.py")
        assert "app.py" in one
        other = await git_diff(state, "pkg/mod.py")
        assert other.strip() == ""

    async def test_diff_detects_renames(self):
        repo = self.make_repo()
        state = await open_state(repo)
        (repo / "app.py").rename(repo / "renamed.py")

        stat = await diff(state)
        assert {f.path for f in stat.files} == {"renamed.py"}

    # ------------------------------------------------------------------
    # accept (re-baselining)

    async def test_accept_changes_rebaselines(self):
        repo = self.make_repo()
        state = await open_state(repo)
        (repo / "app.py").write_text("feature\n")
        state.record(
            session="A", tool="edit_file", turn="t1",
            paths=("app.py",), auto=(),
        )
        assert (await diff(state)).files

        sha = await state.accept()

        assert state.baseline_tree == sha
        assert (await diff(state)).files == []  # /diff resets

        # Later work measures against the new baseline only.
        (repo / "app.py").write_text("feature\nmore\n")
        (repo / "second.py").write_text("two\n")
        state.record(
            session="A", tool="edit_file", turn="t2",
            paths=("app.py", "second.py"), auto=(),
        )
        stat = await diff(state)
        by_path = {f.path: f for f in stat.files}
        assert set(by_path) == {"app.py", "second.py"}
        assert stat.additions == 2

    # ------------------------------------------------------------------
    # the generated git shim (the read-only policy at execution time)

    def shim(self) -> Path:
        """Write git_shim_script() into a temp bin dir and return that dir."""
        if not hasattr(self, "_shim_dir"):
            shim_dir = self.base / "shim"
            shim_dir.mkdir()
            script = shim_dir / "git"
            script.write_text(gitwork.git_shim_script())
            script.chmod(0o755)
            self._shim_dir = shim_dir
        return self._shim_dir

    def run_shim(self, command: str, cwd: Path, *,
                 path: str | None = None) -> subprocess.CompletedProcess:
        """Run ``command`` through ``/bin/sh`` with the shim first on PATH."""
        env = {
            **os.environ,
            "PATH": path or f"{self.shim()}:{os.environ['PATH']}",
        }
        return subprocess.run(
            ["/bin/sh", "-c", command], cwd=cwd, capture_output=True,
            text=True, env=env, check=False,
        )

    def test_shim_blocks_mutations_in_place(self):
        """A blocked git call fails where it stands: statements around it
        still run, and the message says the command did not run."""
        repo = self.make_repo()
        result = self.run_shim(
            "echo before; git reset --hard; echo after", repo)
        assert result.returncode == 0  # the last statement's status
        assert "before" in result.stdout and "after" in result.stdout
        assert "blocked, read-only here" in result.stderr

    def test_shim_stops_at_and(self):
        """``&&`` stops at the block, exactly as a failing command should."""
        repo = self.make_repo()
        result = self.run_shim(
            "echo before && git add -A && echo after", repo)
        assert result.returncode != 0
        assert "before" in result.stdout
        assert "after" not in result.stdout + result.stderr
        assert "blocked, read-only here" in result.stderr

    def test_shim_blocks_every_mutation(self):
        repo = self.make_repo()
        for command in (
            "git checkout other-branch",
            "git switch main",
            "git checkout -b new",
            "git checkout -- app.py",
            "git branch feature",
            "git branch -D agent/x",
            "git tag v1",
            "git tag -d v1",
            "git reset --hard",
            "git reset HEAD~1",
            "git rebase main",
            "git worktree add ../wt",
            "git update-ref refs/heads/main HEAD~1",
            "git symbolic-ref HEAD refs/heads/other",
            "git stash",
            "git clean -fd",
            "git merge main",
            "git push",
            "git pull",
            "git commit --amend",
            "git config user.email a@b.c",
            "git config user.email",
        ):
            assert "blocked, read-only here" in self.run_shim(
                command, repo).stderr, command
        # None of that mutated the repo.
        assert _git(repo, "symbolic-ref", "--short", "HEAD").strip() == "main"
        assert _git(repo, "status", "--porcelain").strip() == ""

    def test_shim_allows_read_only_inspection(self):
        repo = self.make_repo()
        for command in (
            "git status",
            "git status --short",
            "git diff HEAD",
            "git log --oneline",
            "git blame app.py",
            "git show HEAD",
            "git rev-parse HEAD",
            "git config --get user.name",
            "git config --list",
            "git branch",
            "git branch --show-current",
            "git branch -a",
            "git tag",
            "git tag -l 'v1*'",
            "git stash list",
            "git stash show",
            "git remote -v",
            "git reflog",
            "git reflog show",
            "git check-ignore pkg/",
            "git -C . status",
            "git --no-pager log --oneline",
        ):
            result = self.run_shim(command, repo)
            assert "blocked" not in result.stderr, (command, result.stderr)

    def test_shim_reads_through_shell_wrappers(self):
        """An env prefix or a nested ``sh -c`` still resolves to the shim
        (and mentioning git is not running it)."""
        repo = self.make_repo()
        assert "blocked" in self.run_shim(
            "FOO=1 git reset --hard", repo).stderr
        assert "blocked" not in self.run_shim(
            "FOO=1 git status", repo).stderr
        assert "blocked" not in self.run_shim(
            "sh -c 'git status'", repo).stderr
        assert "blocked" in self.run_shim(
            "sh -c 'git reset --hard'", repo).stderr
        assert "blocked" not in self.run_shim(
            'echo "git reset --hard"', repo).stderr
        assert "blocked" not in self.run_shim(
            "grep -q line1 app.py", repo).stderr

    def test_shim_reports_a_missing_real_git(self):
        """Only the shim on PATH: say so instead of silently doing nothing."""
        result = self.run_shim("git status", self.make_repo(),
                               path=str(self.shim()))
        assert result.returncode == 127
        assert "found no real git binary" in result.stderr

    # ------------------------------------------------------------------
    # ignore rules are resolved per snapshot, never pinned

    async def test_mid_session_gitignore_change_takes_effect(self):
        """A `.gitignore` written mid-session applies to the next snapshot;
        paths already in the baseline stay tracked."""
        repo = self.make_repo()
        (repo / "kept.txt").write_text("k\n")
        state = await open_state(repo)

        (repo / ".gitignore").write_text("junk.log\n")
        (repo / "junk.log").write_text("noise\n")
        (repo / "new.txt").write_text("n\n")
        assert {f.path for f in (await diff(state)).files} == {
            ".gitignore", "new.txt",
        }

        # A path that is already in the baseline stays there: the rule does
        # not retroactively untrack it.
        (repo / ".gitignore").write_text("junk.log\nkept.txt\n")
        (repo / "kept.txt").write_text("k2\n")
        assert "kept.txt" in {f.path for f in (await diff(state)).files}


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
