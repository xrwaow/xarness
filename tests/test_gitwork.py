"""Tests for gitwork: direct-write change tracking (tree snapshots,
checkpoints, revert, diff) and the run_bash git-command filter.

Uses a real temp git repo (git is a hard dependency of the feature anyway).
"""

import asyncio
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xarness import gitwork
from xarness.gitwork import (
    GitInfo, TurnCheckpoint, accept_changes, attributed_diff_stat, check_blocked_git,
    checkpoint, detect_repo, diff_stat, git_diff, revert_to_tree, revert_turn,
    setup_tracking, snapshot_tree, tree_diff_stat, undo_last_turn,
)


def _git(cwd: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
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

    async def test_detect_repo_reads_head(self):
        assert await detect_repo(self.base) is None  # plain directory
        repo = self.make_repo()
        info = await detect_repo(repo)
        assert info is not None
        assert info.root == repo.resolve()
        assert info.head_branch == "main"
        assert info.base_ref is not None

    async def test_detect_repo_unborn_head_has_no_base_ref(self):
        repo = self.base / "unborn"
        repo.mkdir()
        _git(repo, "init", "-q", "-b", "main", ".")
        info = await detect_repo(repo)
        assert info is not None
        assert info.base_ref is None

    async def test_setup_tracking_auto_initializes_non_repo(self):
        """No git init from the user: the harness sets everything up itself."""
        workspace = self.base / "plain"
        workspace.mkdir()
        (workspace / "notes.txt").write_text("user's file\n")

        info, notes = await setup_tracking(workspace, "sess1")

        assert info is not None
        assert info.workspace == workspace.resolve()
        # The user's file is untouched, and the repo is harness-managed.
        assert (workspace / "notes.txt").read_text() == "user's file\n"
        assert any("initialized" in n for n in notes)
        out = _git(workspace, "config", "xarness.managed")
        assert out.strip() == "true"

    async def test_setup_tracking_snapshots_uncommitted_state(self):
        """The baseline tree captures uncommitted + untracked files, so /diff
        starts empty even in a dirty repo."""
        repo = self.make_repo()
        (repo / "app.py").write_text("line1\nedited\n")  # uncommitted
        (repo / "draft.txt").write_text("untracked\n")

        info, _ = await setup_tracking(repo, "sess1")

        assert info is not None
        stat = await diff_stat(info)
        assert stat.files == []

    async def test_setup_tracking_respects_allow_init_false(self):
        workspace = self.base / "plain"
        workspace.mkdir()
        info, _ = await setup_tracking(workspace, "sess1", allow_init=False)
        assert info is None
        assert not (workspace / ".git").exists()

    async def test_setup_tracking_excludes_default_noise_dirs(self):
        """The default noise dirs are written to the repo's local
        info/exclude, so snapshots skip them even with no .gitignore."""
        repo = self.make_repo()
        info, _ = await setup_tracking(repo, "sess1")
        assert info is not None
        (repo / "node_modules").mkdir()
        (repo / "node_modules" / "pkg.js").write_text("x\n")
        (repo / "kept.txt").write_text("kept\n")

        sha = await snapshot_tree(repo)
        names = _git(repo, "ls-tree", "-r", "--name-only", sha).splitlines()
        assert "kept.txt" in names
        assert not any(n.startswith("node_modules/") for n in names)

    async def test_ensure_default_excludes_is_idempotent(self):
        """Re-running setup (a session resume) doesn't duplicate the block."""
        repo = self.make_repo()
        await setup_tracking(repo, "sess1")
        await setup_tracking(repo, "sess2")

        exclude = (repo / ".git" / "info" / "exclude").read_text()
        assert exclude.count("# xarness: default ignores") == 1
        assert "/node_modules/" in exclude

    async def test_ensure_default_excludes_leaves_gitignore_alone(self):
        """Only .git/info/exclude is written; the user's .gitignore is theirs."""
        repo = self.make_repo()
        gitignore = repo / ".gitignore"
        gitignore.write_text("mine\n")

        await setup_tracking(repo, "sess1")

        assert gitignore.read_text() == "mine\n"

    async def test_setup_tracking_scopes_to_workspace_subdir(self):
        """--workspace pointing at a subdirectory of a bigger repo: the
        session is scoped to that subtree — snapshots, diffs, and reverts
        only ever touch it."""
        repo = self.make_repo()
        (repo / "f").mkdir()
        (repo / "f" / "notes.txt").write_text("tracked\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "f")
        (repo / "stray.txt").write_text("untracked\n")  # outside f

        info, notes = await setup_tracking(repo / "f", "sess1")

        assert info is not None
        assert info.subtree == "f"
        assert info.agent_workspace == repo / "f"
        assert any("scoped to f/" in n for n in notes)

        # Edits outside the subtree don't show in the diff; edits inside do,
        # with subtree-relative paths. A file created inside the subtree
        # after setup shows up as new.
        (repo / "f" / "draft.txt").write_text("untracked\n")
        (repo / "f" / "notes.txt").write_text("tracked\nagent edit\n")
        (repo / "app.py").write_text("outside the subtree\n")
        stat = await diff_stat(info)
        by_path = {f.path: f for f in stat.files}
        assert set(by_path) == {"notes.txt", "draft.txt"}
        assert by_path["notes.txt"].additions == 1
        assert by_path["draft.txt"].is_new

        # Revert restores the subtree (and only it).
        sha = await checkpoint(info)
        (repo / "f" / "extra.txt").write_text("created\n")
        (repo / "f" / "notes.txt").unlink()
        await revert_to_tree(info, sha)
        assert (repo / "f" / "notes.txt").read_text() == "tracked\nagent edit\n"
        assert not (repo / "f" / "extra.txt").exists()
        assert (repo / "app.py").read_text() == "outside the subtree\n"

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
        info, _ = await setup_tracking(repo, "sess1")
        assert info is not None

        sha = await checkpoint(info)

        # Agent turn: modify, delete, create.
        (repo / "app.py").write_text("line1\nagent edit\n")
        (repo / "pkg" / "mod.py").unlink()
        (repo / "new_file.py").write_text("created\n")

        await revert_to_tree(info, sha)

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
        info, _ = await setup_tracking(repo, "sess1")
        assert info is not None
        sha = await checkpoint(info)

        (repo / "draft.txt").unlink()
        await revert_to_tree(info, sha)

        assert (repo / "draft.txt").read_text() == "untracked\n"
        # Still untracked — the user's index was never involved.
        assert "??" in _git(repo, "status", "--porcelain")

    async def test_revert_keeps_ignored_files(self):
        repo = self.make_repo()
        (repo / ".gitignore").write_text("ignored*\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "gitignore")
        info, _ = await setup_tracking(repo, "sess1")
        assert info is not None
        sha = await checkpoint(info)

        (repo / "ignored.dir").mkdir()
        (repo / "ignored.dir" / "artifact.bin").write_text("x\n")
        await revert_to_tree(info, sha)

        assert (repo / "ignored.dir" / "artifact.bin").exists()

    async def test_revert_to_missing_tree_fails_loudly(self):
        repo = self.make_repo()
        info, _ = await setup_tracking(repo, "sess1")
        assert info is not None
        with self.assertRaises(gitwork.GitWorktreeError):
            await revert_to_tree(info, "0" * 40)

    # ------------------------------------------------------------------
    # revert_turn: reverse one turn's diff, preserving later edits

    async def test_revert_turn_reverses_only_the_turns_diff(self):
        """A turn's edits are reversed; an unrelated file changed after the
        turn (outside the session) survives."""
        repo = self.make_repo()
        info, _ = await setup_tracking(repo, "sess1")
        assert info is not None

        before = await checkpoint(info)
        (repo / "app.py").write_text("agent edit\n")  # the turn's edit
        after = await checkpoint(info)
        (repo / "unrelated.txt").write_text("manual edit\n")  # after the turn

        conflict = await revert_turn(info, before, after)

        assert conflict is None
        assert (repo / "app.py").read_text() == "line1\nline2\n"
        assert (repo / "unrelated.txt").read_text() == "manual edit\n"

    async def test_revert_turn_conflict_leaves_workspace_untouched(self):
        """A manual edit on the same lines the turn changed conflicts; the
        revert reports it and changes nothing."""
        repo = self.make_repo()
        info, _ = await setup_tracking(repo, "sess1")
        assert info is not None

        before = await checkpoint(info)
        (repo / "app.py").write_text("agent change\n")  # the turn's edit
        after = await checkpoint(info)
        (repo / "app.py").write_text("manual change\n")  # overlapping manual edit

        conflict = await revert_turn(info, before, after)

        assert conflict is not None
        # A readable summary, not raw merge-tree output (tree oids, stage
        # lines, git's internal CONFLICT notices).
        assert "app.py" in conflict
        assert "CONFLICT" not in conflict
        assert (repo / "app.py").read_text() == "manual change\n"  # untouched

    async def test_revert_turn_is_a_noop_when_the_turn_made_no_edits(self):
        """before_tree == after_tree (a plan-mode or no-edit turn): nothing to
        reverse, and drift made since is left alone."""
        repo = self.make_repo()
        info, _ = await setup_tracking(repo, "sess1")
        assert info is not None
        tree = await checkpoint(info)
        (repo / "manual.txt").write_text("manual\n")

        result = await revert_turn(info, tree, tree)

        assert result is None
        assert (repo / "manual.txt").read_text() == "manual\n"

    async def test_revert_turn_noop_makes_no_git_calls(self):
        """The no-edit early return skips the whole merge machinery — no
        commit wrapping, no merge-tree, no revert."""
        repo = self.make_repo()
        info, _ = await setup_tracking(repo, "sess1")
        assert info is not None
        tree = await checkpoint(info)

        with mock.patch.object(gitwork, "_wrap_commit", new=mock.AsyncMock()) as wrap, \
                mock.patch.object(gitwork, "_run_git", new=mock.AsyncMock()) as run:
            result = await revert_turn(info, tree, tree)

        assert result is None
        wrap.assert_not_called()
        run.assert_not_called()

    async def test_tree_diff_stat_cache_returns_fresh_rows(self):
        """Repeated diffs of the same immutable tree pair are cached, but each
        call hands back fresh rows so a caller can't corrupt the cache."""
        repo = self.make_repo()
        info, _ = await setup_tracking(repo, "sess1")
        assert info is not None
        before = await checkpoint(info)
        (repo / "app.py").write_text("agent\n")
        after = await checkpoint(info)

        first = await tree_diff_stat(info, before, after)
        first.files[0].additions = 999  # mutate the returned row
        second = await tree_diff_stat(info, before, after)

        assert second.files[0].additions == 1

    async def test_revert_turns_applies_newest_first(self):
        """A range reverts each turn's own diff; the net effect is the state
        before the oldest turn, with no later turn's edit left behind."""
        repo = self.make_repo()
        info, _ = await setup_tracking(repo, "sess1")
        assert info is not None

        before1 = await checkpoint(info)
        (repo / "one.txt").write_text("one\n")
        after1 = await checkpoint(info)
        before2 = await checkpoint(info)
        (repo / "two.txt").write_text("two\n")
        after2 = await checkpoint(info)

        conflict = await gitwork.revert_turns(info, [
            TurnCheckpoint(before1, after1), TurnCheckpoint(before2, after2),
        ])

        assert conflict is None
        assert not (repo / "one.txt").exists()
        assert not (repo / "two.txt").exists()

    async def test_undo_last_turn_is_a_noop_without_turns(self):
        """No turn since the baseline: /undo must not fall back to reverting
        the baseline and destroying changes made outside the session."""
        repo = self.make_repo()
        info, _ = await setup_tracking(repo, "sess1")
        assert info is not None
        (repo / "app.py").write_text("manual edit\n")
        baseline = info.baseline_tree

        result = await undo_last_turn(info, [])

        assert result is None
        assert (repo / "app.py").read_text() == "manual edit\n"
        assert info.baseline_tree == baseline

    async def test_undo_last_turn_reverses_the_most_recent(self):
        repo = self.make_repo()
        info, _ = await setup_tracking(repo, "sess1")
        assert info is not None
        before = await checkpoint(info)
        (repo / "app.py").write_text("agent\n")
        after = await checkpoint(info)

        result = await undo_last_turn(info, [TurnCheckpoint(before, after)])

        assert result is None
        assert (repo / "app.py").read_text() == "line1\nline2\n"

    # ------------------------------------------------------------------
    # attributed diff

    async def test_attributed_diff_tags_agent_and_drift(self):
        repo = self.make_repo()
        info, _ = await setup_tracking(repo, "sess1")
        assert info is not None

        before = await checkpoint(info)
        (repo / "agent.txt").write_text("agent\n")  # turn edit
        after = await checkpoint(info)
        (repo / "manual.txt").write_text("manual\n")  # drift after the turn

        stat = await attributed_diff_stat(info, [TurnCheckpoint(before, after)])
        by_path = {f.path: f for f in stat.files}
        assert by_path["agent.txt"].source == "agent"
        assert by_path["manual.txt"].source == "drift"
        assert by_path["manual.txt"].is_new

    async def test_attributed_diff_keeps_drift_after_a_later_turn(self):
        """Drift is anchored to baseline_tree, not the last turn: a file made
        outside the session still shows (tagged drift) after a later agent
        turn, even though both sides of that turn's diff contain it."""
        repo = self.make_repo()
        info, _ = await setup_tracking(repo, "sess1")
        assert info is not None

        # Drift created outside the session, never accepted.
        (repo / "drift.txt").write_text("outside\n")
        # A later agent turn: its before_tree already contains the drift, and
        # it edits a different file.
        before = await checkpoint(info)
        (repo / "agent.txt").write_text("agent\n")
        after = await checkpoint(info)

        stat = await attributed_diff_stat(info, [TurnCheckpoint(before, after)])
        by_path = {f.path: f for f in stat.files}
        assert by_path["drift.txt"].source == "drift"
        assert by_path["agent.txt"].source == "agent"

    async def test_attributed_diff_tags_drift_between_turns(self):
        """Drift injected between two turns (not just before the first or
        after the last) is still tagged drift, not absorbed."""
        repo = self.make_repo()
        info, _ = await setup_tracking(repo, "sess1")
        assert info is not None

        before1 = await checkpoint(info)
        (repo / "one.txt").write_text("one\n")
        after1 = await checkpoint(info)
        (repo / "mid.txt").write_text("mid\n")  # drift between the turns
        before2 = await checkpoint(info)
        (repo / "two.txt").write_text("two\n")
        after2 = await checkpoint(info)

        stat = await attributed_diff_stat(info, [
            TurnCheckpoint(before1, after1), TurnCheckpoint(before2, after2),
        ])
        by_path = {f.path: f for f in stat.files}
        assert by_path["mid.txt"].source == "drift"
        assert by_path["one.txt"].source == "agent"
        assert by_path["two.txt"].source == "agent"

    async def test_attributed_diff_turn_touched_wins_within_a_file(self):
        """Attribution is per-file: a file a turn touched stays tagged agent
        even if it was also edited outside the session afterwards."""
        repo = self.make_repo()
        info, _ = await setup_tracking(repo, "sess1")
        assert info is not None

        before = await checkpoint(info)
        (repo / "app.py").write_text("agent\n")
        after = await checkpoint(info)
        (repo / "app.py").write_text("agent\nmanual\n")

        stat = await attributed_diff_stat(info, [TurnCheckpoint(before, after)])
        by_path = {f.path: f for f in stat.files}
        assert by_path["app.py"].source == "agent"

    # ------------------------------------------------------------------
    # diff

    async def test_diff_stat_tracks_edits_and_new_files(self):
        repo = self.make_repo()
        info, _ = await setup_tracking(repo, "sess1")
        assert info is not None

        (repo / "app.py").write_text("line1\nline2\nline3\n")
        (repo / "added.txt").write_text("new\n")
        (repo / "pkg" / "mod.py").unlink()

        stat = await diff_stat(info)
        by_path = {f.path: f for f in stat.files}
        assert set(by_path) == {"app.py", "added.txt", "pkg/mod.py"}
        assert by_path["app.py"].additions == 1
        assert by_path["added.txt"].is_new
        assert by_path["pkg/mod.py"].deletions == 1
        assert stat.additions == 2 and stat.deletions == 1

    async def test_full_and_file_diff(self):
        repo = self.make_repo()
        info, _ = await setup_tracking(repo, "sess1")
        assert info is not None
        (repo / "app.py").write_text("line1\nagent\n")

        full = await git_diff(info)
        assert "app.py" in full and "agent" in full
        one = await git_diff(info, "app.py")
        assert "app.py" in one
        other = await git_diff(info, "pkg/mod.py")
        assert other.strip() == ""

    async def test_diff_detects_renames(self):
        repo = self.make_repo()
        info, _ = await setup_tracking(repo, "sess1")
        assert info is not None
        (repo / "app.py").rename(repo / "renamed.py")

        stat = await diff_stat(info)
        assert {f.path for f in stat.files} == {"renamed.py"}

    # ------------------------------------------------------------------
    # accept (re-baselining)

    async def test_accept_changes_rebaselines(self):
        repo = self.make_repo()
        info, _ = await setup_tracking(repo, "sess1")
        assert info is not None
        (repo / "app.py").write_text("feature\n")
        assert (await diff_stat(info)).files

        sha = await accept_changes(info)

        assert info.baseline_tree == sha
        assert (await diff_stat(info)).files == []  # /diff resets

        # Later work measures against the new baseline only.
        (repo / "app.py").write_text("feature\nmore\n")
        (repo / "second.py").write_text("two\n")
        stat = await diff_stat(info)
        by_path = {f.path: f for f in stat.files}
        assert set(by_path) == {"app.py", "second.py"}
        assert stat.additions == 2

    # ------------------------------------------------------------------
    # session-block round trip

    async def test_git_info_block_round_trip(self):
        repo = self.make_repo()
        info, _ = await setup_tracking(repo, "sess1")
        assert info is not None
        block = info.to_block()
        assert all(isinstance(v, (str, int)) for v in block.values())
        restored = GitInfo.from_block(block)
        assert restored == info

    async def test_from_block_rejects_legacy_worktree_block(self):
        with self.assertRaises(KeyError):
            GitInfo.from_block({
                "session_id": "old",
                "base_ref": "abc",
                "branch": "agent/old",
                "worktree": "/tmp/wt",
                "original_workspace": "/tmp/ws",
                "git_common_dir": "/tmp/ws/.git",
            })

    # ------------------------------------------------------------------
    # run_bash git-command filter (unchanged behavior)

    def guard(self, cmd: str) -> str | None:
        return check_blocked_git(cmd, "main", self.base)

    def test_blocks_branch_switching(self):
        assert self.guard("git checkout other-branch") is not None
        assert self.guard("git switch main") is not None
        assert self.guard("git checkout -b new") is not None

    def test_allows_file_restore_and_readonly_git(self):
        (self.base / "file.txt").write_text("x\n")
        assert self.guard("git checkout -- file.txt") is None
        assert self.guard("git checkout file.txt") is None
        assert self.guard("git status") is None
        assert self.guard("git diff HEAD") is None
        assert self.guard("git log --oneline") is None
        assert self.guard("git add -A && git commit -m msg") is None

    def test_blocks_worktree_and_destructive_ops(self):
        assert self.guard("git worktree add ../wt") is not None
        assert self.guard("git branch -D agent/x") is not None
        assert self.guard("git reset --hard") is not None
        assert self.guard("git rebase main") is not None

    def test_blocks_through_shell_operators_and_env_prefixes(self):
        assert self.guard("echo hi && git reset --hard") is not None
        assert self.guard("FOO=1 git reset --hard") is not None
        assert self.guard("sh -c 'git reset --hard'") is not None

    def test_filter_blocks_even_without_branch_info(self):
        assert check_blocked_git("git reset --hard", None, None) is not None


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
