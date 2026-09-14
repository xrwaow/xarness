"""Tests for gitwork: worktree lifecycle, diff computation, merge/reject,
and the run_bash git-command filter.

Uses a real temp git repo (git is a hard dependency of the feature anyway);
monkeypatches gitwork.WORKTREES_DIR so nothing touches ~/.local/share.
"""

import asyncio
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from xarness import gitwork
from xarness.gitwork import (
    MergeOutcome, check_blocked_git, commit_worktree_changes, create_worktree,
    delete_branch, detect_repo, diff_stat, file_diff, full_diff, merge_branch,
    recreate_worktree, remove_worktree, worktree_is_valid,
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
        self._orig_dir = gitwork.WORKTREES_DIR
        gitwork.WORKTREES_DIR = self.base / "worktrees"
        self.addCleanup(setattr, gitwork, "WORKTREES_DIR", self._orig_dir)

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
    # detection + worktree creation

    async def test_detect_repo_none_for_plain_directory(self):
        assert await detect_repo(self.base) is None

    async def test_detect_repo_reads_head(self):
        repo = self.make_repo()
        info = await detect_repo(repo)
        assert info is not None
        assert info.root == repo
        assert info.head_branch == "main"
        sha = _git(repo, "rev-parse", "HEAD").strip()
        assert info.base_ref == sha
        assert info.git_common_dir == repo / ".git"

    async def test_detect_repo_unborn_head_has_no_base_ref(self):
        repo = self.base / "empty"
        repo.mkdir()
        _git(repo, "init", "-q", "-b", "main", ".")
        info = await detect_repo(repo)
        assert info is not None
        assert info.root == repo
        assert info.base_ref is None

    async def test_setup_isolation_auto_initializes_non_repo(self):
        """No git init from the user: the harness sets everything up itself."""
        workspace = self.base / "plain"
        workspace.mkdir()
        (workspace / "notes.txt").write_text("user's file\n")

        info, notes = await gitwork.setup_isolation(workspace, "sess1")

        assert info is not None
        assert info.worktree.is_dir()
        # The user's file was snapshotted as the baseline and is intact.
        assert (workspace / "notes.txt").read_text() == "user's file\n"
        assert (info.worktree / "notes.txt").read_text() == "user's file\n"
        assert any("initialized" in n for n in notes)
        # The repo is marked as harness-managed.
        managed = _git(workspace, "config", "xarness.managed").strip()
        assert managed == "true"

        # Agent edit shows up in the diff against the baseline...
        (info.worktree / "notes.txt").write_text("user's file\nagent edit\n")
        stat = await diff_stat(info.worktree, info.base_ref)
        assert stat.additions == 1

        # ...and accept lands it in the user's (auto-initialized) repo.
        assert await commit_worktree_changes(info) == "committed"
        result = await merge_branch(workspace, info.branch)
        assert result.outcome is MergeOutcome.MERGED
        assert (workspace / "notes.txt").read_text() == "user's file\nagent edit\n"

    async def test_setup_isolation_baselines_unborn_user_repo(self):
        """User ran `git init` themselves but has no commits: their repo is
        used as-is (plus a baseline snapshot), never replaced."""
        workspace = self.base / "unborn"
        workspace.mkdir()
        _git(workspace, "init", "-q", "-b", "main", ".")
        (workspace / "file.py").write_text("x = 1\n")

        info, notes = await gitwork.setup_isolation(workspace, "sess1")

        assert info is not None
        assert any("no commits" in n for n in notes)
        assert await detect_repo(workspace) is not None
        # One commit (the baseline), authored into THEIR repo — files intact.
        assert len(_git(workspace, "log", "--oneline").splitlines()) == 1
        assert (workspace / "file.py").read_text() == "x = 1\n"

    async def test_setup_isolation_respects_allow_init_false(self):
        workspace = self.base / "plain"
        workspace.mkdir()
        info, notes = await gitwork.setup_isolation(workspace, "sess1", allow_init=False)
        assert info is None
        assert not (workspace / ".git").exists()

    async def test_create_worktree_isolates_user_state(self):
        repo = self.make_repo()
        # User has uncommitted changes BEFORE the session starts.
        (repo / "app.py").write_text("user edit\n")
        (repo / "scratch.txt").write_text("untracked\n")
        before_status = _git(repo, "status", "--porcelain")

        info = await create_worktree(await detect_repo(repo), "sess1")

        # Worktree exists outside the repo, on its own branch at base_ref.
        assert info.worktree.is_dir()
        assert info.worktree.parent == gitwork.WORKTREES_DIR
        assert info.branch == "agent/sess1"
        assert (info.worktree / "app.py").read_text() == "line1\nline2\n"
        assert _git(info.worktree, "rev-parse", "--abbrev-ref", "HEAD").strip() == "agent/sess1"

        # The user's directory is byte-for-byte untouched.
        assert _git(repo, "status", "--porcelain") == before_status
        assert (repo / "app.py").read_text() == "user edit\n"
        assert (repo / "scratch.txt").read_text() == "untracked\n"
        assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"

    async def test_create_worktree_copies_untracked_files(self):
        repo = self.make_repo()
        (repo / "uncommitted.cfg").write_text("key=value\n")
        (repo / "sub").mkdir()
        (repo / "sub" / "nested.txt").write_text("nested\n")
        info = await create_worktree(await detect_repo(repo), "sess1")
        assert (info.worktree / "uncommitted.cfg").read_text() == "key=value\n"
        assert (info.worktree / "sub" / "nested.txt").read_text() == "nested\n"
        assert "uncommitted.cfg" in info.copied_untracked

    async def test_create_worktree_can_skip_untracked_copy(self):
        repo = self.make_repo()
        (repo / "uncommitted.cfg").write_text("key=value\n")
        info = await create_worktree(await detect_repo(repo), "sess1", copy_untracked=False)
        assert not (info.worktree / "uncommitted.cfg").exists()
        assert info.copied_untracked == []

    async def test_worktree_is_valid_and_recreate(self):
        repo = self.make_repo()
        info = await create_worktree(await detect_repo(repo), "sess1")
        assert await worktree_is_valid(info)

        # Simulate external removal (git worktree prune + delete the folder).
        await remove_worktree(repo, info.worktree, force=True)
        assert not await worktree_is_valid(info)

        fresh = await recreate_worktree(info)
        assert await worktree_is_valid(fresh)
        assert fresh.worktree == info.worktree
        assert fresh.branch == info.branch

    # ------------------------------------------------------------------
    # diff computation

    async def test_diff_stat_tracks_edits_and_new_files(self):
        repo = self.make_repo()
        info = await create_worktree(await detect_repo(repo), "sess1")
        wt = info.worktree

        # Nothing changed yet.
        stat = await diff_stat(wt, info.base_ref)
        assert stat.files == []

        # Edit a tracked file (+1 -1) and create a new one (+2 -0).
        (wt / "app.py").write_text("line1\nchanged\n")
        (wt / "new.py").write_text("x = 1\ny = 2\n")

        stat = await diff_stat(wt, info.base_ref)
        by_path = {f.path: f for f in stat.files}
        assert by_path["app.py"].additions == 1
        assert by_path["app.py"].deletions == 1
        assert by_path["new.py"].additions == 2
        assert by_path["new.py"].deletions == 0
        assert by_path["new.py"].is_new is True
        assert by_path["app.py"].is_new is False
        assert stat.additions == 3
        assert stat.deletions == 1

    async def test_full_and_file_diff(self):
        repo = self.make_repo()
        info = await create_worktree(await detect_repo(repo), "sess1")
        (info.worktree / "app.py").write_text("edited\n")
        (info.worktree / "other.py").write_text("new\n")

        full = await full_diff(info.worktree, info.base_ref)
        assert "diff --git a/app.py" in full
        assert "diff --git a/other.py" in full

        single = await file_diff(info.worktree, info.base_ref, "app.py")
        assert "diff --git a/app.py" in single
        assert "other.py" not in single

    # ------------------------------------------------------------------
    # accept / reject

    async def test_accept_merges_into_user_branch_no_ff(self):
        repo = self.make_repo()
        info = await create_worktree(await detect_repo(repo), "sess1")
        (info.worktree / "app.py").write_text("agent edit\n")
        assert await commit_worktree_changes(info) == "committed"

        result = await merge_branch(repo, info.branch)
        assert result.outcome is MergeOutcome.MERGED
        assert (repo / "app.py").read_text() == "agent edit\n"
        # --no-ff: a distinct merge commit, not a fast-forward.
        assert len(_git(repo, "log", "--oneline").splitlines()) == 3
        assert "Merge branch" in _git(repo, "log", "-1", "--format=%s")

        await remove_worktree(repo, info.worktree)
        assert not info.worktree.exists()

    async def test_accept_reports_conflict_and_keeps_worktree(self):
        repo = self.make_repo()
        info = await create_worktree(await detect_repo(repo), "sess1")
        (info.worktree / "app.py").write_text("agent edit\n")
        assert await commit_worktree_changes(info) == "committed"

        # The user's directory moves on with a conflicting change.
        (repo / "app.py").write_text("user conflicting edit\n")
        _git(repo, "commit", "-aqm", "user change")

        result = await merge_branch(repo, info.branch)
        assert result.outcome is MergeOutcome.CONFLICT
        # The merge is left in place for manual resolution...
        assert (repo / ".git" / "MERGE_HEAD").exists()
        # ...and the worktree is NOT deleted, so the user can retry.
        assert await worktree_is_valid(info)

    async def test_accept_in_progress_merge_is_reported(self):
        repo = self.make_repo()
        info = await create_worktree(await detect_repo(repo), "sess1")
        (info.worktree / "app.py").write_text("agent edit\n")
        await commit_worktree_changes(info)

        # Simulate an unresolved merge sitting in the user's repo: a
        # conflicting merge that was never resolved or aborted.
        (repo / "app.py").write_text("user conflicting edit\n")
        _git(repo, "commit", "-aqm", "user change")
        _git(repo, "merge", "--no-commit", info.branch, check=False)

        result = await merge_branch(repo, info.branch)
        assert result.outcome is MergeOutcome.IN_PROGRESS

    async def test_commit_skips_untouched_copied_untracked(self):
        repo = self.make_repo()
        (repo / "notes.txt").write_text("user notes\n")
        info = await create_worktree(await detect_repo(repo), "sess1")
        # Agent edits a tracked file but leaves the copied untracked file alone.
        (info.worktree / "app.py").write_text("agent edit\n")

        assert await commit_worktree_changes(info) == "committed"
        # The agent's edit is committed...
        names = _git(info.worktree, "show", "--name-only", "--format=").split()
        assert "app.py" in names
        # ...but the untouched copied file is not.
        assert "notes.txt" not in names

    async def test_commit_clean_worktree(self):
        repo = self.make_repo()
        info = await create_worktree(await detect_repo(repo), "sess1")
        assert await commit_worktree_changes(info) == "clean"

    async def test_reject_removes_worktree_and_branch(self):
        repo = self.make_repo()
        info = await create_worktree(await detect_repo(repo), "sess1")
        (info.worktree / "app.py").write_text("agent edit\n")  # uncommitted

        await remove_worktree(repo, info.worktree, force=True)
        await delete_branch(repo, info.branch)

        assert not info.worktree.exists()
        assert "agent/sess1" not in _git(repo, "branch")
        assert "agent/sess1" not in _git(repo, "worktree", "list")
        # User's directory completely unaffected.
        assert (repo / "app.py").read_text() == "line1\nline2\n"
        assert _git(repo, "status", "--porcelain") == ""

    # ------------------------------------------------------------------
    # shell-command filter (Part 2)

    def guard(self, command, branch="agent/sess1", root=None):
        return check_blocked_git(command, branch, root)

    def test_blocks_branch_switching(self):
        assert self.guard("git checkout main")
        assert self.guard("git checkout other-branch")
        assert self.guard("git checkout -b new-branch")
        assert self.guard("git switch main")
        assert self.guard("git switch -c feature")

    def test_allows_file_restore_and_readonly_git(self):
        assert self.guard("git checkout -- some/file.py") is None
        # A bare word that names an existing path is a file restore, not a
        # ref switch — don't block it even though it looks like a branch name.
        existing = self.base / "some-branch-looking-file"
        existing.write_text("x")
        assert self.guard("git checkout some-branch-looking-file", root=self.base) is None
        assert self.guard("git checkout agent/sess1") is None  # own branch
        assert self.guard("git status") is None
        assert self.guard("git diff") is None
        assert self.guard("git log --oneline") is None
        assert self.guard("git add -A && git commit -m x") is None
        assert self.guard("git show HEAD") is None
        assert self.guard("git blame app.py") is None

    def test_blocks_worktree_and_destructive_ops(self):
        assert self.guard("git worktree add /tmp/x main")
        assert self.guard("git worktree remove /tmp/x")
        assert self.guard("git worktree prune")
        assert self.guard("git branch -d agent/sess1")
        assert self.guard("git branch -D agent/sess1")
        assert self.guard("git branch --delete agent/sess1")
        assert self.guard("git reset --hard HEAD~1")
        assert self.guard("git rebase main")

    def test_no_false_positive_on_file_named_checkout(self):
        # A file literally named checkout.txt must not trip the filter.
        assert self.guard("cat checkout.txt") is None
        assert self.guard("git log -- checkout.txt") is None

    def test_blocks_through_shell_operators_and_env_prefixes(self):
        assert self.guard("git status && git checkout main")
        assert self.guard("echo hi; git worktree prune")
        assert self.guard("GIT_DIR=/tmp git checkout main || true")
        assert self.guard("sh -c 'git checkout main'")
        assert self.guard("sudo git rebase main")

    def test_filter_blocks_even_without_branch_info(self):
        # check_blocked_git itself still blocks identity changes when the
        # branch name is unknown; the app-level guard is what short-circuits
        # to None when the session has no worktree at all.
        assert check_blocked_git("git checkout main", None, None) is not None


if __name__ == "__main__":
    unittest.main()
