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
    delete_branch, detect_repo, diff_stat, git_diff, merge_branch,
    recreate_worktree, remove_worktree, stage_untracked, worktree_is_valid,
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

    async def test_detect_repo_reads_head(self):
        assert await detect_repo(self.base) is None  # plain directory
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

    async def test_setup_isolation_scopes_to_workspace_subdir(self):
        """--workspace pointing at a subdirectory of a bigger repo: the
        session is scoped to that subtree — untracked copy, diff, and agent
        workspace cover only it, never the rest of the repo."""
        repo = self.make_repo()
        (repo / "f").mkdir()
        (repo / "f" / "notes.txt").write_text("tracked\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "add f")
        (repo / "f" / "draft.txt").write_text("untracked\n")  # under f
        (repo / "stray.txt").write_text("untracked\n")  # outside f

        info, notes = await gitwork.setup_isolation(repo / "f", "sess1")

        assert info is not None
        assert info.subtree == "f"
        assert info.agent_workspace == info.worktree / "f"
        # The subtree exists in the worktree even though git doesn't check
        # out untracked directories.
        assert info.agent_workspace.is_dir()
        assert any("scoped to f/" in n for n in notes)
        # Only the subtree's untracked file was copied.
        assert (info.worktree / "f" / "draft.txt").read_text() == "untracked\n"
        assert not (info.worktree / "stray.txt").exists()
        assert info.copied_untracked == ["f/draft.txt"]

        # Diff covers the subtree only, with subtree-relative paths — even
        # when the rest of the worktree somehow changes too. The copied
        # untracked draft shows up as a new file (intent-to-add), same as in
        # the whole-repo case.
        (info.worktree / "f" / "notes.txt").write_text("tracked\nagent edit\n")
        (info.worktree / "app.py").write_text("outside the subtree\n")
        stat = await diff_stat(info.worktree, info.base_ref, info.subtree)
        by_path = {f.path: f for f in stat.files}
        assert set(by_path) == {"notes.txt", "draft.txt"}
        assert by_path["notes.txt"].additions == 1
        assert by_path["notes.txt"].is_new is False
        assert by_path["draft.txt"].is_new is True
        full = await git_diff(info.worktree, info.base_ref, subtree=info.subtree)
        assert "diff --git a/notes.txt" in full
        assert "app.py" not in full
        single = await git_diff(
            info.worktree, info.base_ref, "notes.txt", subtree=info.subtree
        )
        assert "diff --git a/notes.txt" in single

        # A recreated worktree keeps the subtree present and scoped.
        fresh = await recreate_worktree(info)
        assert fresh.subtree == "f"
        assert (fresh.worktree / "f").is_dir()

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

        # copy_untracked=False skips them entirely.
        info = await create_worktree(await detect_repo(repo), "sess2", copy_untracked=False)
        assert not (info.worktree / "uncommitted.cfg").exists()
        assert info.copied_untracked == []

    async def test_worktree_is_valid_and_recreate(self):
        repo = self.make_repo()
        info = await create_worktree(await detect_repo(repo), "sess1")
        assert await worktree_is_valid(info)
        # Nothing changed yet: nothing to commit.
        assert await commit_worktree_changes(info) == "clean"

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

        full = await git_diff(info.worktree, info.base_ref)
        assert "diff --git a/app.py" in full
        assert "diff --git a/other.py" in full

        single = await git_diff(info.worktree, info.base_ref, "app.py")
        assert "diff --git a/app.py" in single
        assert "other.py" not in single

    async def test_diff_detects_renames(self):
        """-M is passed explicitly: a moved file shows up as a rename (one
        numstat entry for the new path, "rename from/to" in the full diff),
        not a full delete+add pair, regardless of diff.renames config."""
        repo = self.make_repo()
        info = await create_worktree(await detect_repo(repo), "sess1")
        wt = info.worktree

        # Pure move (100% similarity) + move-and-edit (2 of 3 lines kept).
        (wt / "moved.py").write_text((wt / "pkg" / "mod.py").read_text())
        (wt / "pkg" / "mod.py").unlink()
        (wt / "app_renamed.py").write_text("line1\nline2\nline3\n")
        (wt / "app.py").unlink()

        stat = await diff_stat(wt, info.base_ref)
        paths = {f.path for f in stat.files}
        assert "pkg/mod.py" not in paths
        assert "app.py" not in paths
        assert "moved.py" in paths
        assert "app_renamed.py" in paths
        by_path = {f.path: f for f in stat.files}
        # Rename-with-edit: numstat counts only the edit's lines.
        assert by_path["app_renamed.py"].additions == 1
        assert by_path["app_renamed.py"].deletions == 0

        full = await git_diff(wt, info.base_ref)
        assert "rename from pkg/mod.py" in full
        assert "rename to moved.py" in full

    async def test_stage_untracked_is_idempotent(self):
        """Re-staging between edits never corrupts the diff: it must always
        reflect current worktree content, and extra calls are no-ops."""
        repo = self.make_repo()
        info = await create_worktree(await detect_repo(repo), "sess1")
        wt = info.worktree

        (wt / "new.py").write_text("v1\n")
        await stage_untracked(wt)
        (wt / "new.py").write_text("v2 edited\n")
        await stage_untracked(wt)
        await stage_untracked(wt)
        (wt / "app.py").write_text("edited\n")

        stat = await diff_stat(wt, info.base_ref)
        by_path = {f.path: f for f in stat.files}
        assert by_path["new.py"].additions == 1
        assert by_path["new.py"].deletions == 0
        assert by_path["new.py"].is_new is True
        assert by_path["app.py"].additions == 1
        assert by_path["app.py"].deletions == 2
        assert stat.additions == 2
        assert stat.deletions == 2

        # A second diff_stat (which re-stages) reports the same thing.
        again = await diff_stat(wt, info.base_ref)
        assert again.additions == stat.additions
        assert again.deletions == stat.deletions

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

    async def test_create_worktree_after_accept_picks_free_branch(self):
        """Resuming a session whose changes were accepted: accept dropped the
        git block but kept the branch, so a fresh worktree for the same
        session id must derive a new branch instead of failing on -b."""
        repo = self.make_repo()
        info = await create_worktree(await detect_repo(repo), "sess1")
        (info.worktree / "app.py").write_text("agent edit\n")
        assert await commit_worktree_changes(info) == "committed"
        assert (await merge_branch(repo, info.branch)).outcome is MergeOutcome.MERGED

        # The resume path (cli._fresh_isolation / app._on_session_selected)
        # calls setup_isolation → create_worktree with the same session id.
        fresh = await create_worktree(await detect_repo(repo), "sess1")
        assert fresh.branch == "agent/sess1-2"
        assert fresh.worktree.is_dir()
        # Worktree is based on the user's current HEAD, i.e. includes the
        # previously accepted change.
        assert (fresh.worktree / "app.py").read_text() == "agent edit\n"
        assert "agent/sess1\n" in _git(repo, "branch")  # kept history intact

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

        # A clean worktree (nothing to commit) is reported as such.
        info = await create_worktree(await detect_repo(repo), "sess2")
        assert await commit_worktree_changes(info) == "nothing-staged"

    async def test_reject_removes_worktree_and_branch(self):
        repo = self.make_repo()
        info = await create_worktree(await detect_repo(repo), "sess1")
        (info.worktree / "app.py").write_text("agent edit\n")  # uncommitted

        await remove_worktree(repo, info.worktree, force=True)
        await delete_branch(repo, info.branch)

        assert not info.worktree.exists()
        # No agent/* branch of any kind survives a reject.
        branches = [b.strip().lstrip("* ") for b in _git(repo, "branch").splitlines()]
        assert not any(b.startswith("agent/") for b in branches)
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
        # A file literally named checkout.txt must not trip the filter.
        assert self.guard("cat checkout.txt") is None
        assert self.guard("git log -- checkout.txt") is None

    def test_blocks_worktree_and_destructive_ops(self):
        assert self.guard("git worktree add /tmp/x main")
        assert self.guard("git worktree remove /tmp/x")
        assert self.guard("git worktree prune")
        assert self.guard("git branch -d agent/sess1")
        assert self.guard("git branch -D agent/sess1")
        assert self.guard("git branch --delete agent/sess1")
        assert self.guard("git reset --hard HEAD~1")
        assert self.guard("git rebase main")

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
