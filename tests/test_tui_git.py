"""TUI integration tests for git-based change tracking: the diff summary
widget, /accept, /reject, and worktree resume reconnection.

Uses a real temp git repo; session storage and worktree locations are
redirected to temp dirs so nothing touches the user's home.
"""

import asyncio
import subprocess
import tempfile
import unittest
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any
from unittest import mock

from textual.widgets import Static
from xarness import gitwork, session_store
from xarness.conversation import Conversation
from xarness.controller import ChatController
from xarness.gitwork import GitInfo, create_worktree, detect_repo, remove_worktree
from xarness.events import ContentDelta, ToolCallArgumentsDone, ToolCallStarted, TurnComplete
from xarness.tui.app import AgentApp
from xarness.tui.confirm_screen import ConfirmScreen
from xarness.tui.widgets import DiffSummary, ErrorLine, NoticeLine
from test_tui import PROFILE, make_registry


class FakeClient:
    """First stream() call plays ``script``; later calls play queued scripts,
    falling back to a plain final answer so multi-turn tests terminate."""

    def __init__(self, script: list[Any]) -> None:
        self.script = script
        self.next_scripts: list[list[Any]] = []
        self._calls = 0

    async def stream(
        self, wire_messages: Sequence[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[Any]:
        self._calls += 1
        if self._calls == 1:
            events = self.script
        elif self.next_scripts:
            events = self.next_scripts.pop(0)
        else:
            events = [ContentDelta("done"), TurnComplete(usage=None)]
        for event in events:
            await asyncio.sleep(0.01)
            yield event


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"git {' '.join(args)}: {result.stderr}"
    return result.stdout


class GitTUITest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(tempfile.mkdtemp(dir=self._tmp.name))

        # Redirect all persistent state into the temp dir.
        for target, value in (
            (gitwork, "WORKTREES_DIR"), (session_store, "SESSIONS_DIR"),
        ):
            patcher = mock.patch.object(target, value, self.base / value.lower())
            patcher.start()
            self.addCleanup(patcher.stop)

        # A repo with one commit.
        self.repo = self.base / "repo"
        self.repo.mkdir()
        _git(self.repo, "init", "-q", "-b", "main", ".")
        _git(self.repo, "config", "user.email", "t@t")
        _git(self.repo, "config", "user.name", "t")
        (self.repo / "app.py").write_text("line1\nline2\n")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-qm", "init")

    async def make_git_app(self, script: list[Any]) -> tuple[AgentApp, GitInfo]:
        """App bound to a fresh worktree of the test repo."""
        repo_info = await detect_repo(self.repo)
        assert repo_info is not None
        info = await create_worktree(repo_info, "testsess")
        client = FakeClient(script)
        from xarness.controller import ChatController
        controller = ChatController(PROFILE, "k", client=client)
        app = AgentApp(
            PROFILE, "k", controller=controller, tool_registry=make_registry(),
            workspace=info.worktree, session_name="testsess", git_info=info,
        )
        return app, info

    async def wait_until(self, predicate, timeout: float = 5.0) -> None:
        for _ in range(int(timeout / 0.05)):
            await asyncio.sleep(0.05)
            if predicate():
                return
        raise AssertionError("condition not met in time")

    async def run_simple_turn(self, app: AgentApp, pilot) -> None:
        await pilot.press("h", "i", "enter")
        await self.wait_until(lambda: not app._turn_busy)

    # ------------------------------------------------------------------
    # diff summary

    async def test_diff_summary_hidden_without_git_session(self):
        client = FakeClient([ContentDelta("hi"), TurnComplete(usage=None)])
        controller = ChatController(PROFILE, "k", client=client)
        app = AgentApp(PROFILE, "k", controller=controller)
        async with app.run_test() as pilot:
            await pilot.press("h", "i", "enter")
            await self.wait_until(lambda: not app._turn_busy)
            summary = app.query_one("#diff-summary", DiffSummary)
            self.assertFalse(summary.has_class("visible"))

    async def test_diff_summary_appears_after_tool_turn(self):
        script = [
            ToolCallStarted("c1", "noop"),
            ToolCallArgumentsDone("c1", "noop", "{}"),
            TurnComplete(has_tool_calls=True),
        ]
        app, info = await self.make_git_app(script)
        client = app.controller._client
        # Queue: turn-1 final answer, then turn-2's tool round + final answer
        # (a turn only recomputes the summary when it ran tools).
        client.next_scripts = [
            [ContentDelta("done"), TurnComplete(usage=None)],
            [
                ToolCallStarted("c2", "noop"),
                ToolCallArgumentsDone("c2", "noop", "{}"),
                TurnComplete(has_tool_calls=True),
            ],
            [ContentDelta("done"), TurnComplete(usage=None)],
        ]
        async with app.run_test() as pilot:
            summary = app.query_one("#diff-summary", DiffSummary)
            # Hidden before any edits.
            self.assertFalse(summary.has_class("visible"))

            # A tool turn with no file changes: recomputed, still hidden.
            await pilot.press("g", "o", "enter")
            await self.wait_until(lambda: not app._turn_busy)
            self.assertFalse(summary.has_class("visible"))

            # The agent edits a file in the worktree; next tool turn the
            # summary must appear with correct counts.
            (info.worktree / "app.py").write_text("line1\nline2\nline3\nline4\n")
            await pilot.press("a", "g", "a", "i", "n", "enter")
            await self.wait_until(lambda: not app._turn_busy)

            self.assertTrue(summary.has_class("visible"))
            text = str(app.query_one("#diff-summary-text").content)
            self.assertIn("Edited 1 file", text)
            self.assertIn("+2", str(app.query_one("#diff-summary-add").content))
            self.assertIn("-0", str(app.query_one("#diff-summary-del").content))

            # Expand: per-file rows (name, dir, stats) relative to the root.
            summary.toggle()
            rows = app.query(".diff-file-row")
            self.assertEqual(len(rows), 1)
            self.assertIn("app.py", str(rows.first().query_one(".diff-file-name").content))
            add_text = str(rows.first().query_one(".diff-file-add").content)
            del_text = str(rows.first().query_one(".diff-file-del").content)
            self.assertIn("+2", add_text)
            self.assertIn("-0", del_text)

    async def test_diff_content_renders_requested_diff(self):
        script = [
            ToolCallStarted("c1", "noop"),
            ToolCallArgumentsDone("c1", "noop", "{}"),
            TurnComplete(has_tool_calls=True),
        ]
        app, info = await self.make_git_app(script)
        (info.worktree / "app.py").write_text("edited\n")
        async with app.run_test() as pilot:
            await pilot.press("g", "o", "enter")
            await self.wait_until(lambda: not app._turn_busy)
            summary = app.query_one("#diff-summary", DiffSummary)
            self.assertTrue(summary.has_class("visible"))

            # A per-file [diff] request renders that file's unified diff.
            app.post_message(DiffSummary.DiffRequested("app.py"))
            await self.wait_until(lambda: summary.active_diff_key == "app.py")
            content = app.query_one("#diff-content")
            self.assertTrue(content.has_class("show"))
            self.assertIn("app.py", str(app.query_one("#diff-text", Static).content))

            # Requesting the same key again collapses the diff.
            app.post_message(DiffSummary.DiffRequested("app.py"))
            await self.wait_until(lambda: summary.active_diff_key is not None and summary.active_diff_key != "app.py")
            self.assertFalse(content.has_class("show"))

    async def test_slash_diff_toggles_panel_and_per_file_diff(self):
        app, info = await self.make_git_app([ContentDelta("hi"), TurnComplete(usage=None)])
        (info.worktree / "app.py").write_text("agent edit\n")
        (info.worktree / "new.py").write_text("brand new\n")
        async with app.run_test() as pilot:
            await pilot.press("h", "i", "enter")
            await self.wait_until(lambda: not app._turn_busy)
            summary = app.query_one("#diff-summary", DiffSummary)

            # /diff: expands the per-file list (no unified diff open).
            app._handle_slash_command("/diff")
            await self.wait_until(lambda: summary.has_class("expanded"))
            self.assertTrue(summary.has_class("visible"))
            self.assertFalse(summary.diff_shown)
            self.assertEqual(len(app.query(".diff-file-row")), 2)
            self.assertFalse(app.query_one("#diff-content").has_class("show"))

            # /diff again: hides the whole panel.
            app._handle_slash_command("/diff")
            await self.wait_until(lambda: not summary.has_class("visible"))
            self.assertFalse(summary.has_class("expanded"))

            # /diff <path>: panel + that file's unified diff, row highlighted.
            app._handle_slash_command("/diff app.py")
            await self.wait_until(lambda: summary.diff_shown and summary.active_diff_key == "app.py")
            self.assertTrue(summary.has_class("visible"))
            rendered = str(app.query_one("#diff-text", Static).content)
            self.assertIn("app.py", rendered)
            self.assertNotIn("new.py", rendered)
            active = app.query(".diff-file-row.active")
            self.assertEqual(len(active), 1)
            self.assertEqual(active.first().diff_key, "app.py")

    async def test_escape_minimizes_diff_then_collapses_list(self):
        """esc #1 minimizes an open diff (list stays up, row stays highlighted,
        focus moves to the list); arrows + enter navigate/open from there;
        esc #2 collapses the list back to the bare summary line."""
        app, info = await self.make_git_app([ContentDelta("hi"), TurnComplete(usage=None)])
        (info.worktree / "app.py").write_text("agent edit\n")
        (info.worktree / "new.py").write_text("brand new\n")
        async with app.run_test() as pilot:
            await pilot.press("h", "i", "enter")
            await self.wait_until(lambda: not app._turn_busy)
            summary = app.query_one("#diff-summary", DiffSummary)
            content = app.query_one("#diff-content")

            # Open one file's diff; the pane takes focus for arrow-key paging.
            app._handle_slash_command("/diff app.py")
            await self.wait_until(lambda: summary.diff_shown)
            await self.wait_until(lambda: app.focused is content)

            # First escape: minimize the diff, keep the list + highlight, and
            # hand focus to the summary for keyboard navigation.
            await pilot.press("escape")
            await self.wait_until(lambda: not summary.diff_shown)
            self.assertTrue(summary.has_class("expanded"))
            self.assertTrue(summary.has_class("visible"))
            self.assertFalse(content.has_class("show"))
            await self.wait_until(lambda: app.focused is summary)
            self.assertEqual(len(app.query(".diff-file-row.active")), 1)

            # Down moves the highlight; enter opens the highlighted diff and
            # refocuses the pane (so arrows page the diff again).
            await pilot.press("down")
            active = app.query(".diff-file-row.active").first()
            self.assertEqual(active.diff_key, "new.py")
            await pilot.press("enter")
            await self.wait_until(
                lambda: summary.diff_shown and summary.active_diff_key == "new.py"
            )
            await self.wait_until(lambda: app.focused is content)

            # Escape minimizes again; a second escape collapses the list —
            # only the summary line remains and focus returns to the input.
            await pilot.press("escape")
            await self.wait_until(lambda: not summary.diff_shown)
            await pilot.press("escape")
            await self.wait_until(lambda: not summary.has_class("expanded"))
            self.assertTrue(summary.has_class("visible"))
            self.assertFalse(content.has_class("show"))
            await self.wait_until(lambda: app.focused is app.query_one("#chat-input"))

    async def test_slash_diff_without_changes_or_git(self):
        app, info = await self.make_git_app([ContentDelta("hi"), TurnComplete(usage=None)])
        async with app.run_test() as pilot:
            await pilot.press("h", "i", "enter")
            await self.wait_until(lambda: not app._turn_busy)

            # No pending changes: a notice, no diff view.
            app._handle_slash_command("/diff")
            await self.wait_until(
                lambda: any("no pending changes" in str(n.content) for n in app.query(NoticeLine))
            )
            self.assertFalse(app.query_one("#diff-summary", DiffSummary).has_class("expanded"))

        # No worktree at all: an error line.
        client = FakeClient([ContentDelta("x"), TurnComplete(usage=None)])
        from xarness.controller import ChatController
        bare = AgentApp(
            PROFILE, "k", controller=ChatController(PROFILE, "k", client=client),
        )
        async with bare.run_test() as pilot:
            bare._handle_slash_command("/diff")
            await self.wait_until(
                lambda: any("no pending-changes tracking" in str(e.content) for e in bare.query(ErrorLine))
            )

    # ------------------------------------------------------------------
    # accept / reject

    async def test_reject_removes_worktree_after_confirmation(self):
        app, info = await self.make_git_app([ContentDelta("hi"), TurnComplete(usage=None)])
        (info.worktree / "app.py").write_text("agent edit\n")
        async with app.run_test() as pilot:
            await pilot.press("h", "i", "enter")
            await self.wait_until(lambda: not app._turn_busy)

            app._handle_slash_command("/reject")
            # Confirmation modal appears; wrong word does nothing.
            await self.wait_until(lambda: isinstance(app.screen, ConfirmScreen))
            await pilot.press("n", "o", "enter")
            await self.wait_until(lambda: not isinstance(app.screen, ConfirmScreen))
            self.assertTrue(info.worktree.exists())  # not confirmed: no-op

            # Type the confirm word: worktree and branch are destroyed.
            app._handle_slash_command("/reject")
            await self.wait_until(lambda: isinstance(app.screen, ConfirmScreen))
            await pilot.press("r", "e", "j", "e", "c", "t", "enter")
            await self.wait_until(lambda: app.git_info is None)

            self.assertFalse(info.worktree.exists())
            self.assertNotIn("agent/testsess", _git(self.repo, "branch"))
            # No agent/* branch of any kind survives a confirmed reject.
            self.assertFalse(any(
                b.strip().lstrip("* ").startswith("agent/")
                for b in _git(self.repo, "branch").splitlines()
            ))
            # User's directory untouched.
            self.assertEqual((self.repo / "app.py").read_text(), "line1\nline2\n")
            # Session JSON no longer carries a git block.
            self.assertIsNone(session_store.load_git_block("testsess"))

    async def test_accept_merges_and_cleans_up(self):
        app, info = await self.make_git_app([ContentDelta("hi"), TurnComplete(usage=None)])
        (info.worktree / "app.py").write_text("agent edit\n")
        async with app.run_test() as pilot:
            await pilot.press("h", "i", "enter")
            await self.wait_until(lambda: not app._turn_busy)

            app._handle_slash_command("/accept")
            await self.wait_until(
                lambda: any("accepted" in str(n.content) for n in app.query(NoticeLine))
            )

            # Changes landed as a merge commit in the user's directory.
            self.assertEqual((self.repo / "app.py").read_text(), "agent edit\n")
            self.assertIn("Merge branch", _git(self.repo, "log", "-1", "--format=%s"))
            self.assertFalse(info.worktree.exists())
            self.assertIsNone(app.git_info)
            self.assertIsNone(session_store.load_git_block("testsess"))

    async def test_accept_conflict_reports_and_preserves_worktree(self):
        app, info = await self.make_git_app([ContentDelta("hi"), TurnComplete(usage=None)])
        (info.worktree / "app.py").write_text("agent edit\n")
        async with app.run_test() as pilot:
            await pilot.press("h", "i", "enter")
            await self.wait_until(lambda: not app._turn_busy)

            # The user's directory moves on with a conflicting change.
            (self.repo / "app.py").write_text("user conflicting edit\n")
            _git(self.repo, "commit", "-aqm", "user change")

            app._handle_slash_command("/accept")
            await self.wait_until(
                lambda: any("conflict" in str(e.content) for e in app.query(ErrorLine))
            )

            # Conflict reported, worktree kept so the user can retry after
            # manually reconciling; git state still persisted.
            self.assertTrue(info.worktree.exists())
            self.assertIsNotNone(app.git_info)
            notices = [str(n.content) for n in app.query(NoticeLine)]
            self.assertFalse(any("accepted" in n for n in notices))

    # ------------------------------------------------------------------
    # resume reconnection

    async def test_resume_reconnects_valid_worktree(self):
        info = await create_worktree(await detect_repo(self.repo), "testsess")
        (info.worktree / "app.py").write_text("pending agent edit\n")
        session_store.save_session("testsess", "m", Conversation(), git=info.to_block())

        client = FakeClient([ContentDelta("x"), TurnComplete(usage=None)])
        from xarness.controller import ChatController
        app = AgentApp(
            PROFILE, "k", controller=ChatController(PROFILE, "k", client=client),
            session_name="testsess",
        )
        async with app.run_test() as pilot:
            # Simulate /resume picking the same session.
            await app._on_session_selected("testsess")
            await pilot.pause()
            await self.wait_until(lambda: app.git_info is not None)

            self.assertEqual(app.git_info.worktree, info.worktree)
            self.assertEqual(app.sandbox.workspace, info.worktree)
            summary = app.query_one("#diff-summary", DiffSummary)
            self.assertTrue(summary.has_class("visible"))
            text = str(app.query_one("#diff-summary-text").content)
            self.assertIn("Edited 1 file", text)

    async def test_resume_recreates_removed_worktree(self):
        info = await create_worktree(await detect_repo(self.repo), "testsess")
        session_store.save_session("testsess", "m", Conversation(), git=info.to_block())
        # Externally removed (git worktree prune + folder deletion).
        await remove_worktree(self.repo, info.worktree, force=True)

        client = FakeClient([ContentDelta("x"), TurnComplete(usage=None)])
        app = AgentApp(
            PROFILE, "k", controller=ChatController(PROFILE, "k", client=client),
            session_name="testsess",
        )
        async with app.run_test() as pilot:
            await app._on_session_selected("testsess")
            await pilot.pause()
            await self.wait_until(lambda: app.git_info is not None)

            self.assertTrue(app.git_info.worktree.exists())
            await self.wait_until(
                lambda: any("recreated" in str(n.content) for n in app.query(NoticeLine))
            )


if __name__ == "__main__":
    unittest.main()
