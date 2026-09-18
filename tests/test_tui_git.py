"""TUI integration tests for git-based change tracking: the diff summary
widget, /undo //retry file reverts, and resume reconnection.

Uses a real temp git repo; session storage is redirected to a temp dir so
nothing touches the user's home. The agent edits the repo directly — there
is no worktree.
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
from xarness import session_store
from xarness.conversation import Conversation
from xarness.controller import ChatController
from xarness.gitwork import GitInfo, diff_stat, setup_tracking
from xarness.events import ContentDelta, ToolCallArgumentsDone, ToolCallStarted, TurnComplete
from xarness.tui.app import AgentApp
from xarness.tui.widgets import DiffSummary, ErrorLine, NoticeLine
from xarness.tui.widgets import ChatInput, UserMessage
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

        # Redirect persistent session storage into the temp dir.
        patcher = mock.patch.object(
            session_store, "SESSIONS_DIR", self.base / "sessions"
        )
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
        """App bound to the test repo with direct-write change tracking."""
        info, _ = await setup_tracking(self.repo, "testsess")
        assert info is not None
        client = FakeClient(script)
        controller = ChatController(PROFILE, "k", client=client)
        app = AgentApp(
            PROFILE, "k", controller=controller, tool_registry=make_registry(),
            workspace=info.agent_workspace, session_name="testsess", git_info=info,
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

            # The agent edits a file in the workspace; next tool turn the
            # summary must appear with correct counts.
            (info.workspace / "app.py").write_text("line1\nline2\nline3\nline4\n")
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
        (info.workspace / "app.py").write_text("edited\n")
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
        (info.workspace / "app.py").write_text("agent edit\n")
        (info.workspace / "new.py").write_text("brand new\n")
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
        (info.workspace / "app.py").write_text("agent edit\n")
        (info.workspace / "new.py").write_text("brand new\n")
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
        app, _info = await self.make_git_app([ContentDelta("hi"), TurnComplete(usage=None)])
        async with app.run_test() as pilot:
            await pilot.press("h", "i", "enter")
            await self.wait_until(lambda: not app._turn_busy)

            # No pending changes: a notice, no diff view.
            app._handle_slash_command("/diff")
            await self.wait_until(
                lambda: any("no pending changes" in str(n.content) for n in app.query(NoticeLine))
            )
            self.assertFalse(app.query_one("#diff-summary", DiffSummary).has_class("expanded"))

        # No tracking at all: an error line.
        client = FakeClient([ContentDelta("x"), TurnComplete(usage=None)])
        bare = AgentApp(
            PROFILE, "k", controller=ChatController(PROFILE, "k", client=client),
        )
        async with bare.run_test() as pilot:
            bare._handle_slash_command("/diff")
            await self.wait_until(
                lambda: any("no change tracking" in str(e.content) for e in bare.query(ErrorLine))
            )

    async def test_accept_locks_in_changes_and_resets_diff(self):
        """/accept makes the current state the new baseline: /diff empties,
        and /undo of a pre-accept turn can no longer revert its file edits.
        Work continues from the accepted state."""
        script = [
            ToolCallStarted("c1", "noop"),
            ToolCallArgumentsDone("c1", "noop", "{}"),
            TurnComplete(has_tool_calls=True),
        ]
        app, info = await self.make_git_app(script)
        client = app.controller._client
        client.next_scripts = [
            [ContentDelta("done"), TurnComplete(usage=None)],
            [ContentDelta("done"), TurnComplete(usage=None)],
        ]
        async with app.run_test() as pilot:
            await self.run_simple_turn(app, pilot)
            # Feature one: the agent edits app.py and creates a file.
            (info.workspace / "app.py").write_text("feature one\n")
            (info.workspace / "feature.py").write_text("new\n")

            app._handle_slash_command("/accept")
            await self.wait_until(
                lambda: any("accepted: 2 file(s)" in str(n.content) for n in app.query(NoticeLine))
            )

            # /diff is reset: nothing pending vs the new baseline.
            summary = app.query_one("#diff-summary", DiffSummary)
            self.assertFalse(summary.has_class("visible"))

            # /undo of the (pre-accept) turn rolls back the conversation but
            # must NOT revert the accepted file changes.
            app._handle_slash_command("/undo")
            await self.wait_until(lambda: app.query_one(ChatInput).text == "hi")
            self.assertEqual((info.workspace / "app.py").read_text(), "feature one\n")
            self.assertTrue((info.workspace / "feature.py").exists())

            # Continue with feature two; only it shows as pending now —
            # the accepted feature is off /diff's radar entirely.
            (info.workspace / "second.py").write_text("feature two\n")
            await pilot.press("m", "o", "r", "e", "enter")
            await self.wait_until(lambda: not app._turn_busy)
            stat = await diff_stat(info)
            self.assertEqual({f.path for f in stat.files}, {"second.py"})

            # The accepted tree is persisted for resume.
            block = session_store.load_git_block("testsess")
            self.assertEqual(block["baseline_tree"], info.baseline_tree)

    async def test_reject_discards_everything_since_last_accept(self):
        """/reject restores the workspace to the accepted baseline, wiping all
        un-accepted changes (agent's and user's) while keeping the session."""
        script = [
            ToolCallStarted("c1", "noop"),
            ToolCallArgumentsDone("c1", "noop", "{}"),
            TurnComplete(has_tool_calls=True),
        ]
        app, info = await self.make_git_app(script)
        client = app.controller._client
        client.next_scripts = [[ContentDelta("done"), TurnComplete(usage=None)]]
        async with app.run_test() as pilot:
            await self.run_simple_turn(app, pilot)
            (info.workspace / "app.py").write_text("un-accepted edit\n")
            (info.workspace / "created.txt").write_text("new\n")

            app._handle_slash_command("/reject")
            await self.wait_until(
                lambda: any("rejected: 2 file(s)" in str(n.content) for n in app.query(NoticeLine))
            )

            self.assertEqual((info.workspace / "app.py").read_text(), "line1\nline2\n")
            self.assertFalse((info.workspace / "created.txt").exists())
            summary = app.query_one("#diff-summary", DiffSummary)
            self.assertFalse(summary.has_class("visible"))

            # A following /undo doesn't resurrect the rejected changes.
            app._handle_slash_command("/undo")
            await self.wait_until(lambda: app.query_one(ChatInput).text == "hi")
            self.assertEqual((info.workspace / "app.py").read_text(), "line1\nline2\n")

    async def test_accept_reject_without_tracking_error(self):
        app, _info = await self.make_git_app([ContentDelta("hi"), TurnComplete(usage=None)])
        app.git_info = None
        app.controller.git_info = None
        async with app.run_test() as pilot:
            app._handle_slash_command("/accept")
            app._handle_slash_command("/reject")
            await self.wait_until(lambda: len(app.query(ErrorLine)) >= 2)
            errors = [str(e.content) for e in app.query(ErrorLine)]
            self.assertTrue(any("/accept: no change tracking" in e for e in errors))
            self.assertTrue(any("/reject: no change tracking" in e for e in errors))

    # ------------------------------------------------------------------
    # resume reconnection

    async def test_resume_reconnects_tracking(self):
        info, _ = await setup_tracking(self.repo, "testsess")
        assert info is not None
        (self.repo / "app.py").write_text("pending agent edit\n")
        session_store.save_session("testsess", "m", Conversation(), git=info.to_block())

        client = FakeClient([ContentDelta("x"), TurnComplete(usage=None)])
        app = AgentApp(
            PROFILE, "k", controller=ChatController(PROFILE, "k", client=client),
            session_name="testsess",
        )
        async with app.run_test() as pilot:
            # Simulate /resume picking the same session.
            await app._on_session_selected("testsess")
            await pilot.pause()
            await self.wait_until(lambda: app.git_info is not None)

            self.assertEqual(app.git_info.workspace, info.workspace)
            self.assertEqual(app.git_info.baseline_tree, info.baseline_tree)
            summary = app.query_one("#diff-summary", DiffSummary)
            self.assertTrue(summary.has_class("visible"))
            text = str(app.query_one("#diff-summary-text").content)
            self.assertIn("Edited 1 file", text)

    async def test_resume_with_legacy_worktree_block_sets_up_fresh_tracking(self):
        """A session persisted by the old worktree-based build resumes with
        fresh direct-write tracking instead of failing."""
        legacy_block = {
            "session_id": "testsess",
            "base_ref": "abc",
            "branch": "agent/testsess",
            "worktree": str(self.base / "gone-worktree"),
            "original_workspace": str(self.repo),
            "git_common_dir": str(self.repo / ".git"),
        }
        session_store.save_session("testsess", "m", Conversation(), git=legacy_block)

        client = FakeClient([ContentDelta("x"), TurnComplete(usage=None)])
        app = AgentApp(
            PROFILE, "k", controller=ChatController(PROFILE, "k", client=client),
            workspace=self.repo, session_name="testsess",
        )
        async with app.run_test() as pilot:
            await app._on_session_selected("testsess")
            await pilot.pause()
            await self.wait_until(lambda: app.git_info is not None)

            self.assertEqual(app.git_info.workspace, self.repo.resolve())
            self.assertTrue(app.git_info.baseline_tree)
            await self.wait_until(
                lambda: any("old worktree" in str(n.content) for n in app.query(NoticeLine))
            )

    # ------------------------------------------------------------------
    # /undo + /retry

    async def test_undo_reverts_file_edits_and_restores_input(self):
        script = [
            ToolCallStarted("c1", "noop"),
            ToolCallArgumentsDone("c1", "noop", "{}"),
            TurnComplete(has_tool_calls=True),
        ]
        app, info = await self.make_git_app(script)
        client = app.controller._client
        client.next_scripts = [[ContentDelta("done"), TurnComplete(usage=None)]]
        async with app.run_test() as pilot:
            await self.run_simple_turn(app, pilot)
            # What the turn's tools did: edit a tracked file, create a new one.
            (info.workspace / "app.py").write_text("agent edit\n")
            (info.workspace / "created.txt").write_text("new\n")

            app._handle_slash_command("/undo")
            await self.wait_until(lambda: app.query_one(ChatInput).text == "hi")

            # File state restored to the turn's checkpoint.
            self.assertEqual(
                (info.workspace / "app.py").read_text(), "line1\nline2\n"
            )
            self.assertFalse((info.workspace / "created.txt").exists())
            # Conversation back to just the system message; input holds the text.
            self.assertEqual(
                [m.role for m in app.controller.conversation.messages], ["system"]
            )
            self.assertTrue(app.query_one(ChatInput).has_focus)
            self.assertFalse(app._turn_busy)
            await self.wait_until(
                lambda: any("undone" in str(n.content) for n in app.query(NoticeLine))
            )

    async def test_retry_reverts_file_edits_and_resends(self):
        script = [
            ToolCallStarted("c1", "noop"),
            ToolCallArgumentsDone("c1", "noop", "{}"),
            TurnComplete(has_tool_calls=True),
        ]
        app, info = await self.make_git_app(script)
        client = app.controller._client
        client.next_scripts = [[ContentDelta("done"), TurnComplete(usage=None)]]
        async with app.run_test() as pilot:
            await self.run_simple_turn(app, pilot)
            (info.workspace / "app.py").write_text("agent edit\n")
            (info.workspace / "created.txt").write_text("new\n")
            client.next_scripts.append(
                [ContentDelta("second answer"), TurnComplete(usage=None)]
            )

            app._handle_slash_command("/retry")
            await self.wait_until(lambda: not app._turn_busy)
            await self.wait_until(
                lambda: any(
                    m.role == "assistant" and m.content == "second answer"
                    for m in app.controller.conversation.messages
                )
            )

            # File edits from the retried turn are gone.
            self.assertEqual(
                (info.workspace / "app.py").read_text(), "line1\nline2\n"
            )
            self.assertFalse((info.workspace / "created.txt").exists())
            # Same user message, fresh response.
            self.assertEqual(
                [m.content for m in app.controller.conversation.messages][1:],
                ["hi", "second answer"],
            )
            self.assertEqual(len(app.query(UserMessage)), 1)

    async def test_user_files_are_visible_to_the_agent_immediately(self):
        """Direct-write mode: files the user adds mid-session are in the
        agent's workspace already — no sync step needed."""
        app, info = await self.make_git_app([ContentDelta("hi"), TurnComplete(usage=None)])
        async with app.run_test() as pilot:
            await self.run_simple_turn(app, pilot)
            (self.repo / "midway.txt").write_text("user file\n")
            self.assertEqual(
                (info.workspace / "midway.txt").read_text(), "user file\n"
            )

    async def test_undo_without_git_rolls_back_messages_only(self):
        """No git tracking: messages and usage still roll back, and the user
        is told file edits could not be reverted."""
        app, _ = await self.make_git_app([ContentDelta("hi"), TurnComplete(usage=None)])
        app.git_info = None
        app.controller.git_info = None
        async with app.run_test() as pilot:
            await pilot.press("h", "i", "enter")
            await self.wait_until(lambda: not app._turn_busy)

            app._handle_slash_command("/undo")
            await self.wait_until(lambda: app.query_one(ChatInput).text == "hi")

            self.assertEqual(
                [m.role for m in app.controller.conversation.messages], ["system"]
            )
            await self.wait_until(
                lambda: any(
                    "could not be reverted" in str(n.content)
                    for n in app.query(NoticeLine)
                )
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
