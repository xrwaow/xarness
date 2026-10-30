"""TUI tests: the Ctrl+C headless modal, no-local-echo rendering, and the
client/server import boundary. The app runs against a fake clientlib
client; the session behaviors themselves are covered in test_session.py."""

import asyncio
import unittest

from textual.app import App
from textual.pilot import Pilot

from xarness import protocol
from xarness.clientlib import ClientError
from xarness.conversation import Timing
from xarness.protocol import (
    CommandResult,
    ConfigInfo,
    DiffInfo,
    Event,
    SessionSummary,
    Snapshot,
    StatusPayload,
)
from xarness.tui import flow as tflow
from xarness.tui.transcript import ToolStatus


class FakeClient:
    """Scripted stand-in for XarnessClient: serves a snapshot and a live
    event queue; records commands."""

    def __init__(self, snapshot: Snapshot) -> None:
        self.snapshot_data = snapshot
        self.queue: asyncio.Queue = asyncio.Queue()
        self.commands: list[tuple[str, Event]] = []
        self.command_results: dict[str, CommandResult] = {}
        self.closed: list[str] = []
        self.refs: list[protocol.RefInfo] = []
        self.ref_updates: list[tuple[str, str, bool]] = []

    async def snapshot(self, name: str) -> Snapshot:
        return self.snapshot_data

    async def attach(self, name: str):
        if getattr(self, "fail_attach_first", False):
            self.fail_attach_first = False
            raise ClientError("connection lost")
        # The real SSE stream's first frame is a fresh Snapshot.
        yield self.snapshot_data
        while True:
            item = await self.queue.get()
            if item is None:
                return
            yield item

    async def command(self, name: str, command) -> CommandResult:
        self.commands.append((name, command))
        return self.command_results.get(
            command.type, CommandResult(cmd_id="t", ok=True, payload=protocol.RollbackResult(user_text="put back")),
        )

    async def sessions(self, **kwargs):
        return []

    async def config(self):
        return protocol.ConfigInfo(
            models=[{"name": "glm-flash", "display_name": "GLM Flash"}],
            sandboxes=[{"name": "default"}],
            presets=["default"], themes=[],
            default_model="glm-flash",
            default_model_display_name="GLM Flash",
        )

    async def diff(self, name: str, path: str | None = None) -> DiffInfo:
        return DiffInfo()

    async def search_files(self, workspace: str, q: str) -> list[str]:
        return []

    async def workspace_refs(self, workspace: str) -> list[protocol.RefInfo]:
        return list(self.refs)

    async def update_workspace_refs(
        self, workspace: str, path: str, add: bool,
    ) -> list[protocol.RefInfo]:
        self.ref_updates.append((workspace, path, add))
        if add:
            self.refs.append(protocol.RefInfo(path=path, mount=f".refs/{path}"))
        else:
            self.refs = [ref for ref in self.refs if ref.path != path]
        return list(self.refs)

    async def create_session(self, request) -> SessionSummary:
        return getattr(self, "create_result", None) or SessionSummary(
            name="new-session", active=True)

    async def set_default_profile(self, name: str) -> ConfigInfo:
        return ConfigInfo(default_model=name)

    async def delete_session(self, name: str) -> CommandResult:
        return CommandResult(cmd_id="t", ok=True)

    async def close_session(self, name: str) -> CommandResult:
        self.closed.append(name)
        return CommandResult(cmd_id="t", ok=True)

    async def aclose(self) -> None:
        pass


def make_snapshot(state: str = "idle") -> Snapshot:
    return Snapshot(
        messages=[],
        status=StatusPayload(
            state=state,  # type: ignore[arg-type]
            session_name="s1", workspace="/tmp/ws", mode="write",
            model_name="m", model_display_name="m", model_id="m",
        ),
    )


def running_status_event() -> Event:
    return protocol.StatusChanged(
        status=StatusPayload(
            state="running",  # type: ignore[arg-type]
            session_name="s1", workspace="/tmp/ws", mode="write",
            model_name="m", model_display_name="m", model_id="m",
        )
    )


class CtrlCModalTest(unittest.IsolatedAsyncioTestCase):
    async def _run(self, snapshot: Snapshot, script):
        from xarness.tui.app import AgentApp
        client = FakeClient(snapshot)
        app = AgentApp(client, workspace=None, session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
            await script(pilot, client, app)

    async def test_running_ctrl_c_shows_modal_and_continue_headless_exits(self) -> None:
        async def script(pilot: Pilot, client: FakeClient, app):
            client.queue.put_nowait(running_status_event())
            await pilot.pause()
            await pilot.press("ctrl+c")
            await pilot.pause()
            from xarness.tui.confirm_screen import ConfirmScreen
            self.assertIsInstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            await pilot.pause()
            self.assertFalse(app.is_running)

        await self._run(make_snapshot(), script)

    async def test_running_ctrl_c_stop_and_quit_cancels_then_exits(self) -> None:
        async def script(pilot: Pilot, client: FakeClient, app):
            client.queue.put_nowait(running_status_event())
            await pilot.pause()
            await pilot.press("ctrl+c")
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()
            cancels = [c for _, c in client.commands if c.type == "cancel"]
            self.assertEqual(len(cancels), 1)
            # "Stop and quit" also unloads the session from the host (it
            # stays saved on disk) — otherwise it lingers in the host's list.
            self.assertEqual(client.closed, ["s1"])
            self.assertFalse(app.is_running)

        await self._run(make_snapshot(), script)

    async def test_running_ctrl_c_escape_stays_in_the_app(self) -> None:
        async def script(pilot: Pilot, client: FakeClient, app):
            client.queue.put_nowait(running_status_event())
            await pilot.pause()
            await pilot.press("ctrl+c")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            self.assertTrue(app.is_running)
            self.assertEqual([c for _, c in client.commands if c.type == "cancel"], [])

        await self._run(make_snapshot(), script)

    async def test_idle_ctrl_c_exits_without_a_modal(self) -> None:
        async def script(pilot: Pilot, client: FakeClient, app):
            await pilot.press("ctrl+c")
            await pilot.pause()
            self.assertFalse(app.is_running)
            from xarness.tui.confirm_screen import ConfirmScreen
            self.assertNotIsInstance(app.screen, ConfirmScreen)

        await self._run(make_snapshot(), script)


class NoLocalEchoTest(unittest.IsolatedAsyncioTestCase):
    async def test_user_message_reaches_the_log_only_via_the_event_stream(self) -> None:
        from xarness.tui.app import AgentApp

        client = FakeClient(make_snapshot())
        app = AgentApp(client, workspace=None, session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
            # Nothing in the model yet: the client does not echo its own sends.
            self.assertEqual(
                [type(e.item).__name__ for e in app._flow_model.entries
                 if type(e.item).__name__ == "UserItem"], [])
            client.queue.put_nowait(protocol.UserMessageAdded(
                text="hello", message_id="m1", send_time="2026-01-01T00:00:00+00:00",
            ))
            await pilot.pause()
            users = [e for e in app._flow_model.entries
                     if isinstance(e.item, tflow.UserItem)]
            self.assertEqual(len(users), 1)
            self.assertEqual(users[0].item.message_id, "m1")
            self.assertEqual(
                [c for _, c in client.commands if c.type == "send_message"], [],
            )

    async def test_status_changed_updates_the_steer_bar(self) -> None:
        from xarness.tui.app import AgentApp
        from xarness.tui.widgets import SteerQueueBar

        client = FakeClient(make_snapshot())
        app = AgentApp(client, workspace=None, session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
            client.queue.put_nowait(protocol.StatusChanged(status=StatusPayload(
                state="running",  # type: ignore[arg-type]
                session_name="s1", workspace="/tmp/ws", mode="write",
                model_name="m", model_display_name="m", model_id="m",
                queued=[protocol.QueuedMessage(text="steer!")],
            )))
            await pilot.pause()
            bar = app.query_one("#steer-queue-bar", SteerQueueBar)
            self.assertEqual(len(bar.query(".steer-item")), 1)
            self.assertTrue(bar.has_class("visible"))


class ImportBoundaryTest(unittest.TestCase):
    """The TUI package imports only clientlib, protocol and pure helpers —
    never the controller, session, tools, gitwork, sandbox or host modules."""

    BANNED = (
        "xarness.controller", "xarness.session", "xarness.tools",
        "xarness.sandbox", "xarness.gitwork", "xarness.host", "xarness.api",
        "xarness.client",
    )

    def test_tui_never_imports_the_backend(self) -> None:
        from pathlib import Path
        tui = Path(__file__).resolve().parent.parent / "src" / "xarness" / "tui"
        for path in tui.glob("*.py"):
            source = path.read_text()
            for banned in self.BANNED:
                self.assertNotIn(
                    f"from ..{banned.split('.', 1)[1]} import", source,
                    f"{path.name} imports {banned}",
                )
                self.assertNotIn(
                    f"import {banned}", source,
                    f"{path.name} imports {banned}",
                )


class ReconnectTest(unittest.IsolatedAsyncioTestCase):
    """On connection loss the TUI respawns the host and reattaches; the
    stream endpoint loads the saved session, with no separate round-trip."""

    async def test_connection_loss_respawns_and_reattaches(self) -> None:
        from unittest import mock

        import xarness.tui.app as app_mod
        from xarness.tui.app import AgentApp

        client = FakeClient(make_snapshot())
        client.fail_attach_first = True
        client.spawned: list[bool] = []

        async def fake_ensure_host(timeout: float = 5.0) -> None:
            client.spawned.append(True)

        with mock.patch.object(app_mod, "ensure_host", fake_ensure_host):
            app = AgentApp(client, workspace=None, session_name="s1")
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.pause()
                await pilot.pause()

        self.assertEqual(client.spawned, [True])


class _WidgetApp(App):
    """Bare app that loads the real stylesheet for widget-level tests."""


class ThinkingDurationTest(unittest.IsolatedAsyncioTestCase):
    """The live path settles WITHOUT a local estimate (the backend owns the
    number); a later TurnComplete fills it in; the first backend number wins."""

    def test_backend_duration_used_not_client_estimate(self) -> None:
        thinking = tflow.ThinkingItem(parts=["hmm"])
        thinking.finish(None)
        self.assertEqual(thinking.summary_text(), "Thought for …")
        thinking.finish(7.0)
        self.assertEqual(thinking.summary_text(), "Thought for 7.0s")

    def test_backend_duration_first_wins(self) -> None:
        thinking = tflow.ThinkingItem()
        thinking.finish(3.0)
        thinking.finish(99.0)  # e.g. a stray second event
        self.assertEqual(thinking.summary_text(), "Thought for 3.0s")

    def test_streaming_expands_then_finish_collapses_and_click_toggles(self) -> None:
        # Streaming: expanded by default, and clickable mid-stream.
        thinking = tflow.ThinkingItem(parts=["hmm"])
        self.assertTrue(thinking.expanded)
        thinking.expanded = False  # user collapsed it mid-stream
        thinking.expanded = True   # ...and reopened it — both allowed
        # Round end: auto-collapse, whatever the user's last toggle was.
        thinking.finish(2.0)
        self.assertFalse(thinking.expanded)
        # A later duration override (TurnComplete) must not re-collapse an
        # item the user re-expanded after it finished.
        thinking.expanded = True
        thinking.finish(3.0)  # _estimated was False: ignored
        self.assertTrue(thinking.expanded)


class CommandPathsTest(unittest.IsolatedAsyncioTestCase):
    """Fire-and-forget command paths must actually send: bare (un-awaited)
    coroutine calls silently did nothing — esc-esc cancel, steer 'send now'
    and discard were all dead."""

    async def test_interrupt_and_steer_commands_are_sent(self) -> None:
        from xarness.protocol import QueuedMessage
        from xarness.tui.app import AgentApp

        client = FakeClient(make_snapshot("running"))
        app = AgentApp(client, workspace=None, session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
            client.queue.put_nowait(running_status_event())
            await pilot.pause()
            # Double-esc interrupts the running turn.
            await pilot.press("escape")
            await pilot.press("escape")
            await pilot.pause()
            # Empty Enter with queued messages flushes them now.
            app.status = app.status.model_copy(update={
                "queued": [QueuedMessage(text="steer")],
            })
            app._refresh_steer_bar()
            await pilot.press("enter")
            await pilot.pause()
            # ctrl+shift+backspace discards the oldest queued message.
            await pilot.press("ctrl+shift+backspace")
            await pilot.pause()
            types = [c.type for _, c in client.commands]
            self.assertEqual(
                types, ["cancel", "flush_queued", "discard_queued"], types,
            )

    async def test_popup_confirm_on_arg_command_completes_instead_of_running(self) -> None:
        """Enter on /filter (takes an argument) must leave the word in the
        input for the user to finish, not execute it with an empty arg."""
        from xarness.tui.app import AgentApp
        from xarness.tui.widgets import ChatInput

        client = FakeClient(make_snapshot())
        app = AgentApp(client, workspace=None, session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
            app._accept_popup_value("filter", was_slash=True)
            await pilot.pause()
            chat_input = app.query_one("#chat-input", ChatInput)
            self.assertEqual(chat_input.text, "/filter ")
            self.assertEqual(client.commands, [])

    async def test_ctrl_c_skips_headless_prompt_when_opened_elsewhere(self) -> None:
        from xarness.tui.app import AgentApp

        client = FakeClient(make_snapshot("running"))
        app = AgentApp(client, workspace=None, session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
            status = running_status_event().status.model_copy(
                update={"clients_attached": 2},
            )
            client.queue.put_nowait(__import__(
                "xarness.protocol", fromlist=["StatusChanged"],
            ).StatusChanged(status=status))
            await pilot.pause()
            await pilot.press("ctrl+c")
            await pilot.pause()
            # No modal shown; the app just exits (the session stays for the
            # other attached client).
            self.assertFalse(app.is_running)


class RefsScreenTest(unittest.IsolatedAsyncioTestCase):
    """The /refs manager lists container mounts (`.refs/<alias>`) and
    dispatches add/remove without ever touching the worktree."""

    async def test_lists_mounts_and_dispatches_remove(self) -> None:
        from xarness.protocol import RefInfo
        from xarness.tui.app import AgentApp
        from xarness.tui.refs_screen import RefsScreen

        client = FakeClient(make_snapshot())
        app = AgentApp(client, workspace=None, session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
            refs = [RefInfo(path="/tmp/notes.md", mount=".refs/notes.md")]
            client.refs = list(refs)
            app.push_screen(RefsScreen(refs))
            await pilot.pause()
            screen = app.screen
            self.assertIsInstance(screen, RefsScreen)
            # The dimmed line is the container path the model uses; the
            # worktree never gets a .refs directory.
            self.assertEqual(screen._refs, refs)
            await pilot.click("#refs-remove-0")
            await pilot.pause()
            self.assertEqual(client.ref_updates[-1][1:], ("/tmp/notes.md", False))
            # Typing a path in the input dispatches an add.
            for key in "/tmp/other.md":
                await pilot.press(key)
            await pilot.press("enter")
            await pilot.pause()
            self.assertEqual(client.ref_updates[-1][1:], ("/tmp/other.md", True))

    async def test_slash_command_opens_the_manager(self) -> None:
        from xarness.tui.app import AgentApp
        from xarness.tui.refs_screen import RefsScreen

        client = FakeClient(make_snapshot())
        app = AgentApp(client, workspace=None, session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
            app._handle_slash_command("/refs")
            await pilot.pause()
            self.assertIsInstance(app.screen, RefsScreen)


class AttachSeedsPumpTest(unittest.IsolatedAsyncioTestCase):
    """The pump attaches to the session by name; reconnects always deliver
    a fresh Snapshot (which re-renders the history exactly once — the
    rendered snapshot is replaced, not appended to)."""

    async def test_attach_is_called_with_the_session_name(self) -> None:

        from xarness.tui.app import AgentApp

        client = FakeClient(make_snapshot())
        calls: list[str] = []

        async def attach(name):
            calls.append(name)
            while True:
                item = await client.queue.get()
                if item is None:
                    return
                yield item

        client.attach = attach
        client.snapshot_data = Snapshot(
            messages=[],
            status=client.snapshot_data.status,
        )
        app = AgentApp(client, workspace=None, session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
        self.assertEqual(calls, ["s1"])


class MultiRoundStreamTest(unittest.IsolatedAsyncioTestCase):
    """Rounds 2+ of a turn must stream into FRESH entries: the previous
    round's thinking block is finished (collapsed) and its answer settled —
    appending to them made later CoT and answers invisible until reload."""

    async def test_second_round_streams_into_new_widgets(self) -> None:
        from xarness.tui.app import AgentApp

        async def script(pilot: Pilot, client: FakeClient, app):
            def items(cls):
                return [e for e in app._flow_model.entries
                        if isinstance(e.item, cls)]
            q = client.queue
            q.put_nowait(protocol.ReasoningDelta(text="think one"))
            q.put_nowait(protocol.ContentDelta(text="answer one"))
            q.put_nowait(protocol.ToolCallStarted(call_id="c1", name="read_file"))
            q.put_nowait(protocol.ToolCallArgumentsDone(
                call_id="c1", name="read_file", arguments_json="{}"))
            q.put_nowait(protocol.TurnComplete(has_tool_calls=True))
            q.put_nowait(protocol.ToolFinished(call_id="c1", ok=True, output="ok"))
            await pilot.pause(); await pilot.pause()
            thinking = items(tflow.ThinkingItem)
            self.assertEqual(len(thinking), 1)
            self.assertTrue(thinking[0].item.done)
            # Round 2 begins: the provider opens it with ProcessingStarted.
            q.put_nowait(protocol.ProcessingStarted())
            q.put_nowait(protocol.ReasoningDelta(text="think two"))
            q.put_nowait(protocol.ContentDelta(text="answer two"))
            await pilot.pause(); await pilot.pause()
            thinking = items(tflow.ThinkingItem)
            self.assertEqual(len(thinking), 2, "round 2 CoT needs its own entry")
            self.assertEqual(thinking[1].item.reasoning, "think two")
            assistants = items(tflow.AssistantItem)
            self.assertEqual(len(assistants), 2, "round 2 answer needs its own entry")
            q.put_nowait(protocol.TurnComplete())
            await pilot.pause()

        app_client = FakeClient(make_snapshot(state="running"))
        app = AgentApp(app_client, workspace=None, session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
            await script(pilot, app_client, app)


class DriftPromptStateTest(unittest.IsolatedAsyncioTestCase):
    async def test_resolved_drift_is_cleared_from_local_status(self) -> None:
        """After DriftResolved, a later StatusChanged must not re-open the
        prompt for a drift that the host already resolved."""
        from xarness.tui.app import AgentApp

        app = AgentApp(FakeClient(make_snapshot()), workspace=None,
                       session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
            app.status = StatusPayload(
                state="idle",
                pending_drift=protocol.PendingDrift(
                    workspace="/tmp/ws", last_known="aaa", current="bbb",
                    files=["a.txt"]),
            )
            await app._handle_event(protocol.DriftResolved(accepted=False))
            self.assertIsNone(app.status.pending_drift)


class LiveToolDetailTest(unittest.IsolatedAsyncioTestCase):
    """While a tool is still MAKING_CALL its streamed arguments must already
    show the useful detail: write_file's +LOC, edit_file's +adds -dels and
    run_bash's command — not just the bare tool name."""

    def test_settled_summary_derives_detail_from_args(self) -> None:
        """The header is not stored separately: once settled, the summary is
        rebuilt from the call's arguments."""
        from xarness.tui.tool_view import ToolCallView, summary_text
        from xarness.tui.transcript import ToolStatus

        def summary(name: str, args: str) -> str:
            return summary_text(ToolCallView(
                name=name, args=args, status=ToolStatus.CALL_SUCCEEDED,
            )).plain

        self.assertEqual(
            summary("write_file", '{"path": "a.py", "content": "a\\nb\\n"}'),
            "write_file +2 a.py",
        )
        self.assertEqual(
            summary("edit_file", '{"path": "b.py", "edits": [{"old_string": '
                    '"x", "new_string": "y"}]}'),
            "edit_file +1 -1 b.py",
        )
        self.assertEqual(
            summary("run_bash", '{"command": "ls -la"}'),
            "run_bash ls -la",
        )
        self.assertEqual(
            summary("read_file", '{"path": "c.py", "start_line": 2}'),
            "read_file c.py [start_line=2]",
        )

    def test_partial_edit_arguments_already_count_lines(self) -> None:
        """A half-streamed edits array is not valid JSON yet; the live label
        must still grow the +adds -dels count instead of showing only a path."""
        from xarness.tui.tool_view import _edit_live

        _name, detail = _edit_live(
            '{"path": "b.py", "edits": [{"old_string": "x", '
            '"new_string": "y'
        )
        self.assertIn("+1 -1", detail)
        self.assertIn("b.py", detail)

    async def test_streamed_arguments_render_live_details(self) -> None:
        from xarness.tui.app import AgentApp

        calls = [
            ("w1", "write_file",
             '{"path": "a.py", "content": "a\\nb\\nc\\n"}', "+3 a.py"),
            ("e1", "edit_file",
             '{"path": "b.py", "edits": [{"old_string": "x", '
             '"new_string": "y\\nz"}]}', "+2 -1 b.py"),
            ("b1", "run_bash", '{"command": "grep -rn foo"}',
             "grep -rn foo"),
        ]
        client = FakeClient(make_snapshot(state="running"))
        app = AgentApp(client, workspace=None, session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
            for call_id, name, args, _expected in calls:
                client.queue.put_nowait(
                    protocol.ToolCallStarted(call_id=call_id, name=name))
                # Stream in two halves: the label must follow partial JSON.
                client.queue.put_nowait(protocol.ToolCallArgumentsDelta(
                    call_id=call_id, text=args[: len(args) // 2]))
                client.queue.put_nowait(protocol.ToolCallArgumentsDelta(
                    call_id=call_id, text=args[len(args) // 2:]))
            await pilot.pause()
            await pilot.pause()

            entries = [e for e in app._flow_model.entries
                       if isinstance(e.item, tflow.ToolItem)]
            self.assertEqual(len(entries), len(calls))
            for entry, (_, _, _, expected) in zip(entries, calls):
                presentation = await app._presenter.present(entry, 80)
                text = "\n".join(strip.text for strip in presentation.strips)
                self.assertIn(expected, text)
            # The same labels are what the live view actually paints (not
            # just what the presenter returns when asked directly).
            flow = app.query_one("#chat-log")
            painted = "\n".join(
                flow.row_text(y) for y in range(flow.row_count))
            self.assertIn("Running write_file +3 a.py", painted)
            self.assertIn("Running run_bash grep -rn foo", painted)

    def test_expanded_edit_file_body_shows_the_diff(self) -> None:
        """edit_file's returned unified diff must render as +/- rows — the
        tool output generator used to collapse its headers/hunks onto one
        line, leaving the expanded block empty."""
        from xarness.tools import _format_edit_diff
        from xarness.tui.tool_view import ToolCallView, _edit_body
        from xarness.tui.transcript import ToolStatus

        output = "applied 1 edit to a.py\n" + _format_edit_diff(
            "a.py", "old\nkeep\n", "new\nkeep\n",
        )
        body = _edit_body(ToolCallView(
            name="edit_file", args='{"path": "a.py"}', output=output,
            status=ToolStatus.CALL_SUCCEEDED, expanded=True,
        ))
        self.assertIn("- old", body.plain)
        self.assertIn("+ new", body.plain)

    async def test_running_run_bash_shows_streamed_output(self) -> None:
        """While the call is still running, each ToolOutputDelta must repaint
        the expanded body (not wait for ToolFinished)."""
        from xarness.tui.app import AgentApp

        client = FakeClient(make_snapshot(state="running"))
        app = AgentApp(client, workspace=None, session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
            client.queue.put_nowait(
                protocol.ToolCallStarted(call_id="b1", name="run_bash"))
            client.queue.put_nowait(protocol.ToolCallArgumentsDone(
                call_id="b1", name="run_bash",
                arguments_json='{"command": "echo hi"}',
            ))
            client.queue.put_nowait(
                protocol.ToolOutputDelta(call_id="b1", text="first\n"))
            await pilot.pause()
            await pilot.pause()
            entry = [e for e in app._flow_model.entries
                     if isinstance(e.item, tflow.ToolItem)][0]
            self.assertIs(entry.item.status, ToolStatus.MAKING_CALL)
            # No auto-expand: a running call stays collapsed until clicked.
            self.assertFalse(entry.item.expanded)
            flow = app.query_one("#chat-log")

            def painted() -> str:
                return "\n".join(
                    flow.row_text(y) for y in range(flow.row_count))

            self.assertNotIn("first", painted())
            # Expand by clicking the entry (the real path, not by flipping
            # the flag and asking the presenter directly).
            from textual_flowview import FlowView
            app.post_message(FlowView.Clicked(flow, entry, 0, 0))
            await pilot.pause()
            await pilot.pause()
            self.assertIn("first", painted())

            client.queue.put_nowait(
                protocol.ToolOutputDelta(call_id="b1", text="second\n"))
            await pilot.pause()
            await pilot.pause()
            self.assertIn("first", painted())
            self.assertIn("second", painted())

    async def test_completed_edit_file_expands_to_diff(self) -> None:
        """End-to-end live path: the finished call's output feeds the
        expanded body, not just a bare "applied" line."""
        from xarness.tools import _format_edit_diff
        from xarness.tui.app import AgentApp

        client = FakeClient(make_snapshot(state="running"))
        app = AgentApp(client, workspace=None, session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
            output = "applied 1 edit to a.py\n" + _format_edit_diff(
                "a.py", "old\nkeep\n", "new\nkeep\n",
            )
            client.queue.put_nowait(
                protocol.ToolCallStarted(call_id="e1", name="edit_file"))
            client.queue.put_nowait(protocol.ToolCallArgumentsDone(
                call_id="e1", name="edit_file",
                arguments_json='{"path": "a.py", "edits": []}',
            ))
            client.queue.put_nowait(protocol.ToolFinished(
                call_id="e1", ok=True, output=output,
            ))
            await pilot.pause()
            await pilot.pause()
            entry = [e for e in app._flow_model.entries
                     if isinstance(e.item, tflow.ToolItem)][0]
            self.assertIn("diff --git", entry.item.output)
            entry.item.expanded = True
            presentation = await app._presenter.present(entry, 80)
            text = "\n".join(strip.text for strip in presentation.strips)
            self.assertIn("- old", text)
            self.assertIn("+ new", text)

class SessionPopupRowTest(unittest.IsolatedAsyncioTestCase):
    """Session popup rows: {title} {age} {dir} {diff}, with a right-aligned
    state word (padded to the longest state word) for loaded sessions —
    waiting=accent, running/compacting=success, idle=muted; saved (not
    loaded) rows show no state word at all."""

    def _summaries(self):
        def s(name, state, active, attached=0):
            return SessionSummary(
                name=name, state=state, active=active, clients_attached=attached,
                last_user_message_at="2026-01-01T00:00:00+00:00",
            )
        return [
            s("waiting-one", "waiting", True),
            s("running-one", "running", True),
            s("compacting-one", "compacting", True),
            s("idle-one", "idle", False),
            s("saved-one", "saved", False),
        ]

    def test_state_word_color_per_state(self) -> None:
        from xarness import theme
        from xarness.tui.list_picker import STATE_WORDS as _STATE_WORDS
        expected = {
            "waiting": "accent", "running": "success",
            "compacting": "success", "idle": "muted",
        }
        for state, key in expected.items():
            self.assertEqual(_STATE_WORDS.get(state), (state, key))
            self.assertIn(key, theme.PALETTE)
        # Saved / not loaded sessions show no state word at all.
        self.assertIsNone(_STATE_WORDS.get("saved"))

    async def test_rows_render_title_age_and_dot(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Label, OptionList

        from xarness.tui.list_picker import ListPicker, session_options

        class Host(App):
            def compose(self) -> ComposeResult:
                yield Label("")

        app = Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(ListPicker(session_options(self._summaries(), None), title="Open session", filterable=True))
            await pilot.pause()
            option_list = app.screen.query_one("#picker-list", OptionList)
            prompts = [
                str(option_list.get_option_at_index(i).prompt)
                for i in range(option_list.option_count)
            ]
            self.assertEqual(len(prompts), 5)
            # Only loaded sessions get a state word; saved-one shows none.
            from xarness.tui.list_picker import STATE_WIDTH as _STATE_WIDTH
            for state, name in (("waiting", "waiting-one"),
                                ("running", "running-one"),
                                ("compacting", "compacting-one"),
                                ("idle", "idle-one")):
                row = next(p for p in prompts if name in p)
                # State word is right-aligned: padded to the column width.
                self.assertIn(state.rjust(_STATE_WIDTH), row,
                              f"{name} missing right-aligned '{state}'")
            saved_row = next(p for p in prompts if "saved-one" in p)
            self.assertFalse(
                any(state.rjust(_STATE_WIDTH) in saved_row
                    for state in ("waiting", "running", "compacting", "idle")))
            for name in ("waiting-one", "running-one", "compacting-one",
                         "idle-one", "saved-one"):
                self.assertTrue(
                    any(name in p for p in prompts), f"{name} missing from rows")
            # Every row shows an age column.
            self.assertTrue(all("m" in p or "h" in p or "d" in p for p in prompts))

    def test_workspace_column_shows_innermost_dir(self) -> None:
        from xarness.tui.list_picker import session_options

        entries = [
            SessionSummary(
                name="nested", workspace="/home/me/projects/my-app",
                state="saved", last_user_message_at="2026-01-01T00:00:00+00:00",
            ),
            SessionSummary(name="elsewhere", state="saved"),
        ]
        rows = {str(o.id): str(o.prompt) for o in session_options(entries, None)}
        self.assertIn("my-app", rows["nested"])
        self.assertNotIn("/home/me/projects/my-app", rows["nested"])
        self.assertNotIn("my-app", rows["elsewhere"])

    async def test_search_filters_and_restores_rows(self) -> None:
        """Regression: typing in the search box must narrow the list and
        deleting the query must bring every row back."""
        from textual.app import App, ComposeResult
        from textual.widgets import Input, OptionList

        from xarness.tui.list_picker import ListPicker, session_options

        entries = [
            SessionSummary(name=f"2026-10-0{i} sess", state="saved")
            for i in range(3)
        ]

        class Host(App):
            def compose(self) -> ComposeResult:
                yield Input()

        app = Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(ListPicker(session_options(entries, None), title="Open session", filterable=True))
            await pilot.pause()
            ol = app.screen.query_one("#picker-list", OptionList)
            self.assertEqual(ol.option_count, 3)
            await pilot.press("0", "0")
            await pilot.pause()
            self.assertEqual(ol.option_count, 1)  # only "2026-10-00 sess"
            await pilot.press("backspace")
            await pilot.pause()
            self.assertEqual(ol.option_count, 3)
            # Rows stay in entry order after re-filtering.
            names = [str(o.id) for o in ol.options]
            self.assertEqual(names, [f"2026-10-0{i} sess" for i in range(3)])
            await pilot.press("9", "9")
            await pilot.pause()
            self.assertEqual(ol.option_count, 0)

class TerminalTitleTest(unittest.IsolatedAsyncioTestCase):
    """The CLI tab title follows the attached session's workspace."""

    async def test_title_follows_attached_session_workspace(self) -> None:
        from xarness.tui.app import AgentApp

        async def script(pilot: Pilot, client: FakeClient, app):
            written: list[str] = []
            app._set_terminal_title = written.append  # type: ignore[method-assign]
            snap = make_snapshot()
            object.__setattr__(
                snap, "status",
                StatusPayload(
                    state="idle", session_name="other", workspace="/other/ws",
                    mode="write", model_name="m", model_display_name="m", model_id="m",
                ),
            )
            client.snapshot_data = snap
            await app._attach("other")
            await pilot.pause()
            self.assertIn("xarness ws", written, f"title not updated on attach: {written}")
            written.clear()
            app._detach()
            self.assertTrue(written, "title not restored on detach")

        client = FakeClient(make_snapshot())
        app = AgentApp(client, workspace=None, session_name=None)
        async with app.run_test() as pilot:
            await pilot.pause()
            await script(pilot, client, app)

    async def test_finished_suffix_clears_when_the_next_turn_starts(self) -> None:
        from xarness.protocol import ProcessingStarted, TurnFinished
        from xarness.tui.app import AgentApp

        client = FakeClient(make_snapshot())
        app = AgentApp(client, workspace=None, session_name=None)
        async with app.run_test() as pilot:
            await pilot.pause()
            written: list[str] = []
            app._set_terminal_title = written.append  # type: ignore[method-assign]
            snap = make_snapshot()
            object.__setattr__(
                snap, "status",
                StatusPayload(
                    state="idle", session_name="other",
                    workspace="/workspace/projects/xarness",
                    mode="write", model_name="m", model_display_name="m", model_id="m",
                ),
            )
            client.snapshot_data = snap
            await app._attach("other")
            await pilot.pause()
            self.assertEqual(written[-1], "xarness xarness")

            await app._handle_event(TurnFinished(elapsed=1.0))
            self.assertEqual(written[-1], "xarness xarness - finished")

            # A new round/turn clears the stale suffix.
            await app._handle_event(ProcessingStarted())
            self.assertEqual(written[-1], "xarness xarness")


class TerminalTitlePathTest(unittest.IsolatedAsyncioTestCase):
    """The tab title uses the workspace's own directory name, never a
    cropped multi-component path: ``/workspace/projects/xarness`` and
    ``/home/me/xarness`` must both read ``xarness xarness``."""

    async def test_title_for_uses_the_last_path_component(self) -> None:
        from xarness.tui.app import AgentApp

        app = AgentApp(FakeClient(make_snapshot()), workspace=None, session_name=None)
        async with app.run_test() as pilot:
            await pilot.pause()
            self.assertEqual(app._title_for("/workspace/projects/xarness"), "xarness")
            self.assertEqual(app._title_for("/home/me/larkspur"), "larkspur")
            self.assertEqual(app._title_for("/home/me/larkspur/"), "larkspur")
            self.assertEqual(app._title_for("/"), "/")


class InterruptSettleTest(unittest.IsolatedAsyncioTestCase):
    """An interrupt ends the turn without a TurnComplete for this client:
    the Thinking shimmer (or a half-streamed tool's writing shimmer) must
    settle when the idle StatusChanged arrives, not shimmer forever."""

    async def test_idle_status_settles_thinking_shimmer(self) -> None:
        from xarness.tui.app import AgentApp

        async def script(pilot: Pilot, client: FakeClient, app):
            q = client.queue
            q.put_nowait(protocol.ReasoningDelta(text="thinking..."))
            await pilot.pause()
            await pilot.pause()
            thinking = [e for e in app._flow_model.entries
                        if isinstance(e.item, tflow.ThinkingItem)]
            self.assertEqual(len(thinking), 1)
            self.assertFalse(thinking[0].item.done)
            # The interrupt: an error Notice plus the explicit end event
            # (the Notice itself settles nothing).
            q.put_nowait(protocol.Notice(text="interrupted", level="error"))
            q.put_nowait(protocol.TurnEnded(outcome="cancelled"))
            await pilot.pause()
            self.assertTrue(
                [e for e in app._flow_model.entries
                 if isinstance(e.item, tflow.ThinkingItem)][0].item.done)

        client = FakeClient(make_snapshot(state="running"))
        app = AgentApp(client, workspace=None, session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
            await script(pilot, client, app)

    async def test_idle_status_settles_tool_writing_shimmer(self) -> None:
        async def script(pilot: Pilot, client: FakeClient, app):
            q = client.queue
            # The call starts waiting: pending strip first, then the tool's
            # Running shimmer replaces it (one ToolItem per call, created at
            # ToolCallStarted; streamed args accumulate in it).
            q.put_nowait(protocol.ProcessingStarted())
            await pilot.pause()
            self.assertTrue([e for e in app._flow_model.entries
                             if isinstance(e.item, tflow.PendingItem)])
            self.assertEqual(app._phase, "pending")
            q.put_nowait(protocol.ToolCallStarted(call_id="c1", name="read_file"))
            q.put_nowait(protocol.ToolCallArgumentsDelta(call_id="c1", text='{"f"'))
            await pilot.pause()
            await pilot.pause()
            blocks = [e for e in app._flow_model.entries
                      if isinstance(e.item, tflow.ToolItem)]
            self.assertEqual(len(blocks), 1)
            self.assertEqual(blocks[0].item.args, '{"f"')
            self.assertIsNone(app._pending_entry, "strip outlived the first content")
            q.put_nowait(protocol.Notice(text="interrupted", level="error"))
            q.put_nowait(protocol.StatusChanged(
                status=StatusPayload(
                    state="idle", session_name="s1", workspace="/tmp/ws",
                    mode="write", model_name="m", model_display_name="m", model_id="m",
                )
            ))
            await pilot.pause()
            # The half-streamed call settles as failed — no eternal shimmer.
            self.assertEqual(blocks[0].item.status, ToolStatus.CALL_FAILED)
            self.assertEqual(app._phase, "idle")
            self.assertFalse(
                [e for e in app._flow_model.entries
                 if isinstance(e.item, tflow.PendingItem) and e.alive],
                "pending strip outlived the interrupt")

    async def test_pending_strip_only_while_waiting_on_provider(self) -> None:
        """The strip means "waiting on the LLM", not "the turn is busy".

        A round that ends with tool calls goes idle (the tools execute with no
        strip); the strip only comes back when the *next* round's
        ``ProcessingStarted`` opens a provider request, then goes away on the
        first delta."""
        from xarness.tui.app import AgentApp

        async def script(pilot: Pilot, client: FakeClient, app):
            q = client.queue
            q.put_nowait(protocol.ProcessingStarted())
            q.put_nowait(protocol.ContentDelta(text="hi"))
            q.put_nowait(protocol.TurnComplete(has_tool_calls=True))
            await pilot.pause()
            # Tools are about to run: no provider request is in flight.
            self.assertEqual(app._phase, "idle")
            self.assertFalse([e for e in app._flow_model.entries
                              if isinstance(e.item, tflow.PendingItem) and e.alive])
            # The next round (after the tools) waits on the provider again.
            q.put_nowait(protocol.ProcessingStarted())
            await pilot.pause()
            self.assertEqual(app._phase, "pending")
            self.assertTrue([e for e in app._flow_model.entries
                             if isinstance(e.item, tflow.PendingItem) and e.alive])
            q.put_nowait(protocol.ContentDelta(text="again"))
            await pilot.pause()
            self.assertEqual(app._phase, "streaming")
            self.assertFalse([e for e in app._flow_model.entries
                              if isinstance(e.item, tflow.PendingItem) and e.alive])
            q.put_nowait(protocol.TurnComplete(has_tool_calls=False))
            await pilot.pause()
            self.assertEqual(app._phase, "idle")

        client = FakeClient(make_snapshot(state="running"))
        app = AgentApp(client, workspace=None, session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
            await script(pilot, client, app)


class ResumeListHeightTest(unittest.IsolatedAsyncioTestCase):
    """The /sessions list must show more than one row at a time (the list
    widgets' 1fr default collapses to a single row inside the auto-height
    modal)."""

    async def test_multiple_rows_visible(self) -> None:
        from textual.widgets import OptionList

        from xarness.tui.app import AgentApp
        from xarness.tui.list_picker import ListPicker, session_options
        summaries = [
            SessionSummary(name=f"s{i}", state="saved", active=False)
            for i in range(5)
        ]
        client = FakeClient(make_snapshot())
        app = AgentApp(client, workspace=None, session_name=None)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app.push_screen(ListPicker(session_options(summaries, None), title="Open session", filterable=True))
            await pilot.pause()
            screen = app.screen
            option_list = screen.query_one("#picker-list", OptionList)
            self.assertGreaterEqual(
                option_list.region.height, 5,
                "list collapsed: not all 5 rows visible at once",
            )
            self.assertEqual(option_list.option_count, 5)


def _msg(role: str, mid: str, content: str) -> protocol.Message:
    return protocol.Message(role=role, id=mid, content=content)


class UndoTrimTest(unittest.IsolatedAsyncioTestCase):
    """Undo-from-here rebuilds the transcript from the new snapshot — a
    data-only rebuild (cheap under FlowView), including messages that
    arrived LIVE during the attach."""

    async def test_history_replaced_rebuilds_after_undo(self) -> None:
        from xarness.tui.app import AgentApp

        messages = [
            _msg("user", "u1", "first"),
            _msg("assistant", "a1", "answer one"),
            _msg("user", "u2", "second"),
            _msg("assistant", "a2", "answer two"),
        ]
        snap = make_snapshot()
        object.__setattr__(snap, "messages", messages)
        client = FakeClient(snap)
        app = AgentApp(client, workspace=None, session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
            # A live turn adds a message the snapshot never contained.
            client.queue.put_nowait(protocol.UserMessageAdded(
                text="live", message_id="u3"))
            await pilot.pause()
            users = [e.item.message_id for e in app._flow_model.entries
                     if isinstance(e.item, tflow.UserItem)]
            self.assertEqual(users, ["u1", "u2", "u3"])

            # Undo-from-here on "second" (u2): u2, a2 and u3 are gone.
            trimmed = make_snapshot()
            object.__setattr__(trimmed, "messages", messages[:2])
            client.queue.put_nowait(protocol.HistoryReplaced(snapshot=trimmed))
            await pilot.pause()
            users = [e.item.message_id for e in app._flow_model.entries
                     if isinstance(e.item, tflow.UserItem)]
            self.assertEqual(users, ["u1"])


class DiffClickMinimizeTest(unittest.IsolatedAsyncioTestCase):
    """Clicking inside the open diff pane minimizes only the diff — the
    expanded file list (and its rows) stays up."""

    async def test_click_on_diff_pane_keeps_file_list(self) -> None:
        from textual.geometry import Offset

        from xarness.tui.app import AgentApp
        from xarness.tui.widgets import DiffSummary

        snap = make_snapshot()
        client = FakeClient(snap)
        app = AgentApp(client, workspace=None, session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
            summary = app.query_one("#diff-summary", DiffSummary)
            await summary.set_summary(make_diff_stat())
            summary.add_class("expanded")
            summary.show_diff("a.py", "--- a/a.py\n+++ b/a.py\n@@\n-x\n+y\n")
            await pilot.pause()
            self.assertTrue(summary.diff_shown)
            self.assertTrue(summary.has_class("expanded"))

            # Click in the middle of the diff text pane.
            content = summary.query_one("#diff-content")
            await pilot.click(content, offset=Offset(2, 1))
            await pilot.pause()
            self.assertFalse(summary.diff_shown)
            # The file list itself stays expanded and populated.
            self.assertTrue(summary.has_class("expanded"))
            self.assertEqual(len(summary.query(".diff-file-row")), 1)


def make_diff_stat():
    from xarness.protocol import DiffFileInfo, DiffInfo

    return DiffInfo(files=[
        DiffFileInfo(path="a.py", additions=2, deletions=1, is_new=False, source="agent"),
    ])


class DraftStatusTest(unittest.IsolatedAsyncioTestCase):
    """A draft chat (no session yet) shows the config's default model — not
    a dash — and Tab toggles a local draft mode that is applied to the
    session created by the first message."""

    async def test_draft_shows_default_model_and_tab_applies_mode(self) -> None:
        from xarness.tui.app import AgentApp

        snap = make_snapshot()
        client = FakeClient(snap)
        app = AgentApp(client, workspace=None, session_name=None)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            bar = app.query_one("#status-bar")
            # Default model visible before any session exists.
            self.assertIn("GLM Flash", str(bar.render()))
            self.assertIn("write", str(bar.render()))

            # Tab in draft mode: no session, so it must not send a command;
            # it records draft state.
            await pilot.press("tab")
            await pilot.pause()
            self.assertEqual(app._draft_mode, "plan")
            self.assertEqual(
                [(n, c) for n, c in client.commands if c.type == "set_mode"], [])

            # First message creates the session and carries the mode over.
            client.create_result = SessionSummary(
                name="s-new", state="idle", active=True)
            await pilot.press("enter")  # empty input: no-op
            app._submit("hello", None)
            await pilot.pause()
            await pilot.pause()
            client.queue.put_nowait(protocol.StatusChanged(status=StatusPayload(
                state="idle", session_name="s-new", workspace="/tmp/ws",
                mode="write", model_name="glm-flash",
                model_display_name="GLM Flash", model_id="glm-flash",
            )))
            await pilot.pause()
            await pilot.pause()
            self.assertEqual(app.session_name, "s-new")
            set_modes = [c for _, c in client.commands if c.type == "set_mode"]
            self.assertEqual([c.mode for c in set_modes], ["plan"])
            # Draft state is spent.
            self.assertEqual(app._draft_mode, "write")

    async def test_draft_model_switch_updates_status_before_message(self) -> None:
        """Picking a model in a draft chat repaints the status bar right
        away (and remembers the choice for the first message), rather than
        leaving the old default until a message creates the session."""
        from xarness.tui.app import AgentApp

        snap = make_snapshot()
        client = FakeClient(snap)

        async def config():
            return protocol.ConfigInfo(
                models=[
                    {"name": "glm-flash", "display_name": "GLM Flash"},
                    {"name": "other", "display_name": "Other Model",
                     "cot_strength": "high", "max_context": 200_000},
                ],
                sandboxes=[{"name": "default"}],
                presets=["default"], themes=[],
                default_model="glm-flash",
                default_model_display_name="GLM Flash",
            )

        client.config = config
        # config_path=None keeps the preference save off the real config file.
        app = AgentApp(client, workspace=None, session_name=None, config_path=None)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            bar = app.query_one("#status-bar")
            self.assertIn("GLM Flash", str(bar.render()))

            async def pick(names, current=None, title=""):
                return "other"

            app._pick = pick
            worker = app._pick_model()
            await worker.wait()
            await pilot.pause()

            # No session exists, so nothing was sent to the host...
            self.assertEqual(app.session_name, None)
            self.assertEqual(
                [c for _, c in client.commands if c.type == "switch_model"], [])
            # ...but the header already shows the picked model, and the
            # first message will create the session on it.
            self.assertEqual(app.model, "other")
            self.assertIn("Other Model", str(bar.render()))


class ReplayShimmerAndTimingTest(unittest.IsolatedAsyncioTestCase):
    """Reloaded history must mount no shimmer at all (the block arrives
    settled), and the persisted per-turn timing line must render."""

    def _snapshot_with_history(self) -> Snapshot:
        snap = make_snapshot()
        snap.messages = [
            protocol.Message(
                role="user", id="u1", content="do it",
                timing=Timing(send="2025-01-01T00:00:00+00:00"),
            ),
            protocol.Message(
                role="assistant", id="a2", content="",
                reasoning="step one\nstep two",
                timing=Timing(reasoning_seconds=4.0),
                tool_calls=[{"call_id": "c1", "name": "read_file",
                             "arguments_json": "{}"}],
            ),
            protocol.Message(
                role="tool", id="t1", tool_call_id="c1", content="file body",
            ),
            protocol.Message(
                role="assistant", id="a1", content="Done.",
                timing=Timing(
                    start="2025-01-01T00:00:01+00:00",
                    end="2025-01-01T00:00:05+00:00",
                ),
            ),
        ]
        return snap

    async def test_hidden_epochs_from_compaction_still_render(self) -> None:
        """The snapshot's ``history`` (older compaction epochs) is folded
        before the current messages, so resuming shows the whole session."""
        from xarness.conversation import SUMMARY_PREFIX
        from xarness.tui.app import AgentApp
        from xarness.tui.transcript import DividerItem, UserItem

        snap = make_snapshot()
        snap.history = [
            protocol.Message(role="user", id="u0", content="first question"),
            protocol.Message(
                role="user", id="s0", kind="summary",
                content=f"{SUMMARY_PREFIX}\n\nthe handoff",
            ),
        ]
        snap.messages = [
            protocol.Message(role="assistant", id="a0", content="after"),
        ]
        app = AgentApp(FakeClient(snap), workspace=None, session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            items = [entry.item for entry in app._flow_model.entries]
            self.assertIsInstance(items[0], UserItem)
            self.assertEqual(items[0].text, "first question")
            self.assertIsInstance(items[1], DividerItem)
            self.assertIn("the handoff", items[1].text)
            self.assertEqual(items[2].chunks, ["after"])

    async def test_replay_has_no_shimmer_and_shows_timing(self) -> None:
        from xarness.tui.app import AgentApp

        client = FakeClient(self._snapshot_with_history())
        app = AgentApp(client, workspace=None, session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            flow = app.query_one("#chat-log")
            # The tool result is paired with its call: succeeded, not failed.
            tools = [e.item for e in app._flow_model.entries
                     if isinstance(e.item, tflow.ToolItem)]
            self.assertEqual(len(tools), 1)
            self.assertEqual(tools[0].status, ToolStatus.CALL_SUCCEEDED)
            # The settled summary shows the tool name, not "Running read_file".
            all_text = "\n".join(flow.row_text(y) for y in range(flow.row_count))
            self.assertIn("read_file", all_text)
            self.assertNotIn("Running read_file", all_text)
            # The replayed CoT block shows its "Thought for Xs" pill.
            thinking = [e.item for e in app._flow_model.entries
                        if isinstance(e.item, tflow.ThinkingItem)]
            self.assertEqual(len(thinking), 1)
            self.assertTrue(thinking[0].done)
            self.assertIn("Thought for", all_text)
            self.assertIn("Worked for 5.0s | 4.0s 1.0s", all_text)


class FlowViewTest(unittest.IsolatedAsyncioTestCase):
    """The virtualized transcript: the WHOLE history lives in the FlowModel
    (one FlowView widget regardless of length — that is the point); only the
    visible rows are painted, so a 1k+-message session costs the same as a
    fresh one."""

    @staticmethod
    def _big_snapshot(count: int = 1200) -> Snapshot:
        from xarness.conversation import Message as Msg
        snap = make_snapshot()
        snap.messages = [
            Msg(role="user" if i % 2 == 0 else "assistant",
                content=f"body {i}", id=f"m{i}")
            for i in range(count)
        ]
        return snap

    async def test_full_history_in_model_one_widget(self) -> None:
        from textual_flowview import FlowView

        from xarness.tui.app import AgentApp

        app = AgentApp(FakeClient(self._big_snapshot()), workspace=None,
                       session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            self.assertEqual(len(list(app.query(FlowView))), 1)
            self.assertEqual(len(app._flow_model), 1200)
            # The tail is painted: the last user message is visible text.
            flow = app.query_one("#chat-log", FlowView)
            tail = "\n".join(
                flow.row_text(y) for y in range(max(0, flow.row_count - 12),
                                                flow.row_count)
            )
            self.assertIn("body 1198", tail)

    async def test_presentations_stay_viewport_bounded(self) -> None:
        # O(viewport) is the whole claim: after loading 1200 entries, only
        # the visible band (+ overscan) has ever been presented.
        from textual_flowview import FlowView

        from xarness.tui.app import AgentApp

        app = AgentApp(FakeClient(self._big_snapshot()), workspace=None,
                       session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            flow = app.query_one("#chat-log", FlowView)
            self.assertLess(len(flow._layout), 100)


class TextSelectionTest(unittest.IsolatedAsyncioTestCase):
    """Mouse drag over the transcript must both highlight and extract text.

    The presenter bakes an explicit palette bg into every cell (bg-less
    cells would fall through to the terminal's default colors), and rich's
    style merge lets cell styles win over the lib's own ``_decorate_line``
    selection pass — so ChatFlowView.render_line re-applies the selection
    span with cell styles overridden. This test guards that override."""

    async def test_drag_selects_and_highlights_transcript_text(self) -> None:
        from xarness import theme
        from xarness.tui.app import AgentApp
        from xarness.tui.flow import UserItem

        app = AgentApp(FakeClient(make_snapshot()), workspace=None,
                       session_name="s1")
        async with app.run_test(size=(80, 24)) as pilot:
            for _ in range(5):
                await pilot.pause()
            app._flow_model.append(UserItem(text="selectable body text"))
            for _ in range(10):
                await pilot.pause()
            view = app.query_one("#chat-log")
            # Drag across the row that actually carries the user text
            # (spacer/separator rows between entries aren't selectable).
            row = next(y for y in range(view.row_count)
                       if "selectable body text" in view.row_text(y))
            await pilot.mouse_down(offset=(10, row))
            await pilot.mouse_up(offset=(40, row))
            await pilot.pause()
            # Text is extractable for the clipboard (the drag starts a few
            # columns into the line, so the first word is clipped).
            selected = app.screen.get_selected_text()
            self.assertIn("body text", selected or "")
            # And the drag paints the theme's selection highlight.
            highlight = theme.PALETTE["highlight"]
            highlighted = any(
                seg.style is not None and seg.style.bgcolor is not None
                and seg.style.bgcolor.get_truecolor("#000").hex == highlight
                for seg in view.render_line(row)
            )
            self.assertTrue(
                highlighted,
                f"no cell on row 5 carries the selection highlight {highlight}",
            )




class DiffSummaryListTest(unittest.IsolatedAsyncioTestCase):
    """The pending-changes panel is a row of per-file cells, matching the
    pre-refactor layout (name flexes, counts sit in their own right-hand
    columns, source/new are separate badges)."""

    async def test_set_summary_lists_files(self) -> None:
        from textual.widgets import Static

        from xarness.protocol import DiffFileInfo, DiffInfo
        from xarness.tui.app import AgentApp
        from xarness.tui.widgets import DiffSummary

        app = AgentApp(FakeClient(make_snapshot()), workspace=None, session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
            summary = app.query_one("#diff-summary", DiffSummary)

            await summary.set_summary(DiffInfo(files=[
                DiffFileInfo(path="a.py", additions=9, deletions=3, source="drift"),
                DiffFileInfo(path="b.py", additions=1, deletions=0, is_new=True),
            ]))
            rows = list(summary.query(".diff-file-row"))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0].diff_key, "a.py")
            self.assertEqual(rows[0].query_one(".diff-file-name", Static).render().plain, "a.py")
            self.assertEqual(
                rows[0].query_one(".diff-file-source", Static).render().plain, "drift"
            )
            self.assertTrue(rows[1].query_one(".diff-file-new"))
            # Counts are separate cells so CSS can right-align them.
            self.assertEqual(
                rows[0].query_one(".diff-file-add", Static).render().plain, "+9"
            )
            self.assertEqual(
                rows[0].query_one(".diff-file-del", Static).render().plain, "-3"
            )


class ShimmerDirectionTest(unittest.TestCase):
    """The sweep must start at the base color and peak at the highlight.

    The lerp helper the refactor reused weights its *first* argument, so
    passing (base, peak) here silently inverted every shimmer."""

    def test_peak_band_is_the_highlight_not_the_base(self) -> None:
        from xarness import theme
        from xarness.tui import flow

        label = "x" * 40
        text = flow.shimmer_text(
            label, "#101010", "#f0f0f0", phase=theme.SHIMMER_BAND_WIDTH,
        )
        colors = {
            span.start: span.style.color.get_truecolor().hex  # type: ignore[union-attr]
            for span in text.spans
        }
        self.assertEqual(colors[0], "#f0f0f0")  # band centre
        self.assertEqual(colors[len(label) - 1], "#101010")  # resting text

    def test_palette_pairs_put_the_highlight_second(self) -> None:
        from xarness import theme

        self.assertEqual(
            theme.SHIMMER_PROCESSING,
            (theme.PALETTE["muted"], theme.PALETTE["text"]),
        )
        self.assertEqual(
            theme.SHIMMER_THINKING,
            (theme.PALETTE["muted"], theme.PALETTE["surface"]),
        )


class UndoCommandTest(unittest.IsolatedAsyncioTestCase):
    async def test_plain_undo_puts_the_user_text_back_in_the_input(self) -> None:
        from xarness.tui.app import AgentApp
        from xarness.tui.widgets import ChatInput

        client = FakeClient(make_snapshot())
        app = AgentApp(client, workspace=None, session_name="s1")
        async with app.run_test() as pilot:
            await pilot.pause()
            await app._run_undo()
            await pilot.pause()
            self.assertEqual(app.query_one("#chat-input", ChatInput).text, "put back")
            self.assertIn("undo", [command.type for _, command in client.commands])


class RelativeTimeFormatTest(unittest.TestCase):
    def test_days_are_not_rendered_as_hours(self) -> None:
        from datetime import datetime, timedelta, timezone

        from xarness.tui.list_picker import relative_time

        now = datetime.now(timezone.utc)
        self.assertEqual(relative_time((now - timedelta(days=2)).isoformat()), "2d")
        self.assertEqual(relative_time((now - timedelta(hours=3)).isoformat()), "3h00m")
        self.assertEqual(relative_time(None), "?")
