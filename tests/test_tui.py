"""TUI tests: the Ctrl+C headless modal, no-local-echo rendering, and the
client/server import boundary. The app runs against a fake clientlib
client; the session behaviors themselves are covered in test_session.py."""

import asyncio
import base64
import hashlib
import json
import unittest
from contextlib import asynccontextmanager

from textual.app import App
from textual.pilot import Pilot

from xarness import protocol
from xarness.clientlib import ClientError
from xarness.conversation import (
    AssistantMessage, TextPart, Timing, ToolMessage, TurnStats, UserMessage,
    Usage,
)
from xarness.protocol import (
    CommandResult,
    ConfigInfo,
    DiffInfo,
    Event,
    SessionSummary,
    Snapshot,
    Status,
)
from xarness.tui import flow as tflow
from xarness.tui.transcript import ToolStatus


@asynccontextmanager
async def run_app(app, size=None):
    """Run ``app`` and hand back a pilot with the first frame settled — what
    nearly every test does before it asserts anything."""
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        yield pilot


def _model_info() -> dict:
    """Full ModelInfo fields (ConfigInfo.models is typed now)."""
    return {"cot_strength": "medium", "max_context": 128_000,
            "supports_vision": False, "auto_compact": True}


class FakeClient:
    """Scripted stand-in for XarnessClient: serves a snapshot and a live
    event queue; records commands."""

    def __init__(self, snapshot: Snapshot) -> None:
        self.snapshot_data = snapshot
        self.queue: asyncio.Queue = asyncio.Queue()
        self.commands: list[tuple[str, Event]] = []
        self.command_results: dict[str, CommandResult] = {}
        self.unloaded: list[str] = []
        self.session_calls: list[dict] = []
        self.refs: list[protocol.RefInfo] = []
        self.ref_updates: list[tuple[str, str, bool]] = []
        # The host's answer to "does closing need a choice?". Scripted per
        # test: the policy is the host's (session.quit_verdict), never the
        # TUI's, so the default is "nothing to ask".
        self.quit_verdict_result = protocol.QuitVerdict(ask=False)
        self.quit_verdict_error: Exception | None = None
        self.uploads: list[tuple[str, bytes, str]] = []
        # The host's answer to /tools (scripted per test).
        self.tools_info = protocol.ToolsInfo()

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
            command.type, CommandResult(user_text="put back"),
        )

    async def upload_image(self, name: str, data: bytes, image_id: str) -> None:
        self.uploads.append((name, data, image_id))

    async def sessions(self, **kwargs):
        self.session_calls.append(kwargs)
        return list(getattr(self, "session_list", []))

    async def config(self):
        return protocol.ConfigInfo(
            models=[dict(_model_info(), name="glm-flash",
                         display_name="GLM Flash")],
            sandboxes=[{"name": "default"}],
            presets=["default"], themes=[],
            default_model="glm-flash",
            default_model_display_name="GLM Flash",
        )

    async def diff(self, name: str, path: str | None = None) -> DiffInfo:
        return DiffInfo()

    async def tools(self, name: str) -> protocol.ToolsInfo:
        return self.tools_info

    async def search_files(self, workspace: str, q: str) -> list[str]:
        return []

    async def workspace_refs(self, workspace: str) -> list[protocol.RefInfo]:
        return list(self.refs)

    async def update_workspace_refs(
        self, workspace: str, path: str, add: bool,
    ) -> list[protocol.RefInfo]:
        self.ref_updates.append((workspace, path, add))
        if add:
            self.refs.append(protocol.RefInfo(path=path, mount=f"/tmp/refs/{path}"))
        else:
            self.refs = [ref for ref in self.refs if ref.path != path]
        return list(self.refs)

    async def create_session(self, request) -> SessionSummary:
        return getattr(self, "create_result", None) or SessionSummary(
            name="new-session", active=True)

    async def set_default_profile(self, name: str) -> ConfigInfo:
        return ConfigInfo(default_model=name)

    async def delete_session(self, name: str) -> CommandResult:
        return CommandResult()

    async def unload_session(self, name: str) -> bool:
        # Quitting must not unload anything: a session is the host's, and the
        # quit tests assert this stays empty.
        self.unloaded.append(name)
        return True

    async def quit_verdict(self, name: str) -> protocol.QuitVerdict:
        if self.quit_verdict_error is not None:
            raise self.quit_verdict_error
        return self.quit_verdict_result

    async def aclose(self) -> None:
        pass


def make_app(client=None, session_name="s1", **kwargs):
    """An AgentApp over a scripted client, with no workspace to attach to."""
    from xarness.tui.app import AgentApp

    return AgentApp(
        client if client is not None else FakeClient(make_snapshot()),
        workspace=None, session_name=session_name, **kwargs,
    )


def make_snapshot(state: str = "idle") -> Snapshot:
    return Snapshot(
        messages=[],
        status=Status(
            state=state,  # type: ignore[arg-type]
            session_name="s1", workspace="/tmp/ws", mode="write",
            model_name="m", model_display_name="m", model_id="m",
        ),
    )


def running_status_event() -> Event:
    return protocol.StatusChanged(status=Status(state="running"))


class ToolsScreenTest(unittest.IsolatedAsyncioTestCase):
    def tools_info(self) -> protocol.ToolsInfo:
        return protocol.ToolsInfo(
            tools=[
                protocol.ToolInfo(
                    name="read_file", description="Read a file",
                ),
                protocol.ToolInfo(
                    name="mcp__fs__cat", source="fs",
                    description="Read through MCP",
                ),
            ],
            servers=[protocol.McpServerInfo(
                name="fs", command="npx", args=["pkg"],
                status="connected",
                tools=[protocol.ToolInfo(
                    name="cat", source="fs", description="Read through MCP",
                )],
            )],
        )

    async def test_tools_modal_lists_sources_and_posts_toggles(self) -> None:
        from textual.widgets import OptionList

        from xarness.tui.tools_screen import ToolsScreen

        client = FakeClient(make_snapshot())
        client.tools_info = self.tools_info()
        app = make_app(client)
        async with run_app(app) as pilot:
            app._open_tools()
            await pilot.pause()
            await pilot.pause()
            self.assertIsInstance(app.screen, ToolsScreen)
            screen = app.screen
            top = screen.query_one("#tools-list")
            bottom = screen.query_one("#mcp-list", OptionList)
            blocks = top.query(".tools-tool")
            self.assertEqual(len(blocks), 2)
            self.assertIn(
                "read_file",
                str(blocks[0].query_one(".tools-tool-name").render()),
            )
            self.assertIn(
                "mcp · fs",
                str(blocks[1].query_one(".tools-tool-name").render()),
            )
            self.assertIn(
                "Read a file",
                str(blocks[0].query_one(".tools-tool-description").render()),
            )
            self.assertIn(
                "Read through MCP",
                str(blocks[1].query_one(".tools-tool-description").render()),
            )
            self.assertEqual(
                [option.id for option in bottom.options],
                ["server:fs", "tool:fs:cat"],
            )
            # Clicking a row posts the command the app sends to the host.
            bottom.focus()
            bottom.action_cursor_down()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            self.assertEqual(
                [(command.type, command.scope)
                 for _, command in client.commands],
                [("set_mcp_server_scope", "workspace")],
            )

    async def test_tools_modal_shows_and_cycles_a_server_scope(self) -> None:
        """The server header names its scope; a click asks for the next."""
        from textual.widgets import OptionList

        from xarness.tui.tools_screen import ToolsScreen

        client = FakeClient(make_snapshot())
        client.tools_info = self.tools_info()
        app = make_app(client)
        async with run_app(app) as pilot:
            app._open_tools()
            await pilot.pause()
            await pilot.pause()
            self.assertIsInstance(app.screen, ToolsScreen)
            bottom = app.screen.query_one("#mcp-list", OptionList)
            self.assertIn("on (global)", bottom.options[0].prompt.plain)
            bottom.focus()
            bottom.action_cursor_down()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            self.assertEqual(
                [(command.type, command.scope)
                 for _, command in client.commands],
                [("set_mcp_server_scope", "workspace")],
            )

    async def test_tools_modal_shows_a_url_server_and_adds_by_url(self) -> None:
        from textual.widgets import Input, OptionList

        from xarness.tui.tools_screen import ToolsScreen

        client = FakeClient(make_snapshot())
        client.tools_info = protocol.ToolsInfo(
            tools=[protocol.ToolInfo(
                name="mcp__exa__search", source="exa",
                description="Search the web",
            )],
            servers=[protocol.McpServerInfo(
                name="exa", url="https://mcp.exa.ai/mcp",
                status="connected", available=True,
                tools=[protocol.ToolInfo(
                    name="search", source="exa", description="Search the web",
                )],
            )],
        )
        app = make_app(client)
        async with run_app(app) as pilot:
            app._open_tools()
            for _ in range(3):
                await pilot.pause()
            screen = app.screen
            self.assertIsInstance(screen, ToolsScreen)
            bottom = screen.query_one("#mcp-list", OptionList)
            header = str(bottom.get_option_at_index(0).prompt)
            self.assertIn("https://mcp.exa.ai/mcp", header)
            self.assertNotIn("None", header)

            # The add field is an inset box, not panel-colored like the
            # surface it sits on.
            add = screen.query_one("#tools-add", Input)
            self.assertNotEqual(
                add.styles.background,
                screen.query_one("#tools-modal").styles.background,
            )

            add.focus()
            add.value = "https://mcp.exa.ai/mcp"
            await pilot.press("enter")
            for _ in range(3):
                await pilot.pause()
            added = [
                command for _, command in client.commands
                if command.type == "add_mcp_server"
            ]
            self.assertEqual(len(added), 1)
            self.assertEqual(added[0].url, "https://mcp.exa.ai/mcp")
            self.assertIsNone(added[0].command)
            self.assertEqual(added[0].name, "exa-ai")

    async def test_tools_requires_a_session(self) -> None:
        from xarness.tui.tools_screen import ToolsScreen

        client = FakeClient(make_snapshot())
        app = make_app(client, session_name=None)
        async with run_app(app) as pilot:
            app._open_tools()
            await pilot.pause()
            self.assertNotIsInstance(app.screen, ToolsScreen)


class CtrlCModalTest(unittest.IsolatedAsyncioTestCase):
    async def _run(self, snapshot: Snapshot, script):
        client = FakeClient(snapshot)
        app = make_app(client)
        async with run_app(app) as pilot:
            await script(pilot, client, app)

    async def test_running_ctrl_c_shows_modal_and_continue_headless_exits(self) -> None:
        async def script(pilot: Pilot, client: FakeClient, app):
            client.quit_verdict_result = protocol.QuitVerdict(
                ask=True, state="running",
            )
            client.queue.put_nowait(running_status_event())
            await pilot.pause()
            await pilot.press("ctrl+c")
            await pilot.pause()
            from xarness.tui.confirm_screen import ConfirmScreen
            self.assertIsInstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            await pilot.pause()
            self.assertFalse(app.is_running)
            self.assertEqual(
                [c for _, c in client.commands if c.type == "cancel"], [],
            )

        await self._run(make_snapshot(), script)

    async def test_running_ctrl_c_stop_and_quit_cancels_then_exits(self) -> None:
        async def script(pilot: Pilot, client: FakeClient, app):
            client.quit_verdict_result = protocol.QuitVerdict(
                ask=True, state="running",
            )
            client.queue.put_nowait(running_status_event())
            await pilot.pause()
            await pilot.press("ctrl+c")
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()
            cancels = [c for _, c in client.commands if c.type == "cancel"]
            self.assertEqual(len(cancels), 1)
            # ...and nothing unloads the session: quitting is a disconnect,
            # the host owns the session's lifetime.
            self.assertEqual(client.unloaded, [])
            self.assertFalse(app.is_running)

        await self._run(make_snapshot(), script)

    async def test_pending_ask_uses_the_answer_specific_question(self) -> None:
        async def script(pilot: Pilot, client: FakeClient, app):
            client.quit_verdict_result = protocol.QuitVerdict(
                ask=True, state="running", pending_ask=True,
            )
            client.queue.put_nowait(running_status_event())
            await pilot.pause()
            await pilot.press("ctrl+c")
            await pilot.pause()
            from textual.widgets import Label

            from xarness.tui.confirm_screen import ConfirmScreen
            self.assertIsInstance(app.screen, ConfirmScreen)
            title = app.screen.query_one("#confirm-title", Label).render()
            self.assertIn("waiting for your answer", str(title))
            await pilot.press("y")
            await pilot.pause()
            self.assertFalse(app.is_running)

        await self._run(make_snapshot(), script)

    async def test_running_ctrl_c_escape_stays_in_the_app(self) -> None:
        async def script(pilot: Pilot, client: FakeClient, app):
            client.quit_verdict_result = protocol.QuitVerdict(
                ask=True, state="running",
            )
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

    async def test_unreachable_host_quits_without_a_modal(self) -> None:
        """No verdict (host gone, timeout, older host): quit, don't block."""
        async def script(pilot: Pilot, client: FakeClient, app):
            client.quit_verdict_error = ClientError("connection refused")
            client.queue.put_nowait(running_status_event())
            await pilot.pause()
            await pilot.press("ctrl+c")
            await pilot.pause()
            from xarness.tui.confirm_screen import ConfirmScreen
            self.assertNotIsInstance(app.screen, ConfirmScreen)
            self.assertFalse(app.is_running)
            self.assertEqual(
                [c for _, c in client.commands if c.type == "cancel"], [],
            )

        await self._run(make_snapshot(), script)


class NoLocalEchoTest(unittest.IsolatedAsyncioTestCase):
    async def test_user_message_reaches_the_log_only_via_the_event_stream(self) -> None:

        client = FakeClient(make_snapshot())
        app = make_app(client)
        async with run_app(app) as pilot:
            # Nothing in the model yet: the client does not echo its own sends.
            self.assertEqual(
                [type(e.item).__name__ for e in app._flow_model.entries
                 if type(e.item).__name__ == "UserItem"], [])
            client.queue.put_nowait(protocol.MessageAdded(
                message=UserMessage(
                    parts=[TextPart(text="hello")], id="m1",
                    timing=Timing(send="2026-01-01T00:00:00+00:00"),
                ),
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
        from xarness.tui.widgets import SteerQueueBar

        client = FakeClient(make_snapshot())
        app = make_app(client)
        async with run_app(app) as pilot:
            client.queue.put_nowait(protocol.StatusChanged(status=Status(
                state="running",  # type: ignore[arg-type]
                session_name="s1", workspace="/tmp/ws", mode="write",
                model_name="m", model_display_name="m", model_id="m",
                queued=[protocol.QueuedMessage(text="steer!")],
            )))
            await pilot.pause()
            bar = app.query_one("#steer-queue-bar", SteerQueueBar)
            self.assertEqual(len(bar.query(".steer-item")), 1)
            self.assertTrue(bar.has_class("visible"))


class ImageUploadTest(unittest.IsolatedAsyncioTestCase):
    """An attached image is uploaded before the send; ImageAttachment is an
    ImageRef subclass, so the ref fast path must not swallow it."""

    @staticmethod
    def _attachment() -> "ImageAttachment":
        from xarness.images import ImageAttachment

        data = b"cat"
        return ImageAttachment(
            id=hashlib.sha256(data).hexdigest(), name="clipboard.png",
            mime="image/png", data_b64=base64.b64encode(data).decode(),
            width=8, height=8,
        )

    async def test_attachment_is_uploaded_and_sent_as_a_ref(self) -> None:
        from xarness.images import ImageRef, token_text

        client = FakeClient(make_snapshot())
        app = make_app(client)
        attachment = self._attachment()
        async with run_app(app) as pilot:
            app._submit(f"look {token_text(attachment)}", [attachment])
            await pilot.pause()

        self.assertEqual(
            client.uploads, [("s1", b"cat", attachment.id)])
        sends = [c for _, c in client.commands if c.type == "send_message"]
        self.assertEqual(len(sends), 1)
        refs = sends[0].images or []
        self.assertEqual(len(refs), 1)
        # The wire carries the ref shape only: no base64 payload.
        self.assertIs(type(refs[0]), ImageRef)
        self.assertEqual(refs[0].id, attachment.id)
        self.assertFalse(hasattr(refs[0], "data_b64"))

    async def test_a_restored_ref_is_not_uploaded_again(self) -> None:
        from xarness.images import ImageRef, token_text

        client = FakeClient(make_snapshot())
        app = make_app(client)
        ref = ImageRef(
            id=hashlib.sha256(b"cat").hexdigest(), name="clipboard.png",
            mime="image/png", width=8, height=8,
        )
        async with run_app(app) as pilot:
            app._submit(f"look {token_text(ref)}", [ref])
            await pilot.pause()

        self.assertEqual(client.uploads, [])
        sends = [c for _, c in client.commands if c.type == "send_message"]
        self.assertEqual([i.id for i in (sends[0].images or [])], [ref.id])


class FailedSendRestoresInputTest(unittest.IsolatedAsyncioTestCase):
    """A message that never reached the session goes back in the input, so
    a rejected/failed send does not lose what the user wrote."""

    @staticmethod
    def _input(app):
        from xarness.tui.chat_input import ChatInput

        return app.query_one("#chat-input", ChatInput)

    @staticmethod
    def _fail_command(client):
        async def command(name, cmd):
            client.commands.append((name, cmd))
            raise ClientError("send rejected")

        client.command = command

    async def test_text_is_put_back_when_the_send_fails(self) -> None:
        client = FakeClient(make_snapshot())
        app = make_app(client)
        self._fail_command(client)
        async with run_app(app) as pilot:
            # The input is cleared on submit, as by a real Enter press.
            self._input(app).load_text("")
            app._submit("hello there")
            await pilot.pause()
            self.assertEqual(self._input(app).text, "hello there")

    async def test_attachments_come_back_with_the_text(self) -> None:
        from xarness.images import token_text

        attachment = ImageUploadTest._attachment()
        client = FakeClient(make_snapshot())
        app = make_app(client)
        self._fail_command(client)
        async with run_app(app) as pilot:
            app._submit(f"look {token_text(attachment)}", [attachment])
            await pilot.pause()
            chat_input = self._input(app)
            self.assertEqual(chat_input.text, f"look {token_text(attachment)}")
            self.assertEqual([a.id for a in chat_input.attachments], [attachment.id])

    async def test_a_new_draft_is_not_clobbered(self) -> None:
        client = FakeClient(make_snapshot())
        app = make_app(client)

        async def command(name, cmd):
            # Typed after submitting but before the failure lands.
            self._input(app).load_text("next thought")
            raise ClientError("send rejected")

        client.command = command
        async with run_app(app) as pilot:
            app._submit("hello there")
            await pilot.pause()
            self.assertEqual(self._input(app).text, "next thought")

    async def test_failed_session_creation_puts_the_text_back(self) -> None:
        client = FakeClient(make_snapshot())
        app = make_app(client, session_name=None)

        async def create_session(request):
            raise ClientError("no host")

        client.create_session = create_session
        async with run_app(app) as pilot:
            app._submit("hello there")
            await pilot.pause()
            self.assertEqual(self._input(app).text, "hello there")


class WordmarkTest(unittest.IsolatedAsyncioTestCase):
    """The wordmark marks an empty session and yields to the first message."""

    @staticmethod
    def _text(app) -> str:
        view = app.query_one("#chat-log")
        return "\n".join(
            view.render_line(y).text for y in range(view.size.height)
        )

    async def test_empty_session_shows_it_and_first_message_clears_it(self) -> None:

        client = FakeClient(make_snapshot())
        app = make_app(client)
        async with run_app(app) as pilot:
            self.assertIn("... ...", self._text(app))

            client.queue.put_nowait(protocol.MessageAdded(
                message=UserMessage(
                    parts=[TextPart(text="hello")], id="m1",
                    timing=Timing(send="2026-01-01T00:00:00+00:00"),
                ),
            ))
            await pilot.pause()
            self.assertNotIn("... ...", self._text(app))

    async def test_new_session_brings_it_back(self) -> None:

        snap = make_snapshot()
        snap.messages.append(UserMessage(
            parts=[TextPart(text="hello")], id="m1",
            timing=Timing(send="2026-01-01T00:00:00+00:00"),
        ))
        client = FakeClient(snap)
        app = make_app(client)
        async with run_app(app) as pilot:
            self.assertNotIn("... ...", self._text(app))

            app._start_new_session()
            await pilot.pause()
            self.assertIn("... ...", self._text(app))


class ImportBoundaryTest(unittest.TestCase):
    """The TUI package imports only clientlib, protocol and pure helpers —
    never the session, tools, gitwork, sandbox or host modules."""

    BANNED = (
        "xarness.session", "xarness.tools",
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
    """On connection loss the TUI re-attaches: a reconnect delivers a fresh
    snapshot, and the client respawns a dead host on the next request."""

    async def test_connection_loss_reattaches(self) -> None:
        from xarness.tui import app as app_module

        client = FakeClient(make_snapshot())
        client.fail_attach_first = True
        attaches: list[str] = []
        real_attach = client.attach

        def attach(name: str):
            attaches.append(name)
            return real_attach(name)

        client.attach = attach
        previous = app_module.RECONNECT_MIN
        app_module.RECONNECT_MIN = 0.01
        try:
            app = make_app(client)
            async with run_app(app) as pilot:
                # The loop backs off before retrying the stream.
                await asyncio.sleep(app_module.RECONNECT_MIN + 0.02)
                await pilot.pause()
        finally:
            app_module.RECONNECT_MIN = previous

        self.assertEqual(attaches, ["s1", "s1"])


class _WidgetApp(App):
    """Bare app that loads the real stylesheet for widget-level tests."""


class ThinkingDurationTest(unittest.IsolatedAsyncioTestCase):
    """The duration belongs to the cot block alone: it is stamped when the
    block ends (first answer token / first tool call), not when the round
    ends, and it is never overwritten afterwards."""

    def test_unknown_duration_is_backfilled_once(self) -> None:
        # A block settled without a stamp (mid-round attach, interrupted
        # round) shows "…" until a stamped settle arrives.
        thinking = tflow.ThinkingItem(parts=["hmm"])
        thinking.finish(None)
        self.assertEqual(thinking.summary_text(), "Thought for …")
        thinking.finish(7.0)
        self.assertEqual(thinking.summary_text(), "Thought for 7.0s")
        thinking.finish(99.0)  # a later event must not extend it
        self.assertEqual(thinking.summary_text(), "Thought for 7.0s")

    def test_streaming_expands_then_finish_collapses_and_click_toggles(self) -> None:
        # Streaming: expanded by default, and clickable mid-stream.
        thinking = tflow.ThinkingItem(parts=["hmm"])
        self.assertTrue(thinking.expanded)
        thinking.expanded = False  # user collapsed it mid-stream
        thinking.expanded = True   # ...and reopened it — both allowed
        # Round end: auto-collapse, whatever the user's last toggle was.
        thinking.finish(2.0)
        self.assertFalse(thinking.expanded)
        # A later duration override (a stray settle) must not re-collapse an
        # item the user re-expanded after it finished.
        thinking.expanded = True
        thinking.finish(3.0)
        self.assertTrue(thinking.expanded)


class CommandPathsTest(unittest.IsolatedAsyncioTestCase):
    """Fire-and-forget command paths must actually send: bare (un-awaited)
    coroutine calls silently did nothing — esc-esc cancel, steer 'send now'
    and discard were all dead."""

    async def test_interrupt_and_steer_commands_are_sent(self) -> None:
        from xarness.protocol import QueuedMessage

        client = FakeClient(make_snapshot("running"))
        app = make_app(client)
        async with run_app(app) as pilot:
            client.queue.put_nowait(running_status_event())
            await pilot.pause()
            # Double-esc interrupts the running turn.
            await pilot.press("escape")
            await pilot.press("escape")
            await pilot.pause()
            # Empty Enter with queued messages flushes them now.
            assert app._status is not None
            app._status.queued = [QueuedMessage(text="steer")]
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
        from xarness.tui.widgets import ChatInput

        client = FakeClient(make_snapshot())
        app = make_app(client)
        async with run_app(app) as pilot:
            app._accept_popup_value("filter", was_slash=True)
            await pilot.pause()
            chat_input = app.query_one("#chat-input", ChatInput)
            self.assertEqual(chat_input.text, "/filter ")
            self.assertEqual(client.commands, [])

    async def test_ctrl_c_skips_headless_prompt_when_streamed_elsewhere(self) -> None:
        """The turn also streams to another client, so this quit has nothing
        to decide and must not offer "Stop and quit"."""
        from xarness.tui.confirm_screen import ConfirmScreen

        client = FakeClient(make_snapshot("running"))
        client.quit_verdict_result = protocol.QuitVerdict(
            ask=False, state="running", other_clients=1,
        )
        app = make_app(client)
        async with run_app(app) as pilot:
            status = running_status_event().status.model_copy(
                update={"clients_attached": 2},
            )
            client.queue.put_nowait(protocol.StatusChanged(status=status))
            await pilot.pause()
            await pilot.press("ctrl+c")
            await pilot.pause()
            self.assertNotIsInstance(app.screen, ConfirmScreen)
            self.assertFalse(app.is_running)
            self.assertEqual(
                [c for _, c in client.commands if c.type == "cancel"], [],
            )
            self.assertEqual(client.unloaded, [])


class AcceptRejectConfirmTest(unittest.IsolatedAsyncioTestCase):
    """`/accept` never asks (its notice reports the foreign files); `/reject`
    asks first when drift — the user's own edits — would be thrown away with
    it."""

    def _app(self):

        client = FakeClient(make_snapshot())
        return client, make_app(client)

    @staticmethod
    def _diff(*kinds: str) -> DiffInfo:
        return DiffInfo(files=[
            protocol.DiffFileInfo(path=f"f{i}.py", kind=kind)
            for i, kind in enumerate(kinds)
        ])

    async def test_accept_of_foreign_rows_posts_without_a_modal(self) -> None:
        """Other agents' and bash rows are accepted silently: the host's
        notice reports the breakdown, so there is nothing to confirm."""
        from xarness.tui.confirm_screen import ConfirmScreen

        client, app = self._app()
        async with run_app(app) as pilot:
            app._last_diff = self._diff("mine", "agent", "auto")
            app._accept()
            await pilot.pause()
            self.assertNotIsInstance(app.screen, ConfirmScreen)
            await pilot.pause()
            self.assertEqual(
                [c.type for _, c in client.commands], ["accept"], client.commands,
            )

    async def test_accept_of_own_rows_posts_without_a_modal(self) -> None:
        from xarness.tui.confirm_screen import ConfirmScreen

        client, app = self._app()
        async with run_app(app) as pilot:
            app._last_diff = self._diff("mine", "mine")
            app._accept()
            await pilot.pause()
            self.assertNotIsInstance(app.screen, ConfirmScreen)
            await pilot.pause()
            self.assertEqual([c.type for _, c in client.commands], ["accept"])

    async def test_reject_confirms_drift_and_a_cancel_posts_nothing(self) -> None:
        from textual.widgets import Label

        from xarness.tui.confirm_screen import ConfirmScreen

        client, app = self._app()
        async with run_app(app) as pilot:
            app._last_diff = self._diff("mine", "drift")
            app._reject_with_confirm()
            await pilot.pause()
            self.assertIsInstance(app.screen, ConfirmScreen)
            title = app.screen.query_one("#confirm-title", Label).render()
            self.assertIn("revert 1 drift file", str(title))
            # Declining leaves the workspace alone.
            await pilot.press("escape")
            await pilot.pause()
            await pilot.pause()
            self.assertEqual(client.commands, [])

    async def test_reject_without_drift_posts_without_a_modal(self) -> None:
        from xarness.tui.confirm_screen import ConfirmScreen

        client, app = self._app()
        async with run_app(app) as pilot:
            app._last_diff = self._diff("agent")
            app._reject_with_confirm()
            await pilot.pause()
            self.assertNotIsInstance(app.screen, ConfirmScreen)
            await pilot.pause()
            self.assertEqual([c.type for _, c in client.commands], ["reject"])


class RefsScreenTest(unittest.IsolatedAsyncioTestCase):
    """The /refs manager lists container mounts (/tmp/refs/<alias>) and
    dispatches add/remove without ever touching the worktree."""

    async def test_lists_mounts_and_dispatches_remove(self) -> None:
        from xarness.protocol import RefInfo
        from xarness.tui.refs_screen import RefsScreen

        client = FakeClient(make_snapshot())
        app = make_app(client)
        async with run_app(app) as pilot:
            refs = [RefInfo(path="/tmp/notes.md", mount="/tmp/refs/notes.md")]
            client.refs = list(refs)
            app.push_screen(RefsScreen(refs))
            await pilot.pause()
            screen = app.screen
            self.assertIsInstance(screen, RefsScreen)
            # The dimmed line is the container path the model uses; the
            # worktree never gets a refs directory.
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
        from xarness.tui.refs_screen import RefsScreen

        client = FakeClient(make_snapshot())
        app = make_app(client)
        async with run_app(app) as pilot:
            app._handle_slash_command("/refs")
            await pilot.pause()
            self.assertIsInstance(app.screen, RefsScreen)


class SlashCommandDispatchTest(unittest.IsolatedAsyncioTestCase):
    """A leading "/" only makes a command when it names a registered one;
    any other input is ordinary chat text, not an unknown-command error."""

    async def test_unknown_slash_text_is_sent_as_a_message(self) -> None:
        from xarness.tui.widgets import ChatInput

        client = FakeClient(make_snapshot())
        app = make_app(client)
        async with run_app(app) as pilot:
            app.on_chat_input_chat_submitted(
                ChatInput.ChatSubmitted("/not-a-command hello"))
            await pilot.pause()
            await pilot.pause()
            sends = [c for _, c in client.commands if c.type == "send_message"]
            self.assertEqual([c.text for c in sends], ["/not-a-command hello"])

    async def test_registered_slash_command_still_runs(self) -> None:
        from xarness.tui.widgets import ChatInput

        client = FakeClient(make_snapshot())
        app = make_app(client)
        async with run_app(app) as pilot:
            app.on_chat_input_chat_submitted(ChatInput.ChatSubmitted("/mode"))
            await pilot.pause()
            await pilot.pause()
            self.assertEqual(
                [c.mode for _, c in client.commands if c.type == "set_mode"],
                ["plan"],
            )
            self.assertEqual(
                [c for _, c in client.commands if c.type == "send_message"], [])


class AttachSeedsPumpTest(unittest.IsolatedAsyncioTestCase):
    """The pump attaches to the session by name; reconnects always deliver
    a fresh Snapshot (which re-renders the history exactly once — the
    rendered snapshot is replaced, not appended to)."""

    async def test_attach_is_called_with_the_session_name(self) -> None:


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
        app = make_app(client)
        async with run_app(app):
            self.assertEqual(calls, ["s1"])


class MultiRoundStreamTest(unittest.IsolatedAsyncioTestCase):
    """Rounds 2+ of a turn must stream into FRESH entries: the previous
    round's thinking block is finished (collapsed) and its answer settled —
    appending to them made later CoT and answers invisible until reload."""

    async def test_second_round_streams_into_new_widgets(self) -> None:

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
            q.put_nowait(protocol.RoundComplete(has_tool_calls=True))
            q.put_nowait(protocol.ToolFinished(call_id="c1", status="ok", content="ok"))
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
            q.put_nowait(protocol.RoundComplete())
            await pilot.pause()

        app_client = FakeClient(make_snapshot(state="running"))
        app = make_app(app_client)
        async with run_app(app) as pilot:
            await script(pilot, app_client, app)


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

    def test_running_run_bash_names_the_command_whatever_its_shape(self) -> None:
        """The live label must not depend on the command's first line being
        non-blank: leading newlines and multi-line scripts (heredocs) still
        have to say what is running, and agree with the settled label."""
        from xarness.tui.tool_view import _bash_settled, live_parts

        cases = {
            "\nls -la": "ls -la",
            "\n\n  rg foo": "rg foo",
            "cd workspace &&\nls": "cd workspace && ls",
            "python - <<PY\nprint(1)\nPY": "python - <<PY print(1) PY",
        }
        for command, expected in cases.items():
            args = json.dumps({"command": command})
            self.assertEqual(live_parts("run_bash", args), ("run_bash", expected))
            self.assertEqual(_bash_settled(args).plain.strip(), expected)

    def test_tools_without_a_bespoke_view_name_themselves(self) -> None:
        """subagent/ask_user must not fall back to the anonymous "tool"."""
        from xarness.tui.tool_view import live_parts

        self.assertEqual(
            live_parts("subagent", '{"prompt": "do the thing"}'),
            ("subagent", ""),
        )
        self.assertEqual(live_parts("ask_user", "{}"), ("ask_user", ""))

    def test_mcp_tools_read_as_server_dot_tool(self) -> None:
        """A namespaced provider name is shown as ``server · tool`` in the
        live label, the settled summary and the expanded body."""
        from xarness.tui.tool_view import (
            ToolCallView,
            body_renderable,
            live_parts,
            summary_text,
        )
        from xarness.tui.transcript import ToolStatus

        args = '{"path": "notes.md"}'
        self.assertEqual(
            live_parts("mcp__fs__read", args), ("fs · read", "notes.md"),
        )
        settled = summary_text(ToolCallView(
            name="mcp__fs__read", args=args, status=ToolStatus.CALL_SUCCEEDED,
        ))
        self.assertEqual(settled.plain, "fs · read notes.md")
        body = body_renderable(ToolCallView(name="mcp__fs__read", args=args))
        self.assertIn('"path": "notes.md"', body.plain)

    def test_running_command_truncates_like_the_settled_one(self) -> None:
        """A long command may not shrink when the call finishes: the live
        detail keeps the settled width."""
        from xarness.tui.tool_view import _bash_settled, live_parts

        args = json.dumps({"command": "rg " + "x" * 200})
        live = live_parts("run_bash", args)[1]
        self.assertEqual(live, _bash_settled(args).plain.strip())
        self.assertLessEqual(len(live), 80)

    async def test_streamed_arguments_render_live_details(self) -> None:

        calls = [
            ("w1", "write_file",
             '{"path": "a.py", "content": "a\\nb\\nc\\n"}', "+3 a.py"),
            ("e1", "edit_file",
             '{"path": "b.py", "edits": [{"old_string": "x", '
             '"new_string": "y\\nz"}]}', "+2 -1 b.py"),
            ("b1", "run_bash", '{"command": "grep -rn foo"}',
             "grep -rn foo"),
            # Leading newline: the command must still name itself live.
            ("b2", "run_bash", '{"command": "\\nla -la /workspace"}',
             "la -la /workspace"),
        ]
        client = FakeClient(make_snapshot(state="running"))
        app = make_app(client)
        async with run_app(app) as pilot:
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
            self.assertIn("Running run_bash la -la /workspace", painted)

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

        client = FakeClient(make_snapshot(state="running"))
        app = make_app(client)
        async with run_app(app) as pilot:
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

        client = FakeClient(make_snapshot(state="running"))
        app = make_app(client)
        async with run_app(app) as pilot:
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
                call_id="e1", status="ok", content=output,
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
    """Session popup rows: {title} {age} {dir} {diff} and a right-aligned
    state word for loaded sessions (none for saved ones)."""

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
            "ask": "warning", "waiting": "accent", "running": "success",
            "compacting": "success", "idle": "muted",
        }
        for state, key in expected.items():
            self.assertEqual(_STATE_WORDS.get(state), (state, key))
            self.assertIn(key, theme.PALETTE)
        # Saved / not loaded sessions show no state word at all.
        self.assertIsNone(_STATE_WORDS.get("saved"))

    def test_theme_rows_show_five_color_squares(self) -> None:
        """Each /theme row previews the palette with five painted swatches
        lined up on the right of the name."""
        from xarness import theme
        from xarness.tui.list_picker import (
            SESSION_ROW_WIDTH, SWATCH, THEME_SWATCH_KEYS, theme_options,
        )

        options = theme_options(list(theme.THEMES), current=theme.CURRENT_THEME)
        self.assertEqual([str(o.id) for o in options], list(theme.THEMES))
        for option in options:
            palette = theme.THEMES[str(option.id)]
            spans = [span for span in option.prompt.spans
                     if span.style and span.style.startswith("#")]
            swatches = [option.prompt.plain[s.start:s.end] for s in spans]
            self.assertEqual(len(swatches), 5, "one square per swatch key")
            self.assertEqual(swatches, [SWATCH] * 5)
            expected = [palette[key] for key in THEME_SWATCH_KEYS]
            self.assertEqual([span.style for span in spans], expected)
            # Name first, swatches flush against the right edge.
            self.assertGreater(spans[0].start, str(option.prompt).index(str(option.id)))
            self.assertEqual(len(str(option.prompt)), SESSION_ROW_WIDTH)

    def test_ask_state_gets_its_own_word(self) -> None:
        """A session blocked on its ask tool reads "ask", not "waiting"."""
        from xarness.tui.list_picker import (
            STATE_WIDTH as _STATE_WIDTH,
            SPACER_ID,
            session_options,
        )

        entries = [
            SessionSummary(
                name="ask-one", state="ask", active=True,
                last_user_message_at="2026-01-01T00:00:00+00:00",
            ),
        ]
        options = [o for o in session_options(entries, None)
                   if str(o.id) != SPACER_ID]
        self.assertEqual(len(options), 1)
        row = str(options[0].prompt)
        self.assertTrue(row.endswith("ask".rjust(_STATE_WIDTH)), row)
        self.assertNotIn("waiting", row)

    def test_only_the_current_workspace_is_listed(self) -> None:
        """Live sessions of another workspace are dropped like saved ones:
        /sessions is about the checkout the TUI runs in."""
        from xarness.tui.list_picker import SPACER_ID, session_options

        def summary(name, workspace, state, active, at):
            return SessionSummary(
                name=name, workspace=workspace, state=state, active=active,
                last_user_message_at=f"2026-01-01T{at}+00:00",
            )

        entries = [
            summary("here-live", "/work/app", "running", True, "04:00"),
            summary("there-live", "/work/other", "running", True, "03:00"),
            summary("here-saved", "/work/app/.", "saved", False, "02:00"),
            summary("there-saved", "/work/other", "saved", False, "01:00"),
        ]
        options = session_options(entries, "/work/app")
        self.assertEqual(
            [str(o.id) for o in options],
            ["here-live", SPACER_ID, "here-saved"],
        )

    def test_titles_pad_to_the_longest_title(self) -> None:
        """Every row's title column is the same fixed width, so the age (and
        everything after it) starts at the same offset on every row."""
        from xarness.tui.list_picker import session_options

        names = ["short", "a-much-longer-title"]
        entries = [
            SessionSummary(
                name=name, state="saved",
                last_user_message_at="2026-01-01T00:00:00+00:00",
            )
            for name in names
        ]
        rows = [str(o.prompt) for o in session_options(entries, None)]
        self.assertEqual(len(rows), 2)
        title_width = max(map(len, names))
        for row, name in zip(rows, names):
            self.assertEqual(row[:title_width], name.ljust(title_width), row)
        # Same age/dir/diff columns: the only difference is the padded title.
        self.assertEqual(rows[0][title_width:], rows[1][title_width:], rows)
        self.assertGreater(len(rows[0][title_width:].strip()), 0, rows)

    async def test_rows_render_title_age_and_dot(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Label, OptionList

        from xarness.tui.list_picker import ListPicker, session_options

        class Host(App):
            def compose(self) -> ComposeResult:
                yield Label("")

        app = Host()
        async with run_app(app) as pilot:
            app.push_screen(ListPicker(session_options(self._summaries(), None), title="Open session", filterable=True))
            await pilot.pause()
            option_list = app.screen.query_one("#picker-list", OptionList)
            prompts = [
                str(option_list.get_option_at_index(i).prompt)
                for i in range(option_list.option_count)
            ]
            self.assertEqual(len(prompts), 6)
            # The live sessions (with a state word) and the saved sessions
            # of this workspace are separated by one blank row of padding,
            # which the list marks disabled so the cursor skips it.
            blank = [i for i, p in enumerate(prompts) if not p.strip()]
            self.assertEqual(len(blank), 1, "missing the one-row section gap")
            self.assertEqual(blank[0], 3, "gap belongs after the live rows")
            spacer = option_list.get_option_at_index(blank[0])
            self.assertTrue(spacer.disabled, "the gap must not be selectable")
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
            # Every session row shows an age column (the gap row does not).
            rows = [p for i, p in enumerate(prompts) if i not in blank]
            self.assertTrue(all("m" in p or "h" in p or "d" in p for p in rows))

    async def test_gap_hidden_while_filtering_and_never_highlighted(self) -> None:
        """The blank gap row is padding only: typing hides it and the cursor
        steps over it."""
        from textual.app import App, ComposeResult
        from textual.widgets import Input, Label, OptionList

        from xarness.tui.list_picker import ListPicker, session_options

        class Host(App):
            def compose(self) -> ComposeResult:
                yield Label("")

        app = Host()
        async with run_app(app) as pilot:
            app.push_screen(ListPicker(
                session_options(self._summaries(), None),
                title="Open session", filterable=True,
            ))
            await pilot.pause()
            option_list = app.screen.query_one("#picker-list", OptionList)
            self.assertEqual(option_list.option_count, 6)
            self.assertEqual(option_list.highlighted, 0)
            # waiting-one -> running-one -> compacting-one -> (gap) -> idle-one
            for _ in range(3):
                await pilot.press("down")
                await pilot.pause()
            self.assertEqual(
                option_list.get_option_at_index(option_list.highlighted).id,
                "idle-one",
            )
            # Typing narrows the list and drops the gap row.
            app.screen.query_one("#picker-search", Input).value = "saved"
            await pilot.pause()
            self.assertEqual(option_list.option_count, 1)
            self.assertEqual(
                option_list.get_option_at_index(option_list.highlighted).id,
                "saved-one",
            )

    async def test_update_options_keeps_filter_and_highlight(self) -> None:
        """The live /sessions refresh replaces rows without moving the
        cursor off the highlighted session or clearing the search text."""
        from textual.app import App, ComposeResult
        from textual.widgets import Input, Label, OptionList

        from xarness.tui.list_picker import ListPicker, session_options

        class Host(App):
            def compose(self) -> ComposeResult:
                yield Label("")

        app = Host()
        async with run_app(app) as pilot:
            app.push_screen(ListPicker(
                session_options(self._summaries(), None),
                title="Open session", filterable=True,
            ))
            await pilot.pause()
            screen = app.screen
            option_list = screen.query_one("#picker-list", OptionList)
            option_list.highlighted = 1  # running-one
            screen.query_one("#picker-search", Input).value = "one"
            await pilot.pause()
            self.assertEqual(option_list.option_count, 5)

            # The host now reports different states for the same sessions.
            def s(name, state, active, attached=0):
                return SessionSummary(
                    name=name, state=state, active=active,
                    clients_attached=attached,
                    last_user_message_at="2026-01-01T00:00:00+00:00",
                )
            updated = [
                s("waiting-one", "idle", False),
                s("running-one", "idle", False),
                s("compacting-one", "idle", False),
                s("idle-one", "idle", False),
                s("saved-one", "saved", False),
            ]
            screen.update_options(session_options(updated, None))
            await pilot.pause()
            self.assertEqual(option_list.option_count, 5, "search text was lost")
            self.assertEqual(
                option_list.get_option_at_index(option_list.highlighted).id,
                "running-one",
            )
            self.assertIn(
                "idle",
                str(option_list.get_option_at_index(option_list.highlighted).prompt),
            )

    async def test_open_picker_refreshes_while_it_stays_open(self) -> None:
        """Rows are live: a session that goes running -> idle while the
        picker is open updates without closing it."""
        from textual.widgets import OptionList

        from xarness.tui import app as app_module
        from xarness.tui.list_picker import ListPicker

        def summary(state: str) -> SessionSummary:
            return SessionSummary(
                name="s1", state=state, active=True, clients_attached=1,
                last_user_message_at="2026-01-01T00:00:00+00:00",
                message_count=1,
            )

        client = FakeClient(make_snapshot())
        client.session_list = [summary("running")]
        previous_interval = app_module.SESSIONS_REFRESH_SECONDS
        app_module.SESSIONS_REFRESH_SECONDS = 0.02
        try:
            app = make_app(client)
            async with run_app(app) as pilot:
                await app._open_sessions_async()
                await pilot.pause()
                self.assertIsInstance(app.screen, ListPicker)
                # The picker's +/- columns come from the host's stats: the
                # TUI must ask for them or every live row reads as clean.
                self.assertTrue(client.session_calls)
                self.assertTrue(all(
                    call.get("stats") is True for call in client.session_calls
                ), client.session_calls)
                option_list = app.screen.query_one("#picker-list", OptionList)
                self.assertIn("running", str(option_list.get_option_at_index(0).prompt))

                client.session_list = [summary("idle")]
                await asyncio.sleep(0.08)
                await pilot.pause()
                self.assertIn("idle", str(option_list.get_option_at_index(0).prompt))
                # The periodic refresh keeps asking for the live diff totals.
                self.assertTrue(all(
                    call.get("stats") is True for call in client.session_calls
                ), client.session_calls)
        finally:
            app_module.SESSIONS_REFRESH_SECONDS = previous_interval

    async def test_sessions_command_scopes_subagents_to_the_current_session(
        self,
    ) -> None:
        """The picker carries the session it was opened from."""
        from textual.widgets import OptionList

        def summary(name: str, parent: str | None = None) -> SessionSummary:
            return SessionSummary(
                name=name, parent_id=parent, state="saved",
                last_user_message_at="2026-01-01T00:00:00+00:00",
            )

        client = FakeClient(make_snapshot())
        client.session_list = [summary("a"), summary("a/sub", "a"),
                               summary("b")]
        for session_name, expected in (
            (None, ["a", "b"]), ("a", ["a", "a/sub", "b"]),
            ("b", ["a", "b"]),
        ):
            app = make_app(client, session_name=session_name)
            async with run_app(app) as pilot:
                await app._open_sessions_async()
                await pilot.pause()
                options = app.screen.query_one("#picker-list", OptionList)
                self.assertEqual(
                    [str(options.get_option_at_index(i).id)
                     for i in range(options.option_count)],
                    expected, f"wrong rows inside {session_name!r}",
                )

    def test_child_sessions_nest_under_their_parent(self) -> None:
        from xarness.tui.list_picker import session_options

        def summary(
            name: str, parent: str | None = None, at: str = "00:00:00",
        ) -> SessionSummary:
            return SessionSummary(
                name=name, parent_id=parent, state="saved", active=True,
                last_user_message_at=f"2026-01-01T{at}+00:00",
            )

        # Recency order puts the child and the grandchild first; the picker
        # moves each under the parent it names. A subagent is stored as
        # ``<parent>/agent-N`` and only its short name is shown.
        entries = [
            summary("root/agent-1", parent="root", at="00:00:03"),
            summary("other", at="00:00:02"),
            summary("root", at="00:00:04"),
            summary("root/agent-1/agent-1", parent="root/agent-1",
                    at="00:00:01"),
        ]
        options = session_options(entries, None, current="root")
        assert [str(o.id) for o in options] == [
            "root", "root/agent-1", "root/agent-1/agent-1", "other",
        ]
        assert str(options[1].prompt).startswith("↳ agent-1")
        assert str(options[2].prompt).startswith("↳ agent-1")
        assert str(options[3].prompt).startswith("other")

    def test_subagents_only_show_under_the_session_they_ran_from(self) -> None:
        """A draft chat lists top-level sessions alone; a subagent appears
        only from its parent (or from the subagent itself)."""
        from xarness.tui.list_picker import session_options

        def summary(name: str, parent: str | None = None) -> SessionSummary:
            return SessionSummary(
                name=name, parent_id=parent, state="saved",
                last_user_message_at="2026-01-01T00:00:00+00:00",
            )

        entries = [summary("sess1"), summary("sess1/sub1", "sess1"),
                   summary("sess1/sub2", "sess1"), summary("sess2")]
        names = lambda current: [str(o.id)  # noqa: E731
                                 for o in session_options(entries, None, current)]
        assert names(None) == ["sess1", "sess2"]
        assert names("sess1") == ["sess1", "sess1/sub1", "sess1/sub2",
                                  "sess2"]
        assert names("sess2") == ["sess1", "sess2"]
        assert names("sess1/sub1") == ["sess1", "sess1/sub1", "sess2"]

    def test_an_active_member_promotes_its_whole_group(self) -> None:
        """Attaching to an idle subagent keeps its parent and siblings in
        the live section, with the hosted children above the saved ones."""
        from xarness.tui.list_picker import SPACER_ID, session_options

        def summary(
            name: str, parent: str | None = None, state: str = "saved",
            active: bool = False, at: str = "00:00:00",
        ) -> SessionSummary:
            return SessionSummary(
                name=name, parent_id=parent, state=state, active=active,
                last_user_message_at=f"2026-01-01T{at}+00:00",
            )

        entries = [
            summary("root", at="00:00:04"),
            summary("root/agent-1", parent="root", at="00:00:01"),
            summary("root/agent-2", parent="root", state="idle", active=True,
                    at="00:00:03"),
            summary("old", at="00:00:02"),
        ]
        options = session_options(entries, None, current="root")
        assert [str(o.id) for o in options] == [
            "root", "root/agent-2", "root/agent-1", SPACER_ID, "old",
        ], [str(o.id) for o in options]
        rows = {str(o.id): str(o.prompt) for o in options}
        assert "idle" in rows["root"]   # promoted, not loaded
        assert "idle" in rows["root/agent-1"]
        assert "idle" in rows["root/agent-2"]
        assert rows["root/agent-1"].startswith("↳ agent-1")
        assert rows["root/agent-2"].startswith("↳ agent-2")

    def test_idle_child_of_an_active_parent_stays_live(self) -> None:
        """A finished subagent is saved/inactive, but while its parent is
        active it belongs above the gap, not with the old sessions."""
        from xarness.tui.list_picker import SPACER_ID, session_options

        def summary(name, state, active, parent=None, at="00:00:00"):
            return SessionSummary(
                name=name, state=state, active=active, parent_id=parent,
                last_user_message_at=f"2026-01-01T{at}+00:00",
            )

        entries = [
            summary("root", "running", True, at="00:00:05"),
            summary("child", "saved", False, parent="root", at="00:00:04"),
            summary("old", "saved", False, at="00:00:01"),
        ]
        options = session_options(entries, None, current="root")
        assert [str(o.id) for o in options] == [
            "root", "child", SPACER_ID, "old",
        ], [str(o.id) for o in options]
        child_row = next(o for o in options if str(o.id) == "child")
        self.assertIn("idle", str(child_row.prompt))

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
        async with run_app(app) as pilot:
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

        async def script(pilot: Pilot, client: FakeClient, app):
            written: list[str] = []
            app._set_terminal_title = written.append  # type: ignore[method-assign]
            snap = make_snapshot()
            object.__setattr__(
                snap, "status",
                Status(
                    session_name="other", workspace="/other/ws",
                    mode="write", model_name="m", model_display_name="m",
                    model_id="m",
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
        app = make_app(client, session_name=None)
        async with run_app(app) as pilot:
            await script(pilot, client, app)

    async def test_finished_suffix_clears_when_the_next_turn_starts(self) -> None:
        from xarness.protocol import ProcessingStarted

        client = FakeClient(make_snapshot())
        app = make_app(client, session_name=None)
        async with run_app(app) as pilot:
            written: list[str] = []
            app._set_terminal_title = written.append  # type: ignore[method-assign]
            snap = make_snapshot()
            object.__setattr__(
                snap, "status",
                Status(
                    session_name="other",
                    workspace="/workspace/projects/xarness",
                    mode="write", model_name="m", model_display_name="m",
                    model_id="m",
                ),
            )
            client.snapshot_data = snap
            await app._attach("other")
            await pilot.pause()
            self.assertEqual(written[-1], "xarness xarness")

            await app._handle_event(protocol.TurnEnded(outcome="finished"))
            self.assertEqual(written[-1], "xarness xarness - finished")

            # A new round/turn clears the stale suffix.
            await app._handle_event(ProcessingStarted())
            self.assertEqual(written[-1], "xarness xarness")


class TerminalTitlePathTest(unittest.IsolatedAsyncioTestCase):
    """The tab title uses the workspace's own directory name, never a
    cropped multi-component path: ``/workspace/projects/xarness`` and
    ``/home/me/xarness`` must both read ``xarness xarness``."""

    async def test_title_for_uses_the_last_path_component(self) -> None:

        app = make_app(session_name=None)
        async with run_app(app):
            self.assertEqual(app._title_for("/workspace/projects/xarness"), "xarness")
            self.assertEqual(app._title_for("/home/me/larkspur"), "larkspur")
            self.assertEqual(app._title_for("/home/me/larkspur/"), "larkspur")
            self.assertEqual(app._title_for("/"), "/")


class InterruptSettleTest(unittest.IsolatedAsyncioTestCase):
    """An interrupt ends the turn without a RoundComplete for this client:"""

    async def test_turn_end_settles_thinking_shimmer(self) -> None:

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
        app = make_app(client)
        async with run_app(app) as pilot:
            await script(pilot, client, app)

    async def test_turn_end_settles_tool_writing_shimmer(self) -> None:
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
            q.put_nowait(protocol.TurnEnded(outcome="cancelled"))
            await pilot.pause()
            # The half-streamed call settles as failed — no eternal shimmer.
            self.assertEqual(blocks[0].item.status, ToolStatus.CALL_FAILED)
            self.assertEqual(app._phase, "idle")
            self.assertEqual(app._anim, {}, "repaint timer outlived the interrupt")
            self.assertFalse(
                [e for e in app._flow_model.entries
                 if isinstance(e.item, tflow.PendingItem) and e.alive],
                "pending strip outlived the interrupt")


        client = FakeClient(make_snapshot(state="running"))
        app = make_app(client)
        async with run_app(app) as pilot:
            await script(pilot, client, app)

    async def test_pending_strip_only_while_waiting_on_provider(self) -> None:
        """The strip means "waiting on the LLM", not "the turn is busy"."""

        async def script(pilot: Pilot, client: FakeClient, app):
            q = client.queue
            q.put_nowait(protocol.ProcessingStarted())
            q.put_nowait(protocol.ContentDelta(text="hi"))
            q.put_nowait(protocol.RoundComplete(has_tool_calls=True))
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
            q.put_nowait(protocol.RoundComplete(has_tool_calls=False))
            await pilot.pause()
            self.assertEqual(app._phase, "idle")

        client = FakeClient(make_snapshot(state="running"))
        app = make_app(client)
        async with run_app(app) as pilot:
            await script(pilot, client, app)


class ResumeListHeightTest(unittest.IsolatedAsyncioTestCase):
    """The /sessions list must show more than one row at a time (the list
    widgets' 1fr default collapses to a single row inside the auto-height
    modal)."""

    async def test_multiple_rows_visible(self) -> None:
        from textual.widgets import OptionList

        from xarness.tui.list_picker import ListPicker, session_options
        summaries = [
            SessionSummary(name=f"s{i}", state="saved", active=False)
            for i in range(5)
        ]
        client = FakeClient(make_snapshot())
        app = make_app(client, session_name=None)
        async with run_app(app, size=(100, 40)) as pilot:
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
    return (
        UserMessage(parts=[TextPart(text=content)], id=mid)
        if role == "user"
        else AssistantMessage(content=content, id=mid)
    )


class UndoTrimTest(unittest.IsolatedAsyncioTestCase):
    """Undo-from-here rebuilds the transcript from the new snapshot — a
    data-only rebuild (cheap under FlowView), including messages that
    arrived LIVE during the attach."""

    async def test_history_replaced_rebuilds_after_undo(self) -> None:

        messages = [
            _msg("user", "u1", "first"),
            _msg("assistant", "a1", "answer one"),
            _msg("user", "u2", "second"),
            _msg("assistant", "a2", "answer two"),
        ]
        snap = make_snapshot()
        object.__setattr__(snap, "messages", messages)
        client = FakeClient(snap)
        app = make_app(client)
        async with run_app(app) as pilot:
            # A live turn adds a message the snapshot never contained.
            client.queue.put_nowait(protocol.MessageAdded(
                message=UserMessage(parts=[TextPart(text="live")], id="u3")))
            await pilot.pause()
            users = [e.item.message_id for e in app._flow_model.entries
                     if isinstance(e.item, tflow.UserItem)]
            self.assertEqual(users, ["u1", "u2", "u3"])

            # Undo-from-here on "second" (u2): u2, a2 and u3 are gone.
            trimmed = make_snapshot()
            object.__setattr__(trimmed, "messages", messages[:2])
            client.queue.put_nowait(protocol.HistoryReplaced(snapshot=trimmed, reason="rollback"))
            await pilot.pause()
            users = [e.item.message_id for e in app._flow_model.entries
                     if isinstance(e.item, tflow.UserItem)]
            self.assertEqual(users, ["u1"])


class DiffClickMinimizeTest(unittest.IsolatedAsyncioTestCase):
    """Clicking inside the open diff pane minimizes only the diff — the
    expanded file list (and its rows) stays up."""

    async def test_click_on_diff_pane_keeps_file_list(self) -> None:
        from textual.geometry import Offset

        from xarness.tui.widgets import DiffSummary

        snap = make_snapshot()
        client = FakeClient(snap)
        app = make_app(client)
        async with run_app(app) as pilot:
            summary = app.query_one("#diff-summary", DiffSummary)
            await summary.set_summary(make_diff_stat())
            summary.add_class("expanded")
            summary.show_diff(("a.py", None), "--- a/a.py\n+++ b/a.py\n@@\n-x\n+y\n")
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
        DiffFileInfo(path="a.py", additions=2, deletions=1, is_new=False, kind="agent"),
    ])


class DraftStatusTest(unittest.IsolatedAsyncioTestCase):
    """A draft chat (no session yet) shows the config's default model — not
    a dash — and Tab toggles a local draft mode that is applied to the
    session created by the first message."""

    async def test_draft_shows_default_model_and_tab_applies_mode(self) -> None:

        snap = make_snapshot()
        client = FakeClient(snap)
        app = make_app(client, session_name=None)
        async with run_app(app) as pilot:
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
            client.queue.put_nowait(protocol.StatusChanged(status=Status(
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

        snap = make_snapshot()
        client = FakeClient(snap)

        async def config():
            return protocol.ConfigInfo(
                models=[
                    dict(_model_info(), name="glm-flash",
                         display_name="GLM Flash"),
                    dict(_model_info(), name="other",
                         display_name="Other Model", cot_strength="high",
                         max_context=200_000),
                ],
                sandboxes=[{"name": "default"}],
                presets=["default"], themes=[],
                default_model="glm-flash",
                default_model_display_name="GLM Flash",
            )

        client.config = config
        # config_path=None keeps the preference save off the real config file.
        app = make_app(client, session_name=None, config_path=None)
        async with run_app(app) as pilot:
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
            UserMessage(
                parts=[TextPart(text="do it")], id="u1",
                timing=Timing(send="2025-01-01T00:00:00+00:00"),
            ),
            AssistantMessage(
                content="", reasoning="step one\nstep two",
                timing=Timing(reasoning_seconds=4.0),
                tool_calls=[{"call_id": "c1", "name": "read_file",
                             "arguments_json": "{}"}],
            ),
            ToolMessage(id="t1", tool_call_id="c1", content="file body"),
            AssistantMessage(
                content="Done.",
                # The host stamps a finished turn's numbers on its last answer;
                # that stamp is what a footer is drawn from.
                timing=Timing(
                    start="2025-01-01T00:00:01+00:00",
                    end="2025-01-01T00:00:05+00:00",
                    stats=TurnStats(total=5.0, busy=4.0, idle=1.0),
                ),
            ),
        ]
        return snap

    async def test_hidden_epochs_from_compaction_still_render(self) -> None:
        """The snapshot's whole message list is rendered, including the
        pre-compaction epochs, so resuming shows the whole session."""
        from xarness.conversation import SUMMARY_PREFIX
        from xarness.tui.transcript import DividerItem, UserItem

        snap = make_snapshot()
        snap.messages = [
            UserMessage(parts=[TextPart(text="first question")], id="u0"),
            UserMessage(
                parts=[TextPart(text=f"{SUMMARY_PREFIX}\n\nthe handoff")],
                id="s0", is_summary=True,
            ),
            AssistantMessage(content="after", id="a0"),
        ]
        app = make_app(FakeClient(snap))
        async with run_app(app) as pilot:
            await pilot.pause()
            items = [entry.item for entry in app._flow_model.entries]
            self.assertIsInstance(items[0], UserItem)
            self.assertEqual(items[0].text, "first question")
            self.assertIsInstance(items[1], DividerItem)
            self.assertIn("the handoff", items[1].text)
            self.assertEqual(items[2].chunks, ["after"])

    async def test_replay_has_no_shimmer_and_shows_timing(self) -> None:

        client = FakeClient(self._snapshot_with_history())
        app = make_app(client)
        async with run_app(app) as pilot:
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
            self.assertIn("Worked for 5.0s (4.0s | 1.0s)", all_text)

    async def test_replay_footer_shows_turn_token_counts(self) -> None:
        """The footer's ↑input ↓output spend is the host's turn total,
        stamped next to the timings — the client sums nothing."""

        snap = self._snapshot_with_history()
        snap.messages[3].timing.stats.usage = Usage(
            input_tokens=3_234, output_tokens=456)
        app = make_app(FakeClient(snap))
        async with run_app(app) as pilot:
            await pilot.pause()
            flow = app.query_one("#chat-log")
            all_text = "\n".join(flow.row_text(y) for y in range(flow.row_count))
            self.assertIn(
                "Worked for 5.0s (4.0s | 1.0s) · ↑3.2k ↓456", all_text)

    async def test_resumed_running_turn_shows_no_footer(self) -> None:
        """The reported bug: continue headless, then reattach while the turn
        still works. The host has not ended that turn, so its last answer
        carries no stats and no \"Worked for\" line is drawn under it."""

        snap = self._snapshot_with_history()
        snap.status.state = "running"  # type: ignore[assignment]
        snap.messages[-1].timing.stats = None  # not over: nothing stamped
        app = make_app(FakeClient(snap))
        async with run_app(app) as pilot:
            await pilot.pause()
            flow = app.query_one("#chat-log")
            all_text = "\n".join(flow.row_text(y) for y in range(flow.row_count))
            self.assertIn("Done.", all_text)
            self.assertNotIn("Worked for", all_text)

    async def test_attach_mid_tool_shows_it_running_not_interrupted(self) -> None:
        """A turn whose tools are still executing has no result for those
        calls yet: the live round in the snapshot must win over the history."""

        snap = make_snapshot("running")
        snap.messages = [
            UserMessage(parts=[TextPart(text="do it")], id="u1"),
            AssistantMessage(
                content="",
                timing=Timing(
                    start="2025-01-01T00:00:00+00:00",
                    end="2025-01-01T00:00:02+00:00",
                ),
                tool_calls=[{"call_id": "c1", "name": "read_file",
                             "arguments_json": "{}"}],
            ),
        ]
        snap.partial_round = protocol.PartialRound(
            tool_calls=[protocol.PartialToolCall(
                call_id="c1", name="read_file", arguments="{}",
            )],
        )
        app = make_app(FakeClient(snap))
        async with run_app(app) as pilot:
            await pilot.pause()
            flow = app.query_one("#chat-log")
            all_text = "\n".join(flow.row_text(y) for y in range(flow.row_count))
            tools = [e.item for e in app._flow_model.entries
                     if isinstance(e.item, tflow.ToolItem)]
            self.assertEqual(len(tools), 1)
            self.assertIs(tools[0].status, ToolStatus.MAKING_CALL)
            self.assertNotEqual(tools[0].output, "interrupted")
            self.assertNotIn("Worked for", all_text)  # the turn is still open

    async def test_unanswered_call_in_a_running_turn_is_not_interrupted(self) -> None:
        """The host says the turn is running, so a call with no result is the
        one in flight — even if the snapshot carries no live round for it
        (an older host, or a gap in the bookkeeping)."""

        snap = make_snapshot("running")
        snap.messages = [
            UserMessage(parts=[TextPart(text="do it")], id="u1"),
            AssistantMessage(
                content="",
                timing=Timing(
                    start="2025-01-01T00:00:00+00:00",
                    end="2025-01-01T00:00:02+00:00",
                ),
                tool_calls=[{"call_id": "c1", "name": "read_file",
                             "arguments_json": "{}"}],
            ),
        ]
        app = make_app(FakeClient(snap))
        async with run_app(app) as pilot:
            await pilot.pause()
            tools = [e.item for e in app._flow_model.entries
                     if isinstance(e.item, tflow.ToolItem)]
            self.assertEqual(len(tools), 1)
            self.assertIs(tools[0].status, ToolStatus.MAKING_CALL)
            self.assertNotEqual(tools[0].output, "interrupted")


class FlowViewTest(unittest.IsolatedAsyncioTestCase):
    """The virtualized transcript: the WHOLE history lives in one FlowModel
    widget regardless of length, and only the visible rows are painted."""

    @staticmethod
    def _big_snapshot(count: int = 1200) -> Snapshot:
        snap = make_snapshot()
        snap.messages = [
            _msg("user" if i % 2 == 0 else "assistant", f"m{i}", f"body {i}")
            for i in range(count)
        ]
        return snap

    async def test_full_history_in_model_one_widget(self) -> None:
        from textual_flowview import FlowView


        app = make_app(FakeClient(self._big_snapshot()))
        async with run_app(app) as pilot:
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


        app = make_app(FakeClient(self._big_snapshot()))
        async with run_app(app) as pilot:
            await pilot.pause()
            flow = app.query_one("#chat-log", FlowView)
            self.assertLess(len(flow._layout), 100)


class PresenterWidthTest(unittest.IsolatedAsyncioTestCase):
    """Non-user rows reserve the one-cell left inset before wrapping."""

    class _Entry:
        def __init__(self, item) -> None:
            from types import SimpleNamespace
            self.item = item
            self._model = SimpleNamespace(entries=[self])

    async def test_status_line_keeps_every_character(self) -> None:
        from xarness.tui.flow import Presenter
        from xarness.tui.transcript import LineItem

        width = 20
        presentation = await Presenter().present(
            self._Entry(LineItem(text="x" * 40, kind="info")), width,
        )

        self.assertEqual(
            "".join(strip.text for strip in presentation.strips).count("x"),
            40,
        )
        for strip in presentation.strips:
            self.assertEqual(strip.cell_length, width)

    async def test_assistant_paragraph_keeps_word_tails(self) -> None:
        from xarness.tui.flow import Presenter
        from xarness.tui.transcript import AssistantItem

        width = 16
        item = AssistantItem(chunks=["aaaaaa bbbb cc"])
        presentation = await Presenter().present(self._Entry(item), width)
        text = "".join(strip.text for strip in presentation.strips)

        self.assertIn("cc", text)
        self.assertEqual(text.count("a") + text.count("b") + text.count("c"), 12)
        for strip in presentation.strips:
            self.assertEqual(strip.cell_length, width)


class TextSelectionTest(unittest.IsolatedAsyncioTestCase):
    """Mouse drag over the transcript must both highlight and extract text."""

    async def test_drag_selects_and_highlights_transcript_text(self) -> None:
        from xarness import theme
        from xarness.tui.flow import UserItem

        app = make_app()
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
    """The pending-changes panel leads with the agent's own files and shows
    one foldable section per other kind below them."""

    async def test_agent_files_lead_and_other_kinds_fold(self) -> None:
        from textual.widgets import Static

        from xarness.protocol import DiffFileInfo, DiffGroup, DiffInfo
        from xarness.tui.widgets import DiffSummary

        app = make_app()
        async with run_app(app, size=(120, 40)) as pilot:
            summary = app.query_one("#diff-summary", DiffSummary)

            await summary.set_summary(DiffInfo(
                files=[
                    DiffFileInfo(path="own.py", additions=4, deletions=1, kind="mine"),
                    DiffFileInfo(path="a.py", additions=9, deletions=3, kind="drift"),
                ],
                groups=[DiffGroup(prefix="build/", files=7, kind="auto")],
                ignored_files=7,
            ))
            summary.add_class("expanded")
            await pilot.pause()
            # Default view: only the agent's file (and the section headers;
            # every session's rows already lead the list, so there is no
            # "agent" section to fold them into).
            self.assertEqual(
                [r.diff_key for r in summary.query(".diff-file-row")],
                [("own.py", None)],
            )
            self.assertEqual(
                [r.diff_section for r in summary.query(".diff-section-row")],
                ["auto", "drift"],
            )
            drift = summary.query_one(".diff-section-row.drift")
            self.assertEqual(
                drift.query_one(".diff-file-add", Static).render().plain, "+9"
            )
            self.assertEqual(
                drift.query_one(".diff-file-del", Static).render().plain, "-3"
            )
            auto = summary.query_one(".diff-section-row.auto")
            self.assertEqual(
                auto.query_one(".diff-section-count", Static).render().plain, "7 files"
            )

            # Clicking the drift header unfolds its file row, which keeps the
            # drift badge and opens its diff on click.
            await pilot.click(drift)
            await pilot.pause()
            rows = list(summary.query(".diff-file-row"))
            self.assertEqual(
                [r.diff_key for r in rows], [("own.py", None), ("a.py", None)]
            )
            self.assertEqual(
                rows[1].query_one(".diff-file-source", Static).render().plain,
                "drift",
            )
            toggle = drift.query_one(".diff-section-toggle", Static)
            self.assertEqual(toggle.render().plain, "\u25be")

            # Clicking again folds it back, keeping the header.
            await pilot.click(drift)
            await pilot.pause()
            self.assertEqual(
                [r.diff_key for r in summary.query(".diff-file-row")],
                [("own.py", None)],
            )
            self.assertEqual(toggle.render().plain, "\u25b8")

    async def test_an_agent_created_group_leads_with_every_session(self) -> None:
        """An ignored file agents edited leads the list, or folds into ``agent``."""
        from xarness.protocol import DiffFileInfo, DiffGroup, DiffInfo
        from xarness.tui.widgets import DiffSummary

        app = make_app()
        async with run_app(app, size=(120, 40)) as pilot:
            summary = app.query_one("#diff-summary", DiffSummary)
            summary.add_class("expanded")
            await summary.set_summary(DiffInfo(
                files=[
                    DiffFileInfo(path="own.py", additions=1, deletions=0,
                                 kind="mine"),
                ],
                groups=[DiffGroup(prefix="dist/bundle.js", files=1, kind="agent")],
            ))
            await pilot.pause()
            group = summary.query_one(".diff-group-row")
            self.assertEqual(group.diff_key, ("dist/bundle.js", None))

            await summary.set_scope("mine")
            await pilot.pause()
            self.assertFalse(summary.query(".diff-group-row"))
            self.assertTrue(summary.query_one(".diff-section-row.agent"))

    async def test_a_shared_file_shows_its_author_count(self) -> None:
        """A path written by more than one session carries a count badge, so
        the row's +A/-D read as that author's delta, not the whole file's."""
        from textual.widgets import Static

        from xarness.protocol import DiffFileInfo, DiffInfo
        from xarness.tui.widgets import DiffSummary

        app = make_app()
        async with run_app(app, size=(120, 40)) as pilot:
            summary = app.query_one("#diff-summary", DiffSummary)
            await summary.set_summary(DiffInfo(files=[
                DiffFileInfo(
                    path="shared.py", additions=2, deletions=0, kind="mine",
                    authors=("s1", "other"),
                ),
                DiffFileInfo(path="solo.py", additions=1, deletions=0, kind="mine"),
            ]))
            summary.add_class("expanded")
            await pilot.pause()

            rows = {r.diff_key: r for r in summary.query(".diff-file-row")}
            self.assertEqual(
                [cell.classes for cell in rows[("shared.py", None)].children], [
                    {"diff-file-name"}, {"diff-file-shared"},
                    {"diff-file-source", "mine"}, {"diff-file-add"},
                    {"diff-file-del"},
                ],
            )
            self.assertEqual(
                rows[("shared.py", None)].query_one(
                    ".diff-file-shared", Static,
                ).render().plain,
                "2",
            )
            self.assertFalse(rows[("solo.py", None)].query(".diff-file-shared"))

            # A row that becomes shared after the fact gains the badge in the
            # right column, without remounting the row.
            await summary.set_summary(DiffInfo(files=[
                DiffFileInfo(
                    path="shared.py", additions=2, deletions=0, kind="mine",
                    authors=("s1", "other", "third"),
                ),
                DiffFileInfo(
                    path="solo.py", additions=1, deletions=0, kind="mine",
                    authors=("s1", "other"),
                ),
            ]))
            await pilot.pause()
            badges = list(summary.query(".diff-file-shared"))
            self.assertEqual(
                [badge.render().plain for badge in badges], ["3", "2"],
            )

    async def test_a_combined_row_credits_the_last_writer(self) -> None:
        """A combined row adds several sessions' edits up, so the badge
        credits the one that wrote the path last."""
        from textual.widgets import Static

        from xarness.protocol import DiffFileInfo, DiffInfo
        from xarness.tui.widgets import DiffSummary

        app = make_app()
        async with run_app(app, size=(120, 40)) as pilot:
            summary = app.query_one("#diff-summary", DiffSummary)
            summary.add_class("expanded")
            await summary.set_summary(DiffInfo(files=[
                DiffFileInfo(path="a.py", additions=3, deletions=1, kind="agent",
                             authors=("s1", "s2")),
                DiffFileInfo(path="b.py", additions=3, deletions=1, kind="mine",
                             authors=("s2", "s1")),
                DiffFileInfo(path="c.py", additions=1, deletions=0, kind="agent",
                             author="s2", authors=("s1", "s2")),
            ]))
            await pilot.pause()

            def badge(key) -> Static:
                rows = {r.diff_key: r for r in summary.query(".diff-row")}
                return rows[key].query_one(".diff-file-source", Static)

            # Another session wrote the path last: dimmed ``agent``.
            self.assertEqual(
                (badge(("a.py", None)).render().plain,
                 badge(("a.py", None)).has_class("agent")),
                ("agent", True),
            )
            # This session did: the same word, accent-colored.
            self.assertEqual(
                (badge(("b.py", None)).render().plain,
                 badge(("b.py", None)).has_class("mine")),
                ("agent", True),
            )
            # A per-author row credits its own author, not the last writer.
            self.assertEqual(badge(("c.py", "s2")).render().plain, "agent")
            self.assertTrue(badge(("c.py", "s2")).has_class("agent"))

    async def test_clicking_an_ignored_group_requests_its_diff(self) -> None:
        """A group row is a click target like a file row: it asks the app for
        the group's own patch (files added from nothing) instead of
        collapsing the whole panel."""
        from xarness.protocol import DiffGroup, DiffInfo
        from xarness.tui.widgets import DiffSummary

        client = FakeClient(make_snapshot())
        requested: list[str | None] = []

        async def diff(name, path=None, author=None):
            requested.append((path, author))
            return DiffInfo(
                text="diff --git a/build/ b/build/\n+++ b/build/out.o\n+obj\n",
            )

        client.diff = diff
        app = make_app(client)
        async with run_app(app, size=(120, 40)) as pilot:
            summary = app.query_one("#diff-summary", DiffSummary)
            await summary.set_summary(DiffInfo(
                groups=[DiffGroup(prefix="build/", files=7, kind="auto")],
                ignored_files=7,
            ))
            summary.add_class("expanded")
            await summary.toggle_section("auto")
            await pilot.pause()

            group = summary.query_one(".diff-group-row")
            await pilot.click(group)
            await pilot.pause()
            self.assertEqual([p for p in requested if p[0]], [("build/", None)])
            self.assertEqual(summary.active_diff_key, ("build/", None))
            self.assertTrue(summary.diff_shown)
            # The list stays up, with the group row highlighted.
            self.assertTrue(summary.has_class("expanded"))
            self.assertTrue(group.has_class("active"))

    async def test_section_headers_follow_a_divider_with_the_arrow_first(self) -> None:
        """A full-width rule separates the agent files from the sections;
        each header reads label, fold arrow, then the totals on the right."""
        from xarness.protocol import DiffFileInfo, DiffGroup, DiffInfo
        from xarness.tui.widgets import DiffSummary

        app = make_app()
        async with run_app(app, size=(120, 40)) as pilot:
            summary = app.query_one("#diff-summary", DiffSummary)
            await summary.set_summary(DiffInfo(
                files=[
                    DiffFileInfo(path="own.py", additions=1, deletions=0, kind="mine"),
                ],
                groups=[DiffGroup(prefix="build/", files=7, kind="auto")],
                ignored_files=7,
            ))
            summary.add_class("expanded")
            await pilot.pause()

            files = summary.query_one("#diff-files")
            divider = summary.query_one(".diff-divider")
            self.assertEqual(list(files.children).index(divider), 1)

            auto = summary.query_one(".diff-section-row.auto")
            self.assertEqual(
                [next(iter(cell.classes)) for cell in auto.children],
                [
                    "diff-section-name",
                    "diff-section-toggle",
                    "diff-section-spacer",
                    "diff-section-count",
                    "diff-file-add",
                    "diff-file-del",
                ],
            )
            # ``agent``/``auto``/``drift`` share a 5-cell label column, so the
            # fold arrows start at the same x on every header.
            names = list(summary.query(".diff-section-name"))
            self.assertEqual({name.size.width for name in names}, {5})
            toggles = list(summary.query(".diff-section-toggle"))
            self.assertEqual(len({toggle.region.x for toggle in toggles}), 1)

    async def test_totals_widen_the_columns_so_badges_line_up(self) -> None:
        """The summary total can be wider than any single row; the +/- columns
        grow to it so the ``agent`` badges (and the totals) line up."""
        from textual.widgets import Static

        from xarness.protocol import DiffFileInfo, DiffInfo
        from xarness.tui.widgets import DiffSummary

        app = make_app()
        async with run_app(app, size=(120, 40)) as pilot:
            summary = app.query_one("#diff-summary", DiffSummary)
            await summary.set_summary(DiffInfo(files=[
                DiffFileInfo(path="a.py", additions=1500, deletions=100, kind="mine"),
                DiffFileInfo(path="b.py", additions=17, deletions=12, kind="agent"),
                DiffFileInfo(path="c.py", additions=1, deletions=0,
                             is_new=True, kind="mine"),
            ]))
            summary.add_class("expanded")
            await pilot.pause()

            summary_add = summary.query_one("#diff-summary-add", Static)
            summary_del = summary.query_one("#diff-summary-del", Static)
            self.assertEqual(summary_add.render().plain, "+1518")
            self.assertEqual(summary_del.render().plain, "-112")

            rows = list(summary.query(".diff-file-row"))
            adds = {row.query_one(".diff-file-add", Static).render().plain for row in rows}
            dels = {row.query_one(".diff-file-del", Static).render().plain for row in rows}
            self.assertEqual({len(text) for text in adds}, {len("+1518")})
            self.assertEqual({len(text) for text in dels}, {len("-112")})
            # The badge before the numbers sits at the summary chip's column,
            # new-badge row included.
            chip = summary.query_one(".diff-filter")
            for row in rows:
                source = row.query_one(".diff-file-source")
                self.assertEqual(source.region.x, chip.region.x)

    async def test_a_count_change_keeps_every_other_row(self) -> None:
        """A refresh that changes one file's counts must not remount the
        whole list (that loses scroll position and the active highlight)."""
        from textual.widgets import Static

        from xarness.protocol import DiffFileInfo, DiffInfo
        from xarness.tui.widgets import DiffSummary

        app = make_app()
        async with run_app(app):
            summary = app.query_one("#diff-summary", DiffSummary)
            await summary.set_summary(DiffInfo(files=[
                DiffFileInfo(path="a.py", additions=1, deletions=0, kind="agent"),
                DiffFileInfo(path="b.py", additions=2, deletions=0, kind="mine"),
            ]))
            first = list(summary.query(".diff-file-row"))
            summary.show_diff(("a.py", None), "--- a/a.py\n+++ b/a.py\n@@\n-x\n+y\n")

            await summary.set_summary(DiffInfo(files=[
                DiffFileInfo(path="a.py", additions=5, deletions=1, kind="agent"),
                DiffFileInfo(path="b.py", additions=2, deletions=0, kind="mine"),
            ]))
            after = list(summary.query(".diff-file-row"))
            self.assertIs(after[0], first[0])
            self.assertIs(after[1], first[1])
            self.assertEqual(
                after[0].query_one(".diff-file-add", Static).render().plain, "+5"
            )
            self.assertTrue(after[0].has_class("active"))
            # A combined row carries the badge too: own work in the accent.
            own = after[1].query_one(".diff-file-source", Static)
            self.assertEqual(own.render().plain, "agent")
            self.assertTrue(own.has_class("mine"))

    async def test_section_rows_and_their_children_are_reused(self) -> None:
        """Section headers and their unfolded rows are keyed too: a refresh
        keeps the widgets when only the counts moved."""
        from textual.widgets import Static

        from xarness.protocol import DiffFileInfo, DiffGroup, DiffInfo
        from xarness.tui.widgets import DiffSummary

        app = make_app()
        async with run_app(app) as pilot:
            summary = app.query_one("#diff-summary", DiffSummary)

            def stat(files: int, ignored: int) -> DiffInfo:
                return DiffInfo(
                    files=[DiffFileInfo(
                        path="x.py", additions=files, deletions=0,
                        kind="drift",
                    )],
                    groups=[DiffGroup(prefix="build/", files=ignored, kind="auto")],
                    ignored_files=ignored,
                )

            await summary.set_summary(stat(1, 2))
            await summary.toggle_section("drift")
            await summary.toggle_section("auto")
            await pilot.pause()
            drift = summary.query_one(".diff-section-row.drift")
            auto = summary.query_one(".diff-section-row.auto")
            row = summary.query_one(".diff-file-row")
            group = summary.query_one(".diff-group-row")
            await summary.set_summary(stat(3, 4))
            await pilot.pause()
            self.assertIs(summary.query_one(".diff-section-row.drift"), drift)
            self.assertIs(summary.query_one(".diff-section-row.auto"), auto)
            self.assertIs(summary.query_one(".diff-file-row"), row)
            self.assertIs(summary.query_one(".diff-group-row"), group)
            count = group.query_one(".diff-group-count", Static)
            self.assertEqual(count.render().plain, "4 files")
            self.assertEqual(
                auto.query_one(".diff-section-count", Static).render().plain,
                "4 files",
            )


class DiffScopeChipTest(unittest.IsolatedAsyncioTestCase):
    """The summary line carries one ``agent`` chip; clicking it cycles the
    scope between this session plus other sessions and this session alone.
    Every section below the list is always there, folded."""

    def _stat(self):
        from xarness.protocol import DiffFileInfo, DiffGroup, DiffInfo

        return DiffInfo(
            files=[
                DiffFileInfo(path="own.py", additions=1, deletions=0, kind="mine"),
                DiffFileInfo(path="other.py", additions=2, deletions=2, kind="agent"),
                DiffFileInfo(path="yours.py", additions=3, deletions=1, kind="drift"),
            ],
            groups=[DiffGroup(prefix="build/", files=7, kind="auto")],
            ignored_files=7,
            additions=6,
            deletions=3,
        )

    async def _summary(self, pilot, app):
        from xarness.tui.widgets import DiffSummary

        # The host builds the rows for the session's scope; these rows are
        # already one per author, so both scopes can be served from the same
        # payload (nothing here is a shared file).
        async def diff(name, path=None, author=None):
            return self._stat()

        app.client.diff = diff
        summary = app.query_one("#diff-summary", DiffSummary)
        await summary.set_summary(self._stat())
        await pilot.pause()
        return summary

    @staticmethod
    def _chip_colors(text) -> list[tuple[str, str]]:
        return [
            (text.plain[span.start:span.end], str(span.style))
            for span in text.spans
        ]

    async def test_the_chip_reads_agent_in_two_tones(self) -> None:
        from xarness import theme
        from xarness.tui.diff_summary import _chip_text

        app = make_app()
        async with run_app(app, size=(120, 40)) as pilot:
            summary = await self._summary(pilot, app)
            chip = summary.query_one(".diff-filter")
            self.assertEqual(chip.render().plain, "agent")
            palette = theme.PALETTE
            both = _chip_text(True)
            self.assertEqual(both.plain, "agent")
            self.assertEqual(
                self._chip_colors(both),
                [("age", palette["accent"]), ("nt", palette["muted"])],
            )
            own = _chip_text(False)
            self.assertEqual(own.plain, "agent")
            self.assertEqual(
                self._chip_colors(own), [("agent", palette["accent"])]
            )

    async def test_clicking_the_chip_cycles_the_scope(self) -> None:
        from textual.widgets import Static


        app = make_app()
        async with run_app(app, size=(120, 40)) as pilot:
            summary = await self._summary(pilot, app)
            summary.add_class("expanded")
            await pilot.pause()
            self.assertEqual(
                [r.diff_key for r in summary.query(".diff-file-row")],
                [("own.py", None), ("other.py", None)],
            )

            def loc() -> tuple[str, str]:
                return tuple(  # type: ignore[return-value]
                    summary.query_one(f"#diff-summary-{side}", Static).render().plain
                    for side in ("add", "del")
                )

            self.assertEqual(loc(), ("+3", "-2"))

            chip = summary.query_one(".diff-filter")
            await pilot.click(chip)
            await pilot.pause()
            self.assertEqual(summary.scope, "mine")
            # Only this session's file leads; the other session's work has
            # moved into the (still folded) dimmed "agent" section.
            self.assertEqual(
                [r.diff_key for r in summary.query(".diff-file-row")],
                [("own.py", None)],
            )
            self.assertEqual(
                [r.diff_section for r in summary.query(".diff-section-row")],
                ["agent", "auto", "drift"],
            )
            from textual.widgets import Static

            others = summary.query_one(".diff-section-row.agent")
            self.assertEqual(
                others.query_one(".diff-section-name", Static).render().plain,
                "agent",
            )
            self.assertEqual(loc(), ("+1", "-0"))
            self.assertEqual(chip.render().plain, "agent")
            self.assertEqual(len(chip.render().spans), 1)

            # Unfolding the section brings the other session's row back.
            await summary.toggle_section("agent")
            await pilot.pause()
            self.assertEqual(
                [r.diff_key for r in summary.query(".diff-file-row")],
                [("own.py", None), ("other.py", None)],
            )

            # Clicking again includes every session inline, which leaves the
            # "agent" section with nothing to hold: it goes away.
            await pilot.click(chip)
            await pilot.pause()
            self.assertEqual(summary.scope, "all")
            self.assertEqual(
                [r.diff_section for r in summary.query(".diff-section-row")],
                ["auto", "drift"],
            )
            self.assertEqual(loc(), ("+3", "-2"))

    async def test_collapsing_a_section_closes_its_open_diff(self) -> None:

        app = make_app()
        async with run_app(app, size=(120, 40)) as pilot:
            summary = await self._summary(pilot, app)
            summary.add_class("expanded")
            summary.show_diff(("yours.py", None), "--- a/yours.py\n+++ b/yours.py\n@@\n-x\n+y\n")
            await pilot.pause()
            self.assertTrue(summary.diff_shown)
            # show_diff unfolded the drift section for it.
            self.assertEqual(
                [r.diff_key for r in summary.query(".diff-file-row")],
                [("own.py", None), ("other.py", None), ("yours.py", None)],
            )

            await summary.toggle_section("drift")
            await pilot.pause()
            # The pane closes with its row instead of floating orphaned.
            self.assertFalse(summary.diff_shown)
            self.assertFalse(summary.query_one("#diff-content").has_class("show"))

    async def test_sections_are_visible_even_when_empty(self) -> None:
        from xarness.protocol import DiffFileInfo, DiffInfo
        from xarness.tui.widgets import DiffSummary

        app = make_app()
        async with run_app(app, size=(120, 40)) as pilot:
            summary = app.query_one("#diff-summary", DiffSummary)
            await summary.set_summary(DiffInfo(files=[
                DiffFileInfo(path="own.py", additions=1, deletions=0, kind="mine"),
            ]))
            await pilot.pause()
            self.assertEqual(
                [r.diff_section for r in summary.query(".diff-section-row")],
                ["auto", "drift"],
            )
            self.assertTrue(summary.query_one(".diff-section-row.drift").has_class("empty"))

    async def test_stepping_skips_folded_files(self) -> None:

        app = make_app()
        async with run_app(app, size=(120, 40)) as pilot:
            summary = await self._summary(pilot, app)
            summary.add_class("expanded")
            await pilot.pause()
            self.assertEqual(
                summary._visible_keys(), [("own.py", None), ("other.py", None)]
            )
            self.assertFalse(summary._request_neighbour(("other.py", None), 1))

            await summary.toggle_section("drift")
            await pilot.pause()
            self.assertEqual(
                summary._visible_keys(),
                [("own.py", None), ("other.py", None), ("yours.py", None)],
            )
            self.assertTrue(summary._request_neighbour(("other.py", None), 1))

    async def test_narrowing_the_scope_keeps_the_open_diff(self) -> None:
        """Switching to this session moves another session's open diff into
        the dimmed `agent` section — unfolding it with the pane rather
        than closing the diff the user is reading."""

        app = make_app()
        async with run_app(app, size=(120, 40)) as pilot:
            summary = await self._summary(pilot, app)
            summary.add_class("expanded")
            summary.show_diff(("other.py", None), "--- a/other.py\n+++ b/other.py\n@@\n-x\n+y\n")
            await pilot.pause()
            self.assertTrue(summary.diff_shown)

            await pilot.click(summary.query_one(".diff-filter"))
            await pilot.pause()
            self.assertTrue(summary.diff_shown)
            self.assertIn("agent", summary._expanded)
            self.assertEqual(
                [r.diff_key for r in summary.query(".diff-file-row")],
                [("own.py", None), ("other.py", None)],
            )

    async def test_a_chip_click_is_saved_on_the_session(self) -> None:
        """The choice is session state: the widget posts it and the app sends
        SetViewState so a reattach (or a later run) shows the same rows."""

        client = FakeClient(make_snapshot())
        app = make_app(client)
        async with run_app(app, size=(120, 40)) as pilot:
            summary = await self._summary(pilot, app)
            summary.add_class("expanded")
            await pilot.pause()
            chip = summary.query_one(".diff-filter")
            await pilot.click(chip)
            for _ in range(3):
                await pilot.pause()
            sent = [c for _, c in client.commands if c.type == "set_view_state"]
            self.assertEqual([c.view.diff_scope for c in sent], ["mine"])

    async def test_the_saved_filter_is_restored_on_attach(self) -> None:
        """A session saved scoped to this session reopens the same way: the
        status carries the choice, and the panel applies it without a click."""

        snap = make_snapshot()
        snap.status.view.diff_scope = "mine"
        app = make_app(FakeClient(snap))
        async with run_app(app, size=(120, 40)) as pilot:
            summary = await self._summary(pilot, app)
            summary.add_class("expanded")
            await pilot.pause()
            self.assertEqual(summary.scope, "mine")
            self.assertEqual(
                [r.diff_key for r in summary.query(".diff-file-row")],
                [("own.py", None)],
            )
            self.assertEqual(
                [r.diff_section for r in summary.query(".diff-section-row")],
                ["agent", "auto", "drift"],
            )

    async def test_a_draft_keeps_its_scope_for_the_session_it_starts(self) -> None:
        """No session yet: the click is not lost — the selection rides out on
        the create request, so the first session starts scoped."""

        client = FakeClient(make_snapshot())
        app = make_app(client, session_name=None)
        async with run_app(app, size=(120, 40)) as pilot:
            summary = await self._summary(pilot, app)
            await summary.toggle_scope()
            await pilot.pause()
            # A draft must not revert the click on the next status refresh.
            app._refresh_bars()
            await pilot.pause()
            self.assertEqual(summary.scope, "mine")
            self.assertEqual(app._selected_diff_scope(), "mine")
            self.assertEqual(
                [c for _, c in client.commands if c.type == "set_view_state"], []
            )

    async def test_a_stale_status_does_not_revert_the_click(self) -> None:
        """Clicking the chip updates the panel immediately; a StatusChanged
        already in flight (with the pre-click value) must not undo it before
        the host's own echo lands."""

        client = FakeClient(make_snapshot())
        app = make_app(client)
        async with run_app(app, size=(120, 40)) as pilot:
            summary = await self._summary(pilot, app)
            summary.add_class("expanded")
            await pilot.pause()
            chip = summary.query_one(".diff-filter")
            await pilot.click(chip)
            await pilot.pause()
            # A status computed before the host applied the command.
            client.queue.put_nowait(
                protocol.StatusChanged(status=make_snapshot().status))
            for _ in range(3):
                await pilot.pause()
            self.assertEqual(summary.scope, "mine")
            # The host's echo settles it; from then on status is authoritative
            # (another client's choice included).
            status = make_snapshot().status
            status.view.diff_scope = "mine"
            client.queue.put_nowait(protocol.StatusChanged(status=status))
            for _ in range(3):
                await pilot.pause()
            self.assertEqual(summary.scope, "mine")
            status = make_snapshot().status
            status.view.diff_scope = "all"
            client.queue.put_nowait(protocol.StatusChanged(status=status))
            for _ in range(3):
                await pilot.pause()
            self.assertEqual(summary.scope, "all")


class CompactingStripTest(unittest.IsolatedAsyncioTestCase):
    """Compaction has no ProcessingStarted: the status event drives the strip."""

    async def test_compacting_status_shows_and_then_clears_the_strip(self) -> None:
        client = FakeClient(make_snapshot("compacting"))
        app = make_app(client)
        async with run_app(app) as pilot:
            strips = [e for e in app._flow_model.entries
                      if isinstance(e.item, tflow.PendingItem)]
            self.assertEqual([e.item.label for e in strips], ["Compacting"])
            self.assertTrue(strips[0].item.streaming())
            client.queue.put_nowait(
                protocol.StatusChanged(status=make_snapshot("running").status))
            await pilot.pause()
            self.assertFalse([e for e in app._flow_model.entries
                              if isinstance(e.item, tflow.PendingItem)
                              and e.alive])
            self.assertEqual(app._phase, "idle")


class ShimmerDirectionTest(unittest.TestCase):
    """The sweep must start at the base color and peak at the highlight."""

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
        # Peak is white, not the theme's text color, in every palette.
        from xarness import theme
        for name in theme.THEMES:
            theme.set_theme(name)
            self.assertEqual(theme.SHIMMER_PROCESSING[1], "#ffffff")
            self.assertEqual(theme.SHIMMER_THINKING[1], "#ffffff")
        theme.set_theme(theme.DEFAULT_THEME)


class UndoCommandTest(unittest.IsolatedAsyncioTestCase):
    async def test_plain_undo_puts_the_user_text_back_in_the_input(self) -> None:
        from xarness.tui.widgets import ChatInput

        client = FakeClient(make_snapshot())
        app = make_app(client)
        async with run_app(app) as pilot:
            await app._run_undo()
            await pilot.pause()
            self.assertEqual(app.query_one("#chat-input", ChatInput).text, "put back")
            self.assertIn("rollback", [command.type for _, command in client.commands])


class UndoAffordanceClickTest(unittest.IsolatedAsyncioTestCase):
    """Click-to-undo is hit-tested: the whole user message box arms the
    ``↩ undo to here`` line it shows, only that line's own text fires it, and
    a click anywhere else closes it."""

    async def _run(self, script) -> None:
        from xarness.tui.flow import UserItem

        client = FakeClient(make_snapshot())
        app = make_app(client)
        async with app.run_test(size=(60, 20)) as pilot:
            for _ in range(5):
                await pilot.pause()
            app._flow_model.append(UserItem(text="hello there", message_id="m1"))
            for _ in range(10):
                await pilot.pause()
            await script(pilot, client, app)

    def _located(self, app, needle: str) -> tuple[int, int]:
        """(x, y) of ``needle`` in the rendered log, in widget coordinates."""
        view = app.query_one("#chat-log")
        for y in range(view.row_count):
            x = view.row_text(y).find(needle)
            if x >= 0:
                return x, y
        raise AssertionError(f"{needle!r} not painted; rows: " + repr(
            [view.row_text(y) for y in range(view.row_count)]))

    def _armed(self, app) -> bool:
        return app._flow_model.entries[-1].item.undo_armed

    async def test_clicking_the_message_arms_the_affordance(self) -> None:
        async def script(pilot, client, app):
            x, y = self._located(app, "hello there")
            await pilot.click("#chat-log", offset=(x, y))
            await pilot.pause()
            self.assertTrue(self._armed(app))
            self._located(app, "undo to here")

        await self._run(script)

    async def test_the_whole_message_box_is_the_click_target(self) -> None:
        async def script(pilot, client, app):
            x, y = self._located(app, "hello there")
            # Every cell of the box arms it: the far right of the text row and
            # the blank padding rows above and below the text. Clicking the
            # message again closes it.
            for ax, ay in ((40, y), (0, y - 1), (40, y + 1)):
                await pilot.click("#chat-log", offset=(ax, ay))
                await pilot.pause()
                self.assertTrue(self._armed(app), (ax, ay))
                await pilot.click("#chat-log", offset=(40, y))
                await pilot.pause()
                self.assertFalse(self._armed(app), (ax, ay))
            self.assertNotIn("rollback", [c.type for _, c in client.commands])

        await self._run(script)

    async def test_only_the_undo_lines_text_fires_the_rollback(self) -> None:
        async def script(pilot, client, app):
            x, y = self._located(app, "hello there")
            await pilot.click("#chat-log", offset=(x, y))
            await pilot.pause()
            self.assertTrue(self._armed(app))
            # A click elsewhere in the already-armed message just closes it.
            await pilot.click("#chat-log", offset=(x, y))
            await pilot.pause()
            self.assertNotIn("rollback", [c.type for _, c in client.commands])
            self.assertFalse(self._armed(app))
            # Arm again, then click the blank tail of the undo line: not the
            # affordance's text, so that only closes it too.
            await pilot.click("#chat-log", offset=(x, y))
            await pilot.pause()
            _, uy = self._located(app, "undo to here")
            await pilot.click("#chat-log", offset=(40, uy))
            await pilot.pause()
            self.assertNotIn("rollback", [c.type for _, c in client.commands])
            self.assertFalse(self._armed(app))
            # Arm one last time and click the affordance's own text.
            await pilot.click("#chat-log", offset=(x, y))
            await pilot.pause()
            ux, uy = self._located(app, "undo to here")
            await pilot.click("#chat-log", offset=(ux + 2, uy))
            await pilot.pause()
            await pilot.pause()
            rollbacks = [c for _, c in client.commands if c.type == "rollback"]
            self.assertEqual(len(rollbacks), 1)
            self.assertEqual(rollbacks[0].message_id, "m1")

        await self._run(script)

    async def test_clicking_outside_the_log_closes_it(self) -> None:
        async def script(pilot, client, app):
            x, y = self._located(app, "hello there")
            await pilot.click("#chat-log", offset=(x, y))
            await pilot.pause()
            self.assertTrue(self._armed(app))
            await pilot.click("#chat-input")
            await pilot.pause()
            self.assertFalse(self._armed(app))

        await self._run(script)


class RelativeTimeFormatTest(unittest.TestCase):
    def test_days_are_not_rendered_as_hours(self) -> None:
        from datetime import datetime, timedelta, timezone

        from xarness.tui.list_picker import relative_time

        now = datetime.now(timezone.utc)
        self.assertEqual(relative_time((now - timedelta(days=2)).isoformat()), "2d")
        self.assertEqual(relative_time((now - timedelta(hours=3)).isoformat()), "3h00m")
        self.assertEqual(relative_time(None), "?")


class ImageTokenCursorTest(unittest.IsolatedAsyncioTestCase):
    """A pasted-image token behaves as one character in the input."""

    async def _input(self, text: str, offset: int):
        from textual.app import App, ComposeResult
        from textual.widgets.text_area import Selection

        from xarness.images import ImageRef, token_text
        from xarness.tui.widgets import ChatInput

        class Host(App):
            def compose(self) -> ComposeResult:
                yield ChatInput(id="chat-input")

        token = token_text(ImageRef(
            id="0" * 64, name="pic.png", mime="image/png", width=8, height=8,
        ))
        full = text.replace("•", token)
        app = Host()
        async with app.run_test() as pilot:
            chat_input = app.query_one("#chat-input", ChatInput)
            chat_input.load_text(full)
            location = chat_input.document.get_location_from_index(
                len(full) if offset < 0 else offset,
            )
            chat_input.selection = Selection(location, location)
            await pilot.pause()
            yield chat_input

    async def test_left_arrow_skips_the_token(self) -> None:
        async for chat_input in self._input("ab •", -1):
            chat_input.action_cursor_left()
            # one press crosses the whole token, landing on its start
            self.assertEqual(chat_input.cursor_location, (0, len("ab ")))

    async def test_right_arrow_skips_the_token(self) -> None:
        async for chat_input in self._input("• cd", 0):
            chat_input.action_cursor_right()
            # one press crosses the whole token, landing past its end
            self.assertEqual(
                chat_input.cursor_location,
                chat_input.document.get_location_from_index(
                    len(chat_input.text) - len(" cd"),
                ),
            )

    async def test_shift_left_selects_the_whole_token(self) -> None:
        async for chat_input in self._input("ab •", -1):
            chat_input.action_cursor_left(select=True)
            self.assertEqual(chat_input.selected_text.count("🖼"), 1)
            self.assertEqual(
                chat_input.selected_text,
                chat_input.text[len("ab "):],
            )

    async def test_backspace_after_token_removes_it(self) -> None:
        async for chat_input in self._input("ab •", -1):
            chat_input.action_delete_left()
            self.assertEqual(chat_input.text, "ab ")
            self.assertEqual(chat_input.attachments, [])


class ImageTokenSelectedStyleTest(unittest.IsolatedAsyncioTestCase):
    """A selected token's chip swaps to the selected style."""

    async def test_token_highlight_follows_selection(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets.text_area import Selection

        from xarness.images import ImageRef, token_text
        from xarness.tui.widgets import ChatInput

        token = token_text(ImageRef(
            id="0" * 64, name="pic.png", mime="image/png", width=8, height=8,
        ))
        full = f"ab {token} cd"

        class Host(App):
            def compose(self) -> ComposeResult:
                yield ChatInput()

        app = Host()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            chat_input.load_text(full)
            await pilot.pause()

            def highlight_names() -> list[str]:
                return [
                    name
                    for ranges in chat_input._highlights.values()
                    for _, _, name in ranges
                ]

            self.assertEqual(highlight_names(), ["image-token"])
            end = chat_input.document.get_location_from_index(len(full))
            start = chat_input.document.get_location_from_index(2)
            chat_input.selection = Selection(start, end)
            await pilot.pause()
            self.assertEqual(highlight_names(), ["image-token-selected"])
            chat_input.selection = Selection(end, end)
            await pilot.pause()
            self.assertEqual(highlight_names(), ["image-token"])


class AskFlowTest(unittest.IsolatedAsyncioTestCase):
    """The ask bar is a picker: up/down move the highlight, enter takes it,
    and typing in the input is always the write-in."""

    @staticmethod
    def _status(*questions) -> Status:
        return Status(
            state="idle",
            session_name="s1", workspace="/tmp/ws", mode="write",
            model_name="m", model_display_name="m", model_id="m",
            pending_ask=protocol.PendingAsk(ask_id="a1", questions=list(questions)),
        )

    async def _pending(self, pilot, client, app, *questions):
        client.queue.put_nowait(protocol.StatusChanged(status=self._status(*questions)))
        await pilot.pause()

    async def test_choices_render_as_a_picker(self) -> None:
        from xarness.tui.widgets import AskBar

        client = FakeClient(make_snapshot())
        async with run_app(make_app(client)) as pilot:
            app = pilot.app
            await self._pending(pilot, client, app, protocol.AskQuestion(
                question="Which storage?", options=["sqlite", "postgres"]))
            bar = app.query_one("#ask-bar", AskBar)
            self.assertTrue(bar.has_class("visible"))
            shown = bar.render().plain
            self.assertIn("Question 1/1: Which storage?", shown)
            self.assertIn("❯ 1) sqlite", shown)  # first choice highlighted
            self.assertIn("\n    2) postgres", shown)
            self.assertIn("↑/↓ choose · enter picks · or type your own answer", shown)

    async def test_arrows_move_the_highlight_and_enter_takes_it(self) -> None:
        from xarness.tui.widgets import AskBar

        client = FakeClient(make_snapshot())
        async with run_app(make_app(client)) as pilot:
            app = pilot.app
            await self._pending(pilot, client, app, protocol.AskQuestion(
                question="Which storage?", options=["sqlite", "postgres"]))
            await pilot.press("down")
            await pilot.pause()
            shown = app.query_one("#ask-bar", AskBar).render().plain
            self.assertIn("❯ 2) postgres", shown)
            self.assertIn("\n    1) sqlite", shown)

            await pilot.press("enter")  # empty input: take the highlight
            await pilot.pause()
            sent = [c for _, c in client.commands if c.type == "answer_ask"]
            self.assertEqual([c.answers for c in sent], [["postgres"]])

    async def test_typing_is_the_write_in(self) -> None:
        from xarness.tui.widgets import ChatInput

        client = FakeClient(make_snapshot())
        async with run_app(make_app(client)) as pilot:
            app = pilot.app
            await self._pending(pilot, client, app, protocol.AskQuestion(
                question="Which storage?", options=["sqlite", "postgres"]))
            app.query_one("#chat-input", ChatInput).text = "duckdb"
            await pilot.press("enter")
            await pilot.pause()
            sent = [c for _, c in client.commands if c.type == "answer_ask"]
            self.assertEqual([c.answers for c in sent], [["duckdb"]])

    async def test_a_free_text_question_shows_no_choices(self) -> None:
        from xarness.tui.widgets import AskBar, ChatInput

        client = FakeClient(make_snapshot())
        async with run_app(make_app(client)) as pilot:
            app = pilot.app
            await self._pending(pilot, client, app,
                                protocol.AskQuestion(question="Project name?"))
            self.assertEqual(
                app.query_one("#ask-bar", AskBar).render().plain,
                "Question 1/1: Project name?",
            )
            chat_input = app.query_one("#chat-input", ChatInput)
            self.assertFalse(chat_input.picker_active)

            # Enter on empty input has nothing to take: no answer is sent.
            await pilot.press("enter")
            await pilot.pause()
            self.assertEqual(
                [c for _, c in client.commands if c.type == "answer_ask"], [],
            )
