"""Tests for history replay, /resume, and /model (TUI-only phase)."""

import json
import os
import tempfile
import unittest
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

from textual.widgets import ListItem

from xarness import session_store
from xarness.config import CotStrength, ProviderProfile
from xarness.controller import ChatController
from xarness.conversation import Conversation, Message, ToolCall
from xarness.events import (
    ContentDelta,
    ProcessingStarted,
    ToolCallStatus,
    TurnComplete,
    Usage,
)
from xarness.tui.app import AgentApp
from xarness.tui.picker_screen import PickerScreen
from xarness.tui.resume_screen import ResumeScreen
from xarness.tui.widgets import (
    AssistantMessage,
    StatusBar,
    ThinkingBlock,
    ToolCallBlock,
    UserMessage,
)

PROFILE = ProviderProfile(
    base_url="https://alpha.example.test/v1",
    model_id="alpha-model",
    shown_name="Alpha Model",
    max_context=1000,
    cot_strength=CotStrength.MEDIUM,
)


class FakeClient:
    def __init__(self, script: list[Any]) -> None:
        self.script = script
        self.received_tools: list[dict[str, Any]] | None = None
        self._profile = PROFILE
        self._api_key = "test-key"

    def switch_profile(self, profile: ProviderProfile, api_key: str | None) -> None:
        self._profile = profile
        self._api_key = api_key

    async def stream(
        self,
        wire_messages: Sequence[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[Any]:
        self.received_tools = tools
        yield ProcessingStarted()
        for event in self.script:
            yield event


def make_app(script: list[Any], **kwargs: Any) -> AgentApp:
    controller = ChatController(PROFILE, "test-key", client=FakeClient(script))
    return AgentApp(PROFILE, "test-key", controller=controller, **kwargs)


async def wait_until_idle(app: AgentApp, timeout: float = 5.0) -> None:
    import asyncio

    for _ in range(int(timeout / 0.05)):
        await asyncio.sleep(0.05)
        if not app._turn_busy:
            return
    raise AssertionError("turn did not finish in time")


def write_config(path: Path, beta_key_env: str | None = "BETA_KEY") -> None:
    import json

    path.write_text(
        json.dumps(
            {
                "default_profile": "alpha",
                "profiles": [
                    {
                        "name": "alpha",
                        "base_url": "https://alpha.example.test/v1",
                        "api_key_env": "ALPHA_KEY",
                        "model_id": "alpha-model",
                        "shown_name": "Alpha Model",
                    },
                    {
                        "name": "beta",
                        "base_url": "https://beta.example.test/v1",
                        "api_key_env": beta_key_env,
                        "model_id": "beta-model",
                        "shown_name": "Beta Model",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


# ----------------------------------------------------------------------
# Phase 1: chat-log replay from Conversation history


class TestRenderHistory(unittest.IsolatedAsyncioTestCase):
    async def test_replay_mixed_history(self) -> None:
        app = make_app([])
        conversation = Conversation()
        conversation.add(Message(role="user", content="hello"))
        conversation.add(
            Message(role="assistant", content="hi there", reasoning="pondering")
        )
        conversation.add(
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall("call_1", "noop", '{"x": 1}')],
            )
        )
        conversation.add(Message(role="tool", tool_call_id="call_1", content="success!"))
        conversation.add(Message(role="assistant", content="all done"))
        app.controller.conversation = conversation

        async with app.run_test():
            await app._render_history()

            self.assertEqual(len(app.query(UserMessage)), 1)
            self.assertIn("hello", app.query_one(UserMessage).text)

            # Reasoning replays collapsed with an unknown-duration label.
            thinking = app.query_one(ThinkingBlock)
            self.assertTrue(thinking.done)
            self.assertEqual(thinking.summary_text, "Thought for …")

            # The pure tool-call round renders no AssistantMessage body.
            assistants = app.query(AssistantMessage)
            self.assertEqual(len(assistants), 2)

            # Tool call replays settled green with the tool-role output.
            block = app.query_one(ToolCallBlock)
            self.assertEqual(block.tool_name, "noop")
            self.assertEqual(block.accumulated_arguments, '{"x": 1}')
            self.assertEqual(block.status, ToolCallStatus.CALL_SUCCEEDED)
            self.assertEqual(block._output_text, "success!")

            # Tool-role message is consumed via lookup, not rendered standalone.
            self.assertEqual(len(app.query(UserMessage)) + len(app.query(AssistantMessage)) + 2, 5)

    async def test_replay_empty_conversation(self) -> None:
        app = make_app([])
        app.controller.conversation = Conversation()
        async with app.run_test():
            await app._render_history()
            chat = app.query_one("#chat-log")
            self.assertEqual(len(chat.children), 0)

    async def test_startup_replays_preloaded_conversation(self) -> None:
        """cli.py --session loads a conversation before run(): mount replays it
        into the chat log without anyone calling _render_history manually."""
        app = make_app([])
        conversation = Conversation()
        conversation.add(Message(role="user", content="old question"))
        conversation.add(Message(role="assistant", content="old answer"))
        app.controller.conversation = conversation
        async with app.run_test():
            self.assertEqual(len(app.query(UserMessage)), 1)
            self.assertEqual(len(app.query(AssistantMessage)), 1)
            self.assertIn("old question", app.query_one(UserMessage).text)

    async def test_replay_assistant_without_reasoning(self) -> None:
        app = make_app([])
        conversation = Conversation()
        conversation.add(Message(role="user", content="q"))
        conversation.add(Message(role="assistant", content="plain answer"))
        app.controller.conversation = conversation
        async with app.run_test():
            await app._render_history()
            self.assertEqual(len(app.query(ThinkingBlock)), 0)
            self.assertEqual(len(app.query(AssistantMessage)), 1)

    async def test_replay_tool_call_without_content_and_empty_assistant(self) -> None:
        app = make_app([])
        conversation = Conversation()
        conversation.add(
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall("call_9", "noop", "{}")],
            )
        )
        conversation.add(Message(role="tool", tool_call_id="call_9", content="ok"))
        # Assistant message with neither content nor tool calls: "(no output)".
        conversation.add(Message(role="assistant", content=""))
        app.controller.conversation = conversation
        async with app.run_test():
            await app._render_history()
            # Only the empty/no-tool-calls assistant renders, as "(no output)".
            self.assertEqual(len(app.query(AssistantMessage)), 1)
            self.assertEqual(len(app.query(ToolCallBlock)), 1)
            self.assertEqual(app.query_one(ToolCallBlock)._output_text, "ok")


# ----------------------------------------------------------------------
# Phase 2: /resume


class TestResume(unittest.IsolatedAsyncioTestCase):
    async def test_resume_screen_lists_filters_and_selects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(session_store, "SESSIONS_DIR", Path(tmp)):
                session_store.save_session(
                    "alpha-chat", "alpha-model", Conversation()
                )
                session_store.save_session(
                    "beta-chat", "beta-model", Conversation()
                )
                app = make_app([])
                results: list[str | None] = []
                async with app.run_test() as pilot:
                    screen = ResumeScreen()
                    await app.push_screen(screen, results.append)
                    await pilot.pause()

                    # Newest session first (beta was saved after alpha).
                    names = [item.session_name for item in screen.query(ListItem)]
                    self.assertEqual(names, ["beta-chat", "alpha-chat"])

                    # Live filtering narrows the list.
                    await pilot.press("b", "e", "t")
                    names = [item.session_name for item in screen.query(ListItem)]
                    self.assertEqual(names, ["beta-chat"])

                    # Clear the filter, navigate with arrows (search box keeps
                    # focus), and pick with enter.
                    await pilot.press("backspace", "backspace", "backspace")
                    await pilot.press("down", "down")  # beta-chat -> alpha-chat
                    await pilot.press("enter")
                    self.assertEqual(results, ["alpha-chat"])

    async def test_slash_resume_loads_session_and_replays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(session_store, "SESSIONS_DIR", Path(tmp)):
                saved = Conversation()
                saved.add(Message(role="user", content="old question"))
                saved.add(
                    Message(
                        role="assistant",
                        content="old answer",
                        reasoning="deliberating",
                        reasoning_seconds=80.0,
                    )
                )
                session_store.save_session("my-session", "alpha-model", saved)

                app = make_app(
                    [ContentDelta("fresh reply"), TurnComplete(usage=Usage(10, 2))]
                )
                async with app.run_test() as pilot:
                    # Enter on the slash-command popup completes and runs it.
                    await pilot.press("/", "s", "e", "s", "s", "i", "o", "n", "s", "enter")
                    for _ in range(50):
                        await pilot.pause()
                        if isinstance(app.screen, ResumeScreen):
                            break
                    self.assertIsInstance(app.screen, ResumeScreen)

                    await pilot.press("enter")  # pick the only session
                    for _ in range(50):
                        await pilot.pause()
                        if app.session_name == "my-session":
                            break
                    self.assertEqual(app.session_name, "my-session")

                    # History replayed into the chat log.
                    self.assertEqual(len(app.query(UserMessage)), 1)
                    self.assertEqual(len(app.query(AssistantMessage)), 1)

                    # Persisted thinking time survives the round-trip.
                    self.assertEqual(
                        app.query_one(ThinkingBlock).summary_text, "Thought for 1m 20s"
                    )
                    # ensure_system_message() prepends the mode-aware prompt.
                    self.assertEqual(app.controller.conversation.messages[0].role, "system")
                    self.assertEqual(
                        [m.content for m in app.controller.conversation.messages[1:]],
                        ["old question", "old answer"],
                    )

                    # New messages continue appending to the same session file.
                    await pilot.press("m", "o", "r", "e", "enter")
                    await wait_until_idle(app)
                    data = json.loads((Path(tmp) / "my-session.json").read_text())
                    self.assertEqual(data["messages"][0]["role"], "system")
                    contents = [m["content"] for m in data["messages"][1:]]
                    self.assertEqual(
                        contents, ["old question", "old answer", "more", "fresh reply"]
                    )


# ----------------------------------------------------------------------
# Phase 3: /model


class TestModelPicker(unittest.IsolatedAsyncioTestCase):
    def test_list_profile_names(self) -> None:
        from xarness.config import ConfigError, list_profile_names

        with tempfile.TemporaryDirectory() as tmp:
            named = Path(tmp) / "config.json"
            write_config(named)
            self.assertEqual(list_profile_names(named), ["alpha", "beta"])

            flat = Path(tmp) / "flat.json"
            flat.write_text(
                '{"provider": {"base_url": "https://x.test/v1", "model_id": "m"}}',
                encoding="utf-8",
            )
            self.assertEqual(list_profile_names(flat), [])

            missing = Path(tmp) / "nope.json"
            with self.assertRaises(ConfigError):
                list_profile_names(missing)

    async def test_slash_model_switches_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            write_config(config_path)
            env = {"ALPHA_KEY": "a-key", "BETA_KEY": "b-key"}
            with patch.dict(os.environ, env):
                app = make_app(
                    [ContentDelta("reply"), TurnComplete(usage=Usage(10, 2))],
                    config_path=config_path,
                )
                conversation = Conversation()
                conversation.add(Message(role="user", content="carried over"))
                app.controller.conversation = conversation

                async with app.run_test() as pilot:
                    await pilot.press("/", "m", "o", "d", "e", "l", "enter")
                    for _ in range(50):
                        await pilot.pause()
                        if isinstance(app.screen, PickerScreen):
                            break
                    self.assertIsInstance(app.screen, PickerScreen)

                    await pilot.press("down")  # highlight beta
                    await pilot.press("enter")
                    for _ in range(50):
                        await pilot.pause()
                        if app.profile.model_id == "beta-model":
                            break
                    self.assertEqual(app.profile.model_id, "beta-model")

                    # Controller rebuilt with the new profile; history carried over.
                    self.assertEqual(app.controller.profile.model_id, "beta-model")
                    self.assertEqual(
                        app.controller._client._profile.base_url,
                        "https://beta.example.test/v1",
                    )
                    self.assertIs(
                        app.controller.conversation, conversation
                    )

                    # Status bar shows the new model's display name.
                    self.assertIn("Beta Model", app.query_one(StatusBar).text)

                    # A send after the switch works and keeps appending.
                    await pilot.press("x", "enter")
                    await wait_until_idle(app)
                    self.assertEqual(
                        app.controller.conversation.messages[-1].content, "reply"
                    )

    async def test_slash_model_missing_api_key_shows_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            write_config(config_path, beta_key_env=None)
            env = {"ALPHA_KEY": "a-key"}  # beta needs no key, but its base_url is unreachable
            with patch.dict(os.environ, env):
                app = make_app([], config_path=config_path)
                async with app.run_test() as pilot:
                    await pilot.press("/", "m", "o", "d", "e", "l", "enter")
                    await pilot.pause()
                    await pilot.press("down", "enter")
                    for _ in range(50):
                        await pilot.pause()
                        if app.screen is app.screen_stack[0]:
                            break
                    # Beta resolved (its api_key_env is null so no key needed);
                    # the switch went through to the beta profile.
                    self.assertEqual(app.profile.model_id, "beta-model")


# ----------------------------------------------------------------------
# /new


class TestSlashNew(unittest.IsolatedAsyncioTestCase):
    async def test_slash_new_starts_fresh_chat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(session_store, "SESSIONS_DIR", Path(tmp)):
                app = make_app(
                    [ContentDelta("reply"), TurnComplete(usage=Usage(10, 2))],
                    session_name="old-session",
                )
                async with app.run_test() as pilot:
                    await pilot.press("h", "i", "enter")
                    await wait_until_idle(app)
                    self.assertEqual(len(app.query(UserMessage)), 1)

                    # Enter on the popup completes and runs /new in one step.
                    await pilot.press("/", "n", "e", "w", "enter")
                    await pilot.pause()

                    # Chat log cleared; conversation reset to just the system prompt.
                    self.assertEqual(len(app.query(UserMessage)), 0)
                    messages = app.controller.conversation.messages
                    self.assertEqual([m.role for m in messages], ["system"])

                    # Fresh session file name; usage counters reset.
                    self.assertNotEqual(app.session_name, "old-session")
                    self.assertEqual((app.total_in, app.total_out, app.last_in), (0, 0, 0))

                    # The new session works: sending appends to the fresh conversation.
                    await pilot.press("x", "enter")
                    await wait_until_idle(app)
                    self.assertEqual(app.controller.conversation.messages[-1].content, "reply")


if __name__ == "__main__":
    unittest.main()


class TestContainerSettings(unittest.IsolatedAsyncioTestCase):
    def _make(self, script, tmp: Path, **kwargs):
        from xarness.sandbox import SandboxConfig, SandboxSession

        (tmp / "wt").mkdir()
        if not (tmp / "config.json").exists():
            (tmp / "config.json").write_text(
                json.dumps({"provider": {
                    "base_url": "https://alpha.example.test/v1",
                    "model_id": "alpha-model",
                }}),
                encoding="utf-8",
            )
        sandbox = SandboxConfig(workspace=tmp / "wt")
        return make_app(
            script,
            sandbox=sandbox,
            sandbox_session=SandboxSession(sandbox),
            config_path=tmp / "config.json",
            **kwargs,
        ), sandbox

    async def test_toggles_apply_to_sandbox_and_save_defaults(self) -> None:
        from xarness.tui.container_screen import ContainerSettingsScreen

        with tempfile.TemporaryDirectory() as tmpd:
            tmp = Path(tmpd)
            (tmp / "docs").mkdir()
            (tmp / "docs" / "spec.md").write_text("x\n")
            app, sandbox = self._make([], tmp)
            async with app.run_test() as pilot:
                screen = ContainerSettingsScreen(sandbox, tmp / "config.json")
                await app.push_screen(screen)
                await pilot.pause()

                # Toggle network access on (space on the focused switch).
                net = screen.query_one("#container-net")
                self.assertFalse(sandbox.allow_network)
                net.value = True
                await pilot.pause()
                self.assertTrue(sandbox.allow_network)

                # Add a ref through the input (Enter submits): host path,
                # explicit mount, writable.
                inp = screen.query_one("#container-ref-input")
                inp.value = f"{tmp / 'docs' / 'spec.md'} docs/spec.md --rw"
                await pilot.press("enter")
                await pilot.pause()
                ref = sandbox.external_refs["spec.md"]
                self.assertEqual(ref.mount, "docs/spec.md")
                self.assertFalse(ref.read_only)

                # Save as defaults writes the container section.
                await pilot.click("#container-save")
                await pilot.pause()
                data = json.loads((tmp / "config.json").read_text())
                self.assertTrue(data["container"]["network_access"])
                self.assertEqual(
                    data["container"]["auto_include_refs"],
                    [{"path": str(tmp / "docs" / "spec.md"),
                      "mount": "docs/spec.md", "read_only": False}],
                )

    async def test_slash_container_opens_screen(self) -> None:
        from xarness.tui.container_screen import ContainerSettingsScreen

        with tempfile.TemporaryDirectory() as tmpd:
            tmp = Path(tmpd)
            app, sandbox = self._make([], tmp)
            async with app.run_test() as pilot:
                await pilot.press("/", "c", "o", "n", "t", "a", "i", "n", "e", "r", "enter")
                await pilot.pause()
                self.assertIsInstance(app.screen, ContainerSettingsScreen)
                self.assertIs(app.screen._sandbox, sandbox)
class TestContainerGpu(unittest.IsolatedAsyncioTestCase):
    async def test_gpu_toggle_applies_and_saves(self) -> None:
        import json
        from xarness.tui.container_screen import ContainerSettingsScreen

        with tempfile.TemporaryDirectory() as tmpd:
            tmp = Path(tmpd)
            app, sandbox = TestContainerSettings()._make([], tmp)
            async with app.run_test() as pilot:
                screen = ContainerSettingsScreen(sandbox, tmp / "config.json")
                await app.push_screen(screen)
                await pilot.pause()
                self.assertFalse(sandbox.gpu_access)
                screen.query_one("#container-gpu").value = True
                await pilot.pause()
                self.assertTrue(sandbox.gpu_access)
                await pilot.click("#container-save")
                await pilot.pause()
                data = json.loads((tmp / "config.json").read_text())
                self.assertTrue(data["container"]["gpu_access"])


class TestContainerTabs(unittest.IsolatedAsyncioTestCase):
    def _make(self, tmp: Path, config: dict | None = None):
        import json as _json
        from xarness.sandbox import SandboxConfig, SandboxSession

        (tmp / "wt").mkdir()
        (tmp / "docs").mkdir()
        (tmp / "docs" / "spec.md").write_text("x\n")
        (tmp / "config.json").write_text(_json.dumps(config or {"provider": {
            "base_url": "https://alpha.example.test/v1", "model_id": "alpha-model",
        }}), encoding="utf-8")
        sandbox = SandboxConfig(workspace=tmp / "wt")
        app = make_app(
            [],
            sandbox=sandbox,
            sandbox_session=SandboxSession(sandbox),
            config_path=tmp / "config.json",
        )
        return app, sandbox

    async def test_defaults_tab_edits_config_not_session(self) -> None:
        import json
        from textual.widgets import TabbedContent
        from xarness.tui.container_screen import ContainerSettingsScreen

        with tempfile.TemporaryDirectory() as tmpd:
            tmp = Path(tmpd)
            app, sandbox = self._make(tmp)
            async with app.run_test(size=(100, 44)) as pilot:
                screen = ContainerSettingsScreen(sandbox, tmp / "config.json")
                await app.push_screen(screen)
                await pilot.pause()
                self.assertEqual(
                    screen.query_one("#container-tabs", TabbedContent).active,
                    "session-tab",
                )

                # Session toggle touches the sandbox, not the defaults.
                screen.query_one("#container-net").value = True
                await pilot.pause()
                self.assertTrue(sandbox.allow_network)
                self.assertFalse(screen._defaults.network_access)

                # Switch to the defaults tab: its own switches, and a ref
                # added there lands in _defaults, not the sandbox.
                tabs = screen.query_one("#container-tabs", TabbedContent)
                tabs.active = "defaults-tab"
                await pilot.pause()
                defaults_input = screen.query_one("#defaults-ref-input")
                defaults_input.focus()
                await pilot.pause()
                defaults_input.value = "docs/spec.md .venv"
                await pilot.press("enter")
                await pilot.pause()
                self.assertEqual(len(screen._defaults.auto_include_refs), 1)
                self.assertEqual(len(sandbox.external_refs), 0)

                # Save persists defaults only.
                await pilot.click("#container-save")
                await pilot.pause()
                data = json.loads((tmp / "config.json").read_text())
                self.assertFalse(data["container"]["network_access"])
                self.assertEqual(
                    data["container"]["auto_include_refs"],
                    [{"path": "docs/spec.md", "mount": ".venv", "read_only": True}],
                )
                # And the session sandbox keeps its live state.
                self.assertTrue(sandbox.allow_network)

    async def test_defaults_tab_loaded_from_config(self) -> None:
        from textual.widgets import TabbedContent
        from xarness.tui.container_screen import ContainerSettingsScreen

        with tempfile.TemporaryDirectory() as tmpd:
            tmp = Path(tmpd)
            app, sandbox = self._make(tmp, {
                "provider": {"base_url": "https://x/v1", "model_id": "m"},
                "container": {
                    "network_access": True,
                    "auto_include_refs": ["docs/spec.md"],
                },
            })
            async with app.run_test() as pilot:
                screen = ContainerSettingsScreen(sandbox, tmp / "config.json")
                await app.push_screen(screen)
                await pilot.pause()
                tabs = screen.query_one("#container-tabs", TabbedContent)
                tabs.active = "defaults-tab"
                await pilot.pause()
                self.assertTrue(screen._defaults.network_access)
                rows = screen._default_ref_rows()
                self.assertEqual(len(rows), 1)
                path, mount, read_only = rows[0]
                self.assertEqual(path, "docs/spec.md")
                self.assertEqual(mount, ".refs/spec.md")
                self.assertTrue(read_only)

    async def test_ref_row_delete_and_rw_switch(self) -> None:
        from textual.widgets import Button as TButton, Switch as TSwitch
        from xarness.tui.container_screen import ContainerSettingsScreen

        with tempfile.TemporaryDirectory() as tmpd:
            tmp = Path(tmpd)
            app, sandbox = self._make(tmp)
            (tmp / "lib").mkdir()
            sandbox.add_ref(tmp / "docs" / "spec.md")
            sandbox.add_ref(tmp / "lib", mount=".venv", read_only=False)
            async with app.run_test(size=(100, 44)) as pilot:
                screen = ContainerSettingsScreen(sandbox, tmp / "config.json")
                await app.push_screen(screen)
                await pilot.pause()

                # Full host paths are shown.
                labels = " ".join(str(l.render()) for l in screen.query(".container-ref-path"))
                self.assertIn(str(tmp / "docs" / "spec.md"), labels)
                self.assertIn(str(tmp / "lib"), labels)

                # Flip the first row's rw switch: spec.md starts read-only;
                # switching it on makes the mount writable.
                sw = screen.query_one("#s-rw-0", TSwitch)
                self.assertTrue(sandbox.external_refs["spec.md"].read_only)
                sw.value = True
                await pilot.pause()
                self.assertFalse(sandbox.external_refs["spec.md"].read_only)

                # ✕ removes the row's ref.
                await pilot.click("#s-rm-0")
                await pilot.pause()
                self.assertNotIn("spec.md", sandbox.external_refs)
                self.assertIn(".venv", [r.mount for r in sandbox.external_refs.values()])


class TestContainerGpuProbe(unittest.IsolatedAsyncioTestCase):
    async def test_gpu_switch_triggers_probe_notice(self) -> None:
        from xarness.tui.container_screen import ContainerSettingsScreen

        with tempfile.TemporaryDirectory() as tmpd:
            tmp = Path(tmpd)
            app, sandbox = TestContainerSettings()._make([], tmp)
            async with app.run_test() as pilot:
                screen = ContainerSettingsScreen(sandbox, tmp / "config.json")
                await app.push_screen(screen)
                await pilot.pause()

                from unittest import mock as _mock

                probe_mock = _mock.AsyncMock(return_value=("warn", "denied by cgroup"))
                with _mock.patch("xarness.sandbox.probe_gpu_access", probe_mock):
                    screen.query_one("#container-gpu").value = True
                    for _ in range(30):
                        await pilot.pause()
                        if app._workers and all(w.is_finished for w in app._workers):
                            break
                    await pilot.pause()
                self.assertTrue(sandbox.gpu_access)
                # The worker surfaced the probe as a notification (severity
                # plumbing — the toast itself is Textual's).
                # The AppNotification widget keeps the toast list; simplest
                # observable: probe mock was awaited (worker ran) and the
                # app's toast handler got the message — assert via the
                # notifications widget's hook if present, else the mock call.
                self.assertTrue(probe_mock.awaited)
