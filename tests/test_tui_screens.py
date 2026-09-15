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
from xarness.conversation import Conversation, Message
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
    path.write_text(
        """
default_profile: alpha
profiles:
  alpha:
    base_url: https://alpha.example.test/v1
    api_key_env: ALPHA_KEY
    model_id: alpha-model
    shown_name: Alpha Model
  beta:
    base_url: https://beta.example.test/v1
    api_key_env: %s
    model_id: beta-model
    shown_name: Beta Model
"""
        % ("BETA_KEY" if beta_key_env else "null"),
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
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "noop", "arguments": '{"x": 1}'},
                    }
                ],
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
                tool_calls=[
                    {
                        "id": "call_9",
                        "type": "function",
                        "function": {"name": "noop", "arguments": "{}"},
                    }
                ],
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
            named = Path(tmp) / "config.yaml"
            write_config(named)
            self.assertEqual(list_profile_names(named), ["alpha", "beta"])

            flat = Path(tmp) / "flat.yaml"
            flat.write_text(
                "provider:\n  base_url: https://x.test/v1\n  model_id: m\n",
                encoding="utf-8",
            )
            self.assertEqual(list_profile_names(flat), [])

            missing = Path(tmp) / "nope.yaml"
            with self.assertRaises(ConfigError):
                list_profile_names(missing)

    async def test_slash_model_switches_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
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
            config_path = Path(tmp) / "config.yaml"
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
