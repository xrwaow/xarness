"""Headless TUI smoke tests using Textual's pilot."""

import asyncio
import unittest
from collections.abc import AsyncIterator, Sequence
from typing import Any

from agentcli.config import CotStrength, ProviderProfile
from agentcli.controller import ChatController
from agentcli.events import (
    ContentDelta,
    ProcessingStarted,
    ReasoningDelta,
    TurnComplete,
    Usage,
)
from agentcli.tui.app import AgentApp
from agentcli.tui.widgets import (
    AssistantMessage,
    ChatInput,
    StatusBar,
    ThinkingBlock,
    UserMessage,
)

PROFILE = ProviderProfile(
    base_url="https://api.example.test/v1",
    model_id="test-model",
    shown_name="Test Model",
    max_context=1000,
    cot_strength=CotStrength.MEDIUM,
)


class FakeClient:
    def __init__(self, script: list[Any], delay: float = 0.01) -> None:
        self.script = script
        self.delay = delay

    async def stream(self, wire_messages: Sequence[dict[str, Any]]) -> AsyncIterator[Any]:
        yield ProcessingStarted()
        for event in self.script:
            await asyncio.sleep(self.delay)
            yield event


def make_app(script: list[Any], delay: float = 0.01) -> tuple[AgentApp, FakeClient]:
    client = FakeClient(script, delay)
    controller = ChatController(PROFILE, "test-key", client=client)
    return AgentApp(PROFILE, "test-key", controller=controller), client


async def wait_until_idle(app: AgentApp, timeout: float = 5.0) -> None:
    for _ in range(int(timeout / 0.05)):
        await asyncio.sleep(0.05)
        if not app._turn_busy:
            return
    raise AssertionError("turn did not finish in time")


class TestChatLoop(unittest.IsolatedAsyncioTestCase):
    async def test_full_turn_with_reasoning(self) -> None:
        app, _client = make_app(
            [
                ReasoningDelta("pondering "),
                ReasoningDelta("deeply"),
                ContentDelta("The answer"),
                ContentDelta(" is 42"),
                TurnComplete(usage=Usage(input_tokens=100, output_tokens=20)),
            ]
        )
        async with app.run_test() as pilot:
            await pilot.press("h", "i")
            await pilot.press("enter")
            await wait_until_idle(app)

            # User message rendered in the scrollback.
            user_messages = app.query(UserMessage)
            self.assertEqual(len(user_messages), 1)
            self.assertIn("hi", user_messages.first().text)

            # Answer streamed and finalized.
            assistant = app.query_one(AssistantMessage)
            self.assertEqual(assistant._text, "The answer is 42")

            # Reasoning block collapsed with a duration label.
            thinking = app.query_one(ThinkingBlock)
            self.assertTrue(thinking.done)
            self.assertTrue(thinking.has_class("collapsed"))
            self.assertRegex(thinking.summary_text, r"Thought for \d+\.\d+s")

            # Reasoning stored in conversation history regardless of display.
            assistant_msg = app.controller.conversation.messages[1]
            self.assertEqual(assistant_msg.reasoning, "pondering deeply")

            # Status bar reflects usage and headroom.
            status = app.query_one(StatusBar).text
            self.assertIn("Test Model", status)
            self.assertIn("100", status)
            self.assertIn("left", status)

            # Input was cleared after submit.
            self.assertEqual(app.query_one(ChatInput).text, "")

    async def test_turn_without_reasoning_skips_thinking_state(self) -> None:
        app, _client = make_app(
            [ContentDelta("plain answer"), TurnComplete(usage=Usage(input_tokens=10, output_tokens=2))]
        )
        async with app.run_test() as pilot:
            await pilot.press("enter")  # empty input: no-op
            await pilot.press("y", "o", "enter")
            await wait_until_idle(app)

            self.assertEqual(len(app.query(ThinkingBlock)), 0)
            self.assertEqual(app.query_one(AssistantMessage)._text, "plain answer")

    async def test_ctrl_t_toggles_thoughts(self) -> None:
        app, _client = make_app(
            [
                ReasoningDelta("secret thoughts"),
                ContentDelta("answer"),
                TurnComplete(usage=Usage(input_tokens=10, output_tokens=2)),
            ]
        )
        async with app.run_test() as pilot:
            await pilot.press("q", "enter")
            await wait_until_idle(app)

            thinking = app.query_one(ThinkingBlock)
            self.assertTrue(thinking.has_class("collapsed"))
            self.assertFalse(thinking.query_one(".thinking-text").display)

            await pilot.press("ctrl+t")
            self.assertTrue(thinking.has_class("expanded"))
            self.assertTrue(thinking.query_one(".thinking-text").display)

            await pilot.press("ctrl+t")
            self.assertTrue(thinking.has_class("collapsed"))

    async def test_queued_message_runs_after_current_turn(self) -> None:
        app, client = make_app([ContentDelta("first reply"), TurnComplete(usage=Usage(10, 2))], delay=0.3)
        async with app.run_test() as pilot:
            await pilot.press("a", "enter")
            # Submit again while the first turn is still running.
            await pilot.press("b", "enter")
            self.assertTrue(app._turn_busy)

            client.script = [
                ContentDelta("second reply"),
                TurnComplete(usage=Usage(30, 5)),
            ]
            await wait_until_idle(app)

            self.assertEqual(len(app.query(UserMessage)), 2)
            self.assertEqual(app.controller.conversation.messages[-1].content, "second reply")
            self.assertEqual(app.total_in, 40)
            self.assertEqual(app.total_out, 7)


if __name__ == "__main__":
    unittest.main()
