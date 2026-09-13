"""Headless TUI smoke tests using Textual's pilot."""

import asyncio
import unittest
from collections.abc import AsyncIterator, Sequence
from typing import Any

from xarness.config import CotStrength, ProviderProfile
from xarness.controller import ChatController
from xarness.events import (
    ContentDelta,
    ProcessingStarted,
    ReasoningDelta,
    ToolCallArgumentsDelta,
    ToolCallStarted,
    ToolCallStatus,
    TurnComplete,
    Usage,
)
from xarness.tui.app import AgentApp
from xarness.tui.widgets import (
    ChatInput,
    StatusBar,
    ThinkingBlock,
    ToolCallBlock,
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
        # Scripts for the 2nd+ stream() calls (e.g. the round after tools).
        self.next_scripts: list[list[Any]] = []
        self._calls = 0
        self.delay = delay
        self.received_tools: list[dict[str, Any]] | None = None

    async def stream(
        self,
        wire_messages: Sequence[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[Any]:
        self.received_tools = tools
        self._calls += 1
        yield ProcessingStarted()
        script = self.script if self._calls == 1 or not self.next_scripts else self.next_scripts.pop(0)
        for event in script:
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
            self.assertEqual(app.controller.conversation.messages[1].content, "The answer is 42")

            # Reasoning block collapsed with a duration label.
            thinking = app.query_one(ThinkingBlock)
            self.assertTrue(thinking.done)
            self.assertFalse(thinking.has_class("expanded"))
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
            self.assertEqual(app.controller.conversation.messages[1].content, "plain answer")

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
            self.assertFalse(thinking.has_class("expanded"))
            self.assertFalse(thinking.query_one(".thinking-row").display)

            await pilot.press("ctrl+t")
            self.assertTrue(thinking.has_class("expanded"))
            self.assertTrue(thinking.query_one(".thinking-row").display)

            await pilot.press("ctrl+t")
            self.assertFalse(thinking.has_class("expanded"))

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

    async def test_tool_call_round_trip(self) -> None:
        """A tool-call round executes test_tool and streams a follow-up round."""
        app, client = make_app(
            [
                ToolCallStarted("call_1", "test_tool"),
                ToolCallArgumentsDelta("call_1", "{}"),
                TurnComplete(has_tool_calls=True),
            ]
        )
        client.next_scripts = [
            [ContentDelta("tool says hi"), TurnComplete(usage=Usage(5, 2))]
        ]
        async with app.run_test() as pilot:
            await pilot.press("r", "u", "n", "enter")
            await wait_until_idle(app)

            # One tool call block, settled green with the tool's output.
            block = app.query_one(ToolCallBlock)
            self.assertEqual(block.tool_name, "test_tool")
            self.assertEqual(block.status, ToolCallStatus.CALL_SUCCEEDED)
            self.assertEqual(block.accumulated_arguments, "{}")
            self.assertEqual(block._output_text, "success!")

            # Conversation: user, assistant tool-call turn, tool result, final answer.
            roles = [m.role for m in app.controller.conversation.messages]
            self.assertEqual(roles, ["user", "assistant", "tool", "assistant"])
            tool_msg = app.controller.conversation.messages[2]
            self.assertEqual(tool_msg.tool_call_id, "call_1")
            self.assertEqual(tool_msg.content, "success!")
            self.assertEqual(app.controller.conversation.messages[3].content, "tool says hi")


if __name__ == "__main__":
    unittest.main()
