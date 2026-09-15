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
    ToolCallArgumentsDone,
    ToolCallStarted,
    ToolCallStatus,
    TurnComplete,
    Usage,
)
from xarness.tools import Tool, ToolRegistry, ToolResult, build_registry
from xarness.tui.app import AgentApp
from xarness.tui.widgets import (
    AskBar,
    ChatInput,
    ErrorLine,
    NoticeLine,
    StatusBar,
    ThinkingBlock,
    ToolCallBlock,
    ToolWritingIndicator,
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


def make_registry(
    ask_callback=None, compact_callback=None
) -> ToolRegistry:
    """Registry for TUI tests: web_search/ask/compact plus a deterministic
    stub tool ("noop") the scripted tool-call rounds can invoke."""
    registry = build_registry(
        None, None, ask_callback=ask_callback, compact_callback=compact_callback
    )

    async def _noop(args: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, output="ok")

    registry.register(Tool(name="noop", description="", parameters_schema={}, handler=_noop))
    return registry


def make_app(script: list[Any], delay: float = 0.01) -> tuple[AgentApp, FakeClient]:
    client = FakeClient(script, delay)
    controller = ChatController(PROFILE, "test-key", client=client)
    app = AgentApp(PROFILE, "test-key", controller=controller)
    # Swap in a registry with a deterministic stub tool: the default one has
    # no sandbox tools here, and web_search would hit the network.
    registry = make_registry(
        ask_callback=app._ask_user, compact_callback=app._compact_conversation
    )
    app.tool_registry = registry
    app.controller.tools = registry
    return app, client


async def wait_until_idle(app: AgentApp, timeout: float = 5.0) -> None:
    for _ in range(int(timeout / 0.05)):
        await asyncio.sleep(0.05)
        if not app._turn_busy:
            return
    raise AssertionError("turn did not finish in time")


async def wait_for(predicate, timeout: float = 5.0) -> None:
    for _ in range(int(timeout / 0.05)):
        await asyncio.sleep(0.05)
        if predicate():
            return
    raise AssertionError("condition not met in time")


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

            # Answer streamed and finalized. (messages[0] is the system prompt.)
            self.assertEqual(app.controller.conversation.messages[2].content, "The answer is 42")

            # Reasoning block collapsed with a duration label.
            thinking = app.query_one(ThinkingBlock)
            self.assertTrue(thinking.done)
            self.assertFalse(thinking.has_class("expanded"))
            self.assertRegex(thinking.summary_text, r"Thought for \d+\.\d+s")

            # Reasoning stored in conversation history regardless of display.
            assistant_msg = app.controller.conversation.messages[2]
            self.assertEqual(assistant_msg.reasoning, "pondering deeply")
            self.assertIsNotNone(assistant_msg.reasoning_seconds)

            # Turn ends with a muted "Worked for" summary line.
            notice = app.query_one(NoticeLine)
            self.assertRegex(str(notice.content), r"Worked for \d+\.\d?s")

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
            self.assertEqual(app.controller.conversation.messages[2].content, "plain answer")

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
        """A message submitted mid-turn is queued and runs after it."""
        class GatedClient:
            """Holds the first turn open until the test releases it, so the
            second submit reliably lands while the turn is busy."""

            def __init__(self) -> None:
                self.gate = asyncio.Event()
                self._calls = 0

            async def stream(self, wire_messages, tools=None):
                self._calls += 1
                yield ProcessingStarted()
                if self._calls == 1:
                    yield ContentDelta("first reply")
                    await self.gate.wait()
                    yield TurnComplete(usage=Usage(10, 2))
                else:
                    yield ContentDelta("second reply")
                    yield TurnComplete(usage=Usage(30, 5))

        client = GatedClient()
        controller = ChatController(PROFILE, "test-key", client=client)
        app = AgentApp(PROFILE, "test-key", controller=controller)
        async with app.run_test() as pilot:
            await pilot.press("a", "enter")
            for _ in range(100):
                await asyncio.sleep(0.05)
                if app._turn_busy:
                    break
            self.assertTrue(app._turn_busy)

            # Submit again while the first turn is still running.
            await pilot.press("b", "enter")
            self.assertEqual(app._queued, ["b"])

            client.gate.set()
            await wait_until_idle(app)

            self.assertEqual(len(app.query(UserMessage)), 2)
            self.assertEqual(app.controller.conversation.messages[-1].content, "second reply")
            self.assertEqual(app.total_in, 40)
            self.assertEqual(app.total_out, 7)

    async def test_tool_call_round_trip(self) -> None:
        """A tool-call round executes the stub tool and streams a follow-up round."""
        app, client = make_app(
            [
                ToolCallStarted("call_1", "noop"),
                ToolCallArgumentsDelta("call_1", "{}"),
                ToolCallArgumentsDone("call_1", "noop", "{}"),
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
            self.assertEqual(block.tool_name, "noop")
            self.assertEqual(block.status, ToolCallStatus.CALL_SUCCEEDED)
            self.assertEqual(block.accumulated_arguments, "{}")
            self.assertEqual(block._output_text, "ok")

            # Conversation: system, user, assistant tool-call turn, tool result,
            # final answer.
            roles = [m.role for m in app.controller.conversation.messages]
            self.assertEqual(
                roles, ["system", "user", "assistant", "tool", "assistant"]
            )
            tool_msg = app.controller.conversation.messages[3]
            self.assertEqual(tool_msg.tool_call_id, "call_1")
            self.assertEqual(tool_msg.content, "ok")
            self.assertEqual(app.controller.conversation.messages[4].content, "tool says hi")

    async def test_parallel_tool_calls_show_single_writing_indicator(self) -> None:
        """While args stream: one 'Writing tools' shimmer, no per-call blocks.
        Blocks appear (and the indicator leaves) once args are done."""
        app, client = make_app(
            [
                ToolCallStarted("call_1", "noop"),
                ToolCallStarted("call_2", "noop"),
                ToolCallArgumentsDelta("call_1", "{}"),
                ToolCallArgumentsDone("call_1", "noop", "{}"),
                ToolCallArgumentsDone("call_2", "noop", '"x"'),
                TurnComplete(has_tool_calls=True),
            ],
            delay=0.3,
        )
        client.next_scripts = [
            [ContentDelta("done"), TurnComplete(usage=Usage(5, 2))]
        ]
        async with app.run_test() as pilot:
            # Submit directly: pilot.press drains the whole turn, which would
            # hide the mid-round state this test inspects.
            app._submit("go")

            # Wait for call_1's args to finish: its block mounts while call_2
            # is still streaming — one block plus the shared indicator.
            for _ in range(100):
                await asyncio.sleep(0.05)
                if len(app.query(ToolCallBlock)) == 1:
                    break
            self.assertEqual(len(app.query(ToolCallBlock)), 1)
            self.assertEqual(len(app.query(ToolWritingIndicator)), 1)

            await wait_until_idle(app)
            # Indicator gone; both blocks settled.
            self.assertEqual(len(app.query(ToolWritingIndicator)), 0)
            self.assertEqual(len(app.query(ToolCallBlock)), 2)

    async def test_ask_tool_returns_user_answers(self) -> None:
        """The ask tool shows questions one at a time in the AskBar; answers
        typed into the normal chat input become the tool result."""
        app, client = make_app(
            [
                ToolCallStarted("call_1", "ask"),
                ToolCallArgumentsDone(
                    "call_1", "ask", '{"questions": ["Which db?", "Confirm?"]}'
                ),
                TurnComplete(has_tool_calls=True),
            ],
            delay=0.05,
        )
        client.next_scripts = [
            [ContentDelta("got it"), TurnComplete(usage=Usage(5, 2))]
        ]
        async with app.run_test() as pilot:
            await pilot.press("a", "s", "k", "enter")

            ask_bar = app.query_one("#ask-bar", AskBar)
            await wait_for(lambda: ask_bar.has_class("visible"))
            # First question only, shown above the input.
            self.assertIn("Question 1/2", str(ask_bar.content))
            self.assertIn("Which db?", str(ask_bar.content))
            self.assertNotIn("Confirm?", str(ask_bar.content))
            self.assertTrue(app.query_one(ChatInput).has_focus)

            # Answer it; the second question appears in the same bar.
            await pilot.press("p", "g", "enter")
            await wait_for(lambda: "Confirm?" in str(ask_bar.content))
            self.assertIn("Question 2/2", str(ask_bar.content))

            await pilot.press("y", "e", "s", "enter")
            await wait_until_idle(app)

            tool_msg = app.controller.conversation.messages[3]
            self.assertIn("Q: Which db?", tool_msg.content)
            self.assertIn("A: pg", tool_msg.content)
            self.assertIn("A: yes", tool_msg.content)
            self.assertEqual(app.controller.conversation.messages[4].content, "got it")

            # The bar hides and the placeholder is restored once done.
            await wait_for(lambda: not ask_bar.has_class("visible"))
            self.assertNotIn("answer", str(app.query_one(ChatInput).placeholder).lower())

    async def test_ask_tool_escape_skips_questions(self) -> None:
        """Esc while a question is pending skips it; the turn keeps running."""
        app, client = make_app(
            [
                ToolCallStarted("call_1", "ask"),
                ToolCallArgumentsDone(
                    "call_1", "ask", '{"questions": ["Which db?"]}'
                ),
                TurnComplete(has_tool_calls=True),
            ],
            delay=0.05,
        )
        client.next_scripts = [
            [ContentDelta("moving on"), TurnComplete(usage=Usage(5, 2))]
        ]
        async with app.run_test() as pilot:
            await pilot.press("a", "s", "k", "enter")
            ask_bar = app.query_one("#ask-bar", AskBar)
            await wait_for(lambda: ask_bar.has_class("visible"))

            await pilot.press("escape")
            await wait_until_idle(app)

            tool_msg = app.controller.conversation.messages[3]
            self.assertIn("skipped", tool_msg.content)
            self.assertEqual(app.controller.conversation.messages[4].content, "moving on")
            self.assertFalse(ask_bar.has_class("visible"))

    async def test_slash_theme_opens_picker_and_switches(self) -> None:
        from xarness import theme
        from xarness.tui.picker_screen import PickerScreen

        app, _client = make_app([ContentDelta("x"), TurnComplete(usage=Usage(1, 1))])
        async with app.run_test() as pilot:
            theme.set_theme("ayu-darker")
            # Bare /theme opens the picker (same interaction as /model).
            await pilot.press("/", "t", "h", "e", "m", "e", "enter")
            await pilot.pause()
            self.assertIsInstance(app.screen, PickerScreen)

            # The list is focused: up/down moves the cursor, enter selects.
            list_view = app.screen.query_one("#picker-list")
            self.assertTrue(list_view.has_focus)
            await pilot.press("down")
            self.assertEqual(list_view.index, 1)
            await pilot.press("enter")
            await pilot.pause()
            self.assertEqual(theme.CURRENT_THEME, "one-light")
            self.assertEqual(app.get_css_variables()["c-bg"], theme.PALETTE["bg"])

            # Unknown theme name reports the available options.
            app._handle_theme_command("nope")
            await pilot.pause()
            errors = app.query(ErrorLine)
            self.assertTrue(errors)
            self.assertIn("unknown theme", str(errors.last().content))


if __name__ == "__main__":
    unittest.main()
