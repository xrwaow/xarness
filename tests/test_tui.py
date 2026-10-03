"""Headless TUI smoke tests using Textual's pilot."""

import asyncio
import unittest
from collections.abc import AsyncIterator, Sequence
from typing import Any

from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Static

from xarness.config import CotStrength, ProviderProfile
from xarness.conversation import Message
from xarness.controller import ChatController
from xarness.events import (
    ContentDelta,
    ProcessingStarted,
    ReasoningDelta,
    StreamError,
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
    AssistantMessage,
    ChatInput,
    CompactionSummary,
    ErrorLine,
    MessageLine,
    NoticeLine,
    PendingIndicator,
    ShimmerText,
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


def make_registry(ask_callback=None) -> ToolRegistry:
    """Registry for TUI tests: ask_user plus a deterministic stub tool
    ("noop") the scripted tool-call rounds can invoke."""
    registry = build_registry(None, None, ask_callback=ask_callback)

    async def _noop(args: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, output="ok")

    registry.register(Tool(name="noop", description="", parameters_schema={}, handler=_noop))
    return registry


def make_app(script: list[Any], delay: float = 0.01) -> tuple[AgentApp, FakeClient]:
    client = FakeClient(script, delay)
    controller = ChatController(PROFILE, "test-key", client=client)
    app = AgentApp(PROFILE, "test-key", controller=controller)
    # Swap in a registry with a deterministic stub tool: the default one has
    # no sandbox tools here.
    registry = make_registry(ask_callback=app._ask_user)
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

    async def test_queued_message_steers_next_round(self) -> None:
        """A message typed mid-turn is injected at the round boundary —
        after the tool answers, before the next LLM call — so the very
        next round already sees it, and its (queued) marker clears."""

        class GatedClient:
            def __init__(self) -> None:
                self.gate = asyncio.Event()
                self._calls = 0
                self.round_user_texts: list[list[str]] = []

            async def stream(self, wire_messages, tools=None):
                self._calls += 1
                self.round_user_texts.append(
                    [m["content"] for m in wire_messages if m.get("role") == "user"]
                )
                yield ProcessingStarted()
                if self._calls == 1:
                    yield ToolCallStarted("call_1", "noop")
                    yield ToolCallArgumentsDelta("call_1", "{}")
                    yield ToolCallArgumentsDone("call_1", "noop", "{}")
                    yield TurnComplete(has_tool_calls=True)
                    # Hold the turn open so the steer message lands mid-round.
                    await self.gate.wait()
                else:
                    yield ContentDelta("steered reply")
                    yield TurnComplete(usage=Usage(20, 4))

        client = GatedClient()
        controller = ChatController(PROFILE, "test-key", client=client)
        app = AgentApp(PROFILE, "test-key", controller=controller)
        async with app.run_test() as pilot:
            await pilot.press("a", "enter")
            await wait_for(lambda: app._turn_busy)

            # Queue while the tool round is in flight.
            await pilot.press("b", "enter")
            self.assertEqual(app._queued, ["b"])
            # The send hint lives in the steer bar above the input now.
            hints = app.query_one("#steer-queue-bar").query(".steer-hint")
            self.assertEqual(len(hints), 1)

            client.gate.set()
            await wait_until_idle(app)

            # The queued message was delivered into the same turn, right
            # after the tool result, before the follow-up round.
            roles = [m.role for m in app.controller.conversation.messages]
            self.assertEqual(
                roles, ["system", "user", "assistant", "tool", "user", "assistant"]
            )
            self.assertEqual(app.controller.conversation.messages[4].content, "b")
            self.assertEqual(client.round_user_texts[1], ["a", "b"])
            self.assertEqual(app._queued, [])
            self.assertEqual(app.query(UserMessage)[1]._suffix, "")

    async def test_enter_on_empty_input_sends_queued_message_now(self) -> None:
        """Enter with an empty input while a message is queued interrupts the
        running turn and sends the queued message immediately."""

        class GatedClient:
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
                    yield TurnComplete(usage=Usage(20, 4))

        client = GatedClient()
        controller = ChatController(PROFILE, "test-key", client=client)
        app = AgentApp(PROFILE, "test-key", controller=controller)
        async with app.run_test() as pilot:
            await pilot.press("a", "enter")
            await wait_for(lambda: app._turn_busy)

            await pilot.press("b", "enter")
            self.assertEqual(app._queued, ["b"])

            # Empty input + Enter: send now, don't wait for the turn to end.
            await pilot.press("enter")
            await wait_until_idle(app)

            self.assertEqual(app._queued, [])
            self.assertEqual(len(app.query(UserMessage)), 2)
            self.assertEqual(app.controller.conversation.messages[-1].content, "second reply")
            notices = [str(n.content) for n in app.query(ErrorLine)]
            self.assertTrue(any("interrupted" in n for n in notices))

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

    async def test_tool_call_header_detail(self) -> None:
        """Settled headers carry a dimmed argument summary per tool."""
        from xarness.tui.widgets import _tool_header_detail

        def detail(name, args, out=""):
            t = _tool_header_detail(name, args, out)
            return None if t is None else t.plain

        self.assertEqual(detail("read_file", '{"path": "src/app.py"}'), " src/app.py")
        self.assertEqual(
            detail("read_file", '{"path": "index.html", "start_line": 270, "end_line": 300}'),
            " index.html [start_line=270, end_line=300]",
        )
        self.assertEqual(detail("write_file", '{"path": "new.py"}'), " new.py")
        self.assertEqual(detail("edit_file", '{"path": "f.py", "edits": []}'), " f.py")
        self.assertEqual(
            detail("run_bash", '{"command": "git status\\nls"}'), " git status"
        )
        self.assertEqual(detail("web_search", '{"query": "tui toolkit"}'), " tui toolkit")
        # ask_user renders as a plain "ask_user" — no argument summary.
        self.assertIsNone(detail("ask_user", '{"questions": ["Which one?", "Why?"]}'))
        # Malformed / missing args fall back to the plain header.
        self.assertIsNone(detail("read_file", "not json"))
        self.assertIsNone(detail("unknown_tool", "{}"))

    async def test_tool_output_containing_a_diff_is_rendered_as_one(self) -> None:
        """An edit's diff shows up in the expanded tool block — just the diff,
        with the theme's diff styling; the raw old/new arguments and the
        'applied' prose are not shown."""
        app, _client = make_app([ContentDelta("x"), TurnComplete(usage=Usage(1, 1))])
        async with app.run_test() as pilot:
            block = ToolCallBlock("call_1", "edit_file")
            await app.query_one("#chat-log", VerticalScroll).mount(block)
            block.append_arguments('{"path": "f.py"}')
            block.set_result(
                ToolCallStatus.CALL_SUCCEEDED,
                output=(
                    "applied edit to f.py\n"
                    "diff --git a/f.py b/f.py\n"
                    "--- a/f.py\n"
                    "+++ b/f.py\n"
                    "@@ -1,2 +1,2 @@\n"
                    " def a():\n"
                    "-    old_line()\n"
                    "+    new_line()\n"
                ),
            )
            await pilot.pause()

            body = block.query_one(".toolcall-body", Static).content
            self.assertIsInstance(body, Text)
            self.assertNotIn('{"path": "f.py"}', body.plain)
            self.assertNotIn("applied edit to f.py", body.plain)
            # The diff is rendered (hunk + changed lines, no per-file header
            # — the path is already in the block's summary row), not dumped
            # as raw `diff --git`/`---`/`+++` text.
            self.assertNotIn("── f.py", body.plain)
            self.assertIn("@@ -1,2 +1,2 @@", body.plain)
            self.assertIn("old_line()", body.plain)
            self.assertIn("new_line()", body.plain)
            self.assertNotIn("diff --git", body.plain)
            # Not plain text: the +/- lines carry the theme's diff styling.
            self.assertTrue(body.spans)

    async def test_read_file_highlight_matches_message_fences(self) -> None:
        """The read_file body uses the exact same highlight function as the
        assistant message's markdown fences — including the ANSI-mode branch,
        where the palette theme would quantize to unreadable 16-color output."""
        from unittest.mock import PropertyMock, patch

        from textual.app import active_app

        from xarness.tui.widgets import PaletteFence, highlight_code

        code = 'def foo():\n    return "bar"\n'
        app, _client = make_app([ContentDelta("x"), TurnComplete(usage=Usage(1, 1))])
        async with app.run_test():
            # Same output as the fence path, non-ANSI…
            self.assertEqual(
                highlight_code(code, path="f.py"),
                PaletteFence.highlight(code, language="", path="f.py"),
            )
            # …and in ANSI mode the fence's ANSI theme is used, not the
            # palette-driven CodeHighlightTheme.
            token = active_app.set(app)
            try:
                with patch.object(
                    type(app), "native_ansi_color", new_callable=PropertyMock, return_value=True
                ):
                    self.assertEqual(
                        highlight_code(code, path="f.py"),
                        PaletteFence.highlight(
                            code, language="", ansi=True, dark=True, path="f.py"
                        ),
                    )
            finally:
                active_app.reset(token)

    async def test_tool_code_blocks_render_readable_on_dark_background(self) -> None:
        """Truncated read_file (with a `[showing lines …]` note) and write_file
        bodies must be a single Content, not a rich Group. A Group renders
        through Rich without the widget's background, so the Content's
        `$text` (auto) styles resolve to black and unstyled tokens
        (punctuation, operators) come out black-on-dark — unlike the identical
        LLM-message fences."""
        from textual.content import Content

        from xarness import theme

        previous = theme.CURRENT_THEME
        theme.set_theme("ayu-darker")
        self.addCleanup(theme.set_theme, previous)

        app, _client = make_app([ContentDelta("x"), TurnComplete(usage=Usage(1, 1))])
        async with app.run_test(size=(120, 60)) as pilot:
            log = app.query_one("#chat-log", VerticalScroll)

            read = ToolCallBlock("c1", "read_file")
            await log.mount(read)
            read.append_arguments('{"path": "f.py", "start_line": 1, "end_line": 2}')
            read.set_result(
                ToolCallStatus.CALL_SUCCEEDED,
                output=(
                    "def foo(a, b):\n    return a + b\n"
                    "\n[showing lines 1-2 of 900; pass start_line/end_line for more]"
                ),
            )
            read.toggle()

            write = ToolCallBlock("c2", "write_file")
            await log.mount(write)
            write.append_arguments('{"path": "g.py", "content": "def foo():\\n    return 1\\n"}')
            write.set_result(ToolCallStatus.CALL_SUCCEEDED, output="applied write to g.py")
            write.toggle()
            await pilot.pause()
            await pilot.pause()

            for block in (read, write):
                body = block.query_one(".toolcall-body", Static).content
                self.assertIsInstance(body, Content)

            # Every code/note row renders light text on the dark background
            # (the left border marker is a dim border color by design).
            for strip in app.screen._compositor.render_strips():
                row = "".join(segment.text for segment in strip)
                if not any(marker in row for marker in ("def foo(a, b)", "def foo():", "applied write")):
                    continue
                for segment in strip:
                    if not segment.text.strip() or segment.text.strip() == "│":
                        continue
                    style = segment.style
                    if style is None or style.color is None or style.bgcolor is None:
                        continue
                    self.assertGreater(
                        sum(style.color.get_truecolor()),
                        200,
                        f"dark tool text {segment.text!r} on {style.bgcolor}",
                    )

    async def test_parallel_tool_calls_show_single_writing_indicator(self) -> None:
        """While args stream: one 'Writing <tool>' shimmer, no per-call blocks.
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
            # The label names the call still being written (call_2).
            self.assertEqual(
                " ".join(app.query(ToolWritingIndicator)[0]._label()).rstrip(), "noop"
            )

            await wait_until_idle(app)
            # Indicator gone; both blocks settled.
            self.assertEqual(len(app.query(ToolWritingIndicator)), 0)
            self.assertEqual(len(app.query(ToolCallBlock)), 2)

    async def test_write_file_count_counts_partial_content(self) -> None:
        """write_file's +count totals complete JSON exactly and partial JSON
        (mid-stream) from the escaped newlines seen so far."""
        from xarness.tui.widgets import _write_file_loc

        self.assertEqual(_write_file_loc('{"path": "g.py", "content": "a\\nb\\nc"}'), 3)
        self.assertEqual(_write_file_loc('{"path": "g.py", "content": "a\\nb"}'), 2)
        self.assertEqual(_write_file_loc('{"path": "g.py", "content": "a\\nb\\n'), 2)
        self.assertEqual(_write_file_loc('{"path": "g.py", "content": "'), 0)
        self.assertEqual(_write_file_loc('{"path": "g.py"}'), None)
        # An escaped quote inside content doesn't end the string early.
        self.assertEqual(_write_file_loc('{"content": "a\\"b\\nc"}'), 2)

    async def test_write_file_streams_count_in_writing_indicator(self) -> None:
        """While write_file's content streams, the shared indicator grows a
        live '+count'; the label follows the next tool call after it."""
        app, client = make_app(
            [
                ToolCallStarted("call_1", "write_file"),
                ToolCallArgumentsDelta("call_1", '{"path": "g.py", "content": "a\\nb\\n'),
                ToolCallArgumentsDelta("call_1", 'c\\nd"}'),
                ToolCallArgumentsDone(
                    "call_1", "write_file", '{"path": "g.py", "content": "a\\nb\\nc\\nd"}'
                ),
                TurnComplete(has_tool_calls=True),
            ],
            delay=0.3,
        )
        client.next_scripts = [
            [ContentDelta("done"), TurnComplete(usage=Usage(5, 2))]
        ]
        async with app.run_test() as pilot:
            app._submit("go")

            # Wait for the first content fragment: "a\nb\n" = 2 lines so far.
            for _ in range(100):
                await asyncio.sleep(0.05)
                indicators = app.query(ToolWritingIndicator)
                if indicators and "+2" in " ".join(indicators[0]._label()):
                    break
            self.assertEqual(
                " ".join(app.query(ToolWritingIndicator)[0]._label()),
                "write_file +2 g.py",
            )

            # The second fragment completes the JSON: "a\nb\nc\nd" = 4 lines.
            for _ in range(100):
                await asyncio.sleep(0.05)
                indicators = app.query(ToolWritingIndicator)
                if indicators and "+4" in " ".join(indicators[0]._label()):
                    break
            self.assertEqual(
                " ".join(app.query(ToolWritingIndicator)[0]._label()),
                "write_file +4 g.py",
            )

            await wait_until_idle(app)
            self.assertEqual(len(app.query(ToolWritingIndicator)), 0)

    async def test_settled_summary_drops_ran_and_colors_edit_counts(self) -> None:
        """Settled blocks read 'read_file index.html [start_line=…]' with no
        'Ran' verb, and edit_file's +N -M counts get the diff colors."""
        from xarness import theme
        from xarness.tui.widgets import _header_detail_text

        app, _client = make_app([ContentDelta("x"), TurnComplete(usage=Usage(1, 1))])
        async with app.run_test() as pilot:
            log = app.query_one("#chat-log", VerticalScroll)
            block = ToolCallBlock("c1", "read_file")
            await log.mount(block)
            block.append_arguments('{"path": "index.html", "start_line": 270, "end_line": 300}')
            block.set_result(ToolCallStatus.CALL_SUCCEEDED, output="data", header="index.html")
            await pilot.pause()

            summary = block.query_one(".toolcall-summary-text", Static).content
            self.assertEqual(summary.plain, "read_file index.html [start_line=270, end_line=300]")

            detail = _header_detail_text("edit_file", "+3 -1 src/app.py")
            self.assertEqual(detail.plain, " +3 -1 src/app.py")
            spans = {(detail.plain[s.start:s.end], str(s.style)) for s in detail.spans}
            self.assertIn(("+3", theme.PALETTE["diff_add"]), spans)
            self.assertIn(("-1", theme.PALETTE["diff_del"]), spans)

    async def test_write_file_header_shows_line_count(self) -> None:
        """Settled write_file blocks read 'write_file +N path' with the count
        in the diff-add color; a legacy path-only header stays dimmed."""
        from xarness import theme
        from xarness.tui.widgets import _header_detail_text

        detail = _header_detail_text("write_file", "+4 src/g.py")
        self.assertEqual(detail.plain, " +4 src/g.py")
        spans = {(detail.plain[s.start:s.end], str(s.style)) for s in detail.spans}
        self.assertIn(("+4", theme.PALETTE["diff_add"]), spans)

        legacy = _header_detail_text("write_file", "src/g.py")
        self.assertEqual(legacy.plain, " src/g.py")

    async def test_auto_compact_uses_configured_threshold(self) -> None:
        """_maybe_auto_compact fires at the profile's auto_compact_threshold
        (from config), not a hardcoded 90%; it stays off unless enabled."""
        app, _client = make_app([ContentDelta("x"), TurnComplete(usage=Usage(1, 1))])
        async with app.run_test() as pilot:
            self.assertFalse(app.auto_compact)  # config default: off
            app.profile = app.profile.model_copy(
                update={"auto_compact": True, "auto_compact_threshold": 0.1}
            )
            app.auto_compact = True
            calls: list[str] = []

            async def fake_compact() -> str:
                calls.append("compact")
                return "compacted: 900 → 90 tokens (freed 810)"

            app._run_compaction = fake_compact  # type: ignore[method-assign]

            # 100/1000 = 10% ≥ threshold: compaction fires. (last_in is a
            # read-only view of the conversation's per-message usage.)
            app.controller.conversation.add(
                Message(role="assistant", content="", usage=Usage(100, 0))
            )
            await app._maybe_auto_compact()
            self.assertEqual(calls, ["compact"])

            # 50/1000 = 5% < threshold: nothing happens.
            calls.clear()
            app.controller.conversation.messages[-1].usage = Usage(50, 0)
            await app._maybe_auto_compact()
            self.assertEqual(calls, [])

    async def test_tab_toggles_mode(self) -> None:
        """Tab in the input toggles plan/write mode, same as /mode."""
        from xarness.prompts import GENERAL_SYSTEM_PROMPT, system_prompt_for

        app, _client = make_app([ContentDelta("x"), TurnComplete(usage=Usage(1, 1))])
        async with app.run_test() as pilot:
            await pilot.press("tab")
            await pilot.pause()
            self.assertEqual(app.mode, "plan")
            self.assertIn("PLAN mode", app.controller.conversation.messages[0].content)

            await pilot.press("tab")
            await pilot.pause()
            self.assertEqual(app.mode, "write")
            self.assertIn("WRITE mode", app.controller.conversation.messages[0].content)

    async def test_ask_tool_returns_user_answers(self) -> None:
        """The ask tool shows questions one at a time in the AskBar; answers
        typed into the normal chat input become the tool result."""
        app, client = make_app(
            [
                ToolCallStarted("call_1", "ask_user"),
                ToolCallArgumentsDone(
                    "call_1", "ask_user", '{"questions": ["Which db?", "Confirm?"]}'
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
                ToolCallStarted("call_1", "ask_user"),
                ToolCallArgumentsDone(
                    "call_1", "ask_user", '{"questions": ["Which db?"]}'
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

    async def test_slash_mode_updates_system_message_and_tools(self) -> None:
        """A mid-session /mode rewrites the system message in place so it
        always matches the active mode's tool set."""
        from xarness.prompts import GENERAL_SYSTEM_PROMPT, system_prompt_for

        app, _client = make_app([ContentDelta("x"), TurnComplete(usage=Usage(1, 1))])
        async with app.run_test() as pilot:
            self.assertEqual(
                app.controller.conversation.messages[0].content,
                system_prompt_for(GENERAL_SYSTEM_PROMPT, "write"),
            )

            app._handle_slash_command("/mode")
            await pilot.pause()

            self.assertEqual(app.mode, "plan")
            self.assertEqual(
                app.controller.conversation.messages[0].content,
                system_prompt_for(GENERAL_SYSTEM_PROMPT, "plan"),
            )
            self.assertIn("PLAN mode", app.controller.conversation.messages[0].content)
            # The controller streams with the new mode's registry.
            self.assertIs(app.controller.tools, app.tool_registry)
            self.assertEqual(
                {t["function"]["name"] for t in app.controller.tools.schema()},
                {t["function"]["name"] for t in app.tool_registry.schema()},
            )

            # And back again.
            app._handle_slash_command("/mode")
            await pilot.pause()
            self.assertIn("WRITE mode", app.controller.conversation.messages[0].content)

    async def test_slash_undo_restores_input_and_history(self) -> None:
        app, _client = make_app([ContentDelta("answer"), TurnComplete(usage=Usage(10, 4))])
        async with app.run_test() as pilot:
            await pilot.press("h", "i", "enter")
            await wait_until_idle(app)
            self.assertEqual(app.total_in, 10)

            app._handle_slash_command("/undo")
            await wait_for(lambda: app.query_one(ChatInput).text == "hi")

            # Conversation back to just the system message; usage rolled back.
            self.assertEqual(
                [m.role for m in app.controller.conversation.messages], ["system"]
            )
            self.assertEqual((app.total_in, app.total_out, app.last_in), (0, 0, 0))
            # The user message is back in the input box (focused), not sent.
            chat_input = app.query_one(ChatInput)
            self.assertEqual(chat_input.text, "hi")
            self.assertTrue(chat_input.has_focus)
            self.assertFalse(app._turn_busy)

    async def test_slash_retry_resends_the_same_message(self) -> None:
        app, client = make_app([ContentDelta("bad answer"), TurnComplete(usage=Usage(10, 4))])
        client.next_scripts = [
            [ContentDelta("better answer"), TurnComplete(usage=Usage(8, 3))]
        ]
        async with app.run_test() as pilot:
            await pilot.press("h", "i", "enter")
            await wait_until_idle(app)

            app._handle_slash_command("/retry")
            await wait_until_idle(app)

            # The old turn was replaced by a fresh one with the same text.
            self.assertEqual(
                [m.content for m in app.controller.conversation.messages][1:],
                ["hi", "better answer"],
            )
            self.assertEqual(len(app.query(UserMessage)), 1)
            # Only the surviving turn's usage counts.
            self.assertEqual((app.total_in, app.total_out), (8, 3))

    async def test_slash_undo_without_a_turn_is_a_noop(self) -> None:
        app, _client = make_app([ContentDelta("x"), TurnComplete(usage=Usage(1, 1))])
        async with app.run_test() as pilot:
            app._handle_slash_command("/undo")
            await pilot.pause()
            app._handle_slash_command("/retry")
            await pilot.pause()

            notices = [str(n.content) for n in app.query(NoticeLine)]
            self.assertTrue(any("nothing to undo" in n for n in notices))
            self.assertTrue(any("nothing to retry" in n for n in notices))
            self.assertEqual(
                [m.role for m in app.controller.conversation.messages], ["system"]
            )

    async def test_escape_needs_a_second_press_to_interrupt(self) -> None:
        """A single Esc only arms the interrupt and prompts for confirmation;
        a second one cancels the running turn."""

        class GatedClient:
            def __init__(self) -> None:
                self.gate = asyncio.Event()

            async def stream(self, wire_messages, tools=None):
                yield ProcessingStarted()
                yield ContentDelta("working")
                await self.gate.wait()
                yield TurnComplete(usage=Usage(10, 2))

        client = GatedClient()
        controller = ChatController(PROFILE, "test-key", client=client)
        app = AgentApp(PROFILE, "test-key", controller=controller)
        async with app.run_test() as pilot:
            await pilot.press("a", "enter")
            await wait_for(lambda: app._turn_busy)

            await pilot.press("escape")
            await pilot.pause()
            # Still running: the first Esc only asked for confirmation.
            self.assertTrue(app._turn_busy)
            warnings = [str(w.content) for w in app.query(MessageLine)]
            self.assertTrue(any("press esc again" in w for w in warnings))

            await pilot.press("escape")
            await wait_until_idle(app)
            self.assertFalse(app._turn_busy)
            errors = [str(e.content) for e in app.query(ErrorLine)]
            self.assertTrue(any("interrupted" in e for e in errors))

    async def test_click_a_message_to_undo_back_to_it(self) -> None:
        """Clicking a user message reveals the undo affordance; clicking that
        rolls the conversation back to the clicked turn."""
        app, client = make_app([ContentDelta("one"), TurnComplete(usage=Usage(5, 3))])
        client.next_scripts = [
            [ContentDelta("two"), TurnComplete(usage=Usage(6, 4))],
            [ContentDelta("three"), TurnComplete(usage=Usage(7, 5))],
        ]
        async with app.run_test() as pilot:
            for key in ("a", "b", "c"):
                await pilot.press(key, "enter")
                await wait_until_idle(app)

            first = app.query(UserMessage).first()
            # Scroll the log home so the (older) first message is on screen.
            app.query_one("#chat-log", VerticalScroll).scroll_home(animate=False)
            await pilot.pause()
            await pilot.click(first, offset=(3, 1))
            await pilot.pause()
            self.assertTrue(first.has_class("undo-armed"))

            # The affordance hugs its own label, not the whole message line.
            button = first.query_one(".user-undo")
            self.assertLess(button.region.width, first.region.width)

            await pilot.click(button, offset=(2, 0))
            await wait_for(lambda: app.query_one(ChatInput).text == "a")

            # Every turn from the clicked one on is gone; its text is back in
            # the input box ready to edit or resend.
            self.assertEqual(
                [m.role for m in app.controller.conversation.messages], ["system"]
            )
            self.assertEqual(len(app.query(UserMessage)), 0)

    async def test_compaction_shows_a_shimmering_indicator(self) -> None:
        """While the summarizer runs, a shimmering "Compacting" line stands in
        for the (otherwise silent) pause."""
        app, _client = make_app([ContentDelta("one"), TurnComplete(usage=Usage(5, 40))])
        async with app.run_test() as pilot:
            await pilot.press("h", "i", "enter")
            await wait_until_idle(app)

            gate = asyncio.Event()

            async def gated_compact() -> str:
                await gate.wait()
                return "compacted: 40 → 7 tokens (freed 33)"

            app.controller.compact = gated_compact  # type: ignore[method-assign]
            task = asyncio.create_task(app._run_compaction())
            await wait_for(lambda: app._compacting)
            await pilot.pause()

            labels = [
                indicator.query_one("#pending-shimmer", ShimmerText).label
                for indicator in app.query(PendingIndicator)
            ]
            self.assertIn("Compacting", labels)

            gate.set()
            await task
            await wait_for(lambda: not app._compacting)
            await pilot.pause()
            self.assertEqual(len(app.query(PendingIndicator)), 0)

    async def test_interrupt_saves_the_partial_answer(self) -> None:
        """Interrupting mid-answer keeps the text (and reasoning) the model had
        produced, up to the last non-thinking block."""

        class GatedClient:
            def __init__(self) -> None:
                self.gate = asyncio.Event()

            async def stream(self, wire_messages, tools=None):
                yield ProcessingStarted()
                yield ReasoningDelta("thinking hard")
                yield ContentDelta("partial answer")
                await self.gate.wait()
                yield TurnComplete(usage=Usage(10, 2))

        client = GatedClient()
        controller = ChatController(PROFILE, "test-key", client=client)
        app = AgentApp(PROFILE, "test-key", controller=controller)
        async with app.run_test() as pilot:
            await pilot.press("h", "i", "enter")
            await wait_for(lambda: app.query(AssistantMessage))

            await pilot.press("escape")
            await pilot.press("escape")
            await wait_until_idle(app)

            roles = [m.role for m in app.controller.conversation.messages]
            self.assertEqual(roles, ["system", "user", "assistant"])
            saved = app.controller.conversation.messages[-1]
            self.assertEqual(saved.content, "partial answer")
            self.assertEqual(saved.reasoning, "thinking hard")

    async def test_interrupt_before_any_answer_saves_nothing(self) -> None:
        """A turn interrupted while still thinking (no answer text) adds no
        assistant message."""

        class GatedClient:
            def __init__(self) -> None:
                self.gate = asyncio.Event()

            async def stream(self, wire_messages, tools=None):
                yield ProcessingStarted()
                yield ReasoningDelta("still thinking")
                await self.gate.wait()
                yield TurnComplete(usage=Usage(10, 2))

        client = GatedClient()
        controller = ChatController(PROFILE, "test-key", client=client)
        app = AgentApp(PROFILE, "test-key", controller=controller)
        async with app.run_test() as pilot:
            await pilot.press("h", "i", "enter")
            await wait_for(lambda: app.query(ThinkingBlock))

            await pilot.press("escape")
            await pilot.press("escape")
            await wait_until_idle(app)

            self.assertEqual(
                [m.role for m in app.controller.conversation.messages],
                ["system", "user"],
            )

    async def test_stream_error_saves_the_partial_answer(self) -> None:
        """A round that ends with a stream error keeps its partial answer, the
        same as a user interrupt — not just a cancellation."""
        app, _client = make_app([
            ContentDelta("half an answer"),
            StreamError("boom"),
        ])
        async with app.run_test() as pilot:
            await pilot.press("h", "i", "enter")
            await wait_until_idle(app)

            self.assertEqual(
                [m.role for m in app.controller.conversation.messages],
                ["system", "user", "assistant"],
            )
            self.assertEqual(
                app.controller.conversation.messages[-1].content, "half an answer"
            )
            errors = [str(e.content) for e in app.query(ErrorLine)]
            self.assertTrue(any("boom" in e for e in errors))

    async def test_compaction_reports_token_counts_and_mounts_the_handoff(self):
        app, client = make_app([ContentDelta("one"), TurnComplete(usage=Usage(5, 40))])
        async with app.run_test() as pilot:
            await pilot.press("h", "i", "enter")
            await wait_until_idle(app)

            client.script = [
                ContentDelta("summary text"), TurnComplete(usage=Usage(100, 7))
            ]
            result = await app._run_compaction()

            # Before/after on one line, then the summary itself.
            first_line, _, rest = result.partition("\n")
            self.assertEqual(first_line, "compacted: 40 → 7 tokens (freed 33)")
            self.assertIn("summary text", rest)
            # The summarization round's spend is in the session totals.
            self.assertEqual((app.total_in, app.total_out), (105, 47))

            # The handoff itself is in the log, rendered as a distinct
            # padded block (not a plain user message), below the divider.
            summaries = app.query(CompactionSummary)
            self.assertEqual(len(summaries), 1)
            self.assertIn("summary text", summaries[0].text)

    async def test_resume_replays_the_compaction_handoff_distinctly(self) -> None:
        """A compacted session replays with the handoff as a CompactionSummary
        (padded, not a plain user message) below the divider."""
        app, client = make_app([ContentDelta("one"), TurnComplete(usage=Usage(5, 40))])
        async with app.run_test() as pilot:
            await pilot.press("h", "i", "enter")
            await wait_until_idle(app)
            client.script = [
                ContentDelta("summary text"), TurnComplete(usage=Usage(100, 7))
            ]
            await app._run_compaction()
            await app._render_history()

            self.assertEqual(len(app.query(CompactionSummary)), 1)
            self.assertIn(
                "summary text",
                app.query(CompactionSummary)[0].text,
            )
            notices = [str(n.content) for n in app.query(MessageLine)]
            self.assertTrue(any("compacted" in n for n in notices))


class TestChatScroll(unittest.IsolatedAsyncioTestCase):
    """The chat log follows the stream, but only while the user is at the
    bottom — scrolling up mid-turn must not be undone."""

    @staticmethod
    def _wheel_up(chat, times: int = 1) -> None:
        # What Textual's MouseScrollUp handler does; used directly because the
        # test drives the scroll position rather than the terminal's mouse.
        for _ in range(times):
            chat._scroll_up_for_pointer(animate=False)

    @staticmethod
    def _wheel_down(chat, times: int = 1) -> None:
        for _ in range(times):
            chat._scroll_down_for_pointer(animate=False)

    def _long_script(self) -> list[Any]:
        script: list[Any] = [ContentDelta(f"line {i}\n\n") for i in range(300)]
        script.append(TurnComplete(usage=Usage(10, 10)))
        return script

    async def test_stream_follows_while_at_the_bottom(self) -> None:
        app, _client = make_app(self._long_script(), delay=0.005)
        async with app.run_test(size=(80, 24)):
            chat = app.query_one("#chat-log", VerticalScroll)
            app._submit("hi")
            await wait_for(lambda: chat.max_scroll_y > 20)
            self.assertTrue(chat.is_anchored)
            self.assertEqual(chat.scroll_y, chat.max_scroll_y)
            await wait_until_idle(app)

    async def test_scrolling_up_mid_turn_is_not_yanked_back(self) -> None:
        app, _client = make_app(self._long_script(), delay=0.005)
        async with app.run_test(size=(80, 24)):
            chat = app.query_one("#chat-log", VerticalScroll)
            app._submit("hi")
            # The live tail's rendered height is capped, so the log grows
            # only a little mid-stream; wait for it to become scrollable.
            await wait_for(lambda: chat.max_scroll_y > 2)

            self._wheel_up(chat, times=2)
            parked = chat.scroll_y

            # Still streaming: the viewport stays put while the live tail
            # keeps receiving text (growth shows up in content, not height).
            live = app.query_one(".assistant-live", Static)
            seen = {str(live.content)}
            for _ in range(10):
                await asyncio.sleep(0.02)
                self.assertEqual(chat.scroll_y, parked)
                seen.add(str(live.content))
            self.assertGreater(len(seen), 1)
            self.assertLess(chat.scroll_y, chat.max_scroll_y)
            self.assertTrue(chat._anchor_released)
            await wait_until_idle(app)

    async def test_scrolling_back_to_the_bottom_resumes_following(self) -> None:
        app, _client = make_app(self._long_script(), delay=0.005)
        async with app.run_test(size=(80, 24)):
            chat = app.query_one("#chat-log", VerticalScroll)
            app._submit("hi")
            await wait_for(lambda: chat.max_scroll_y > 20)

            self._wheel_up(chat, times=5)
            self.assertTrue(chat._anchor_released)

            self._wheel_down(chat, times=int(chat.max_scroll_y) + 5)
            self.assertFalse(chat._anchor_released)
            # Following again: the viewport tracks the growing log.
            for _ in range(10):
                await asyncio.sleep(0.02)
                self.assertEqual(chat.scroll_y, chat.max_scroll_y)
            await wait_until_idle(app)


if __name__ == "__main__":
    unittest.main()
