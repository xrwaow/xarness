"""The xarness chat application.

Owns the event loop between :class:`ChatController` (conversation + API) and
the widget tree. All state transitions — processing → thinking → answer
streaming — are driven by stream events, never by timers.
"""

from __future__ import annotations

import asyncio
from typing import ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, VerticalScroll
from textual.worker import Worker

from .. import theme
from ..config import ProviderProfile
from ..controller import ChatController
from ..events import (
    ContentDelta,
    ReasoningDelta,
    StreamError,
    ToolCallArgumentsDelta,
    ToolCallStarted,
    ToolCallStatus,
    TurnComplete,
)
from ..tools import ToolRegistry, default_registry
from .widgets import (
    AssistantMessage,
    ChatInput,
    ErrorLine,
    PendingIndicator,
    StatusBar,
    ThinkingBlock,
    ToolCallBlock,
    UserMessage,
)


class AgentApp(App[None]):
    CSS_PATH = "styles.tcss"
    TITLE = "xarness"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+t", "toggle_thoughts", "Thoughts", priority=True),
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("escape", "interrupt", "Interrupt", priority=True),
    ]

    def __init__(
        self,
        profile: ProviderProfile,
        api_key: str | None,
        controller: ChatController | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        super().__init__()
        self.profile = profile
        self.tool_registry = tool_registry or default_registry()
        self.controller = controller or ChatController(
            profile, api_key, tool_registry=self.tool_registry
        )
        self.total_in = 0
        self.total_out = 0
        self.last_in = 0
        self._turn_busy = False
        self._queued: list[str] = []
        self._last_thinking: ThinkingBlock | None = None
        self._worker: Worker | None = None

    def get_css_variables(self) -> dict[str, str]:
        return {**super().get_css_variables(), **theme.CSS_VARIABLES}

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat-log")
        yield StatusBar(id="status-bar")
        with Container(id="input-wrap"):
            yield ChatInput(
                id="chat-input",
                placeholder="Send a message…  (Enter: send · Shift+Enter: newline · Ctrl+T: thoughts)",
            )

    def on_mount(self) -> None:
        self.query_one("#chat-input", ChatInput).focus()
        self._refresh_status()

    # ------------------------------------------------------------------
    # Status bar

    def _refresh_status(self) -> None:
        context_used = self.last_in + self.total_out
        self.query_one("#status-bar", StatusBar).update_status(
            shown_name=self.profile.display_name,
            effort=self.profile.cot_strength.value,
            total_in=self.total_in,
            total_out=self.total_out,
            context_used=context_used,
            max_context=self.profile.max_context,
        )

    # ------------------------------------------------------------------
    # Submission / turn lifecycle

    def on_chat_input_chat_submitted(self, event: ChatInput.ChatSubmitted) -> None:
        self._submit(event.text)

    def _submit(self, text: str) -> None:
        chat = self.query_one("#chat-log", VerticalScroll)
        follow = self._at_bottom(chat)
        user_message = UserMessage(text)
        if self._turn_busy:
            user_message.mark_queued()
            self._queued.append(text)
        chat.mount(user_message)
        if follow:
            chat.scroll_end(animate=False)
        if not self._turn_busy:
            self._worker = self._run_turn(text)

    def action_interrupt(self) -> None:
        if self._turn_busy and self._worker is not None:
            self._worker.cancel()

    @work(group="turn")
    async def _run_turn(self, text: str) -> None:
        """Drive the full exchange: rounds of (thinking → answer → tool calls).

        Each round streams one model response. If the round requested tool
        calls, they're executed, their blocks settle, results are recorded
        into the conversation, and the next round streams — until a round
        comes back with no tool calls.
        """
        self._turn_busy = True
        chat = self.query_one("#chat-log", VerticalScroll)
        tool_blocks: dict[str, ToolCallBlock] = {}
        indicator: PendingIndicator | None = None
        indicator_live = False
        thinking: ThinkingBlock | None = None

        stream = self.controller.send(text)
        try:
            while True:
                indicator = PendingIndicator()
                indicator_live = True
                await chat.mount(indicator)

                thinking = None
                assistant = None
                tool_blocks = {}
                round_has_tools = False

                async for event in stream:
                    follow = self._at_bottom(chat)

                    if isinstance(event, ReasoningDelta):
                        if thinking is None:
                            await self._dismiss_indicator(indicator)
                            indicator_live = False
                            thinking = ThinkingBlock()
                            self._last_thinking = thinking
                            await chat.mount(thinking)
                        thinking.append_reasoning(event.text)
                    elif isinstance(event, ContentDelta):
                        if assistant is None:
                            if indicator_live:
                                await self._dismiss_indicator(indicator)
                                indicator_live = False
                            if thinking is not None:
                                thinking.finish(duration=None)
                            assistant = AssistantMessage()
                            await chat.mount(assistant)
                        await assistant.append_delta(event.text)
                    elif isinstance(event, ToolCallStarted):
                        if indicator_live:
                            await self._dismiss_indicator(indicator)
                            indicator_live = False
                        if thinking is not None:
                            thinking.finish(duration=None)
                        block = ToolCallBlock(event.call_id, event.name)
                        tool_blocks[event.call_id] = block
                        await chat.mount(block)
                    elif isinstance(event, ToolCallArgumentsDelta):
                        block = tool_blocks.get(event.call_id)
                        if block is not None:
                            block.append_arguments(event.text)
                    elif isinstance(event, TurnComplete):
                        round_has_tools = event.has_tool_calls
                        await self._complete_round(event, thinking, assistant)
                    elif isinstance(event, StreamError):
                        if indicator_live:
                            await self._dismiss_indicator(indicator)
                            indicator_live = False
                        await chat.mount(ErrorLine(event.message))

                    if follow:
                        chat.scroll_end(animate=False)

                if indicator_live:
                    await self._dismiss_indicator(indicator)

                if not round_has_tools:
                    break

                # Execute the calls this round requested, in stream order.
                for call_id, block in tool_blocks.items():
                    result = await self.tool_registry.call(block.tool_name, block.accumulated_arguments)
                    if result.parse_error:
                        status = ToolCallStatus.PARSING_ERROR
                    elif result.ok:
                        status = ToolCallStatus.CALL_SUCCEEDED
                    else:
                        status = ToolCallStatus.CALL_FAILED
                    block.set_result(status, output=result.output, error=result.error)
                    self.controller.record_tool_result(call_id, result)

                stream = self.controller.continue_after_tools()
        except asyncio.CancelledError:
            if indicator is not None and indicator_live:
                await self._dismiss_indicator(indicator)
            if thinking is not None and not thinking.done:
                thinking.finish(duration=None)
            for block in tool_blocks.values():
                if block.status is ToolCallStatus.MAKING_CALL:
                    block.set_result(ToolCallStatus.CALL_FAILED, error="interrupted")
            await chat.mount(ErrorLine("interrupted"))
        finally:
            self._turn_busy = False
            self._worker = None
            if self._queued:
                self._worker = self._run_turn(self._queued.pop(0))

    async def _dismiss_indicator(self, indicator: PendingIndicator | None) -> None:
        if indicator is not None and indicator.is_mounted:
            await indicator.remove()

    async def _complete_round(
        self,
        event: TurnComplete,
        thinking: ThinkingBlock | None,
        assistant: AssistantMessage | None,
    ) -> None:
        chat = self.query_one("#chat-log", VerticalScroll)
        if assistant is not None:
            await assistant.finalize()
        elif thinking is None and not event.has_tool_calls:
            await chat.mount(ErrorLine("empty response from model"))
        if thinking is not None:
            thinking.finish(event.reasoning_seconds)

        if event.usage is not None:
            self.total_in += event.usage.input_tokens
            self.total_out += event.usage.output_tokens
            self.last_in = event.usage.input_tokens
        self._refresh_status()

    def action_toggle_thoughts(self) -> None:
        if self._last_thinking is not None:
            self._last_thinking.toggle()

    # ------------------------------------------------------------------
    # Scrolling

    def _at_bottom(self, chat: VerticalScroll) -> bool:
        return chat.scroll_y >= chat.max_scroll_y - 2
