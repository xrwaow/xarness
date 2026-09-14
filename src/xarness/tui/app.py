"""The xarness chat application.

Owns the event loop between :class:`ChatController` (conversation + API) and
the widget tree. All state transitions — processing → thinking → answer
streaming — are driven by stream events, never by timers.
"""

from __future__ import annotations

import asyncio
import time
from typing import ClassVar, Literal

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, VerticalScroll
from textual.widgets import OptionList, TextArea
from textual.worker import Worker

from pathlib import Path

from .. import theme
from ..config import ConfigError, ProviderProfile, load_all_profiles, resolve_api_key
from ..controller import ChatController
from ..conversation import Conversation, Message
from ..file_search import search_files
from ..events import (
    ContentDelta, ReasoningDelta, StreamError, ToolCallArgumentsDone,
    ToolCallStarted, ToolCallStatus, TurnComplete,
)
from ..prompts import GENERAL_SYSTEM_PROMPT, system_prompt_for
from ..sandbox import SandboxConfig, SandboxSession
from ..tools import ToolRegistry, build_registry
from .ask_screen import AskScreen
from .picker_screen import PickerScreen
from .resume_screen import ResumeScreen
from .widgets import (
    AssistantMessage, ChatInput, ErrorLine, NoticeLine, PendingIndicator,
    ShimmerText, StatusBar, SuggestionPopup, ThinkingBlock, ToolCallBlock,
    ToolWritingIndicator, UserMessage, _format_duration,
)


SLASH_COMMANDS = [
    ("tools", "list available tools"),
    ("model", "choose what model and reasoning effort to use"),
    ("sessions", "resume a previous session"),
    ("mode", "switch between plan (read-only) and write mode"),
    ("theme", "choose a color theme"),
    ("new", "start a new chat"),
]


class AgentApp(App[None]):
    CSS_PATH = "styles.tcss"
    TITLE = "xarness"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+t", "toggle_thoughts", "Thoughts", priority=True),
        Binding("ctrl+c", "copy_or_quit", "Copy / Quit", priority=True),
        Binding("escape", "interrupt", "Interrupt", priority=True),
    ]

    def __init__(
        self,
        profile: ProviderProfile,
        api_key: str | None,
        controller: ChatController | None = None,
        tool_registry: ToolRegistry | None = None,
        workspace: Path | None = None,
        session_name: str | None = None,
        config_path: Path | None = None,
        profile_name: str | None = None,
        sandbox: SandboxConfig | None = None,
        sandbox_session: SandboxSession | None = None,
    ) -> None:
        super().__init__()
        self.profile = profile
        self.api_key = api_key
        # Kept on the app so /mode can rebuild a registry without re-deriving
        # the sandbox setup.
        self.sandbox = sandbox
        self.sandbox_session = sandbox_session
        self.mode: Literal["plan", "write"] = "write"
        self.tool_registry = tool_registry or build_registry(
            sandbox, sandbox_session, mode=self.mode,
            ask_callback=self._ask_user, compact_callback=self._compact_conversation,
        )
        self.controller = controller or ChatController(
            profile, api_key, tool_registry=self.tool_registry
        )
        self.workspace = workspace
        self.session_name = session_name
        self.config_path = config_path
        self.profile_name = profile_name
        self.total_in = 0
        self.total_out = 0
        self.last_in = 0
        self._turn_busy = False
        self._queued: list[str] = []
        self._last_thinking: ThinkingBlock | None = None
        self._worker: Worker | None = None
        self._at_search_timer = None
        self._profiles_cache: dict[str, ProviderProfile] = {}
        self.ensure_system_message()

    def ensure_system_message(self) -> None:
        """Make sure the conversation leads with the mode-aware system prompt.

        Called at startup, after a conversation swap (/resume, cli.py's
        existing-session load), and on /mode. An existing system message is
        updated IN PLACE rather than appended: there is exactly one, at index
        0, and rewriting it means a mode switch applies from the very next
        round onward without duplicating prompts in the history.
        """
        conversation = self.controller.conversation
        content = system_prompt_for(GENERAL_SYSTEM_PROMPT, self.mode)
        if conversation.messages and conversation.messages[0].role == "system":
            conversation.messages[0].content = content
        else:
            conversation.messages.insert(0, Message(role="system", content=content))

    def get_css_variables(self) -> dict[str, str]:
        return {**super().get_css_variables(), **theme.CSS_VARIABLES}

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat-log")
        with Container(id="input-wrap"):
            yield SuggestionPopup(id="suggestion-popup")
            yield ChatInput(
                id="chat-input",
                placeholder="Send a message…  (Enter: send · Shift+Enter: newline · Ctrl+T: thoughts)",
            )
        yield StatusBar(id="status-bar")

    async def on_mount(self) -> None:
        self.query_one("#chat-input", ChatInput).focus()
        self._refresh_status()
        # A conversation loaded before mount (e.g. `--session <name>` resume
        # in cli.py) has never been rendered — replay it into the chat log.
        if any(m.role != "system" for m in self.controller.conversation.messages):
            await self._render_history()

    # ------------------------------------------------------------------
    # Status bar

    def _refresh_status(self) -> None:
        context_used = self.last_in + self.total_out
        self.query_one("#status-bar", StatusBar).update_status(
            shown_name=self.profile.display_name,
            effort=self.profile.cot_strength.value,
            mode=self.mode,
            total_in=self.total_in,
            total_out=self.total_out,
            context_used=context_used,
            max_context=self.profile.max_context,
        )

    # ------------------------------------------------------------------
    # Submission / turn lifecycle

    def on_chat_input_chat_submitted(self, event: ChatInput.ChatSubmitted) -> None:
        text = event.text.strip()
        if text.startswith("/"):
            self._handle_slash_command(text)
            return
        self._submit(text)

    def _handle_slash_command(self, text: str) -> None:
        parts = text[1:].split(maxsplit=1)
        cmd = parts[0] if parts else ""
        chat = self.query_one("#chat-log", VerticalScroll)
        if cmd == "tools":
            names = ", ".join(t["function"]["name"] for t in self.tool_registry.schema())
            chat.mount(ErrorLine(f"tools: {names}"))
        elif cmd == "sessions":
            self.push_screen(ResumeScreen(), self._on_session_selected)
        elif cmd == "new":
            self._start_new_session()
        elif cmd == "model":
            if self.config_path is None:
                chat.mount(ErrorLine("/model unavailable: no config path known"))
                return
            try:
                self._profiles_cache = load_all_profiles(self.config_path)
            except ConfigError as exc:
                chat.mount(ErrorLine(f"/model failed: {exc}"))
                return
            self.push_screen(
                PickerScreen(
                    list(self._profiles_cache),
                    current=self.profile_name,
                    title="Switch model",
                ),
                self._on_model_selected,
            )
        elif cmd == "mode":
            new_mode = "plan" if self.mode == "write" else "write"
            self.mode = new_mode
            self.tool_registry = build_registry(
                self.sandbox, self.sandbox_session, mode=new_mode,
                ask_callback=self._ask_user, compact_callback=self._compact_conversation,
            )
            self.controller.tools = self.tool_registry  # rewire to the new registry
            self.ensure_system_message()
            self._refresh_status()
        elif cmd == "theme":
            arg = parts[1].strip() if len(parts) > 1 else ""
            if arg:
                self._handle_theme_command(arg)
            else:
                # No argument: open the picker, same interaction as /model.
                self.push_screen(
                    PickerScreen(
                        list(theme.THEMES),
                        current=theme.CURRENT_THEME,
                        title="Color theme",
                    ),
                    self._on_theme_selected,
                )
        else:
            chat.mount(ErrorLine(f"unknown command: /{cmd}"))

    def _on_theme_selected(self, name: str | None) -> None:
        if name:
            self._apply_theme(name)

    def _handle_theme_command(self, arg: str) -> None:
        """Apply an explicitly named theme, or report the available ones."""
        chat = self.query_one("#chat-log", VerticalScroll)
        if arg not in theme.THEMES:
            chat.mount(ErrorLine(f"unknown theme '{arg}'; available: {', '.join(theme.THEMES)}"))
            return
        self._apply_theme(arg)

    def _apply_theme(self, name: str) -> None:
        """Switch palette live: CSS variables, input caret, shimmers, dots,
        summaries, status bar — anything that baked palette colors at render
        time must be re-rendered here, or the switched theme drifts from the
        same theme set at startup."""
        theme.set_theme(name)
        self.refresh_css()
        self.query_one("#chat-input", ChatInput).apply_input_theme()
        for shimmer in self.query(ShimmerText):
            colors = (
                theme.SHIMMER_PROCESSING if shimmer.id == "pending-shimmer"
                else theme.SHIMMER_THINKING
            )
            shimmer.set_colors(*colors)
        for message in self.query(UserMessage):
            message.apply_palette()
        for block in self.query(ToolCallBlock):
            block.recolor()
        for thinking in self.query(ThinkingBlock):
            thinking.recolor()
        for block in self.query(ToolCallBlock):
            block.recolor()
        self._refresh_status()

    def _start_new_session(self) -> None:
        """Slash /new: fresh conversation, fresh session file (unless disabled)."""
        self.controller.conversation = Conversation()
        if self.session_name is not None:
            from ..session_store import new_session_name
            self.session_name = new_session_name()
        self.ensure_system_message()
        self.total_in = self.total_out = self.last_in = 0
        self._last_thinking = None
        self._queued.clear()
        chat = self.query_one("#chat-log", VerticalScroll)
        chat.remove_children()
        self._refresh_status()

    async def _on_session_selected(self, name: str | None) -> None:
        if not name:
            return
        from ..session_store import load_session
        self.controller.conversation = load_session(name)
        self.session_name = name
        self.ensure_system_message()
        await self._render_history()

    def _on_model_selected(self, name: str | None) -> None:
        if not name:
            return
        profile = self._profiles_cache.get(name)
        if profile is None:
            return
        try:
            api_key = resolve_api_key(profile)
        except ConfigError as exc:
            chat = self.query_one("#chat-log", VerticalScroll)
            chat.mount(ErrorLine(f"could not switch model: {exc}"))
            return
        # Swap the provider underneath the existing controller: client,
        # conversation, and tools all carry over.
        self.controller.switch_profile(profile, api_key)
        self.profile = profile
        self.profile_name = name
        self._refresh_status()

    async def _render_history(self) -> None:
        """Rebuild #chat-log from self.controller.conversation.messages.

        Used by /resume. Best-effort reconstruction: tool-call status is
        inferred from whether the recorded tool-role content starts with
        "error: " (see ChatController.record_tool_result), since the
        original ToolCallStatus enum value isn't itself persisted.
        """
        chat = self.query_one("#chat-log", VerticalScroll)
        await chat.remove_children()
        messages = self.controller.conversation.messages
        tool_results = {m.tool_call_id: m for m in messages if m.role == "tool" and m.tool_call_id}

        for message in messages:
            if message.role == "user":
                await chat.mount(UserMessage(message.content))
            elif message.role == "assistant":
                if message.reasoning:
                    thinking = ThinkingBlock()
                    await chat.mount(thinking)
                    thinking.append_reasoning(message.reasoning)
                    thinking.finish(message.reasoning_seconds, estimate_if_unknown=False)
                # Tool-call-only rounds render no AssistantMessage body; an
                # empty assistant message with no tool calls replays as
                # "(no output)", matching the live-stream finalize path.
                if message.content or not message.tool_calls:
                    assistant = AssistantMessage()
                    await chat.mount(assistant)
                    if message.content:
                        await assistant.append_delta(message.content)
                    await assistant.finalize()
                for call in message.tool_calls or []:
                    call_id = call.get("id", "")
                    fn = call.get("function", {})
                    block = ToolCallBlock(call_id, fn.get("name", ""))
                    await chat.mount(block)
                    block.append_arguments(fn.get("arguments", ""))
                    result = tool_results.get(call_id)
                    if result is not None:
                        is_error = result.content.startswith("error: ")
                        status = ToolCallStatus.CALL_FAILED if is_error else ToolCallStatus.CALL_SUCCEEDED
                        block.set_result(
                            status,
                            output="" if is_error else result.content,
                            error=result.content[len("error: "):] if is_error else "",
                        )
        chat.scroll_end(animate=False)

    def on_chat_input_slash_query(self, event: ChatInput.SlashQuery) -> None:
        q = event.query.lower()
        matches = [(name, f"/{name}  {desc}") for name, desc in SLASH_COMMANDS if name.startswith(q)]
        # Exact match first (stable sort keeps declaration order otherwise), so
        # typing "/mode" highlights /mode rather than /model.
        matches.sort(key=lambda m: m[0] != q)
        self._show_popup(matches)

    def on_chat_input_at_query(self, event: ChatInput.AtQuery) -> None:
        self._cancel_at_search()
        self._at_search_timer = self.set_timer(0.15, lambda: self._run_at_search(event.query))

    def _cancel_at_search(self) -> None:
        if self._at_search_timer is not None:
            self._at_search_timer.stop()
            self._at_search_timer = None

    @work(exclusive=True, group="at-search")
    async def _run_at_search(self, query: str) -> None:
        chat_input = self.query_one("#chat-input", ChatInput)
        if chat_input.popup_active != "at":
            return  # dismissed (e.g. escape) while the search was pending
        root = self.workspace or Path.cwd()
        results = await search_files(root, query)
        self._show_popup([(r, r) for r in results])

    def on_chat_input_popup_dismiss(self, event: ChatInput.PopupDismiss) -> None:
        self._cancel_at_search()
        self._hide_popup()

    def on_chat_input_popup_nav(self, event: ChatInput.PopupNav) -> None:
        self.query_one("#suggestion-popup", SuggestionPopup).move_highlight(event.direction)

    def on_chat_input_popup_confirm(self, event: ChatInput.PopupConfirm) -> None:
        popup = self.query_one("#suggestion-popup", SuggestionPopup)
        value = popup.selected_value
        chat_input = self.query_one("#chat-input", ChatInput)
        was_slash = chat_input.popup_active == "slash"
        chat_input.popup_active = None
        self._hide_popup()
        chat_input.focus()
        if value is None:
            return
        if was_slash:
            # Slash commands take no arguments: complete and run in one Enter.
            chat_input.load_text("")
            self._handle_slash_command(f"/{value}")
        else:
            chat_input.insert_mention(value)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Mouse click on a popup option: insert it immediately, keep focus on input."""
        event.stop()
        chat_input = self.query_one("#chat-input", ChatInput)
        was_slash = chat_input.popup_active == "slash"
        value = event.option.id
        self._hide_popup()
        chat_input.popup_active = None
        chat_input.focus()
        if value is None:
            return
        if was_slash:
            # Consistent with Enter: clicking a slash command runs it.
            chat_input.load_text("")
            self._handle_slash_command(f"/{value}")
        else:
            chat_input.insert_mention(value)

    def _show_popup(self, items: list[tuple[str, str]]) -> None:
        popup = self.query_one("#suggestion-popup", SuggestionPopup)
        popup.set_items(items)
        if items:
            popup.add_class("visible")
        else:
            popup.remove_class("visible")

    def _hide_popup(self) -> None:
        self.query_one("#suggestion-popup", SuggestionPopup).remove_class("visible")

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

    def action_copy_or_quit(self) -> None:
        """ctrl+c: copy the active selection if there is one, quit otherwise."""
        focused = self.focused
        if isinstance(focused, TextArea) and focused.selected_text:
            self.copy_to_clipboard(focused.selected_text)
            return
        selected = self.screen.get_selected_text()
        if selected:
            self.copy_to_clipboard(selected)
            return
        self.exit()

    def action_interrupt(self) -> None:
        chat_input = self.query_one("#chat-input", ChatInput)
        popup = self.query_one("#suggestion-popup", SuggestionPopup)
        if chat_input.popup_active is not None or popup.has_class("visible"):
            chat_input.popup_active = None
            self._cancel_at_search()
            self._hide_popup()
            return
        if self._turn_busy and self._worker is not None:
            self._worker.cancel()

    async def _ask_user(self, questions: list[str]) -> list[str] | None:
        """Callback for the ask tool: surface the questions, await the answers.

        Runs inside the turn worker, so wait_for_dismiss is allowed; it makes
        the awaited result the screen's dismiss value (the answers).
        """
        return await self.push_screen(AskScreen(questions), wait_for_dismiss=True)

    async def _compact_conversation(self) -> str:
        """Callback for the compact tool: summarize + truncate the history."""
        return await self.controller.compact()

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
        turn_start = time.monotonic()
        tool_blocks: dict[str, ToolCallBlock] = {}
        indicator: PendingIndicator | None = None
        indicator_live = False
        writing: ToolWritingIndicator | None = None
        pending_writes = 0
        thinking: ThinkingBlock | None = None
        had_stream_error = False

        stream = self.controller.send(text)
        try:
            while True:
                indicator = PendingIndicator()
                indicator_live = True
                await chat.mount(indicator)

                thinking = None
                assistant = None
                tool_blocks = {}
                writing = None
                pending_writes = 0
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
                        # One shared "Writing tools" shimmer while arguments
                        # stream; per-call blocks appear once args are complete.
                        pending_writes += 1
                        if writing is None:
                            writing = ToolWritingIndicator()
                            await chat.mount(writing)
                    elif isinstance(event, ToolCallArgumentsDone):
                        block = ToolCallBlock(event.call_id, event.name)
                        block.append_arguments(event.arguments_json)
                        tool_blocks[event.call_id] = block
                        await chat.mount(block)
                        pending_writes = max(0, pending_writes - 1)
                        if pending_writes == 0 and writing is not None:
                            await self._dismiss_indicator(writing)
                            writing = None
                    elif isinstance(event, TurnComplete):
                        round_has_tools = event.has_tool_calls
                        await self._complete_round(event, thinking, assistant)
                    elif isinstance(event, StreamError):
                        if indicator_live:
                            await self._dismiss_indicator(indicator)
                            indicator_live = False
                        await chat.mount(ErrorLine(event.message))
                        had_stream_error = True

                    if follow:
                        chat.scroll_end(animate=False)

                if writing is not None:
                    await self._dismiss_indicator(writing)
                    writing = None
                if indicator_live:
                    await self._dismiss_indicator(indicator)

                if not round_has_tools:
                    if not had_stream_error:
                        worked = _format_duration(time.monotonic() - turn_start)
                        await chat.mount(NoticeLine(f"worked for {worked}"))
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
            if writing is not None and writing.is_mounted:
                await writing.remove()
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

        if self.session_name:
            from ..session_store import save_session
            save_session(self.session_name, self.profile.model_id, self.controller.conversation)

    def action_toggle_thoughts(self) -> None:
        if self._last_thinking is not None:
            self._last_thinking.toggle()

    # ------------------------------------------------------------------
    # Scrolling

    def _at_bottom(self, chat: VerticalScroll) -> bool:
        return chat.scroll_y >= chat.max_scroll_y - 2
