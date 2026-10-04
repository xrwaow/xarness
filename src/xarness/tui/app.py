"""The xarness chat application.

Owns the event loop between :class:`ChatController` (conversation + API) and
the widget tree. All state transitions — processing → thinking → answer
streaming — are driven by stream events, never by timers.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Literal

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Horizontal, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static, TextArea
from textual.worker import Worker

from .. import theme
from ..config import (
    ConfigError, ProviderProfile, load_all_profiles, resolve_api_key, save_preferences,
)
from ..conversation import SUMMARY_PREFIX, Conversation, Message
from ..controller import ChatController, RollbackPlan
from ..events import (
    ContentDelta, ReasoningDelta, StreamError, ToolCallArgumentsDelta,
    ToolCallArgumentsDone, ToolCallStarted, ToolCallStatus, TurnComplete, Usage,
)
from ..file_search import search_files
from .. import images as image_loader
from ..images import ImageAttachment, ImageError
from ..gitwork import (
    DiffStat, GitInfo, GitWorktreeError, RevertConflict, accept_changes,
    attributed_diff_stat, check_blocked_git, diff_stat, git_diff, revert_to_tree,
    setup_tracking, snapshot_tree, tree_diff_stat,
)
from ..prompts import GENERAL_SYSTEM_PROMPT, system_prompt_for
from ..sandbox import SandboxConfig, SandboxSession
from ..tools import ToolRegistry, build_registry
from .confirm_screen import ConfirmScreen
from .container_screen import ContainerSettingsScreen
from .picker_screen import PickerScreen
from .resume_screen import ResumeScreen
from .widgets import (
    AskBar, AssistantMessage, ChatInput, CompactionSummary, DiffSummary,
    ErrorLine, GeneratingBar,
    MessageLine, NoticeLine, PendingIndicator, ShimmerText, StatusBar,
    SteerQueueBar, SuggestionPopup, ThinkingBlock, ToolCallBlock,
    ToolWritingIndicator, UserMessage, _format_duration,
)


INPUT_PLACEHOLDER = (
    "Send a message…  (Enter: send · Shift+Enter: newline · Tab: mode · Ctrl+T: thoughts)"
)
ASK_PLACEHOLDER = "Type your answer…  (Enter: send · Esc: skip)"

# An @-mention token that resolves to an image file attaches it (its literal
# text still goes to the model). A quoted form (@"path with spaces.png")
# covers paths pasted from terminals; plain tokens may not contain spaces.
_IMAGE_TOKEN_RE = re.compile(
    r'@("[^"\n]+\.(?:png|jpe?g|webp|gif|bmp)"|[^\s@]+\.(?:png|jpe?g|webp|gif|bmp))\b',
    re.IGNORECASE,
)

# How long a first Esc stays "armed" waiting for the confirming second one.
INTERRUPT_ARM_SECONDS = 2.5

# Tools that modify workspace files — a successful call means a diff now
# exists, so the pending-changes bar should surface above the input.
_WRITING_TOOLS = frozenset({"write_file", "edit_file"})


def _format_drift_detail(stat: DiffStat, limit: int = 6) -> str:
    """A short plain-text summary of drifted files for the confirm modal."""
    if not stat.files:
        return ""
    lines = []
    for f in stat.files[:limit]:
        tag = " (new)" if f.is_new else ""
        lines.append(f"  {f.path}{tag}  +{f.additions} -{f.deletions}")
    if len(stat.files) > limit:
        lines.append(f"  … and {len(stat.files) - limit} more")
    return "\n".join(lines)


# Divider mounted in the chat log at every compaction point (live and on
# /resume): everything above is history the model no longer sees verbatim.
COMPACTED_DIVIDER = (
    "— compacted: the model sees only the summary below; "
    "messages above are kept here for you —"
)


class _RoundView:
    """Widget state for one streaming round: pending indicator, thinking
    block, assistant message, tool-writing shimmer, tool-call blocks. The
    turn driver (_run_turn) only loops rounds and executes tools."""

    def __init__(self, app: "AgentApp", chat: "VerticalScroll") -> None:
        self.app = app
        self.chat = chat
        self.indicator: PendingIndicator | None = None
        self.indicator_live = False
        self.thinking: ThinkingBlock | None = None
        self.assistant: AssistantMessage | None = None
        self.writing: ToolWritingIndicator | None = None
        self.pending_writes = 0
        self.tool_blocks: dict[str, ToolCallBlock] = {}
        self.round_has_tools = False
        self.had_stream_error = False

    async def mount(self) -> None:
        self.indicator = PendingIndicator()
        self.indicator_live = True
        await self.chat.mount(self.indicator)

    async def handle(self, event) -> None:
        if isinstance(event, ReasoningDelta):
            if self.thinking is None:
                await self._hide_indicator()
                self.thinking = ThinkingBlock()
                self.app._last_thinking = self.thinking
                await self.app._mount_spaced(self.chat, self.thinking)
            self.thinking.append_reasoning(event.text)
        elif isinstance(event, ContentDelta):
            if self.assistant is None:
                await self._hide_indicator()
                if self.thinking is not None:
                    self.thinking.finish(duration=None)
                self.assistant = AssistantMessage()
                await self.app._mount_spaced(self.chat, self.assistant)
            await self.assistant.append_delta(event.text)
        elif isinstance(event, ToolCallStarted):
            await self._hide_indicator()
            if self.thinking is not None:
                self.thinking.finish(duration=None)
            # One shared "Writing <tool>" shimmer while arguments stream; the
            # indicator accumulates the streamed args itself. Per-call blocks
            # appear once args are complete.
            self.pending_writes += 1
            if self.writing is None:
                self.writing = ToolWritingIndicator(event.name)
                await self.chat.mount(self.writing)
            else:
                self.writing.set_tool(event.name)
            self.writing.track_call(event.call_id, event.name)
        elif isinstance(event, ToolCallArgumentsDelta):
            if self.writing is not None:
                self.writing.append_args_delta(event.call_id, event.text)
        elif isinstance(event, ToolCallArgumentsDone):
            block = ToolCallBlock(event.call_id, event.name)
            await self.chat.mount(block)
            # After mount: append_arguments refreshes the summary label in
            # place; before it, the widget isn't mounted and the update no-ops.
            block.append_arguments(event.arguments_json)
            self.tool_blocks[event.call_id] = block
            self.pending_writes = max(0, self.pending_writes - 1)
            if self.pending_writes == 0 and self.writing is not None:
                await self.app._dismiss_indicator(self.writing)
                self.writing = None
        elif isinstance(event, TurnComplete):
            self.round_has_tools = event.has_tool_calls
            await self.app._complete_round(event, self.thinking, self.assistant)
        elif isinstance(event, StreamError):
            await self._hide_indicator()
            if self.assistant is not None:
                # No TurnComplete follows an error: settle the streamed text
                # here, or it stays in the unselectable live Static.
                await self.assistant.finalize()
            await self.chat.mount(ErrorLine(event.message))
            self.had_stream_error = True
            self.app._clear_hint_lines()

    async def _hide_indicator(self) -> None:
        if self.indicator_live:
            await self.app._dismiss_indicator(self.indicator)
            self.indicator_live = False

    async def finish(self) -> None:
        """Dismiss the round's transient indicators at the round's end."""
        if self.writing is not None:
            await self.app._dismiss_indicator(self.writing)
            self.writing = None
        await self._hide_indicator()

    async def cancel(self) -> None:
        """Settle the round's widgets for an interrupted turn."""
        if self.writing is not None and self.writing.is_mounted:
            await self.writing.remove()
        if self.indicator is not None and self.indicator_live:
            await self.app._dismiss_indicator(self.indicator)
        if self.thinking is not None and not self.thinking.done:
            self.thinking.finish(duration=None)
        if self.assistant is not None:
            # Interrupted rounds never reach TurnComplete/finalize: settle the
            # partial answer so it's selectable (and matches the /resume replay).
            await self.assistant.finalize()
        self.app._clear_hint_lines()
        for block in self.tool_blocks.values():
            if block.status is ToolCallStatus.MAKING_CALL:
                block.set_result(ToolCallStatus.CALL_FAILED, error="interrupted")


SLASH_COMMANDS = [
    ("tools", "list available tools"),
    ("model", "choose what model and reasoning effort to use"),
    ("sessions", "resume a previous session"),
    ("mode", "switch between plan (read-only) and write mode"),
    ("container", (
        "network access, .gitignore shadowing, external references, "
        "auto-compact, tool output limit; changes are saved to the config "
        "automatically"
    )),
    ("theme", "choose a color theme"),
    ("new", "start a new chat"),
    ("delete", "delete the current session and start a new one (your files are left exactly as they are)"),
    ("diff", "show pending changes (optionally: /diff <path>)"),
    ("accept", "lock in the changes made so far (they stop showing in /diff and can no longer be undone)"),
    ("reject", "discard all changes made since the last /accept"),
    ("undo", "drop the last turn and revert its file edits"),
    ("retry", "revert the last turn's file edits and resend your message"),
]


class AgentApp(App[None]):
    CSS_PATH = "styles.tcss"
    TITLE = "xarness"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+t", "toggle_thoughts", "Thoughts", priority=True),
        Binding("ctrl+c", "copy_or_quit", "Copy / Quit", priority=True),
        Binding("ctrl+shift+c", "copy_selection", "Copy", priority=True),
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
        tool_output_limit: int | None = None,
        git_info: GitInfo | None = None,
        startup_notices: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.profile = profile
        self.api_key = api_key
        # Kept on the app so /mode can rebuild a registry without re-deriving
        # the sandbox setup.
        self.sandbox = sandbox
        self.sandbox_session = sandbox_session
        self.mode: Literal["plan", "write"] = "write"
        # The app owns the sandbox's mode (build_registry doesn't mutate it).
        if sandbox is not None:
            sandbox.read_only = self.mode != "write"
        # Auto-compact: run compaction automatically when the context window
        # passes the profile's auto_compact_threshold (checked at the end of
        # each turn). Off unless the config enables it; /container toggles.
        self.auto_compact = self.profile.auto_compact
        self.tool_registry = tool_registry or build_registry(
            sandbox, sandbox_session, mode=self.mode,
            ask_callback=self._ask_user,
            git_guard=self._make_git_guard(),
        )
        if tool_output_limit is not None:
            self.tool_registry.max_output_chars = tool_output_limit
        self.controller = controller or ChatController(
            profile, api_key, tool_registry=self.tool_registry
        )
        self.controller.git_info = git_info
        self.workspace = workspace
        self.session_name = session_name
        self.config_path = config_path
        self.profile_name = profile_name
        self.git_info = git_info
        self.startup_notices = startup_notices or []
        self._turn_busy = False
        # /auto_compact: run compaction automatically when the context window
        # passes the profile's auto_compact_threshold (checked at the end of
        # each turn). Off unless the config enables it; /auto_compact toggles.
        self.auto_compact = self.profile.auto_compact
        # True while a compaction is rewriting the history: submissions
        # queue (like a busy turn) instead of racing the rewrite. Auto-
        # compaction runs at the end of a turn; /new //delete wait it out.
        self._compacting = False
        self._queued: list[tuple[str, list[ImageAttachment] | None]] = []
        self._ask_future: asyncio.Future[str | None] | None = None
        self._last_thinking: ThinkingBlock | None = None
        self._worker: Worker | None = None
        # Double-Esc interrupt: the first Esc arms it (and shows a prompt),
        # the second one within INTERRUPT_ARM_SECONDS actually cancels.
        self._interrupt_armed = False
        self._interrupt_timer = None
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
        content = system_prompt_for(GENERAL_SYSTEM_PROMPT, self.mode, self.sandbox)
        if conversation.messages and conversation.messages[0].role == "system":
            conversation.messages[0].content = content
        else:
            conversation.messages.insert(0, Message(role="system", content=content))

    def get_css_variables(self) -> dict[str, str]:
        return {**super().get_css_variables(), **theme.CSS_VARIABLES}

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat-log")
        with Container(id="input-wrap"):
            yield GeneratingBar(id="generating-bar")
            yield DiffSummary(id="diff-summary")
            yield AskBar(id="ask-bar")
            yield SuggestionPopup(id="suggestion-popup")
            yield SteerQueueBar(id="steer-queue-bar")
            with Horizontal(id="input-row"):
                yield Static("›", id="input-prompt")
                yield ChatInput(
                    id="chat-input",
                    placeholder=INPUT_PLACEHOLDER,
                )
        yield StatusBar(id="status-bar")

    async def on_mount(self) -> None:
        self.query_one("#chat-input", ChatInput).focus()
        self._refresh_status()
        # Follow new output only while the user is parked at the bottom.
        # Textual's anchor does exactly that: the log stays pinned to the end as
        # content streams, the anchor is released the moment the user scrolls up
        # (so nothing yanks the viewport back), and it re-arms once they scroll
        # down to the bottom again.
        self.query_one("#chat-log", VerticalScroll).anchor()
        # Container settings live per workspace now: apply whatever this
        # workspace has saved before any turn runs.
        if self.workspace is not None:
            from ..session_store import load_workspace_container
            self._restore_session_container(load_workspace_container(str(self.workspace)))
        # A conversation loaded before mount (e.g. `--session <name>` resume
        # in cli.py) has never been rendered — replay it into the chat log.
        if any(m.role != "system" for m in self.controller.conversation.messages):
            await self._render_history()
        for notice in self.startup_notices:
            self._post_line(NoticeLine(notice))
        # On a resumed session this reconnects the pending-changes summary.
        await self.refresh_diff_summary()
        # A session loaded before mount (e.g. a future `--session <name>` resume)
        # may have drifted while it was closed; surface it the same way a
        # /sessions resume does, rather than deciding for the user.
        drift = await self._detect_resume_drift()
        if drift is not None and self.git_info is not None:
            self._prompt_resume_drift(self.git_info, drift[0], drift[1])

    # ------------------------------------------------------------------
    # Status bar

    def _refresh_status(self) -> None:
        # The last round's prompt already includes the entire prior
        # conversation (including earlier assistant outputs), so the current
        # context is that prompt plus only the newest round's output.
        context_used = self.last_in + self.last_out
        self.query_one("#status-bar", StatusBar).update_status(
            shown_name=self.profile.display_name,
            effort=self.profile.cot_strength.value,
            mode=self.mode,
            total_in=self.total_in,
            total_out=self.total_out,
            context_used=context_used,
            max_context=self.profile.max_context,
            workspace=str(self.workspace) if self.workspace else None,
        )

    # ------------------------------------------------------------------
    # Submission / turn lifecycle

    def on_chat_input_chat_submitted(self, event: ChatInput.ChatSubmitted) -> None:
        # While the ask tool waits for an answer, every submission is the
        # answer — never a chat message or slash command. Empty text is
        # ignored so the "send now" Enter can't submit a blank answer.
        if self._ask_future is not None and not self._ask_future.done():
            if event.text.strip():
                self._ask_future.set_result(event.text)
            return
        text = event.text.strip()
        if not text:
            # Empty input + Enter while messages are queued: send now.
            self._flush_queued_now()
            return
        if text.startswith("/"):
            self._handle_slash_command(text)
            return
        self._submit(text, event.images)

    def _handle_slash_command(self, text: str) -> None:
        parts = text[1:].split(maxsplit=1)
        cmd = parts[0] if parts else ""
        if cmd == "tools":
            names = ", ".join(t["function"]["name"] for t in self.tool_registry.schema())
            self._post_line(MessageLine(f"tools: {names}", kind="info"))
        elif cmd == "sessions":
            self.push_screen(
                ResumeScreen(workspace=str(self.workspace) if self.workspace else None),
                self._on_session_selected,
            )
        elif cmd == "new":
            if self._compacting:
                self._post_line(ErrorLine("/new: wait for the compaction to finish first"))
            else:
                self._start_new_session()
        elif cmd == "delete":
            if self._turn_busy or self._compacting:
                self._post_line(ErrorLine("/delete: wait for the current turn or compaction to finish first"))
            else:
                self._start_delete()
        elif cmd == "accept":
            preflight = self._git_action_preflight("accept")
            if preflight is not None:
                self._start_accept(*preflight)
        elif cmd == "reject":
            preflight = self._git_action_preflight("reject")
            if preflight is not None:
                self._start_reject(*preflight)
        elif cmd == "diff":
            arg = parts[1].strip() if len(parts) > 1 else ""
            self._show_diff_command(arg)
        elif cmd == "model":
            if self.config_path is None:
                self._post_line(ErrorLine("/model unavailable: no config path known"))
                return
            try:
                self._profiles_cache = load_all_profiles(self.config_path)
            except ConfigError as exc:
                self._post_line(ErrorLine(f"/model failed: {exc}"))
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
            self._switch_mode()
        elif cmd == "container":
            if self.sandbox is None:
                self._post_line(ErrorLine(
                    "/container unavailable: filesystem/bash tools are disabled "
                    "(bubblewrap missing), so there is no container to configure"
                ))
            else:
                self.push_screen(ContainerSettingsScreen(
                    self.sandbox, self.config_path,
                    session_auto_compact=self.auto_compact,
                    session_output_limit=self.tool_registry.max_output_chars,
                    default_auto_compact=self.profile.auto_compact,
                    profile_name=self.profile_name,
                    on_auto_compact=self._set_auto_compact,
                    on_output_limit=self._set_output_limit,
                    workspace=str(self.workspace) if self.workspace else None,
                ))
        elif cmd == "undo":
            self._run_undo(resend=False)
        elif cmd == "retry":
            self._run_undo(resend=True)
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
            self._post_line(ErrorLine(f"unknown command: /{cmd}"))

    def _switch_mode(self) -> None:
        """Toggle plan/write mode; shared by /mode and the Tab shortcut."""
        new_mode = "plan" if self.mode == "write" else "write"
        self.mode = new_mode
        if self.sandbox is not None:
            self.sandbox.read_only = new_mode != "write"
        self.tool_registry = build_registry(
            self.sandbox, self.sandbox_session, mode=new_mode,
            ask_callback=self._ask_user,
            git_guard=self._make_git_guard(),
        )
        self.controller.tools = self.tool_registry  # rewire to the new registry
        self.ensure_system_message()
        self._refresh_status()

    # ------------------------------------------------------------------
    # /undo + /retry

    @property
    def total_in(self) -> int:
        return self.controller.usage_total.input_tokens

    @property
    def total_out(self) -> int:
        return self.controller.usage_total.output_tokens

    def _last_usage(self) -> Usage | None:
        messages = self.controller.conversation.messages
        return next((m.usage for m in reversed(messages) if m.usage), None)

    @property
    def last_in(self) -> int:
        return self._last_usage().input_tokens if self._last_usage() else 0

    @property
    def last_out(self) -> int:
        return self._last_usage().output_tokens if self._last_usage() else 0

    def _set_auto_compact(self, value: bool) -> None:
        """Container popup, workspace tab: flip the live auto-compact flag.

        It is persisted per workspace (see the workspace store), never as the
        global profile default."""
        self.auto_compact = value

    def _set_output_limit(self, limit: int) -> None:
        """Container popup, workspace tab: per-tool output cap for this
        workspace."""
        self.tool_registry.max_output_chars = limit

    @work(group="undo", exclusive=True)
    async def _run_undo(self, resend: bool) -> None:
        """Shared body of /undo (drop the last turn, put its user message
        back in the input) and /retry (drop it and resend it).

        Only file edits are reverted — the workspace is restored to the
        turn's checkpoint snapshot. Anything run_bash did beyond the files
        (installs, background jobs, network calls) is not undone."""
        label = "retry" if resend else "undo"
        if self._turn_busy or self._compacting:
            self._post_line(MessageLine(
                f"/{label}: wait for the current turn or compaction to finish first",
                kind="warn",
            ))
            return
        plan = self.controller.rollback_plan()
        if plan is None:
            self._post_line(NoticeLine(f"nothing to {label}"))
            return
        await self._apply_rollback(plan, resend=resend)

    @work(group="undo", exclusive=True)
    async def _run_undo_to(self, message: Message) -> None:
        """Click-to-undo: drop the clicked user message and everything after
        it (file edits reverted), putting that message back in the input."""
        if self._turn_busy or self._compacting:
            self._post_line(MessageLine(
                "wait for the current turn or compaction to finish first",
                kind="warn",
            ))
            return
        plan = self.controller.rollback_plan_to(message)
        if plan is None:
            self._post_line(NoticeLine("nothing to undo there"))
            return
        await self._apply_rollback(plan, resend=False, scope="messages from here on")

    def on_user_message_undo_requested(self, event: UserMessage.UndoRequested) -> None:
        event.stop()
        message = event.user_message.message
        if message is not None:
            self._run_undo_to(message)

    async def _apply_rollback(
        self, plan: RollbackPlan, resend: bool, scope: str = "last turn"
    ) -> None:
        """Carry out a rollback plan: rewrite the history, revert the files,
        and report what happened. Shared by /undo, /retry, and click-to-undo."""
        if plan.compaction_only:
            # Undoing the compaction itself: history restored, the turn that
            # triggered it stays intact, no files reverted, input untouched.
            self.controller.apply_rollback(plan)
            await self._render_history()
            self._refresh_status()
            self._persist_git_state()
            if resend:
                self._post_line(NoticeLine(
                    "compaction undone: pre-compaction history restored — "
                    "nothing to retry (the turn is still in the conversation)"
                ))
            else:
                self._post_line(NoticeLine(
                    "compaction undone: pre-compaction history restored"
                ))
            return
        # Reverse the turn's file edits FIRST, against the live workspace.
        # A conflict aborts the whole undo, leaving both the workspace and
        # the conversation exactly as they were.
        label = "retry" if resend else "undo"
        try:
            await self.controller.revert_changes(plan.revert_turns)
        except RevertConflict as exc:
            self._post_line(ErrorLine(
                f"/{label} conflict: the turn's changes overlap edits made "
                f"since — nothing was changed\n{exc}"
            ))
            return
        except (GitWorktreeError, OSError) as exc:
            self._post_line(ErrorLine(f"/{label} failed: {exc}"))
            return
        self.controller.apply_rollback(plan)
        await self._render_history()
        self._refresh_status()
        await self.refresh_diff_summary()
        if self.git_info is None:
            note = "no git tracking for this session — file edits could not be reverted"
        elif not plan.revert_turns:
            note = "no file edits to revert"
        else:
            note = "file edits reverted"
        if resend:
            self._post_line(NoticeLine(f"retrying: {scope} rolled back ({note})"))
            self._submit(plan.user_text)
        else:
            chat_input = self.query_one("#chat-input", ChatInput)
            chat_input.load_text(plan.user_text)
            chat_input.focus()
            self._post_line(NoticeLine(f"undone: {scope} removed ({note})"))
        # Persist the rolled-back state: without this, /new + /sessions
        # resumes the session as it was before the undo.
        self._persist_git_state()

    # ------------------------------------------------------------------
    # Git change tracking (diff summary + accept/reject)

    def _make_git_guard(self) -> Callable[[str], str | None]:
        """run_bash veto closure; reads self.git_info at call time so /resume
        rebinding takes effect without rebuilding the guard."""

        def guard(command: str) -> str | None:
            info = self.git_info
            if info is None:
                return None
            return check_blocked_git(command, None, info.workspace)

        return guard

    async def refresh_diff_summary(self) -> None:
        """Recompute the pending-changes bar (hidden when empty or gitless)."""
        try:
            widget = self.query_one("#diff-summary", DiffSummary)
        except NoMatches:
            return  # app shutting down; DOM already pruned
        info = self.git_info
        if info is None:
            await widget.clear()
            return
        try:
            # Attribute each file to its source: agent turns vs. changes made
            # outside the session (drift).
            stat = await attributed_diff_stat(info, self.controller.turn_checkpoints())
        except GitWorktreeError:
            await widget.clear()  # tracking broken (e.g. repo gone): stay quiet
            return
        if not stat.files:
            await widget.clear()
        else:
            await widget.set_summary(stat)

    async def on_diff_summary_diff_requested(self, event: DiffSummary.DiffRequested) -> None:
        info = self.git_info
        if info is None:
            return
        widget = self.query_one("#diff-summary", DiffSummary)
        if widget.active_diff_key == event.key:
            widget.hide_diff()  # same affordance again: collapse the diff
            return
        try:
            if event.key is None:
                text = await git_diff(info)
            else:
                text = await git_diff(info, event.key)
        except GitWorktreeError as exc:
            widget.hide_diff()
            self._post_line(ErrorLine(f"diff failed: {exc}"))
            return
        widget.show_diff(event.key, text)

    @work(group="diff-view", exclusive=True)
    async def _show_diff_command(self, arg: str) -> None:
        """Slash /diff: toggle the pending-changes panel — the first /diff
        expands the per-file list, /diff again hides it entirely. With an
        argument, /diff <path> shows that one file's unified diff."""
        info = self.git_info
        if info is None:
            self._post_line(ErrorLine(
                "/diff: no change tracking for this session — the workspace "
                "could not be tracked with git (auto-init disabled via --no-init-repo, "
                "or repo setup failed), so the agent's edits cannot be diffed"
            ))
            return
        widget = self.query_one("#diff-summary", DiffSummary)

        if not arg:
            # Toggle: fully shown file list (no diff open) → hide the panel.
            if widget.has_class("visible") and widget.has_class("expanded") and not widget.diff_shown:
                await widget.clear()
                return
            await self.refresh_diff_summary()
            if not widget.has_class("visible"):
                self._post_line(NoticeLine("no pending changes to diff"))
                return
            widget.add_class("expanded")
            return

        try:
            text = await git_diff(info, arg)
        except GitWorktreeError as exc:
            widget.hide_diff()
            self._post_line(ErrorLine(f"/diff failed: {exc}"))
            return
        if not text.strip():
            self._post_line(NoticeLine(f"no pending changes for {arg}"))
            return
        # Rebuild the row list first (a previous /diff toggle may have cleared
        # it) so the active-row highlight has something to attach to.
        await self.refresh_diff_summary()
        widget.show_diff(arg, text)

    def _rebind_tracking(self, info: GitInfo) -> None:
        """Attach (possibly resumed) change-tracking state to this session.

        The sandbox already points at the user's real workspace — only the
        git_info wiring (checkpoints, diff summary, git guard) changes."""
        self.git_info = info
        self.controller.git_info = info

    def _persist_git_state(self) -> None:
        if self.session_name:
            from ..session_store import save_session
            save_session(
                self.session_name, self.profile.model_id, self.controller.conversation,
                git=self.git_info.to_block() if self.git_info else None,
                workspace=str(self.workspace) if self.workspace else None,
            )

    def _rewrite_checkpoints(self, sha: str) -> None:
        """Point every turn's checkpoints at ``sha`` (the accepted tree), so
        /undo //retry can no longer revert file state past an accept: both the
        before and after tree become the accepted state, making each turn's
        diff empty."""
        for message in self.controller.conversation.messages:
            if message.role == "user" and message.checkpoint_sha is not None:
                message.checkpoint_sha = sha
                message.after_tree = sha
        if self.controller.conversation.compact_snapshot:
            for message in self.controller.conversation.compact_snapshot:
                if message.role == "user" and message.checkpoint_sha is not None:
                    message.checkpoint_sha = sha
                    message.after_tree = sha

    def _git_action_preflight(self, label: str) -> tuple[GitInfo, bool] | None:
        """Shared /accept //reject guards; mounts an error line if blocked.

        Returns the git info plus whether a turn is still generating — these
        are allowed mid-turn (snapshots use throwaway indexes and the running
        turn's checkpoint is rewritten like any other), the caller just warns
        that the agent may keep editing."""
        info = self.git_info
        if info is None:
            self._post_line(ErrorLine(
                f"/{label}: no change tracking for this session (the workspace "
                "isn't a git repo, or tracking setup failed)"
            ))
            return None
        return info, self._turn_busy or self._compacting

    @work(group="git-action", exclusive=True)
    async def _start_accept(self, info: GitInfo, mid_turn: bool = False) -> None:
        """Lock in the changes made so far: they become the new baseline —
        off /diff's radar and out of /undo's reach. Work continues from
        here, change-per-feature. Allowed mid-turn: the running turn's
        checkpoint is rewritten along with the rest, so its eventual
        after-tree measures from the accepted baseline too. (Guarding
        happens synchronously in the slash-command handler.)"""
        if info is None:
            return
        try:
            stat = await diff_stat(info)
        except GitWorktreeError:
            stat = None
        try:
            sha = await accept_changes(info)
        except GitWorktreeError as exc:
            self._post_line(ErrorLine(f"/accept failed: {exc}"))
            return
        self._rewrite_checkpoints(sha)
        self._persist_git_state()
        await self.refresh_diff_summary()
        if mid_turn:
            self._post_line(NoticeLine(
                "note: the current turn is still generating — it may keep "
                "editing files on top of the accepted state"
            ))
        if stat is None or not stat.files:
            self._post_line(NoticeLine(
                "accepted: no pending changes — baseline reset; /diff and /undo "
                "now measure from here"
            ))
        else:
            self._post_line(NoticeLine(
                f"accepted: {len(stat.files)} file(s) +{stat.additions} -{stat.deletions} "
                "locked in; /diff is reset and /undo can no longer revert them"
            ))

    @work(group="git-action", exclusive=True)
    async def _start_reject(self, info: GitInfo, mid_turn: bool = False) -> None:
        """Discard every change made since the last /accept: the workspace is
        restored to the accepted baseline. The conversation keeps going (the
        agent sees the reverted files on its next turn). Allowed mid-turn —
        the generating turn simply keeps working from the reverted files.
        (Guarding happens synchronously in the slash-command handler.)"""
        if info is None:
            return
        try:
            stat = await diff_stat(info)
            await revert_to_tree(info, info.baseline_tree)
        except GitWorktreeError as exc:
            self._post_line(ErrorLine(f"/reject failed: {exc}"))
            return
        self._rewrite_checkpoints(info.baseline_tree)
        self._persist_git_state()
        await self.refresh_diff_summary()
        if mid_turn:
            self._post_line(NoticeLine(
                "note: the current turn is still generating — it may keep "
                "editing files on top of the reverted state"
            ))
        if not stat.files:
            self._post_line(NoticeLine("rejected: nothing to discard"))
        else:
            self._post_line(NoticeLine(
                f"rejected: {len(stat.files)} file(s) +{stat.additions} -{stat.deletions} "
                "since the last accept were reverted"
            ))

    @work(group="git-action", exclusive=True)
    async def _start_delete(self) -> None:
        """Slash /delete: end the current session for good. Deleting a session
        is pure bookkeeping — the saved session file is removed and a fresh
        session starts; the workspace is never touched (no revert, no accept).
        (Guarding happens synchronously in the slash-command handler; the
        worker assumes it passed.)"""
        deleted = False
        if self.session_name is not None:
            from ..session_store import delete_session
            deleted = delete_session(self.session_name)
        self._start_new_session()
        await self.refresh_diff_summary()
        # Posted after _start_new_session: it wipes the chat log.
        state = "saved session removed" if deleted else "no saved session file"
        self._post_line(NoticeLine(
            f"deleted: {state} — started a fresh session; your files are "
            "untouched"
        ))

    def _on_theme_selected(self, name: str | None) -> None:
        if name:
            self._handle_theme_command(name)

    def _handle_theme_command(self, arg: str) -> None:
        """Apply an explicitly named theme, or report the available ones."""
        if arg not in theme.THEMES:
            self._post_line(ErrorLine(f"unknown theme '{arg}'; available: {', '.join(theme.THEMES)}"))
            return
        self._apply_theme(arg)
        self._save_preference(default_theme=arg)

    def _save_preference(
        self,
        *,
        default_theme: str | None = None,
        default_profile: str | None = None,
        auto_compact: bool | None = None,
        profile: str | None = None,
    ) -> None:
        """Write a chosen theme/model/auto-compact setting back to the config
        file so it becomes the default for the next session. Failures are
        reported, never fatal."""
        if self.config_path is None:
            return
        try:
            save_preferences(
                self.config_path,
                default_theme=default_theme,
                default_profile=default_profile,
                auto_compact=auto_compact,
                profile=profile,
            )
        except ConfigError as exc:
            self._post_line(NoticeLine(f"note: could not save preference: {exc}"))

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
        for thinking in self.query(ThinkingBlock):
            thinking.recolor()
        for block in self.query(ToolCallBlock):
            block.recolor()
        self._refresh_status()
        self.query_one("#diff-summary", DiffSummary).recolor()

    def _start_new_session(self) -> None:
        """Slash /new: fresh conversation, fresh session file (unless disabled)."""
        self.controller.conversation = Conversation()
        self.controller.usage_total = Usage(input_tokens=0, output_tokens=0)
        self.controller._absorbed_usage = Usage(input_tokens=0, output_tokens=0)
        self.controller._absorbed_spent = False
        self.controller.last_compaction = None
        if self.session_name is not None:
            from ..session_store import new_session_name
            self.session_name = new_session_name()
        self.ensure_system_message()
        self._last_thinking = None
        self._queued.clear()
        self._refresh_steer_bar()
        chat = self.query_one("#chat-log", VerticalScroll)
        chat.remove_children()
        # Fresh, empty log: follow it again from the top of the new session.
        chat.anchor()
        self._refresh_status()

    def _restore_session_container(self, block: dict | None) -> None:
        """Reapply a session's saved container settings on resume."""
        if not isinstance(block, dict):
            return
        if self.sandbox is not None:
            self.sandbox.apply_session_settings(block)
        limit = block.get("tool_output_limit")
        if isinstance(limit, int) and limit > 0:
            self._set_output_limit(limit)
        if isinstance(block.get("auto_compact"), bool):
            self.auto_compact = block["auto_compact"]

    async def _on_session_selected(self, name: str | None) -> None:
        if not name:
            return
        from ..session_store import load_state
        chat = self.query_one("#chat-log", VerticalScroll)
        state = load_state(name)
        self.controller.conversation = state.conversation
        self.session_name = name
        if self.workspace is not None:
            from ..session_store import load_workspace_container
            self._restore_session_container(load_workspace_container(str(self.workspace)))
        self.ensure_system_message()
        # Per-message usage is persisted: restore the status-bar totals.
        self._refresh_status()
        # Notices mount after _render_history (which clears the chat log).
        notes: list[str] = []
        errors: list[str] = []
        block = state.git
        if block is not None:
            try:
                info = GitInfo.from_block(block)
            except (KeyError, TypeError, ValueError):
                info = None
            if info is not None:
                self._rebind_tracking(info)
                notes.append(f"reconnected change tracking for {info.agent_workspace}")
            else:
                # Session predates direct-write tracking (old worktree block).
                notes.append(
                    "note: old worktree session; switched to direct edits with fresh tracking"
                )
                await self._setup_tracking_for_resume(name, notes, errors)
        elif self.sandbox is not None:
            # Session predates git tracking. Set up tracking now if possible
            # (same policy as cli.py — including auto-init on a bare directory).
            await self._setup_tracking_for_resume(name, notes, errors)
        # Workspace changes made outside this session (while it was closed)
        # are surfaced, not decided on, after the history is re-rendered.
        drift = await self._detect_resume_drift()
        await self._render_history()
        for note in notes:
            self._post_line(NoticeLine(note))
        for error in errors:
            self._post_line(ErrorLine(error))
        await self.refresh_diff_summary()
        if drift is not None and self.git_info is not None:
            self._prompt_resume_drift(self.git_info, drift[0], drift[1])

    async def _detect_resume_drift(self) -> tuple[str, str] | None:
        """(last_known, current) when the workspace changed outside this
        session since it was last open, else None.

        ``last_known`` is the last turn's after-tree, or the baseline when
        the session has no reversible turns (e.g. an older session saved
        before after-trees were recorded)."""
        info = self.git_info
        if info is None:
            return None
        turns = self.controller.turn_checkpoints()
        last_known = turns[-1].after_tree if turns else info.baseline_tree
        try:
            current = await snapshot_tree(info.workspace, info.subtree)
        except GitWorktreeError:
            return None
        if current == last_known:
            return None
        return last_known, current

    @work(group="resume-drift", exclusive=True)
    async def _prompt_resume_drift(self, info: GitInfo, last_known: str, current: str) -> None:
        """The workspace changed outside this session while it was closed.

        Show what drifted and let the user choose: accept it as the new
        baseline, or keep tracking against the last known state. Never pick
        for them."""
        try:
            stat = await tree_diff_stat(info, last_known, current)
        except GitWorktreeError:
            return
        detail = _format_drift_detail(stat)
        choice = await self.push_screen_wait(ConfirmScreen(
            "Workspace changed outside this session since it was last open:",
            confirm_label="Accept as baseline",
            cancel_label="Keep tracking",
            detail=detail,
        ))
        if not choice:
            self._post_line(NoticeLine(
                "keeping tracking against the last known state — the drift "
                "shows up in /diff"
            ))
            return
        try:
            await accept_changes(info)
        except GitWorktreeError as exc:
            self._post_line(ErrorLine(f"could not accept the drift: {exc}"))
            return
        self._rewrite_checkpoints(info.baseline_tree)
        self._persist_git_state()
        self._post_line(NoticeLine(
            "drift accepted: the workspace is the new baseline; /diff and "
            "/undo now measure from here"
        ))
        await self.refresh_diff_summary()

    async def _setup_tracking_for_resume(
        self, name: str, notes: list[str], errors: list[str]
    ) -> None:
        """Set up fresh change tracking for a resumed session that has none
        (legacy worktree block or pre-tracking session). The sandbox already
        points at the real workspace."""
        from .. import gitwork
        try:
            info, setup_notes = await gitwork.setup_tracking(
                self.workspace or Path.cwd(), name
            )
        except GitWorktreeError as exc:
            errors.append(f"git change tracking unavailable: {exc}")
            return
        if info is not None:
            self._rebind_tracking(info)
            notes.extend(setup_notes)

    def _on_model_selected(self, name: str | None) -> None:
        if not name:
            return
        profile = self._profiles_cache.get(name)
        if profile is None:
            return
        try:
            api_key = resolve_api_key(profile)
        except ConfigError as exc:
            self._post_line(ErrorLine(f"could not switch model: {exc}"))
            return
        # Swap the provider underneath the existing controller: client,
        # conversation, and tools all carry over.
        self.controller.switch_profile(profile, api_key)
        self.profile = profile
        self.profile_name = name
        self._refresh_status()
        self._save_preference(default_profile=name)

    async def _mount_spaced(
        self,
        chat: "VerticalScroll",
        widget: "ThinkingBlock | AssistantMessage",
    ) -> None:
        """Mount a cot/assistant block, keeping a blank line above it when it
        directly follows tool call blocks (which carry no spacing of their
        own)."""
        children = chat.children
        if children and isinstance(children[-1], ToolCallBlock):
            widget.add_class("after-tools")
        await chat.mount(widget)

    async def _render_history(self) -> None:
        """Rebuild #chat-log from the conversation.

        Used by /resume. Best-effort reconstruction: tool-call status is
        inferred from whether the recorded tool-role content starts with
        "error: " (see ChatController.record_tool_result), since the
        original ToolCallStatus enum value isn't itself persisted.

        If the conversation was compacted, the pre-compaction messages are
        rendered first (they are history, kept for the human), then a
        divider, then the summary and everything after it — mirroring what
        the model actually sees.
        """
        chat = self.query_one("#chat-log", VerticalScroll)
        await chat.remove_children()
        conversation = self.controller.conversation
        messages = conversation.messages
        pre_compact = conversation.compact_snapshot or []
        all_messages = [*pre_compact, *messages]
        tool_results = {
            m.tool_call_id: m for m in all_messages if m.role == "tool" and m.tool_call_id
        }

        for message in pre_compact:
            await self._render_message(chat, message, tool_results)
        if pre_compact:
            await chat.mount(NoticeLine(COMPACTED_DIVIDER))
        for message in messages:
            await self._render_message(chat, message, tool_results)
        chat.scroll_end(animate=False)

    async def _render_message(
        self,
        chat: "VerticalScroll",
        message: Message,
        tool_results: dict[str, Message],
    ) -> None:
        """Render one persisted message into the chat log."""
        if message.role == "user":
            if message.content.startswith(SUMMARY_PREFIX):
                # The handoff an auto-compaction left behind: padded and
                # accent-colored, not a plain user message.
                await chat.mount(CompactionSummary(
                    message.content.removeprefix(SUMMARY_PREFIX).strip()
                ))
            else:
                widget = UserMessage(message.content, message.images)
                widget.message = message
                await chat.mount(widget)
        elif message.role == "assistant":
            if message.reasoning:
                thinking = ThinkingBlock()
                await self._mount_spaced(chat, thinking)
                thinking.append_reasoning(message.reasoning)
                thinking.finish(message.reasoning_seconds, estimate_if_unknown=False)
            # Tool-call-only rounds render no AssistantMessage body; an
            # empty assistant message with no tool calls replays as
            # "(no output)", matching the live-stream finalize path.
            if message.content or not message.tool_calls:
                assistant = AssistantMessage()
                await self._mount_spaced(chat, assistant)
                if message.content:
                    await assistant.append_delta(message.content)
                await assistant.finalize()
            for call in message.tool_calls or []:
                block = ToolCallBlock(call.call_id, call.name)
                await chat.mount(block)
                block.append_arguments(call.arguments_json)
                result = tool_results.get(call.call_id)
                if result is not None:
                    is_error = result.content.startswith("error: ")
                    status = ToolCallStatus.CALL_FAILED if is_error else ToolCallStatus.CALL_SUCCEEDED
                    block.set_result(
                        status,
                        output="" if is_error else result.content,
                        error=result.content[len("error: "):] if is_error else "",
                        header=result.header or "",
                    )

    def on_chat_input_mode_toggle(self, event: ChatInput.ModeToggle) -> None:
        self._switch_mode()

    def on_chat_input_slash_query(self, event: ChatInput.SlashQuery) -> None:
        q = event.query.lower()
        # Table-style rows: commands padded to a shared column, descriptions
        # dimmed — same layout language as the session list.
        width = max(len(name) for name, _ in SLASH_COMMANDS)
        matches = [
            (
                name,
                f"/{name:<{width}}  [dim]{desc}[/]",
            )
            for name, desc in SLASH_COMMANDS
            if name.startswith(q)
        ]
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

    def _post_line(self, widget: Static) -> None:
        """Mount a notice/error line into the chat log.

        Whether it lands in view is the chat log's business: it is anchored,
        so it follows the new line only if the user is already at the bottom.
        """
        try:
            chat = self.query_one("#chat-log", VerticalScroll)
        except NoMatches:
            return  # app shutting down; DOM already pruned
        chat.mount(widget)

    def _refresh_steer_bar(self) -> None:
        """Mirror the queued steer messages into the bar above the input.
        They move into the chat log (as UserMessages) when actually sent."""
        try:
            bar = self.query_one("#steer-queue-bar", SteerQueueBar)
        except NoMatches:
            return  # app shutting down
        bar.update_items([text for text, _ in self._queued])

    def _extract_images(self, text: str) -> list[ImageAttachment]:
        """Load every ``@path`` token in ``text`` that names an image file.

        The token itself stays in the message text (the model reads
        ``@file.png`` literally); the image rides along as an attachment.
        Duplicates resolve once; unreadable files are reported and skipped.
        """
        images: list[ImageAttachment] = []
        seen: set[str] = set()
        root = self.workspace or Path.cwd()
        for match in _IMAGE_TOKEN_RE.finditer(text):
            raw = match.group(1)
            path = image_loader.parse_image_path(raw)
            if path is None or not image_loader.is_image_path(path):
                continue
            if not path.is_absolute():
                path = root / path
            try:
                resolved = str(path.resolve())
            except OSError:
                continue
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            try:
                images.append(image_loader.load_image(path))
            except ImageError as exc:
                self._post_line(ErrorLine(str(exc)))
        return images

    def _check_vision(self, images: list[ImageAttachment]) -> bool:
        """True when the current profile can take images (or there are none)."""
        if not images or self.controller.profile.supports_vision:
            return True
        self._post_line(ErrorLine(
            f"{self.controller.profile.display_name} is not configured with "
            "supports_vision: true — image attachments were not sent"
        ))
        return False

    def _submit(self, text: str, images: list[ImageAttachment] | None = None) -> None:
        chat = self.query_one("#chat-log", VerticalScroll)
        # Editor tokens come pre-ordered from the input; @-mentioned paths
        # typed manually are scanned out of the text and appended.
        all_images = list(images or [])
        for scanned in self._extract_images(text):
            if scanned not in all_images:
                all_images.append(scanned)
        if not self._check_vision(all_images):
            return
        if self._turn_busy or self._compacting:
            # Queued: shown in the steer bar above the input until sent.
            self._queued.append((text, all_images or None))
            self._refresh_steer_bar()
            return
        user_message = UserMessage(text, all_images or None)
        chat.mount(user_message)
        self._worker = self._run_turn(text, user_message, all_images or None)

    def _flush_queued_now(self) -> None:
        """Enter on an empty input while messages are queued: send now.

        Interrupts the running turn; the worker's cleanup pops the oldest
        queued message and starts it as a turn immediately, instead of
        waiting for the next round boundary.
        """
        if self._turn_busy and self._queued and self._worker is not None:
            self._worker.cancel()

    def _copy_selection(self) -> bool:
        """Copy the active selection (TextArea or screen); True if copied."""
        focused = self.focused
        if isinstance(focused, TextArea) and focused.selected_text:
            self.copy_to_clipboard(focused.selected_text)
            return True
        selected = self.screen.get_selected_text()
        if selected:
            self.copy_to_clipboard(selected)
            return True
        return False

    async def action_copy_or_quit(self) -> None:
        """ctrl+c: copy the active selection if there is one, quit otherwise.

        Quitting is immediate, but an in-flight turn is cancelled and given a
        moment to unwind first — otherwise the worker keeps running against a
        closed app and sprays cancellation errors on the way out.
        """
        if self._copy_selection():
            return
        self._disarm_interrupt()
        worker = self._worker if self._turn_busy else None
        if worker is not None:
            worker.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(worker), timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        self.exit()

    def action_copy_selection(self) -> None:
        """ctrl+shift+c: copy the active selection, if any (never quits)."""
        self._copy_selection()

    def action_interrupt(self) -> None:
        # The priority escape binding shadows the modals' own cancel bindings,
        # so cancel the open modal here rather than interrupting the turn.
        if isinstance(self.screen, ModalScreen):
            self.screen.dismiss(None)
            return
        chat_input = self.query_one("#chat-input", ChatInput)
        popup = self.query_one("#suggestion-popup", SuggestionPopup)
        if chat_input.popup_active is not None or popup.has_class("visible"):
            chat_input.popup_active = None
            self._cancel_at_search()
            self._hide_popup()
            return
        if self._ask_future is not None and not self._ask_future.done():
            # Esc while a question is pending: skip the questions, keep the
            # turn running (the ask tool reports the skip to the model).
            self._ask_future.set_result(None)
            return
        diff_summary = self.query_one("#diff-summary", DiffSummary)
        if diff_summary.diff_shown:
            # First esc: minimize the open diff, keeping the file list up and
            # its last-viewed row highlighted for arrow-key navigation.
            diff_summary.minimize_diff()
            return
        if self.focused is diff_summary and diff_summary.has_class("expanded"):
            # Second esc: close the file list too; only the summary remains.
            diff_summary.collapse()
            return
        if self._turn_busy and self._worker is not None:
            if self._interrupt_armed:
                # Second esc within the window: actually stop the turn.
                self._disarm_interrupt()
                self._worker.cancel()
            else:
                # First esc only arms the interrupt, so a stray keypress can't
                # kill a turn mid-flight.
                self._interrupt_armed = True
                self._post_line(
                    MessageLine("press esc again to interrupt", kind="warn", hint=True)
                )
                self._interrupt_timer = self.set_timer(
                    INTERRUPT_ARM_SECONDS, self._disarm_interrupt
                )

    def _disarm_interrupt(self) -> None:
        self._interrupt_armed = False
        if self._interrupt_timer is not None:
            self._interrupt_timer.stop()
            self._interrupt_timer = None

    async def _ask_user(self, questions: list[str]) -> list[str] | None:
        """Callback for the ask tool: show the questions one at a time in the
        AskBar above the input; the user answers through the normal chat
        input (Enter submits, Esc skips all questions).

        Runs inside the turn worker, awaiting a future that the input
        handler resolves — same event loop, so the await is safe.
        """
        ask_bar = self.query_one("#ask-bar", AskBar)
        chat_input = self.query_one("#chat-input", ChatInput)
        answers: list[str] = []
        try:
            for index, question in enumerate(questions):
                ask_bar.show_question(f"Question {index + 1}/{len(questions)}: {question}")
                chat_input.placeholder = ASK_PLACEHOLDER
                chat_input.focus()
                future: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
                self._ask_future = future
                answer = await future
                if answer is None:  # esc: skip the remaining questions
                    return None
                answers.append(answer)
            return answers
        finally:
            self._ask_future = None
            ask_bar.hide()
            chat_input.placeholder = INPUT_PLACEHOLDER

    @work(group="turn")
    async def _run_turn(
        self,
        text: str,
        user_widget: UserMessage | None = None,
        images: list[ImageAttachment] | None = None,
    ) -> None:
        """Drive the full exchange: rounds of (thinking → answer → tool calls).

        Each round streams one model response; its widgets are owned by a
        :class:`_RoundView`. If the round requested tool calls, they're
        executed, their blocks settle, results are recorded into the
        conversation, and the next round streams — until a round comes back
        with no tool calls.
        """
        self._turn_busy = True
        chat = self.query_one("#chat-log", VerticalScroll)
        gen_bar = self.query_one("#generating-bar", GeneratingBar)
        gen_bar.add_class("active")
        turn_start = time.monotonic()
        turn_had_tools = False
        view = _RoundView(self, chat)

        stream = self.controller.send(text, images)
        try:
            while True:
                view = _RoundView(self, chat)
                await view.mount()

                async for event in stream:
                    if user_widget is not None and user_widget.message is None:
                        # Bind the widget to its conversation message so
                        # click-to-undo can locate this turn. The user message
                        # is the last one added when the round starts.
                        user_widget.message = next(
                            (m for m in reversed(self.controller.conversation.messages)
                             if m.role == "user"),
                            None,
                        )
                    await view.handle(event)

                await view.finish()

                # A round that ended without finishing — a provider or
                # transport error, say — still streamed something. Keep it
                # (up to the last non-thinking block) instead of dropping the
                # partial answer; a completed round has nothing partial left,
                # so this is a no-op then.
                if self.controller.save_interrupted_round():
                    self._persist_git_state()

                if not view.round_has_tools:
                    if not view.had_stream_error:
                        worked = _format_duration(time.monotonic() - turn_start)
                        self._post_line(NoticeLine(f"Worked for {worked}"))
                    break

                # Execute the calls this round requested, in stream order.
                turn_had_tools = True
                round_wrote_files = False
                for call_id, block in view.tool_blocks.items():
                    result = await self.tool_registry.call(
                        block.tool_name,
                        block.accumulated_arguments,
                        output_sink=block.append_output,
                    )
                    if result.parse_error:
                        status = ToolCallStatus.PARSING_ERROR
                    elif result.ok:
                        status = ToolCallStatus.CALL_SUCCEEDED
                    else:
                        status = ToolCallStatus.CALL_FAILED
                    block.set_result(
                        status, output=result.output, error=result.error, header=result.header
                    )
                    self.controller.record_tool_result(call_id, result)
                    if result.ok and block.tool_name in _WRITING_TOOLS:
                        round_wrote_files = True
                if round_wrote_files and self.git_info is not None:
                    # The model just changed a file: surface the diff panel
                    # (header only — the file list stays collapsed) above the
                    # input without waiting for the turn to end.
                    await self.refresh_diff_summary()

                # Steer: anything typed while this round was streaming is
                # injected here — after the tool answers, before the next
                # LLM call — so the model sees it right away instead of the
                # queued message waiting for the whole turn to end.
                if self._queued:
                    for queued_text, queued_images in self._queued:
                        self.controller.inject_user_message(queued_text, queued_images)
                    injected = self.controller.conversation.messages[-len(self._queued):]
                    chat = self.query_one("#chat-log", VerticalScroll)
                    for (queued_text, queued_images), message in zip(self._queued, injected):
                        # Land the message in the transcript only now — after
                        # the round's answer/tool calls it follows.
                        widget = UserMessage(queued_text, queued_images)
                        widget.message = message
                        widget.mark_sent()
                        chat.mount(widget)
                    self._queued.clear()
                    self._refresh_steer_bar()

                stream = self.controller.continue_after_tools()
            if not view.had_stream_error:
                # Auto-compact fires at the turn boundary: the history has no
                # pending tool calls to break pairing.
                await self._maybe_auto_compact()
        except asyncio.CancelledError:
            # `view` is the round in flight (reassigned at each loop top).
            await view.cancel()
            # Keep whatever the model had produced before the interruption —
            # everything up to the last non-thinking block — so the next turn
            # and a later /resume still see it. An empty or reasoning-only
            # round saves nothing, leaving the history untouched.
            if self.controller.save_interrupted_round():
                self._persist_git_state()
            self._post_line(ErrorLine("interrupted"))
        finally:
            # Record where the turn left the workspace so /undo //retry can
            # reverse exactly this turn's diff later (interruptions included),
            # and persist it for a later /resume.
            await self.controller.finish_turn()
            self._persist_git_state()
            gen_bar.remove_class("active")
            self._turn_busy = False
            self._worker = None
            # Cheapest correct trigger for the diff summary: after any turn
            # that ran tools, recompute once — not per keystroke or per round.
            if turn_had_tools and self.git_info is not None:
                await self.refresh_diff_summary()
            self._disarm_interrupt()
            if self._queued:
                queued_text, queued_images = self._queued.pop(0)
                widget = UserMessage(queued_text, queued_images)
                self.query_one("#chat-log", VerticalScroll).mount(widget)
                self._refresh_steer_bar()
                self._worker = self._run_turn(queued_text, widget, queued_images)

    async def _dismiss_indicator(self, indicator: PendingIndicator | None) -> None:
        if indicator is not None and indicator.is_mounted:
            await indicator.remove()

    async def _run_compaction(self) -> str:
        """Compact via the controller and refresh everything that depends on
        it. Returns the summary, prefixed with the token-count line
        ("compacted: N → M tokens (freed K)")."""
        chat = self.query_one("#chat-log", VerticalScroll)
        self._compacting = True
        # A shimmering "Compacting" line while the summarizer runs, so the
        # pause reads as work rather than a hang.
        indicator = PendingIndicator(
            "Compacting", theme.SHIMMER_COMPACTING, hint=""
        )
        await chat.mount(indicator)
        try:
            summary = await self.controller.compact()
        finally:
            if indicator.is_mounted:
                await indicator.remove()
            self._compacting = False
        self._refresh_status()
        # Mark the compaction point in the chat log — same divider /resume
        # renders between the kept history and the summary — and show the
        # handoff itself, rendered like the replayed one below it.
        self._post_line(NoticeLine(COMPACTED_DIVIDER))
        for message in self.controller.conversation.messages:
            if message.role == "user" and message.content.startswith(SUMMARY_PREFIX):
                await chat.mount(CompactionSummary(
                    message.content.removeprefix(SUMMARY_PREFIX).strip()
                ))
                break
        counts = self._compaction_counts()
        return f"compacted: {counts}\n{summary}" if counts else summary

    def _compaction_counts(self) -> str | None:
        '"N → M tokens (freed K)" for the last compaction, or None.'
        counts = self.controller.last_compaction
        if counts is None:
            return None
        before, after = counts
        return f"{before:,} → {after:,} tokens (freed {max(0, before - after):,})"

    def _post_compaction_notice(self, label: str = "auto-compacted") -> None:
        counts = self._compaction_counts()
        if counts is None:
            self._post_line(NoticeLine(f"{label}: nothing to compact yet"))
        else:
            self._post_line(NoticeLine(f"{label}: {counts}"))

    async def _maybe_auto_compact(self) -> None:
        """End-of-turn hook for /container's auto-compact: compact at the
        profile's auto_compact_threshold of the context window.

        Runs only at a turn boundary, so there are no pending tool calls to
        break pairing. The context estimate matches the status bar's
        (last round's prompt + cumulative output)."""
        if not self.auto_compact:
            return
        if self.last_in + self.last_out < self.profile.auto_compact_threshold * self.profile.max_context:
            return
        try:
            await self._run_compaction()
        except RuntimeError:
            return
        self._post_compaction_notice("auto-compacted")

    def _clear_hint_lines(self) -> None:
        """Remove transient hint lines (e.g. 'press esc again to interrupt')
        from the chat log. Called when the assistant's message settles, so
        keypress coaching doesn't linger in the scrollback."""
        try:
            chat = self.query_one("#chat-log", VerticalScroll)
        except NoMatches:
            return  # app shutting down; DOM already pruned
        for line in chat.query(".msg.hint"):
            line.remove()

    async def _complete_round(
        self,
        event: TurnComplete,
        thinking: ThinkingBlock | None,
        assistant: AssistantMessage | None,
    ) -> None:
        chat = self.query_one("#chat-log", VerticalScroll)
        self._clear_hint_lines()
        if assistant is not None:
            await assistant.finalize()
        elif thinking is None and not event.has_tool_calls:
            await chat.mount(ErrorLine("empty response from model"))
        if thinking is not None:
            thinking.finish(event.reasoning_seconds)

        self._refresh_status()

        if self.session_name:
            from ..session_store import save_session
            save_session(
                self.session_name, self.profile.model_id, self.controller.conversation,
                git=self.git_info.to_block() if self.git_info else None,
                workspace=str(self.workspace) if self.workspace else None,
            )

    def action_toggle_thoughts(self) -> None:
        if self._last_thinking is not None:
            self._last_thinking.toggle()
