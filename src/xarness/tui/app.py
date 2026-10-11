"""The xarness chat application — a thin client of the host API.

Owns the widget tree and renders the event stream of one headless
:class:`~xarness.session.Session` running in the host process. All state
and logic — the turn loop, tool execution, steering, undo/retry,
accept/reject, compaction, persistence — live in the session; this process
only renders events and sends commands. It never imports the controller,
session, tools, gitwork or sandbox modules.
"""

from __future__ import annotations

import asyncio
import httpx
from pathlib import Path
from typing import ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Horizontal, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static, TextArea
from textual.worker import Worker

from .. import theme
from ..clientlib import ClientError, XarnessClient
from ..config import DEFAULT_CONFIG_PATH, save_preferences
from ..conversation import SUMMARY_PREFIX, Message
from ..events import ToolCallStatus
from ..images import ImageAttachment
from ..protocol import (
    AnswerAsk, AskRequested, AskResolved, Cancel, Command, ContentDelta,
    DiscardQueued, DriftResolved, Event, FilterAdd, FilterRemove,
    FlushQueued, HistoryReplaced, ImageAttachmentModel, MessageModel,
    Notice, ReasoningDelta, Reject, ResolveDrift,
    Retry, SendMessage, SessionDeleted, SetMode, SetSandbox, Snapshot,
    StatusChanged, StatusPayload, StreamError, Accept, SwitchModel,
    ToolCallArgumentsDelta, ToolCallArgumentsDone, ToolCallStarted,
    ToolExecStarted, ToolFinished, ToolOutputDelta, TurnComplete,
    TurnFinished, Undo, UndoTo, UserMessageAdded,
)
from .confirm_screen import ConfirmScreen
from .picker_screen import PickerScreen
from .resume_screen import ResumeScreen
from .widgets import (
    AskBar, AssistantMessage, ChatInput, CompactionSummary, DiffSummary,
    ErrorLine, GeneratingBar,
    MessageLine, NoticeLine, PendingIndicator, ShimmerText, StatusBar,
    crop_path,
    SteerQueueBar, SuggestionPopup, ThinkingBlock, ToolCallBlock,
    ToolWritingIndicator, UserMessage, _format_duration,
)


# Max characters of the workspace path shown in the terminal title —
# matches the status bar's crop_path default.
_TITLE_PATH_LEN = 16

INPUT_PLACEHOLDER = (
    "Send a message…  (Enter: send · Shift+Enter: newline · Tab: mode · Ctrl+T: thoughts)"
)
ASK_PLACEHOLDER = "Type your answer…  (Enter: send · Esc: skip)"

# How long a first Esc stays "armed" waiting for the confirming second one.
INTERRUPT_ARM_SECONDS = 2.5

# How long to wait for a cancel command to unwind before exiting anyway.
CANCEL_TIMEOUT = 2.0

# Divider mounted in the chat log at every compaction point (live and on
# /resume): everything above is history the model no longer sees verbatim.
COMPACTED_DIVIDER = (
    "— compacted: the model sees only the summary below; "
    "messages above are kept here for you —"
)


class ContinueHeadlessScreen(ModalScreen[bool | None]):
    """Ctrl+C while a request is still running: continue it in the
    background in the host, or cancel it and quit.

    ``True`` = continue headless, ``False`` = stop and quit, ``None``
    (escape) = stay in the app. Styled entirely by the existing modal CSS
    (same conventions as ConfirmScreen — no new visual style)."""

    BINDINGS = [
        ("escape", "stay", "Stay"),
        ("y", "continue_headless", "Continue headless"),
        ("n", "stop_quit", "Stop and quit"),
    ]

    def compose(self) -> ComposeResult:
        from textual.containers import Vertical
        from textual.widgets import Button, Label
        with Vertical(id="confirm-modal"):
            yield Label(
                "A request is still running. Continue it in the background?",
                id="confirm-title", markup=False,
            )
            with Horizontal(id="confirm-buttons"):
                yield Button("Continue headless", id="confirm-yes", variant="primary")
                yield Button("Stop and quit", id="confirm-no", variant="error")

    def on_button_pressed(self, event) -> None:
        self.dismiss(event.button.id == "confirm-yes")

    def action_continue_headless(self) -> None:
        self.dismiss(True)

    def action_stop_quit(self) -> None:
        self.dismiss(False)

    def action_stay(self) -> None:
        self.dismiss(None)


class _RoundView:
    """Widget state for one streaming round: pending indicator, thinking
    block, assistant message, tool-writing shimmer, tool-call blocks. Fed
    by the event pump; the session drives pacing."""

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
        # Set by cancel(): indicators mounted afterwards (a mount cancelled
        # mid-flight still lands its widget) remove themselves on_mount.
        self.cancelled = False

    async def mount(self) -> None:
        self.indicator = PendingIndicator(cancelled=lambda: self.cancelled)
        self.indicator_live = True
        await self.chat.mount(self.indicator)

    async def handle(self, event: Event) -> None:
        if isinstance(event, ReasoningDelta):
            if self.thinking is not None and self.thinking.done:
                # A new round of the same turn (previous round ended with
                # tool calls): the finished thinking block and settled
                # answer belong to the transcript — start fresh widgets, or
                # round 2+ CoT and answers stream into finished (collapsed,
                # dropped-text) widgets and stay invisible until reload.
                await self._new_round()
            if self.thinking is None:
                await self._hide_indicator()
                self.thinking = ThinkingBlock()
                self.app._last_thinking = self.thinking
                await self.app._mount_spaced(self.chat, self.thinking)
            self.thinking.append_reasoning(event.text)
        elif isinstance(event, ContentDelta):
            if self.assistant is not None and self.assistant.finalized:
                await self._new_round()
            if self.assistant is None:
                await self._hide_indicator()
                if self.thinking is not None:
                    self.thinking.finish(duration=None, estimate_if_unknown=False)
                self.assistant = AssistantMessage()
                await self.app._mount_spaced(self.chat, self.assistant)
            await self.assistant.append_delta(event.text)
        elif isinstance(event, ToolCallStarted):
            await self._hide_indicator()
            if self.thinking is not None:
                self.thinking.finish(duration=None, estimate_if_unknown=False)
            # One shared "Writing <tool>" shimmer while arguments stream; the
            # indicator accumulates the streamed args itself. Per-call blocks
            # appear once args are complete.
            self.pending_writes += 1
            if self.writing is None:
                self.writing = ToolWritingIndicator(
                    event.name, cancelled=lambda: self.cancelled
                )
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
        elif isinstance(event, ToolExecStarted):
            pass  # the block's amber "Running" dot is already showing
        elif isinstance(event, ToolOutputDelta):
            block = self.tool_blocks.get(event.call_id)
            if block is not None:
                block.append_output(event.text)
        elif isinstance(event, ToolFinished):
            # The ToolCallStatus mapping lives here (client-side): the
            # session reports the outcome, the client picks the dot color.
            if event.parse_error:
                status = ToolCallStatus.PARSING_ERROR
            elif event.ok:
                status = ToolCallStatus.CALL_SUCCEEDED
            else:
                status = ToolCallStatus.CALL_FAILED
            block = self.tool_blocks.get(event.call_id)
            if block is not None:
                block.set_result(
                    status, output=event.output, error=event.error, header=event.header,
                )
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

    async def _new_round(self) -> None:
        """Reset the per-round widgets for the turn's next round (the tool
        blocks of earlier rounds stay put — their results may still land)."""
        self.thinking = None
        self.assistant = None
        self.round_has_tools = False
        self.had_stream_error = False

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
        # Flip first: any indicator whose mount was cancelled mid-flight (and
        # so still lands after this method returns) removes itself on_mount.
        self.cancelled = True
        if self.writing is not None and self.writing.is_mounted:
            await self.writing.remove()
        if self.indicator is not None and self.indicator_live:
            await self.app._dismiss_indicator(self.indicator)
        if self.thinking is not None and not self.thinking.done:
            self.thinking.finish(duration=None, estimate_if_unknown=False)
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
    ("sandbox", "choose this session's sandbox profile (idle only)"),
    ("filter", "add/remove/list paths excluded from the sandbox mounts"),
    ("theme", "choose a color theme"),
    ("new", "leave this session and start a fresh chat"),
    ("delete", "delete the current session and start a new one (your files are left exactly as they are)"),
    ("diff", "show pending changes (optionally: /diff <path>)"),
    ("accept", "lock in the changes made so far (they stop showing in /diff and can no longer be undone)"),
    ("reject", "discard all changes made since the last /accept"),
    ("undo", "drop the last turn and revert its file edits"),
    ("retry", "revert the last turn's file edits and resend your message"),
]


class ChatLog(VerticalScroll):
    """The transcript scroll.

    The scrollbar is hidden app-wide, so Textual's only built-in anchor
    release (grabbing the scrollbar) never fires: without this, the log
    stays pinned to the bottom forever and every relayout — a streaming
    flush, expanding a block — yanks the view back down while you're
    reading, which reads as scroll/expand lag. Release the anchor as soon
    as the user scrolls up; scrolling back to the bottom re-arms it."""

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        if new_value < old_value and self._anchored and not self._anchor_released:
            self.release_anchor()


class AgentApp(App[None]):
    CSS_PATH = "styles.tcss"
    TITLE = "xarness"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+t", "toggle_thoughts", "Thoughts", priority=True),
        Binding("ctrl+c", "copy_or_quit", "Copy / Quit", priority=True),
        Binding("ctrl+shift+c", "copy_selection", "Copy", priority=True),
        Binding("escape", "interrupt", "Interrupt", priority=True),
        Binding("ctrl+shift+backspace", "steer_discard", "Discard queued", priority=True),
    ]

    def __init__(
        self,
        client: XarnessClient,
        workspace: Path | None = None,
        session_name: str | None = None,
        config_path: Path | None = DEFAULT_CONFIG_PATH,
        model: str | None = None,
        sandbox: str | None = None,
        preset: str | None = None,
        network: bool | None = None,
        gpu: bool | None = None,
        no_init_repo: bool = False,
        no_session: bool = False,
        startup_notices: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.client = client
        self.workspace = workspace
        self.session_name = session_name
        self.config_path = config_path
        self.model = model
        self.sandbox_name = sandbox
        self.preset = preset
        # Tri-state sandbox overrides (None = not given).
        self.network = network
        self.gpu = gpu
        self.no_init_repo = no_init_repo
        self.no_session = no_session
        self.startup_notices = startup_notices or []
        # Latest status from the session (drives the bars); a draft has none.
        self.status: StatusPayload | None = None
        self._view: _RoundView | None = None
        # Message ids currently rendered in the chat log (see
        # _mount_user_message / _on_history_replaced: the trim path).
        self._rendered_ids: set[str] = set()
        self._pump_worker: Worker | None = None
        self._ask_future: asyncio.Future[str | None] | None = None
        self._ask_id: str | None = None
        self._ask_worker: Worker | None = None
        self._drift_modal_open = False
        self._last_thinking: ThinkingBlock | None = None
        # Double-Esc interrupt: the first Esc arms it (and shows a prompt),
        # the second one within INTERRUPT_ARM_SECONDS actually cancels.
        self._interrupt_armed = False
        self._interrupt_timer = None
        self._at_search_timer = None
        self._config_cache = None
        # Draft-chat state (before any session exists): the mode Tab toggles
        # locally and is applied to the session once it is created.
        self._draft_mode = "write"

    # ------------------------------------------------------------------
    # layout

    def get_css_variables(self) -> dict[str, str]:
        return {**super().get_css_variables(), **theme.CSS_VARIABLES}

    def compose(self) -> ComposeResult:
        yield ChatLog(id="chat-log")
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
        self._set_default_title()
        # Follow new output only while the user is parked at the bottom.
        # Textual's anchor does exactly that: the log stays pinned to the end as
        # content streams, the anchor is released the moment the user scrolls up
        # (so nothing yanks the viewport back), and it re-arms once they scroll
        # down to the bottom again.
        self.query_one("#chat-log", ChatLog).anchor()
        self._load_draft_defaults()
        for notice in self.startup_notices:
            self._post_line(NoticeLine(notice))
        if self.session_name is not None:
            await self._attach(self.session_name)

    async def on_unmount(self) -> None:
        await self.client.aclose()

    # ------------------------------------------------------------------
    # Attach + event pump

    async def _attach(self, name: str) -> None:
        """Attach to a hosted session: snapshot first, then live events."""
        if self._pump_worker is not None:
            self._pump_worker.cancel()
            self._pump_worker = None
        self.session_name = name
        try:
            snap = await self.client.snapshot(name)
        except (ClientError, httpx.HTTPError) as exc:
            self._post_line(ErrorLine(f"could not attach to {name}: {exc}"))
            return
        self.status = snap.status
        self._refresh_title()
        await self._render_history(snap)
        self._refresh_status()
        self._refresh_steer_bar()
        if snap.status.finished_while_away:
            self._post_line(NoticeLine("finished while you were away"))
        await self.refresh_diff_summary()
        # Seed the pump with the seq we just rendered: the host replays only
        # the missed events instead of sending a snapshot, so the history is
        # not rendered a second time on attach.
        self._pump_worker = self._pump_events(name, snap.seq)

    @work(group="stream", exit_on_error=False)
    async def _pump_events(self, name: str, last_seq: int | None = None) -> None:
        """Consume the session's event stream, reconnecting with backoff.

        The clientlib reader replays with Last-Event-ID across dropped
        connections; a fresh Snapshot arrives when the replay window was
        missed. Never crashes the app on connection loss."""
        try:
            async for item in self.client.attach(name, last_seq):
                if isinstance(item, Snapshot):
                    self.status = item.status
                    await self._render_history(item)
                    self._view = None
                else:
                    await self._handle_event(item)
        except asyncio.CancelledError:
            raise
        except (ClientError, httpx.HTTPError) as exc:
            self._view = None
            # Recover outside this worker: _attach cancels the pump worker,
            # and a task cannot reliably cancel and replace itself.
            self.call_later(self._recover, name, exc)
        except Exception as exc:  # noqa: BLE001 - never crash on connection loss
            self._view = None
            self.call_later(self._recover, name, exc)

    async def _recover(self, name: str, exc: Exception) -> None:
        """Connection loss: respawn the host if it died, reload the session
        on the (fresh) host, and reattach. One attempt: a failure is
        reported and the user reattaches via /sessions."""
        self._post_line(ErrorLine(f"connection lost: {exc} — reconnecting"))
        try:
            from ..clientlib import ensure_host
            await ensure_host()
            try:
                await self.client.load_session(name)
            except ClientError:
                pass  # the fresh host may already host it (load is idempotent)
            await self._attach(name)
        except ClientError as recover_exc:
            self._post_line(ErrorLine(
                f"could not reconnect: {recover_exc} — /sessions to reattach",
            ))

    async def _handle_event(self, event: Event) -> None:
        """Route one live event to the widgets."""
        if isinstance(event, StatusChanged):
            was_streaming = self.status is not None and self.status.state in (
                "running", "compacting",
            )
            self.status = event.status
            self._refresh_status()
            self._refresh_steer_bar()
            self._refresh_generating()
            await self._maybe_prompt_drift()
            if (
                was_streaming and self.status.state not in ("running", "compacting")
                and self._view is not None
            ):
                # The turn ended without a TurnComplete for this client —
                # an interrupt (esc / stop-and-quit) or a forced steer
                # cancelling it. Settle the live round, or the Thinking
                # shimmer (and a half-streamed tool's writing shimmer)
                # keeps animating next to the "interrupted" error line.
                await self._view.cancel()
                self._view = None
            return
        if isinstance(event, Notice):
            if event.level == "error":
                self._post_line(ErrorLine(event.text))
            else:
                self._post_line(NoticeLine(event.text))
            return
        if isinstance(event, UserMessageAdded):
            await self._mount_user_message(event)
            return
        if isinstance(event, HistoryReplaced):
            await self._on_history_replaced(event)
            return
        if isinstance(event, AskRequested):
            self._ask_worker = self._ask_flow(event.ask_id, event.questions)
            return
        if isinstance(event, AskResolved):
            self._on_ask_resolved(event)
            return
        if isinstance(event, DriftResolved):
            self._drift_modal_open = False
            return
        if isinstance(event, TurnFinished):
            self._post_line(NoticeLine(f"Worked for {_format_duration(event.elapsed)}"))
            self._refresh_diff_summary_soon()
            return
        if isinstance(event, SessionDeleted):
            # The session went away (deleted, or closed by this or another
            # client — a Notice with the specifics precedes this event).
            self._detach()
            return
        # Everything else is a streaming-round event.
        if self._view is None:
            self._view = _RoundView(self, self.query_one("#chat-log", VerticalScroll))
            await self._view.mount()
        await self._view.handle(event)
        if isinstance(event, ToolFinished):
            self._refresh_diff_summary_soon()

    # ------------------------------------------------------------------
    # Commands

    async def _send_command(self, command: Command) -> dict | None:
        """Send one command to the session; returns its result payload.

        Errors (busy rejection, unknown target, git failure) surface as an
        ErrorLine — the session broadcast the effects to everyone anyway."""
        if self.session_name is None:
            return None
        try:
            result = await self.client.command(self.session_name, command)
        except (ClientError, httpx.HTTPError) as exc:
            self._post_line(ErrorLine(str(exc)))
            return None
        return result.payload

    def _post_command(self, command: Command) -> None:
        """Fire-and-forget command send: results are ignored (the effects
        arrive as events / StatusChanged anyway). Every caller that does not
        await _send_command must go through this — calling the coroutine
        bare silently does nothing."""
        self.run_worker(
            self._send_command(command), group="post-command", exit_on_error=False,
        )

    # ------------------------------------------------------------------
    # Terminal title

    def _set_terminal_title(self, title: str) -> None:
        """Write an OSC 0 sequence so the terminal tab/window shows `title`."""
        driver = self._driver
        if driver is None:
            return
        driver.write(f"\x1b]0;{title}\x07")
        driver.flush()

    def _set_default_title(self) -> None:
        """Default title: `xarness <workdir>`, path cropped like the status
        bar (tail kept, whole components) but without the leading `...`."""
        path = str(self.workspace or Path.cwd())
        cropped = crop_path(path, _TITLE_PATH_LEN)
        if cropped.startswith(".../"):
            cropped = cropped[4:]
        self._set_terminal_title(f"xarness {cropped}")

    def _refresh_title(self) -> None:
        """Follow the attached session: its workspace in the tab title, or
        the default when detached — otherwise switching to a session in a
        different workspace keeps the stale title."""
        workspace = self.status.workspace if (
            self.session_name is not None and self.status is not None
        ) else None
        if workspace:
            cropped = crop_path(workspace, _TITLE_PATH_LEN)
            if cropped.startswith(".../"):
                cropped = cropped[4:]
            self._set_terminal_title(f"xarness {cropped}")
        else:
            self._set_default_title()

    # ------------------------------------------------------------------
    # Status bar / bars

    def _refresh_status(self) -> None:
        status = self.status
        if status is None:
            # Draft chat: the backend tells us what a fresh session starts
            # with (/config defaults), so show that instead of a dash.
            config = self._config_cache
            self.query_one("#status-bar", StatusBar).update_status(
                shown_name=(
                    config.default_model_display_name if config else None
                ) or "—",
                effort="medium", mode=self._draft_mode,
                total_in=0, total_out=0, context_used=0, max_context=128_000,
                workspace=str(self.workspace) if self.workspace else None,
            )
            return
        self.query_one("#status-bar", StatusBar).update_status(
            shown_name=status.model_display_name,
            effort=status.cot_strength,
            mode=status.mode,
            total_in=status.total_in,
            total_out=status.total_out,
            context_used=status.context_used,
            max_context=status.max_context,
            workspace=status.workspace,
        )

    @work(group="draft-status", exit_on_error=False)
    async def _load_draft_defaults(self) -> None:
        """Draft chat: fetch /config so the status bar can show the default
        model instead of a dash (and /model works without a round trip
        later). Harmless no-op once a session is attached."""
        if self.session_name is not None or self._config_cache is not None:
            return
        try:
            self._config_cache = await self.client.config()
        except (ClientError, httpx.HTTPError):
            return
        if self.session_name is None:
            self._refresh_status()

    def _refresh_generating(self) -> None:
        try:
            bar = self.query_one("#generating-bar", GeneratingBar)
        except NoMatches:
            return
        state = self.status.state if self.status else "idle"
        if state in ("running", "compacting"):
            bar.add_class("active")
        else:
            bar.remove_class("active")

    def _refresh_steer_bar(self) -> None:
        """Mirror the session's queued steer messages into the bar above the
        input. The queue is session state (StatusChanged.queued), so every
        client's bar shows it."""
        try:
            bar = self.query_one("#steer-queue-bar", SteerQueueBar)
        except NoMatches:
            return  # app shutting down
        queued = self.status.queued if self.status else []
        bar.update_items([q.text for q in queued])

    def _refresh_diff_summary_soon(self) -> None:
        """Recompute the pending-changes bar soon (debounced per event)."""
        self.set_timer(0.05, lambda: self.call_later(self.refresh_diff_summary))

    async def refresh_diff_summary(self) -> None:
        """Recompute the pending-changes bar (hidden when empty or gitless)."""
        try:
            widget = self.query_one("#diff-summary", DiffSummary)
        except NoMatches:
            return  # app shutting down; DOM already pruned
        if self.session_name is None:
            await widget.clear()
            return
        try:
            info = await self.client.diff(self.session_name)
        except ClientError:
            await widget.clear()  # tracking broken (e.g. repo gone): stay quiet
            return
        if not info.files:
            await widget.clear()
        else:
            await widget.set_summary(info)

    # ------------------------------------------------------------------
    # Submission / ask flow

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

    def _submit(self, text: str, images: list[ImageAttachment] | None = None) -> None:
        """Send a message. No local echo: the message lands in the transcript
        when UserMessageAdded arrives, identically for every client."""
        self.call_later(self._submit_async, text, images)

    async def _submit_async(self, text: str, images: list[ImageAttachment] | None) -> None:
        if self.session_name is None:
            # Draft chat: the session comes into existence on the server with
            # the first message.
            try:
                summary = await self.client.create_session(
                    workspace=str(self.workspace) if self.workspace else None,
                    model=self.model,
                    sandbox=self.sandbox_name,
                    preset=self.preset,
                    network=self.network,
                    gpu=self.gpu,
                    no_init_repo=self.no_init_repo,
                )
            except (ClientError, httpx.HTTPError) as exc:
                self._post_line(ErrorLine(f"could not start a session: {exc}"))
                return
            self.model = None
            self.sandbox_name = None
            self.preset = None
            self.network = None
            self.gpu = None
            await self._attach(summary.name)
            # Carry the draft-mode toggle (Tab before any message) into the
            # freshly created session.
            if self._draft_mode != "write":
                self._post_command(SetMode(mode=self._draft_mode))
            self._draft_mode = "write"
        # Pasted/dragged images ride along as base64 attachments; @path
        # tokens in the text are resolved server-side.
        models = [
            ImageAttachmentModel(
                name=i.name, mime=i.mime, data_b64=i.data_b64,
                width=i.width, height=i.height,
            )
            for i in (images or [])
        ]
        await self._send_command(SendMessage(text=text, images=models or None))

    async def _mount_user_message(self, event: UserMessageAdded) -> None:
        """Land a user message in the transcript (sent or steered)."""
        chat = self.query_one("#chat-log", VerticalScroll)
        images = [
            ImageAttachment(name=i.name, mime=i.mime, data_b64=i.data_b64,
                            width=i.width, height=i.height)
            for i in (event.images or [])
        ]
        widget = UserMessage(event.text, images or None)
        widget.message = Message(role="user", content=event.text, id=event.message_id or "")
        widget.mark_sent()
        await self._mount_spaced(chat, widget)
        # Track the rendered id: HistoryReplaced's trim path compares these
        # against the new history to trim in place instead of re-rendering
        # the whole chat (undo-from-here on a message that arrived live
        # during this attach must trim, not reload).
        if event.message_id:
            self._rendered_ids.add(event.message_id)

    # ------------------------------------------------------------------
    # Ask tool (event-driven)

    @work(group="ask")
    async def _ask_flow(self, ask_id: str, questions: list[str]) -> None:
        """Show the questions one at a time in the AskBar; the user answers
        through the normal chat input (Enter submits, Esc skips all). The
        collected answers go back as one ``answer_ask`` command — any client
        may answer; first answer wins."""
        ask_bar = self.query_one("#ask-bar", AskBar)
        chat_input = self.query_one("#chat-input", ChatInput)
        answers: list[str] = []
        try:
            for index, question in enumerate(questions):
                ask_bar.show_question(f"Question {index + 1}/{len(questions)}: {question}")
                chat_input.placeholder = ASK_PLACEHOLDER
                chat_input.focus()
                future: asyncio.Future[str | None] = (
                    asyncio.get_running_loop().create_future()
                )
                self._ask_future = future
                self._ask_id = ask_id
                answer = await future
                if answer is None:  # esc: skip the remaining questions
                    await self._send_command(AnswerAsk(ask_id=ask_id, answers=None))
                    return
                answers.append(answer)
            await self._send_command(AnswerAsk(ask_id=ask_id, answers=answers))
        finally:
            self._ask_future = None
            self._ask_id = None
            ask_bar.hide()
            chat_input.placeholder = INPUT_PLACEHOLDER

    def _on_ask_resolved(self, event: AskResolved) -> None:
        """Another client answered first: close this client's prompt."""
        if self._ask_id is not None and self._ask_id == event.ask_id:
            if self._ask_future is not None and not self._ask_future.done():
                self._ask_future.cancel()
            if self._ask_worker is not None:
                self._ask_worker.cancel()
                self._ask_worker = None

    # ------------------------------------------------------------------
    # Drift prompt (session state; any client may resolve)

    async def _maybe_prompt_drift(self) -> None:
        """The workspace changed outside this session while it was closed.

        Show what drifted and let the user choose: accept it as the new
        baseline, or keep tracking against the last known state. Never pick
        for them. The prompt is session state — if another client resolves
        it first, DriftResolved closes this modal's follow-up."""
        if (
            self.status is None or self.status.pending_drift is None
            or self._drift_modal_open
        ):
            return
        self._drift_modal_open = True
        choice = await self.push_screen_wait(ConfirmScreen(
            "Workspace changed outside this session since it was last open:",
            confirm_label="Accept as baseline",
            cancel_label="Keep tracking",
        ))
        self._drift_modal_open = False
        if self.status is None or self.status.pending_drift is None:
            return  # resolved elsewhere while the modal was open
        if choice is None:
            return  # dismissed: leave the prompt pending
        await self._send_command(ResolveDrift(accept=bool(choice)))

    # ------------------------------------------------------------------
    # Slash commands

    def _handle_slash_command(self, text: str) -> None:
        parts = text[1:].split(maxsplit=1)
        cmd = parts[0] if parts else ""
        if cmd == "tools":
            names = ", ".join(self.status.tools) if self.status and self.status.tools else "none"
            self._post_line(MessageLine(f"tools: {names}", kind="info"))
        elif cmd == "sessions":
            self._open_sessions()
        elif cmd == "new":
            self._start_new_session()
        elif cmd == "delete":
            self._start_delete()
        elif cmd == "accept":
            self._post_command(Accept())
        elif cmd == "reject":
            self._post_command(Reject())
        elif cmd == "undo":
            self._post_command(Undo())
        elif cmd == "retry":
            self._post_command(Retry())
        elif cmd == "diff":
            arg = parts[1].strip() if len(parts) > 1 else ""
            self._show_diff_command(arg)
        elif cmd == "model":
            self._open_model_picker()
        elif cmd == "sandbox":
            self._open_sandbox_picker()
        elif cmd == "filter":
            arg = parts[1].strip() if len(parts) > 1 else ""
            self._handle_filter_command(arg)
        elif cmd == "mode":
            self._switch_mode()
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
        """Toggle plan/write mode; shared by /mode and the Tab shortcut.
        With no session attached the toggle is local draft state, applied to
        the session the moment the first message creates it."""
        if self.session_name is None:
            self._draft_mode = "plan" if self._draft_mode == "write" else "write"
            self._refresh_status()
            return
        current = self.status.mode if self.status else "write"
        new_mode = "plan" if current == "write" else "write"
        self._post_command(SetMode(mode=new_mode))

    async def _run_undo_to(self, message_id: str) -> None:
        """Click-to-undo: drop the clicked user message and everything after
        it (file edits reverted), putting that message back in the input."""
        payload = await self._send_command(UndoTo(message_id=message_id))
        self._put_back_user_text(payload)

    def _put_back_user_text(self, payload: dict | None) -> None:
        """The user text an /undo wants back in the input box (the command
        result goes only to the issuer — everyone else just saw the
        HistoryReplaced broadcast)."""
        user_text = (payload or {}).get("user_text")
        if user_text:
            chat_input = self.query_one("#chat-input", ChatInput)
            chat_input.load_text(user_text)
            chat_input.focus()

    def on_user_message_undo_requested(self, event: UserMessage.UndoRequested) -> None:
        event.stop()
        message = event.user_message.message
        if message is not None and message.id:
            self.call_later(self._run_undo_to, message.id)

    def _start_new_session(self) -> None:
        """Slash /new: leave the current session (it keeps running in the
        host) and return to a draft chat."""
        self._detach()
        self._post_line(NoticeLine(
            "started a fresh chat — the previous session keeps running in "
            "the background (/sessions to go back)"
        ))

    def _detach(self) -> None:
        """Return to a draft chat: no session attached, empty log."""
        if self._pump_worker is not None:
            self._pump_worker.cancel()
            self._pump_worker = None
        self.session_name = None
        self.status = None
        self._view = None
        self._last_thinking = None
        self._refresh_title()
        chat = self.query_one("#chat-log", VerticalScroll)
        chat.remove_children()
        # Fresh, empty log: follow it again from the top of the new session.
        chat.anchor()
        self._refresh_status()
        self._refresh_steer_bar()
        self._refresh_generating()
        self.call_later(self.refresh_diff_summary)

    def _start_delete(self) -> None:
        """Slash /delete: end the current session for good. Deleting a
        session is pure bookkeeping — the saved session file is removed and
        a fresh session starts; the workspace is never touched."""
        self.call_later(self._delete_async)

    async def _delete_async(self) -> None:
        name = self.session_name
        self._detach()
        if name is None:
            self._post_line(NoticeLine("no session to delete"))
            return
        try:
            await self.client.delete_session(name)
        except (ClientError, httpx.HTTPError) as exc:
            self._post_line(ErrorLine(f"/delete failed: {exc}"))
            return
        self._post_line(NoticeLine(
            "deleted: saved session removed — started a fresh session; "
            "your files are untouched"
        ))

    def _open_sessions(self) -> None:
        self.call_later(self._open_sessions_async)

    async def _open_sessions_async(self) -> None:
        try:
            entries = await self.client.sessions()
        except (ClientError, httpx.HTTPError) as exc:
            self._post_line(ErrorLine(f"/sessions failed: {exc}"))
            return
        self.push_screen(
            ResumeScreen(
                entries,
                workspace=str(self.workspace) if self.workspace else None,
            ),
            self._on_session_selected,
        )

    async def _on_session_selected(self, name: str | None) -> None:
        if not name:
            return
        if name == self.session_name:
            return
        try:
            active = {s.name: s for s in await self.client.sessions()}
            summary = active.get(name)
            if summary is None or not summary.active:
                await self.client.load_session(name)
            await self._attach(name)
        except (ClientError, httpx.HTTPError) as exc:
            self._post_line(ErrorLine(f"could not open {name}: {exc}"))

    def _open_model_picker(self) -> None:
        self.call_later(self._open_model_picker_async)

    async def _open_model_picker_async(self) -> None:
        try:
            self._config_cache = await self.client.config()
        except (ClientError, httpx.HTTPError) as exc:
            self._post_line(ErrorLine(f"/model failed: {exc}"))
            return
        names = [m["name"] for m in self._config_cache.models]
        self.push_screen(
            PickerScreen(
                names,
                current=self.status.model_name if self.status else None,
                title="Switch model",
            ),
            self._on_model_selected,
        )

    def _on_model_selected(self, name: str | None) -> None:
        if not name:
            return
        self._post_command(SwitchModel(name=name))
        self._save_preference(default_profile=name)

    def _open_sandbox_picker(self) -> None:
        self.call_later(self._open_sandbox_picker_async)

    async def _open_sandbox_picker_async(self) -> None:
        try:
            self._config_cache = await self.client.config()
        except (ClientError, httpx.HTTPError) as exc:
            self._post_line(ErrorLine(f"/sandbox failed: {exc}"))
            return
        names = [s["name"] for s in self._config_cache.sandboxes]
        self.push_screen(
            PickerScreen(
                names,
                current=self.status.sandbox_name if self.status else None,
                title="Sandbox profile (applies when the session is idle)",
            ),
            self._on_sandbox_selected,
        )

    def _on_sandbox_selected(self, name: str | None) -> None:
        if not name:
            return
        self._post_command(SetSandbox(name=name))

    def _handle_filter_command(self, arg: str) -> None:
        sub = arg.split(maxsplit=1)
        action = sub[0] if sub else "list"
        path = sub[1].strip() if len(sub) > 1 else ""
        if action == "add" and path:
            self._post_command(FilterAdd(path=path))
        elif action == "remove" and path:
            self._post_command(FilterRemove(path=path))
        elif action == "list":
            filters = self.status.filter_list if self.status else []
            if filters:
                self._post_line(MessageLine(
                    "filtered out of the sandbox mounts: " + ", ".join(filters),
                    kind="info",
                ))
            else:
                self._post_line(MessageLine("no paths filtered out", kind="info"))
        else:
            self._post_line(ErrorLine("usage: /filter add <path> | /filter remove <path> | /filter list"))

    # ------------------------------------------------------------------
    # Theme (client-side preference)

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
        except Exception as exc:  # noqa: BLE001 - preference save is best-effort
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

    # ------------------------------------------------------------------
    # Rendering

    async def _mount_spaced(
        self,
        chat: "VerticalScroll",
        widget: "ThinkingBlock | AssistantMessage | UserMessage",
    ) -> None:
        """Mount a cot/assistant/user block, keeping a blank line above it when
        it directly follows tool call blocks (which carry no spacing of their
        own)."""
        children = chat.children
        if children and isinstance(children[-1], ToolCallBlock):
            widget.add_class("after-tools")
        await chat.mount(widget)

    async def _on_history_replaced(self, event: HistoryReplaced) -> None:
        """The conversation was rewritten (undo, retry, accept, compaction,
        drift accept): re-render, trimming the removed tail when possible."""
        self._view = None
        new_ids = {
            m.message_id for m in event.snapshot.messages if m.message_id
        }
        if hasattr(self, "_rendered_ids") and self._rendered_ids == new_ids:
            # Nothing added or removed (e.g. /accept, which only moves the
            # baseline): the rendered transcript is already correct — don't
            # rebuild the whole chat, just the parts that depend on git
            # state.
            self._refresh_status()
            self.call_later(self.refresh_diff_summary)
            return
        dropped = self._rendered_ids - new_ids if hasattr(self, "_rendered_ids") else set()
        self._rendered_ids = new_ids
        if dropped and await self._trim_history(dropped):
            self._refresh_status()
            self.call_later(self.refresh_diff_summary)
            return
        await self._render_history(event.snapshot)
        self._refresh_status()
        self.call_later(self.refresh_diff_summary)

    async def _trim_history(self, dropped_ids: set[str]) -> bool:
        """Remove the rendered widgets for `dropped_ids` and everything
        after them, keeping the rest of the transcript untouched.

        Returns False — the caller falls back to a full _render_history —
        when no rendered widget matches (e.g. a conversation loaded but never
        rendered, or a widget without a message binding)."""
        chat = self.query_one("#chat-log", VerticalScroll)
        children = list(chat.children)
        cut = next(
            (
                i
                for i, widget in enumerate(children)
                if getattr(widget, "message", None) is not None
                and getattr(widget.message, "id", None) in dropped_ids
            ),
            None,
        )
        if cut is None:
            return False
        for widget in children[cut:]:
            await widget.remove()
        chat.scroll_end(animate=False)
        return True

    async def _render_history(self, snap: Snapshot) -> None:
        """Rebuild #chat-log from a snapshot.

        Best-effort reconstruction: tool-call status is inferred from whether
        the recorded tool-role content starts with "error: ", since the
        original ToolCallStatus enum value isn't itself persisted.

        If the conversation was compacted, the pre-compaction messages are
        rendered first (they are history, kept for the human), then a
        divider, then the summary and everything after it — mirroring what
        the model actually sees. The current turn's buffered events are fed
        through a fresh _RoundView, so a mid-turn attach rebuilds the
        thinking block, partial answer and running tool blocks."""
        self._view = None
        self._rendered_ids = {
            m.message_id for m in snap.messages if m.message_id
        }
        chat = self.query_one("#chat-log", VerticalScroll)
        await chat.remove_children()
        messages = snap.messages
        pre_compact = snap.compact_snapshot
        all_messages = [*pre_compact, *messages]
        tool_results = {
            m.tool_call_id: m for m in all_messages
            if m.role == "tool" and m.tool_call_id
        }

        for message in pre_compact:
            await self._render_message(chat, message, tool_results)
        if pre_compact:
            await chat.mount(NoticeLine(COMPACTED_DIVIDER))
        for message in messages:
            await self._render_message(chat, message, tool_results)
        chat.scroll_end(animate=False)
        # Mid-turn attach: rebuild the live round from the buffered events.
        for event in snap.turn_events:
            await self._handle_event(event)

    async def _render_message(
        self,
        chat: "VerticalScroll",
        message: MessageModel,
        tool_results: dict[str, MessageModel],
    ) -> None:
        """Render one persisted message into the chat log."""
        if message.role == "user":
            if message.content.startswith(SUMMARY_PREFIX):
                # The handoff an auto-compaction left behind: padded and
                # accent-colored, not a plain user message.
                summary = CompactionSummary(
                    message.content.removeprefix(SUMMARY_PREFIX).strip()
                )
                summary.message = Message(
                    role="user", content=message.content, id=message.message_id or "",
                )
                await chat.mount(summary)
            else:
                images = [
                    ImageAttachment(name=i.name, mime=i.mime, data_b64=i.data_b64,
                                    width=i.width, height=i.height)
                    for i in (message.images or [])
                ]
                widget = UserMessage(message.content, images or None)
                widget.message = Message(
                    role="user", content=message.content, id=message.message_id or "",
                )
                await self._mount_spaced(chat, widget)
        elif message.role == "assistant":
            if message.reasoning:
                thinking = ThinkingBlock()
                await self._mount_spaced(chat, thinking)
                thinking.append_reasoning(message.reasoning)
                thinking.finish(message.reasoning_seconds, estimate_if_unknown=False)
            # Tool-call-only rounds render no AssistantMessage body; an
            # empty assistant message with no tool calls replays as
            # "(no output)" — except a reasoning-only partial (saved on
            # cancel): its thinking block is the whole message.
            if message.content or (not message.tool_calls and not message.reasoning):
                assistant = AssistantMessage()
                await self._mount_spaced(chat, assistant)
                if message.content:
                    await assistant.append_delta(message.content)
                await assistant.finalize()
            for call in message.tool_calls or []:
                function = call.get("function") or {}
                block = ToolCallBlock(call.get("id", ""), function.get("name", ""))
                await chat.mount(block)
                block.append_arguments(function.get("arguments", ""))
                result = tool_results.get(call.get("id", ""))
                if result is not None:
                    is_error = result.content.startswith("error: ")
                    status = (
                        ToolCallStatus.CALL_FAILED if is_error
                        else ToolCallStatus.CALL_SUCCEEDED
                    )
                    block.set_result(
                        status,
                        output="" if is_error else result.content,
                        error=result.content[len("error: "):] if is_error else "",
                        header=result.header or "",
                    )
                else:
                    # The call never produced a result: the turn was
                    # interrupted mid-execution and the session was saved.
                    # It can never complete now — render it as failed instead
                    # of an eternally shimmering "Running …".
                    block.set_result(ToolCallStatus.CALL_FAILED, error="interrupted")

    # ------------------------------------------------------------------
    # Popups (@ search runs server-side against the session's workspace)

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
        root = str(self.workspace or Path.cwd())
        try:
            results = await self.client.search_files(root, query)
        except ClientError:
            results = []
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

    # ------------------------------------------------------------------
    # Steer queue commands

    def action_steer_discard(self) -> None:
        """ctrl+shift+backspace: discard the oldest queued steer message."""
        if self.status is None or not self.status.queued:
            return
        self._post_command(DiscardQueued())

    def _flush_queued_now(self) -> None:
        """Enter on an empty input while messages are queued: send now.

        Cancels the running turn; the session starts the queued message
        immediately instead of waiting for the next round boundary."""
        if self.status and self.status.queued and self.status.state == "running":
            self._post_command(FlushQueued())

    # ------------------------------------------------------------------
    # Copy / quit / interrupt

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

    @work(group="quit", exclusive=True)
    async def action_copy_or_quit(self) -> None:
        """ctrl+c: copy the active selection if there is one, quit otherwise.

        Quitting while a request is still running asks first: the session
        lives in the host, so it can keep running headless — or be cancelled
        on the way out. Idle quits immediately. (A worker: push_screen_wait
        requires one.)"""
        if self._copy_selection():
            return
        self._disarm_interrupt()
        state = self.status.state if self.status else "idle"
        has_pending_ask = bool(self.status and self.status.pending_ask)
        busy = self.session_name is not None and (
            state in ("running", "compacting") or has_pending_ask
        )
        if busy:
            # Another client has the session open: the session keeps running
            # for them regardless, and "stop and quit" would yank it away —
            # so don't ask, just detach and exit.
            if self.status is not None and self.status.clients_attached > 1:
                self.exit()
                return
            choice = await self.push_screen_wait(ContinueHeadlessScreen())
            if choice is None:
                return  # esc: stay in the app
            if choice:
                # Close the event stream and exit; the session keeps
                # running in the host.
                self.exit()
                return
            # Stop and quit: cancel, wait for it to unwind, then unload the
            # session from the host (it stays saved on disk) and exit.
            try:
                await asyncio.wait_for(
                    self._send_command(Cancel()), timeout=CANCEL_TIMEOUT,
                )
            except (asyncio.TimeoutError, ClientError):
                pass
            if self.session_name is not None:
                try:
                    await self.client.close_session(self.session_name)
                except (ClientError, httpx.HTTPError):
                    pass  # exiting anyway; the host grace-exits on its own
            self.exit()
            return
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
        if self.status is not None and self.status.state in ("running", "compacting"):
            if self._interrupt_armed:
                # Second esc within the window: actually stop the turn.
                self._disarm_interrupt()
                self._post_command(Cancel())
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

    # ------------------------------------------------------------------
    # /diff

    @work(group="diff-view", exclusive=True)
    async def _show_diff_command(self, arg: str) -> None:
        """Slash /diff: toggle the pending-changes panel — the first /diff
        expands the per-file list, /diff again hides it entirely. With an
        argument, /diff <path> shows that one file's unified diff."""
        if self.session_name is None:
            self._post_line(ErrorLine(
                "/diff: no session attached — send a message first"
            ))
            return
        widget = self.query_one("#diff-summary", DiffSummary)
        try:
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
            info = await self.client.diff(self.session_name, path=arg)
            text = await self._diff_text(arg)
            if not text.strip():
                self._post_line(NoticeLine(f"no pending changes for {arg}"))
                return
            # Rebuild the row list first (a previous /diff toggle may have
            # cleared it) so the active-row highlight has something to attach to.
            await self.refresh_diff_summary()
            widget.show_diff(arg, text)
        except (ClientError, httpx.HTTPError) as exc:
            widget.hide_diff()
            self._post_line(ErrorLine(f"/diff failed: {exc}"))

    async def _diff_text(self, path: str) -> str:
        """One file's unified diff (the summary endpoint gives per-file
        stats; the text comes from the same API's diff endpoint, which
        populates ``text`` for path-specific requests)."""
        info = await self.client.diff(self.session_name, path=path)
        return info.text or ""

    async def on_diff_summary_diff_requested(self, event: DiffSummary.DiffRequested) -> None:
        if self.session_name is None:
            return
        widget = self.query_one("#diff-summary", DiffSummary)
        if widget.active_diff_key == event.key:
            # Same affordance again: close the diff pane only — the file
            # list stays up with the row still highlighted (same state as
            # esc-minimize), so keyboard navigation keeps its context.
            widget.minimize_diff()
            return
        try:
            text = await self._diff_text(event.key or "")
        except (ClientError, httpx.HTTPError) as exc:
            widget.hide_diff()
            self._post_line(ErrorLine(f"diff failed: {exc}"))
            return
        widget.show_diff(event.key, text)

    # ------------------------------------------------------------------
    # misc

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
        self._clear_hint_lines()
        if assistant is not None:
            await assistant.finalize()
        elif thinking is None and not event.has_tool_calls:
            await chat.mount(ErrorLine("empty response from model"))
        if thinking is not None:
            thinking.finish(event.reasoning_seconds)
        if self._view is not None:
            await self._view.finish()
        self._refresh_status()

    def action_toggle_thoughts(self) -> None:
        if self._last_thinking is not None:
            self._last_thinking.toggle()
