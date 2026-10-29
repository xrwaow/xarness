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
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from rich.cells import cell_len
from rich.segment import Segment
from rich.style import Style

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Horizontal
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.strip import Strip
from textual.widgets import OptionList, Static, TextArea
from textual.worker import Worker

from .. import theme
from ..clientlib import ClientError, XarnessClient, ensure_host
from ..config import DEFAULT_CONFIG_PATH, save_preferences
from ..images import ImageAttachment
from ..protocol import (
    AnswerAsk, AskRequested, AskResolved, Cancel, Command, ContentDelta,
    CreateSessionRequest, DiffChanged, DiscardQueued, DriftDetected,
    DriftResolved, Event, CompactionDone, RollbackApplied,
    CommandPayload, RollbackResult,
    FilterAdd, FilterRemove,
    FlushQueued, HistoryReplaced,
    Notice, ProcessingStarted, ReasoningDelta, Reject, ResolveDrift,
    Retry, SendMessage, SessionDeleted, SetMode, SetSandbox, Snapshot,
    StatusChanged, StatusPayload, StreamError, Accept, SwitchModel,
    ToolCallStarted, TurnComplete,
    TurnEnded, TurnFinished, Undo, UndoTo, UserMessageAdded,
)
from textual_flowview import Anchor, FlowModel, FlowView

from ..util import format_duration as _format_duration
from . import flow as tflow
from .flow import LineItem, PendingItem
from .transcript import Transcript
from .confirm_screen import ConfirmScreen
from .list_picker import ListPicker, name_options, session_options
from .refs_screen import RefsScreen
from .diff_summary import DiffSummary
from .widgets import (
    AskBar, ChatInput, GeneratingBar, StatusBar,
    SteerQueueBar, SuggestionPopup,
)

# Reconnect backoff for the one recovery loop (seconds).
RECONNECT_MIN = 0.25
RECONNECT_MAX = 4.0

INPUT_PLACEHOLDER = (
    "Send a message…  (Enter: send · Shift+Enter: newline · Tab: mode)"
)
ASK_PLACEHOLDER = "Type your answer…  (Enter: send · Esc: skip)"


# How long a first Esc stays "armed" waiting for the confirming second one.
INTERRUPT_ARM_SECONDS = 2.5

# How long to wait for a cancel command to unwind before exiting anyway.
CANCEL_TIMEOUT = 2.0

COMMANDS: dict[str, "SlashCommand"] = {}


@dataclass(frozen=True)
class SlashCommand:
    """One slash command: the popup, dispatcher and usage errors share it."""

    name: str
    description: str
    handler: Callable[["AgentApp", str], None]
    # Takes an argument: the popup completes the word and lets the user type
    # it instead of executing immediately (Enter on "filter" used to run the
    # command with an empty arg).
    takes_arg: bool = False


def _command(name: str, description: str, handler, takes_arg: bool = False):
    COMMANDS[name] = SlashCommand(name, description, handler, takes_arg)


_command("tools", "list available tools",
         lambda s, a: s._notice(
             "tools: " + (
                 ", ".join(s.view_status.tools) if s.view_status.tools else "none"
             )
         ))
_command("model", "choose what model and reasoning effort to use",
         lambda s, a: s._pick_model())
_command("sessions", "resume a previous session",
         lambda s, a: s.call_later(s._open_sessions_async))
_command("mode", "switch between plan (read-only) and write mode",
         lambda s, a: s._switch_mode())
_command("sandbox", "choose this session's sandbox profile (idle only)",
         lambda s, a: s._pick_sandbox())
_command("filter", "add/remove/list paths excluded from the sandbox mounts",
         lambda s, a: s._handle_filter_command(a), takes_arg=True)
_command("refs", "add/remove this workspace's external reference paths",
         lambda s, a: s._handle_refs_command(a), takes_arg=True)
_command("theme", "choose a color theme",
         lambda s, a: s._handle_theme_command(a) if a else s._pick_theme(),
         takes_arg=True)
_command("new", "leave this session and start a fresh chat",
         lambda s, a: s._start_new_session())
_command("delete",
         "delete the current session and start a new one (your files are left exactly as they are)",
         lambda s, a: s.call_later(s._delete_async))
_command("diff", "show pending changes (optionally: /diff <path>)",
         lambda s, a: s._show_diff_command(a), takes_arg=True)
_command("accept",
         "lock in the changes made so far (they stop showing in /diff and can no longer be undone)",
         lambda s, a: s._post_command(Accept()))
_command("reject", "discard all changes made since the last /accept",
         lambda s, a: s._post_command(Reject()))
_command("undo", "drop the last turn and revert its file edits",
         lambda s, a: s.call_later(s._run_undo))
_command("retry", "revert the last turn's file edits and resend your message",
         lambda s, a: s._post_command(Retry()))


# Transcript virtualization: the chat log is a textual_flowview.FlowView —
# it paints only the visible rows, so the whole history lives in the model
# and no windowing/pagination is needed at any transcript length.


class ChatFlowView(FlowView):
    """FlowView with re-entrant presentation defused.

    ``present`` runs on the event loop via a Textual worker, and Textual's
    eager task factory starts ``create_task`` coroutines *synchronously on
    the current stack*. So a present kicked off from layout/scroll/compositor
    paint re-enters the paint/present cycle on the same stack, ~30 frames
    deeper per hop — in the wild that marched to the recursion limit
    mid-render and permanently pinned entries to stale presentations.

    Always deferring to a fresh event-loop turn breaks every such cycle; the
    extra frame of latency is invisible. ``_pending_present`` dedupes on
    entry id so an entry never queues twice. (Long-term fix: in
    textual_flowview itself.)"""

    can_focus = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._pending_present: set[int] = set()

    def _present_entry(self, entry) -> None:  # type: ignore[override]
        if entry.id in self._pending_present:
            return
        self._pending_present.add(entry.id)
        self.call_after_refresh(self._do_present_entry, entry)

    def _do_present_entry(self, entry) -> None:
        self._pending_present.discard(entry.id)
        super()._present_entry(entry)

    def render_line(self, y: int) -> Strip:
        strip = super().render_line(y)
        # Empty/unstyled cells (blank rows of an empty or partially-filled
        # log) carry no bg and fall through to the TERMINAL's default colors
        # — the widget CSS background does not reach raw strip cells. Fill
        # them with the palette so e.g. a fresh draft chat doesn't show the
        # terminal's (dark) background after a /theme switch. Cell styles
        # win, so presenter-painted rows are untouched.
        if any(seg.style is None or seg.style.bgcolor is None for seg in strip):
            strip = strip.apply_style(Style(bgcolor=theme.PALETTE["bg"]))
        # The lib's _decorate_line applies the selection highlight with
        # Strip.apply_style, and rich's style merge lets each cell's own
        # bgcolor win — so rows whose cells carry an explicit background
        # (the user-message box's surface) highlight nothing when selected.
        # Re-apply the selection span here with the cell styles overridden.
        selection = self.text_selection
        if selection is None:
            return strip
        span = selection.get_span(y + int(self.scroll_offset.y))
        if span is None:
            return strip
        width = strip.cell_length
        if width <= 0:
            return strip
        body_start = min(self._left_gutter_w(), width)
        body_end = max(body_start, width - self._right_gutter_w())
        body_text = strip.crop(body_start, body_end).text
        n = len(body_text)
        # The span is in *character* offsets into the body (end == -1 means
        # "to end of line"); convert to cells, or CJK / emoji rows highlight
        # the wrong columns. Same conversion the lib's _decorate_line does.
        end_c = n if span[1] == -1 else span[1]
        start_c = max(0, min(span[0], n))
        end_c = max(start_c, min(end_c, n))
        start_cell = body_start + cell_len(body_text[:start_c])
        end_cell = body_start + cell_len(body_text[:end_c])
        if end_cell <= start_cell:
            return strip
        sel_style = self.selection_style
        out = []
        x = 0
        for segment in strip:
            if start_cell <= x < end_cell:
                # ``+`` lets the right operand win: selection colors override
                # any cell bg/fg, while meta (content offsets) is kept. Blank
                # filler cells can carry style=None — use the selection style
                # alone there (None + Style raises TypeError).
                base = segment.style
                out.append(Segment(
                    segment.text,
                    base + sel_style if base is not None else sel_style,
                ))
            else:
                out.append(segment)
            x += segment.cell_length
        return Strip(out, width)


class AgentApp(App[None]):
    CSS_PATH = "styles.tcss"
    TITLE = "xarness"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c", "copy_or_quit", "Copy / Quit", priority=True),
        Binding("ctrl+shift+c", "copy_selection", "Copy", priority=True),
        Binding("escape", "interrupt", "Interrupt", priority=True),
        Binding("ctrl+shift+backspace", "steer_discard", "Discard queued", priority=True),
    ]

    def __init__(
        self,
        client: XarnessClient | None,
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
        # The pure reducer owns the transcript data; the FlowModel holds one
        # widget entry per item. _entries maps id(item) -> entry so live
        # mutations repaint the right row without a second copy of state.
        self._transcript = Transcript()
        self._flow_model: "FlowModel" = FlowModel()
        self._entries: dict[int, object] = {}
        self._anim: dict[int, object] = {}
        self._presenter = tflow.Presenter()
        # Transient hint lines ("press esc again …"), removed when the
        # assistant's message settles.
        self._hint_entries: list = []
        self._pump_worker: Worker | None = None
        self._ask_future: asyncio.Future[str | None] | None = None
        self._ask_id: str | None = None
        self._ask_worker: Worker | None = None
        self._drift_modal_open = False
        # Double-Esc interrupt: the first Esc arms it (and shows a prompt),
        # the second one within INTERRUPT_ARM_SECONDS actually cancels.
        self._interrupt_armed = False
        self._interrupt_timer = None
        self._at_search_timer = None
        # Commands go through one ordered queue worker: the server lock is
        # FIFO by arrival, so concurrent HTTP sends could reorder commands.
        self._command_queue: asyncio.Queue[Command] = asyncio.Queue()
        self._command_worker: Worker | None = None
        self._config_cache = None
        # Live phase — the single owner of the ephemeral pending strip.
        # pending = a round is waiting on the provider (turn start and every
        # post-tools round); the transitions below are the strip's only
        # append/remove sites, so a dropped round can never orphan it.
        self._phase = "idle"
        self._pending_entry = None
        # Draft-chat state (before any session exists): the mode Tab toggles
        # locally and is applied to the session once it is created.
        self._draft_mode = "write"

    # -- live phase (the pending strip's only owner) ---------------------

    def _phase_pending(self) -> None:
        """A round is waiting on the provider: show the strip."""
        if self._phase == "pending":
            return
        self._phase = "pending"
        self._pending_entry = self._append_item(PendingItem())

    def _phase_streaming(self) -> None:
        """Content is arriving: the strip's job is done."""
        self._phase = "streaming"
        self._drop_pending()

    def _phase_idle(self) -> None:
        """Nothing in flight: no strip."""
        self._phase = "idle"
        self._drop_pending()

    def _drop_pending(self) -> None:
        if self._pending_entry is not None:
            self._remove_entry(self._pending_entry)
            self._pending_entry = None

    # ------------------------------------------------------------------
    # layout

    def get_css_variables(self) -> dict[str, str]:
        return {**super().get_css_variables(), **theme.CSS_VARIABLES}

    def compose(self) -> ComposeResult:
        yield ChatFlowView(
            model=self._flow_model,
            presenter=self._presenter,
            id="chat-log",
            anchor=Anchor.STICKY_BOTTOM,
            spacing=0,
            overscan=8,
            estimated_height=2,
        )
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
        if self.client is None:
            # Connect inside the app's event loop: a client created under
            # another asyncio.run() would carry a closed loop and break on
            # first use under Textual.
            try:
                self.client = await XarnessClient.connect_unix()
            except ClientError as exc:
                self._error(f"could not connect to the host: {exc}")
                return
        self.query_one("#chat-input", ChatInput).focus()
        self._refresh_bars()
        self._set_default_title()
        # Follow new output only while the user is parked at the bottom:
        # FlowView's STICKY_BOTTOM anchor does exactly that (released on a
        # scroll-up gesture, re-armed at the tail).
        self._load_draft_defaults()
        for notice in self.startup_notices:
            self._notice(notice)
        if self.session_name is not None:
            await self._attach(self.session_name)

    async def on_unmount(self) -> None:
        if self._command_worker is not None:
            self._command_worker.cancel()
            self._command_worker = None
        if self.client is not None:
            await self.client.aclose()

    # ------------------------------------------------------------------
    # Attach + event pump

    async def _attach(self, name: str) -> None:
        """Attach to a hosted session; the stream's first frame is its
        snapshot, so there is no separate GET to race with it."""
        if self._pump_worker is not None:
            self._pump_worker.cancel()
            self._pump_worker = None
        self.session_name = name
        self._pump_worker = self._pump_events(name)

    @work(group="stream", exit_on_error=False)
    async def _pump_events(self, name: str) -> None:
        """Consume the session's event stream, reconnecting with backoff.

        Every reconnect delivers a fresh Snapshot (full re-render). Never
        crashes the app on connection loss."""
        try:
            async for item in self.client.attach(name):
                if isinstance(item, Snapshot):
                    self.status = item.status
                    self._set_vision(item.status.supports_vision)
                    self._refresh_title()
                    await self._render_history(item)
                    self._refresh_bars()
                    if item.status.finished_while_away:
                        self._notice("finished while you were away")
                        self._set_title_state("finished")
                    await self.refresh_diff_summary()
                    self._sync_ask()
                    self._maybe_prompt_drift()
                else:
                    await self._handle_event(item)
        except asyncio.CancelledError:
            raise
        except (ClientError, httpx.HTTPError) as exc:
            self.call_later(self._recover, name, exc)
        except Exception as exc:  # noqa: BLE001 - never crash on connection loss
            self.call_later(self._recover, name, exc)

    async def _recover(self, name: str, exc: Exception) -> None:
        """Connection loss: the one recovery path. Respawn the host if it
        died, then reattach; retry with backoff until it succeeds or the user
        switches/detaches away from this session. The stream endpoint hosts a
        saved session on demand, so there is no separate ``load_session``
        round-trip."""
        self._error(f"connection lost: {exc} — reconnecting")
        backoff = RECONNECT_MIN
        while self.session_name == name:
            try:
                await ensure_host()
                await self._attach(name)
                return
            except (ClientError, httpx.HTTPError):
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_MAX)

    async def _handle_event(self, event: Event) -> None:
        """Route one live event to the widgets."""
        if isinstance(event, StatusChanged):
            self.status = event.status
            self._set_vision(event.status.supports_vision)
            self._refresh_bars()
            self._sync_refs_screen()
            self._sync_ask()
            # Spawned, not awaited: it is a worker (push_screen_wait needs
            # one) and must not block the event pump on its modal.
            self._maybe_prompt_drift()
            return
        if isinstance(event, Notice):
            if event.level == "error":
                self._error(event.text)
            elif event.level == "warn":
                self._line(event.text, kind="warn")
            else:
                self._notice(event.text)
            return
        if isinstance(event, UserMessageAdded):
            self._sync_entries(self._transcript.apply(event))
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
        if isinstance(event, DiffChanged):
            await self._show_diff_summary(event.summary)
            return
        if isinstance(event, CompactionDone):
            self._notice(self._compaction_text(event))
            return
        if isinstance(event, RollbackApplied):
            self._notice(self._rollback_text(event))
            return
        if isinstance(event, DriftDetected):
            # The prompt is session state (see _maybe_prompt_drift); the
            # status carries it. No separate UI action needed.
            return
        if isinstance(event, DriftResolved):
            self._drift_modal_open = False
            # The prompt is resolved: drop the pending flag locally too, or
            # the next StatusChanged would reopen the modal for a drift that
            # no longer exists.
            if self.status is not None:
                self.status.pending_drift = None
            return
        if isinstance(event, TurnEnded):
            # The turn is fully unwound: settle anything still live and stop
            # the shimmer/status without inferring it from a status change.
            self._sync_entries(self._transcript.cancel())
            self._phase_idle()
            self._clear_hint_lines()
            if event.outcome == "finished":
                self._set_title_state("finished")
            else:
                self._set_default_title()
            return
        if isinstance(event, TurnFinished):
            self._notice(f"Worked for {_format_duration(event.elapsed)}")
            self._set_title_state("finished")
            return
        if isinstance(event, SessionDeleted):
            # The session went away (deleted, or closed by this or another
            # client — a Notice with the specifics precedes this event).
            self._detach()
            return
        if isinstance(event, ProcessingStarted):
            # A round opened and is waiting on the provider: the strip goes
            # up (this is the start of a turn AND every post-tools round).
            # The reducer opens the live round; there is no app-side reset.
            self._sync_entries(self._transcript.apply(event))
            self._phase_pending()
            # Clear a stale "finished" suffix as soon as the next turn starts.
            self._set_default_title()
            return
        # Everything else is a streaming-round event: the reducer folds it and
        # reports the items it created or mutated.
        if isinstance(event, (ReasoningDelta, ContentDelta, ToolCallStarted)):
            self._phase_streaming()
        self._sync_entries(self._transcript.apply(event))
        if isinstance(event, TurnComplete):
            self._complete_round(event)
        elif isinstance(event, StreamError):
            self._phase_idle()
            self._error(event.message)
            self._clear_hint_lines()

    @staticmethod
    def _compaction_text(event: CompactionDone) -> str:
        """Render a compaction outcome (counts + model summary) as one line."""
        if not event.before:
            if event.automatic:
                return "auto-compacted: nothing to compact yet"
            return event.summary or "nothing to compact yet"
        counts = (
            f"{event.before:,} → {event.after:,} tokens "
            f"(freed {max(0, event.before - event.after):,})"
        )
        prefix = "auto-compacted" if event.automatic else "compacted"
        return f"{prefix}: {counts}\n{event.summary}"

    @staticmethod
    def _rollback_text(event: RollbackApplied) -> str:
        """Render an undo/retry outcome from the typed fields."""
        if event.compaction_undone:
            text = "compaction undone: pre-compaction history restored"
            if event.resend:
                text += " — nothing to retry (the turn is still in the conversation)"
            return text
        note = {
            "reverted": "file edits reverted",
            "no_edits": "no file edits to revert",
            "no_tracking": (
                "no git tracking for this session — file edits could not be reverted"
            ),
        }[event.note]
        if event.resend:
            return f"retrying: {event.scope} rolled back ({note})"
        return f"undone: {event.scope} removed ({note})"

    # ------------------------------------------------------------------
    # Commands

    @contextmanager
    def _reporting(self, prefix: str = ""):
        """Turn a ClientError raised inside the block into an error line in
        the transcript (prefixed)."""
        try:
            yield
        except ClientError as exc:
            self._error(f"{prefix}{exc}")

    async def _send_command(self, command: Command) -> CommandPayload | None:
        """Send one command to the session; returns its result payload.

        Errors (busy rejection, unknown target, git failure) surface as an
        error line — the session broadcast the effects to everyone anyway."""
        if self.session_name is None:
            return None
        try:
            result = await self.client.command(self.session_name, command)
        except ClientError as exc:
            self._error(str(exc))
            return None
        return result.payload

    def _post_command(self, command: Command) -> None:
        """Queue a fire-and-forget command: results are ignored (the effects
        arrive as events / StatusChanged anyway). One worker drains the queue
        in order, so commands cannot overtake each other at the server."""
        self._command_queue.put_nowait(command)
        if self._command_worker is None or self._command_worker.is_finished:
            self._command_worker = self.run_worker(
                self._drain_commands(), group="command-queue", exclusive=True,
            )

    async def _drain_commands(self) -> None:
        while True:
            command = await self._command_queue.get()
            try:
                await self._send_command(command)
            finally:
                self._command_queue.task_done()

    # ------------------------------------------------------------------
    # Terminal title

    def _set_terminal_title(self, title: str) -> None:
        """Write an OSC 0 sequence so the terminal tab/window shows `title`."""
        driver = self._driver
        if driver is None:
            return
        driver.write(f"\x1b]0;{title}\x07")
        driver.flush()

    def _title_for(self, path: str) -> str:
        """The workspace's own directory name for the terminal title.

        The status bar can afford a cropped path; a tab title cannot, and
        two workspaces named ``xarness`` under different parents must read
        the same (``xarness xarness``, not ``xarness projects/xarness``)."""
        name = Path(path.rstrip("/")).name
        return name or path

    def _base_title(self) -> str:
        """`xarness <workdir>` for the attached session, or the local draft."""
        workspace = self.view_status.workspace if self.session_name is not None else None
        path = workspace or str(self.workspace or Path.cwd())
        return f"xarness {self._title_for(path)}"

    def _set_default_title(self) -> None:
        """Default title: `xarness <workdir>` (no activity suffix)."""
        self._set_terminal_title(self._base_title())

    def _set_title_state(self, state: str) -> None:
        """`xarness <workdir> - <state>` (finished / ask), always derived from
        the current base so states never stack up."""
        self._set_terminal_title(f"{self._base_title()} - {state}")

    def _refresh_title(self) -> None:
        """Follow the attached session: its workspace in the tab title, or
        the default when detached — otherwise switching to a session in a
        different workspace keeps the stale title."""
        self._set_default_title()

    # ------------------------------------------------------------------
    # Status bar / bars

    def _refresh_bars(self) -> None:
        """Refresh everything driven by ``self.status``: the status bar, the
        generating bar and the steer queue."""
        self._refresh_status()
        self._refresh_steer_bar()
        self._refresh_generating()

    def _refresh_status(self) -> None:
        status = self.view_status
        self.query_one("#status-bar", StatusBar).update_status(
            display_name=status.model_display_name,
            effort=status.cot_strength,
            mode=status.mode,
            total_in=status.total_in,
            total_out=status.total_out,
            context_used=status.context_used,
            max_context=status.max_context,
            workspace=status.workspace,
        )

    @property
    def view_status(self) -> StatusPayload:
        """The status the UI shows: the real session status, or — for a draft
        chat — one synthesized from ``/config`` defaults plus the pre-session
        model/mode choices. Never ``None``, so view code has no fake-status
        branch and ``session_name``/``supports_vision`` need not be mirrored."""
        if self.status is not None:
            return self.status
        config = self._config_cache
        target = self.model or (config.default_model if config else None)
        entry = next(
            (m for m in (config.models if config else []) if m.get("name") == target),
            None,
        )
        if entry is not None:
            display_name = entry.get("display_name") or entry.get("name") or "—"
            effort = entry.get("cot_strength") or "medium"
            max_context = entry.get("max_context") or 128_000
        else:
            display_name = (
                config.default_model_display_name if config else None
            ) or "—"
            effort = "medium"
            max_context = 128_000
        return StatusPayload(
            state="idle",
            mode=self._draft_mode,
            model_name=target or "",
            model_display_name=display_name,
            cot_strength=effort,
            max_context=max_context,
            workspace=str(self.workspace) if self.workspace else None,
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
        except ClientError:
            return
        if self.session_name is None:
            self._sync_vision_from_config(self.model)
            self._refresh_status()

    def _set_vision(self, supported: bool) -> None:
        try:
            self.query_one("#chat-input", ChatInput).supports_vision = supported
        except NoMatches:
            pass

    def _sync_vision_from_config(self, model: str | None = None) -> None:
        config = self._config_cache
        if config is None:
            return
        name = model or config.default_model
        entry = next((m for m in config.models if m.get("name") == name), None)
        if entry is not None:
            self._set_vision(bool(entry.get("supports_vision")))

    def on_chat_input_image_refused(
        self, event: ChatInput.ImageRefused,
    ) -> None:
        self._line(
            f"{event.name}: the active model does not support image input "
            "(supports_vision is false in the config)",
            kind="warn",
        )

    def _refresh_generating(self) -> None:
        try:
            bar = self.query_one("#generating-bar", GeneratingBar)
        except NoMatches:
            return
        state = self.view_status.state
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
        queued = self.view_status.queued
        bar.update_items([q.text for q in queued])

    async def refresh_diff_summary(self) -> None:
        """Recompute the pending-changes bar (hidden when empty or gitless).
        Live updates normally arrive as ``DiffChanged``; this is the attach/
        history-replaced path."""
        if self.session_name is None:
            await self._show_diff_summary(None)
            return
        try:
            info = await self.client.diff(self.session_name)
        except ClientError:
            info = None  # tracking broken (e.g. repo gone): stay quiet
        await self._show_diff_summary(info)

    async def _show_diff_summary(self, info) -> None:
        """Render one DiffInfo (or clear when empty/None)."""
        try:
            widget = self.query_one("#diff-summary", DiffSummary)
        except NoMatches:
            return  # app shutting down; DOM already pruned
        if info is None or not info.files:
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
        # The user is acting: follow the output their submission produces
        # (their own message, or /tools-style command output landing at the
        # bottom). Without this, a log scrolled up to read stays parked even
        # though the sticky anchor re-arms only at the bottom.
        self._flow_view().scroll_to_bottom(animate=False)
        if text.startswith("/"):
            self._handle_slash_command(text)
            return
        self._submit(text, event.images)

    def _submit(self, text: str, images: list[ImageAttachment] | None = None) -> None:
        """Send a message. No local echo: the message lands in the transcript
        when UserMessageAdded arrives, identically for every client."""
        self.call_later(self._submit_async, text, images)

    async def _ensure_session(self) -> bool:
        """Create the draft's session if it does not exist yet. Returns
        False when the session could not be started (error already shown)."""
        if self.session_name is not None:
            return True
        try:
            summary = await self.client.create_session(CreateSessionRequest(
                workspace=str(self.workspace) if self.workspace else None,
                model=self.model,
                sandbox=self.sandbox_name,
                preset=self.preset,
                network=self.network,
                gpu=self.gpu,
                no_init_repo=self.no_init_repo,
                no_session=self.no_session,
            ))
        except ClientError as exc:
            self._error(f"could not start a session: {exc}")
            return False
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
        return True

    async def _submit_async(self, text: str, images: list[ImageAttachment] | None) -> None:
        # Draft chat: the session comes into existence on the server with
        # the first message.
        if self.session_name is None and not await self._ensure_session():
            return
        # Pasted/dragged images ride along as base64 attachments; SendMessage
        # and UserMessageAdded carry the same ImageAttachment shape the input
        # produced, so they pass through unchanged.
        await self._send_command(SendMessage(text=text, images=images or None))

    # ------------------------------------------------------------------
    # Ask tool (event-driven)

    @work(group="ask", exclusive=True)
    async def _ask_flow(self, ask_id: str, questions: list[str]) -> None:
        """Show the questions one at a time in the AskBar; the user answers
        through the normal chat input (Enter submits, Esc skips all). The
        collected answers go back as one ``answer_ask`` command — any client
        may answer; first answer wins."""
        self._set_title_state("ask")
        self._ask_id = ask_id
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
            self._set_default_title()

    def _on_ask_resolved(self, event: AskResolved) -> None:
        """Another client answered first: close this client's prompt.
        Cancelling the worker unwinds the awaited future into its finally
        block, which hides the ask bar and restores the input."""
        if self._ask_id is not None and self._ask_id == event.ask_id:
            self._close_ask()

    def _close_ask(self) -> None:
        self._ask_id = None
        if self._ask_worker is not None:
            self._ask_worker.cancel()
            self._ask_worker = None

    def _sync_ask(self) -> None:
        """Reconcile the local ask prompt with the session's ``pending_ask``
        state: open it on attach (or when a status update reveals it) and
        close it when the session no longer has one. This keeps
        ``AskResolved`` from being the only way a stale prompt closes."""
        pending = self.view_status.pending_ask
        if pending is None:
            if self._ask_id is not None:
                self._close_ask()
            return
        if self._ask_id == pending.ask_id:
            return
        self._ask_worker = self._ask_flow(pending.ask_id, pending.questions)

    # ------------------------------------------------------------------
    # Drift prompt (session state; any client may resolve)

    @work(group="drift-prompt", exclusive=True)
    async def _maybe_prompt_drift(self) -> None:
        """The workspace changed outside this session while it was closed.

        Show what drifted and let the user choose: accept it as the new
        baseline, or keep tracking against the last known state. Never pick
        for them. The prompt is session state — if another client resolves
        it first, DriftResolved closes this modal's follow-up.

        A worker, not an await inside the event pump: push_screen_wait
        requires one, and blocking the pump on a modal would stall the
        event stream while the prompt is open."""
        if (
            self.status is None or self.status.pending_drift is None
            or self._drift_modal_open
        ):
            return
        self._drift_modal_open = True
        detail = self._drift_detail(self.status.pending_drift)
        choice = await self.push_screen_wait(ConfirmScreen(
            "Workspace changed outside this session since it was last open:",
            detail=detail,
            confirm_label="Accept as baseline",
            cancel_label="Keep tracking",
        ))
        self._drift_modal_open = False
        if self.status is None or self.status.pending_drift is None:
            return  # resolved elsewhere while the modal was open
        if choice is None:
            return  # dismissed: leave the prompt pending
        await self._send_command(ResolveDrift(accept=bool(choice)))

    @staticmethod
    def _drift_detail(pending) -> str:
        """Render the structured drift state for the confirmation modal."""
        files = pending.files or []
        if files:
            shown = ", ".join(files[:8])
            if len(files) > 8:
                shown += f" (+{len(files) - 8} more)"
        else:
            shown = "(paths unavailable)"
        return (
            "the workspace changed while this session was closed "
            f"(last known tree {pending.last_known[:7]}, now "
            f"{pending.current[:7]})\n\n{shown}\n\n"
            "accept to make the workspace the new baseline, or keep "
            "tracking against the last known state"
        )

    # ------------------------------------------------------------------
    # Slash commands

    def _handle_slash_command(self, text: str) -> None:
        parts = text[1:].split(maxsplit=1)
        cmd = parts[0] if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""
        entry = COMMANDS.get(cmd)
        if entry is None:
            self._error(f"unknown command: /{cmd}")
            return
        entry.handler(self, arg)

    def _switch_mode(self) -> None:
        """Toggle plan/write mode; shared by /mode and the Tab shortcut.
        With no session attached the toggle is local draft state, applied to
        the session the moment the first message creates it."""
        if self.session_name is None:
            self._draft_mode = "plan" if self._draft_mode == "write" else "write"
            self._refresh_status()
            return
        current = self.view_status.mode
        new_mode = "plan" if current == "write" else "write"
        self._post_command(SetMode(mode=new_mode))

    async def _run_undo_to(self, message_id: str) -> None:
        """Click-to-undo: drop the clicked user message and everything after
        it (file edits reverted), putting that message back in the input."""
        payload = await self._send_command(UndoTo(message_id=message_id))
        self._put_back_user_text(payload)

    async def _run_undo(self) -> None:
        self._put_back_user_text(await self._send_command(Undo()))

    def _put_back_user_text(self, payload: CommandPayload | None) -> None:
        """Restore an undone message's text and images to the input box."""
        if not isinstance(payload, RollbackResult) or not payload.user_text:
            return
        chat_input = self.query_one("#chat-input", ChatInput)
        chat_input.restore(payload.user_text, list(payload.images or ()))
        chat_input.focus()

    def on_flow_view_clicked(self, event: FlowView.Clicked) -> None:
        """In-transcript clicks: expand/collapse tool calls and thinking
        blocks, arm/trigger the user message's undo affordance."""
        entry, item = event.entry, event.entry.item
        if item.handle_click(self):
            entry.update()

    def _start_new_session(self) -> None:
        """Slash /new: leave the current session (it keeps running in the
        host) and return to a draft chat."""
        self._detach()
        self._notice(
            "started a fresh chat — the previous session keeps running in "
            "the background (/sessions to go back)"
        )

    def _detach(self) -> None:
        """Return to a draft chat: no session attached, empty log."""
        if self._pump_worker is not None:
            self._pump_worker.cancel()
            self._pump_worker = None
        self.session_name = None
        self.status = None
        self._refresh_title()
        self._flow_model.clear()
        self._hint_entries = []
        self._refresh_bars()
        self.call_later(self.refresh_diff_summary)

    async def _delete_async(self) -> None:
        name = self.session_name
        self._detach()
        if name is None:
            self._notice("no session to delete")
            return
        with self._reporting("/delete failed: "):
            await self.client.delete_session(name)
        self._notice(
            "deleted: saved session removed — started a fresh session; "
            "your files are untouched"
        )

    async def _open_sessions_async(self) -> None:
        entries = []
        with self._reporting("/sessions failed: "):
            entries = await self.client.sessions()
        if not entries:
            return
        self.push_screen(
            ListPicker(
                session_options(
                    entries,
                    str(self.workspace) if self.workspace else None,
                ),
                title="Open session", filterable=True,
            ),
            self._on_session_selected,
        )

    async def _on_session_selected(self, name: str | None) -> None:
        if not name:
            return
        if name == self.session_name:
            return
        # No explicit load: the stream endpoint hosts a saved session on
        # demand, and the first frame is its snapshot.
        with self._reporting(f"could not open {name}: "):
            await self._attach(name)

    async def _pick(
        self, names: list[str], *, current: str | None, title: str,
    ) -> str | None:
        """Show a picker screen and wait for the choice (None = dismissed).

        Awaited only from within a worker: push_screen_wait requires an
        active worker context."""
        return await self.push_screen_wait(
            ListPicker(name_options(names, current), title=title)
        )

    @work(group="picker", exclusive=True)
    async def _pick_model(self) -> None:
        with self._reporting("/model failed: "):
            self._config_cache = await self.client.config()
        if self._config_cache is None:
            return
        name = await self._pick(
            [m["name"] for m in self._config_cache.models],
            current=self.view_status.model_name or None,
            title="Switch model",
        )
        if not name:
            return
        if self.session_name is None:
            # Draft chat: there is no session to switch yet. Remember the
            # choice (the first message passes it to create_session) and
            # repaint the status bar now, so the new model is visible
            # before the user sends anything.
            self.model = name
            self._sync_vision_from_config(name)
            self._refresh_status()
            # No session yet: the host still owns the config file.
            try:
                await self.client.set_default_profile(name)
            except ClientError as exc:
                self._error(f"could not save default model: {exc}")
        else:
            self._post_command(SwitchModel(name=name, persist_default=True))
        # The choice is now the config's default for the next session, so
        # keep the cached config in step: a later /new (which drops back to
        # a draft) read its status from here.
        config = self._config_cache
        if config is not None:
            entry = next(
                (m for m in config.models if m.get("name") == name), None,
            )
            config.default_model = name
            config.default_model_display_name = (
                (entry or {}).get("display_name") or name
            )
            self._refresh_status()

    @work(group="picker", exclusive=True)
    async def _pick_sandbox(self) -> None:
        with self._reporting("/sandbox failed: "):
            self._config_cache = await self.client.config()
        if self._config_cache is None:
            return
        name = await self._pick(
            [s["name"] for s in self._config_cache.sandboxes],
            current=self.view_status.sandbox_name,
            title="Sandbox profile (applies when the session is idle)",
        )
        if name:
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
            filters = self.view_status.filter_list
            if filters:
                self._notice(
                    "filtered out of the sandbox mounts: " + ", ".join(filters)
                )
            else:
                self._notice("no paths filtered out")
        else:
            self._error("usage: /filter add <path> | /filter remove <path> | /filter list")

    # ------------------------------------------------------------------
    # External references (per workspace)

    def _handle_refs_command(self, arg: str) -> None:
        if arg:
            self._error("usage: /refs")
        else:
            self._open_refs()

    @work(group="refs", exit_on_error=False)
    async def _open_refs(self) -> None:
        """Open the manager from the workspace, without creating a session."""
        workspace = self.view_status.workspace or str(Path.cwd())
        if not workspace:
            self._error("no workspace to manage references for")
            return
        try:
            refs = await self.client.workspace_refs(workspace)
        except ClientError as exc:
            self._error(str(exc))
            return
        self.push_screen(RefsScreen(refs))

    def on_refs_screen_add_ref(self, event: RefsScreen.AddRef) -> None:
        self._ref_update(True, event.path)

    def on_refs_screen_remove_ref(self, event: RefsScreen.RemoveRef) -> None:
        self._ref_update(False, event.path)

    def _sync_refs_screen(self) -> None:
        """Keep an open RefsScreen in step with the session's status."""
        if isinstance(self.screen, RefsScreen):
            self.screen.post_message(
                RefsScreen.RefsChanged(list(self.view_status.refs))
            )

    @work(group="refs", exit_on_error=False)
    async def _ref_update(self, add: bool, path: str) -> None:
        workspace = self.view_status.workspace or str(Path.cwd())
        if not workspace:
            self._error("no workspace to manage references for")
            return
        try:
            refs = await self.client.update_workspace_refs(workspace, path, add)
        except ClientError as exc:
            self._error(str(exc))
            return
        if self.status is not None:
            self.status.refs = refs
            self._refresh_bars()
        if isinstance(self.screen, RefsScreen):
            self.screen.post_message(RefsScreen.RefsChanged(refs))

    # ------------------------------------------------------------------
    # Theme (client-side preference)

    @work(group="picker", exclusive=True)
    async def _pick_theme(self) -> None:
        name = await self._pick(
            list(theme.THEMES), current=theme.CURRENT_THEME, title="Color theme",
        )
        if name:
            self._handle_theme_command(name)

    def _handle_theme_command(self, arg: str) -> None:
        """Apply an explicitly named theme, or report the available ones."""
        if arg not in theme.THEMES:
            self._error(f"unknown theme '{arg}'; available: {', '.join(theme.THEMES)}")
            return
        self._apply_theme(arg)
        self._save_preference(default_theme=arg)

    def _save_preference(self, *, default_theme: str) -> None:
        """Write the client-side theme choice to the config file.
        Failures are reported, never fatal."""
        if self.config_path is None:
            return
        try:
            save_preferences(
                self.config_path, default_theme=default_theme,
            )
        except Exception as exc:  # noqa: BLE001 - preference save is best-effort
            self._notice(f"note: could not save preference: {exc}")

    def _apply_theme(self, name: str) -> None:
        """Switch palette live: CSS variables, input caret, shimmers, dots,
        summaries, status bar — anything that baked palette colors at render
        time must be re-rendered here, or the switched theme drifts from the
        same theme set at startup.

        The transcript recolors in place: ``_presenter.invalidate()`` busts
        the per-item strip caches (every cache key carries the presenter
        epoch), and bumping each entry's revision makes FlowView re-present
        it — on screen now, or lazily when scrolled back into view. This is
        deliberately NOT a rebuild from the last snapshot: a snapshot only
        covers what was persisted at attach, so a rebuild would wipe every
        entry that arrived via live events (the current turn's user message,
        cot, tool and answer blocks). Working in place also means the switch
        needs no idle gate — it is safe mid-turn."""
        theme.set_theme(name)
        self.refresh_css()
        self.query_one("#chat-input", ChatInput).apply_input_theme()
        self._presenter.invalidate()
        self._refresh_status()
        self.query_one("#diff-summary", DiffSummary).recolor()
        # Recolor every live entry in place. Chunked: each update() can kick
        # off present/layout/scroll work, and hundreds of them back-to-back
        # in one handler stack is how the paint/present recursion blew the
        # recursion limit (which is what pinned entries to stale strips).
        entries = [e for e in self._flow_model.entries if e.alive]
        if entries:
            self._recolor_chunk(entries)

    def _recolor_chunk(self, entries: list) -> None:
        """Bump a batch of entries' revisions, then defer the rest."""
        for entry in entries[:16]:
            entry.update()
        rest = entries[16:]
        if rest:
            self.call_after_refresh(self._recolor_chunk, rest)

    # ------------------------------------------------------------------
    # Transcript (the FlowView model)

    def _flow_view(self) -> "FlowView":
        return self.query_one("#chat-log", FlowView)

    def _append_item(self, item):
        """Append one transcript item to the FlowModel, registering shimmer
        repaints for the items that have one."""
        entry = self._flow_model.append(item)
        if item.streaming():
            self._start_anim(entry, item)
        return entry

    def _register_item(self, item):
        entry = self._append_item(item)
        self._entries[id(item)] = entry
        return entry

    def _register_items(self, items) -> None:
        """Register a whole rendered history in one FlowModel operation.

        Appending one item at a time makes FlowView rebuild its entry index
        on every insert — O(n²) on a long session. ``extend`` sends a single
        insert-many notification, so loading a large transcript stays O(n)."""
        items = list(items)
        if not items:
            return
        entries = self._flow_model.extend(items)
        for item, entry in zip(items, entries):
            self._entries[id(item)] = entry
            if item.streaming():
                self._start_anim(entry, item)

    def _sync_entries(self, items) -> None:
        """Create entries for new items; repaint the rest (and stop their
        shimmer once they settle)."""
        for item in items:
            entry = self._entries.get(id(item))
            if entry is None:
                self._register_item(item)
            else:
                if not item.streaming():
                    self._stop_anim(entry)
                entry.update()

    def _remove_entry(self, entry) -> None:
        self._stop_anim(entry)
        if entry.alive:
            entry.remove()

    def _start_anim(self, entry, item) -> None:
        """Repaint the shimmering entry while on screen (FlowView pauses the
        timer offscreen). Phase comes from the shared clock at render time
        (flow.shimmer_now), so the tick only invalidates."""
        try:
            view = self._flow_view()
        except NoMatches:
            return  # app shutting down / not mounted yet

        def tick(e) -> None:
            if not item.streaming():
                # Completed (or settled by some path that missed its stop):
                # kill the animation here, like the interrupt path does — a
                # leaked tick must never keep a finished cot/tool shimmering.
                self._stop_anim(entry)
                e.update()
                return
            e.update()

        self._anim[id(entry)] = view.animate_entry(entry, 0.08, tick)

    def _stop_anim(self, entry) -> None:
        handle = self._anim.pop(id(entry), None)
        if handle is not None:
            handle.stop()

    def _line(self, text: str, kind: str = "info", hint: bool = False) -> None:
        """Append a one-line status message (notice/warn/error/hint) to the
        transcript. Whether it is in view is FlowView's business: it follows
        the tail only if the user is already at the bottom."""
        self._flow_view()  # raises NoMatches when the app is shutting down
        entry = self._flow_model.append(LineItem(text=text, kind=kind))
        if hint:
            self._hint_entries.append(entry)

    def _error(self, text: str) -> None:
        self._line(text, kind="error")

    def _notice(self, text: str) -> None:
        self._line(text)

    def _clear_hint_lines(self) -> None:
        """Remove transient hint lines (e.g. 'press esc again to interrupt')
        from the transcript. Called when the assistant's message settles, so
        keypress coaching doesn't linger in the scrollback."""
        for entry in self._hint_entries:
            self._remove_entry(entry)
        self._hint_entries = []

    def _complete_round(self, event: TurnComplete) -> None:
        self._clear_hint_lines()
        if (self._transcript.assistant is None
                and self._transcript.thinking is None
                and not event.has_tool_calls):
            self._error("empty response from model")
        # The round is over: clear the strip even when tool calls follow. The
        # strip means "waiting on the provider", not "the turn is busy"; while
        # the tools execute there is no LLM request in flight. The next round's
        # ProcessingStarted (after the tools finish) puts it back up.
        self._phase_idle()
        self._refresh_status()

    async def _on_history_replaced(self, event: HistoryReplaced) -> None:
        """The conversation was rewritten (undo, retry, accept, compaction,
        drift accept): re-render from the new snapshot. A data-only rebuild
        is cheap under FlowView, so there is no incremental trim path."""
        # The snapshot also carries the fresh status (a drift resolve clears
        # pending_drift there); adopt it so local prompts/bars don't keep
        # showing state the host already moved past.
        self.status = event.snapshot.status
        self._refresh_bars()
        await self._render_history(event.snapshot)
        self._refresh_status()
        self.call_later(self.refresh_diff_summary)

    async def _render_history(self, snap: Snapshot) -> None:
        """Rebuild the transcript from a snapshot via the pure reducer.

        Tool-call status is rebuilt from the persisted ``ok`` flag on each
        tool result, so a reload shows the same success/failure state. A
        compacted conversation keeps the pre-compaction messages, then a
        divider, then the summary — mirroring what the model sees. The
        snapshot's ``partial_round`` renders the in-flight round, so a
        mid-turn attach needs no event replay."""
        self._flow_model.clear()
        self._hint_entries = []
        self._entries = {}
        self._anim = {}
        self._presenter.reset()
        self._register_items(self._transcript.load(snap))
        try:
            self._flow_view().scroll_to_bottom(animate=False)
        except NoMatches:
            pass

    # ------------------------------------------------------------------
    # Popups (@ search runs server-side against the session's workspace)

    def on_chat_input_mode_toggle(self, event: ChatInput.ModeToggle) -> None:
        self._switch_mode()

    def on_chat_input_slash_query(self, event: ChatInput.SlashQuery) -> None:
        q = event.query.lower()
        # Table-style rows: commands padded to a shared column, descriptions
        # dimmed — same layout language as the session list.
        width = max(len(name) for name in COMMANDS)
        matches = [
            (
                name,
                f"/{name:<{width}}  [dim]{COMMANDS[name].description}[/]",
            )
            for name in COMMANDS
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
        chat_input = self.query_one("#chat-input", ChatInput)
        was_slash = chat_input.popup_active == "slash"
        self._accept_popup_value(popup.selected_value, was_slash)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Mouse click on a popup option: insert it immediately, keep focus on input."""
        event.stop()
        chat_input = self.query_one("#chat-input", ChatInput)
        was_slash = chat_input.popup_active == "slash"
        self._accept_popup_value(event.option.id, was_slash)

    def _accept_popup_value(self, value: str | None, was_slash: bool) -> None:
        """Complete the confirmed/clicked popup value: slash commands take no
        arguments, so they complete and run in one Enter; @-mentions are
        inserted at the cursor."""
        chat_input = self.query_one("#chat-input", ChatInput)
        chat_input.popup_active = None
        self._hide_popup()
        chat_input.focus()
        if value is None:
            return
        if was_slash:
            command = COMMANDS.get(value)
            if command is not None and command.takes_arg:
                # If the user typed this command (any prefix of it) and hit
                # Enter, run it right away — its empty-arg behavior is the
                # picker/list view, never a destructive default. Only when
                # the highlighted command is NOT what was typed (arrow
                # navigation to a different command) do we just complete the
                # word and let the user type the argument.
                typed = chat_input.text.strip()
                if typed.startswith("/") and f"/{value}".startswith(typed):
                    chat_input.load_text("")
                    self._handle_slash_command(f"/{value}")
                    return
                chat_input.load_text(f"/{value} ")
                return
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

    # ------------------------------------------------------------------
    # Steer queue commands

    def action_steer_discard(self) -> None:
        """ctrl+shift+backspace: discard the oldest queued steer message."""
        if not self.view_status.queued:
            return
        self._post_command(DiscardQueued())

    def _flush_queued_now(self) -> None:
        """Enter on an empty input while messages are queued: send now.

        Cancels the running turn; the session starts the queued message
        immediately instead of waiting for the next round boundary."""
        status = self.view_status
        if status.queued and status.state == "running":
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
        state = self.view_status.state
        has_pending_ask = bool(self.view_status.pending_ask)
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
            choice = await self.push_screen_wait(ConfirmScreen(
                "A request is still running. Continue it in the background?",
                confirm_label="Continue headless",
                cancel_label="Stop and quit",
            ))
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
                except ClientError:
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
                self._line("press esc again to interrupt", kind="warn", hint=True)
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
            self._error(
                "/diff: no session attached — send a message first"
            )
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
                    self._notice("no pending changes to diff")
                    return
                widget.add_class("expanded")
                return
            text = (await self.client.diff(self.session_name, path=arg)).text or ""
            if not text.strip():
                self._notice(f"no pending changes for {arg}")
                return
            # Rebuild the row list first (a previous /diff toggle may have
            # cleared it) so the active-row highlight has something to attach to.
            await self.refresh_diff_summary()
            widget.show_diff(arg, text)
        except ClientError as exc:
            widget.hide_diff()
            self._error(f"/diff failed: {exc}")

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
            text = (await self.client.diff(
                self.session_name, path=event.key)).text or ""
        except ClientError as exc:
            widget.hide_diff()
            self._error(f"diff failed: {exc}")
            return
        widget.show_diff(event.key, text)

    # ------------------------------------------------------------------