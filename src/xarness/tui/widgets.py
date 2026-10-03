"""Widgets for the xarness chat TUI.

Widgets are dumb views: they render what the app feeds them from stream
events and hold no conversation state of their own (the reasoning text inside
``ThinkingBlock`` is display state only; the authoritative copy lives in
``ChatController.conversation``).
"""

from __future__ import annotations

import difflib
import json
import random
import re
import time
from typing import ClassVar, cast

from rich.console import Group, RenderableType
from rich.style import Style
from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.css.query import NoMatches
from textual.highlight import highlight
from textual.message import Message
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Markdown, OptionList, Static, TextArea
from textual.widgets.markdown import MarkdownFence
from textual.widgets.option_list import Option
from textual.widgets.text_area import TextAreaTheme

from .. import theme
from ..conversation import Message as ConversationMessage
from ..events import ToolCallStatus
from ..gitwork import DiffStat


_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_SLASH_RE = re.compile(r"^/(\w*)$")
_AT_RE = re.compile(r"(?:^|\s)@(\S*)$")


def _split_settled(text: str) -> tuple[str, str]:
    """Split text at the end of the last complete fenced code block.

    Returns (settled, remainder). `settled` is everything up through the
    last fully-closed ``` ... ``` pair (including any plain text before it);
    `remainder` is whatever comes after — plain text and/or a fence that's
    still open. Multiple complete fences in one chunk are all included in
    `settled` in one pass via finditer's non-overlapping matches.

    Known limitation: naive backtick pairing, not a real markdown parser —
    fine for normal LLM output, could mis-pair on unusual nested backtick
    usage inside a fence body.
    """
    last_end = 0
    for match in _FENCE_RE.finditer(text):
        last_end = match.end()
    return text[:last_end], text[last_end:]

def _fmt_tokens(count: int) -> str:
    if count < 1_000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1_000:.1f}k"
    return f"{count / 1_000_000:.2f}M"


def _format_duration(seconds: float) -> str:
    """Human duration: 5.3s · 42s · 2m 15s · 1h 04m."""
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _lerp_hex(c1: str, c2: str, t: float) -> str:
    c1, c2 = c1.lstrip("#"), c2.lstrip("#")
    r1, g1, b1 = int(c1[0:2], 16), int(c1[2:4], 16), int(c1[4:6], 16)
    r2, g2, b2 = int(c2[0:2], 16), int(c2[2:4], 16), int(c2[4:6], 16)
    r = round(r1 + (r2 - r1) * t)
    g = round(g1 + (g2 - g1) * t)
    b = round(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


class PaletteFence(MarkdownFence):
    """MarkdownFence that highlights with the xarness palette.

    Textual's fence uses its own dark-only code theme, which renders
    unreadable colors on light backgrounds. In ANSI mode the terminal's own
    palette is still used; otherwise the palette-driven CodeHighlightTheme
    applies (and follows live /theme switches).
    """

    @classmethod
    def highlight(
        cls, code: str, language: str, ansi: bool = False, dark: bool = False,
        path: str | None = None,
    ) -> Content:
        from textual.highlight import ANSIDarkHighlightTheme, ANSILightHighlightTheme

        if ansi:
            ansi_theme = ANSIDarkHighlightTheme if dark else ANSILightHighlightTheme
            return highlight(code, language=language or None, path=path, theme=ansi_theme)
        return highlight(code, language=language or None, path=path, theme=theme.CodeHighlightTheme)


def highlight_code(code: str, path: str | None = None, language: str = "") -> Content:
    """Highlight a code block the same way assistant-message fences do.

    Tool-call bodies (read_file/write_file) must match the LLM message
    fences exactly, including the ANSI-mode branch: on terminals without
    truecolor, the palette-driven CodeHighlightTheme quantizes its hex
    colors down to the 16-color ANSI palette and tokens collapse to
    black-on-dark. Reading the flags off the active app keeps the two
    paths in lockstep.
    """
    from textual.app import active_app

    app = active_app.get()
    return PaletteFence.highlight(
        code,
        language=language or None,
        ansi=app.native_ansi_color,
        dark=app.current_theme.dark,
        path=path,
    )


class SettledMarkdown(Markdown):
    """One settled assistant chunk as a Textual Markdown widget.

    Markdown renders its blocks as Content visuals, which carry the per-cell
    offset metadata Textual's selection system needs — so a mouse drag
    selects individual characters and ctrl+shift+c copies exactly what's
    highlighted. (A Static rendering rich.markdown has no such metadata, so
    selection degrades to whole-widget blocks.) Fenced code keeps the
    palette-driven highlighter, and the widget sizes to its content inside
    the scrolling chat log instead of claiming free height."""

    BLOCKS = {
        **Markdown.BLOCKS,
        "fence": PaletteFence,
        "code_block": PaletteFence,
    }

    def __init__(self, text: str) -> None:
        super().__init__(text, classes="assistant-md")

    def on_mount(self) -> None:
        self.styles.height = "auto"
        self.styles.margin = 0
        self.styles.padding = 0


class ShimmerText(Static):
    """A sweeping brightness band moving through static text, looping.

    This is the Codex-style "Processing…" / "Thinking…" treatment — not a
    spinner glyph. Continuously advances ``phase`` and recolors each
    character by distance from the peak, at ``theme.SHIMMER_FPS``.
    """

    phase = reactive(0.0)

    def __init__(
        self,
        label: str,
        base_color: str,
        peak_color: str,
        suffix: str = "",
        **kwargs,
    ) -> None:
        super().__init__("", markup=False, **kwargs)
        self.label = label
        # Static text appended (space-separated) after the animated part:
        # the tool-call arguments/detail never shimmer.
        self.suffix = suffix
        self.base_color = base_color
        self.peak_color = peak_color
        self._timer = None

    def on_mount(self) -> None:
        self._timer = self.set_interval(1 / theme.SHIMMER_FPS, self._tick)
        self._render_frame()

    def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()

    def set_colors(self, base_color: str, peak_color: str) -> None:
        """Recolor in place (used when the app switches themes live)."""
        self.base_color = base_color
        self.peak_color = peak_color
        self._render_frame()

    def set_label(self, label: str, suffix: str | None = None) -> None:
        """Swap the text in place (ToolWritingIndicator follows whichever
        tool call is currently streaming). ``suffix`` replaces the static
        detail when given, keeping it out of the animated band."""
        self.label = label
        if suffix is not None:
            self.suffix = suffix
        self._render_frame()

    def _tick(self) -> None:
        if not self.is_mounted:
            return
        span = len(self.label) + theme.SHIMMER_BAND_WIDTH * 2
        self.phase = (self.phase + theme.SHIMMER_SPEED / theme.SHIMMER_FPS) % span
        self._render_frame()

    def _render_frame(self) -> None:
        text = Text()
        peak = self.phase - theme.SHIMMER_BAND_WIDTH
        half_band = max(theme.SHIMMER_BAND_WIDTH / 2, 1e-6)
        for i, ch in enumerate(self.label):
            dist = abs(i - peak)
            intensity = max(0.0, 1.0 - dist / half_band)
            color = _lerp_hex(self.base_color, self.peak_color, intensity)
            text.append(ch, style=Style(color=color, bold=intensity > 0.6))
        if self.suffix:
            # A space separates the animated label from the static args
            # detail; the suffix sits above the base color, but well below
            # the shimmer peak so it reads as secondary.
            suffix_color = _lerp_hex(self.base_color, self.peak_color, 0.35)
            text.append(" " + self.suffix, style=Style(color=suffix_color))
        self.update(text)


class SuggestionPopup(OptionList):
    """Floating suggestion list for slash commands and @-file mentions."""

    can_focus = False

    def set_items(self, items: list[tuple[str, str]]) -> None:
        self.clear_options()
        for value, label in items:
            self.add_option(Option(label, id=value))
        if items:
            self.highlighted = 0

    @property
    def selected_value(self) -> str | None:
        if self.highlighted is None:
            return None
        option = self.get_option_at_index(self.highlighted)
        return option.id if option else None

    def move_highlight(self, delta: int) -> None:
        if self.option_count == 0:
            return
        current = self.highlighted or 0
        self.highlighted = (current + delta) % self.option_count


class UserMessage(Vertical):
    """A submitted user message: full-width surface highlight, dim marker.

    Marker + content columns so wrapped lines hang-indent under the text
    after "›" instead of returning to the left edge of the surface. Text
    color comes from CSS ($c-text) so live /theme switches recolor it.

    Clicking the message reveals a small "↩ undo to here" affordance;
    clicking that posts :class:`UndoRequested` so the app can roll the
    conversation back to this turn (see ChatController.rollback_plan_to).

    Initial content is rendered in compose(): Textual dispatches on_mount
    before the widget counts as mounted, so mount-time updates get skipped
    by any is_mounted guard and must not be the only render path.
    """

    class UndoRequested(Message):
        """The user asked to undo back to this message."""

        def __init__(self, user_message: "UserMessage") -> None:
            super().__init__()
            self.user_message = user_message

    def __init__(self, text: str) -> None:
        super().__init__(classes="msg user")
        self._text = text
        self._suffix = ""
        # The conversation message this widget renders, bound by the app once
        # the turn is under way (live) or when replaying history. Click-to-undo
        # needs it to locate the turn in the conversation.
        self.message: ConversationMessage | None = None

    def compose(self):
        with Horizontal(classes="user-row"):
            yield Static("›", classes="user-marker", markup=False)
            yield Static(self._build_content(), classes="user-content", markup=False)
        yield _UndoButton(self)

    def _build_content(self) -> Text:
        line = Text(self._text)
        if self._suffix:
            line.append(self._suffix, style=theme.PALETTE["muted"])
        return line

    @property
    def text(self) -> str:
        return self._text

    def apply_palette(self) -> None:
        """(Re)render the content with the current palette (used by /theme).
        Pre-mount calls are no-ops: compose() renders from the same state."""
        if not self.is_mounted:
            return
        self.query_one(".user-content", Static).update(self._build_content())

    def mark_sent(self) -> None:
        """The queued message reached the model; drop the queued marker."""
        self._suffix = ""
        self.apply_palette()

    def on_click(self, event: events.Click) -> None:
        """Reveal (or hide) the undo affordance. Only messages the app has
        bound to a conversation turn are undoable."""
        if self.message is None:
            return
        self.toggle_class("undo-armed")
        event.stop()


class CompactionSummary(Static):
    """The handoff summary an auto-compaction leaves in the transcript.

    Deliberately not rendered as a plain user message: padded on both
    sides and colored with the accent, so the model-facing handoff reads
    as a distinct event in the scrollback — live (mounted right after the
    compaction runs) and on /resume replay. Colors come from CSS, so
    /theme switches apply."""

    DEFAULT_CSS = """
    CompactionSummary {
        width: 1fr;
        height: auto;
        padding: 0 2;
        color: $c-accent;
    }
    """

    def __init__(self, text: str) -> None:
        self._text = text
        super().__init__(Text(text), classes="msg compaction-summary", markup=False)

    @property
    def text(self) -> str:
        return self._text

    def apply_palette(self) -> None:
        """Re-render on /theme (color lives in CSS; nothing to rebuild)."""


class SteerQueueBar(Vertical):
    """Queued steer messages, pinned above the chat input.

    While a turn is busy, submitted messages wait here instead of landing
    in the chat log; each becomes a UserMessage in the log only when it is
    actually sent (at a round boundary or turn end), so the transcript
    reads in send order. Colors come from CSS, so /theme switches apply."""

    def update_items(self, texts: list[str]) -> None:
        for old in self.query(".steer-item, .steer-hint"):
            old.remove()
        for text in texts:
            self.mount(Static(f"› {text}", classes="steer-item", markup=False))
        if texts:
            self.add_class("visible")
            # Only the oldest message is "first in line" — enter sends it.
            self.mount(Static("↵ enter to send now", classes="steer-hint", markup=False))
        else:
            self.remove_class("visible")


class _UndoButton(Static):
    """The click-to-undo affordance a UserMessage reveals when clicked."""

    def __init__(self, owner: UserMessage) -> None:
        super().__init__("↩ undo to here", classes="user-undo", markup=False)
        self._owner = owner

    def on_click(self, event: events.Click) -> None:
        self._owner.post_message(UserMessage.UndoRequested(self._owner))
        event.stop()


class AssistantMessage(Vertical):
    """Streams the answer, settling each closed code fence as it completes.

    ``.assistant-content`` holds an ordered stack: zero or more finalized
    SettledMarkdown widgets (rendered once a fence closes, never re-rendered,
    one cheap Static per chunk) followed by exactly one live Static showing the still-streaming tail.
    finalize() runs the same settle logic once more against any leftover
    text at turn-end, so there's a single code path for "this chunk is done,
    render it" whether it happens mid-stream or at the very end.
    """

    # Coalesced live-tail rendering: re-render at a fixed rate, not per token.
    _FLUSH_INTERVAL = 0.05

    def __init__(self) -> None:
        super().__init__(classes="msg assistant")
        self._live_text = ""
        self._has_settled = False
        self._finalized = False
        self._live_dirty = False
        self._flush_timer: Timer | None = None

    def compose(self):
        with Horizontal(classes="assistant-row"):
            yield Static("•", classes="assistant-marker", markup=False)
            with Vertical(classes="assistant-content"):
                yield Static("", classes="assistant-live", markup=False)

    def on_mount(self) -> None:
        self._live = self.query_one(".assistant-live", Static)

    async def append_delta(self, text: str) -> None:
        self._live_text += text
        settled, remainder = _split_settled(self._live_text)
        if settled:
            await self._settle(settled)
            self._live_text = remainder
        # Display-only rstrip: models often end a message with blank lines
        # before requesting tool calls, and the raw live tail renders them as
        # a blank gap until finalize() swaps in Markdown (which strips them).
        # _live_text itself stays intact for the fence-settling logic above.
        self._live_dirty = True
        if self._flush_timer is None:
            self._flush_timer = self.set_timer(self._FLUSH_INTERVAL, self._flush)

    def _flush(self) -> None:
        """Coalesced live-tail re-render: at most ~20 Hz instead of per token."""
        self._flush_timer = None
        if not self._live_dirty or not self.is_mounted:
            return
        self._live_dirty = False
        self._live.update(self._live_text.rstrip())

    async def _settle(self, chunk: str) -> None:
        self._has_settled = True
        content = self.query_one(".assistant-content", Vertical)
        live = content.query_one(".assistant-live", Static)
        await content.mount(SettledMarkdown(chunk), before=live)

    async def finalize(self) -> None:
        """Settle whatever's left in the live tail; drop the now-empty Static.

        Idempotent: also called after stream errors and interrupts (paths
        that never see TurnComplete) so the streamed text always ends up in
        selectable SettledMarkdown rather than the plain live Static."""
        if self._finalized:
            return
        self._finalized = True
        if self._flush_timer is not None:
            self._flush_timer.stop()
            self._flush_timer = None
        self._live_dirty = False
        content = self.query_one(".assistant-content", Vertical)
        live = content.query_one(".assistant-live", Static)
        tail = self._live_text
        await live.remove()
        if tail.strip():
            await content.mount(SettledMarkdown(tail))
        elif not self._has_settled:
            await content.mount(SettledMarkdown("(no output)"))


class ThinkingBlock(Vertical):
    """Reasoning channel for a turn.

    Expanded while streaming so the live reasoning is visible; collapsed
    once finished. Clickable at any time — including while still streaming —
    to hide or reveal the reasoning text. Shows "Thinking" (shimmering) while
    active, "Thought for Xs" (plain) once done.

    Following new reasoning text is left to the chat log's anchor: while it is
    armed the growing text keeps the log pinned to the end, and once the user
    scrolls up the log stays where they put it.
    """

    # Coalesced rendering: deltas accumulate and the text re-renders at a
    # fixed rate instead of once per token.
    _FLUSH_INTERVAL = 0.05

    def __init__(self) -> None:
        super().__init__(classes="msg thinking expanded")  # open while streaming
        self._reasoning_parts: list[str] = []
        self._text_dirty = False
        self._flush_timer: Timer | None = None
        self._duration: float | None = None
        self._done = False
        self._start = time.monotonic()
        self.summary_text = "Thinking"

    @property
    def reasoning(self) -> str:
        return "".join(self._reasoning_parts)

    def compose(self):
        with Horizontal(classes="thinking-summary"):
            yield ShimmerText("Thinking", *theme.SHIMMER_THINKING, id="thinking-shimmer")
            yield Static("", id="thinking-elapsed", classes="thinking-elapsed", markup=False)
        with Horizontal(classes="thinking-row"):
            yield Static("", classes="thinking-marker", markup=False)
            yield Static("", classes="thinking-text", markup=False)

    def on_mount(self) -> None:
        self._refresh_text()

    @property
    def done(self) -> bool:
        return self._done

    def append_reasoning(self, text: str) -> None:
        self._reasoning_parts.append(text)
        self._schedule_flush()

    def _schedule_flush(self) -> None:
        """Coalesce rapid deltas into one re-render per flush interval."""
        self._text_dirty = True
        if self._flush_timer is None and self.is_mounted:
            self._flush_timer = self.set_timer(self._FLUSH_INTERVAL, self._flush)

    def _flush(self) -> None:
        self._flush_timer = None
        if self._text_dirty and self.is_mounted:
            self._text_dirty = False
            self._refresh_text()

    def _visible_reasoning(self) -> str:
        return self.reasoning

    def finish(self, duration: float | None, estimate_if_unknown: bool = True) -> None:
        """Collapse to a static summary. First call wins.

        When duration is None: if estimate_if_unknown (the live-streaming
        case), fall back to elapsed-since-start. If not (history replay,
        where "elapsed since start" would be meaningless), show "…" instead.
        """
        if self._done:
            return
        self._done = True
        if duration is not None:
            self._duration = duration
        elif estimate_if_unknown:
            self._duration = time.monotonic() - self._start
        else:
            self._duration = None
        self.remove_class("expanded")  # shrink once finished
        self._drop_text()
        self._swap_to_static_summary()

    def finish_unknown(self) -> None:
        """Finish without a measurable duration (history replay) — the summary
        shows "Thought for …s" instead of a number."""
        if self._done:
            return
        self._done = True
        self._duration = None
        self.remove_class("expanded")
        self._drop_text()
        self._swap_to_static_summary()

    def toggle(self) -> None:
        if self.has_class("expanded"):
            self.remove_class("expanded")
            self._drop_text()
        else:
            self.add_class("expanded")
            self._refresh_text()

    def on_click(self, event: events.Click) -> None:
        self.toggle()
        event.stop()

    def _drop_text(self) -> None:
        """Release the rendered reasoning (collapsed blocks show nothing;
        the raw text stays in _reasoning_parts for re-expansion)."""
        if self._flush_timer is not None:
            self._flush_timer.stop()
            self._flush_timer = None
        self._text_dirty = False
        if self.is_mounted:
            try:
                self.query_one(".thinking-text", Static).update("")
            except NoMatches:
                pass  # DOM pruned during app shutdown

    def _refresh_text(self) -> None:
        if self.is_mounted:
            self.query_one(".thinking-text", Static).update(self._visible_reasoning())

    def _swap_to_static_summary(self) -> None:
        summary_row = self.query_one(".thinking-summary", Horizontal)
        shimmer = summary_row.query("#thinking-shimmer")
        if shimmer:
            shimmer.remove()
        elapsed_static = self.query_one("#thinking-elapsed", Static)
        elapsed_static.remove_class("thinking-elapsed")
        elapsed_static.add_class("thinking-done-summary")
        self._refresh_summary_color()

    def recolor(self) -> None:
        """Re-render the settled summary with the current palette (/theme)."""
        if self._done and self.is_mounted:
            self._refresh_summary_color()

    def _refresh_summary_color(self) -> None:
        duration = _format_duration(self._duration) if self._duration is not None else "…"
        self.summary_text = f"Thought for {duration}"
        elapsed_static = self.query_one(".thinking-done-summary", Static)
        elapsed_static.update(Text(self.summary_text, style=f"italic {theme.PALETTE['muted']}"))


def _shorten(text: str, limit: int = 60) -> str:
    """Collapse whitespace and truncate long text for a one-line header."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _json_tool_args(args_text: str) -> dict:
    """Best-effort parse of a tool call's streamed arguments JSON."""
    try:
        parsed = json.loads(args_text)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _partial_string(args_text: str, key: str, unescape: bool = False) -> str | None:
    """Best-effort extraction of a string argument from tool-call JSON that
    may still be streaming: the text after ``"key": "`` up to the (not yet
    arrived, possibly) closing quote. Returns None when the key hasn't
    appeared at all."""
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"', args_text)
    if match is None:
        return None
    rest = args_text[match.end():]
    # Stop at the unescaped closing quote; also tolerate a dangling backslash
    # from a fragment boundary landing mid-escape.
    i = 0
    while i < len(rest):
        if rest[i] == "\\":
            i += 2
            continue
        if rest[i] == '"':
            rest = rest[:i]
            break
        i += 1
    rest = rest.rstrip("\\")
    if unescape:
        rest = rest.replace('\\"', '"').replace("\\\\", "\\").replace("\\n", "\n")
    return rest


def _write_file_loc(args_text: str) -> int | None:
    """Best-effort line count of write_file's ``content`` argument while its
    JSON is still streaming. Complete JSON counts exactly; a partial one
    counts the escaped ``\\n`` sequences after the "content" key. A
    trailing newline doesn't start a new line."""
    content = _json_tool_args(args_text).get("content")
    if isinstance(content, str):
        return _line_count(content)
    raw = _partial_string(args_text, "content")
    if raw is None:
        return None
    return _line_count(raw.replace("\\\\", "").replace("\\n", "\n"))


def _line_count(text: str) -> int:
    """Number of lines in ``text``; a trailing newline ends a line, not
    starts one (so "a\\nb\\n" is 2 lines, "a\\nb\\nc" is 3)."""
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _live_edit_counts(args_text: str) -> str | None:
    """'+N -M' running diffstat of edit_file's arguments while they stream
    (complete JSON only — partially streamed edits aren't counted)."""
    args = _json_tool_args(args_text)
    edits = args.get("edits")
    if not isinstance(edits, list):
        old, new = args.get("old_string"), args.get("new_string")
        if isinstance(old, str) and old:
            edits = [{"old_string": old, "new_string": new if isinstance(new, str) else ""}]
        else:
            return None
    adds = dels = 0
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        old, new = edit.get("old_string", ""), edit.get("new_string", "")
        if not isinstance(old, str) or not old:
            continue
        diff = list(difflib.unified_diff(
            old.splitlines(), new.splitlines() if isinstance(new, str) else [], n=0, lineterm=""
        ))
        adds += sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
        dels += sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
    if not adds and not dels:
        return None
    return f"+{adds} -{dels}"


def _stream_label(tool_name: str, args_text: str) -> str:
    """Live one-line label for a call whose arguments are streaming or that
    is executing: '<tool> <detail>', e.g. 'write_file +4 g.py',
    'run_bash git status', 'edit_file +2 -1 app.py'. Falls back to the bare
    tool name until enough of the arguments has arrived."""
    args = _json_tool_args(args_text)
    path = args.get("path")
    if not isinstance(path, str):
        path = _partial_string(args_text, "path")

    if tool_name == "write_file":
        loc = _write_file_loc(args_text)
        detail = " ".join(
            part for part in (f"+{loc}" if loc is not None else None, path) if part
        )
        return f"write_file {detail}".rstrip()
    if tool_name == "edit_file":
        if not path:
            return "edit_file"
        counts = _live_edit_counts(args_text)
        return f"edit_file {counts} {path}" if counts else "edit_file"
    if tool_name == "read_file" and path:
        return f"read_file {path}"
    if tool_name in "run_bash":
        command = args.get("command")
        if not isinstance(command, str):
            command = _partial_string(args_text, "command", unescape=True)
        if command:
            # One line only — later lines (a compound command, say) don't fit.
            return tool_name + " " + _shorten(command.splitlines()[0])
    return tool_name


def _stream_parts(tool_name: str, args_text: str) -> tuple[str, str]:
    """Split the live stream label into the shimmering part (the tool name)
    and the static detail (the arguments-derived part, '' when none yet)."""
    label = _stream_label(tool_name, args_text)
    detail = label[len(tool_name):].strip()
    return tool_name, detail


def _header_detail_text(tool_name: str, header: str) -> Text | None:
    """The stored header summary, laid out after the verb + tool name.

    Every tool's detail appends dimmed after a space, all on one line
    (``edit_file +3 -1 src/app.py``). run_bash's command is truncated for
    the header — the full command shows in the expanded body.
    """
    if not header:
        return None
    muted = theme.PALETTE["muted"]
    if tool_name == "edit_file":
        # The tool's header is "+N -M path"; the counts get the diff's
        # add/del colors so they read at a glance, path stays dimmed.
        detail = Text(" ", style=muted)
        counts = re.match(r"^(\+\d+) (-\d+) (.+)$", header)
        if counts:
            adds, dels, path = counts.groups()
            detail.append(adds, style=theme.PALETTE["diff_add"])
            detail.append(" ", style=muted)
            detail.append(dels, style=theme.PALETTE["diff_del"])
            detail.append(f" {path}", style=muted)
        else:
            detail.append(header, style=muted)
        return detail
    if tool_name == "write_file":
        # The tool's header is "+N path"; the count gets the diff's add
        # color so it reads at a glance, path stays dimmed. Older sessions
        # stored just the path — fall back to plain dimmed.
        detail = Text(" ", style=muted)
        counts = re.match(r"^(\+\d+) (.+)$", header)
        if counts:
            loc, path = counts.groups()
            detail.append(loc, style=theme.PALETTE["diff_add"])
            detail.append(f" {path}", style=muted)
        else:
            detail.append(header, style=muted)
        return detail
    if tool_name in "run_bash":
        detail = Text(" ", style=muted)
        detail.append(_shorten(header, 80), style=muted)
        return detail
    detail = Text(" ")
    detail.append(header, style=muted)
    return detail


def _tool_header_detail(tool_name: str, args_text: str, output_text: str) -> Text | None:
    """The dimmed detail that follows the verb in the settled header, e.g.
    ``read_file [dimmed]src/app.py``. Returns None (plain verb + name)
    for tools without a useful summary or unparseable arguments.

    Kept best-effort and defensive: model-supplied arguments are not trusted
    and a malformed call must never break rendering.
    """
    muted = theme.PALETTE["muted"]
    args = _json_tool_args(args_text)

    def as_str(value: object) -> str:
        return value if isinstance(value, str) else ""

    if tool_name in ("read_file", "write_file", "edit_file"):
        path = as_str(args.get("path"))
        if not path:
            return None
        detail = Text(" ")
        detail.append(_shorten(path), style=muted)
        if tool_name == "read_file":
            # Surface the requested range: read_file index.html [start_line=270, end_line=30]
            ranges = []
            if args.get("start_line") is not None:
                ranges.append(f"start_line={args['start_line']}")
            if args.get("end_line") is not None:
                ranges.append(f"end_line={args['end_line']}")
            if ranges:
                detail.append(f" [{', '.join(ranges)}]", style=muted)
        return detail
    if tool_name in "run_bash":
        command = as_str(args.get("command"))
        if not command:
            return None
        first = command.splitlines()[0]
        detail = Text(" ")
        detail.append(_shorten(first, 80), style=muted)
        return detail
    if tool_name == "web_search":
        query = as_str(args.get("query"))
        if not query:
            return None
        detail = Text(" ")
        detail.append(_shorten(query), style=muted)
        return detail
    return None


_EXPAND_PLAIN_TOOLS = frozenset({"run_bash"})


class ToolCallBlock(Vertical):
    """One tool call: colored status dot + name, expandable to see args/output.

    Collapsed by default, same interaction pattern as ThinkingBlock — click
    anytime to toggle. The dot is amber while the call is in flight, green
    once it succeeds, red on failure (including argument parse errors).
    """

    _STATUS_COLOR_KEY: ClassVar[dict[ToolCallStatus, str]] = {
        ToolCallStatus.MAKING_CALL: "warning",
        ToolCallStatus.PARSING_ERROR: "error",
        ToolCallStatus.CALL_SUCCEEDED: "success",
        ToolCallStatus.CALL_FAILED: "error",
    }

    _STATUS_VERB: ClassVar[dict[ToolCallStatus, str]] = {
        ToolCallStatus.MAKING_CALL: "Running",
        ToolCallStatus.PARSING_ERROR: "Failed",
        ToolCallStatus.CALL_SUCCEEDED: "",  # no verb: just "read_file path"
        ToolCallStatus.CALL_FAILED: "Failed",
    }

    def __init__(self, call_id: str, name: str) -> None:
        super().__init__(classes="msg toolcall")  # no "expanded" — starts collapsed
        self.call_id = call_id
        # DOMNode already exposes a read-only `name`; the tool's name lives here.
        self.tool_name = name
        self.accumulated_arguments = ""
        self._header = ""
        self._status = ToolCallStatus.MAKING_CALL
        self._output_text = ""
        self._body_dirty = False
        self._body_timer: Timer | None = None

    def compose(self):
        with Horizontal(classes="toolcall-summary"):
            yield Static(Text("•", style=theme.PALETTE["warning"]), classes="toolcall-dot", markup=False)
            yield ShimmerText(f"Running {self.tool_name}", *theme.SHIMMER_THINKING, classes="toolcall-shimmer")
        with Horizontal(classes="toolcall-row"):
            yield Static("", classes="toolcall-marker", markup=False)
            yield Static("", classes="toolcall-body", markup=False)

    @property
    def status(self) -> ToolCallStatus:
        return self._status

    def append_arguments(self, text: str) -> None:
        self.accumulated_arguments += text
        # Arguments can stream in large chunks (write_file bodies); the body
        # is only visible when expanded, so skip re-rendering while collapsed
        # and render once on expand instead.
        if self.has_class("expanded"):
            self._refresh_body()
        # The shimmer animates only the tool name; the args-derived detail
        # ('g.py +4', 'git status') renders statically after it.
        if self.is_mounted:
            try:
                name, detail = _stream_parts(
                    self.tool_name, self.accumulated_arguments
                )
                self.query_one(".toolcall-shimmer", ShimmerText).set_label(
                    f"Running {name}", detail
                )
            except NoMatches:
                pass  # DOM pruned during app shutdown

    def append_output(self, text: str) -> None:
        """Output streaming in while the call runs (run_bash) — the expanded
        body grows live; a collapsed block only accumulates the text (the
        final result replaces it via set_result anyway)."""
        self._output_text += text
        if self.is_mounted and self.has_class("expanded"):
            self._body_dirty = True
            if self._body_timer is None:
                self._body_timer = self.set_timer(0.05, self._flush_body)

    def _flush_body(self) -> None:
        """Coalesced body re-render while output streams in."""
        self._body_timer = None
        if self._body_dirty and self.is_mounted and self.has_class("expanded"):
            self._body_dirty = False
            self._refresh_body()

    def set_result(
        self,
        status: ToolCallStatus,
        output: str = "",
        error: str = "",
        header: str = "",
    ) -> None:
        """Move to a terminal state; output and error are mutually exclusive.

        ``header`` is the tool-computed summary line persisted with the
        conversation; when absent (older sessions), the header falls back to
        a best-effort reconstruction from the call's arguments.
        """
        self._status = status
        self._output_text = output or error
        self._header = header
        self._refresh_dot()
        self._refresh_body(force=True)
        self._render_summary()

    def toggle(self) -> None:
        if self.has_class("expanded"):
            self.remove_class("expanded")
            self._clear_body()
        else:
            self.add_class("expanded")
            self._refresh_body(force=True)
        self._render_summary()

    def _clear_body(self) -> None:
        """Drop the rendered body on collapse; the raw args/output text stays
        in memory and is re-rendered on the next expand."""
        if self._body_timer is not None:
            self._body_timer.stop()
            self._body_timer = None
        self._body_dirty = False
        if self.is_mounted:
            try:
                self.query_one(".toolcall-body", Static).update("")
            except NoMatches:
                pass  # DOM pruned during app shutdown

    def on_click(self, event: events.Click) -> None:
        self.toggle()
        event.stop()

    def recolor(self) -> None:
        """Re-render with the current palette (/theme).

        The status dot, any expanded body (diffs, code blocks), and the
        settled summary row (edit_file's +N -M counts use diff colors)
        bake palette colors into rich styles, so all need a re-render."""
        self._refresh_dot()
        self._refresh_body()
        self._render_summary()

    def _refresh_dot(self) -> None:
        if self.is_mounted:
            color = theme.PALETTE[self._STATUS_COLOR_KEY[self._status]]
            try:
                self.query_one(".toolcall-dot", Static).update(Text("•", style=color))
            except NoMatches:
                pass  # DOM pruned during app shutdown

    def _refresh_body(self, force: bool = False) -> None:
        if not self.is_mounted:
            return
        if not force and not self.has_class("expanded"):
            return  # collapsed: body isn't visible; rendered on expand
        try:
            static = self.query_one(".toolcall-body", Static)
        except NoMatches:
            return  # DOM pruned during app shutdown
        if self.tool_name == "edit_file":
            # The raw old_string/new_string arguments add nothing on expand —
            # show just the diff (or the error text when the edit failed).
            split = _split_tool_diff(self._output_text)
            if split is not None:
                # The path is already in the summary row; skip the diff's
                # own '── path' header line.
                static.update(_render_diff(split[1], show_headers=False))
                return
            static.update(self._output_text)
            return
        args = _json_tool_args(self.accumulated_arguments)
        path = args.get("path")
        if self.tool_name == "read_file" and isinstance(path, str) and path:
            # Show the read content as a syntax-highlighted code block; on
            # failure (or while arguments stream in) stay plain.
            if self._status is ToolCallStatus.CALL_SUCCEEDED:
                static.update(_render_read_file(path, self._output_text))
                return
        if self.tool_name == "write_file":
            # Show the written content as a syntax-highlighted code block
            # rather than the raw JSON arguments. While the arguments are
            # still streaming in (no complete JSON yet), fall through to the
            # plain rendering below.
            content = args.get("content")
            if isinstance(path, str) and path and isinstance(content, str):
                # highlight_code: the same highlighting the assistant
                # message's markdown code fences use, so tool code blocks
                # and LLM code blocks match. Language is guessed from the
                # path. The path itself is already in the summary row.
                #
                # Keep this one Content rather than a rich Group: a Group
                # renders through Rich, where the widget's background is
                # missing when the Content's `$text` (auto) styles resolve,
                # so unstyled tokens (punctuation, operators) come out black
                # on a dark background. A bare Content renders as a Visual
                # with the widget's style as the base, matching the fences.
                body = highlight_code(content, path=path)
                if self._output_text:
                    body = body + Content("\n" + self._output_text)
                static.update(body)
                return
        args = self.accumulated_arguments
        if self.tool_name == "run_bash":
            # Show the command itself, not its JSON wrapper — the command
            # line reads in the normal text color (the body's default is
            # the muted tool-output color); output stays muted.
            command = self._bash_command()
            if command:
                rendered = Text(command, style=theme.PALETTE["text"])
                if self._output_text:
                    rendered.append("\n" + self._output_text, style=theme.PALETTE["muted"])
                split = _split_tool_diff(
                    command + ("\n" + self._output_text if self._output_text else "")
                )
                if split is None:
                    static.update(rendered)
                    return
                rendered.append_text(_render_diff(split[1]))
                static.update(rendered)
                return
        body = args
        if self._output_text:
            body = f"{body}\n{self._output_text}" if body else self._output_text
        split = _split_tool_diff(body)
        if split is None:
            static.update(body)
            return
        # Unified diffs (edit_file's result, `git diff` from run_bash) render
        # with the theme's diff colors; anything before them stays plain.
        prose, diff = split
        rendered = Text(prose.rstrip("\n"))
        if rendered:
            rendered.append("\n")
        rendered.append_text(_render_diff(diff))
        static.update(rendered)

    def _bash_command(self) -> str:
        """The called command: parsed from the arguments, falling back to the
        stored header (which is the command) for resumed sessions."""
        command = _json_tool_args(self.accumulated_arguments).get("command")
        if isinstance(command, str) and command:
            return command
        return self._header

    def _render_summary(self) -> None:
        """(Re)build the settled summary row. Called on set_result and on
        toggle — when expanded, the argument-summary tools (run_bash, glob,
        grep, ls) drop the header detail since the full arguments/commands
        are shown in the body."""
        try:
            summary_row = self.query_one(".toolcall-summary", Horizontal)
        except NoMatches:
            return  # DOM pruned during app shutdown
        if self._status is ToolCallStatus.MAKING_CALL:
            # Still executing: the shimmer carries the live label
            # ('run_bash git status'). Leave it alone.
            return
        shimmer = summary_row.query(".toolcall-shimmer")
        if shimmer:
            shimmer.remove()
        for old in summary_row.query(".toolcall-summary-text"):
            old.remove()
        verb = self._STATUS_VERB[self._status]
        # Verb + tool name inherit the stylesheet's muted color; the detail
        # suffix (path, command, token counts, …) bakes the palette's muted
        # style in so it reads dimmer next to it. Succeeded calls drop the
        # verb entirely: "read_file index.html [start_line=…]".
        summary = Text(f"{verb} {self.tool_name}" if verb else self.tool_name)
        hide_detail = self.has_class("expanded") and self.tool_name in _EXPAND_PLAIN_TOOLS
        if self._status is ToolCallStatus.CALL_SUCCEEDED and not hide_detail:
            if self.tool_name == "read_file":
                # The stored header is just the path; the [start_line=…]
                # range suffix comes from the call's arguments.
                detail = _tool_header_detail(
                    self.tool_name, self.accumulated_arguments, self._output_text
                )
                if detail is None:
                    detail = _header_detail_text(self.tool_name, self._header)
            else:
                detail = _header_detail_text(self.tool_name, self._header)
                if detail is None:
                    detail = _tool_header_detail(
                        self.tool_name, self.accumulated_arguments, self._output_text
                    )
            if detail is not None:
                summary.append_text(detail)
        summary_row.mount(Static(summary, classes="toolcall-summary-text", markup=False))


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_DIFF_GIT_RE = re.compile(r"^diff --git ", re.MULTILINE)


_SHOWING_LINES_RE = re.compile(
    r"^\[showing lines \d+-\d+ of \d+(?:;[^\]]*)?\]", re.MULTILINE
)


def _render_read_file(path: str, output: str) -> RenderableType:
    """read_file result as a syntax-highlighted code block (the path lives
    in the summary row). The bracketed ``[showing lines …]`` truncation
    notes are lifted out so they don't get lexed as code; large-file
    outlines stay plain prose."""
    if "# File outline for" in output:
        return Text(output.rstrip("\n"))
    notes = _SHOWING_LINES_RE.findall(output)
    if notes:
        output = _SHOWING_LINES_RE.sub("", output).strip("\n")
        # One Content, not a rich Group — see the write_file branch in
        # _refresh_body for why a Group turns unstyled tokens black.
        return highlight_code(output, path=path) + Content("\n" + "\n".join(notes))
    return highlight_code(output, path=path)


def _split_tool_diff(text: str) -> tuple[str, str] | None:
    """Split tool output into ``(prose, unified diff)`` at its first
    ``diff --git`` line; ``None`` when the output isn't a diff."""
    match = _DIFF_GIT_RE.search(text)
    if match is None:
        return None
    return text[: match.start()], text[match.start() :]


def _render_diff(text: str, show_headers: bool = True) -> Text:
    """Unified diff as GitHub-style rich text.

    Each file collapses to one `── path` header with new/deleted/renamed
    badges (the index/mode/---/+++ prologue is parsed and dropped, binary
    changes become a short notice); +/- lines get a full-line background
    tint from the theme's diff colors, with old/new line numbers in a
    muted gutter. ``show_headers=False`` drops the per-file header line —
    used for edit_file, whose path is already in the summary row. Plain
    stateful parse — no external diff library."""
    p = theme.PALETTE
    out = Text()
    header_path: str | None = None
    header_badges: list[str] = []
    header_renamed = False

    def flush_header() -> None:
        nonlocal header_path, header_renamed
        if header_path is None:
            return
        if show_headers:
            if out:
                out.append("\n")  # blank separator between files
            out.append("── ", style=p["border"])
            out.append(header_path, style=f"bold {p['text']}")
            if "new" in header_badges:
                out.append("  new", style=f"italic {p['accent']}")
            if "deleted" in header_badges:
                out.append("  deleted", style=f"italic {p['error']}")
            if header_renamed:
                out.append("  renamed", style=f"italic {p['accent2']}")
            out.append("\n")
        header_path = None
        header_badges.clear()
        header_renamed = False

    def row(old: str, new: str, marker: str, body: str, fg: str, bg: str | None) -> None:
        tint = f" on {bg}" if bg else ""
        out.append(f"{old:>4} ", style=p["diff_meta"] + tint)
        out.append(f"{new:>4} ", style=p["diff_meta"] + tint)
        out.append(f"{marker} ", style=fg + tint)
        out.append(body, style=fg + tint)
        out.append("\n", style=tint)

    old_no = new_no = 0
    in_header = False
    for line in text.splitlines():
        if line.startswith("diff --git"):
            flush_header()
            header_path = line.split(" b/", 1)[1] if " b/" in line else line[11:].strip()
            in_header = True
            continue
        if in_header and line.startswith(("@@", "Binary")):
            in_header = False  # prologue ended
        elif in_header:
            if line.startswith("new file mode"):
                header_badges.append("new")
            elif line.startswith("deleted file mode"):
                header_badges.append("deleted")
            elif line.startswith("rename "):
                header_renamed = True
            # index/mode/similarity/---/+++ lines: parsed above, not shown
            continue

        if line.startswith("@@"):
            flush_header()
            m = _HUNK_RE.match(line)
            if m:
                old_no, new_no = int(m.group(1)), int(m.group(3))
            out.append(line, style=p["diff_hunk"])
            out.append("\n")
        elif line.startswith("Binary"):
            flush_header()
            out.append("binary file differs", style=f"italic {p['diff_meta']}")
            out.append("\n")
        elif line.startswith("+"):
            row("", str(new_no), "+", line[1:], p["diff_add"], p["diff_add_bg"])
            new_no += 1
        elif line.startswith("-"):
            row(str(old_no), "", "-", line[1:], p["diff_del"], p["diff_del_bg"])
            old_no += 1
        elif line.startswith("\\"):
            out.append("      ⋯ no newline at end of file\n", style=f"italic {p['diff_meta']}")
        else:
            flush_header()
            row(str(old_no), str(new_no), " ", line[1:] if line else "", p["text"], None)
            old_no += 1
            new_no += 1
    flush_header()
    return out


# Sentinel: no diff section currently shown (None is a valid *key* — the full diff).
_NO_DIFF: object = object()


class DiffText(VerticalScroll):
    """Capped, focusable diff pane with its own scrollbar. Up/down page by
    the pane's full visible height and stop at the pane's top/bottom — they
    never switch files. Left/right step to the previous/next file's diff;
    with the full diff open, right opens the first file's diff.
    At the first/last file the arrows do nothing."""

    can_focus = True

    def _summary(self) -> "DiffSummary | None":
        node = self.parent
        while node is not None and not isinstance(node, DiffSummary):
            node = node.parent
        return node  # type: ignore[return-value]

    def _step_file(self, delta: int) -> bool:
        summary = self._summary()
        if summary is None:
            return False
        key = summary._active_key
        if not isinstance(key, str):
            # Full multi-file diff open: right steps into the first file.
            if key is None and delta == 1 and summary._stat is not None:
                if summary._stat.files:
                    summary.post_message(summary.DiffRequested(summary._stat.files[0].path))
                    return True
            return False
        return summary._request_neighbour(key, delta)

    def action_scroll_up(self) -> None:
        self.scroll_page_up(animate=False)

    def action_scroll_down(self) -> None:
        self.scroll_page_down(animate=False)

    def key_left(self) -> None:
        self._step_file(-1)

    def key_right(self) -> None:
        self._step_file(1)


class DiffSummary(Vertical):
    """Live "Edited N files +A -D" bar above the chat input.

    Collapsed by default, same click-to-expand interaction as ThinkingBlock:
    clicking the summary line toggles the per-file list; clicking a file row
    posts a ``DiffRequested`` the App fulfills with a git call, since the
    widget holds no git state itself. Hidden entirely when there are no
    pending changes.

    Keyboard, once the list has focus (escape minimizes an open diff into
    this state): up/down move the highlighted row, enter opens its diff, and
    a further escape (via the app's priority interrupt binding) collapses
    the list back to the bare summary line.
    """

    can_focus = True  # arrow-key navigation over the file list once minimized

    class DiffRequested(Message):
        def __init__(self, key: str | None) -> None:
            self.key = key  # None = the full multi-file diff
            super().__init__()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._stat: DiffStat | None = None
        self._active_key: str | None | object = _NO_DIFF
        self._active_text = ""

    def _request_neighbour(self, key: str, delta: int) -> bool:
        """Post a DiffRequested for the file adjacent to ``key`` in the file
        list (delta -1/+1); True if a request was posted."""
        if self._stat is None:
            return False
        paths = [f.path for f in self._stat.files]
        try:
            index = paths.index(key)
        except ValueError:
            return False
        neighbor = index + delta
        if not 0 <= neighbor < len(paths):
            return False
        self.post_message(self.DiffRequested(paths[neighbor]))
        return True

    def compose(self):
        with Horizontal(id="diff-summary-row"):
            yield Static("", id="diff-summary-text", markup=False)
            yield Static("", id="diff-summary-add", markup=False)
            yield Static("", id="diff-summary-del", markup=False)
        with Vertical(id="diff-body"):
            yield Vertical(id="diff-files")
            # Focusable scroll container: click it (or tab to it) and the
            # diff scrolls with the keyboard; the wheel scrolls on hover.
            with DiffText(id="diff-content"):
                yield Static("", id="diff-text", markup=False)

    @property
    def active_diff_key(self) -> str | None | object:
        return self._active_key

    @property
    def diff_shown(self) -> bool:
        return self._active_key is not _NO_DIFF

    def toggle(self) -> None:
        if self.has_class("expanded"):
            self.remove_class("expanded")
        else:
            self.add_class("expanded")

    def minimize_diff(self) -> None:
        """Hide the open diff pane but keep the file list up with the last
        viewed file's row still highlighted, and take focus so arrow keys +
        enter can open another file's diff (escape again collapses)."""
        self._active_key = _NO_DIFF
        self._active_text = ""
        if self.is_mounted:
            self.query_one("#diff-text", Static).update("")
            self.query_one("#diff-content", DiffText).remove_class("show")
        self.focus()

    def collapse(self) -> None:
        """Close the expanded file list; only the summary line remains."""
        self.remove_class("expanded")
        self.app.query_one("#chat-input", ChatInput).focus()

    def on_click(self, event: events.Click) -> None:
        # Rows carry a ``diff_key`` attribute; clicks on anything else toggle
        # the expanded file list.
        node = event.widget
        while node is not None and node is not self:
            diff_key = getattr(node, "diff_key", _NO_DIFF)
            if diff_key is not _NO_DIFF:
                event.stop()
                self.post_message(self.DiffRequested(cast("str | None", diff_key)))
                return
            node = node.parent
        self.toggle()
        event.stop()

    def on_key(self, event: events.Key) -> None:
        """File-list navigation while focused with no diff open: up/down move
        the row highlight, enter opens the highlighted file's diff. With a
        diff open the keys belong to the diff pane (scroll/page), so stay out
        of the way — they only reach here bubbled up from DiffText."""
        if self.diff_shown or not self.has_class("expanded") or self._stat is None:
            return
        if event.key in ("up", "down", "enter"):
            event.prevent_default()
            event.stop()
            if event.key == "enter":
                self._open_highlighted()
            else:
                self._move_highlight(-1 if event.key == "up" else 1)

    def _active_row_index(self) -> int | None:
        for index, row in enumerate(self.query(".diff-file-row")):
            if row.has_class("active"):
                return index
        return None

    def _move_highlight(self, delta: int) -> None:
        rows = list(self.query(".diff-file-row"))
        if not rows:
            return
        current = self._active_row_index()
        index = 0 if current is None else max(0, min(len(rows) - 1, current + delta))
        self._mark_active_row(cast("str | None", getattr(rows[index], "diff_key", None)))

    def _open_highlighted(self) -> None:
        rows = list(self.query(".diff-file-row"))
        index = self._active_row_index()
        if index is None or index >= len(rows):
            return
        key = getattr(rows[index], "diff_key", None)
        if key is not None:
            self.post_message(self.DiffRequested(cast("str | None", key)))

    async def set_summary(self, stat: DiffStat) -> None:
        """Rebuild the summary + per-file rows; becomes visible."""
        self._stat = stat
        count = len(stat.files)
        # Colors are CSS ($c-* variables) applied to the Statics — keep the
        # Text unstyled so /theme recolors these rows live.
        self.query_one("#diff-summary-text", Static).update(
            Text(f"Edited {count} file{'s' if count != 1 else ''}")
        )
        self.query_one("#diff-summary-add", Static).update(Text(f"+{stat.additions}"))
        self.query_one("#diff-summary-del", Static).update(Text(f"-{stat.deletions}"))

        files = self.query_one("#diff-files", Vertical)
        await files.remove_children()
        # Pad the +/- counts to a uniform width across the file list so the
        # right-aligned tail (and the "new" badge) sits at the same column
        # on every row regardless of digit count.
        add_w = max(len(f"+{f.additions}") for f in stat.files)
        del_w = max(len(f"-{f.deletions}") for f in stat.files)
        for f in stat.files:
            # Row layout: name  [new]  +N -M — the whole row is clickable
            # and requests that file's unified diff. The "new" badge gets
            # its own column so it never wraps the file name.
            row = Horizontal(classes="diff-file-row")
            row.diff_key = f.path  # type: ignore[attr-defined]
            await files.mount(row)
            row.mount(Static(Text(f.path), classes="diff-file-name", markup=False))
            # Source tag: "agent" for a turn's edits, "drift" for changes
            # made outside the session.
            source_class = "drift" if "drift" in f.source else "agent"
            row.mount(Static(
                Text(f.source), classes=f"diff-file-source {source_class}", markup=False,
            ))
            if f.is_new:
                row.mount(Static(Text("new"), classes="diff-file-new", markup=False))
            row.mount(Static(
                Text(f"+{f.additions}".rjust(add_w)), classes="diff-file-add", markup=False,
            ))
            row.mount(Static(
                Text(f"-{f.deletions}".rjust(del_w)), classes="diff-file-del", markup=False,
            ))
        self.add_class("visible")

    async def clear(self) -> None:
        """Hide entirely — no pending changes (or no worktree)."""
        self._stat = None
        self._active_key = _NO_DIFF
        self._active_text = ""
        self.remove_class("expanded")
        self.remove_class("visible")
        if self.is_mounted:
            self.query_one("#diff-text", Static).update("")
            self.query_one("#diff-content", DiffText).remove_class("show")
            await self.query_one("#diff-files", Vertical).remove_children()

    def show_diff(self, key: str | None, text: str) -> None:
        """Display one file's (or the full) unified diff inline."""
        self._active_key = key
        self._active_text = text
        self.add_class("visible")
        self.add_class("expanded")
        content = self.query_one("#diff-content", DiffText)
        self.query_one("#diff-text", Static).update(_render_diff(text))
        content.add_class("show")
        content.focus()  # arrow keys page / step files right away
        # Defer until the container is laid out, or scrolling is a no-op.
        content.call_after_refresh(content.scroll_home, animate=False)
        self._mark_active_row(key)

    def hide_diff(self) -> None:
        self._active_key = _NO_DIFF
        self._active_text = ""
        if self.is_mounted:
            self.query_one("#diff-text", Static).update("")
            self.query_one("#diff-content", DiffText).remove_class("show")
            self._mark_active_row(None)
            # The diff pane may have held focus; hand it back to the input.
            self.app.query_one("#chat-input", ChatInput).focus()

    def _mark_active_row(self, key: str | None) -> None:
        """Highlight the row whose diff is currently displayed."""
        for row in self.query(".diff-file-row"):
            if getattr(row, "diff_key", _NO_DIFF) == key:
                row.add_class("active")
            else:
                row.remove_class("active")

    def recolor(self) -> None:
        """Re-render with the current palette (/theme).

        All row/summary/button colors are CSS variables, so the stylesheet
        recolors them live; only the diff text bakes palette colors into
        rich styles and needs a re-render."""
        if not self.is_mounted:
            return
        if self._active_key is not _NO_DIFF:
            self.query_one("#diff-text", Static).update(_render_diff(self._active_text))


class GeneratingBar(Static):
    """Random-walk brightness bar shown above the input while generating.

    Each column drifts up or down a brightness ladder one step per tick, so
    the bar breathes organically instead of cycling like a spinner. Every
    cell is the same bottom-anchored half-block glyph with the brightness
    in its *foreground* color: shade glyphs (░▒▓█) are ambiguous-width and
    render double in some terminals, and background-colored spaces get
    glyph-ized back into shades by Textual's driver. Toggled with the
    ``active`` class by ``_run_turn``.
    """

    GLYPH = "▄"
    WIDTH = 8
    LEVELS = 8
    TICK = 0.16

    def __init__(self, **kwargs) -> None:
        super().__init__(markup=False, **kwargs)
        self._cols: list[int] = []

    def _level_color(self, level: int) -> str:
        """Foreground color for a brightness level (0 = dim, max = accent)."""
        t = level / (self.LEVELS - 1) if self.LEVELS > 1 else 1.0
        return _lerp_hex(theme.PALETTE["bg"], theme.PALETTE["accent"], t)

    def on_mount(self) -> None:
        self._cols = [random.randrange(self.LEVELS) for _ in range(self.WIDTH)]
        self.set_interval(self.TICK, self._tick)

    def _tick(self) -> None:
        if not self.is_mounted:
            return
        top = self.LEVELS - 1
        for i in range(self.WIDTH):
            self._cols[i] = max(0, min(top, self._cols[i] + random.choice((-1, 0, 1))))
        line = Text("  ")
        for level in self._cols:
            line.append(self.GLYPH, style=self._level_color(level))
        self.update(line)


class PendingIndicator(Horizontal):
    """Shimmering busy state, shown until the first token of any kind.

    Also reused for compaction ("Compacting"), which has no interrupt hint.
    """

    def __init__(
        self,
        label: str = "Processing",
        colors: tuple[str, str] | None = None,
        hint: str = "esc to interrupt",
    ) -> None:
        super().__init__(classes="msg pending")
        self._label = label
        self._colors = colors or theme.SHIMMER_PROCESSING
        self._hint = hint
        self._start = time.monotonic()

    def compose(self):
        yield ShimmerText(self._label, *self._colors, id="pending-shimmer")
        yield Static("", id="pending-elapsed", classes="pending-elapsed", markup=False)

    def on_mount(self) -> None:
        self.set_interval(0.2, self._tick)
        self._tick()

    def _tick(self) -> None:
        if not self.is_mounted:
            return
        elapsed = time.monotonic() - self._start
        detail = f"{elapsed:.0f}s · {self._hint}" if self._hint else f"{elapsed:.0f}s"
        self.query_one("#pending-elapsed", Static).update(
            Text(f"  ({detail})", style=theme.PALETTE["muted"])
        )


class ToolWritingIndicator(Horizontal):
    """Single amber-dot shimmer shown while tool-call arguments are still
    streaming — replaces the per-block 'Running [tool]' shinies, which only
    make sense once a call actually executes. The shimmer sweeps over the
    tool name only; the arguments-derived detail ('write_file g.py +4',
    'run_bash git stat…') is appended as static text as fragments arrive.

    Owns the streaming-arguments state itself: fragments are accumulated per
    call_id here (the only consumer), not mirrored through the turn driver.
    The label follows the most recently updated call when several stream."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(classes="msg toolwriting")
        self.tool_name = tool_name
        self._names: dict[str, str] = {}
        self._args: dict[str, str] = {}
        self._current: str | None = None

    def compose(self):
        yield Static(Text("•", style=theme.PALETTE["warning"]), classes="toolcall-dot", markup=False)
        # Yellow from the theme (the warning palette key): base and peak are
        # the same color so the text stays solidly yellow, with the shimmer
        # band showing as a bold sweep.
        yield ShimmerText(
            self._label()[0], theme.PALETTE["warning"], theme.PALETTE["warning"],
            suffix=self._label()[1],
            classes="toolwriting-shimmer",
        )

    def _label(self) -> tuple[str, str]:
        """The animated tool name and the static args detail."""
        name = self._names.get(self._current or "", self.tool_name)
        return _stream_parts(name, self._args.get(self._current or "", ""))

    def set_tool(self, tool_name: str) -> None:
        """Follow the tool call that just started streaming."""
        self.tool_name = tool_name
        self._current = None
        self._refresh()

    def track_call(self, call_id: str, name: str) -> None:
        """Register a newly started call; its arguments accumulate here."""
        self._names[call_id] = name
        self._args.setdefault(call_id, "")
        self._current = call_id
        self.tool_name = name
        self._refresh()

    def append_args_delta(self, call_id: str, text: str) -> None:
        """Feed one streamed argument fragment so the label grows live detail."""
        if call_id not in self._names:
            return
        self._args[call_id] += text
        self._current = call_id
        self._refresh()

    def _refresh(self) -> None:
        if not self.is_mounted:
            return
        try:
            name, detail = self._label()
            self.query_one(".toolwriting-shimmer", ShimmerText).set_label(
                name, detail
            )
        except NoMatches:
            pass  # DOM pruned during app shutdown


class MessageLine(Static):
    """Unified one-line status message in the scrollback.

    One widget for every non-conversational line — notices, warnings,
    confirmations, errors — with ``kind`` picking the glyph and the CSS class
    ($c-muted / $c-warning / $c-success / $c-error), so theme switches recolor
    every kind live. :class:`ErrorLine` and :class:`NoticeLine` are the two
    named shorthands the rest of the app uses.
    """

    _GLYPHS = {"error": "✗ ", "warn": "! ", "success": "✓ "}

    def __init__(self, message: str, kind: str = "notice") -> None:
        super().__init__(
            Text(f"{self._GLYPHS.get(kind, '')}{message}"), classes=f"msg {kind}"
        )


class ErrorLine(MessageLine):
    """A request-level failure, rendered inline in the scrollback."""

    def __init__(self, message: str) -> None:
        super().__init__(message, kind="error")


class NoticeLine(MessageLine):
    """Muted one-line status notice (e.g. 'worked for 2m 15s')."""

    def __init__(self, message: str) -> None:
        super().__init__(message, kind="notice")


class AskBar(Static):
    """The ask tool's current question, shown above the input while pending.

    Hidden until a question is active; the user answers through the normal
    chat input. Content wraps to the terminal width (unlike the old modal's
    auto-width labels, which clipped long questions).
    """

    def show_question(self, text: str) -> None:
        self.update(Text(text))
        self.add_class("visible")

    def hide(self) -> None:
        self.remove_class("visible")


def crop_path(path: str, max_len: int = 16) -> str:
    """Shorten a path from the left, keeping the tail: .../sub/file.

    Whole components are dropped until the remainder fits, so the result
    always shows complete directory names (never a partial one)."""
    if len(path) <= max_len:
        return path
    parts = path.rstrip("/").split("/")
    while parts and len("/".join(parts)) > max_len:
        parts.pop(0)
    tail = "/".join(parts)
    return f".../{tail}" if tail else path


class StatusBar(Static):
    """Persistent footer: model, effort, session tokens, context headroom."""

    def update_status(
        self,
        *,
        shown_name: str,
        effort: str,
        mode: str,
        total_in: int,
        total_out: int,
        context_used: int,
        max_context: int,
        workspace: str | None = None,
    ) -> None:
        palette = theme.PALETTE
        percent_left = max(0, round(100 * (1 - context_used / max_context))) if max_context else 100

        line = Text()
        line.append(shown_name, style=f"bold {palette['status']}")
        line.append(f" · effort {effort}", style=palette["muted"])
        line.append(f" · {mode}", style=palette["accent2"] if mode == "write" else palette["text"])
        line.append("  │  ", style=palette["border"])
        line.append("in ", style=palette["muted"])
        line.append(_fmt_tokens(total_in), style=palette["text"])
        line.append(" · out ", style=palette["muted"])
        line.append(_fmt_tokens(total_out), style=palette["text"])
        line.append("  │  ", style=palette["border"])
        line.append("ctx ", style=palette["muted"])
        line.append(f"{_fmt_tokens(context_used)}/{_fmt_tokens(max_context)}", style=palette["text"])
        line.append(f" · {percent_left}% left", style=palette["muted"])
        if workspace:
            line.append("  │  ", style=palette["border"])
            line.append(crop_path(workspace), style=palette["muted"])
        self._text = str(line)
        self.update(line)

    @property
    def text(self) -> str:
        return self._text


class ChatInput(TextArea):
    """Persistent multi-line input bar: Enter submits, Shift+Enter newlines.

    Detects '/' (slash commands, must start the input) and '@' (file
    mentions, anywhere) and posts query messages so App can drive a
    SuggestionPopup. When a popup is active, arrow/enter/tab/escape are
    redirected to popup navigation instead of normal TextArea behavior.
    """

    class ChatSubmitted(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    class SlashQuery(Message):
        def __init__(self, query: str) -> None:
            self.query = query
            super().__init__()

    class AtQuery(Message):
        def __init__(self, query: str) -> None:
            self.query = query
            super().__init__()

    class PopupNav(Message):
        def __init__(self, direction: int) -> None:
            self.direction = direction
            super().__init__()

    class PopupConfirm(Message):
        pass

    class PopupDismiss(Message):
        pass

    class ModeToggle(Message):
        """Tab pressed with no popup active — toggle plan/write mode."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "submit", "Send", priority=True),
        Binding("shift+enter", "newline", "Newline", priority=True, show=False),
        Binding("alt+enter", "newline", "Newline", priority=True, show=False),
        # priority beats Widget's inherited tab→focus_next; _on_key still
        # claims tab first while a popup is open (popup confirm).
        Binding("tab", "toggle_mode", "Mode", priority=True, show=False),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.popup_active: str | None = None

    def on_mount(self) -> None:
        self.apply_input_theme()

    def apply_input_theme(self) -> None:
        """(Re)build the TextArea theme from the current palette.

        Called on mount and again when /theme switches palettes live; the
        cursor and selection colors come from the theme config.
        """
        base = TextAreaTheme.get_builtin_theme("css")
        no_line_highlight = TextAreaTheme(
            name="xarness-input",
            base_style=base.base_style,
            gutter_style=base.gutter_style,
            cursor_style=Style(bgcolor=theme.PALETTE["cursor"], color=theme.PALETTE["bg"]),
            cursor_line_style=Style(),
            cursor_line_gutter_style=base.cursor_line_gutter_style,
            bracket_matching_style=base.bracket_matching_style,
            selection_style=Style(bgcolor=theme.PALETTE["highlight"], color=theme.PALETTE["bg"]),
            syntax_styles=base.syntax_styles,
        )
        self.register_theme(no_line_highlight)
        # Assigning the same name wouldn't re-run the theme watcher, so bounce
        # through the builtin to force the new colors to be applied.
        self.theme = "css"
        self.theme = "xarness-input"

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._check_triggers()

    def _check_triggers(self) -> None:
        text = self.text
        slash_match = _SLASH_RE.match(text) if "\n" not in text else None
        if slash_match:
            self.popup_active = "slash"
            self.post_message(self.SlashQuery(slash_match.group(1)))
            return

        row, col = self.cursor_location
        line = self.document.get_line(row)
        prefix = line[:col]
        at_match = _AT_RE.search(prefix)
        if at_match:
            self.popup_active = "at"
            self.post_message(self.AtQuery(at_match.group(1)))
            return

        if self.popup_active is not None:
            self.popup_active = None
            self.post_message(self.PopupDismiss())

    def _on_key(self, event: events.Key) -> None:
        if self.popup_active is not None and event.key in ("up", "down", "enter", "tab", "escape"):
            event.prevent_default()
            event.stop()
            if event.key == "up":
                self.post_message(self.PopupNav(-1))
            elif event.key == "down":
                self.post_message(self.PopupNav(1))
            elif event.key in ("enter", "tab"):
                self.post_message(self.PopupConfirm())
            elif event.key == "escape":
                self.popup_active = None
                self.post_message(self.PopupDismiss())
            return
        # NOTE: no super()._on_key() here — Textual already dispatches
        # TextArea._on_key via the MRO; calling it manually creates an
        # un-awaited coroutine (RuntimeWarning) with no extra effect.

    def action_submit(self) -> None:
        if self.popup_active is not None:
            self.post_message(self.PopupConfirm())
            return
        text = self.text.strip()
        if not text:
            # Empty Enter is still an event: the app uses it to "send now"
            # a message that is queued behind a running turn.
            self.post_message(self.ChatSubmitted(""))
            return
        self.load_text("")
        self.post_message(self.ChatSubmitted(text))

    def action_newline(self) -> None:
        self.insert("\n")

    def action_toggle_mode(self) -> None:
        self.post_message(self.ModeToggle())

    def insert_mention(self, value: str) -> None:
        row, col = self.cursor_location
        line = self.document.get_line(row)
        prefix = line[:col]
        at_col = prefix.rfind("@")
        if at_col == -1:
            return
        self.replace(f"@{value} ", (row, at_col), (row, col))
        self.popup_active = None
