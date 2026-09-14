"""Widgets for the xarness chat TUI.

Widgets are dumb views: they render what the app feeds them from stream
events and hold no conversation state of their own (the reasoning text inside
``ThinkingBlock`` is display state only; the authoritative copy lives in
``ChatController.conversation``).
"""

from __future__ import annotations

import re
import time
from typing import ClassVar, cast

from rich.style import Style
from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Markdown, OptionList, Static, TextArea
from textual.widgets.option_list import Option
from textual.widgets.text_area import TextAreaTheme

from .. import theme
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


class InlineMarkdown(Markdown):
    """Markdown that sizes to its content instead of claiming free space.

    Textual's Markdown is internally scroll-container-based and defaults to
    height: 1fr — fine standalone, wrong when nested inside our own
    scrolling chat log. Kept as a safety net even though the height bug we
    actually hit turned out to be ThinkingBlock, not this — a long enough
    assistant response could still exercise this path.
    """

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
        **kwargs,
    ) -> None:
        super().__init__("", markup=False, **kwargs)
        self.label = label
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

    Initial content is rendered in compose(): Textual dispatches on_mount
    before the widget counts as mounted, so mount-time updates get skipped
    by any is_mounted guard and must not be the only render path.
    """

    def __init__(self, text: str) -> None:
        super().__init__(classes="msg user")
        self._text = text
        self._suffix = ""

    def compose(self):
        with Horizontal(classes="user-row"):
            yield Static("›", classes="user-marker", markup=False)
            yield Static(self._build_content(), classes="user-content", markup=False)

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

    def mark_queued(self) -> None:
        """Note that this message is waiting for the current turn to finish."""
        self._suffix = "   (queued)"
        self.apply_palette()


class AssistantMessage(Vertical):
    """Streams the answer, settling each closed code fence as it completes.

    ``.assistant-content`` holds an ordered stack: zero or more finalized
    InlineMarkdown widgets (rendered once a fence closes, never re-rendered)
    followed by exactly one live Static showing the still-streaming tail.
    finalize() runs the same settle logic once more against any leftover
    text at turn-end, so there's a single code path for "this chunk is done,
    render it" whether it happens mid-stream or at the very end.
    """

    def __init__(self) -> None:
        super().__init__(classes="msg assistant")
        self._live_text = ""
        self._has_settled = False

    def compose(self):
        with Horizontal(classes="assistant-row"):
            yield Static("•", classes="assistant-marker", markup=False)
            with Vertical(classes="assistant-content"):
                yield Static("", classes="assistant-live", markup=False)

    async def append_delta(self, text: str) -> None:
        self._live_text += text
        settled, remainder = _split_settled(self._live_text)
        if settled:
            await self._settle(settled)
            self._live_text = remainder
        self.query_one(".assistant-live", Static).update(self._live_text)

    async def _settle(self, chunk: str) -> None:
        self._has_settled = True
        content = self.query_one(".assistant-content", Vertical)
        live = content.query_one(".assistant-live", Static)
        await content.mount(InlineMarkdown(chunk, classes="assistant-md"), before=live)

    async def finalize(self) -> None:
        """Settle whatever's left in the live tail; drop the now-empty Static."""
        content = self.query_one(".assistant-content", Vertical)
        live = content.query_one(".assistant-live", Static)
        tail = self._live_text
        await live.remove()
        if tail.strip():
            await content.mount(InlineMarkdown(tail, classes="assistant-md"))
        elif not self._has_settled:
            await content.mount(InlineMarkdown("(no output)", classes="assistant-md"))


class ThinkingBlock(Vertical):
    """Reasoning channel for a turn.

    Collapsed by default. Clickable at any time — including while still
    streaming — to reveal or hide the live reasoning text. Shows "Thinking"
    (shimmering) while active, "Thought for Xs" (plain) once done.

    Auto-scrolls the chat log to follow new reasoning text, but only while
    expanded AND still actively thinking — never after it's settled, so
    re-expanding a finished thought later doesn't yank the viewport.
    """

    def __init__(self) -> None:
        super().__init__(classes="msg thinking")  # no "expanded" — starts collapsed
        self._reasoning = ""
        self._duration: float | None = None
        self._done = False
        self._start = time.monotonic()
        self.summary_text = "Thinking"

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
        self._reasoning += text
        self._refresh_text()
        if self.has_class("expanded") and not self._done:
            self._follow_scroll()

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
        self._swap_to_static_summary()

    def finish_unknown(self) -> None:
        """Finish without a measurable duration (history replay) — the summary
        shows "Thought for …s" instead of a number."""
        if self._done:
            return
        self._done = True
        self._duration = None
        self._swap_to_static_summary()

    def toggle(self) -> None:
        if self.has_class("expanded"):
            self.remove_class("expanded")
        else:
            self.add_class("expanded")
            if not self._done:
                self._follow_scroll()

    def on_click(self, event: events.Click) -> None:
        self.toggle()
        event.stop()

    def _follow_scroll(self) -> None:
        if self.is_mounted and self.parent is not None:
            self.parent.scroll_end(animate=False)

    def _refresh_text(self) -> None:
        if self.is_mounted:
            self.query_one(".thinking-text", Static).update(self._reasoning)

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
        ToolCallStatus.CALL_SUCCEEDED: "Ran",
        ToolCallStatus.CALL_FAILED: "Failed",
    }

    def __init__(self, call_id: str, name: str) -> None:
        super().__init__(classes="msg toolcall")  # no "expanded" — starts collapsed
        self.call_id = call_id
        # DOMNode already exposes a read-only `name`; the tool's name lives here.
        self.tool_name = name
        self.accumulated_arguments = ""
        self._status = ToolCallStatus.MAKING_CALL
        self._output_text = ""

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
        self._refresh_body()

    def set_result(self, status: ToolCallStatus, output: str = "", error: str = "") -> None:
        """Move to a terminal state; output and error are mutually exclusive."""
        self._status = status
        self._output_text = output or error
        self._refresh_dot()
        self._refresh_body()
        self._swap_to_static_summary()

    def toggle(self) -> None:
        if self.has_class("expanded"):
            self.remove_class("expanded")
        else:
            self.add_class("expanded")

    def on_click(self, event: events.Click) -> None:
        self.toggle()
        event.stop()

    def recolor(self) -> None:
        """Re-apply the status-dot color with the current palette (/theme)."""
        self._refresh_dot()

    def _refresh_dot(self) -> None:
        if self.is_mounted:
            color = theme.PALETTE[self._STATUS_COLOR_KEY[self._status]]
            self.query_one(".toolcall-dot", Static).update(Text("•", style=color))

    def _refresh_body(self) -> None:
        if not self.is_mounted:
            return
        body = self.accumulated_arguments
        if self._output_text:
            body = f"{body}\n{self._output_text}" if body else self._output_text
        self.query_one(".toolcall-body", Static).update(body)

    def _swap_to_static_summary(self) -> None:
        summary_row = self.query_one(".toolcall-summary", Horizontal)
        shimmer = summary_row.query(".toolcall-shimmer")
        if shimmer:
            shimmer.remove()
        verb = self._STATUS_VERB[self._status]
        summary_row.mount(Static(f"{verb} {self.tool_name}", classes="toolcall-summary-text", markup=False))


def _colorize_diff(text: str) -> Text:
    """Unified diff as rich text: + lines green, - lines red, hunk headers
    and metadata dimmed. Plain line-prefix coloring — no diff parser."""
    palette = theme.PALETTE
    out = Text()
    for line in text.splitlines():
        if line.startswith(("diff ", "index ", "+++ ", "--- ", "new file", "deleted file", "old mode", "new mode", "similarity ", "rename ", "Binary ")):
            style = palette["muted"]
        elif line.startswith("@@"):
            style = palette["status"]
        elif line.startswith("+"):
            style = palette["success"]
        elif line.startswith("-"):
            style = palette["error"]
        else:
            style = palette["text"]
        out.append(line, style=style)
        out.append("\n", style=palette["text"])
    return out


# Sentinel: no diff section currently shown (None is a valid *key* — the full diff).
_NO_DIFF: object = object()


class DiffSummary(Vertical):
    """Live "Edited N files +A -D" bar above the chat input.

    Collapsed by default, same click-to-expand interaction as ThinkingBlock:
    clicking the summary line toggles the per-file list; each file row's
    [diff] affordance (and the summary line's own [diff], which opens the
    full multi-file diff) posts a ``DiffRequested`` the App fulfills with a
    git call, since the widget holds no git state itself. Hidden entirely
    when there are no pending changes.
    """

    class DiffRequested(Message):
        def __init__(self, key: str | None) -> None:
            self.key = key  # None = the full multi-file diff
            super().__init__()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._stat: DiffStat | None = None
        self._active_key: str | None | object = _NO_DIFF
        self._active_text = ""

    def compose(self):
        with Horizontal(id="diff-summary-row"):
            yield Static("", id="diff-summary-text", markup=False)
            full_btn = Static(
                Text("[diff]", style=theme.PALETTE["status"]), id="diff-open-full", markup=False
            )
            full_btn.diff_key = None  # type: ignore[attr-defined]
            yield full_btn
        with Vertical(id="diff-body"):
            yield Vertical(id="diff-files")
            yield Static("", id="diff-content", markup=False)

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

    def on_click(self, event: events.Click) -> None:
        # Rows (and the summary line's [diff]) carry a ``diff_key`` attribute;
        # clicks on anything else toggle the expanded file list.
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

    async def set_summary(self, stat: DiffStat) -> None:
        """Rebuild the summary + per-file rows; becomes visible."""
        self._stat = stat
        count = len(stat.files)
        label = Text()
        label.append(
            f"Edited {count} file{'s' if count != 1 else ''}",
            style=f"bold {theme.PALETTE['text']}",
        )
        label.append("  ", style=theme.PALETTE["border"])
        label.append(f"+{stat.additions}", style=theme.PALETTE["success"])
        label.append(" ", style=theme.PALETTE["text"])
        label.append(f"-{stat.deletions}", style=theme.PALETTE["error"])
        self.query_one("#diff-summary-text", Static).update(label)

        files = self.query_one("#diff-files", Vertical)
        await files.remove_children()
        for f in stat.files:
            # Row layout: name  dir/  +N -M — the whole row is clickable and
            # requests that file's unified diff.
            row = Horizontal(classes="diff-file-row")
            row.diff_key = f.path  # type: ignore[attr-defined]
            await files.mount(row)
            name = Text(f.path, style=theme.PALETTE["text"])
            if f.is_new:
                name.append("  new", style=f"italic {theme.PALETTE['accent']}")
            row.mount(Static(name, classes="diff-file-name", markup=False))
            directory = f.path.rsplit("/", 1)[0] + "/" if "/" in f.path else "."
            row.mount(Static(
                Text(directory, style=theme.PALETTE["muted"]),
                classes="diff-file-dir", markup=False,
            ))
            stats = Text(f"+{f.additions}", style=theme.PALETTE["success"])
            stats.append(f" -{f.deletions}", style=theme.PALETTE["error"])
            row.mount(Static(stats, classes="diff-file-stats", markup=False))
        self.add_class("visible")

    async def clear(self) -> None:
        """Hide entirely — no pending changes (or no worktree)."""
        self._stat = None
        self._active_key = _NO_DIFF
        self._active_text = ""
        self.remove_class("expanded")
        self.remove_class("visible")
        if self.is_mounted:
            content = self.query_one("#diff-content", Static)
            content.update("")
            content.remove_class("show")
            await self.query_one("#diff-files", Vertical).remove_children()

    def show_diff(self, key: str | None, text: str) -> None:
        """Display one file's (or the full) unified diff inline."""
        self._active_key = key
        self._active_text = text
        self.add_class("visible")
        self.add_class("expanded")
        content = self.query_one("#diff-content", Static)
        content.update(_colorize_diff(text))
        content.add_class("show")
        content.scroll_home(animate=False)
        self._mark_active_row(key)

    def hide_diff(self) -> None:
        self._active_key = _NO_DIFF
        self._active_text = ""
        if self.is_mounted:
            content = self.query_one("#diff-content", Static)
            content.update("")
            content.remove_class("show")
            self._mark_active_row(None)

    def _mark_active_row(self, key: str | None) -> None:
        """Highlight the row whose diff is currently displayed."""
        for row in self.query(".diff-file-row"):
            if getattr(row, "diff_key", _NO_DIFF) == key:
                row.add_class("active")
            else:
                row.remove_class("active")

    def recolor(self) -> None:
        """Re-render with the current palette (/theme)."""
        if not self.is_mounted:
            return
        self.query_one("#diff-open-full", Static).update(
            Text("[diff]", style=theme.PALETTE["status"])
        )
        if self._active_key is not _NO_DIFF:
            self.query_one("#diff-content", Static).update(_colorize_diff(self._active_text))


class PendingIndicator(Horizontal):
    """Shimmering 'Processing' state, shown until the first token of any kind."""

    def __init__(self) -> None:
        super().__init__(classes="msg pending")
        self._start = time.monotonic()

    def compose(self):
        yield ShimmerText("Processing", *theme.SHIMMER_PROCESSING, id="pending-shimmer")
        yield Static("", id="pending-elapsed", classes="pending-elapsed", markup=False)

    def on_mount(self) -> None:
        self.set_interval(0.2, self._tick)
        self._tick()

    def _tick(self) -> None:
        if not self.is_mounted:
            return
        elapsed = time.monotonic() - self._start
        self.query_one("#pending-elapsed", Static).update(
            Text(f"  ({elapsed:.0f}s · esc to interrupt)", style=theme.PALETTE["muted"])
        )


class ToolWritingIndicator(Horizontal):
    """Single amber-dot 'Writing tools' shimmer shown while tool-call
    arguments are still streaming — replaces the per-block 'Running [tool]'
    shinies, which only make sense once a call actually executes."""

    def compose(self):
        yield Static(Text("•", style=theme.PALETTE["warning"]), classes="toolcall-dot", markup=False)
        yield ShimmerText("Writing tools", *theme.SHIMMER_THINKING, classes="toolwriting-shimmer")


class ErrorLine(Static):
    """A request-level failure, rendered inline in the scrollback.

    Color comes from CSS ($c-error) so theme switches recolor it live.
    """

    def __init__(self, message: str) -> None:
        super().__init__(Text(f"✗ {message}"), classes="msg error")


class NoticeLine(Static):
    """Muted one-line status notice (e.g. 'worked for 2m 15s').

    Color comes from CSS ($c-muted) so theme switches recolor it live.
    """

    def __init__(self, message: str) -> None:
        super().__init__(Text(message), classes="msg notice")


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
    ) -> None:
        palette = theme.PALETTE
        percent_left = max(0, round(100 * (1 - context_used / max_context))) if max_context else 100

        line = Text()
        line.append(shown_name, style=f"bold {palette['status']}")
        line.append(f" · effort {effort}", style=palette["muted"])
        line.append(f" · {mode}", style=palette["accent2"] if mode == "write" else palette["muted"])
        line.append("  │  ", style=palette["border"])
        line.append("in ", style=palette["muted"])
        line.append(_fmt_tokens(total_in), style=palette["text"])
        line.append(" · out ", style=palette["muted"])
        line.append(_fmt_tokens(total_out), style=palette["text"])
        line.append("  │  ", style=palette["border"])
        line.append("ctx ", style=palette["muted"])
        line.append(f"{_fmt_tokens(context_used)}/{_fmt_tokens(max_context)}", style=palette["text"])
        line.append(f" · {percent_left}% left", style=palette["muted"])
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

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "submit", "Send", priority=True),
        Binding("shift+enter", "newline", "Newline", priority=True, show=False),
        Binding("alt+enter", "newline", "Newline", priority=True, show=False),
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
            return
        self.load_text("")
        self.post_message(self.ChatSubmitted(text))

    def action_newline(self) -> None:
        self.insert("\n")

    def insert_mention(self, value: str) -> None:
        row, col = self.cursor_location
        line = self.document.get_line(row)
        prefix = line[:col]
        at_col = prefix.rfind("@")
        if at_col == -1:
            return
        self.replace(f"@{value} ", (row, at_col), (row, col))
        self.popup_active = None
