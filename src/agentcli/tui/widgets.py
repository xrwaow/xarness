"""Widgets for the agentcli chat TUI.

Widgets are dumb views: they render what the app feeds them from stream
events and hold no conversation state of their own (the reasoning text inside
``ThinkingBlock`` is display state only; the authoritative copy lives in
``ChatController.conversation``).
"""

from __future__ import annotations

import re
import time
from typing import ClassVar

from rich.style import Style
from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Markdown, Static, TextArea
from textual.widgets.text_area import TextAreaTheme

from .. import theme


_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)

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

    def set_label(self, label: str) -> None:
        """Swap the label and restart the sweep from the leading edge."""
        self.label = label
        self.phase = 0.0
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


class UserMessage(Static):
    """A submitted user message: full-width surface highlight, dim marker."""

    def __init__(self, text: str) -> None:
        super().__init__("", classes="msg user", markup=False)
        self._text = text
        self._apply()

    @property
    def text(self) -> str:
        return self._text

    def _apply(self, suffix: str = "") -> None:
        line = Text()
        line.append("› ", style=theme.PALETTE["muted"])
        line.append(self._text, style=theme.PALETTE["text"])
        if suffix:
            line.append(suffix, style=theme.PALETTE["muted"])
        self.update(line)

    def mark_queued(self) -> None:
        """Note that this message is waiting for the current turn to finish."""
        self._apply("   (queued)")


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

    def finish(self, duration: float | None) -> None:
        """Collapse to a static summary. First call wins — app.py calls this
        twice (once when content starts, once at turn-complete); a later call
        is timed against full generation finishing, not reasoning ending, so
        it's ignored once we already have a duration."""
        if self._done:
            return
        self._done = True
        self._duration = duration if duration is not None else time.monotonic() - self._start
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
        duration = f"{self._duration:.1f}s" if self._duration is not None else "…"
        elapsed_static.update(Text(f"Thought for {duration}", style=f"italic {theme.PALETTE['muted']}"))


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


class ErrorLine(Static):
    """A request-level failure, rendered inline in the scrollback."""

    def __init__(self, message: str) -> None:
        super().__init__(Text(f"✗ {message}", style=theme.PALETTE["error"]), classes="msg error")


class StatusBar(Static):
    """Persistent footer: model, effort, session tokens, context headroom."""

    def update_status(
        self,
        *,
        shown_name: str,
        effort: str,
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
    """Persistent multi-line input bar: Enter submits, Shift+Enter newlines."""

    class ChatSubmitted(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "submit", "Send", priority=True),
        Binding("shift+enter", "newline", "Newline", priority=True, show=False),
        Binding("alt+enter", "newline", "Newline", priority=True, show=False),
    ]

    def on_mount(self) -> None:
        base = TextAreaTheme.get_builtin_theme("css")
        no_line_highlight = TextAreaTheme(
            name="agentcli-input",
            base_style=base.base_style,
            gutter_style=base.gutter_style,
            cursor_style=base.cursor_style,
            cursor_line_style=Style(),
            cursor_line_gutter_style=base.cursor_line_gutter_style,
            bracket_matching_style=base.bracket_matching_style,
            selection_style=base.selection_style,
            syntax_styles=base.syntax_styles,
        )
        self.register_theme(no_line_highlight)
        self.theme = "agentcli-input"

    def action_submit(self) -> None:
        text = self.text.strip()
        if not text:
            return
        self.load_text("")
        self.post_message(self.ChatSubmitted(text))

    def action_newline(self) -> None:
        self.insert("\n")
