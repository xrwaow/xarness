"""Widgets for the xarness chat TUI.

Widgets are dumb views: they render what the app feeds them from stream
events and hold no conversation state of their own (the reasoning text inside
``ThinkingBlock`` is display state only; the authoritative copy lives in
``ChatController.conversation``).
"""

from __future__ import annotations

import random
import re

from rich.console import RenderableType
from rich.style import Style
from rich.text import Text
from textual.containers import Vertical
from textual.content import Content
from textual.highlight import highlight
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from .. import theme
from ..themes import blend_hex


def _fmt_tokens(count: int) -> str:
    if count < 1_000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1_000:.1f}k"
    return f"{count / 1_000_000:.2f}M"


def _content_to_rich_text(content: Content) -> Text:
    """Convert a highlighted Content into rich Text with concrete colors.

    Content span styles are Textual markup ("$c-accent", "80%" alpha,
    "ansi_red", …) that only resolve when Textual renders the Content
    through the app's CSS variables. The flow transcript renders through
    rich instead, where they would fall back to black — so resolve them
    against the palette here: variables to hex, a trailing percentage to a
    blend toward the background, ansi_* names to rich color names."""
    p = theme.PALETTE

    def resolve(style: str) -> str:
        out: list[str] = []
        last_color: str | None = None
        for token in style.split():
            if token.endswith("%") and last_color is not None:
                out[-1] = blend_hex(last_color, p["bg"], float(token[:-1]) / 100)
                continue
            if token.startswith("$c-"):
                last_color = p.get(token[3:], p["text"])
                out.append(last_color)
            elif token == "$text":
                last_color = p["text"]
                out.append(last_color)
            elif token.startswith("ansi_"):
                out.append(token[5:])
            else:
                out.append(token)
        return " ".join(out)

    text = Text(content.plain)
    for span in content.spans:
        try:
            text.stylize(Style.parse(resolve(span.style)), span.start, span.end)
        except Exception:  # noqa: BLE001 - an unparsable style beats no code
            pass
    return text


def highlight_code(code: str, path: str | None = None, language: str = "") -> Text:
    """Highlight a code block the same way assistant-message fences do.

    Tool-call bodies (read_file/write_file) must match the LLM message
    fences exactly, including the ANSI-mode branch: on terminals without
    truecolor, the palette-driven CodeHighlightTheme quantizes its hex
    colors down to the 16-color ANSI palette and tokens collapse to
    black-on-dark. Reading the flags off the active app keeps the two
    paths in lockstep. Returns resolved rich Text (not Content) because
    the flow transcript renders through rich, where Content's $c-*
    variable styles never resolve.
    """
    from textual.app import active_app
    from textual.highlight import ANSIDarkHighlightTheme, ANSILightHighlightTheme

    app = active_app.get()
    if app.native_ansi_color:
        code_theme = (
            ANSIDarkHighlightTheme if app.current_theme.dark
            else ANSILightHighlightTheme
        )
    else:
        code_theme = theme.CodeHighlightTheme
    content = highlight(
        code, language=language or None, path=path, theme=code_theme,
    )
    return _content_to_rich_text(content)


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
            # Only the oldest message is "first in line" — enter sends it;
            # ctrl+shift+backspace discards it.
            self.mount(Static(
                "↵ enter to send now   ctrl+shift+backspace to discard",
                classes="steer-hint",
                markup=False,
            ))
        else:
            self.remove_class("visible")



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
        body = highlight_code(output, path=path)
        body.append("\n" + "\n".join(notes), style=theme.PALETTE["muted"])
        return body
    return highlight_code(output, path=path)


def _split_tool_diff(text: str) -> tuple[str, str] | None:
    """Split tool output into ``(prose, unified diff)`` at its first
    ``diff --git`` line; ``None`` when the output isn't a diff."""
    match = _DIFF_GIT_RE.search(text)
    if match is None:
        return None
    return text[: match.start()], text[match.start() :]


_LEGACY_DIFF_BREAK_RE = re.compile(
    r"(?<!\n)(?=(?:--- [ab]/|\+\+\+ [ab]/|@@ -\d))"
)
_LEGACY_HUNK_END_RE = re.compile(
    r"(@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@)(?=[^\n])"
)


def _normalize_legacy_diff(text: str) -> str:
    """Re-break the malformed one-line diffs older edit_file calls stored.

    A previous generator joined ``difflib`` output with ``""`` after passing
    keepends lines, collapsing the ``---``/``+++``/``@@`` headers and the
    first changed line together. Replaying such a session must still render
    the hunk, so split those headers back onto their own lines. Valid diffs
    are untouched (their headers already follow a newline).
    """
    text = _LEGACY_DIFF_BREAK_RE.sub("\n", text)
    return _LEGACY_HUNK_END_RE.sub(r"\1\n", text)


def _render_diff(text: str, show_headers: bool = True) -> Text:
    """Unified diff as GitHub-style rich text.

    Each file collapses to one `── path` header with new/deleted/renamed
    badges (the index/mode/---/+++ prologue is parsed and dropped, binary
    changes become a short notice); +/- lines get a full-line background
    tint from the theme's diff colors, with old/new line numbers in a
    muted gutter. ``show_headers=False`` drops the per-file header line —
    used for edit_file, whose path is already in the summary row. Plain
    stateful parse — no external diff library."""
    text = _normalize_legacy_diff(text)
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
        return blend_hex(theme.PALETTE["bg"], theme.PALETTE["accent"], t)

    def on_mount(self) -> None:
        self._cols = [random.randrange(self.LEVELS) for _ in range(self.WIDTH)]
        self.set_interval(self.TICK, self._tick)

    def _tick(self) -> None:
        if not self.is_mounted:
            return
        if not self.has_class("active"):
            return  # hidden: don't churn repaints while the user scrolls
        top = self.LEVELS - 1
        for i in range(self.WIDTH):
            self._cols[i] = max(0, min(top, self._cols[i] + random.choice((-1, 0, 1))))
        line = Text(" ")
        for level in self._cols:
            line.append(self.GLYPH, style=self._level_color(level))
        self.update(line)


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
        self.update(line)


# ChatInput (and its image-token machinery) live in chat_input.py;
# DiffSummary/DiffText in diff_summary.py. Re-exported here for the
# historical import path.
from .chat_input import ChatInput  # noqa: E402,F401
from .diff_summary import DiffSummary, DiffText  # noqa: E402,F401
