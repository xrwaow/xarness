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
from typing import ClassVar, cast

from rich.console import RenderableType
from rich.style import Style
from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.highlight import highlight
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import OptionList, Static, TextArea
from textual.widgets.markdown import MarkdownFence
from textual.widgets.option_list import Option
from textual.widgets.text_area import Selection, TextAreaTheme

from .. import images
from .. import theme
from ..protocol import DiffInfo as DiffStat  # same shape as gitwork.DiffStat
from ..images import ImageAttachment, ImageError, token_text


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


def _content_to_rich_text(content: Content) -> Text:
    """Convert a highlighted Content into rich Text with concrete colors.

    Content span styles are Textual markup ("$c-accent", "80%" alpha,
    "ansi_red", …) that only resolve when Textual renders the Content
    through the app's CSS variables. The flow transcript renders through
    rich instead, where they would fall back to black — so resolve them
    against the palette here: variables to hex, a trailing percentage to a
    blend toward the background, ansi_* names to rich color names."""
    p = theme.PALETTE

    def blend(color: str, pct: float) -> str:
        factor = pct / 100.0
        bg = p["bg"].lstrip("#")
        fg = color.lstrip("#")
        mixed = "".join(
            f"{round(int(fg[i:i + 2], 16) * factor + int(bg[i:i + 2], 16) * (1 - factor)):02x}"
            for i in (0, 2, 4)
        )
        return f"#{mixed}"

    def resolve(style: str) -> str:
        out: list[str] = []
        last_color: str | None = None
        for token in style.split():
            if token.endswith("%") and last_color is not None:
                out[-1] = blend(last_color, float(token[:-1]))
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

    app = active_app.get()
    content = PaletteFence.highlight(
        code,
        language=language or None,
        ansi=app.native_ansi_color,
        dark=app.current_theme.dark,
        path=path,
    )
    return _content_to_rich_text(content)


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
        # the expanded file list — except clicks inside the open diff pane,
        # which only minimize the diff (the file list and its highlight stay
        # up, same as esc), not collapse the whole panel.
        node = event.widget
        while node is not None and node is not self:
            diff_key = getattr(node, "diff_key", _NO_DIFF)
            if diff_key is not _NO_DIFF:
                event.stop()
                self.post_message(self.DiffRequested(cast("str | None", diff_key)))
                return
            if node.id == "diff-content":
                event.stop()
                if self.diff_shown:
                    self.minimize_diff()
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
            # Row layout: name  [new]  {source}  +N -M — the whole row is
            # and requests that file's unified diff. The "new" badge gets
            # its own column so it never wraps the file name.
            row = Horizontal(classes="diff-file-row")
            row.diff_key = f.path  # type: ignore[attr-defined]
            await files.mount(row)
            row.mount(Static(Text(f.path), classes="diff-file-name", markup=False))
            if f.is_new:
                row.mount(Static(Text("new"), classes="diff-file-new", markup=False))
            # Source tag: "agent" for a turn's edits, "drift" for changes
            # made outside the session.
            source_class = "drift" if "drift" in f.source else "agent"
            row.mount(Static(
                Text(f.source), classes=f"diff-file-source {source_class}", markup=False,
            ))
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
        if not self.has_class("active"):
            return  # hidden: don't churn repaints while the user scrolls
        top = self.LEVELS - 1
        for i in range(self.WIDTH):
            self._cols[i] = max(0, min(top, self._cols[i] + random.choice((-1, 0, 1))))
        line = Text(" ")
        for level in self._cols:
            line.append(self.GLYPH, style=self._level_color(level))
        self.update(line)


class MessageLine(Static):
    """Unified one-line status message in the scrollback.

    One widget for every non-conversational line — notices, warnings,
    confirmations, errors — with ``kind`` picking the glyph and the CSS class
    ($c-muted / $c-warning / $c-success / $c-error), so theme switches recolor
    every kind live. :class:`ErrorLine` and :class:`NoticeLine` are the two
    named shorthands the rest of the app uses.
    """

    _GLYPHS = {"error": "✗ ", "warn": "! ", "success": "✓ "}

    def __init__(
        self, message: str, kind: str = "notice", hint: bool = False
    ) -> None:
        # Never mounted since the FlowView migration: the app converts a
        # MessageLine into a flow.LineItem — these attributes carry the data.
        self.message_text = message
        self.kind = kind
        self.hint = hint
        super().__init__(
            Text(f"{self._GLYPHS.get(kind, '')}{message}"),
            classes=f"msg {kind}" + (" hint" if hint else ""),
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
        def __init__(self, text: str, images: list[ImageAttachment] | None = None) -> None:
            self.text = text
            # Attached images in the order their tokens appear in the text.
            self.images = images or []
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
        # Attached images, kept in token order (the editor's image segments).
        # Synced against the document on every edit: deleting a token drops
        # its attachment.
        self.attachments: list[ImageAttachment] = []

    def on_mount(self) -> None:
        self.apply_input_theme()
        self._refresh_token_highlights()

    def _refresh_token_highlights(self) -> None:
        """Paint image tokens with the ``image-token`` style.

        TextArea draws per-range highlights from its internal map (normally
        fed by tree-sitter); with no language set we own that map. Ranges
        are byte offsets into the line (the renderer maps them back to
        codepoints), which matters because 🖼 is multi-byte. Tokens never
        span lines, so per-line scanning is enough."""
        self._highlights.clear()
        for row in range(self.document.line_count):
            line = self.document.get_line(row)
            for match in images.IMAGE_TOKEN_RE.finditer(line):
                self._highlights[row].append(
                    (
                        len(line[: match.start()].encode("utf-8")),
                        len(line[: match.end()].encode("utf-8")),
                        "image-token",
                    )
                )

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
            syntax_styles={
                **base.syntax_styles,
                # Image attachment tokens ([🖼 name W×H]) get a dimmed chip
                # look, matching .user-image-chip in the transcript.
                "image-token": Style(bgcolor=theme.PALETTE["border"], color=theme.PALETTE["muted"]),
            },
        )
        self.register_theme(no_line_highlight)
        # Assigning the same name wouldn't re-run the theme watcher, so bounce
        # through the builtin to force the new colors to be applied.
        self.theme = "css"
        self.theme = "xarness-input"

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._check_triggers()
        self._sync_attachments()
        self._refresh_token_highlights()

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

    def action_newline(self) -> None:
        self.insert("\n")

    def action_toggle_mode(self) -> None:
        self.post_message(self.ModeToggle())

    def _on_paste(self, event: events.Paste) -> None:
        """A pasted path or ``file://`` URI naming an image attaches it.

        Terminals (kitty, iTerm2, ...) paste a clipboard image as a file
        URI or path; the image becomes a ``[🖼 name W×H]`` token in the text
        (deletable as one unit). Raw image bytes cannot cross a plain
        terminal, so there is nothing to intercept when the paste event
        never fires.
        """
        if "\n" in event.text:
            return
        path = images.parse_image_path(event.text)
        if path is None or not images.is_image_path(path) or not path.is_file():
            return
        try:
            attachment = images.load_image(path)
        except ImageError:
            return  # not a decodable image: paste the text as-is
        event.prevent_default()
        event.stop()
        self.insert_image(attachment)

    def insert_image(self, attachment: ImageAttachment) -> None:
        """Attach an image and insert its placeholder token at the cursor."""
        if attachment not in self.attachments:
            self.attachments.append(attachment)
        self.insert(token_text(attachment))

    def _sync_attachments(self) -> None:
        """Drop attachments whose token is no longer in the document."""
        text = self.text
        self.attachments = [
            attachment for attachment in self.attachments
            if token_text(attachment) in text
        ]

    def _ordered_attachments(self) -> list[ImageAttachment]:
        """Attachments in the order their tokens appear in the text."""
        by_token: dict[str, ImageAttachment] = {}
        for attachment in self.attachments:
            by_token.setdefault(token_text(attachment), attachment)
        return [
            by_token[token] for token in images.find_image_tokens(self.text)
            if token in by_token
        ]

    # -- atomic image tokens -------------------------------------------

    def _plain_cursor_offset(self) -> int:
        """The cursor as a plain-text offset into the document."""
        row, col = self.cursor_location
        return sum(len(self.document.get_line(r)) + 1 for r in range(row)) + col

    def _offset_to_location(self, offset: int) -> tuple[int, int]:
        for row in range(self.document.line_count):
            line_len = len(self.document.get_line(row))
            if offset <= line_len:
                return (row, offset)
            offset -= line_len + 1
        return self.document.end

    def _delete_token_at_cursor(self, direction: int) -> bool:
        """Backspace/Delete at an image token's edge: remove the whole token
        (and its attachment) as if it were one character."""
        if self.selected_text:
            return False
        offset = self._plain_cursor_offset()
        for match in images.IMAGE_TOKEN_RE.finditer(self.text):
            start, end = match.span()
            hit = start < offset <= end if direction < 0 else start <= offset < end
            if not hit:
                continue
            start_loc = self._offset_to_location(start)
            end_loc = self._offset_to_location(end)
            self.delete(start_loc, end_loc, maintain_selection_offset=False)
            self.selection = Selection(start_loc, start_loc)
            self._sync_attachments()
            return True
        return False

    def action_delete_left(self) -> None:
        if self._delete_token_at_cursor(-1):
            return
        super().action_delete_left()

    def action_delete_right(self) -> None:
        if self._delete_token_at_cursor(1):
            return
        super().action_delete_right()

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
        images_out = self._ordered_attachments()
        self.load_text("")
        self.attachments = []
        self.post_message(self.ChatSubmitted(text, images_out))

    def insert_mention(self, value: str) -> None:
        """Complete an @-mention. Image files are attached instead: their
        token replaces the typed ``@query``."""
        attachment = self._try_load_mention_image(value)
        if attachment is not None:
            self.attachments.append(attachment)
            self.insert(token_text(attachment))
            self.popup_active = None
            return
        row, col = self.cursor_location
        line = self.document.get_line(row)
        prefix = line[:col]
        at_col = prefix.rfind("@")
        if at_col == -1:
            return
        self.replace(f"@{value} ", (row, at_col), (row, col))
        self.popup_active = None

    def _try_load_mention_image(self, value: str) -> ImageAttachment | None:
        path = images.parse_image_path(value)
        if path is None or not images.is_image_path(path) or not path.is_file():
            return None
        try:
            attachment = images.load_image(path)
        except ImageError:
            return None
        if attachment in self.attachments:
            return None  # already attached: let the plain @-mention go out
        return attachment
