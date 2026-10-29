"""Virtualized transcript: items + presenter for the FlowView-backed chat log.

The chat log is a ``textual_flowview.FlowView``: it paints only the visible
rows (O(viewport) — one widget regardless of transcript length), so a 1k+
message session scrolls as smoothly as a fresh one. This module owns the
*items* (the transcript's data — one per user message, assistant answer,
thinking block, tool call, notice, …) and the :class:`TranscriptPresenter`
that turns each into a :class:`~textual_flowview.Presentation`.

Interactions that were widget behaviors before the migration map onto
FlowView's model:

* click-to-expand tool calls / thinking blocks — ``AgentApp.on_flow_view_clicked``
  flips the item's ``expanded`` flag and calls ``entry.update()``;
* streaming — deltas mutate the item and bump the entry; the presenter keeps
  per-item strip caches so a flush re-renders only the live tail;
* the shimmer ("Thinking", "Running <tool>", "Processing") — pure style:
  the sweep phase is derived from the shared wall-time clock at render time
  (``shimmer_now``); ``FlowView.animate_entry`` ticks only drive repaints
  (and only while the entry is on screen).
"""

from __future__ import annotations

import io
import re
import time
import traceback
from functools import singledispatch
from pathlib import Path
from typing import ClassVar

from rich import box
from rich.console import Console
from rich.markdown import Heading as RichHeading
from rich.markdown import Markdown as RichMarkdown
from rich.markdown import TableElement as RichTableElement
from rich.padding import Padding
from rich.segment import Segment
from rich.style import Style
from rich.table import Table
from rich.text import Text
from textual.strip import Strip
from textual_flowview import Presentation

from .. import images, theme
from ..images import token_text
from .tool_view import (
    ToolCallView, body_renderable, live_parts as _stream_parts,
    summary_text as _tool_summary_text,
)
from .transcript import (
    AssistantItem, DividerItem, LineItem, PendingItem, ThinkingItem, ToolItem,
    ToolStatus, UserItem,
)
from ..theme import blend_hex as _lerp_hex
from .widgets import highlight_code

__all__ = [
    "AssistantItem", "DividerItem", "LineItem", "PendingItem", "Presenter",
    "shimmer_now",
    "ThinkingItem", "ToolItem", "ToolStatus", "UserItem",
    "render_strips",
]


# ---------------------------------------------------------------------------
# strip rendering

_console = Console(file=io.StringIO(), legacy_windows=False, color_system="truecolor")
_theme_pushed = False


def push_rich_theme() -> None:
    """Re-point rich.markdown's style names at the current palette (pop the
    previous theme so repeated /theme switches don't grow the stack)."""
    global _theme_pushed
    if _theme_pushed:
        _console.pop_theme()
    _console.push_theme(theme.rich_markdown_theme())
    _theme_pushed = True


push_rich_theme()


def render_strips(renderable, width: int) -> list[Strip]:
    """Render a Rich renderable into body strips of exactly ``width`` cells."""
    if width <= 0:
        return []
    options = _console.options.update(width=width, min_width=width, max_width=width)
    lines = _console.render_lines(renderable, options, pad=True)
    return [Strip(line, width) for line in lines]


_RENDER_LOG = Path("~/.local/state/xarness/tui-render.log").expanduser()


def _log_render_traceback(item, exc: BaseException) -> None:
    """Append the traceback of a failed item render to a log file.

    FlowView's error panel keeps only `{type}: {exc}`; without this the
    traceback is lost and render bugs can't be diagnosed from a crash
    report. Never raises."""
    try:
        _RENDER_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_RENDER_LOG, "a", encoding="utf-8") as f:
            f.write(f"--- render failed for {type(item).__name__} ---\n")
            f.write(traceback.format_exc(limit=50))
    except BaseException:  # noqa: BLE001 - logging must never raise
        pass


def shimmer_now(span: float) -> float:
    """Shared shimmer clock — phase derived from wall time, so the sweep is
    pure presentation: no per-item state to advance, reset, or leak."""
    return (time.monotonic() * theme.SHIMMER_SPEED) % span


def shimmer_text(
    label: str, base_color: str, peak_color: str, phase: float, suffix: str = "",
) -> Text:
    """One frame of the shimmer sweep (the former ShimmerText widget)."""
    text = Text()
    peak = phase - theme.SHIMMER_BAND_WIDTH
    half_band = max(theme.SHIMMER_BAND_WIDTH / 2, 1e-6)
    for i, ch in enumerate(label):
        dist = abs(i - peak)
        intensity = max(0.0, 1.0 - dist / half_band)
        # ``blend_hex(fg, bg, alpha)`` weights its first argument, so pass
        # the peak first to get the base at intensity 0 and peak at 1.
        color = _lerp_hex(peak_color, base_color, intensity)
        text.append(ch, style=Style(color=color, bold=intensity > 0.6))
    if suffix:
        suffix_color = _lerp_hex(peak_color, base_color, 0.35)
        text.append(" " + suffix, style=Style(color=suffix_color))
    return text


_GLYPHS = {"error": "✗ ", "warn": "! ", "success": "✓ "}


# ---------------------------------------------------------------------------
# presenter


class Presenter:
    """FlowPresenter for the transcript items. Cheap by construction: each
    item caches its rendered strips and a flush re-renders only what changed
    (the live tail), so ``entry.update()`` at streaming rate stays bounded."""

    def __init__(self) -> None:
        self.epoch = 0  # bumped on /theme; part of every cache key
        # Per-entry strip caches live here, not on the (pure-data) items:
        # id(entry) -> (key, strips), and a per-entry chunk cache for the
        # settled assistant blocks.
        self._cache: dict[int, tuple] = {}
        self._chunks: dict[int, dict] = {}
        # (len(args), live label parts) — streaming args are re-parsed only
        # when the text grows, not on every shimmer frame.
        self._args_cache: dict[int, tuple] = {}

    def invalidate(self) -> None:
        self.epoch += 1
        self._cache.clear()
        self._chunks.clear()
        self._args_cache.clear()
        push_rich_theme()

    def reset(self) -> None:
        """Forget all cached strips (the transcript was rebuilt)."""
        self._cache.clear()
        self._chunks.clear()
        self._args_cache.clear()
        self._index_cache = None

    async def present(self, entry, width: int) -> Presentation:
        item = entry.item
        try:
            strips = render_item(item, self, entry, width)
        except Exception as exc:
            # FlowView only surfaces `{type}: {exc}` — without the traceback
            # a render bug is undebuggable. Log the full trace (re-raise so
            # the entry still shows the error panel; this is NOT a fallback).
            _log_render_traceback(item, exc)
            raise
        # Every non-user row carries the old `padding: 0 1` left inset.
        # User rows are the full-width surface box and skip it (a transparent
        # gutter column would punch a hole in the box's background).
        # Cells MUST carry an explicit bg: FlowView's strip path does not
        # route transparent cells through the widget CSS, so bg-less cells
        # fall through to the TERMINAL's default colors (dark bg / light fg
        # on most terminals) — indistinguishable from the dark palettes by
        # luck, glaringly wrong on light ones.
        base_style = Style(color=theme.PALETTE["text"], bgcolor=theme.PALETTE["bg"])
        if not item.full_width:
            inset = Strip.blank(1, base_style)
            strips = [
                Strip.join([inset, s]).adjust_cell_length(width) for s in strips
            ]
        # Vertical rhythm (the old widget CSS is ignored here — the presenter
        # paints raw strips, so it owns the spacing):
        if item.full_width:
            # Old UserMessage CSS: `padding: 1 0` (surface-coloured blank row
            # inside the box, above and below the text).
            pad = Strip.blank(width, Style(bgcolor=theme.PALETTE["surface"]))
            strips = [pad, *strips, pad]
        leading = item.leading_blanks(self._previous_item(entry))
        if leading:
            # Styled: unstyled filler cells make the selection highlighter
            # crash (None + Style) when a drag spans the blank row.
            strips = [
                Strip.blank(width, base_style) for _ in range(leading)
            ] + strips
        # Give every cell an explicit palette base (fg AND bg). Cell styles
        # win, so user pads, diff tints and syntax colors are kept. The
        # drag-selection highlight is unaffected: ChatFlowView.render_line
        # re-applies the selection span with cell styles overridden.
        strips = [s.apply_style(base_style) for s in strips]
        return Presentation(height=len(strips), strips=strips)

    def _previous_item(self, entry) -> object | None:
        """The item immediately above ``entry`` (O(1) per entry via index cache)."""
        try:
            flat = entry._model.entries
        except AttributeError:
            return None
        cache = getattr(self, "_index_cache", None)
        if cache is None or cache[0] is not flat:
            cache = (flat, {id(e): i for i, e in enumerate(flat)})
            self._index_cache = cache
        i = cache[1].get(id(entry), 0)
        return flat[i - 1].item if i else None

    # -- user ---------------------------------------------------------------

    def _user(self, entry, width: int) -> list[Strip]:
        item: UserItem = entry.item
        key = (self.epoch, width, item.text, item.undo_armed, len(item.images))

        def build():
            p = theme.PALETTE
            body = Text(item.text, style=p["text"])
            token_style = Style(bgcolor=p["border"], color=p["muted"])
            for match in images.IMAGE_TOKEN_RE.finditer(item.text):
                body.stylize(token_style, match.start(), match.end())
            rows: list = [body]
            for image in item.images:
                if token_text(image) in item.text:
                    continue  # the token itself already shows the attachment
                rows.append(Text(token_text(image), style=p["muted"]))
            if item.undo_armed and item.message_id:
                rows.append(Text("↩ undo to here", style=f"italic {p['accent']}"))
            # First row carries the "›" marker; wrapped lines hang-indent under
            # it. Every row gets the surface background (the old UserMessage
            # CSS: `background: $c-surface`), padded to the full view width.
            bg = Style(bgcolor=p["surface"])
            strips: list[Strip] = []
            for i, r in enumerate(rows):
                block = _prefix_strips(render_strips(r, width - 2), width)
                if i == 0:
                    block[0] = Strip.join([
                        Strip([_seg("› ", p["muted"])], 2),
                        block[0].crop(2, width),
                    ])
                strips.extend(s.apply_style(bg) for s in block)
            return strips

        return _cached(self, entry, key, build)

    # -- assistant ------------------------------------------------------------

    def _assistant(self, entry, width: int) -> list[Strip]:
        item: AssistantItem = entry.item
        p = theme.PALETTE
        body_w = width - 2
        body: list[Strip] = []
        chunks = self._chunks.setdefault(id(entry), {})
        if len(chunks) > 2_000:
            chunks.clear()
        for i, chunk in enumerate(item.chunks):
            # A heading on the very first line of the answer gets no blank
            # line above it; a heading that opens a later chunk does.
            pad_leading = bool(body)
            key = (i, self.epoch, width, pad_leading)
            cached = chunks.get(key)
            if cached is None:
                cached = _settled_strips(chunk, body_w, pad_leading=pad_leading)
                chunks[key] = cached
            body.extend(cached)
        tail = item.live.rstrip()
        if tail:
            body.extend(render_strips(Text(tail, style=p["text"]), body_w))
        if not body:
            return []
        # Marker sits on the first body line ('• text…'), wrapped lines
        # hang-indent under it at column 2 — the old assistant-marker row.
        body = _prefix_strips(body, width)
        marker = Strip.join([
            Strip([_seg("• ", p["muted"])], 2),
            body[0].crop(2, width),
        ])
        return [marker, *body[1:]]

    # -- thinking ---------------------------------------------------------

    def _thinking(self, entry, width: int) -> list[Strip]:
        item: ThinkingItem = entry.item
        p = theme.PALETTE
        # Shimmer phase derives at render time; settled items keep a stable
        # cache key (phase None).
        phase = None if item.done else shimmer_now(
            len(item.summary_text()) + 2 * theme.SHIMMER_BAND_WIDTH
        )
        key = (self.epoch, width, item.expanded, len(item.reasoning),
               item.done, item.duration, phase)

        def build():
            if item.done:
                pill = Text(
                    item.summary_text(),
                    style=f"italic {p['muted']}",
                )
            else:
                pill = shimmer_text(
                    item.summary_text(), *theme.SHIMMER_THINKING, phase=phase,
                )
            strips = render_strips(pill, width)
            if item.expanded:
                strips.extend(_prefix_strips(
                    render_strips(
                        Text(item.reasoning, style=f"italic {p['reasoning']}"),
                        width - 2,
                    ),
                    width,
                ))
            return strips

        return _cached(self, entry, key, build)

    # -- tool call ----------------------------------------------------------

    def _tool(self, entry, width: int) -> list[Strip]:
        item: ToolItem = entry.item
        p = theme.PALETTE
        label = f"Running {item.name}"
        phase = (
            shimmer_now(len(label) + 2 * theme.SHIMMER_BAND_WIDTH)
            if item.status is ToolStatus.MAKING_CALL else None
        )
        key = (self.epoch, width, item.expanded, item.status, len(item.args),
               len(item.output), phase)
        view = ToolCallView(
            name=item.name, args=item.args, output=item.output,
            status=item.status, expanded=item.expanded,
        )
        if item.status is ToolStatus.MAKING_CALL:
            cached = self._args_cache.get(id(entry))
            if cached is None or cached[0] != len(item.args):
                cached = (len(item.args), _stream_parts(item.name, item.args))
                self._args_cache[id(entry)] = cached
            live_name, live_detail = cached[1]

        def build():
            if item.status is ToolStatus.MAKING_CALL:
                name, detail = live_name, live_detail
                dot = Text("•", style=p["warning"])
                head = shimmer_text(
                    f"Running {name}", *theme.SHIMMER_THINKING,
                    phase=phase, suffix=detail,
                )
            else:
                dot = Text("•", style=p[item.STATUS_COLOR_KEY[item.status]])
                head = _tool_head(item, view)
            dot_strips = render_strips(Text.assemble(dot, " ", head), width)
            strips = list(dot_strips)
            if item.expanded:
                strips.extend(_prefix_strips(
                    render_strips(body_renderable(view), width - 2), width,
                    _seg("│ ", p["border"]),
                ))
            return strips

        return _cached(self, entry, key, build)

    # -- the pending strip ---------------------------------------------------

    def _shimmering(self, item: PendingItem, width: int) -> list[Strip]:
        p = theme.PALETTE
        head = shimmer_text(
            item.label, *theme.SHIMMER_PROCESSING,
            phase=shimmer_now(len(item.label) + 2 * theme.SHIMMER_BAND_WIDTH),
        )
        if item.hint:
            head.append(f"  ({item.hint})", style=p["muted"])
        return render_strips(head, width)

    # -- one-liners -----------------------------------------------------------

    def _line(self, item: LineItem, width: int) -> list[Strip]:
        p = theme.PALETTE
        color_key = {
            "error": "error", "warn": "warning", "success": "success",
            "info": "muted", "hint": "muted",
        }.get(item.kind, "muted")
        glyph = _GLYPHS.get(item.kind, "")
        text = Text(f"{glyph}{item.text}", style=p[color_key])
        return render_strips(text, width)


class _BorderedTableElement(RichTableElement):
    """Rich's markdown table, drawn as a real boxed table.

    rich.markdown defaults to ``box.SIMPLE`` (no outer or column borders),
    which reads as loose text in the transcript. Render through rich's
    ``Table`` with a rounded border instead, keeping the palette's
    ``markdown.table.border``/``markdown.table.header`` styles and the
    cell wrapping a narrow transcript needs."""

    def __rich_console__(self, console, options):
        table = Table(
            box=box.ROUNDED,
            style="markdown.table.border",
            show_edge=True,
            show_lines=True,
            collapse_padding=False,
        )

        if self.header is not None and self.header.row is not None:
            for column in self.header.row.cells:
                heading = column.content.copy()
                heading.stylize("markdown.table.header")
                # overflow="fold" wraps cell text instead of ellipsizing it.
                table.add_column(heading, justify="left", overflow="fold")

        if self.body is not None:
            for row in self.body.rows:
                table.add_row(*[cell.content for cell in row.cells])

        # Leave the last cell of the line blank: a table that reaches the
        # terminal's final column sits in the pending-wrap state, and some
        # terminals drop/overwrite that last glyph (usually a right border).
        yield Padding(table, (0, 1, 0, 0))


class _Heading(RichHeading):
    """Left-aligned headings.

    rich's Heading implementation differs between versions (14.1 renders h1
    as a centered Panel; later versions center it via LEVEL_ALIGN), so
    implement the small TextElement contract directly instead of relying on
    either.
    """

    @classmethod
    def create(cls, markdown, token):
        return cls(token.tag)

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.style_name = f"markdown.{tag}"

    def on_enter(self, context) -> None:
        self.text = Text()
        context.enter_style(self.style_name)

    def on_leave(self, context) -> None:
        context.leave_style()

    def __rich_console__(self, console, options):
        text = self.text.copy()
        text.justify = "left"
        yield text


class _Markdown(RichMarkdown):
    """rich.markdown with boxed tables and left-aligned headings."""

    elements: ClassVar[dict[str, type]] = dict(
        RichMarkdown.elements,
        heading_open=_Heading,
        table_open=_BorderedTableElement,
    )


_HEADING_LINE_RE = re.compile(r"(?m)^(?=#{1,6}[ \t])")


def _settled_strips(
    text: str, width: int, *, pad_leading: bool = False,
) -> list[Strip]:
    """Render a settled assistant chunk: markdown prose plus highlighted
    fences.

    rich.markdown's inter-block blank lines are dropped so prose stays
    contiguous, except that each heading gets one blank line above it
    (padding-top: 1). Headings are split into their own parts so that blank
    can be inserted independently of rich's version-specific heading layout.
    ``pad_leading`` controls whether a heading on the chunk's first line gets
    that padding (False for the first line of an answer).
    """
    # (renderable, is_prose, blank_line_before)
    parts: list[tuple[object, bool, bool]] = []
    pos = 0

    def add_prose(chunk: str) -> None:
        prose = chunk.strip("\n")
        if not prose.strip():
            return
        # rich.markdown collapses single newlines into spaces; make them
        # hard breaks so "line1\nline2" renders as two lines.
        prose = re.sub(r"(?<!\n)\n(?!\n)", "  \n", prose)
        seen_content = pad_leading or bool(parts)
        for segment in _HEADING_LINE_RE.split(prose):
            if not segment.strip():
                continue
            is_heading = _HEADING_LINE_RE.match(segment) is not None
            parts.append((_Markdown(segment), True, is_heading and seen_content))
            seen_content = True

    for match in re.finditer(r"```([^\n`]*)\n(.*?)```", text, re.DOTALL):
        add_prose(text[pos: match.start()])
        parts.append((
            highlight_code(match.group(2), language=match.group(1).strip()),
            False,
            False,
        ))
        pos = match.end()
    add_prose(text[pos:])

    strips: list[Strip] = []
    for part, is_prose, pad_top in parts:
        rendered = render_strips(part, width)
        if is_prose:
            # rich.markdown emits a blank line between every block
            # (paragraphs, headings, lists...). Drop all blank strips so
            # prose renders contiguously.
            rendered = [s for s in rendered if s.text.strip()]
        else:
            while rendered and not rendered[0].text.strip():
                rendered = rendered[1:]
            while rendered and not rendered[-1].text.strip():
                rendered = rendered[:-1]
        if pad_top and rendered:
            strips.append(Strip.blank(width))
        strips.extend(rendered)
    return strips


def _tool_head(item: ToolItem, view: ToolCallView) -> Text:
    """The settled tool line: verb + name, plus a one-line detail when the
    summary adds more than the bare tool name."""
    verb = item.STATUS_VERB[item.status]
    summary = Text(f"{verb} {item.name}" if verb else item.name)
    if item.status is ToolStatus.CALL_SUCCEEDED and not (
        item.expanded and item.name == "run_bash"
    ):
        detail = _tool_summary_text(view)
        if detail.plain != item.name:
            summary.append_text(detail[len(item.name):])
    return summary


@singledispatch
def render_item(item, presenter, entry, width: int) -> list[Strip]:
    return render_strips(Text(repr(item)), width)


@render_item.register
def _(item: UserItem, presenter, entry, width: int) -> list[Strip]:
    return presenter._user(entry, width)


@render_item.register
def _(item: AssistantItem, presenter, entry, width: int) -> list[Strip]:
    return presenter._assistant(entry, width)


@render_item.register
def _(item: ThinkingItem, presenter, entry, width: int) -> list[Strip]:
    return presenter._thinking(entry, width)


@render_item.register
def _(item: ToolItem, presenter, entry, width: int) -> list[Strip]:
    return presenter._tool(entry, width)


@render_item.register
def _(item: PendingItem, presenter, entry, width: int) -> list[Strip]:
    return presenter._shimmering(item, width)


@render_item.register
def _(item: LineItem, presenter, entry, width: int) -> list[Strip]:
    return presenter._line(item, width)


@render_item.register
def _(item: DividerItem, presenter, entry, width: int) -> list[Strip]:
    return render_strips(Text(item.text, style=theme.PALETTE["accent"]), width)


def _cached(presenter, entry, key, build):
    """Per-entry strip cache: return the strips cached under ``key``, or
    build (and cache) them. A change of width/epoch/content misses."""
    cached = presenter._cache.get(id(entry))
    if cached is not None and cached[0] == key:
        return cached[1]
    value = build()
    presenter._cache[id(entry)] = (key, value)
    return value


def _prefix_strips(
    strips: list[Strip], width: int, prefix_seg: Segment | None = None,
) -> list[Strip]:
    """Prefix already-rendered body strips (rendered at ``width - 2``) with a
    two-cell column to ``width``: a blank indent when ``prefix_seg`` is None
    (the hanging indent under a marker row), otherwise the given two-cell
    segment — e.g. the '│ ' expand bar of an expanded tool body."""
    prefix = Strip.blank(2) if prefix_seg is None else Strip([prefix_seg], 2)
    return [Strip.join([prefix, s]).adjust_cell_length(width) for s in strips]


def _seg(text: str, style) -> Segment:
    return Segment(text, Style.parse(style) if isinstance(style, str) else style)
