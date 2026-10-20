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
from pathlib import Path
from dataclasses import dataclass, field

from rich.console import Console
from rich.markdown import Markdown as RichMarkdown
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual.strip import Strip
from textual_flowview import Presentation

from .. import images, theme
from ..conversation import Message as ConversationMessage
from ..events import ToolCallStatus
from ..images import token_text
from .widgets import (
    _json_tool_args,
    _lerp_hex,
    _render_diff,
    _render_read_file,
    _split_settled,
    _split_tool_diff,
    _stream_parts,
    _tool_header_detail,
    _header_detail_text,
    highlight_code,
)

__all__ = [
    "AssistantItem", "DividerItem", "LineItem", "PendingItem", "Presenter",
    "shimmer_now",
    "ThinkingItem", "ToolItem", "ToolCallStatus", "UserItem",
    "render_strips",
]


# ---------------------------------------------------------------------------
# strip rendering

_console = Console(file=io.StringIO(), legacy_windows=False, color_system="truecolor")
_console.push_theme(theme.rich_markdown_theme())


def push_rich_theme() -> None:
    """Re-point rich.markdown's style names at the current palette."""
    _console.push_theme(theme.rich_markdown_theme())


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
        color = _lerp_hex(base_color, peak_color, intensity)
        text.append(ch, style=Style(color=color, bold=intensity > 0.6))
    if suffix:
        suffix_color = _lerp_hex(base_color, peak_color, 0.35)
        text.append(" " + suffix, style=Style(color=suffix_color))
    return text


# ---------------------------------------------------------------------------
# items


@dataclass
class UserItem:
    """A submitted user message (no local echo: created on UserMessageAdded)."""

    text: str
    images: list = field(default_factory=list)
    suffix: str = ""  # "· queued" until the message is actually sent
    undo_armed: bool = False
    message: ConversationMessage | None = None
    _cache: tuple = field(default=(), repr=False)

    @classmethod
    def from_message(cls, message: ConversationMessage) -> "UserItem":
        """Build the transcript item for a persisted (or just-landed) user
        message, carrying the message itself for click-to-undo."""
        return cls(
            text=message.content,
            images=message.images or [],
            message=ConversationMessage(
                role="user", content=message.content, id=message.id or "",
            ),
        )

    def undo_row(self, height: int) -> int | None:
        """Row index of the "↩ undo to here" affordance, when armed.
        (The presenter wraps the box in blank/pad/…/pad/blank, so the last
        real row sits 2 above the bottom.)"""
        if self.undo_armed and self.message is not None:
            return height - 3
        return None


@dataclass
class AssistantItem:
    """One model answer: settled chunks (closed fences settled once) + live tail."""

    chunks: list[str] = field(default_factory=list)
    live: str = ""
    finalized: bool = False
    # (chunk index, epoch, width) -> strips. Markdown rendering dominates the
    # cost of a streaming flush; chunks never change once settled.
    chunk_cache: dict = field(default_factory=dict, repr=False)

    def append_delta(self, text: str) -> None:
        """Fold a delta into the live tail, settling any closed fence."""
        self.live += text
        settled, remainder = _split_settled(self.live)
        if settled:
            self.chunks.append(settled)
            self.live = remainder

    def finalize(self) -> None:
        """Settle the tail; idempotent (also called after errors/interrupts)."""
        if self.finalized:
            return
        self.finalized = True
        tail = self.live
        if tail.strip():
            self.chunks.append(tail)
            self.live = ""
        elif not self.chunks:
            self.chunks.append("(no output)")


@dataclass
class ThinkingItem:
    """Reasoning channel. Click toggles the text; the pill shows the time.

    Expanded while streaming (the live CoT is the interesting part), auto-
    collapsed by :meth:`finish` when the round moves on; a click toggles at
    any time — including mid-stream — so the user can tuck it away and
    reopen it. Replays construct settled items with ``expanded=False``."""

    parts: list[str] = field(default_factory=list)
    done: bool = False
    duration: float | None = None
    expanded: bool = True
    # True while the duration is provisional (unknown): a later finish with
    # the session-recorded seconds overrides it.
    _estimated: bool = False
    _cache: tuple = field(default=(), repr=False)

    @property
    def reasoning(self) -> str:
        return "".join(self.parts)

    def finish(self, duration: float | None) -> None:
        """Collapse to a summary. First backend duration wins; a later one
        overrides only an unknown ("…") duration — the live path settles
        without a local estimate, then TurnComplete fills the number in."""
        if self.done:
            if duration is not None and self._estimated:
                self.duration = duration
                self._estimated = False
            return
        self.done = True
        self.duration = duration
        self._estimated = duration is None
        self.expanded = False  # auto-collapse once the round moves on

    def summary_text(self) -> str:
        if not self.done:
            return "Thinking"
        if self.duration is not None:
            from ..util import format_duration
            return f"Thought for {format_duration(self.duration)}"
        return "Thought for …"


@dataclass
class ToolItem:
    """One tool call: status dot + summary; click expands args/output."""

    call_id: str
    name: str
    args: str = ""
    output: str = ""
    header: str = ""
    status: ToolCallStatus = ToolCallStatus.MAKING_CALL
    expanded: bool = False
    _cache: tuple = field(default=(), repr=False)

    STATUS_COLOR_KEY = {
        ToolCallStatus.MAKING_CALL: "warning",
        ToolCallStatus.PARSING_ERROR: "error",
        ToolCallStatus.CALL_SUCCEEDED: "success",
        ToolCallStatus.CALL_FAILED: "error",
    }
    STATUS_VERB = {
        ToolCallStatus.MAKING_CALL: "Running",
        ToolCallStatus.PARSING_ERROR: "Failed",
        ToolCallStatus.CALL_SUCCEEDED: "",
        ToolCallStatus.CALL_FAILED: "Failed",
    }

    def bash_command(self) -> str:
        command = _json_tool_args(self.args).get("command")
        if isinstance(command, str) and command:
            return command
        return self.header

    def body_renderable(self):
        """Port of ToolCallBlock._refresh_body — the expanded body."""
        if self.name == "edit_file":
            split = _split_tool_diff(self.output)
            if split is not None:
                return _render_diff(split[1], show_headers=False)
            return Text(self.output)
        args = _json_tool_args(self.args)
        path = args.get("path")
        if (
            self.name == "read_file" and isinstance(path, str) and path
            and self.status is ToolCallStatus.CALL_SUCCEEDED
        ):
            return _render_read_file(path, self.output)
        if self.name == "write_file":
            content = args.get("content")
            if isinstance(path, str) and path and isinstance(content, str):
                body = highlight_code(content, path=path)
                if self.output:
                    body.append("\n" + self.output)
                return body
        body = self.args
        if self.name == "run_bash":
            command = self.bash_command()
            if command:
                body = command
        if self.output:
            body = f"{body}\n{self.output}" if body else self.output
        split = _split_tool_diff(body)
        if split is None:
            return Text(body)
        prose, diff = split
        rendered = Text(prose.rstrip("\n"))
        if rendered:
            rendered.append("\n")
        rendered.append_text(_render_diff(diff))
        return rendered

    def summary_text(self) -> Text:
        verb = self.STATUS_VERB[self.status]
        summary = Text(f"{verb} {self.name}" if verb else self.name)
        if self.status is ToolCallStatus.CALL_SUCCEEDED and not (
            self.expanded and self.name in ("run_bash",)
        ):
            detail = _header_detail_text(self.name, self.header)
            if detail is None:
                detail = _tool_header_detail(self.name, self.args)
            if detail is not None:
                summary.append_text(detail)
        return summary


@dataclass
class PendingItem:
    """The ephemeral "Processing · esc to interrupt" strip, shown while a
    round waits on the provider. Owned by the app's phase machine — never
    persisted or replayed; the shimmer comes from the shared clock."""

    label: str = "Processing"
    hint: str = "esc to interrupt"
    _cache: tuple = field(default=(), repr=False)


@dataclass
class LineItem:
    """A one-line status message (notice / warn / error / success / hint)."""

    text: str
    kind: str = "notice"


@dataclass
class DividerItem:
    """A divider line (the compaction handoff marker)."""

    text: str


_GLYPHS = {"error": "✗ ", "warn": "! ", "success": "✓ "}


# ---------------------------------------------------------------------------
# presenter


class Presenter:
    """FlowPresenter for the transcript items. Cheap by construction: each
    item caches its rendered strips and a flush re-renders only what changed
    (the live tail), so ``entry.update()`` at streaming rate stays bounded."""

    def __init__(self) -> None:
        self.epoch = 0  # bumped on /theme; part of every cache key

    def invalidate(self) -> None:
        self.epoch += 1
        push_rich_theme()

    async def present(self, entry, width: int) -> Presentation:
        item = entry.item
        try:
            strips = self._present_item(item, width)
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
        if not isinstance(item, UserItem):
            inset = Strip.blank(1, base_style)
            strips = [
                Strip.join([inset, s]).adjust_cell_length(width) for s in strips
            ]
        # Vertical rhythm (the old widget CSS is ignored here — the presenter
        # paints raw strips, so it owns the spacing):
        if isinstance(item, UserItem):
            # Old UserMessage CSS: `padding: 1 0` (surface-coloured blank row
            # inside the box, above and below the text).
            pad = Strip.blank(width, Style(bgcolor=theme.PALETTE["surface"]))
            strips = [pad, *strips, pad]
        leading = self._leading_blanks(entry, item)
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

    def _leading_blanks(self, entry, item) -> int:
        """Blank lines painted above an entry — the old widget CSS margins,
        now owned by the presenter.

        user   → 1, except flush under a status line (e.g. "Worked for …")
                 and at the very top of the transcript (nothing above)
        cot    → 1
        answer → 1, except flush under its own cot
        tools/shimmers → 1 after a user message (a turn can open with a
                 tool call, and it needs the same gap as a reply), else 0
                 (they pack tight under the reply above them)
        status lines → 1 (they belong to the reply above them)"""
        if isinstance(item, UserItem):
            prev = self._previous_item(entry)
            return 0 if prev is None or isinstance(prev, LineItem) else 1
        if isinstance(item, (ToolItem, PendingItem)):
            return isinstance(self._previous_item(entry), UserItem)
        if isinstance(item, LineItem):
            return 1
        if isinstance(item, (ThinkingItem, AssistantItem)):
            prev = self._previous_item(entry)
            flush = isinstance(item, AssistantItem) and isinstance(prev, ThinkingItem)
            return 0 if flush else 1
        return 0

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

    def _present_item(self, item, width: int) -> list[Strip]:
        if isinstance(item, UserItem):
            return self._user(item, width)
        if isinstance(item, AssistantItem):
            return self._assistant(item, width)
        if isinstance(item, ThinkingItem):
            return self._thinking(item, width)
        if isinstance(item, ToolItem):
            return self._tool(item, width)
        if isinstance(item, PendingItem):
            return self._shimmering(item, width)
        if isinstance(item, LineItem):
            return self._line(item, width)
        if isinstance(item, DividerItem):
            return render_strips(Text(item.text, style=theme.PALETTE["accent"]), width)
        # pragma: no cover — every item type is handled above
        return render_strips(Text(repr(item)), width)

    # -- user ---------------------------------------------------------------

    def _user(self, item: UserItem, width: int) -> list[Strip]:
        key = (self.epoch, width, item.text, item.suffix, item.undo_armed,
               len(item.images))

        def build():
            p = theme.PALETTE
            body = Text(item.text, style=p["text"])
            token_style = Style(bgcolor=p["border"], color=p["muted"])
            for match in images.IMAGE_TOKEN_RE.finditer(item.text):
                body.stylize(token_style, match.start(), match.end())
            if item.suffix:
                body.append(" " + item.suffix, style=p["muted"])
            rows: list = [body]
            for image in item.images:
                if token_text(image) in item.text:
                    continue  # the token itself already shows the attachment
                rows.append(Text(token_text(image), style=p["muted"]))
            if item.undo_armed and item.message is not None:
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

        return _cached(item, key, build)

    # -- assistant ------------------------------------------------------------

    def _assistant(self, item: AssistantItem, width: int) -> list[Strip]:
        p = theme.PALETTE
        body_w = width - 2
        body: list[Strip] = []
        if len(item.chunk_cache) > 2_000:
            item.chunk_cache.clear()
        for i, chunk in enumerate(item.chunks):
            key = (i, self.epoch, width)
            cached = item.chunk_cache.get(key)
            if cached is None:
                cached = _settled_strips(chunk, body_w)
                item.chunk_cache[key] = cached
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

    def _thinking(self, item: ThinkingItem, width: int) -> list[Strip]:
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

        return _cached(item, key, build)

    # -- tool call ----------------------------------------------------------

    def _tool(self, item: ToolItem, width: int) -> list[Strip]:
        p = theme.PALETTE
        label = f"Running {item.name}"
        phase = (
            shimmer_now(len(label) + 2 * theme.SHIMMER_BAND_WIDTH)
            if item.status is ToolCallStatus.MAKING_CALL else None
        )
        key = (self.epoch, width, item.expanded, item.status, len(item.args),
               len(item.output), item.header, phase)

        def build():
            if item.status is ToolCallStatus.MAKING_CALL:
                name, detail = _stream_parts(item.name, item.args)
                dot = Text("•", style=p["warning"])
                head = shimmer_text(
                    f"Running {name}", *theme.SHIMMER_THINKING,
                    phase=phase, suffix=detail,
                )
            else:
                dot = Text("•", style=p[item.STATUS_COLOR_KEY[item.status]])
                head = item.summary_text()
            dot_strips = render_strips(Text.assemble(dot, " ", head), width)
            strips = list(dot_strips)
            if item.expanded:
                strips.extend(_prefix_strips(
                    render_strips(item.body_renderable(), width - 2), width,
                    _seg("│ ", p["border"]),
                ))
            return strips

        return _cached(item, key, build)

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
        }.get(item.kind, "muted")
        glyph = _GLYPHS.get(item.kind, "")
        text = Text(f"{glyph}{item.text}", style=p[color_key])
        return render_strips(text, width)


def _settled_strips(text: str, width: int) -> list[Strip]:
    """A settled assistant chunk: markdown prose + palette-highlighted fences
    (ported from the 8957ce64 fast path). No blank lines between blocks —
    rich.markdown's inter-block spacing is stripped after rendering."""
    parts = []
    pos = 0

    def add_prose(chunk: str) -> None:
        prose = chunk.strip("\n")
        if prose.strip():
            # rich.markdown collapses single newlines into spaces; make them
            # hard breaks so "line1\nline2" renders as two lines.
            prose = re.sub(r"(?<!\n)\n(?!\n)", "  \n", prose)
            parts.append((RichMarkdown(prose), True))

    for match in re.finditer(r"```([^\n`]*)\n(.*?)```", text, re.DOTALL):
        add_prose(text[pos: match.start()])
        parts.append((highlight_code(match.group(2), language=match.group(1).strip()), False))
        pos = match.end()
    add_prose(text[pos:])
    if not parts:
        return []
    strips: list[Strip] = []
    for part, is_prose in parts:
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
        strips.extend(rendered)
    return strips


def _cached(item, key, build):
    """Per-item strip cache: return the strips cached under ``key``, or build
    them (and cache) via ``build()``. Items carry a single ``_cache`` slot —
    a fresh key (width/epoch/content change) misses and rebuilds."""
    if item._cache and item._cache[0] == key:
        return item._cache[1]
    value = build()
    item._cache = (key, value)
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
