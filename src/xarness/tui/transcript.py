"""Pure transcript reducer: items + the live/replay event fold.

No Textual / Rich dependency. Both a fresh snapshot (``load``) and the live
event stream (``apply``) build the same item types here, so a mid-turn attach
and a live turn cannot drift apart. Callers own presentation: they map each
item to a widget entry and re-render the items ``apply`` reports as changed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..protocol import (
    SUMMARY_PREFIX, ContentDelta, Level, Message as ConversationMessage,
    ProcessingStarted, ReasoningDelta, StreamError, ToolCallArgumentsDelta,
    ToolCallArgumentsDone, ToolCallStarted, ToolFinished, ToolOutputDelta,
    TurnComplete, UserMessageAdded, image_refs, is_summary,
)
from ..util import format_duration

_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)

# Live tool output is capped so a runaway command cannot pin megabytes in the
# transcript; the full output is still persisted on the result message.
MAX_LIVE_TOOL_OUTPUT = 65_536


def _parse_iso(stamp: str | None) -> datetime | None:
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp)
    except ValueError:
        return None


def turn_time_summary(messages) -> str | None:
    """The turn's timing line: total | running (streaming + tools) | idle.

    Derived client-side from the persisted wall-clock stamps, so the host
    stores no second, preformatted copy."""
    user_index = next(
        (i for i, m in reversed(list(enumerate(messages)))
         if m.role == "user" and m.timing.send and not is_summary(m)), None,
    )
    turn = messages[user_index + 1:] if user_index is not None else messages
    assistants = [
        m for m in turn if m.role == "assistant" and m.timing.start and m.timing.end
    ]
    if not assistants:
        return None
    end = _parse_iso(assistants[-1].timing.end)
    start = _parse_iso(
        messages[user_index].timing.send if user_index is not None
        else assistants[0].timing.start
    )
    if end is None or start is None:
        return None
    total = max(0.0, (end - start).total_seconds())

    def duration(message, *, reasoning: bool = False) -> float:
        start = _parse_iso(message.timing.start)
        end = _parse_iso(message.timing.end)
        if start is None or end is None:
            return 0.0
        elapsed = max(0.0, (end - start).total_seconds())
        if not reasoning or message.timing.reasoning_seconds is None:
            return elapsed
        return min(message.timing.reasoning_seconds, elapsed)

    streaming = sum(duration(m, reasoning=True) for m in assistants)
    tools = sum(duration(m) for m in turn if m.role == "tool")
    other = max(0.0, total - streaming - tools)
    return (f"{format_duration(total)} | "
            f"{format_duration(streaming + tools)} {format_duration(other)}")


def turn_time_summaries(messages) -> dict[int, str]:
    """Map each turn's final assistant message (by object identity) to its
    derived timing line."""
    starts = [
        i for i, m in enumerate(messages)
        if m.role == "user" and not is_summary(m)
    ]
    summaries: dict[int, str] = {}
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(messages)
        turn = messages[start:end]
        summary = turn_time_summary(turn)
        if not summary:
            continue
        last = next((m for m in reversed(turn) if m.role == "assistant"), None)
        if last is not None:
            summaries[id(last)] = summary
    return summaries


def split_settled(text: str) -> tuple[str, str]:
    """Split at the end of the last complete fenced block: (settled, tail)."""
    last_end = 0
    for match in _FENCE_RE.finditer(text):
        last_end = match.end()
    return text[:last_end], text[last_end:]


class ToolStatus(Enum):
    """Lifecycle of one tool call, driving the status-dot color in the UI."""

    MAKING_CALL = "making_call"  # yellow — arguments streaming / call in flight
    PARSING_ERROR = "parsing_error"  # red — model produced invalid arguments JSON
    CALL_SUCCEEDED = "call_succeeded"  # green
    CALL_FAILED = "call_failed"  # red — tool ran but errored


# ---------------------------------------------------------------------------
# items


class TranscriptItem:
    """Behaviour shared by every transcript row (data stays in dataclasses)."""

    full_width = False

    def streaming(self) -> bool:
        return False

    def leading_blanks(self, prev: "TranscriptItem | None") -> int:
        return 0

    def handle_click(self, app) -> bool:
        """React to a click on this row; return True when the row changed."""
        return False


@dataclass
class UserItem(TranscriptItem):
    """A submitted user message (no local echo: created on UserMessageAdded)."""

    full_width = True

    text: str
    images: list = field(default_factory=list)
    undo_armed: bool = False
    message_id: str | None = None

    @classmethod
    def from_message(cls, message: ConversationMessage) -> "UserItem":
        return cls(
            text=message.content,
            images=image_refs(message),
            message_id=message.id or None,
        )

    def handle_click(self, app) -> bool:
        """Arm the undo affordance; a second click undoes to this message."""
        if not self.message_id:
            return False
        if self.undo_armed:
            self.undo_armed = False
            app.call_later(app._run_undo_to, self.message_id)
        else:
            self.undo_armed = True
        return True

    def leading_blanks(self, prev) -> int:
        return 0 if prev is None or isinstance(prev, LineItem) else 1


@dataclass
class AssistantItem(TranscriptItem):
    """One model answer: settled chunks (closed fences settled once) + live tail."""

    chunks: list[str] = field(default_factory=list)
    live: str = ""
    finalized: bool = False

    def append_delta(self, text: str) -> None:
        """Fold a delta into the live tail, settling any closed fence."""
        self.live += text
        settled, remainder = split_settled(self.live)
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

    def leading_blanks(self, prev) -> int:
        return 0 if isinstance(prev, ThinkingItem) else 1


@dataclass
class ThinkingItem(TranscriptItem):
    """Reasoning channel. Click toggles the text; the pill shows the time."""

    parts: list[str] = field(default_factory=list)
    done: bool = False
    duration: float | None = None
    expanded: bool = True

    @property
    def reasoning(self) -> str:
        return "".join(self.parts)

    def finish(self, duration: float | None) -> None:
        """Collapse to a summary. The first duration (stamped when the block
        ends) wins; a later one only fills an unknown ("…") duration."""
        if self.done:
            if duration is not None and self.duration is None:
                self.duration = duration
            return
        self.done = True
        self.duration = duration
        self.expanded = False  # auto-collapse once the round moves on

    def summary_text(self) -> str:
        if not self.done:
            return "Thinking"
        if self.duration is not None:
            from ..util import format_duration
            return f"Thought for {format_duration(self.duration)}"
        return "Thought for …"

    def streaming(self) -> bool:
        return not self.done

    def handle_click(self, app) -> bool:
        """Toggleable at any time — mid-stream too (the user can collapse
        the live CoT and reopen it; finish() re-collapses at round end)."""
        self.expanded = not self.expanded
        return True

    def leading_blanks(self, prev) -> int:
        return 1


@dataclass
class ToolItem(TranscriptItem):
    """One tool call: status dot + summary; click expands args/output."""

    call_id: str
    name: str
    args: str = ""
    output: str = ""
    status: ToolStatus = ToolStatus.MAKING_CALL
    expanded: bool = False

    STATUS_COLOR_KEY = {
        ToolStatus.MAKING_CALL: "warning",
        ToolStatus.PARSING_ERROR: "error",
        ToolStatus.CALL_SUCCEEDED: "success",
        ToolStatus.CALL_FAILED: "error",
    }
    STATUS_VERB = {
        ToolStatus.MAKING_CALL: "Running",
        ToolStatus.PARSING_ERROR: "Failed",
        ToolStatus.CALL_SUCCEEDED: "",
        ToolStatus.CALL_FAILED: "Failed",
    }

    def streaming(self) -> bool:
        return self.status is ToolStatus.MAKING_CALL

    def handle_click(self, app) -> bool:
        self.expanded = not self.expanded
        return True

    def leading_blanks(self, prev) -> int:
        return int(isinstance(prev, UserItem))


@dataclass
class PendingItem(TranscriptItem):
    """The ephemeral "Processing · esc to interrupt" strip."""

    label: str = "Processing"
    hint: str = "esc to interrupt"

    def streaming(self) -> bool:
        return True

    def leading_blanks(self, prev) -> int:
        return int(isinstance(prev, UserItem))


@dataclass
class LineItem(TranscriptItem):
    """A one-line status message (info / warn / error / success / hint)."""

    text: str
    kind: Level = "info"

    def leading_blanks(self, prev) -> int:
        return 1


@dataclass
class DividerItem(TranscriptItem):
    """A divider line (the compaction handoff marker)."""

    text: str


# ---------------------------------------------------------------------------
# reducer


class Transcript:
    """Fold persisted messages and live events into a flat item list.

    ``live`` indexes only the current round's thinking/answer/tool items;
    messages from ``load`` are settled history and never touch them.
    """

    def __init__(self) -> None:
        self.items: list = []
        self.thinking: ThinkingItem | None = None
        self.assistant: AssistantItem | None = None
        self.tool_blocks: dict[str, ToolItem] = {}
        self.cancelled = False
        # Emission time of the first reasoning delta in this round. The cot
        # block ends at the first answer/tool token, so that event's stamp
        # minus this one is the block's own duration (never the round's).
        self._reasoning_t: float | None = None

    # -- replay -----------------------------------------------------------

    def load(self, snap) -> list:
        self.items = []
        self.start_round()
        # ``history`` holds epochs hidden from the model by compaction. The
        # human still sees them: fold them first, exactly like the current
        # epoch, so a resumed session shows everything before the handoff.
        history = list(getattr(snap, "history", None) or [])
        messages = [*history, *snap.messages]
        results = {
            m.tool_call_id: m for m in messages
            if m.role == "tool" and m.tool_call_id
        }
        summaries = turn_time_summaries(messages)
        for message in messages:
            self._add_message(message, results, summaries.get(id(message)))
        self._load_partial(getattr(snap, "partial_round", None))
        return self.items

    def _add_message(
        self, message, results: dict, summary: str | None = None,
    ) -> None:
        if message.role == "user":
            if is_summary(message):
                self.items.append(DividerItem(
                    message.content.removeprefix(SUMMARY_PREFIX).strip(),
                ))
            else:
                self.items.append(UserItem.from_message(message))
        elif message.role == "assistant":
            if message.reasoning:
                self.items.append(ThinkingItem(
                    parts=[message.reasoning], done=True,
                    duration=message.timing.reasoning_seconds, expanded=False,
                ))
            if message.content or (not message.tool_calls and not message.reasoning):
                item = AssistantItem(finalized=True)
                item.chunks = [message.content] if message.content else ["(no output)"]
                self.items.append(item)
            for call in message.tool_calls or []:
                item = ToolItem(
                    call_id=call.call_id, name=call.name,
                    args=call.arguments_json,
                )
                result = results.get(call.call_id)
                if result is not None:
                    item.status = (
                        ToolStatus.CALL_SUCCEEDED if result.ok
                        else ToolStatus.CALL_FAILED
                    )
                    item.output = result.content
                else:
                    item.status = ToolStatus.CALL_FAILED
                    item.output = "interrupted"
                self.tool_blocks[call.call_id] = item
                self.items.append(item)
            if summary:
                self.items.append(LineItem(text=f"Worked for {summary}"))

    def _load_partial(self, partial) -> None:
        """Render the snapshot's live round on top of the persisted history.

        Partial tool calls whose assistant message is already in the history
        are matched by ``call_id`` and updated in place, never duplicated."""
        if partial is None:
            return
        if partial.reasoning:
            self.thinking = ThinkingItem(parts=[partial.reasoning])
            self.items.append(self.thinking)
        if partial.content:
            self.assistant = AssistantItem()
            self.assistant.append_delta(partial.content)
            self.items.append(self.assistant)
        for call in partial.tool_calls:
            item = self.tool_blocks.get(call.call_id)
            if item is None:
                item = ToolItem(call_id=call.call_id, name=call.name)
                self.tool_blocks[call.call_id] = item
                self.items.append(item)
            item.name = call.name or item.name
            item.args = call.arguments
            item.output = call.output
            if call.ok is None:
                item.status = ToolStatus.MAKING_CALL
            elif call.parse_error:
                item.status = ToolStatus.PARSING_ERROR
            elif call.ok:
                item.status = ToolStatus.CALL_SUCCEEDED
            else:
                item.status = ToolStatus.CALL_FAILED
        if self.thinking is not None and (partial.content or partial.tool_calls):
            # Reasoning precedes the answer/tool phase; settle it without a
            # duration. The stamp that would have ended the block is not part
            # of the snapshot, and inventing one here would be a client-side
            # estimate of a number the host owns.
            self.thinking.finish(None)

    # -- live -------------------------------------------------------------

    def start_round(self) -> None:
        """Begin a fresh round (also used to reset after cancel/attach)."""
        self.thinking = None
        self.assistant = None
        self.tool_blocks = {}
        self.cancelled = False
        self._reasoning_t = None

    def _add(self, item):
        self.items.append(item)
        return item

    def _settle_thinking(self, at: float | None) -> ThinkingItem | None:
        """Close the live cot block at event stamp ``at`` and return it.

        ``at`` is the emission time of the first event *after* the reasoning
        channel (first answer token or first tool call), which is exactly where
        the host ends its ``reasoning_seconds`` measurement — so the live pill
        matches what a replayed/resumed transcript shows. ``None`` settles the
        block with an unknown duration instead of inventing a number.
        """
        item = self.thinking
        if item is None:
            return None
        duration = (
            at - self._reasoning_t
            if at is not None and self._reasoning_t is not None
            else None
        )
        item.finish(duration)
        return item

    def apply(self, event) -> list:
        """Fold one live event; return the items created or mutated."""
        if isinstance(event, UserMessageAdded):
            return [self._add(UserItem(
                text=event.text,
                images=list(event.images or []),
                message_id=event.message_id,
            ))]
        if isinstance(event, ProcessingStarted):
            self.start_round()
            return []
        if self.cancelled:
            return []
        if isinstance(event, ReasoningDelta):
            if self.thinking is None:
                self.thinking = self._add(ThinkingItem())
                self._reasoning_t = event.t
            self.thinking.parts.append(event.text)
            return [self.thinking]
        if isinstance(event, ContentDelta):
            changed = []
            if self.assistant is None:
                # The first answer token ends the cot block: stamp its own
                # duration now (and repaint the pill, or it would keep the
                # shimmer text until some later event happened to update it).
                settled = self._settle_thinking(event.t)
                if settled is not None:
                    changed.append(settled)
                self.assistant = self._add(AssistantItem())
            self.assistant.append_delta(event.text)
            changed.append(self.assistant)
            return changed
        if isinstance(event, ToolCallStarted):
            # The first tool call ends the cot block too — tool-argument
            # streaming is call time, not thinking time.
            settled = self._settle_thinking(event.t)
            item = self._add(ToolItem(call_id=event.call_id, name=event.name))
            self.tool_blocks[event.call_id] = item
            return [item, settled] if settled is not None else [item]
        if isinstance(event, ToolCallArgumentsDelta):
            item = self.tool_blocks.get(event.call_id)
            if item is None:
                return []
            item.args += event.text
            return [item]
        if isinstance(event, ToolCallArgumentsDone):
            item = self.tool_blocks.get(event.call_id)
            if item is None:
                return []
            # Some providers only send the function name on a later fragment
            # than the first; the authoritative name rides on Done.
            if event.name:
                item.name = event.name
            item.args = event.arguments_json
            return [item]
        if isinstance(event, ToolOutputDelta):
            item = self.tool_blocks.get(event.call_id)
            if item is None or len(item.output) >= MAX_LIVE_TOOL_OUTPUT:
                return []
            room = MAX_LIVE_TOOL_OUTPUT - len(item.output)
            item.output += event.text[:room]
            return [item]
        if isinstance(event, ToolFinished):
            item = self.tool_blocks.get(event.call_id)
            if item is None:
                return []
            if event.parse_error:
                item.status = ToolStatus.PARSING_ERROR
            elif event.ok:
                item.status = ToolStatus.CALL_SUCCEEDED
            else:
                item.status = ToolStatus.CALL_FAILED
            item.output = event.output or event.error
            return [item]
        if isinstance(event, TurnComplete):
            return self.settle(event.t)
        if isinstance(event, StreamError):
            return self.settle(None)
        return []

    def settle(self, at: float | None) -> list:
        """Settle the live round, ending any still-open cot block at ``at``.

        Normally the block was already closed by the first answer/tool event;
        this only stamps the duration for a round that never produced one
        (e.g. reasoning-only), and leaves it unknown for errors/cancels.
        """
        changed = []
        settled = self._settle_thinking(at)
        if settled is not None:
            changed.append(settled)
        if self.assistant is not None:
            self.assistant.finalize()
            changed.append(self.assistant)
        return changed

    def cancel(self) -> list:
        """Settle the round for an interrupted turn; ignore later events."""
        self.cancelled = True
        return self.settle(None)
