"""In-memory conversation state, kept separate from the widget tree.

Roles are modeled faithfully to the OpenAI wire format — including the
reserved ``tool`` role and tool-call fields — so a future tool-calling layer
can append tool-call/tool-result messages here without touching UI code.
Reasoning text and token usage are stored locally per assistant message; by
default reasoning is also sent back on later rounds (``keep_reasoning``), but
usage never reaches the wire.

``ToolCall`` and ``Message`` are pydantic models so the very same objects
are persisted (session JSON) and shown to clients (snapshots) — there is no
second mirrored model to keep in sync.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .events import Usage
from .util import format_duration
from .images import IMAGE_TOKEN_RE, ImageAttachment, token_text


# Prefix of the summary user message compaction leaves behind. Identifying
# it by content (not identity) keeps working across a session save/load.
SUMMARY_PREFIX = "[earlier conversation, summarized]"

def is_summary(message: "Message") -> bool:
    """True for the summary user message a compaction leaves behind."""
    return message.role == "user" and message.content.startswith(SUMMARY_PREFIX)


def turn_time_summary(messages: list["Message"]) -> str | None:
    """The turn's timing line: total | running | idle.

    Computed from the persisted wall-clock stamps, so the same string can be
    stamped on the turn's last assistant message (reloads show it) and sent
    in TurnFinished (the live clients show it). None when the stamps are
    missing (old sessions, or nothing streamed).

    - total: turn user message send_time → last assistant end_time
    - running: streaming (per round, end_time − reasoning_start; start_time
      when the round had no chain of thought) + tool execution durations
    - idle: what's left (per-round latency and round boundaries)
    """
    turn_user_index = next(
        (i for i, m in reversed(list(enumerate(messages)))
         if m.role == "user" and m.send_time and not is_summary(m)),
        None,
    )
    turn = messages[turn_user_index + 1:] if turn_user_index is not None else messages
    assistants = [
        m for m in turn if m.role == "assistant" and m.start_time and m.end_time
    ]
    if not assistants:
        return None
    total_end = _parse_iso(assistants[-1].end_time)
    total_start = (
        _parse_iso(messages[turn_user_index].send_time)
        if turn_user_index is not None
        else _parse_iso(assistants[0].start_time)
    )
    if total_end is None or total_start is None:
        return None
    total = max(0.0, (total_end - total_start).total_seconds())
    streaming = 0.0
    for message in assistants:
        end = _parse_iso(message.end_time)
        first = _parse_iso(message.reasoning_start) or _parse_iso(message.start_time)
        if end is not None and first is not None:
            streaming += max(0.0, (end - first).total_seconds())
    tools = 0.0
    for message in turn:
        if message.role != "tool":
            continue
        start = _parse_iso(message.start_time)
        end = _parse_iso(message.end_time)
        if start is not None and end is not None:
            tools += max(0.0, (end - start).total_seconds())
    other = max(0.0, total - streaming - tools)
    fmt = format_duration
    return f"{fmt(total)} | {fmt(streaming + tools)} {fmt(other)}"


def normalize_time_summary(summary: str) -> str:
    """Re-render a persisted timing line in the current shape:
    'total | running idle' with compact durations ('1m02s').

    Older versions persisted 'total | streaming tools other' (three
    numbers) and spaced durations ('1m 02s'); both are re-rendered — the
    three numbers merge into running = streaming + tools. Anything that
    doesn't parse as a timing line passes through unchanged, and a line
    already in the current shape is returned as-is.
    """
    unit_seconds = {"s": 1, "m": 60, "h": 3600}

    parts = [part.strip() for part in summary.split("|")]
    if len(parts) != 2:
        return summary
    # Split the tail into durations via its number+unit pairs ('1m02s' and
    # the legacy '1m 02s' both give (1,m),(02,s)). Units descend h→m→s
    # within one duration, so a pair continues the previous duration when
    # it steps down the chain (h→m, m→s) and starts a new one otherwise.
    # Whitespace is irrelevant.
    durations: list[float] = []
    previous_unit = ""
    current_has_hours = False
    for number, unit in re.findall(r"(\d+(?:\.\d+)?)([hms])", parts[1]):
        value = float(number) * unit_seconds[unit]
        # 'h' durations end at 'm' (format_duration never emits h+m+s), so
        # an 'm' pair that completes an 'h' does not absorb a following 's'.
        continues = (
            (previous_unit, unit) == ("h", "m")
            or ((previous_unit, unit) == ("m", "s") and not current_has_hours)
        )
        if continues and durations:
            durations[-1] += value
        else:
            durations.append(value)
            current_has_hours = unit == "h"
        previous_unit = unit
    if len(durations) == 3:
        # Legacy three-number line: streaming + tools merge into running.
        running, idle = durations[0] + durations[1], durations[2]
    elif len(durations) == 2:
        running, idle = durations
    else:
        return summary
    total = 0.0
    for number, unit in re.findall(r"(\d+(?:\.\d+)?)([hms])", parts[0]):
        total += float(number) * unit_seconds[unit]
    if total <= 0:
        return summary
    rendered = (
        f"{format_duration(total)} | {format_duration(running)} {format_duration(idle)}"
    )
    return rendered if rendered != summary else summary


def _parse_iso(stamp: str | None) -> datetime | None:
    """Parse one of the ISO stamps carried on messages (None on failure)."""
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp)
    except ValueError:
        return None


# The placeholder tokens attached images leave in user text (see
# images.token_text); matches are replaced by image content parts on the wire.
_TOKEN_RE = IMAGE_TOKEN_RE


class ToolCall(BaseModel):
    """One tool call the model requested — the single shape used everywhere
    (stream events, history, wire format, TUI)."""

    call_id: str
    name: str
    arguments_json: str

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.call_id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments_json},
        }

    @classmethod
    def from_wire(cls, call: dict[str, Any]) -> ToolCall:
        fn = call.get("function") or {}
        return cls(
            call_id=call.get("id", ""), name=fn.get("name", ""),
            arguments_json=fn.get("arguments", ""),
        )


class Message(BaseModel):
    """One conversation message, mirroring the API's message object."""

    role: str  # "user" | "assistant" | "system" | "tool"
    content: str = ""
    # Stable id (uuid4), persisted. Click-to-undo addresses messages by
    # id, not object identity.
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    # Wall-clock UTC ISO 8601 stamps (None for old sessions). User messages
    # carry send_time (steering: when the client sent it, not when it was
    # injected); assistant messages start_time/end_time around the request,
    # reasoning_start/reasoning_end for the chain of thought; tool result
    # messages start_time/end_time for the call's execution. Never sent to
    # the provider (to_wire does not include them).
    send_time: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    reasoning_start: str | None = None
    reasoning_end: str | None = None
    # Local-only: reasoning-channel text for assistant messages.
    reasoning: str | None = None
    # Local-only: token usage reported for the round that produced this
    # message (assistant messages only; never sent over the wire).
    usage: Usage | None = None
    # Local-only: git checkpoint tree sha taken when this user message was
    # sent — the workspace state *before* the turn's edits. /undo //retry
    # reverse the turn's diff (checkpoint_sha → after_tree).
    checkpoint_sha: str | None = None
    # Local-only: workspace tree sha after the turn's edits finished — the
    # state the turn produced. Recorded once the turn completes; together
    # with checkpoint_sha it lets /undo reverse exactly this turn's changes
    # without disturbing edits made since.
    after_tree: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    # Local-only: the tool call's one-line header summary (tool-role messages
    # only) — computed by the tool, persisted so /resume renders the same
    # headers. Never sent over the wire.
    header: str | None = None
    # Local-only: "Worked for X | Y Z W" timing of the turn that produced
    # this assistant message (last one of the turn only) — total, streaming,
    # idle-waiting-for-tools, and the leftover latency. Computed by the
    # session at turn end from the persisted stamps, so every client and
    # every reload shows the same line.
    time_summary: str | None = None
    # Images attached to a user message (persisted; sent over the wire as
    # base64 ``image_url`` content parts on every round). None = text only.
    images: list[ImageAttachment] | None = None

    @field_validator("tool_calls", mode="before")
    @classmethod
    def _coerce_tool_calls(cls, value: object) -> object:
        """Accept tool calls in either shape: the wire shape persisted in
        session files and sent to providers (``{id, type, function}``), and
        the internal shape a ``model_dump()`` round-trip produces
        (``{call_id, name, arguments_json}`` — snapshots serialize Message
        as-is, so this is what crosses the host→client JSON boundary)."""
        if isinstance(value, list):
            out = []
            for c in value:
                if isinstance(c, ToolCall):
                    out.append(c)
                elif isinstance(c, dict) and (
                    "call_id" in c or "arguments_json" in c
                ):
                    out.append(ToolCall.model_validate(c))
                else:
                    out.append(ToolCall.from_wire(c))
            return out
        return value

    @property
    def reasoning_duration(self) -> float | None:
        """Seconds spent in the reasoning channel, derived from the
        reasoning_start/reasoning_end stamps (None when unknown)."""
        if not self.reasoning_start or not self.reasoning_end:
            return None
        start = _parse_iso(self.reasoning_start)
        end = _parse_iso(self.reasoning_end)
        if start is None or end is None:
            return None
        return max(0.0, (end - start).total_seconds())

    def to_wire(self, include_reasoning: bool = False) -> dict[str, Any]:
        message: dict[str, Any] = {"role": self.role}
        # Per spec, an assistant message that only calls tools should omit
        # content entirely — some providers reject an empty string there.
        if self.content or self.role != "assistant" or not self.tool_calls:
            message["content"] = self.content
        if self.role == "user" and self.images:
            # Multipart content. The text may carry image placeholder tokens
            # ("[🖼 name W×H]") marking where each image belongs; content
            # parts are emitted in that order so "compare [img1] with
            # [img2]" stays meaningful to the model. Tokens without a
            # matching attachment (or images without a token) still come
            # through: text parts first, unmatched images appended at the end.
            parts: list[dict[str, Any]] = []
            last = 0
            # Match tokens by placeholder text, not ordinal: an unknown or
            # duplicate token must not shift later images.
            by_token = {token_text(image): image for image in self.images}
            used: set[int] = set()
            for match in _TOKEN_RE.finditer(self.content):
                chunk = self.content[last:match.start()]
                if chunk:
                    parts.append({"type": "text", "text": chunk})
                image = by_token.get(match.group(0))
                if image is not None and id(image) not in used:
                    used.add(id(image))
                    parts.append(
                        {"type": "image_url", "image_url": {"url": image.data_url()}}
                    )
                else:
                    parts.append({"type": "text", "text": match.group(0)})
                last = match.end()
            chunk = self.content[last:]
            if chunk:
                parts.append({"type": "text", "text": chunk})
            parts.extend(
                {"type": "image_url", "image_url": {"url": image.data_url()}}
                for image in self.images if id(image) not in used
            )
            message["content"] = parts
        if include_reasoning and self.role == "assistant" and self.reasoning is not None:
            # DeepSeek-style field name; providers that don't know it ignore it.
            message["reasoning_content"] = self.reasoning
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.tool_calls is not None:
            message["tool_calls"] = [call.to_wire() for call in self.tool_calls]
        return message


@dataclass(slots=True)
class Conversation:
    """Ordered message history; the single source of truth for the chat."""

    messages: list[Message] = field(default_factory=list)
    # Local-only: index of the current turn's user message (set by the
    # controller when the turn begins; /undo //retry roll back to it —
    # including across a mid-turn compaction). The turn snapshot is always
    # the messages[:turn_start + 1] prefix of the list, so an index stands
    # in for a full copy and survives save/load unchanged.
    turn_start: int | None = None
    # Local-only: the full message list as it stood just before the most
    # recent compaction, so /undo can restore pre-compaction history. Only
    # the most recent compaction is recoverable.
    compact_snapshot: list[Message] | None = None

    def add(self, message: Message) -> Message:
        self.messages.append(message)
        return message

    def last_turn_message(self) -> Message | None:
        """The most recent user message that opened a tracked turn (carries
        a checkpoint sha), or None."""
        return next(
            (m for m in reversed(self.messages)
             if m.role == "user" and m.checkpoint_sha is not None),
            None,
        )

    def repair_orphaned_tool_calls(self, now: str | None = None) -> bool:
        """Give any assistant tool_call without a matching tool message a
        synthetic ``error: interrupted`` result, so the next API request is
        valid. Runs after a cancel and on a session load. Returns True when
        anything was added."""
        answered = {
            m.tool_call_id for m in self.messages
            if m.role == "tool" and m.tool_call_id
        }
        changed = False
        for message in self.messages:
            if message.role != "assistant" or not message.tool_calls:
                continue
            for call in message.tool_calls:
                if call.call_id not in answered:
                    self.add(Message(
                        role="tool", tool_call_id=call.call_id,
                        content="error: interrupted",
                        end_time=now,
                    ))
                    answered.add(call.call_id)
                    changed = True
        return changed

    def to_wire(self, keep_reasoning: bool = False) -> list[dict[str, Any]]:
        # Orphaned tool calls (turn interrupted mid-execution, session saved
        # mid-turn) are answered synthetically by Session._repair_tool_calls,
        # which runs on cancel and on session load, so every message can go
        # out as-is.
        return [
            message.to_wire(include_reasoning=keep_reasoning)
            for message in self.messages
        ]
