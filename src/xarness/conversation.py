"""In-memory conversation state, kept separate from the widget tree.

Roles are modeled faithfully to the OpenAI wire format — including the
reserved ``tool`` role and tool-call fields — so a future tool-calling layer
can append tool-call/tool-result messages here without touching UI code.
Reasoning text and token usage are stored locally per assistant message; by
default reasoning is also sent back on later rounds (``keep_reasoning``), but
usage never reaches the wire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .events import Usage

# Prefix of the summary user message compaction leaves behind. Identifying
# it by content (not identity) keeps working across a session save/load.
SUMMARY_PREFIX = "[earlier conversation, summarized]"


@dataclass(slots=True)
class ToolCall:
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
        return cls(call.get("id", ""), fn.get("name", ""), fn.get("arguments", ""))


@dataclass(slots=True)
class Message:
    """One conversation message, mirroring the API's message object."""

    role: str  # "user" | "assistant" | "system" | "tool"
    content: str = ""
    # Local-only: reasoning-channel text for assistant messages.
    reasoning: str | None = None
    # Local-only: seconds spent in the reasoning channel (None if unknown).
    reasoning_seconds: float | None = None
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

    def to_wire(self, include_reasoning: bool = False) -> dict[str, Any]:
        message: dict[str, Any] = {"role": self.role}
        # Per spec, an assistant message that only calls tools should omit
        # content entirely — some providers reject an empty string there.
        if self.content or self.role != "assistant" or not self.tool_calls:
            message["content"] = self.content
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

    def to_wire(self, keep_reasoning: bool = False) -> list[dict[str, Any]]:
        return [message.to_wire(include_reasoning=keep_reasoning) for message in self.messages]
