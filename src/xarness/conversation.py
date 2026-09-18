"""In-memory conversation state, kept separate from the widget tree.

Roles are modeled faithfully to the OpenAI wire format — including the
reserved ``tool`` role and tool-call fields — so a future tool-calling layer
can append tool-call/tool-result messages here without touching UI code.
Reasoning text and token usage are stored locally per assistant message but
are never sent back over the wire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .events import Usage


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
    # Local-only: git checkpoint sha taken when this user message was sent —
    # the state /undo //retry restore the worktree to.
    checkpoint_sha: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    # Local-only: the tool call's one-line header summary (tool-role messages
    # only) — computed by the tool, persisted so /resume renders the same
    # headers. Never sent over the wire.
    header: str | None = None

    def to_wire(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": self.role}
        # Per spec, an assistant message that only calls tools should omit
        # content entirely — some providers reject an empty string there.
        if self.content or self.role != "assistant" or not self.tool_calls:
            message["content"] = self.content
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.tool_calls is not None:
            message["tool_calls"] = self.tool_calls
        return message


@dataclass(slots=True)
class Conversation:
    """Ordered message history; the single source of truth for the chat."""

    messages: list[Message] = field(default_factory=list)
    # Local-only: the message list as it stood when the current turn's user
    # message was sent (set by the controller, used by /undo and /retry to
    # roll the conversation back — including across a mid-turn compaction).
    undo_snapshot: list[Message] | None = None

    def add(self, message: Message) -> Message:
        self.messages.append(message)
        return message

    def to_wire(self) -> list[dict[str, Any]]:
        return [message.to_wire() for message in self.messages]
