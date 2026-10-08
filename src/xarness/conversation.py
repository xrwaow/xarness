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
import re
from typing import Any

from .events import Usage
from .images import ImageAttachment

# Prefix of the summary user message compaction leaves behind. Identifying
# it by content (not identity) keeps working across a session save/load.
SUMMARY_PREFIX = "[earlier conversation, summarized]"

# The placeholder tokens attached images leave in user text (see
# images.token_text); matches are replaced by image content parts on the wire.
_TOKEN_RE = re.compile(r"\[🖼 [^\]\n]+\]")


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
    # Images attached to a user message (persisted; sent over the wire as
    # base64 ``image_url`` content parts on every round). None = text only.
    images: list[ImageAttachment] | None = None

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
            index = 0
            for match in _TOKEN_RE.finditer(self.content):
                chunk = self.content[last:match.start()]
                if chunk:
                    parts.append({"type": "text", "text": chunk})
                if index < len(self.images):
                    image = self.images[index]
                    index += 1
                    parts.append(
                        {"type": "image_url", "image_url": {"url": image.data_url()}}
                    )
                else:
                    # More tokens than attachments: keep the marker text.
                    parts.append({"type": "text", "text": match.group(0)})
                last = match.end()
            chunk = self.content[last:]
            if chunk:
                parts.append({"type": "text", "text": chunk})
            parts.extend(
                {"type": "image_url", "image_url": {"url": image.data_url()}}
                for image in self.images[index:]
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

    def to_wire(self, keep_reasoning: bool = False) -> list[dict[str, Any]]:
        answered = {
            m.tool_call_id for m in self.messages if m.role == "tool" and m.tool_call_id
        }
        wire: list[dict[str, Any]] = []
        for message in self.messages:
            orphaned = (
                message.role == "assistant"
                and message.tool_calls
                and any(c.call_id not in answered for c in message.tool_calls)
            )
            if not orphaned:
                wire.append(message.to_wire(include_reasoning=keep_reasoning))
                continue
            # A call without a result (turn interrupted mid-execution, session
            # saved) would be rejected as an unpaired tool call: send only the
            # answered ones, or drop the message when none remain and it
            # carries no text.
            paired = [c for c in message.tool_calls if c.call_id in answered]
            if paired or message.content:
                wire_message = message.to_wire(include_reasoning=keep_reasoning)
                if paired:
                    wire_message["tool_calls"] = [c.to_wire() for c in paired]
                else:
                    del wire_message["tool_calls"]
                wire.append(wire_message)
        return wire
