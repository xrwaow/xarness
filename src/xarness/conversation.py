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

import base64
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from .gitwork import TurnCheckpoint
from .images import (
    IMAGE_TOKEN_RE,
    ImageAttachment,
    ImageRef,
    legacy_token_text,
    token_text,
)

# Prefix of the summary user message compaction leaves behind. Identifying
# it by content (not identity) keeps working across a session save/load.
SUMMARY_PREFIX = "[earlier conversation, summarized]"

def is_summary(message: "Message") -> bool:
    """True for the summary user message a compaction leaves behind."""
    return message.kind == "summary"


class TextPart(BaseModel):
    """One text segment of a message."""

    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    """One image segment, ordered where it appeared in the text.

    ``ref`` is the persisted/wire shape (content hash + display metadata).
    ``data`` is a transient fallback for callers without an image store; it
    is excluded from every serialization, so it never reaches a session
    file or a snapshot."""

    type: Literal["image"] = "image"
    ref: ImageRef
    data: bytes | None = Field(default=None, exclude=True)


MessagePart = TextPart | ImagePart


def build_parts(
    text: str, images: list[ImageAttachment] | None,
    store: Any = None,
) -> list[MessagePart]:
    """Build ordered text/image parts from a user message.

    The text carries one ``[🖼 name W×H]`` token per attachment; parts are
    emitted in token order so ``compare [img1] with [img2]`` keeps its
    meaning. A token without a matching attachment stays text, and an
    attachment without a token is appended at the end. ``store`` (an object
    with ``save(ImageAttachment) -> ImageRef``) writes the bytes to the blob
    store; without one the parts carry transient ``data`` instead.
    """
    if not images:
        return [TextPart(text=text)] if text else []
    by_token = {token_text(image): image for image in images}
    legacy = {legacy_token_text(image): image for image in images}
    used: set[int] = set()
    parts: list[MessagePart] = []

    def add_image(image: ImageAttachment) -> None:
        ref = store.save(image) if store is not None else image.to_ref()
        parts.append(ImagePart(
            ref=ref, data=None if store is not None else image.data(),
        ))

    last = 0
    for match in IMAGE_TOKEN_RE.finditer(text):
        chunk = text[last:match.start()]
        if chunk:
            parts.append(TextPart(text=chunk))
        image = by_token.get(match.group(0)) or legacy.get(match.group(0))
        if image is not None and id(image) not in used:
            used.add(id(image))
            add_image(image)
        else:
            parts.append(TextPart(text=match.group(0)))
        last = match.end()
    chunk = text[last:]
    if chunk:
        parts.append(TextPart(text=chunk))
    for image in images:
        if id(image) not in used:
            add_image(image)
    return parts


def image_refs(message: "Message") -> list[ImageRef]:
    """The image references carried by a message, in text order."""
    return [
        part.ref for part in message.parts or () if isinstance(part, ImagePart)
    ]


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


class Usage(BaseModel):
    """Token usage for a turn. ``approximate`` marks locally estimated counts.

    A pydantic model (not a dataclass) so the very same object crosses the
    wire in snapshots and persists losslessly."""

    input_tokens: int = 0
    output_tokens: int = 0
    approximate: bool = False


class Timing(BaseModel):
    """When a message happened and (for a turn's last answer) how long it took.

    One sub-model instead of six fields scattered on ``Message``. Stamps are
    wall-clock UTC ISO 8601; ``summary`` is the derived turn timing line
    ("total | running idle") stamped only on the turn's last assistant
    message so every client and every reload shows the same one."""

    send: str | None = None
    start: str | None = None
    end: str | None = None
    reasoning_seconds: float | None = None


class Message(BaseModel):
    """One conversation message, mirroring the API's message object."""

    role: Literal["user", "assistant", "system", "tool"]
    content: str = ""
    # "summary" marks the handoff message compaction appends; "interrupted"
    # marks a partial assistant message saved when a turn was cancelled.
    kind: Literal["normal", "summary", "interrupted"] = "normal"
    # Compaction epoch: the conversation shows only messages whose epoch
    # equals the current one (plus later appends). Undo-compaction flips the
    # epoch back instead of restoring a second copy of the history.
    epoch: int = 0
    # Stable id (uuid4), persisted. Click-to-undo addresses messages by
    # id, not object identity.
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    # Timestamps and the derived timing line. Never sent to the provider.
    timing: Timing = Field(default_factory=Timing)
    # Local-only: reasoning-channel text for assistant messages.
    reasoning: str | None = None
    # Local-only: token usage reported for the round that produced this
    # message (assistant messages only; never sent over the wire).
    usage: Usage | None = None
    # Tool result outcome, persisted explicitly instead of encoded in the
    # content prefix, so the TUI can rebuild the call's status faithfully.
    ok: bool = True
    parse_error: bool = False
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    # Ordered text/image segments of a user message. Images are stored by
    # content hash (see ImageRef); the provider wire format is assembled in
    # to_wire, so persistence and snapshots carry no base64 payloads.
    parts: list[TextPart | ImagePart] | None = None

    def to_wire(
        self, include_reasoning: bool = False,
        image_loader: Callable[[str], bytes | None] | None = None,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {"role": self.role}
        # Per spec, an assistant message that only calls tools should omit
        # content entirely — some providers reject an empty string there.
        if self.content or self.role != "assistant" or not self.tool_calls:
            message["content"] = self.content
        image_parts = [p for p in self.parts or () if isinstance(p, ImagePart)]
        if self.role == "user" and image_parts:
            # Multipart content, in the order the text/image parts were
            # recorded. The bytes are resolved from the session blob store
            # (or carried transiently when there is no store).
            content: list[dict[str, Any]] = []
            for part in self.parts or ():
                if isinstance(part, TextPart):
                    if part.text:
                        content.append({"type": "text", "text": part.text})
                    continue
                data = part.data
                if data is None and image_loader is not None:
                    data = image_loader(part.ref.id)
                if data is None:
                    # The blob is gone: keep the message valid rather than
                    # silently dropping the attachment entirely.
                    content.append({"type": "text", "text": token_text(part.ref)})
                    continue
                encoded = base64.b64encode(data).decode("ascii")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{part.ref.mime};base64,{encoded}"},
                })
            message["content"] = content
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


class Turn(BaseModel):
    """One tracked user turn: the user message that opened it and the git
    trees on either side of its edits. ``after_tree is None`` means open."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    user_message_id: str
    before_tree: str | None = None
    after_tree: str | None = None
    # /accept locks a turn: its files can no longer be reverted by /undo.
    locked: bool = False


@dataclass(slots=True)
class RollbackPlan:
    """What /undo or /retry needs to do, computed by
    :meth:`Conversation.rollback_plan`.

    Usage totals are monotonic (they count everything ever spent, including
    undone turns), so a plan carries no usage bookkeeping."""

    user_text: str  # the turn's user message (input box / resend)
    revert_turns: list[TurnCheckpoint]  # file diffs to reverse (oldest first)
    keep: list[Message]  # what the conversation becomes
    images: list[ImageRef] | None = None
    # Epoch to restore (undoing a compaction flips it back instead of
    # restoring a second copy of the history).
    epoch: int = 0
    # True for a compaction-only undo: no messages are dropped, the turn
    # stays intact, no files are reverted, and the input box is not touched.
    compaction_only: bool = False


@dataclass(slots=True)
class Conversation:
    """Ordered message history; the single source of truth for the chat.

    ``messages`` is append-only across compaction: old epochs stay in the
    list and are simply hidden by ``visible_messages`` until an undo flips
    ``epoch`` back. ``turns`` owns the per-turn git checkpoints, so a user
    message carries no turn state of its own."""

    messages: list[Message] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    epoch: int = 0

    def add(self, message: Message) -> Message:
        message.epoch = self.epoch
        self.messages.append(message)
        return message

    def visible_messages(self) -> list[Message]:
        return [m for m in self.messages if m.epoch == self.epoch]

    @property
    def open_turn(self) -> Turn | None:
        """The still-open turn (its after_tree not yet recorded), or None."""
        return next((t for t in reversed(self.turns) if t.after_tree is None), None)

    def begin_turn(self, user_message: Message, before_tree: str | None) -> Turn:
        turn = Turn(user_message_id=user_message.id, before_tree=before_tree)
        self.turns.append(turn)
        return turn

    # -- rollback planning ------------------------------------------------

    def message_by_id(self, message_id: str) -> Message | None:
        return next((m for m in self.messages if m.id == message_id), None)

    def rollback_plan(self) -> RollbackPlan | None:
        """Plan the rollback of the last user turn, or None if there is
        nothing to undo (no user message, or no response to it yet).

        If the most recent thing that happened is a compaction (a summary in
        the visible history, no real user turn after it), /undo //retry undo
        the compaction itself first by flipping the epoch back; the turn that
        triggered it stays intact and no files are touched."""
        visible = self.visible_messages()
        if self.epoch > 0 and visible and is_summary(visible[-1]):
            return RollbackPlan(
                user_text="", revert_turns=[], keep=list(self.messages),
                epoch=self.epoch - 1, compaction_only=True,
            )
        turn = self.turns[-1] if self.turns else None
        message = self.message_by_id(turn.user_message_id) if turn else None
        if message is None:
            # Legacy session without a tracked turn: fall back to the last
            # real user message that has a response after it.
            message = next(
                (m for m in reversed(visible)
                 if m.role == "user" and not is_summary(m) and m is not visible[-1]),
                None,
            )
        if message is None or (visible and message is visible[-1]):
            return None  # the turn produced no response yet
        return self.rollback_plan_to(message)

    def rollback_plan_to(self, message: Message) -> RollbackPlan | None:
        """Plan a rollback to an arbitrary earlier user message
        (click-to-undo): drop that message and everything after it, restoring
        the workspace to the checkpoint taken when it was sent.

        Searches the full append-only history, so messages the last compaction
        summarized away are found too (the epoch simply drops back to theirs).
        Returns None when the message is not a user turn in this conversation.
        """
        if message.role != "user" or is_summary(message):
            return None
        index = next(
            (i for i, m in enumerate(self.messages) if m is message), None,
        )
        if index is None:
            return None
        dropped_ids = {m.id for m in self.messages[index:]}
        # Revert only closed, unlocked turns that are being dropped; their
        # file edits can still be undone.
        revert = [
            TurnCheckpoint(t.before_tree, t.after_tree)
            for t in reversed(self.turns)
            if t.user_message_id in dropped_ids and not t.locked
            and t.before_tree and t.after_tree
        ]
        return RollbackPlan(
            user_text=message.content, revert_turns=revert[::-1],
            keep=list(self.messages[:index]), epoch=message.epoch,
            images=image_refs(message) or None,
        )

    def apply_rollback(self, plan: RollbackPlan) -> None:
        """Truncate the conversation to the plan. File edits are reversed
        separately, so a conflict can abort before the conversation changes."""
        self.messages = list(plan.keep)
        self.epoch = plan.epoch
        kept = {m.id for m in self.messages}
        self.turns = [t for t in self.turns if t.user_message_id in kept]

    def lock_turns(self, tree: str) -> None:
        """Point every turn's checkpoints at ``tree`` (an accepted/rejected
        baseline) and mark the closed ones locked. The open turn keeps
        ``after_tree = None`` so finish_turn still records its end."""
        open_turn = self.open_turn
        for turn in self.turns:
            turn.before_tree = tree
            if turn is not open_turn:
                turn.after_tree = tree
                turn.locked = True

    def repair_orphaned_tool_calls(self, now: str | None = None) -> bool:
        """Give any assistant tool_call without a matching tool message a
        synthetic interrupted result (``kind="interrupted"``, ``ok=False``),
        so the next API request is valid. Runs after a cancel and on a
        session load. Returns True when anything was added."""
        visible = self.visible_messages()
        answered = {
            m.tool_call_id for m in visible
            if m.role == "tool" and m.tool_call_id
        }
        changed = False
        for message in visible:
            if message.role != "assistant" or not message.tool_calls:
                continue
            for call in message.tool_calls:
                if call.call_id not in answered:
                    self.add(Message(
                        role="tool", tool_call_id=call.call_id,
                        content="interrupted", kind="interrupted", ok=False,
                        end_time=now,
                    ))
                    answered.add(call.call_id)
                    changed = True
        return changed

    def to_wire(
        self, keep_reasoning: bool = False, system: str | None = None,
        image_loader: Callable[[str], bytes | None] | None = None,
    ) -> list[dict[str, Any]]:
        # Orphaned tool calls (turn interrupted mid-execution, session saved
        # mid-turn) are answered synthetically by Session._repair_tool_calls,
        # which runs on cancel and on session load, so every message can go
        # out as-is. The system prompt is derived per request, never stored.
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.extend(
            message.to_wire(
                include_reasoning=keep_reasoning, image_loader=image_loader,
            )
            for message in self.visible_messages()
        )
        return messages
