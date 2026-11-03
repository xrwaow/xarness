"""In-memory conversation state, kept separate from the widget tree.

Messages are a discriminated union on ``role`` — :class:`UserMessage`,
:class:`AssistantMessage`, :class:`ToolMessage` — so every field is valid
for exactly the role that uses it. They are pydantic models so the very
same objects are persisted (session JSON) and shown to clients
(snapshots) — there is no second mirrored model to keep in sync.

History is append-only across compaction: a compaction appends one summary
user message, and the *visible* slice is ``messages[i:]`` where ``i`` is
just past the last summary. Undoing a compaction pops that summary; a
rollback to an earlier message simply truncates (which drops the summary
with it, when one is in the dropped range).

Reasoning text and token usage are stored locally per assistant message; by
default reasoning is also sent back on later rounds (``keep_reasoning``),
but usage never reaches the wire.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, NamedTuple, Union

from pydantic import BaseModel, Field, PrivateAttr, TypeAdapter

from .gitwork import TurnCheckpoint
from .images import IMAGE_TOKEN_RE, ImageAttachment, ImageRef, token_text

# Every state a tool result can be in. ``parse_error`` means the model's
# arguments were invalid (it should retry with valid ones); ``interrupted``
# is the synthetic result an orphaned tool call gets after a cancel.
ToolStatus = Literal["ok", "error", "parse_error", "interrupted"]


class ToolResult(NamedTuple):
    """One tool call's outcome: a status and the single text both the model
    and the UI see. A failing call keeps its real output in ``content`` (a
    failing test run must reach the model, not just its exit code)."""

    status: ToolStatus = "ok"
    content: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

# The marker a compaction's summary message starts with (a rendering detail;
# identification is the ``is_summary`` flag).
SUMMARY_PREFIX = "[earlier conversation, summarized]"


class TextPart(BaseModel):
    """One text segment of a message."""

    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    """One image segment, ordered where it appeared in the text.

    ``ref`` is the persisted/wire shape (content hash + display metadata);
    the bytes live in the session's image store."""

    type: Literal["image"] = "image"
    ref: ImageRef


MessagePart = TextPart | ImagePart


def build_parts(
    text: str, images: list["ImageRef"] | None, store: Any,
) -> list[MessagePart]:
    """Build ordered text/image parts from a user message.

    The text carries one ``[🖼 name W×H #id]`` token per attachment; parts are
    emitted in token order so ``compare [img1] with [img2]`` keeps its
    meaning. A token without a matching attachment stays text, and an
    attachment without a token is appended at the end. ``store`` (an object
    with ``save(ImageAttachment) -> ImageRef``) is mandatory: it writes
    attachment bytes to the blob store.
    """
    if not images:
        return [TextPart(text=text)] if text else []
    by_token = {token_text(image): image for image in images}
    used: set[int] = set()
    parts: list[MessagePart] = []

    def add_image(image: ImageRef) -> None:
        parts.append(ImagePart(ref=store.save(image) if isinstance(
            image, ImageAttachment) else image))

    last = 0
    for match in IMAGE_TOKEN_RE.finditer(text):
        chunk = text[last:match.start()]
        if chunk:
            parts.append(TextPart(text=chunk))
        image = by_token.get(match.group(0))
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
        part.ref for part in getattr(message, "parts", None) or ()
        if isinstance(part, ImagePart)
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
    """Token usage for one round.

    A pydantic model (not a dataclass) so the very same object crosses the
    wire in snapshots and persists losslessly. ``cached_tokens`` /
    ``reasoning_tokens`` account for prefix-cache hits and thinking-channel
    spend when the provider reports them."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    approximate: bool = False

    def context_tokens(self, keep_reasoning: bool = True) -> int:
        """The context size this round implies (input side).

        Reasoning tokens count only when they are actually re-sent: with
        ``keep_reasoning`` off the next request drops them, so counting
        them would overestimate the context and trigger compaction early."""
        output = self.output_tokens
        if not keep_reasoning and self.reasoning_tokens:
            output = max(0, output - self.reasoning_tokens)
        return self.input_tokens + output


class Timing(BaseModel):
    """When a message happened and (for a turn's last answer) how long it took.

    One sub-model instead of six fields scattered on each message. Stamps are
    wall-clock UTC ISO 8601; ``reasoning_seconds`` is the derived thinking
    duration stamped only on the turn's last assistant message so every
    client and every reload shows the same one."""

    send: str | None = None
    start: str | None = None
    end: str | None = None
    reasoning_seconds: float | None = None


class BaseMessage(BaseModel):
    """Fields every message carries."""

    # Stable id (uuid4), persisted. Click-to-undo addresses messages by
    # id, not object identity.
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    # Timestamps. Never sent to the provider.
    timing: Timing = Field(default_factory=Timing)

    # True only for the handoff message a compaction appends. A field (not a
    # property) so ``UserMessage`` can persist it, but excluded everywhere
    # else: only the user role ever sets it, and every caller can read it
    # without a ``role == "user" and ...`` guard.
    is_summary: bool = Field(default=False, exclude=True)


class UserMessage(BaseMessage):
    """A user turn opener or steering message. Text and images live in
    ``parts`` — the single source, in order; ``content`` is the joined
    text, derived."""

    role: Literal["user"] = "user"
    parts: list[TextPart | ImagePart] = Field(default_factory=list)
    # True for the handoff message compaction appends (re-shared: the base
    # declaration excludes it, this one persists it).
    is_summary: bool = False

    @property
    def content(self) -> str:
        """The joined text of the message's text parts (derived, never stored)."""
        return "".join(
            part.text for part in self.parts if isinstance(part, TextPart)
        )

    def to_wire(
        self, include_reasoning: bool = False,
        image_loader: "Callable[[ImageRef], str | None] | None" = None,
    ) -> dict[str, Any]:
        image_parts = [p for p in self.parts if isinstance(p, ImagePart)]
        if not image_parts:
            return {"role": "user", "content": self.content}
        # Multipart content, in the order the text/image parts were
        # recorded. The image data-URL comes from the loader (the session
        # blob store's cache).
        content: list[dict[str, Any]] = []
        for part in self.parts:
            if isinstance(part, TextPart):
                if part.text:
                    content.append({"type": "text", "text": part.text})
                continue
            url = image_loader(part.ref) if image_loader is not None else None
            if url is None:
                # The blob is gone: keep the message valid rather than
                # silently dropping the attachment entirely.
                content.append({"type": "text", "text": token_text(part.ref)})
                continue
            content.append({"type": "image_url", "image_url": {"url": url}})
        return {"role": "user", "content": content}


class AssistantMessage(BaseMessage):
    """One model response: the final text, the reasoning channel, and any
    tool calls it made. ``interrupted`` marks a partial message saved when
    a turn was cancelled."""

    role: Literal["assistant"] = "assistant"
    content: str = ""
    # Local-only: reasoning-channel text for assistant messages.
    reasoning: str | None = None
    # Local-only: token usage reported for the round that produced this
    # message (never sent over the wire).
    usage: Usage | None = None
    tool_calls: list[ToolCall] | None = None
    interrupted: bool = False

    def to_wire(
        self, include_reasoning: bool = False,
        image_loader: "Callable[[ImageRef], str | None] | None" = None,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant"}
        # Per spec, an assistant message that only calls tools should omit
        # content entirely — some providers reject an empty string there.
        if self.content or not self.tool_calls:
            message["content"] = self.content
        if include_reasoning and self.reasoning:
            # DeepSeek-style field name; providers that don't know it ignore it.
            message["reasoning_content"] = self.reasoning
        if self.tool_calls is not None:
            message["tool_calls"] = [call.to_wire() for call in self.tool_calls]
        return message


class ToolMessage(BaseMessage):
    """One tool result, addressed by ``tool_call_id``. ``content`` is the
    single text both the model and the UI see — a failing call keeps its
    real output there."""

    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str = ""
    status: ToolStatus = "ok"

    def to_wire(
        self, include_reasoning: bool = False,
        image_loader: "Callable[[ImageRef], str | None] | None" = None,
    ) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": self.tool_call_id,
                "content": self.content}


Message = Annotated[
    Union[UserMessage, AssistantMessage, ToolMessage],
    Field(discriminator="role"),
]


def user_message(
    text: str,
    images: list[ImageAttachment] | None = None,
    store: Any = None,
    **kwargs: Any,
) -> UserMessage:
    """A user message with its text/image parts in order."""
    return UserMessage(parts=build_parts(text, images, store), **kwargs)


MESSAGE_ADAPTER = TypeAdapter(Message)


@dataclass(slots=True)
class RollbackPlan:
    """What /undo or /retry needs to do, computed by
    :meth:`Conversation.rollback_plan`.

    Usage totals are monotonic (they count everything ever spent, including
    undone turns), so a plan carries no usage bookkeeping."""

    user_text: str  # the turn's user message (input box / resend)
    revert_turns: list[TurnCheckpoint]  # file diffs to reverse (oldest first)
    # The conversation becomes ``messages[:drop_from]``. Rolling back past a
    # compaction's summary truncates the summary away automatically.
    drop_from: int
    images: list[ImageRef] | None = None
    # True for a compaction undo: no messages but the summary are dropped,
    # the turn stays intact, no files are reverted.
    compaction: bool = False


class Conversation(BaseModel):
    """Ordered message history; the single source of truth for the chat.

    ``messages`` is append-only across compaction: a compaction appends a
    summary user message and everything before it drops out of the visible
    slice (but stays for the human transcript) until the summary is popped.

    Turn checkpoints live in ``turns``, keyed by the opening user message's
    id — messages are truly immutable and append-only is sound (a turn end,
    an accept or a rollback mutates only this side table, never a persisted
    message row). A message id without an entry is a steering message, a
    summary, or a legacy message from before tracking."""

    messages: list[Message] = Field(default_factory=list)
    turns: dict[str, TurnCheckpoint] = Field(default_factory=dict)

    # First index of the visible slice (just at/after the last summary).
    # Maintained by every mutator below, so no caller has to re-scan.
    _visible_from: int = PrivateAttr(default=0)
    # Watermark: the first message index not known to be on disk, or None
    # when the whole history is persisted. ``Conversation`` is the only
    # class that mutates messages, so it owns this invariant; the persist
    # code only reads and resets it.
    _dirty_from: int | None = PrivateAttr(default=None)

    def add(self, message: Message) -> Message:
        index = len(self.messages)
        if message.is_summary:
            self._visible_from = index
        self._mark_dirty(index)
        self.messages.append(message)
        return message

    def _mark_dirty(self, index: int) -> None:
        """Lower the persist watermark to ``index`` (the lowest index whose
        row on disk is now stale)."""
        self._dirty_from = index if self._dirty_from is None else min(
            self._dirty_from, index)

    def take_dirty_from(self) -> int:
        """The persist watermark, reset to "everything is on disk".

        Called synchronously by the persist code when it serializes: the
        returned index and the reset describe the same list state, so a
        failed write restores it by marking from there again. A clean
        conversation (or a state-only write, e.g. a checkpoint rebaseline)
        returns ``len(messages)``: nothing in the message tail changed."""
        index = len(self.messages) if self._dirty_from is None \
            else self._dirty_from
        self._dirty_from = None
        return index

    def mark_clean(self) -> None:
        """Declare the whole history on disk (a freshly loaded session)."""
        self._dirty_from = None

    def _recompute_visible(self) -> None:
        """Recompute the visible slice's start (after a bulk rewrite)."""
        self._visible_from = next(
            (i for i in range(len(self.messages) - 1, -1, -1)
             if self.messages[i].is_summary), 0,
        )

    def visible_index(self) -> int:
        """First index of the visible slice: the last summary message (the
        model sees the summary plus everything after it)."""
        return self._visible_from

    def visible_messages(self) -> list[Message]:
        return self.messages[self.visible_index():]

    def history_messages(self) -> list[Message]:
        """Everything the visible slice hides (compacted-away transcript)."""
        return self.messages[:self.visible_index()]

    @property
    def open_turn(self) -> TurnCheckpoint | None:
        """The still-open turn's checkpoint (``closed`` not yet set), or
        None.

        ``turns`` is insertion-ordered and entries are added in message
        order, so the last unclosed entry is the open turn — no walk over
        the whole message list."""
        for turn in reversed(list(self.turns.values())):
            if not turn.closed:
                return turn
        return None

    def close_crashed_turns(self) -> list[TurnCheckpoint]:
        """Close every turn left open by a crash (a loaded session can never
        have a running turn).

        Tracked turns (``before_tree`` set) with no after tree get the
        before tree as a placeholder — "no edits attributed" — and are
        returned so the caller can stamp the real workspace tree over it
        and mark it seen. Untracked turns (``before_tree`` None) are just
        closed: the host's tree-based repair never saw them, and a turn
        left open forever makes ``open_turn`` and ``open_turn_before``
        return stale results."""
        repaired: list[TurnCheckpoint] = []
        for turn in self.turns.values():
            if turn.closed:
                continue
            turn.closed = True
            if turn.before_tree is not None and turn.after_tree is None:
                turn.after_tree = turn.before_tree
                repaired.append(turn)
        return repaired

    def begin_turn(self, user_message: UserMessage, before_tree: str | None) -> None:
        """Attach a fresh checkpoint to the turn-opening user message.

        Always called — with ``before_tree=None`` when there is no git
        tracking or the snapshot failed — so every real turn has an entry
        and a rollback plan never has to guess from "the last user
        message" (which could be steering)."""
        self.turns[user_message.id] = TurnCheckpoint(
            before_tree=before_tree, after_tree=None,
        )

    # -- rollback planning ------------------------------------------------

    def message_by_id(self, message_id: str) -> Message | None:
        return next((m for m in self.messages if m.id == message_id), None)

    def last_user_message_at(self) -> str | None:
        """Latest user send_time (derived, never stored separately).

        All messages, not just the visible slice: after a compaction with no
        new user message the visible slice starts at the summary and the
        session would sort by name instead of recency. Messages are appended
        in time order, so the first user message found from the end is the
        answer — O(1) in practice."""
        for message in reversed(self.messages):
            if message.role == "user" and message.timing.send:
                return message.timing.send
        return None

    def rollback_plan(self) -> RollbackPlan | None:
        """Plan the rollback of the last user turn, or None if there is
        nothing to undo (no user message, or no response to it yet).

        If the most recent thing that happened is a compaction (a summary is
        the last message), /undo //retry undo the compaction itself first by
        popping the summary; the turn that triggered it stays intact and no
        files are touched."""
        if self.messages and self.messages[-1].is_summary:
            return RollbackPlan(
                user_text="", revert_turns=[],
                drop_from=len(self.messages) - 1, compaction=True,
            )
        turn_message = next(
            (m for m in reversed(self.messages)
             if m.role == "user" and m.id in self.turns),
            None,
        )
        if turn_message is None:
            # Legacy message without a tracked turn: fall back to the last
            # real user message that has a response after it.
            visible = self.visible_messages()
            turn_message = next(
                (m for m in reversed(visible)
                 if m.role == "user" and not m.is_summary
                 and m is not visible[-1]),
                None,
            )
        if turn_message is None or (
            self.messages and turn_message is self.messages[-1]
        ):
            return None  # the turn produced no response yet
        return self.rollback_plan_to(turn_message)

    def rollback_plan_to(self, message: Message) -> RollbackPlan | None:
        """Plan a rollback to an arbitrary earlier user message
        (click-to-undo): drop that message and everything after it, restoring
        the workspace to the checkpoint taken when it was sent.

        Searches the full append-only history, so messages the last compaction
        summarized away are found too (truncating past the summary drops it).
        Returns None when the message is not a user turn in this conversation.
        """
        if message.role != "user" or message.is_summary:
            return None
        index = next(
            (i for i, m in enumerate(self.messages) if m is message), None,
        )
        if index is None:
            return None
        # Revert only closed turns that are being dropped and actually
        # changed files; their file edits can still be undone.
        revert = [
            self.turns[message.id] for message in reversed(self.messages[index:])
            if (turn := self.turns.get(message.id)) is not None
            and turn.before_tree and turn.after_tree
            and turn.before_tree != turn.after_tree
        ]
        return RollbackPlan(
            user_text=message.content, revert_turns=revert[::-1], drop_from=index,
            images=image_refs(message) or None,
        )

    def apply_rollback(self, plan: RollbackPlan) -> None:
        """Truncate the conversation to the plan. File edits are reversed
        separately, so a conflict can abort before the conversation changes.

        The dropped turns' checkpoints are pruned too: turns are keyed by
        message id, and a pruned id must never resurrect."""
        self.messages = self.messages[:plan.drop_from]
        self._mark_dirty(min(plan.drop_from, len(self.messages)))
        self._recompute_visible()
        keep = {m.id for m in self.messages}
        self.turns = {
            message_id: turn for message_id, turn in self.turns.items()
            if message_id in keep
        }

    def lock_turns(self, tree: str) -> None:
        """Point every turn's checkpoints at ``tree`` (an accepted/rejected
        baseline). A turn whose before/after trees are equal can no longer be
        reverted — reverting it is a no-op — which is what "locked" means.
        The open turn keeps ``after_tree = None`` so finish_turn still
        records its end."""
        open_turn = self.open_turn
        for turn in self.turns.values():
            turn.before_tree = tree
            if turn is not open_turn:
                turn.after_tree = tree

    def turn_checkpoints(self) -> list[TurnCheckpoint]:
        """The reversible turns since the baseline, oldest first."""
        return [
            turn for message in self.messages
            if (turn := self.turns.get(message.id)) is not None
            and turn.before_tree and turn.after_tree
            and turn.before_tree != turn.after_tree
        ]

    def known_trees(self) -> list[str]:
        """Every after-tree the conversation ever recorded (locked turns
        included): the workspace states the session has accounted for."""
        return [
            turn.after_tree for message in self.messages
            if (turn := self.turns.get(message.id)) is not None
            and turn.after_tree
        ]

    def repair_orphaned_tool_calls(self, now: str | None = None) -> bool:
        """Give any assistant tool_call without a matching tool message a
        synthetic interrupted result, so the next API request is valid. Runs
        after a cancel and on a session load.

        Returns True when anything was inserted — the inserted rows shift a
        tail-only persistence watermark, which this lowers itself."""
        answered = {
            m.tool_call_id for m in self.messages
            if m.role == "tool" and m.tool_call_id
        }
        lowest: int | None = None
        # Insert each synthetic result right after the assistant message
        # that owns the call: appending at the end of the history would
        # leave the tool message non-adjacent to its assistant message
        # (a session saved mid-tool-loop and then steered), which some
        # providers reject.
        for index in range(len(self.messages) - 1, -1, -1):
            message = self.messages[index]
            if message.role != "assistant" or not message.tool_calls:
                continue
            missing = [c for c in message.tool_calls if c.call_id not in answered]
            if not missing:
                continue
            # One tool message per missing call, all directly after the
            # owning assistant message.
            for offset, call in enumerate(missing, start=1):
                answered.add(call.call_id)
                self.messages.insert(index + offset, ToolMessage(
                    tool_call_id=call.call_id,
                    content="interrupted", status="interrupted",
                    timing=Timing(end=now),
                ))
            self._recompute_visible()
            lowest = index if lowest is None else min(lowest, index)
        if lowest is not None:
            self._mark_dirty(lowest)
        return lowest is not None

    def to_wire(
        self, keep_reasoning: bool = False, system: str | None = None,
        image_loader: Callable[[ImageRef], str | None] | None = None,
    ) -> list[dict[str, Any]]:
        # Orphaned tool calls (turn interrupted mid-execution, session saved
        # mid-turn) are answered synthetically by Session._repair_tool_calls,
        # which runs on cancel and on session load, so every message can go
        # out as-is. The system prompt is derived per request, never stored.
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        for message in self.visible_messages():
            if (
                message.role == "assistant" and message.interrupted
                and not message.content and not message.tool_calls
            ):
                # A reasoning-only partial: an empty assistant content string
                # is rejected by some providers, and the reasoning alone is
                # not worth a message slot on the wire.
                continue
            messages.append(
                message.to_wire(
                    include_reasoning=keep_reasoning, image_loader=image_loader,
                )
            )
        return messages
