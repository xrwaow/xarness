"""Glue between the API client and the session.

Owns the :class:`Conversation` (plain data, no widgets) and enriches raw
client events with cross-event information: reasoning timing, usage fallback
estimation, and appending finished messages (assistant text, tool calls, tool
results) to the history.

The controller is a *round runner*: :meth:`stream_round` streams exactly one
round — one full model response. If that round made tool calls
(``RoundComplete.has_tool_calls``), the caller executes them, calls
:meth:`record_tool_result` for each, then calls :meth:`stream_round` again.
The controller does not loop internally; the session stays in control of
pacing tool execution and persistence between rounds.

Turn bookkeeping (per-turn git checkpoints, rollback) lives on the
conversation itself; :meth:`finish_turn` records the after-tree of the open
turn.
"""

from __future__ import annotations

import base64
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .client import ChatClient, StreamClient
from .compaction import CHARS_PER_TOKEN
from .compaction import compact as run_compaction
from .compaction import summarize as run_summarize
from .config import ProviderProfile
from .conversation import (
    AssistantMessage,
    Conversation,
    Message,
    RollbackPlan,
    Timing,
    ToolCall,
    ToolMessage,
    Usage,
    user_message,
)
from .gitwork import (
    GitInfo,
    GitWorktreeError,
    RevertConflict,
    TurnCheckpoint,
    revert_turns,
)
from .gitwork import snapshot_state as git_snapshot_state
from .images import ImageRef
from .protocol import (
    ContentDelta,
    ReasoningDelta,
    RoundComplete,
    StreamEvent,
    ToolCallArgumentsDone,
    ToolCallStarted,
)
from .util import utcnow_iso as _utcnow_iso

if TYPE_CHECKING:
    from .compaction import CompactionResult
    from .tools import ToolRegistry


# A round's live partial output, mutated in place while it streams.
@dataclass(slots=True)
class PartialRound:
    reasoning_parts: list[str]
    content_parts: list[str]


class ChatController:
    """One instance per session; the session consumes ``stream_round()``."""

    def __init__(
        self,
        profile: ProviderProfile,
        api_key: str | None,
        client: StreamClient | None = None,
        tool_registry: "ToolRegistry | None" = None,
    ) -> None:
        self.profile = profile
        self.conversation = Conversation()
        self._client = client or ChatClient(profile, api_key)
        self.tools = tool_registry
        # Set by the session when it has git change tracking; drives
        # per-turn checkpoints.
        self.git_info: GitInfo | None = None
        # Cumulative session spend, folded together from every round's
        # usage — including compaction's own summarization round and
        # everything an /undo later removes. Monotonic: it counts everything
        # ever spent this session.
        self.usage_total = Usage(input_tokens=0, output_tokens=0)
        # Usage of the most recent round, kept for the status payload
        # (instead of re-scanning the message list on every status emit).
        self.last_usage = Usage(input_tokens=0, output_tokens=0)
        # Live view of the round currently streaming. Cleared when the round
        # completes; read by save_interrupted_round() when a turn is cancelled.
        self._partial_round: PartialRound | None = None
        # The system prompt is derived at request time (never persisted).
        self.system_prompt: str | None = None
        # Optional hook, called with each user message as it lands in the
        # history (send and inject_user_message). The session uses it to
        # emit MessageAdded — there is no local echo in the client.
        self.on_user_message: Callable[[Message], None] | None = None
        # Image blob storage, supplied by the session (content-addressed).
        # Without one, parts carry the bytes transiently (tests, headless
        # throwaway conversations).
        self.image_store: Any = None

    def _image_url(self, ref: ImageRef) -> str | None:
        """A data URL for one stored image (from the blob store's cache);
        None when there is no store or the blob is gone."""
        if self.image_store is None:
            return None
        loader = getattr(self.image_store, "data_url", None)
        if loader is not None:
            return loader(ref)
        data = self.image_store.load(ref.id)
        if data is None:
            return None
        return f"data:{ref.mime};base64,{base64.b64encode(data).decode('ascii')}"

    async def begin_turn(
        self, text: str, images: list[ImageRef] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Add the user message, open the turn's checkpoint, and stream the
        first round.

        The checkpoint is ALWAYS attached — with ``before_tree=None`` when
        there is no tracking or the snapshot failed — so every turn has an
        entry and a rollback never has to guess its scope."""
        message = user_message(text, images, self.image_store)
        before_tree = await self._take_checkpoint()
        self.conversation.add(message)
        self.conversation.begin_turn(message, before_tree)
        if self.on_user_message is not None:
            self.on_user_message(message)
        async for event in self.stream_round():
            yield event

    def stream_round(self) -> AsyncIterator[StreamEvent]:
        """Stream one round (see the module docstring)."""
        return self._stream_round()

    def record_tool_result(
        self, call_id: str, result: Any,
        start_time: str | None = None, end_time: str | None = None,
    ) -> None:
        """Append a ``tool`` message with the outcome of one call.

        ``content`` is stored as-is — a failing call keeps its real output,
        so failing test output reaches the model."""
        self.conversation.add(ToolMessage(
            tool_call_id=call_id,
            content=result.content,
            status=result.status,
            timing=Timing(start=start_time, end=end_time),
        ))

    def save_interrupted_round(self) -> bool:
        """Persist the partial round a turn was streaming when it ended early.

        Called both when the user interrupts a turn (cancel, or a forced
        steer that cancels it) and when a round ends without completing (a
        provider or transport error). Keeps the assistant text and the
        partial chain of thought that had arrived — including a
        reasoning-only segment (a cancel mid-thinking now saves the CoT,
        instead of discarding it) — and drops any half-streamed tool call
        (it has no result to pair with, and an orphaned tool call would be
        rejected on the next request). Returns True when a message was
        added; a round with neither text nor reasoning adds nothing, so a
        turn that had not started answering leaves the history untouched.
        """
        partial = self._partial_round
        self._partial_round = None
        if partial is None:
            return False
        content = "".join(partial.content_parts)
        reasoning = "".join(partial.reasoning_parts) or None
        if not content and not reasoning:
            return False
        self.conversation.add(AssistantMessage(
            content=content, reasoning=reasoning, interrupted=True,
        ))
        return True

    def inject_user_message(
        self, text: str, images: list[ImageRef] | None = None,
        send_time: str | None = None,
    ) -> None:
        """Add a user message mid-turn (steering).

        Called at a round boundary — after tool results are recorded, before
        the next round streams — so a message typed while the agent works is
        seen by the very next model call instead of waiting for the turn to
        end. ``send_time`` is when the client sent it (steered messages keep
        their original stamp). No checkpoint of its own: it belongs to the
        turn that is already running, whose snapshot /undo restores anyway.
        """
        message = user_message(
            text, images, self.image_store, timing=Timing(send=send_time),
        )
        self.conversation.add(message)
        if self.on_user_message is not None:
            self.on_user_message(message)

    def switch_profile(self, profile: ProviderProfile, api_key: str | None) -> None:
        """Point the session at a new provider profile.

        The client instance, conversation, and tool registry carry over —
        only the endpoint/key change. Clients that don't support in-place
        re-targeting (test fakes) simply keep streaming their script.
        """
        self.profile = profile
        switch = getattr(self._client, "switch_profile", None)
        if switch is not None:
            switch(profile, api_key)

    # ------------------------------------------------------------------
    # per-turn git checkpoint

    async def _take_checkpoint(self) -> str | None:
        """Snapshot the workspace's current state as a tree and return the
        sha. None when there is no git tracking or the checkpoint failed."""
        info = self.git_info
        if info is None:
            return None
        try:
            return await git_snapshot_state(info)
        except GitWorktreeError:
            return None

    # ------------------------------------------------------------------
    # /undo + /retry

    def rollback_plan(self) -> RollbackPlan | None:
        """Plan the rollback of the last user turn (see
        :meth:`Conversation.rollback_plan`)."""
        return self.conversation.rollback_plan()

    def rollback_plan_to(self, message: Message) -> RollbackPlan | None:
        """Plan a rollback to an arbitrary earlier user message (see
        :meth:`Conversation.rollback_plan_to`)."""
        return self.conversation.rollback_plan_to(message)

    def apply_rollback(self, plan: RollbackPlan) -> None:
        """Truncate the conversation to the plan. File edits are reversed
        separately (see :meth:`revert_changes`) so a conflict can abort before
        the conversation changes. Usage totals are monotonic and untouched."""
        self.conversation.apply_rollback(plan)

    @property
    def open_turn(self) -> TurnCheckpoint | None:
        """The still-open turn's checkpoint, or None."""
        return self.conversation.open_turn

    async def finish_turn(self) -> None:
        """Record the workspace state after the turn's edits as the open
        turn's after tree, so /undo can reverse exactly this turn's diff
        later, and mark the turn closed. No-op without an open turn; a
        failed snapshot closes the turn without an after tree (its edits
        just become unrevertable, like untracked turns')."""
        turn = self.conversation.open_turn
        if turn is None:
            return
        info = self.git_info
        if info is not None and turn.before_tree:
            try:
                turn.after_tree = await git_snapshot_state(info)
            except GitWorktreeError:
                turn.after_tree = None
            else:
                # The turn's end state is now the workspace the session
                # knows about; resume drift must measure from here, not from
                # the previous turn.
                info.mark_seen(turn.after_tree)
        turn.closed = True

    def open_turn_before(self) -> str | None:
        """The still-open turn's before tree, or None."""
        turn = self.conversation.open_turn
        return turn.before_tree if turn is not None else None

    def turn_checkpoints(self) -> list[TurnCheckpoint]:
        """The reversible turns since the baseline, oldest first."""
        return self.conversation.turn_checkpoints()

    async def revert_changes(self, turns: list[TurnCheckpoint]) -> None:
        """Reverse the given turns' file edits (newest first) onto the live
        workspace.

        Raises :class:`RevertConflict` when a turn's changes overlap
        unresolvably with edits made since (the workspace is left untouched),
        or :class:`GitWorktreeError` on another git failure. No tracking, or
        an empty turn list, is a clean no-op."""
        info = self.git_info
        if info is None or not turns:
            return
        conflict = await revert_turns(info, turns)
        if conflict is not None:
            raise RevertConflict(conflict)

    # ------------------------------------------------------------------
    # compaction

    async def compact(self) -> "CompactionResult | None":
        """Summarize the conversation and shrink the visible slice to the
        summary message (old messages stay in the append-only list and come
        back if the summary is popped). Returns the
        :class:`~xarness.compaction.CompactionResult`, or None when there
        was nothing to do."""
        result = await run_compaction(
            self.conversation, self._client, summarizer=self._summarize,
            system=self.system_prompt,
            keep_reasoning=self.profile.keep_reasoning,
            image_loader=self._image_url,
        )
        if result is None:
            return None
        if result.usage is not None:
            self._fold_usage(result.usage)
        return result

    async def _summarize(
        self, wire_messages: list[dict[str, Any]],
    ) -> tuple[str, Usage | None]:
        """One-off handoff round through the same client, no tools."""
        return await run_summarize(self._client, wire_messages)

    # ------------------------------------------------------------------
    # streaming

    async def _stream_round(self) -> AsyncIterator[StreamEvent]:
        partial = PartialRound(reasoning_parts=[], content_parts=[])
        done_calls: list[ToolCall] = []
        first_reasoning_at: float | None = None
        # The cot block ends where the answer or the first tool call starts —
        # tool-argument streaming counts as call time, not thinking time.
        first_answer_at: float | None = None
        # Expose this round's partial output live, so a cancelled turn can
        # persist whatever had streamed before the interruption.
        self._partial_round = partial
        started_at = _utcnow_iso()

        tools_schema = self.tools.schema() if self.tools is not None else None
        async for event in self._client.stream(
            self.conversation.to_wire(
                keep_reasoning=self.profile.keep_reasoning,
                system=self.system_prompt,
                image_loader=self._image_url,
            ),
            tools=tools_schema,
        ):
            if isinstance(event, ReasoningDelta):
                if first_reasoning_at is None:
                    first_reasoning_at = time.monotonic()
                partial.reasoning_parts.append(event.text)
            elif isinstance(event, (ContentDelta, ToolCallStarted)):
                if first_answer_at is None:
                    first_answer_at = time.monotonic()
                if isinstance(event, ContentDelta):
                    partial.content_parts.append(event.text)
            elif isinstance(event, ToolCallArgumentsDone):
                # Authoritative once the stream is done parsing; the client
                # has already assembled the full arguments for this call.
                done_calls.append(ToolCall(
                    call_id=event.call_id, name=event.name,
                    arguments_json=event.arguments_json,
                ))
            elif isinstance(event, RoundComplete):
                self._partial_round = None  # round completed; nothing partial left
                reasoning = "".join(partial.reasoning_parts) or None
                reasoning_seconds: float | None = None
                if first_reasoning_at is not None:
                    end = first_answer_at or time.monotonic()
                    reasoning_seconds = end - first_reasoning_at
                usage = event.usage or self._estimate_usage(partial.content_parts)
                self._fold_usage(usage)
                self.conversation.add(AssistantMessage(
                    content="".join(partial.content_parts),
                    reasoning=reasoning,
                    tool_calls=done_calls or None,
                    usage=usage,
                    # Wall-clock stamps taken at the moment the round
                    # actually streamed (never reconstructed later).
                    timing=Timing(
                        start=started_at, end=_utcnow_iso(),
                        reasoning_seconds=reasoning_seconds,
                    ),
                ))
                event = RoundComplete(
                    usage=usage,
                    has_tool_calls=bool(done_calls),
                )
            yield event

    def _fold_usage(self, usage: Usage) -> None:
        """Add one round's usage to the cumulative session total. Copies, so
        already-emitted status payloads keep their point-in-time values."""
        self.usage_total = Usage(
            input_tokens=self.usage_total.input_tokens + usage.input_tokens,
            output_tokens=self.usage_total.output_tokens + usage.output_tokens,
            cached_tokens=self.usage_total.cached_tokens + usage.cached_tokens,
            reasoning_tokens=self.usage_total.reasoning_tokens + usage.reasoning_tokens,
        )
        self.last_usage = usage.model_copy()

    def _estimate_usage(self, content_parts: list[str]) -> Usage:
        prompt_chars = sum(
            len(message.content or "")
            for message in self.conversation.visible_messages()
        )
        return Usage(
            input_tokens=prompt_chars // CHARS_PER_TOKEN,
            output_tokens=sum(len(part) for part in content_parts) // CHARS_PER_TOKEN,
            approximate=True,
        )
