"""Glue between the API client and the UI.

Owns the :class:`Conversation` (plain data, no widgets) and enriches raw
client events with cross-event information: reasoning timing, usage fallback
estimation, and appending finished messages (assistant text, tool calls, tool
results) to the history.

``send()`` streams exactly one round — one full model response. If that round
made tool calls (``TurnComplete.has_tool_calls``), the app executes them,
calls :meth:`record_tool_result` for each, then calls
:meth:`continue_after_tools` for the next round. The controller does not loop
internally; the UI stays in control of pacing tool execution and rendering
between rounds.

``send()`` also takes a per-turn checkpoint: a git tree snapshot of the
workspace is recorded on the user message, and the conversation is
snapshotted — the machinery behind /undo and /retry.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from .client import ChatClient, StreamClient
from .compaction import compact as run_compaction
from .compaction import summarize as run_summarize
from .config import ProviderProfile
from .conversation import (
    Conversation,
    Message,
    RollbackPlan,
    Timing,
    ToolCall,
    build_parts,
)
from .gitwork import (
    GitInfo,
    GitWorktreeError,
    RevertConflict,
    TurnCheckpoint,
    revert_turns,
)
from .gitwork import snapshot_state as git_snapshot_state
from .images import ImageAttachment
from .protocol import (
    ContentDelta,
    ReasoningDelta,
    StreamEvent,
    ToolCallArgumentsDone,
    TurnComplete,
    Usage,
)
from .tools import ToolRegistry, ToolResult
from .util import utcnow_iso as _utcnow_iso

# Rough chars-per-token for the fallback estimate when the provider sends no
# usage data. Deliberately conservative; results are labeled approximate.
_CHARS_PER_TOKEN = 4


# A round's live partial output, mutated in place while it streams.
@dataclass(slots=True)
class PartialRound:
    reasoning_parts: list[str]
    content_parts: list[str]


class ChatController:
    """One instance per session; the UI consumes ``send()`` event streams."""

    def __init__(
        self,
        profile: ProviderProfile,
        api_key: str | None,
        client: StreamClient | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.profile = profile
        self.conversation = Conversation()
        self._client = client or ChatClient(profile, api_key)
        self.tools = tool_registry
        # Set by the app when the session has git change tracking; drives
        # per-turn checkpoints.
        self.git_info: GitInfo | None = None
        # Cumulative session spend (input + output), folded together from
        # every round's usage — including compaction's own summarization
        # round and everything an /undo later removes. Monotonic: it counts
        # everything ever spent this session.
        self.usage_total = Usage(input_tokens=0, output_tokens=0)
        # Usage of the most recent round, kept for the status payload
        # (instead of re-scanning the message list on every status emit).
        self.last_usage = Usage(input_tokens=0, output_tokens=0)
        # (before, after) token counts of the most recent compaction, for the
        # compaction notice.
        self.last_compaction: tuple[int, int] | None = None
        # Live view of the round currently streaming. Cleared when the round
        # completes; read by save_interrupted_round() when a turn is cancelled.
        self._partial_round: PartialRound | None = None
        # The system prompt is derived at request time (never persisted).
        self.system_prompt: str | None = None
        # Optional hook, called with each user message as it lands in the
        # history (send and inject_user_message). The session uses it to emit
        # UserMessageAdded — there is no local echo in the client.
        self.on_user_message: Callable[[Message], None] | None = None
        # Image blob storage, supplied by the session (content-addressed).
        # Without one, parts carry the bytes transiently (tests, headless
        # throwaway conversations).
        self.image_store: Any = None

    def _user_message(
        self, text: str, images: list[ImageAttachment] | None = None, **kwargs: Any,
    ) -> Message:
        """A user message with its text/image parts in order."""
        parts = build_parts(text, images, self.image_store) if images else None
        return Message(role="user", content=text, parts=parts, **kwargs)

    def _image_loader(self, image_id: str) -> bytes | None:
        if self.image_store is None:
            return None
        return self.image_store.load(image_id)

    async def send(self, user_text: str, images: list[ImageAttachment] | None = None) -> AsyncIterator[StreamEvent]:
        """Send a user message, streaming the first round.

        ``images`` are attachments from the client; the bytes are stored
        once and the history keeps content-addressed references."""
        message = self._user_message(user_text, images)
        before_tree = await self._take_checkpoint()
        self.conversation.add(message)
        # The turn is open from here until finish_turn records its after tree.
        self.conversation.begin_turn(message, before_tree)
        if self.on_user_message is not None:
            self.on_user_message(message)
        async for event in self._stream_round():
            yield event

    async def continue_after_tools(self) -> AsyncIterator[StreamEvent]:
        """Stream the next round, after tool results have been recorded."""
        async for event in self._stream_round():
            yield event

    def record_tool_result(
        self, call_id: str, result: ToolResult,
        start_time: str | None = None, end_time: str | None = None,
    ) -> None:
        """Append a ``tool`` role message with the outcome of one call."""
        self.conversation.add(
            Message(
                role="tool",
                tool_call_id=call_id,
                content=result.output if result.ok else (result.error or "error"),
                ok=result.ok,
                parse_error=result.parse_error,
                timing=Timing(start=start_time, end=end_time),
            )
        )

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
        self.conversation.add(
            Message(
                role="assistant",
                content=content,
                reasoning=reasoning,
                kind="interrupted",
            )
        )
        return True

    def inject_user_message(
        self, text: str, images: list[ImageAttachment] | None = None,
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
        message = self._user_message(text, images, timing=Timing(send=send_time))
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
        self.last_compaction = None

    @property
    def open_turn(self):
        """The still-open :class:`Turn` being generated, or None."""
        return self.conversation.open_turn

    async def finish_turn(self) -> None:
        """Record the workspace state after the turn's edits as the open
        turn's after tree, so /undo can reverse exactly this turn's diff
        later. No-op without git tracking, when no turn is open, or when the
        snapshot fails."""
        turn = self.conversation.open_turn
        info = self.git_info
        if info is None or turn is None or not turn.before_tree:
            return
        try:
            turn.after_tree = await git_snapshot_state(info)
        except GitWorktreeError:
            return

    def open_turn_before(self) -> str | None:
        """The still-open turn's before tree, or None."""
        turn = self.conversation.open_turn
        return turn.before_tree if turn is not None else None

    def turn_checkpoints(self) -> list[TurnCheckpoint]:
        """The reversible turns since the baseline, oldest first. Locked
        (accepted) turns are excluded: their edits must not be reverted."""
        return [
            TurnCheckpoint(t.before_tree, t.after_tree)
            for t in self.conversation.turns
            if not t.locked and t.before_tree and t.after_tree
        ]

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

    async def compact(self) -> str:
        """Summarize the conversation and replace the visible history with a
        single handoff summary by advancing the compaction epoch. Old messages
        stay in the append-only list and come back if the compaction is
        undone. Returns the summary text; before/after token counts are left
        in :attr:`last_compaction` for the compaction notice."""
        result = await run_compaction(
            self.conversation, self._client, summarizer=self._summarize,
        )
        if result is None:
            self.last_compaction = None
            return "nothing to compact yet"
        if result.usage is not None:
            self._fold_usage(result.usage)
        self.last_compaction = (result.before_tokens, result.after_tokens)
        return result.summary

    async def _summarize(self, transcript: str) -> tuple[str, Usage | None]:
        """One-off handoff round through the same client, no tools."""
        return await run_summarize(self._client, transcript)

    # ------------------------------------------------------------------
    # streaming

    async def _stream_round(self) -> AsyncIterator[StreamEvent]:
        partial = PartialRound(reasoning_parts=[], content_parts=[])
        done_calls: list[ToolCall] = []
        first_reasoning_at: float | None = None
        first_content_at: float | None = None
        # Expose this round's partial output live, so a cancelled turn can
        # persist whatever had streamed before the interruption.
        self._partial_round = partial
        started_at = _utcnow_iso()

        tools_schema = self.tools.schema() if self.tools is not None else None
        async for event in self._client.stream(
            self.conversation.to_wire(
                keep_reasoning=self.profile.keep_reasoning,
                system=self.system_prompt,
                image_loader=self._image_loader,
            ),
            tools=tools_schema,
        ):
            if isinstance(event, ReasoningDelta):
                if first_reasoning_at is None:
                    first_reasoning_at = time.monotonic()
                partial.reasoning_parts.append(event.text)
            elif isinstance(event, ContentDelta):
                if first_content_at is None:
                    first_content_at = time.monotonic()
                partial.content_parts.append(event.text)
            elif isinstance(event, ToolCallArgumentsDone):
                # Authoritative once the stream is done parsing; the client
                # has already assembled the full arguments for this call.
                done_calls.append(ToolCall(
                    call_id=event.call_id, name=event.name,
                    arguments_json=event.arguments_json,
                ))
            elif isinstance(event, TurnComplete):
                self._partial_round = None  # round completed; nothing partial left
                reasoning = "".join(partial.reasoning_parts) or None
                reasoning_seconds: float | None = None
                if first_reasoning_at is not None:
                    end = first_content_at or time.monotonic()
                    reasoning_seconds = end - first_reasoning_at
                usage = event.usage or self._estimate_usage(partial.content_parts)
                self._fold_usage(usage)
                self.conversation.add(
                    Message(
                        role="assistant",
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
                    )
                )
                event = TurnComplete(
                    usage=usage,
                    has_tool_calls=bool(done_calls),
                )
            yield event

    def _fold_usage(self, usage: Usage) -> None:
        """Add one round's usage to the cumulative session total."""
        self.usage_total.input_tokens += usage.input_tokens
        self.usage_total.output_tokens += usage.output_tokens
        self.last_usage = usage

    def _estimate_usage(self, content_parts: list[str]) -> Usage:
        prompt_chars = sum(
            len(message.content or "")
            for message in self.conversation.visible_messages()
        )
        return Usage(
            input_tokens=prompt_chars // _CHARS_PER_TOKEN,
            output_tokens=sum(len(part) for part in content_parts) // _CHARS_PER_TOKEN,
            approximate=True,
        )
