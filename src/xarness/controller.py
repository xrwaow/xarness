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
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .client import ChatClient
from .config import ProviderProfile
from .conversation import SUMMARY_PREFIX, Conversation, Message, ToolCall
from .events import (
    ContentDelta,
    ReasoningDelta,
    StreamError,
    StreamEvent,
    ToolCallArgumentsDelta,
    ToolCallArgumentsDone,
    ToolCallStarted,
    TurnComplete,
    Usage,
)
from .gitwork import GitInfo, GitWorktreeError, RevertConflict, TurnCheckpoint
from .images import ImageAttachment
from .gitwork import checkpoint as git_checkpoint
from .gitwork import revert_turns
from .tools import ToolRegistry, ToolResult

# Rough chars-per-token for the fallback estimate when the provider sends no
# usage data. Deliberately conservative; results are labeled approximate.
_CHARS_PER_TOKEN = 4

# A round's live partial output, mutated in place while it streams.
@dataclass(slots=True)
class PartialRound:
    reasoning_parts: list[str]
    content_parts: list[str]


def _is_summary(message: Message) -> bool:
    """True for the summary user message a compaction leaves behind."""
    return message.role == "user" and message.content.startswith(SUMMARY_PREFIX)


def _last_user_turn_index(messages: list[Message]) -> int | None:
    """Index of the last real user turn (summary placeholders don't count)
    that has a response after it, or None."""
    index = max(
        (i for i, m in enumerate(messages) if m.role == "user" and not _is_summary(m)
         and i < len(messages) - 1),
        default=None,
    )
    return index


def _summary_index(messages: list[Message]) -> int | None:
    """Index of the first compaction summary in the list, or None."""
    return next((i for i, m in enumerate(messages) if _is_summary(m)), None)


def _message_key(message: Message) -> tuple:
    """Value key for message identity across a save/load roundtrip."""
    return (message.role, message.content, message.tool_call_id)


def _turns_in(messages: list[Message]) -> list[TurnCheckpoint]:
    """The turns in ``messages`` that can be reversed, oldest first.

    A turn is a user message carrying both a before tree (``checkpoint_sha``)
    and an after tree. Turns from sessions saved before after-trees existed
    are skipped: their files cannot be reversed without risking the loss of
    edits made since, so /undo falls back to conversation-only rollback.

    ``revert_turns`` walks the list newest-first, so this returns
    chronological order.
    """
    return [
        TurnCheckpoint(m.checkpoint_sha, m.after_tree)
        for m in messages
        if m.role == "user" and m.checkpoint_sha and m.after_tree
    ]


class StreamClient(Protocol):
    """Anything that can stream a round; ChatClient is the real implementation."""

    def stream(
        self,
        wire_messages: Sequence[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]: ...


@dataclass(slots=True)
class RollbackPlan:
    """What /undo or /retry needs to do, computed by :meth:`rollback_plan`."""

    user_text: str  # the turn's user message (input box / resend)
    revert_turns: list[TurnCheckpoint]  # file diffs to reverse (oldest first)
    keep: list[Message]  # what the conversation becomes
    dropped: list[Message]  # messages whose usage must be subtracted
    # True when the restore point is before the most recent compaction, so
    # apply_rollback must drop the compact snapshot (the summary it covers
    # is no longer part of the history).
    clears_compaction: bool = False
    # True for a compaction-only undo: the pre-compaction history is
    # restored verbatim, the turn stays intact, no files are reverted, and
    # the input box is not touched.
    compaction_only: bool = False


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
        # round. /undo subtracts the usage of the messages it drops.
        self.usage_total = Usage(input_tokens=0, output_tokens=0)
        # Usage folded into the totals for messages that compaction later
        # removed during the current turn. /undo must subtract it too, since
        # the message objects that carry it no longer exist.
        self._absorbed_usage = Usage(input_tokens=0, output_tokens=0)
        # True once a rollback subtracted _absorbed_usage: a later compaction
        # undo must re-fold the restored turns' spend (see apply_rollback).
        self._absorbed_spent = False
        # (before, after) token counts of the most recent compaction, for the
        # compaction notice.
        self.last_compaction: tuple[int, int] | None = None
        # Live view of the round currently streaming. Cleared when the round
        # completes; read by save_interrupted_round() when a turn is cancelled.
        self._partial_round: PartialRound | None = None

    async def send(self, user_text: str, images: list[ImageAttachment] | None = None) -> AsyncIterator[StreamEvent]:
        """Send a user message, streaming the first round.

        ``images`` are base64 attachments carried on the user message and
        resent with the history on every round."""
        self.conversation.add(
            Message(
                role="user",
                content=user_text,
                images=images or None,
                checkpoint_sha=await self._take_checkpoint(),
            )
        )
        # Snapshot marker for /undo //retry: the index of this turn's user
        # message. The turn snapshot is the messages[:turn_start + 1] prefix,
        # so restoring it rolls back the whole turn — even one that compacted
        # the history mid-flight.
        self.conversation.turn_start = len(self.conversation.messages) - 1
        self._absorbed_usage = Usage(input_tokens=0, output_tokens=0)
        self._absorbed_spent = False
        async for event in self._stream_round():
            yield event

    async def continue_after_tools(self) -> AsyncIterator[StreamEvent]:
        """Stream the next round, after tool results have been recorded."""
        async for event in self._stream_round():
            yield event

    def record_tool_result(self, call_id: str, result: ToolResult) -> None:
        """Append a ``tool`` role message with the outcome of one call."""
        content = result.output if result.ok else f"error: {result.error}"
        self.conversation.add(
            Message(
                role="tool",
                tool_call_id=call_id,
                content=content,
                header=result.header or None,
            )
        )

    def save_interrupted_round(self) -> bool:
        """Persist the partial round a turn was streaming when it ended early.

        Called both when the user interrupts a turn and when a round ends
        without completing (a provider or transport error). Keeps the
        assistant text and reasoning that had arrived, but drops a trailing
        reasoning-only segment — thinking that never led to an answer — and
        any half-streamed tool call (it has no result to pair with, and an
        orphaned tool call would be rejected on the next request). Returns
        True when a message was added; an empty or reasoning-only round adds
        nothing, so a turn that had not answered yet leaves the history
        untouched.
        """
        partial = self._partial_round
        self._partial_round = None
        if partial is None:
            return False
        content = "".join(partial.content_parts)
        if not content:
            return False
        self.conversation.add(
            Message(
                role="assistant",
                content=content,
                reasoning="".join(partial.reasoning_parts) or None,
            )
        )
        return True

    def inject_user_message(self, text: str, images: list[ImageAttachment] | None = None) -> None:
        """Add a user message mid-turn (steering).

        Called at a round boundary — after tool results are recorded, before
        the next round streams — so a message typed while the agent works is
        seen by the very next model call instead of waiting for the turn to
        end. No checkpoint of its own: it belongs to the turn that is already
        running, whose snapshot /undo restores anyway.
        """
        self.conversation.add(Message(role="user", content=text, images=images or None))

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
            return await git_checkpoint(info)
        except GitWorktreeError:
            return None

    # ------------------------------------------------------------------
    # /undo + /retry

    def rollback_plan(self) -> RollbackPlan | None:
        """Plan the rollback of the last user turn, or None if there is
        nothing to undo (no user message, or no response to it yet).

        If the most recent thing that happened is a compaction (a summary in
        the history, no real user turn after it), /undo //retry undo the
        compaction itself first: the full pre-compaction history is
        restored, the turn that triggered it stays intact, and no files are
        touched. The next /undo then removes that turn normally.

        Otherwise, with a turn snapshot (the normal case) the conversation
        is restored to exactly what it was when the turn began. Without one
        (resumed session) it falls back to truncating at the last user
        message. Either way the turn's user message is dropped: /undo stops
        there, /retry re-sends its text (which takes a fresh checkpoint).
        """
        messages = self.conversation.messages
        compact_snapshot = self.conversation.compact_snapshot
        summary_idx = _summary_index(messages)
        last_real = _last_user_turn_index(messages)
        if (
            compact_snapshot is not None and summary_idx is not None
            and (last_real is None or last_real < summary_idx)
        ):
            # Undo the compaction: restore the pre-compaction history
            # verbatim. The summary and everything after it leave the
            # history (their usage is subtracted); the compaction turn's own
            # messages come back — matched by value, since after a save/load
            # the snapshot and the message list are distinct objects.
            keep = list(compact_snapshot)
            kept_values = {_message_key(m) for m in keep}
            dropped = [m for m in messages if _message_key(m) not in kept_values]
            return RollbackPlan(
                user_text="", revert_turns=[], keep=keep, dropped=dropped,
                clears_compaction=True, compaction_only=True,
            )
        turn_start = self.conversation.turn_start
        if turn_start is not None and 0 <= turn_start < len(messages) and messages[turn_start].role == "user":
            if turn_start == len(messages) - 1:
                return None  # the turn produced no response yet
            keep = list(messages[:turn_start])
            user_text = messages[turn_start].content
            dropped = list(messages[turn_start:])
            # If the restore point is before the compaction (the undone turn
            # is the one that compacted), the summary is gone from the
            # history — the compact snapshot is no longer reachable.
            clears = not any(_is_summary(m) for m in keep)
        else:
            index = last_real
            if index is None:
                return None
            keep = list(messages[:index])
            user_text = messages[index].content
            dropped = list(messages[index:])
            clears = False
        return RollbackPlan(
            user_text=user_text, revert_turns=_turns_in(dropped), keep=keep,
            dropped=dropped, clears_compaction=clears,
        )

    def rollback_plan_to(self, message: Message) -> RollbackPlan | None:
        """Plan a rollback to an arbitrary earlier user message (click-to-undo):
        drop that message and everything after it, restoring the workspace to
        the checkpoint taken when it was sent.

        Handles messages the last compaction summarized away too — the
        pre-compaction snapshot is restored first, then truncated at the
        clicked turn. Returns None when the message is not a user turn in this
        conversation. Even with nothing to drop (no later messages, no file
        edits) the plan is still produced so the message's text is put back
        in the input.
        """
        if message.role != "user":
            return None
        messages = self.conversation.messages
        index = next((i for i, m in enumerate(messages) if m is message), None)
        if index is not None:
            # Even when the message is the latest one (nothing after it to
            # drop) we still produce a plan: the text goes back into the
            # input, with empty drop/revert lists.
            keep = list(messages[:index])
            dropped = list(messages[index:])
            clears = not any(_is_summary(m) for m in keep)
            return RollbackPlan(
                user_text=message.content, revert_turns=_turns_in(dropped),
                keep=keep, dropped=dropped, clears_compaction=clears,
            )
        snapshot = self.conversation.compact_snapshot
        if snapshot is None:
            return None
        idx = next((i for i, m in enumerate(snapshot) if m is message), None)
        if idx is None:
            return None
        keep = list(snapshot[:idx])
        kept_ids = {id(m) for m in keep}
        # Only the live history's messages count for usage subtraction: the
        # snapshot's own spend was already folded when compaction removed it.
        dropped = [m for m in messages if id(m) not in kept_ids]
        # Revert every dropped turn — the ones the snapshot summarized away
        # (from the clicked turn onward) plus those that ran after compaction.
        revert = _turns_in([*snapshot[idx:], *dropped])
        return RollbackPlan(
            user_text=message.content, revert_turns=revert,
            keep=keep, dropped=dropped, clears_compaction=True,
        )

    def _subtract_usage(self, usage: Usage) -> None:
        """Remove one round's spend from the cumulative total (floored at 0)."""
        self.usage_total.input_tokens = max(
            0, self.usage_total.input_tokens - usage.input_tokens
        )
        self.usage_total.output_tokens = max(
            0, self.usage_total.output_tokens - usage.output_tokens
        )

    def apply_rollback(self, plan: RollbackPlan) -> None:
        """Truncate the conversation, subtract the dropped turns' usage from
        the running totals, and clear the snapshot. The file edits are
        reversed separately (see :meth:`revert_changes`) so a conflict can
        abort before the conversation changes."""
        self.conversation.messages = list(plan.keep)
        self.conversation.turn_start = None
        current_keys = {_message_key(m) for m in self.conversation.messages}
        for message in plan.dropped:
            if message.usage is not None:
                self._subtract_usage(message.usage)
        # Usage of messages compaction destroyed (their objects are gone, so
        # the loop above can't see them). A compaction undo restores those
        # very objects — their spend stays folded until the turn itself is
        # undone later.
        if not plan.compaction_only:
            self._subtract_usage(self._absorbed_usage)
            if self._absorbed_usage.input_tokens or self._absorbed_usage.output_tokens:
                # Consumed: a later compaction undo restores those turns, so
                # it must re-fold their spend (below) to keep the totals and
                # a later per-turn undo from double-counting.
                self._absorbed_spent = True
        elif self._absorbed_spent:
            # The absorbed subtraction already removed the restored turns'
            # spend; fold it back so the totals match the restored history.
            for message in plan.keep:
                if (_message_key(message) not in current_keys
                        and message.usage is not None):
                    self._fold_usage(message.usage)
        self._absorbed_usage = Usage(input_tokens=0, output_tokens=0)
        self._absorbed_spent = False
        self.last_compaction = None
        if plan.clears_compaction:
            # The restore point is before the compaction: its summary is no
            # longer in the history, so the pre-compaction snapshot (which
            # would only offer a stale second undo of the same turn) is
            # dropped too.
            self.conversation.compact_snapshot = None

    async def finish_turn(self) -> None:
        """Record the workspace state after the turn's edits as ``after_tree``
        on the turn's user message, so /undo can reverse exactly this turn's
        diff later. Called by the app once a turn's rounds and tools have
        finished (including on interruption). No-op without git tracking, when
        no turn is open, or when the snapshot fails."""
        info = self.git_info
        if info is None:
            return
        turn = next(
            (m for m in reversed(self.conversation.messages)
             if m.role == "user" and m.checkpoint_sha is not None),
            None,
        )
        if turn is None:
            return
        try:
            turn.after_tree = await git_checkpoint(info)
        except GitWorktreeError:
            return

    def turn_checkpoints(self) -> list[TurnCheckpoint]:
        """The reversible turns since the baseline, oldest first.

        Includes turns the last compaction summarized away (kept in
        ``compact_snapshot``), so /diff attribution and the resume drift check
        still see their edits."""
        pre = self.conversation.compact_snapshot or []
        return _turns_in([*pre, *self.conversation.messages])

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
        """Summarize the conversation and replace older messages with it.

        Compaction runs at a turn boundary (auto-compact), so there are no
        pending tool calls to break pairing. Keeps the system prompt;
        everything after it becomes a single handoff summary user message
        (task, what's known so far, what to do next). Returns the summary
        text; the before/after token counts are left in
        :attr:`last_compaction` for the compaction notice.
        """
        messages = self.conversation.messages
        compactable = messages[1:]
        if not compactable:
            self.last_compaction = None
            return "nothing to compact yet"

        before_tokens = sum(self._message_tokens(m) for m in compactable)
        transcript = "\n\n".join(self._render_for_summary(m) for m in compactable)
        summary, usage = await self._summarize(transcript)
        summary_message = Message(
            role="user",
            content=f"{SUMMARY_PREFIX}\n\n{summary}",
            usage=usage,
        )
        # The summarization round's own spend is part of the session total.
        if usage is not None:
            self._fold_usage(usage)
        # Snapshot for /undo: the full pre-compaction history. Taken here (not
        # on entry) so an empty compaction never clobbers an older snapshot.
        self.conversation.compact_snapshot = list(messages)
        self.conversation.messages = [messages[0], summary_message]
        self.last_compaction = (before_tokens, self._message_tokens(summary_message))
        # Usage of removed messages that belongs to the current turn (at or
        # after turn_start): /undo must subtract it even though the message
        # objects are gone. Pre-turn usage stays folded, matching rollback
        # semantics; send() resets the tracker each turn.
        self._absorbed_spent = False
        turn_start = self.conversation.turn_start
        if turn_start is not None:
            for index in range(1, len(messages)):
                removed = messages[index]
                if index >= turn_start and removed.usage is not None:
                    self._absorbed_usage.input_tokens += removed.usage.input_tokens
                    self._absorbed_usage.output_tokens += removed.usage.output_tokens
        return summary

    @staticmethod
    def _message_tokens(message: Message) -> int:
        """Rough context contribution of one message, in tokens.

        Assistant messages carry their round's real usage; its output tokens
        are what the message added to the context (input tokens would count
        the whole prompt again). Everything else falls back to the
        chars-per-token heuristic.
        """
        if message.usage is not None:
            return message.usage.output_tokens
        return len(message.content or "") // _CHARS_PER_TOKEN

    @staticmethod
    def _render_for_summary(message: Message) -> str:
        """One message as plain text for the summarizer prompt."""
        role = message.role
        parts = [f"{role}: {message.content}"] if message.content else [f"{role}:"]
        for call in message.tool_calls or []:
            parts.append(f"  {role} called {call.name}({call.arguments_json})")
        if role == "tool" and message.tool_call_id:
            parts.append(f"  (result for call {message.tool_call_id})")
        return "\n".join(parts)

    async def _summarize(self, transcript: str) -> tuple[str, Usage | None]:
        """One-off handoff round through the same client, no tools.

        Returns the handoff text and the round's usage, so its cost can be
        folded into the session totals instead of being silently dropped."""
        prompt = (
            "You are handing this conversation off to a fresh context. Write "
            "the handoff the next assistant instance needs to continue the "
            "work seamlessly. Cover, in this order:\n"
            "1. The task: what the user is trying to accomplish, in their "
            "terms.\n"
            "2. What's known so far: key decisions, file paths, relevant "
            "commands and their results, constraints discovered.\n"
            "3. What's next: the current state of any work in progress and "
            "the immediate next steps.\n"
            "Be concise — a few short paragraphs at most. Reply with the "
            "handoff only.\n\n"
            "--- conversation ---\n"
            f"{transcript}"
        )
        parts: list[str] = []
        usage: Usage | None = None
        async for event in self._client.stream([{"role": "user", "content": prompt}]):
            if isinstance(event, ContentDelta):
                parts.append(event.text)
            elif isinstance(event, StreamError):
                raise RuntimeError(event.message)
            elif isinstance(event, TurnComplete):
                usage = event.usage
                break
        summary = "".join(parts).strip()
        if not summary:
            raise RuntimeError("summarization returned no content")
        return summary, usage

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

        tools_schema = self.tools.schema() if self.tools is not None else None
        async for event in self._client.stream(
            self.conversation.to_wire(keep_reasoning=self.profile.keep_reasoning), tools=tools_schema
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
                done_calls.append(ToolCall(event.call_id, event.name, event.arguments_json))
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
                        reasoning_seconds=reasoning_seconds,
                        tool_calls=done_calls or None,
                        usage=usage,
                    )
                )
                event = TurnComplete(
                    usage=usage,
                    reasoning_seconds=reasoning_seconds,
                    has_tool_calls=bool(done_calls),
                )
            yield event

    def _fold_usage(self, usage: Usage) -> None:
        """Add one round's usage to the cumulative session total."""
        self.usage_total.input_tokens += usage.input_tokens
        self.usage_total.output_tokens += usage.output_tokens

    def _estimate_usage(self, content_parts: list[str]) -> Usage:
        prompt_chars = sum(len(message.content or "") for message in self.conversation.messages)
        return Usage(
            input_tokens=prompt_chars // _CHARS_PER_TOKEN,
            output_tokens=sum(len(part) for part in content_parts) // _CHARS_PER_TOKEN,
            approximate=True,
        )
