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
from typing import Any, Callable, Protocol

from .client import ChatClient
from .config import ProviderProfile
from .conversation import SUMMARY_PREFIX, Conversation, Message, ToolCall, is_summary
from .events import (
    ContentDelta,
    ReasoningDelta,
    StreamError,
    StreamEvent,
    ToolCallArgumentsDone,
    TurnComplete,
    Usage,
)
from .gitwork import GitInfo, GitWorktreeError, RevertConflict, TurnCheckpoint
from .images import ImageAttachment
from .util import mono_to_iso as _mono_to_iso, utcnow_iso as _utcnow_iso
from .gitwork import snapshot_state as git_snapshot_state
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


def _last_user_turn_index(messages: list[Message]) -> int | None:
    """Index of the last real user turn (summary placeholders don't count)
    that has a response after it, or None."""
    index = max(
        (i for i, m in enumerate(messages) if m.role == "user" and not is_summary(m)
         and i < len(messages) - 1),
        default=None,
    )
    return index


def _summary_index(messages: list[Message]) -> int | None:
    """Index of the first compaction summary in the list, or None."""
    return next((i for i, m in enumerate(messages) if is_summary(m)), None)


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
    """What /undo or /retry needs to do, computed by :meth:`rollback_plan`.

    Usage totals are monotonic (they count everything ever spent, including
    undone turns), so a plan carries no usage bookkeeping."""

    user_text: str  # the turn's user message (input box / resend)
    revert_turns: list[TurnCheckpoint]  # file diffs to reverse (oldest first)
    keep: list[Message]  # what the conversation becomes
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
        # The user message that opened the turn currently being generated, or
        # None between turns. Explicit state (not inferred from after_tree)
        # because a mid-turn /accept //reject rewrites every checkpoint, and a
        # mid-turn compaction moves this message into compact_snapshot — both
        # would hide the still-open turn from a message-list scan.
        self._open_turn: Message | None = None
        # Optional hook, called with each user message as it lands in the
        # history (send and inject_user_message). The session uses it to emit
        # UserMessageAdded — there is no local echo in the client.
        self.on_user_message: Callable[[Message], None] | None = None

    async def send(self, user_text: str, images: list[ImageAttachment] | None = None) -> AsyncIterator[StreamEvent]:
        """Send a user message, streaming the first round.

        ``images`` are base64 attachments carried on the user message and
        resent with the history on every round."""
        message = Message(
            role="user",
            content=user_text,
            images=images or None,
            checkpoint_sha=await self._take_checkpoint(),
        )
        self.conversation.add(message)
        # The turn is open from here until finish_turn records its after tree.
        self._open_turn = message
        # Snapshot marker for /undo //retry: the index of this turn's user
        # message. The turn snapshot is the messages[:turn_start + 1] prefix,
        # so restoring it rolls back the whole turn — even one that compacted
        # the history mid-flight.
        self.conversation.turn_start = len(self.conversation.messages) - 1
        if self.on_user_message is not None:
            self.on_user_message(self.conversation.messages[-1])
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
        content = result.output if result.ok else f"error: {result.error}"
        self.conversation.add(
            Message(
                role="tool",
                tool_call_id=call_id,
                content=content,
                header=result.header or None,
                start_time=start_time,
                end_time=end_time,
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
        message = Message(
            role="user", content=text, images=images or None, send_time=send_time,
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
            # history (their spend stays counted — totals are monotonic);
            # the compaction turn's own messages come back.
            keep = list(compact_snapshot)
            return RollbackPlan(
                user_text="", revert_turns=[], keep=keep,
                clears_compaction=True, compaction_only=True,
            )
        turn_start = self.conversation.turn_start
        if turn_start is not None and 0 <= turn_start < len(messages) and messages[turn_start].role == "user":
            if turn_start == len(messages) - 1:
                return None  # the turn produced no response yet
            return self.rollback_plan_to(messages[turn_start])
        if last_real is None:
            return None
        return self.rollback_plan_to(messages[last_real])

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
            # input.
            keep = list(messages[:index])
            clears = not any(is_summary(m) for m in keep)
            return RollbackPlan(
                user_text=message.content, revert_turns=_turns_in(messages[index:]),
                keep=keep, clears_compaction=clears,
            )
        snapshot = self.conversation.compact_snapshot
        if snapshot is None:
            return None
        idx = next((i for i, m in enumerate(snapshot) if m is message), None)
        if idx is None:
            return None
        keep = list(snapshot[:idx])
        # Revert every dropped turn — the ones the snapshot summarized away
        # (from the clicked turn onward) plus those that ran after compaction.
        revert = _turns_in([*snapshot[idx:], *messages])
        return RollbackPlan(
            user_text=message.content, revert_turns=revert,
            keep=keep, clears_compaction=True,
        )

    def apply_rollback(self, plan: RollbackPlan) -> None:
        """Truncate the conversation and clear the snapshot. The file edits
        are reversed separately (see :meth:`revert_changes`) so a conflict
        can abort before the conversation changes. Usage totals are
        monotonic and untouched."""
        self.conversation.messages = list(plan.keep)
        self.conversation.turn_start = None
        self.last_compaction = None
        if plan.clears_compaction:
            # The restore point is before the compaction: its summary is no
            # longer in the history, so the pre-compaction snapshot (which
            # would only offer a stale second undo of the same turn) is
            # dropped too.
            self.conversation.compact_snapshot = None

    @property
    def open_turn(self) -> Message | None:
        """The user message of the turn being generated, or None.

        /accept and /reject rewrite every checkpoint mid-turn; the open turn
        is identified by identity, not by scanning the message list."""
        return self._open_turn

    async def finish_turn(self) -> None:
        """Record the workspace state after the turn's edits as ``after_tree``
        on the turn's user message, so /undo can reverse exactly this turn's
        diff later. Called by the app once a turn's rounds and tools have
        finished (including on interruption). No-op without git tracking, when
        no turn is open, or when the snapshot fails.

        Uses the explicitly-open turn, not ``last_turn_message``: a turn whose
        checkpoint failed must not restamp the previous (closed) turn with the
        new after tree."""
        turn = self._open_turn
        self._open_turn = None
        info = self.git_info
        if info is None or turn is None or not turn.checkpoint_sha:
            return
        try:
            turn.after_tree = await git_snapshot_state(info)
        except GitWorktreeError:
            return

    def open_turn_before(self) -> str | None:
        """The still-open turn's before tree, or None.

        /diff attribution folds this into the agent-touched set so the
        in-flight turn's edits are not shown as drift while streaming — even
        after a mid-turn /accept //reject rewrote the checkpoint, or a
        mid-turn compaction moved the message into ``compact_snapshot``.
        """
        turn = self._open_turn
        if turn is None or not turn.checkpoint_sha:
            return None
        return turn.checkpoint_sha

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

        Auto-compact runs at a round boundary (every tool result recorded,
        every assistant message stamped), so there are no pending tool calls
        to break pairing. Keeps the system prompt;
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
        # The turn snapshot prefix must stay valid mid-turn: anchor it on the
        # summary message, so /undo //retry never slice past the compacted
        # history with the pre-compaction index.
        self.conversation.turn_start = len(self.conversation.messages) - 1
        self.last_compaction = (before_tokens, self._message_tokens(summary_message))
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
        started_at = _utcnow_iso()

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
                reasoning_end = _mono_to_iso(first_content_at) or _utcnow_iso()
                self.conversation.add(
                    Message(
                        role="assistant",
                        content="".join(partial.content_parts),
                        reasoning=reasoning,
                        tool_calls=done_calls or None,
                        usage=usage,
                        # Wall-clock stamps taken at the moment the round
                        # actually streamed (never reconstructed later).
                        start_time=started_at,
                        end_time=_utcnow_iso(),
                        reasoning_start=_mono_to_iso(first_reasoning_at),
                        reasoning_end=reasoning_end,
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
        self.last_usage = usage

    def _estimate_usage(self, content_parts: list[str]) -> Usage:
        prompt_chars = sum(len(message.content or "") for message in self.conversation.messages)
        return Usage(
            input_tokens=prompt_chars // _CHARS_PER_TOKEN,
            output_tokens=sum(len(part) for part in content_parts) // _CHARS_PER_TOKEN,
            approximate=True,
        )
