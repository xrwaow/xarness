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
from .conversation import SUMMARY_PREFIX, Conversation, Message
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
from .gitwork import GitInfo, GitWorktreeError
from .gitwork import checkpoint as git_checkpoint
from .gitwork import revert_to_tree
from .tools import ToolRegistry, ToolResult

# Rough chars-per-token for the fallback estimate when the provider sends no
# usage data. Deliberately conservative; results are labeled approximate.
_CHARS_PER_TOKEN = 4


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
    checkpoint_sha: str | None  # git state to restore; None = cannot revert
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
        # Set by compact() when it summarizes away the compact tool call
        # itself: the next record_tool_result call is that call's own result
        # and must not be appended (the history ends on the summary).
        self._drop_next_tool_result = False
        # Usage folded into the totals for messages that compaction later
        # removed during the current turn. /undo must subtract it too, since
        # the message objects that carry it no longer exist.
        self._absorbed_usage = Usage(input_tokens=0, output_tokens=0)
        # (before, after) token counts of the most recent compaction, for the
        # compact tool's result line.
        self.last_compaction: tuple[int, int] | None = None

    async def send(self, user_text: str) -> AsyncIterator[StreamEvent]:
        """Send a user message, streaming the first round."""
        self.conversation.add(
            Message(role="user", content=user_text, checkpoint_sha=await self._take_checkpoint())
        )
        # Snapshot for /undo //retry: the conversation as it stood when this
        # turn began (user message included). Restoring it rolls back the
        # whole turn — even one that compacted the history mid-flight.
        self.conversation.undo_snapshot = list(self.conversation.messages)
        self._drop_next_tool_result = False
        self._absorbed_usage = Usage(input_tokens=0, output_tokens=0)
        async for event in self._stream_round():
            yield event

    async def continue_after_tools(self) -> AsyncIterator[StreamEvent]:
        """Stream the next round, after tool results have been recorded."""
        async for event in self._stream_round():
            yield event

    def record_tool_result(self, call_id: str, result: ToolResult) -> None:
        """Append a ``tool`` role message with the outcome of one call."""
        if self._drop_next_tool_result:
            # compact() summarized away its own tool call; recording its
            # result would leave an orphaned tool message on the wire.
            self._drop_next_tool_result = False
            return
        content = result.output if result.ok else f"error: {result.error}"
        self.conversation.add(
            Message(
                role="tool",
                tool_call_id=call_id,
                content=content,
                header=result.header or None,
            )
        )

    def inject_user_message(self, text: str) -> None:
        """Add a user message mid-turn (steering).

        Called at a round boundary — after tool results are recorded, before
        the next round streams — so a message typed while the agent works is
        seen by the very next model call instead of waiting for the turn to
        end. No checkpoint of its own: it belongs to the turn that is already
        running, whose snapshot /undo restores anyway.
        """
        self.conversation.add(Message(role="user", content=text))

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
        snapshot = self.conversation.undo_snapshot
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
                user_text="", checkpoint_sha=None, keep=keep, dropped=dropped,
                clears_compaction=True, compaction_only=True,
            )
        if snapshot and snapshot[-1].role == "user":
            snapshot_ids = {id(m) for m in snapshot}
            if not any(id(m) not in snapshot_ids for m in messages):
                return None  # the turn produced no response yet
            keep = list(snapshot[:-1])
            user_text = snapshot[-1].content
            sha = snapshot[-1].checkpoint_sha
            kept_ids = {id(m) for m in keep}
            dropped = [m for m in messages if id(m) not in kept_ids]
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
            sha = messages[index].checkpoint_sha
            dropped = list(messages[index:])
            clears = False
        return RollbackPlan(
            user_text=user_text, checkpoint_sha=sha, keep=keep, dropped=dropped,
            clears_compaction=clears,
        )

    def _subtract_usage(self, usage: Usage) -> None:
        """Remove one round's spend from the cumulative total (floored at 0)."""
        self.usage_total.input_tokens = max(
            0, self.usage_total.input_tokens - usage.input_tokens
        )
        self.usage_total.output_tokens = max(
            0, self.usage_total.output_tokens - usage.output_tokens
        )

    def apply_rollback(self, plan: RollbackPlan) -> str | None:
        """Truncate the conversation, subtract the dropped turns' usage from
        the running totals, and clear the snapshot. Returns the checkpoint
        sha so the caller can revert the file edits."""
        self.conversation.messages = list(plan.keep)
        self.conversation.undo_snapshot = None
        for message in plan.dropped:
            if message.usage is not None:
                self._subtract_usage(message.usage)
        # Usage of messages compaction destroyed (their objects are gone, so
        # the loop above can't see them). A compaction undo restores those
        # very objects — their spend stays folded until the turn itself is
        # undone later.
        if not plan.compaction_only:
            self._subtract_usage(self._absorbed_usage)
        self._absorbed_usage = Usage(input_tokens=0, output_tokens=0)
        self.last_compaction = None
        if plan.clears_compaction:
            # The restore point is before the compaction: its summary is no
            # longer in the history, so the pre-compaction snapshot (which
            # would only offer a stale second undo of the same turn) is
            # dropped too.
            self.conversation.compact_snapshot = None
        return plan.checkpoint_sha

    async def revert_changes(self, sha: str | None) -> bool:
        """Restore the workspace to a checkpoint tree. False when there is no
        git tracking / no sha / the revert failed — the caller should tell
        the user file edits could not be reverted."""
        info = self.git_info
        if info is None or sha is None:
            return False
        try:
            await revert_to_tree(info, sha)
        except (GitWorktreeError, OSError):
            return False
        return True

    # ------------------------------------------------------------------
    # compaction

    async def compact(self) -> str:
        """Summarize the conversation and replace older messages with it.

        Keeps the system prompt; everything after it becomes a single summary
        user message. When compaction is requested mid-turn, the compact tool
        call itself is summarized away too and its result is never recorded —
        the model never sees the compaction mechanics, the history just ends
        on the summary. (If the same round requested other tools as well,
        the assistant message is kept so those results stay paired on the
        wire.) Returns the summary text; the before/after token counts are
        left in :attr:`last_compaction` for the compact tool's result line.
        """
        messages = self.conversation.messages
        # Mid-turn compaction: if the trailing assistant round requested only
        # this compact call, summarize it away and suppress its tool result.
        # With other pending calls in the same round, keep the message so
        # their results stay paired on the wire.
        last = messages[-1] if messages else None
        pending_calls = last is not None and last.role == "assistant" and bool(last.tool_calls)
        drop_call_round = pending_calls and len(last.tool_calls) == 1
        keep_from = len(messages) - (1 if pending_calls and not drop_call_round else 0)
        compactable = messages[1:keep_from]
        if not compactable:
            self.last_compaction = None
            return "nothing to compact yet"
        self._drop_next_tool_result = drop_call_round

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
        self.conversation.messages = [messages[0], summary_message, *messages[keep_from:]]
        self.last_compaction = (before_tokens, self._message_tokens(summary_message))
        # Usage of removed messages that belongs to the current turn (not in
        # the undo snapshot): /undo must subtract it even though the message
        # objects are gone. Pre-turn usage stays folded, matching rollback
        # semantics; send() resets the tracker each turn.
        snapshot = self.conversation.undo_snapshot
        if snapshot:
            snapshot_ids = {id(m) for m in snapshot}
            for removed in messages[1:keep_from]:
                if removed.usage is not None and id(removed) not in snapshot_ids:
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
            fn = call.get("function", {})
            parts.append(f"  {role} called {fn.get('name', '?')}({fn.get('arguments', '')})")
        if role == "tool" and message.tool_call_id:
            parts.append(f"  (result for call {message.tool_call_id})")
        return "\n".join(parts)

    async def _summarize(self, transcript: str) -> tuple[str, Usage | None]:
        """One-off summarization round through the same client, no tools.

        Returns the summary text and the round's usage, so its cost can be
        folded into the session totals instead of being silently dropped."""
        prompt = (
            "Summarize the following conversation between a user and a coding "
            "assistant. Preserve the task the user wants, key decisions, file "
            "paths, and the current state of any work in progress. Be concise — "
            "a few short paragraphs at most. Reply with the summary only.\n\n"
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
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        pending: dict[str, dict[str, str]] = {}  # call_id -> {name, arguments}
        call_order: list[str] = []
        first_reasoning_at: float | None = None
        first_content_at: float | None = None

        tools_schema = self.tools.schema() if self.tools is not None else None
        async for event in self._client.stream(self.conversation.to_wire(), tools=tools_schema):
            if isinstance(event, ReasoningDelta):
                if first_reasoning_at is None:
                    first_reasoning_at = time.monotonic()
                reasoning_parts.append(event.text)
            elif isinstance(event, ContentDelta):
                if first_content_at is None:
                    first_content_at = time.monotonic()
                content_parts.append(event.text)
            elif isinstance(event, ToolCallStarted):
                pending[event.call_id] = {"name": event.name, "arguments": ""}
                call_order.append(event.call_id)
            elif isinstance(event, ToolCallArgumentsDelta):
                if event.call_id in pending:
                    pending[event.call_id]["arguments"] += event.text
            elif isinstance(event, ToolCallArgumentsDone):
                # Authoritative once the stream is done parsing.
                pending[event.call_id] = {"name": event.name, "arguments": event.arguments_json}
            elif isinstance(event, TurnComplete):
                reasoning = "".join(reasoning_parts) or None
                reasoning_seconds: float | None = None
                if first_reasoning_at is not None:
                    end = first_content_at or time.monotonic()
                    reasoning_seconds = end - first_reasoning_at
                tool_calls_wire = [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": pending[call_id]["name"],
                            "arguments": pending[call_id]["arguments"],
                        },
                    }
                    for call_id in call_order
                ]
                usage = event.usage or self._estimate_usage(content_parts)
                self._fold_usage(usage)
                self.conversation.add(
                    Message(
                        role="assistant",
                        content="".join(content_parts),
                        reasoning=reasoning,
                        reasoning_seconds=reasoning_seconds,
                        tool_calls=tool_calls_wire or None,
                        usage=usage,
                    )
                )
                event = TurnComplete(
                    usage=usage,
                    reasoning_seconds=reasoning_seconds,
                    has_tool_calls=bool(call_order),
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
