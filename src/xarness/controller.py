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

``send()`` also takes a per-turn checkpoint: untracked files are re-synced
from the user's workspace, a git checkpoint of the worktree is recorded on
the user message, and the conversation is snapshotted — the machinery behind
/undo and /retry.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .client import ChatClient
from .config import ProviderProfile
from .conversation import Conversation, Message
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
from .gitwork import revert_to_checkpoint, sync_untracked_files
from .tools import ToolRegistry, ToolResult

# Rough chars-per-token for the fallback estimate when the provider sends no
# usage data. Deliberately conservative; results are labeled approximate.
_CHARS_PER_TOKEN = 4


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
        # Set by the app when the session has a git-isolated worktree; drives
        # per-turn untracked re-sync and checkpoints.
        self.git_info: GitInfo | None = None
        # Cumulative session spend (input + output), folded together from
        # every round's usage — including compaction's own summarization
        # round. /undo subtracts the usage of the messages it drops.
        self.usage_total = Usage(input_tokens=0, output_tokens=0)
        # (before, after) token counts of the most recent compaction, for the
        # compact tool's result line.
        self.last_compaction: tuple[int, int] | None = None

    async def send(self, user_text: str) -> AsyncIterator[StreamEvent]:
        """Send a user message, streaming the first round."""
        await self._sync_untracked()
        self.conversation.add(
            Message(role="user", content=user_text, checkpoint_sha=await self._take_checkpoint())
        )
        # Snapshot for /undo //retry: the conversation as it stood when this
        # turn began (user message included). Restoring it rolls back the
        # whole turn — even one that compacted the history mid-flight.
        self.conversation.undo_snapshot = list(self.conversation.messages)
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
    # per-turn git checkpoint + untracked re-sync

    async def _sync_untracked(self) -> None:
        """Copy untracked files the user added to their workspace since the
        worktree was created, so the agent sees them this turn. Best-effort."""
        info = self.git_info
        if info is None:
            return
        try:
            await sync_untracked_files(info)
        except (GitWorktreeError, OSError):
            pass  # never block a turn on housekeeping

    async def _take_checkpoint(self) -> str | None:
        """Commit the worktree's current state onto the agent branch and
        return the sha (or the current HEAD when already clean). None when
        there is no git isolation or the checkpoint failed."""
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

        With a snapshot (the normal case) the conversation is restored to
        exactly what it was when the turn began — which also un-does a
        mid-turn compaction. Without one (resumed session) it falls back to
        truncating at the last user message. Either way the turn's user
        message is dropped: /undo stops there, /retry re-sends its text
        (which takes a fresh checkpoint).
        """
        messages = self.conversation.messages
        snapshot = self.conversation.undo_snapshot
        if snapshot and snapshot[-1].role == "user":
            snapshot_ids = {id(m) for m in snapshot}
            if not any(id(m) not in snapshot_ids for m in messages):
                return None  # the turn produced no response yet
            keep = list(snapshot[:-1])
            user_text = snapshot[-1].content
            sha = snapshot[-1].checkpoint_sha
            kept_ids = {id(m) for m in keep}
            dropped = [m for m in messages if id(m) not in kept_ids]
        else:
            index = max(
                (i for i, m in enumerate(messages) if m.role == "user" and i < len(messages) - 1),
                default=None,
            )
            if index is None:
                return None
            keep = messages[:index]
            user_text = messages[index].content
            sha = messages[index].checkpoint_sha
            dropped = messages[index:]
        return RollbackPlan(
            user_text=user_text, checkpoint_sha=sha, keep=keep, dropped=dropped
        )

    def apply_rollback(self, plan: RollbackPlan) -> str | None:
        """Truncate the conversation, subtract the dropped turns' usage from
        the running totals, and clear the snapshot. Returns the checkpoint
        sha so the caller can revert the worktree."""
        self.conversation.messages = list(plan.keep)
        self.conversation.undo_snapshot = None
        for message in plan.dropped:
            if message.usage is not None:
                self.usage_total.input_tokens = max(
                    0, self.usage_total.input_tokens - message.usage.input_tokens
                )
                self.usage_total.output_tokens = max(
                    0, self.usage_total.output_tokens - message.usage.output_tokens
                )
        self.last_compaction = None
        return plan.checkpoint_sha

    async def revert_worktree(self, sha: str | None) -> bool:
        """Restore the worktree to a checkpoint sha. False when there is no
        git isolation / no sha / the revert failed — the caller should tell
        the user file edits could not be reverted."""
        info = self.git_info
        if info is None or sha is None:
            return False
        try:
            await revert_to_checkpoint(info, sha)
        except (GitWorktreeError, OSError):
            return False
        return True

    # ------------------------------------------------------------------
    # compaction

    async def compact(self) -> str:
        """Summarize the conversation and replace older messages with it.

        Keeps the system prompt and the trailing assistant tool-call message
        (the tool result that follows a mid-turn compaction must pair with
        it on the wire); everything in between becomes a single summary user
        message. Returns the summary text; the before/after token counts are
        left in :attr:`last_compaction` for the compact tool's result line.
        """
        messages = self.conversation.messages
        keep_from = len(messages)
        if messages and messages[-1].role == "assistant" and messages[-1].tool_calls:
            keep_from -= 1  # preserve the round that requested this compaction
        compactable = messages[1:keep_from]
        if len(messages) < 2 or not compactable:
            self.last_compaction = None
            return "nothing to compact yet"

        before_tokens = sum(self._message_tokens(m) for m in compactable)
        transcript = "\n\n".join(self._render_for_summary(m) for m in compactable)
        summary, usage = await self._summarize(transcript)
        summary_message = Message(
            role="user",
            content=f"[earlier conversation, summarized]\n\n{summary}",
            usage=usage,
        )
        # The summarization round's own spend is part of the session total.
        if usage is not None:
            self._fold_usage(usage)
        self.conversation.messages = [messages[0], summary_message, *messages[keep_from:]]
        after_tokens = self._message_tokens(summary_message)
        self.last_compaction = (before_tokens, after_tokens)
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
