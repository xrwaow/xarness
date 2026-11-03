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

It owns no workspace state: git checkpoints, rollback and diff bookkeeping
belong to the session (which is the thing that has a workspace and a
:class:`~xarness.gitwork.GitInfo`).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .client import ChatClient, StreamClient
from .compaction import CHARS_PER_TOKEN
from .compaction import compact as run_compaction
from .config import ProviderProfile
from .conversation import (
    AssistantMessage,
    Conversation,
    Message,
    Timing,
    ToolCall,
    ToolMessage,
    Usage,
    user_message,
)
from .images import ImageRef
from .protocol import (
    ContentDelta,
    PartialRound,
    PartialToolCall,
    ReasoningDelta,
    RoundComplete,
    StreamEvent,
    ToolCallArgumentsDelta,
    ToolCallArgumentsDone,
    ToolCallStarted,
)
from .util import utcnow_iso as _utcnow_iso

if TYPE_CHECKING:
    from .compaction import CompactionResult
    from .tools import ToolRegistry


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
        # Cumulative session spend, folded together from every round's
        # usage — including compaction's own summarization round and
        # everything an /undo later removes. Monotonic: it counts everything
        # ever spent this session.
        self.usage_total = Usage(input_tokens=0, output_tokens=0)
        # Usage of the most recent round, kept for the status payload
        # (instead of re-scanning the message list on every status emit).
        self.last_usage = Usage(input_tokens=0, output_tokens=0)
        # The live round currently streaming — ONE accumulator for its
        # reasoning, answer text and tool calls, shared with the session
        # (whose snapshot sends it) and with save_interrupted_round(), so
        # attaching, cancelling and persisting all read the same state.
        self.partial_round = PartialRound()
        # The system prompt is derived at request time (never persisted).
        self.system_prompt: str | None = None
        # Optional hook, called with each user message as it lands in the
        # history (send and inject_user_message). The session uses it to
        # emit MessageAdded — there is no local echo in the client.
        self.on_user_message: Callable[[Message], None] | None = None
        # Image blob storage, supplied by the session (content-addressed).
        # Every controller that can hold image messages has one; a plain
        # text-only controller (tests) never looks at it.
        self.image_store: Any = None

    def _image_url(self, ref: ImageRef) -> str | None:
        """A data URL for one stored image (from the blob store's cache);
        None when the blob is gone."""
        return self.image_store.data_url(ref)

    async def begin_turn(
        self, text: str, images: list[ImageRef] | None,
        before_tree: str | None,
    ) -> AsyncIterator[StreamEvent]:
        """Add the user message, open the turn's checkpoint, and stream the
        first round.

        The checkpoint is ALWAYS attached — with ``before_tree=None`` when
        there is no tracking or the snapshot failed — so every turn has an
        entry and a rollback never has to guess its scope."""
        message = user_message(text, images, self.image_store)
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
        partial = self.partial_round
        content = partial.content
        reasoning = partial.reasoning or None
        self.reset_partial()
        if not content and not reasoning:
            return False
        self.conversation.add(AssistantMessage(
            content=content, reasoning=reasoning, interrupted=True,
        ))
        return True

    def reset_partial(self) -> None:
        """Drop the live round state (turn start, turn end, round boundary)."""
        self.partial_round = PartialRound()

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
    # compaction

    async def compact(self) -> "CompactionResult | None":
        """Summarize the conversation and shrink the visible slice to the
        summary message (old messages stay in the append-only list and come
        back if the summary is popped). Returns the
        :class:`~xarness.compaction.CompactionResult`, or None when there
        was nothing to do."""
        result = await run_compaction(
            self.conversation, self._client,
            system=self.system_prompt,
            keep_reasoning=self.profile.keep_reasoning,
            image_loader=self._image_url,
        )
        if result is None:
            return None
        if result.usage is not None:
            self._fold_usage(result.usage)
        return result

    # ------------------------------------------------------------------
    # streaming

    async def _stream_round(self) -> AsyncIterator[StreamEvent]:
        partial = self.partial_round = PartialRound()
        done_calls: list[ToolCall] = []
        first_reasoning_at: float | None = None
        # The cot block ends where the answer or the first tool call starts —
        # tool-argument streaming counts as call time, not thinking time.
        first_answer_at: float | None = None
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
                partial.reasoning += event.text
            elif isinstance(event, (ContentDelta, ToolCallStarted)):
                if first_answer_at is None:
                    first_answer_at = time.monotonic()
                if isinstance(event, ContentDelta):
                    partial.content += event.text
                else:
                    partial.tool_calls.append(PartialToolCall(
                        call_id=event.call_id, name=event.name,
                    ))
            elif isinstance(event, ToolCallArgumentsDelta):
                if (call := partial.find_call(event.call_id)) is not None:
                    call.arguments += event.text
            elif isinstance(event, ToolCallArgumentsDone):
                if (call := partial.find_call(event.call_id)) is not None:
                    call.arguments = event.arguments_json
                # Authoritative once the stream is done parsing; the client
                # has already assembled the full arguments for this call.
                done_calls.append(ToolCall(
                    call_id=event.call_id, name=event.name,
                    arguments_json=event.arguments_json,
                ))
            elif isinstance(event, RoundComplete):
                reasoning = partial.reasoning or None
                content = partial.content
                # The message is in the history now: only the tool calls
                # still executing remain live.
                partial.reasoning = ""
                partial.content = ""
                reasoning_seconds: float | None = None
                if first_reasoning_at is not None:
                    end = first_answer_at or time.monotonic()
                    reasoning_seconds = end - first_reasoning_at
                usage = event.usage or self._estimate_usage(content)
                self._fold_usage(usage)
                self.conversation.add(AssistantMessage(
                    content=content,
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

    def _estimate_usage(self, content: str) -> Usage:
        prompt_chars = sum(
            len(message.content or "")
            for message in self.conversation.visible_messages()
        )
        return Usage(
            input_tokens=prompt_chars // CHARS_PER_TOKEN,
            output_tokens=len(content) // CHARS_PER_TOKEN,
            approximate=True,
        )
