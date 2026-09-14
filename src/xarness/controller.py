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
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
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

    async def send(self, user_text: str) -> AsyncIterator[StreamEvent]:
        """Send a user message, streaming the first round."""
        self.conversation.add(Message(role="user", content=user_text))
        async for event in self._stream_round():
            yield event

    async def continue_after_tools(self) -> AsyncIterator[StreamEvent]:
        """Stream the next round, after tool results have been recorded."""
        async for event in self._stream_round():
            yield event

    def record_tool_result(self, call_id: str, result: ToolResult) -> None:
        """Append a ``tool`` role message with the outcome of one call."""
        content = result.output if result.ok else f"error: {result.error}"
        self.conversation.add(Message(role="tool", tool_call_id=call_id, content=content))

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

    async def compact(self) -> str:
        """Summarize the conversation and replace older messages with it.

        Keeps the system prompt and the trailing assistant tool-call message
        (the tool result that follows a mid-turn compaction must pair with
        it on the wire); everything in between becomes a single summary user
        message. Returns the summary text.
        """
        messages = self.conversation.messages
        keep_from = len(messages)
        if messages and messages[-1].role == "assistant" and messages[-1].tool_calls:
            keep_from -= 1  # preserve the round that requested this compaction
        compactable = messages[1:keep_from]
        if len(messages) < 2 or not compactable:
            return "nothing to compact yet"

        transcript = "\n\n".join(self._render_for_summary(m) for m in compactable)
        summary = await self._summarize(transcript)
        summary_message = Message(
            role="user",
            content=f"[earlier conversation, summarized]\n\n{summary}",
        )
        self.conversation.messages = [messages[0], summary_message, *messages[keep_from:]]
        return summary

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

    async def _summarize(self, transcript: str) -> str:
        """One-off summarization round through the same client, no tools."""
        prompt = (
            "Summarize the following conversation between a user and a coding "
            "assistant. Preserve the task the user wants, key decisions, file "
            "paths, and the current state of any work in progress. Be concise — "
            "a few short paragraphs at most. Reply with the summary only.\n\n"
            "--- conversation ---\n"
            f"{transcript}"
        )
        parts: list[str] = []
        async for event in self._client.stream([{"role": "user", "content": prompt}]):
            if isinstance(event, ContentDelta):
                parts.append(event.text)
            elif isinstance(event, StreamError):
                raise RuntimeError(event.message)
            elif isinstance(event, TurnComplete):
                break
        summary = "".join(parts).strip()
        if not summary:
            raise RuntimeError("summarization returned no content")
        return summary

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
                self.conversation.add(
                    Message(
                        role="assistant",
                        content="".join(content_parts),
                        reasoning=reasoning,
                        reasoning_seconds=reasoning_seconds,
                        tool_calls=tool_calls_wire or None,
                    )
                )
                usage = event.usage or self._estimate_usage(content_parts)
                event = TurnComplete(
                    usage=usage,
                    reasoning_seconds=reasoning_seconds,
                    has_tool_calls=bool(call_order),
                )
            yield event

    def _estimate_usage(self, content_parts: list[str]) -> Usage:
        prompt_chars = sum(len(message.content or "") for message in self.conversation.messages)
        return Usage(
            input_tokens=prompt_chars // _CHARS_PER_TOKEN,
            output_tokens=sum(len(part) for part in content_parts) // _CHARS_PER_TOKEN,
            approximate=True,
        )