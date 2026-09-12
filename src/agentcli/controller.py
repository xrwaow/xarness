"""Glue between the API client and the UI.

Owns the :class:`Conversation` (plain data, no widgets) and enriches raw
client events with cross-event information: reasoning timing, usage fallback
estimation, and appending the finished assistant message to the history.
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
    StreamEvent,
    TurnComplete,
    Usage,
)

# Rough chars-per-token for the fallback estimate when the provider sends no
# usage data. Deliberately conservative; results are labeled approximate.
_CHARS_PER_TOKEN = 4


class StreamClient(Protocol):
    """Anything that can stream a turn; ChatClient is the real implementation."""

    def stream(
        self, wire_messages: Sequence[dict[str, Any]]
    ) -> AsyncIterator[StreamEvent]: ...


class ChatController:
    """One instance per session; the UI consumes ``send()`` event streams."""

    def __init__(
        self,
        profile: ProviderProfile,
        api_key: str,
        client: StreamClient | None = None,
    ) -> None:
        self.profile = profile
        self.conversation = Conversation()
        self._client = client or ChatClient(profile, api_key)

    async def send(self, user_text: str) -> AsyncIterator[StreamEvent]:
        """Send a user message, yielding enriched stream events."""
        self.conversation.add(Message(role="user", content=user_text))

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        first_reasoning_at: float | None = None
        first_content_at: float | None = None

        async for event in self._client.stream(self.conversation.to_wire()):
            if isinstance(event, ReasoningDelta):
                if first_reasoning_at is None:
                    first_reasoning_at = time.monotonic()
                reasoning_parts.append(event.text)
            elif isinstance(event, ContentDelta):
                if first_content_at is None:
                    first_content_at = time.monotonic()
                content_parts.append(event.text)
            elif isinstance(event, TurnComplete):
                reasoning = "".join(reasoning_parts) or None
                self.conversation.add(
                    Message(role="assistant", content="".join(content_parts), reasoning=reasoning)
                )
                reasoning_seconds: float | None = None
                if first_reasoning_at is not None:
                    end = first_content_at or time.monotonic()
                    reasoning_seconds = end - first_reasoning_at
                usage = event.usage or self._estimate_usage(content_parts)
                event = TurnComplete(usage=usage, reasoning_seconds=reasoning_seconds)
            yield event

    def _estimate_usage(self, content_parts: list[str]) -> Usage:
        prompt_chars = sum(len(message.content or "") for message in self.conversation.messages)
        return Usage(
            input_tokens=prompt_chars // _CHARS_PER_TOKEN,
            output_tokens=sum(len(part) for part in content_parts) // _CHARS_PER_TOKEN,
            approximate=True,
        )
