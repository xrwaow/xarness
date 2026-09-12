"""Async OpenAI-compatible chat client.

Streaming over SSE via httpx. This module knows nothing about the UI: it
consumes wire-format message dicts and yields typed stream events from
``agentcli.events``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from .config import CotStrength, ProviderProfile
from .events import (
    ContentDelta,
    ProcessingStarted,
    ReasoningDelta,
    StreamError,
    StreamEvent,
    TurnComplete,
    Usage,
)

_TIMEOUT = httpx.Timeout(connect=15.0, read=300.0, write=30.0, pool=15.0)

# cot_strength -> the reasoning-effort-style parameter this provider expects.
# Kept isolated so per-provider adjustments never touch the rest of the code.
# "off" omits the parameter entirely, which is the portable way to ask for no
# deliberate reasoning across OpenAI-compatible APIs.
_REASONING_EFFORT: dict[CotStrength, str | None] = {
    CotStrength.OFF: None,
    CotStrength.LOW: "low",
    CotStrength.MEDIUM: "medium",
    CotStrength.HIGH: "high",
}


def reasoning_effort_for(strength: CotStrength) -> str | None:
    """Map config ``cot_strength`` to the provider's reasoning-effort param."""
    return _REASONING_EFFORT[strength]


class ChatClient:
    """Streams chat completions from an OpenAI-compatible endpoint."""

    def __init__(self, profile: ProviderProfile, api_key: str) -> None:
        self._profile = profile
        self._api_key = api_key

    def build_payload(self, wire_messages: Sequence[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._profile.model_id,
            "messages": list(wire_messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        effort = reasoning_effort_for(self._profile.cot_strength)
        if effort is not None:
            payload["reasoning_effort"] = effort
        return payload

    async def stream(self, wire_messages: Sequence[dict[str, Any]]) -> AsyncIterator[StreamEvent]:
        """Yield stream events for one turn.

        Always starts with :class:`ProcessingStarted`; ends with either
        :class:`TurnComplete` (usage included when the provider returned it)
        or :class:`StreamError`.
        """
        url = self._profile.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "text/event-stream",
        }
        yield ProcessingStarted()

        usage: Usage | None = None
        try:
            async with (
                httpx.AsyncClient(timeout=_TIMEOUT) as client,
                client.stream(
                    "POST", url, json=self.build_payload(wire_messages), headers=headers
                ) as response,
            ):
                if response.status_code != 200:
                    detail = (await response.aread()).decode(errors="replace").strip()
                    yield StreamError(f"HTTP {response.status_code} from {url}: {detail[:500]}")
                    return

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue  # SSE comments / event:/retry: lines
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if isinstance(chunk.get("error"), dict):
                        message = chunk["error"].get("message") or str(chunk["error"])
                        yield StreamError(f"stream error from {url}: {message}")
                        return

                    reported = _usage_from_chunk(chunk)
                    if reported is not None:
                        usage = reported
                    for event in _delta_events(chunk):
                        yield event
        except httpx.HTTPError as exc:
            yield StreamError(f"request to {url} failed: {exc}")
            return

        yield TurnComplete(usage=usage)


def _delta_events(chunk: dict[str, Any]) -> list[StreamEvent]:
    """Extract reasoning/content deltas from one SSE chunk.

    Checks both common OpenAI-compatible reasoning field names:
    ``reasoning_content`` (DeepSeek-style) and ``reasoning`` (OpenRouter-style).
    """
    events: list[StreamEvent] = []
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            events.append(ReasoningDelta(reasoning))
        content = delta.get("content")
        if isinstance(content, str) and content:
            events.append(ContentDelta(content))
    return events


def _usage_from_chunk(chunk: dict[str, Any]) -> Usage | None:
    raw = chunk.get("usage")
    if not isinstance(raw, dict):
        return None
    input_tokens = raw.get("prompt_tokens", raw.get("input_tokens")) or 0
    output_tokens = raw.get("completion_tokens", raw.get("output_tokens")) or 0
    return Usage(input_tokens=int(input_tokens), output_tokens=int(output_tokens))
