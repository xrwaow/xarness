"""Async OpenAI-compatible chat client.

Streaming over SSE via httpx. This module knows nothing about the UI: it
consumes wire-format message dicts and yields typed stream events from
``xarness.events``.
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
    ToolCallArgumentsDelta,
    ToolCallArgumentsDone,
    ToolCallStarted,
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


class _ToolCallAccumulator:
    """Tracks in-progress tool calls by stream index across chunks.

    OpenAI-compatible streams only send ``id``/``function.name`` on the first
    delta for a given index; later deltas for that index carry only
    ``function.arguments`` fragments. Indexed by position, not id, because
    the id isn't guaranteed present on every fragment.
    """

    def __init__(self) -> None:
        self._by_index: dict[int, dict[str, str]] = {}  # index -> {call_id, name, args}
        self._order: list[int] = []

    def feed(self, tool_call_deltas: list[dict[str, Any]]) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        for tc in tool_call_deltas:
            index = tc.get("index", 0)
            fn = tc.get("function") or {}
            if index not in self._by_index:
                call_id = tc.get("id") or f"call_{index}"
                name = fn.get("name") or ""
                self._by_index[index] = {"call_id": call_id, "name": name, "args": ""}
                self._order.append(index)
                events.append(ToolCallStarted(call_id=call_id, name=name))
            entry = self._by_index[index]
            # Some providers repeat the id/name on every fragment; keep the
            # first non-empty name we see.
            if fn.get("name") and not entry["name"]:
                entry["name"] = fn["name"]
            args_fragment = fn.get("arguments")
            if isinstance(args_fragment, str) and args_fragment:
                entry["args"] += args_fragment
                events.append(ToolCallArgumentsDelta(call_id=entry["call_id"], text=args_fragment))
        return events

    def finish(self) -> list[StreamEvent]:
        return [
            ToolCallArgumentsDone(
                call_id=self._by_index[i]["call_id"],
                name=self._by_index[i]["name"],
                arguments_json=self._by_index[i]["args"],
            )
            for i in self._order
        ]

    @property
    def has_calls(self) -> bool:
        return bool(self._order)


class ChatClient:
    """Streams chat completions from an OpenAI-compatible endpoint."""

    def __init__(self, profile: ProviderProfile, api_key: str | None) -> None:
        self._profile = profile
        self._api_key = api_key

    def build_payload(
        self,
        wire_messages: Sequence[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._profile.model_id,
            "messages": list(wire_messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        effort = reasoning_effort_for(self._profile.cot_strength)
        if effort is not None:
            payload["reasoning_effort"] = effort
        if tools:
            payload["tools"] = tools
        return payload

    async def stream(
        self,
        wire_messages: Sequence[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield stream events for one round.

        Always starts with :class:`ProcessingStarted`; ends with either
        :class:`TurnComplete` (``has_tool_calls`` set if any were made, usage
        included when the provider returned it) or :class:`StreamError`.
        """
        url = self._profile.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Accept": "text/event-stream",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        yield ProcessingStarted()

        usage: Usage | None = None
        tool_calls = _ToolCallAccumulator()
        try:
            async with (
                httpx.AsyncClient(timeout=_TIMEOUT) as client,
                client.stream(
                    "POST", url, json=self.build_payload(wire_messages, tools), headers=headers
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
                    for event in _delta_events(chunk, tool_calls):
                        yield event
        except httpx.HTTPError as exc:
            yield StreamError(f"request to {url} failed: {exc}")
            return

        for event in tool_calls.finish():
            yield event
        yield TurnComplete(usage=usage, has_tool_calls=tool_calls.has_calls)


def _delta_events(chunk: dict[str, Any], tool_calls: _ToolCallAccumulator) -> list[StreamEvent]:
    """Extract reasoning/content/tool-call deltas from one SSE chunk.

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
        tc_deltas = delta.get("tool_calls")
        if isinstance(tc_deltas, list) and tc_deltas:
            events.extend(tool_calls.feed(tc_deltas))
    return events


def _usage_from_chunk(chunk: dict[str, Any]) -> Usage | None:
    raw = chunk.get("usage")
    if not isinstance(raw, dict):
        return None
    input_tokens = raw.get("prompt_tokens", raw.get("input_tokens")) or 0
    output_tokens = raw.get("completion_tokens", raw.get("output_tokens")) or 0
    return Usage(input_tokens=int(input_tokens), output_tokens=int(output_tokens))
