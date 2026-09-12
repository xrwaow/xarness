"""Typed stream events consumed by the UI.

The API client produces these; the Textual layer never sees HTTP. The set is
deliberately small and provider-agnostic so a tool-calling layer can add new
event types later without changing existing consumers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ProcessingStarted:
    """Emitted once, right after the request is sent (state 1: processing)."""


@dataclass(slots=True)
class ReasoningDelta:
    """A chunk of reasoning/thinking-channel text (state 2: thinking)."""

    text: str


@dataclass(slots=True)
class ContentDelta:
    """A chunk of the final answer (state 3: answer streaming)."""

    text: str


@dataclass(slots=True)
class Usage:
    """Token usage for a turn. ``approximate`` marks locally estimated counts."""

    input_tokens: int
    output_tokens: int
    approximate: bool = False


@dataclass(slots=True)
class TurnComplete:
    """End of a turn, with usage if the provider reported it."""

    usage: Usage | None = None
    reasoning_seconds: float | None = None


@dataclass(slots=True)
class StreamError:
    """A request-level failure (connection error, non-200, in-stream error)."""

    message: str


StreamEvent = ProcessingStarted | ReasoningDelta | ContentDelta | TurnComplete | StreamError
