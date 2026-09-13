"""Typed stream events consumed by the UI.

The API client produces these; the Textual layer never sees HTTP. The set is
deliberately small and provider-agnostic so a tool-calling layer can add new
event types later without changing existing consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
class ToolCallStarted:
    """First sight of a tool call: id + name known, arguments may still stream."""

    call_id: str
    name: str


@dataclass(slots=True)
class ToolCallArgumentsDelta:
    """A raw JSON argument fragment for a call already announced via ToolCallStarted."""

    call_id: str
    text: str


@dataclass(slots=True)
class ToolCallArgumentsDone:
    """Arguments fully received for one call — safe to parse and execute."""

    call_id: str
    name: str
    arguments_json: str


class ToolCallStatus(Enum):
    """Lifecycle of one tool call, driving the status-dot color in the UI."""

    MAKING_CALL = "making_call"  # yellow — arguments streaming / call in flight
    PARSING_ERROR = "parsing_error"  # red — model produced invalid arguments JSON
    CALL_SUCCEEDED = "call_succeeded"  # green
    CALL_FAILED = "call_failed"  # red — tool ran but errored


@dataclass(slots=True)
class Usage:
    """Token usage for a turn. ``approximate`` marks locally estimated counts."""

    input_tokens: int
    output_tokens: int
    approximate: bool = False


@dataclass(slots=True)
class TurnComplete:
    """End of one model round, with usage if the provider reported it.

    A "round" is one full model response (reasoning + content + possibly tool
    calls). ``has_tool_calls`` tells the consumer whether another round
    follows — tool results must be appended and the model called again — or
    whether this round is the final answer.
    """

    usage: Usage | None = None
    reasoning_seconds: float | None = None
    has_tool_calls: bool = False


@dataclass(slots=True)
class StreamError:
    """A request-level failure (connection error, non-200, in-stream error)."""

    message: str


StreamEvent = (
    ProcessingStarted
    | ReasoningDelta
    | ContentDelta
    | ToolCallStarted
    | ToolCallArgumentsDelta
    | ToolCallArgumentsDone
    | TurnComplete
    | StreamError
)
