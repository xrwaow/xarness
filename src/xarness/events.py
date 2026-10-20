"""Typed stream events consumed by the UI, and the Usage model.

The API client produces the stream events; the Textual layer never sees
HTTP. They are the very same pydantic models the protocol uses (re-exported
in :mod:`xarness.protocol`), so a stream event crosses the wire as-is with
no mirrored copy or conversion step. The set is deliberately small and
provider-agnostic so a tool-calling layer can add new event types later
without changing existing consumers.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

State = Literal["idle", "running", "compacting"]
Mode = Literal["plan", "write"]


class EventBase(BaseModel):
    """Base for wire events.

    Unknown fields are ignored, not rejected: the host and client can be
    different builds across a restart, and a newer host may add event
    fields a client doesn't know yet."""

    model_config = ConfigDict(extra="ignore")

    # Server-side emission time. Clients use differences between these
    # stamps to derive live durations (reasoning time, turn time).
    t: float = Field(default_factory=time.monotonic)


class ProcessingStarted(EventBase):
    """Emitted once, right after the request is sent (state 1: processing)."""

    type: Literal["processing_started"] = "processing_started"


class ReasoningDelta(EventBase):
    """A chunk of reasoning/thinking-channel text (state 2: thinking)."""

    type: Literal["reasoning_delta"] = "reasoning_delta"
    text: str


class ContentDelta(EventBase):
    """A chunk of the final answer (state 3: answer streaming)."""

    type: Literal["content_delta"] = "content_delta"
    text: str


class ToolCallStarted(EventBase):
    """First sight of a tool call: id + name known, arguments may still stream."""

    type: Literal["tool_call_started"] = "tool_call_started"
    call_id: str
    name: str


class ToolCallArgumentsDelta(EventBase):
    """A raw JSON argument fragment for a call already announced via ToolCallStarted."""

    type: Literal["tool_call_arguments_delta"] = "tool_call_arguments_delta"
    call_id: str
    text: str


class ToolCallArgumentsDone(EventBase):
    """Arguments fully received for one call — safe to parse and execute."""

    type: Literal["tool_call_arguments_done"] = "tool_call_arguments_done"
    call_id: str
    name: str
    arguments_json: str


class Usage(BaseModel):
    """Token usage for a turn. ``approximate`` marks locally estimated counts.

    A pydantic model (not a dataclass) so the very same object crosses the
    wire in snapshots and persists losslessly."""

    input_tokens: int = 0
    output_tokens: int = 0
    approximate: bool = False


class TurnComplete(EventBase):
    """End of one model round, with usage if the provider reported it.

    A "round" is one full model response (reasoning + content + possibly tool
    calls). ``has_tool_calls`` tells the consumer whether another round
    follows — tool results must be appended and the model called again — or
    whether this round is the final answer.
    """

    type: Literal["turn_complete"] = "turn_complete"
    usage: Usage | None = None
    has_tool_calls: bool = False


class StreamError(EventBase):
    """A request-level failure (connection error, non-200, in-stream error)."""

    type: Literal["stream_error"] = "stream_error"
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
