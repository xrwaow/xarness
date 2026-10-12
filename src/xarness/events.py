"""Typed stream events consumed by the UI, and the Usage model.

The API client produces the stream events; the Textual layer never sees
HTTP. They are the very same pydantic models the protocol uses (re-exported
in :mod:`xarness.protocol`), so a stream event crosses the wire as-is with
no mirrored copy or conversion step. The set is deliberately small and
provider-agnostic so a tool-calling layer can add new event types later
without changing existing consumers.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict


class EventBase(BaseModel):
    """Base for wire events: a session-assigned ``seq`` plus tolerance.

    Unknown fields are ignored, not rejected: the host and client can be
    different builds across a restart, and a newer host may add event
    fields a client doesn't know yet. Positional construction is allowed
    (``ContentDelta("hi")``) to keep call sites terse."""

    seq: int = 0
    model_config = ConfigDict(extra="ignore")

    def __init__(self, *args: object, **kwargs: object) -> None:
        if args:
            if kwargs:
                raise TypeError("pass fields positionally or by keyword, not both")
            names = [
                name for name in type(self).model_fields if name not in ("type", "seq")
            ]
            if len(args) > len(names):
                raise TypeError(f"too many positional arguments for {type(self).__name__}")
            kwargs = dict(zip(names, args))
        super().__init__(**kwargs)


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


class ToolCallStatus(Enum):
    """Lifecycle of one tool call, driving the status-dot color in the UI."""

    MAKING_CALL = "making_call"  # yellow — arguments streaming / call in flight
    PARSING_ERROR = "parsing_error"  # red — model produced invalid arguments JSON
    CALL_SUCCEEDED = "call_succeeded"  # green
    CALL_FAILED = "call_failed"  # red — tool ran but errored


class Usage(BaseModel):
    """Token usage for a turn. ``approximate`` marks locally estimated counts.

    A pydantic model (not a dataclass) so the very same object crosses the
    wire in snapshots and persists losslessly. Constructed positionally in
    many places, so ``Usage(10, 4)`` keeps working."""

    input_tokens: int = 0
    output_tokens: int = 0
    approximate: bool = False

    def __init__(
        self, input_tokens: int = 0, output_tokens: int = 0, **kwargs: object
    ) -> None:
        super().__init__(
            input_tokens=input_tokens, output_tokens=output_tokens, **kwargs
        )


class TurnComplete(EventBase):
    """End of one model round, with usage if the provider reported it.

    A "round" is one full model response (reasoning + content + possibly tool
    calls). ``has_tool_calls`` tells the consumer whether another round
    follows — tool results must be appended and the model called again — or
    whether this round is the final answer.
    """

    type: Literal["turn_complete"] = "turn_complete"
    usage: Usage | None = None
    reasoning_seconds: float | None = None
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
