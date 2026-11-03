"""SSE stream events: transient deltas and authoritative history updates.

``StreamEvent`` are the frames a round streams directly from the provider;
``HostEvent`` the ones the session itself emits (history, status, prompts,
turn lifecycle). ``Event`` is the discriminated union of both — the only
frame type a client's ``/events`` stream carries after the snapshot.
"""

from __future__ import annotations

import time
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from ..conversation import Message, ToolStatus, Usage
from .views import Activity, DiffInfo, SessionConfigView, Snapshot

PROTOCOL_VERSION = 11

# Error convention: REST routes fail with HTTP 4xx (400 bad request, 404
# missing session/path); a ``command`` always returns 200 with
# ``CommandResult.ok=False`` and ``error`` text, because a rejected command is
# a valid, attributable outcome the UI renders inline.


Level = Literal["info", "warn", "error"]

# Why a HistoryReplaced fired. accept/reject/drift stay valid (an older host
# may still send them) but are no longer emitted — they carry no message
# change, so they use :class:`CheckpointsChanged` instead.
HistoryReason = Literal["compaction", "rollback", "drift", "accept", "reject"]


class EventBase(BaseModel):
    """Base for wire events.

    Unknown fields are ignored, not rejected: the host and client can be
    different builds across a restart, and a newer host may add event
    fields a client doesn't know yet."""

    model_config = ConfigDict(extra="ignore")

    # Server-side emission time. Clients use differences between these
    # values (never the absolute value — it is a monotonic clock reading,
    # not a wall stamp) to order events and to time block durations.
    t: float = Field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# stream events (what a model round streams)

class ProcessingStarted(EventBase):
    """Emitted once per round, right after the request is sent."""

    type: Literal["processing_started"] = "processing_started"


class ReasoningDelta(EventBase):
    """A chunk of reasoning/thinking-channel text."""

    type: Literal["reasoning_delta"] = "reasoning_delta"
    text: str


class ContentDelta(EventBase):
    """A chunk of the final answer."""

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


class RoundComplete(EventBase):
    """End of one model round, with usage if the provider reported it.

    A "round" is one full model response (reasoning + content + possibly tool
    calls). ``has_tool_calls`` tells the consumer whether another round
    follows — tool results must be appended and the model called again — or
    whether this round is the final answer.
    """

    type: Literal["round_complete"] = "round_complete"
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
    | RoundComplete
    | StreamError
)


# ---------------------------------------------------------------------------
# host events

class ToolExecStarted(EventBase):
    type: Literal["tool_exec_started"] = "tool_exec_started"
    call_id: str


class ToolOutputDelta(EventBase):
    type: Literal["tool_output_delta"] = "tool_output_delta"
    call_id: str
    text: str


class ToolFinished(EventBase):
    type: Literal["tool_finished"] = "tool_finished"
    call_id: str
    status: ToolStatus
    content: str = ""


class MessageAdded(EventBase):
    """A message landed in the history. The only way a user message reaches
    any client — there is no local echo, so the sender renders it exactly
    like everyone else. Assistant/tool messages ride along so a client that
    missed the stream deltas still gets the authoritative copy (id, timing,
    usage) before the next snapshot."""

    type: Literal["message_added"] = "message_added"
    message: Message


class StatusChanged(EventBase):
    """The session's activity changed (state, usage, queue, prompts).

    Carries only the fast half; the config view travels separately
    (:class:`ConfigChanged`) so per-round updates never re-send static data."""

    type: Literal["status_changed"] = "status_changed"
    activity: Activity = Field(default_factory=lambda: Activity(state="idle"))


class ConfigChanged(EventBase):
    """The session's config view changed (mode, model, sandbox, filter,
    refs, tools). Emitted only on actual settings changes."""

    type: Literal["config_changed"] = "config_changed"
    config: SessionConfigView = Field(default_factory=SessionConfigView)


class Notice(EventBase):
    type: Literal["notice"] = "notice"
    text: str
    level: Level = "info"


class HistoryReplaced(EventBase):
    """The conversation was rewritten (a compaction, a rollback, or a
    tracking rebaseline); clients re-render from the attached snapshot.

    Carries the structured outcome of the rewrite so clients render it
    without parsing prose."""

    type: Literal["history_replaced"] = "history_replaced"
    snapshot: Snapshot
    reason: HistoryReason
    # Compaction: token counts around the rewrite; whether it fired
    # automatically; False marks "nothing to compact yet".
    automatic: bool = False
    before_tokens: int = 0
    after_tokens: int = 0
    # Rollback: scope description, how many turns had file edits reversed,
    # whether the user text is being resent, and what happened to the files.
    scope: str = "last turn"
    reverted: int = 0
    resend: bool = False
    note: Literal["reverted", "no_edits", "no_tracking"] = "reverted"


class CheckpointsChanged(EventBase):
    """Turn checkpoints were rebaselined (accept, reject, drift resolve) but
    NO message changed — clients refresh the diff/pending view and status,
    not the transcript. A full ``HistoryReplaced`` (snapshot attached) is
    reserved for rewrites that actually changed history: rollback and
    compaction."""

    type: Literal["checkpoints_changed"] = "checkpoints_changed"
    reason: HistoryReason = "accept"


class AskResolved(EventBase):
    type: Literal["ask_resolved"] = "ask_resolved"
    ask_id: str
    answers: list[str] | None = None


class DriftResolved(EventBase):
    type: Literal["drift_resolved"] = "drift_resolved"
    accepted: bool


class DiffChanged(EventBase):
    """The pending-changes summary changed (a mutating tool ran, or the turn
    ended). Clients render this instead of polling ``GET /diff`` after every
    tool result — including read-only ones."""

    type: Literal["diff_changed"] = "diff_changed"
    summary: DiffInfo = Field(default_factory=DiffInfo)


class TurnEnded(EventBase):
    """The turn is fully unwound (history persisted, state back to idle).

    Emitted on every path — finished, cancelled, failed — so clients never
    have to infer "the turn is over" from a StatusChanged transition."""

    type: Literal["turn_ended"] = "turn_ended"
    outcome: Literal["finished", "cancelled", "failed"] = "finished"
    elapsed: float = 0.0


class SessionDeleted(EventBase):
    type: Literal["session_deleted"] = "session_deleted"
    name: str = ""


HostEvent = (
    ToolExecStarted
    | ToolOutputDelta
    | ToolFinished
    | MessageAdded
    | StatusChanged
    | ConfigChanged
    | Notice
    | HistoryReplaced
    | CheckpointsChanged
    | AskResolved
    | DriftResolved
    | DiffChanged
    | TurnEnded
    | SessionDeleted
)

Event = Annotated[
    Union[StreamEvent, HostEvent],  # type: ignore[misc]
    Field(discriminator="type"),
]

Snapshot.model_rebuild()
EVENT_ADAPTER = TypeAdapter(Event)
