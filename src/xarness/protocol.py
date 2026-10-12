"""Wire protocol models shared by the backend, the TUI client and the API.

One source of truth for every event and command that crosses a process
boundary. Pydantic models with a ``type`` discriminator; FastAPI generates
the OpenAPI schema from these directly, so the docs can never drift from
the code.

Usage, ImageAttachment, ToolCall and Message are the very models defined in
:mod:`xarness.events` / :mod:`xarness.conversation` (re-exported below) — a
message is persisted, converted to a provider request and shown to a client
as one object, with no mirrored copy to keep in sync. The transient stream
deltas are likewise the models defined in :mod:`xarness.events`.
"""

from __future__ import annotations

from typing import Annotated, Any, get_args, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .conversation import Message
from .events import (
    ContentDelta,
    EventBase,
    ProcessingStarted,
    ReasoningDelta,
    StreamError,
    ToolCallArgumentsDelta,
    ToolCallArgumentsDone,
    ToolCallStarted,
    TurnComplete,
    Usage,
)
from .images import ImageAttachment

PROTOCOL_VERSION = 1

# ---------------------------------------------------------------------------
# primitives

# The one and only usage/message/image shapes: the events/conversation models
# cross the wire as-is (they are pydantic models), so there is no mirrored
# protocol copy to keep in sync. Aliases keep the protocol namespace complete.
UsageModel = Usage
MessageModel = Message
ImageAttachmentModel = ImageAttachment


class QueuedMessage(BaseModel):
    """A steering message waiting to be injected at a round boundary."""

    text: str
    images: list[ImageAttachmentModel] | None = None


class PendingAsk(BaseModel):
    """A live ask-tool prompt (session state, answerable by any client)."""

    ask_id: str
    questions: list[str]


class PendingDrift(BaseModel):
    """A live resume-drift prompt (session state, resolvable by any client)."""

    workspace: str
    detail: str = ""


class StatusPayload(BaseModel):
    """Everything a client needs for its status bar / steer bar / prompts."""

    state: Literal["idle", "running", "compacting"]
    session_name: str | None = None
    workspace: str | None = None
    mode: Literal["plan", "write"] = "write"
    model_name: str = ""
    model_display_name: str = ""
    model_id: str = ""
    cot_strength: str = "medium"
    max_context: int = 128_000
    sandbox_name: str | None = None
    use_container: bool = True
    enable_tracking: bool = True
    filter_list: list[str] = Field(default_factory=list)
    total_in: int = 0
    total_out: int = 0
    last_in: int = 0
    last_out: int = 0
    context_used: int = 0
    queued: list[QueuedMessage] = Field(default_factory=list)
    pending_ask: PendingAsk | None = None
    pending_drift: PendingDrift | None = None
    clients_attached: int = 0
    # Tool names exposed to the model (for the /tools command).
    tools: list[str] = Field(default_factory=list)
    # True when a turn finished while no client was attached; cleared on
    # the next subscribe ("finished while you were away").
    finished_while_away: bool = False


class Snapshot(BaseModel):
    """The full state a client receives when it attaches (SSE first event)."""

    seq: int = 0
    messages: list[MessageModel] = Field(default_factory=list)
    compact_snapshot: list[MessageModel] = Field(default_factory=list)
    turn_start: int | None = None
    status: StatusPayload = Field(default_factory=StatusPayload)
    # Events buffered during the current turn (empty when idle).
    turn_events: list["Event"] = Field(default_factory=list)


class ServerInfo(BaseModel):
    """GET /health response."""

    version: str
    protocol_version: int = PROTOCOL_VERSION
    pid: int
    uptime_seconds: float
    persistent: bool = False
    unix_socket: str | None = None


class SessionSummary(BaseModel):
    """One row of GET /sessions (active or saved)."""

    name: str
    workspace: str | None = None
    state: str = "saved"  # idle | running | compacting | waiting | saved
    active: bool = False
    clients_attached: int = 0
    model: str | None = None
    sandbox: str | None = None
    last_user_message_at: str | None = None
    message_count: int = 0
    tracking: bool = False


class ServerStats(BaseModel):
    """GET /server/stats: the dashboard's data."""

    host: ServerInfo
    started_at: str
    sessions_by_state: dict[str, int] = Field(default_factory=dict)
    clients_attached: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    sessions: list[SessionSummary] = Field(default_factory=list)


class ConfigInfo(BaseModel):
    """GET /config: names and non-secret fields for the client's pickers."""

    models: list[dict[str, Any]] = Field(default_factory=list)
    sandboxes: list[dict[str, Any]] = Field(default_factory=list)
    presets: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    # What a fresh session starts on (from the default preset), so the
    # client's draft status bar shows real values before any session exists.
    default_model: str | None = None
    default_model_display_name: str | None = None
    default_sandbox: str | None = None


class WorkspaceInfo(BaseModel):
    """One known workspace directory."""

    path: str
    session_count: int = 0
    active_sessions: int = 0


class DiffFileInfo(BaseModel):
    path: str
    additions: int = 0
    deletions: int = 0
    is_new: bool = False
    # "agent" (a turn's edits) or "drift" (made outside the session).
    source: Literal["agent", "drift"] = "agent"


class DiffInfo(BaseModel):
    files: list[DiffFileInfo] = Field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    # Unified diff text — only populated when a specific path is requested
    # (the summary poll passes no path and must stay cheap).
    text: str = ""


# ---------------------------------------------------------------------------
# events

# EventBase and the transient stream deltas (ProcessingStarted, ReasoningDelta,
# ContentDelta, ToolCallStarted, ToolCallArgumentsDelta, ToolCallArgumentsDone,
# TurnComplete, StreamError) come from :mod:`xarness.events` — see the import
# above.







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
    ok: bool
    parse_error: bool = False
    output: str = ""
    error: str = ""
    header: str = ""




class UserMessageAdded(EventBase):
    """A user message landed in the history (sent or steered).

    This is the only way a user message reaches any client — there is no
    local echo, so the sender renders it exactly like everyone else."""

    type: Literal["user_message_added"] = "user_message_added"
    text: str
    images: list[ImageAttachmentModel] | None = None
    message_id: str | None = None
    send_time: str | None = None


class StatusChanged(EventBase):
    type: Literal["status_changed"] = "status_changed"
    status: StatusPayload = Field(default_factory=StatusPayload)


class Notice(EventBase):
    type: Literal["notice"] = "notice"
    text: str
    level: Literal["info", "warn", "error"] = "info"


class HistoryReplaced(EventBase):
    """The conversation was rewritten; clients re-render from scratch."""

    type: Literal["history_replaced"] = "history_replaced"
    snapshot: Snapshot


class AskRequested(EventBase):
    type: Literal["ask_requested"] = "ask_requested"
    ask_id: str
    questions: list[str]


class AskResolved(EventBase):
    type: Literal["ask_resolved"] = "ask_resolved"
    ask_id: str
    answers: list[str] | None = None


class DriftResolved(EventBase):
    type: Literal["drift_resolved"] = "drift_resolved"
    accepted: bool


class TurnFinished(EventBase):
    type: Literal["turn_finished"] = "turn_finished"
    elapsed: float = 0.0
    # "Worked for X | Y Z W" — total | streaming | tools | other; the same
    # string the session persists on the turn's last assistant message.
    summary: str = ""


class SessionDeleted(EventBase):
    type: Literal["session_deleted"] = "session_deleted"
    name: str = ""


Event = Annotated[
    Union[
        ProcessingStarted,
        ReasoningDelta,
        ContentDelta,
        ToolCallStarted,
        ToolCallArgumentsDelta,
        ToolCallArgumentsDone,
        ToolExecStarted,
        ToolOutputDelta,
        ToolFinished,
        TurnComplete,
        StreamError,
        UserMessageAdded,
        StatusChanged,
        Notice,
        HistoryReplaced,
        AskRequested,
        AskResolved,
        DriftResolved,
        TurnFinished,
        SessionDeleted,
    ],
    Field(discriminator="type"),
]
Snapshot.model_rebuild()


def _union_members(annotation) -> tuple[type[BaseModel], ...]:
    """The member models of a discriminated ``Annotated[Union[...]]``."""
    return get_args(get_args(annotation)[0])


EVENT_TYPES: dict[str, type[BaseModel]] = {
    model.model_fields["type"].default: model  # type: ignore[index]
    for model in _union_members(Event)
}


EVENT_ADAPTER = TypeAdapter(Event)


# ---------------------------------------------------------------------------
# commands


class CommandBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SendMessage(CommandBase):
    type: Literal["send_message"] = "send_message"
    text: str
    images: list[ImageAttachmentModel] | None = None


class Cancel(CommandBase):
    type: Literal["cancel"] = "cancel"


class AnswerAsk(CommandBase):
    type: Literal["answer_ask"] = "answer_ask"
    ask_id: str
    answers: list[str] | None = None  # None = skipped (esc)


class ResolveDrift(CommandBase):
    type: Literal["resolve_drift"] = "resolve_drift"
    accept: bool


class DiscardQueued(CommandBase):
    type: Literal["discard_queued"] = "discard_queued"


class FlushQueued(CommandBase):
    type: Literal["flush_queued"] = "flush_queued"


class Undo(CommandBase):
    type: Literal["undo"] = "undo"


class Retry(CommandBase):
    type: Literal["retry"] = "retry"


class UndoTo(CommandBase):
    type: Literal["undo_to"] = "undo_to"
    message_id: str


class Compact(CommandBase):
    type: Literal["compact"] = "compact"


class Accept(CommandBase):
    type: Literal["accept"] = "accept"


class Reject(CommandBase):
    type: Literal["reject"] = "reject"


class SetMode(CommandBase):
    type: Literal["set_mode"] = "set_mode"
    mode: Literal["plan", "write"]


class SwitchModel(CommandBase):
    type: Literal["switch_model"] = "switch_model"
    name: str


class SetSandbox(CommandBase):
    type: Literal["set_sandbox"] = "set_sandbox"
    name: str


class FilterAdd(CommandBase):
    type: Literal["filter_add"] = "filter_add"
    path: str


class FilterRemove(CommandBase):
    type: Literal["filter_remove"] = "filter_remove"
    path: str


Command = Annotated[
    Union[
        SendMessage,
        Cancel,
        AnswerAsk,
        ResolveDrift,
        DiscardQueued,
        FlushQueued,
        Undo,
        Retry,
        UndoTo,
        Compact,
        Accept,
        Reject,
        SetMode,
        SwitchModel,
        SetSandbox,
        FilterAdd,
        FilterRemove,
    ],
    Field(discriminator="type"),
]

COMMAND_TYPES: dict[str, type[BaseModel]] = {
    model.model_fields["type"].default: model  # type: ignore[index]
    for model in _union_members(Command)
}


class CommandResult(BaseModel):
    """Reply to one command, delivered only to its issuer."""

    cmd_id: str = ""
    ok: bool
    error: str | None = None
    payload: dict[str, Any] | None = None


class CommandEnvelope(BaseModel):
    """POST /sessions/{name}/commands body: one command + a client-generated id."""

    cmd_id: str
    command: Command


class CreateSessionRequest(BaseModel):
    """POST /v1/sessions body."""

    workspace: str | None = None
    model: str | None = None
    sandbox: str | None = None
    preset: str | None = None
    no_init_repo: bool = False
    network: bool | None = None
    gpu: bool | None = None
    no_session: bool = False
    first_message: str | None = None


