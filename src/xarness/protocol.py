"""Wire protocol models shared by the backend, the TUI client and the API.

One source of truth for every event and command that crosses a process
boundary. Pydantic models with a ``type`` discriminator; FastAPI generates
the OpenAPI schema from these directly, so the docs can never drift from
the code.

Usage, ImageAttachment, ToolCall and Message are the very models defined in
:mod:`xarness.conversation` / :mod:`xarness.images` — a message is persisted,
converted to a provider request and shown to a client as one object, with no
mirrored copy to keep in sync. Every event (including the transient stream
deltas) is defined here.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .conversation import (  # noqa: F401 - re-exported for clients
    SUMMARY_PREFIX,
    Message,
    Usage,
    image_refs,
    is_summary,
)
from .images import ImageAttachment, ImageRef

PROTOCOL_VERSION = 9

# Error convention: REST routes fail with HTTP 4xx (400 bad request, 404
# missing session/path); a ``command`` always returns 200 with
# ``CommandResult.ok=False`` and ``error`` text, because a rejected command is
# a valid, attributable outcome the UI renders inline.

# ---------------------------------------------------------------------------
# stream events
#
# The provider-stream subset (ProcessingStarted … StreamError) is emitted by
# the API client while a round streams. Every other event below is host
# lifecycle/UI traffic. All of them are defined here, next to the commands and
# envelopes they cross the wire with, so there is one event vocabulary.

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
    """Emitted once per round, right after the request is sent (state 1)."""

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

# ---------------------------------------------------------------------------
# primitives



Level = Literal["info", "warn", "error", "success", "hint"]


class QueuedMessage(BaseModel):
    """A steering message waiting to be injected at a round boundary."""

    text: str
    images: list[ImageRef] | None = None


class PendingAsk(BaseModel):
    """A live ask-tool prompt (session state, answerable by any client)."""

    ask_id: str
    questions: list[str]


class PendingDrift(BaseModel):
    """A live resume-drift prompt (session state, resolvable by any client).

    Structured, not prose: clients render their own message from the trees
    and changed paths."""

    workspace: str
    last_known: str = ""
    current: str = ""
    files: list[str] = Field(default_factory=list)


class RefInfo(BaseModel):
    """One external reference bound into the sandbox: the host ``path`` and
    the workspace-relative ``mount`` the model addresses it by (normally
    ``.refs/<alias>``; nothing is created there in the worktree)."""

    path: str
    mount: str


class WorkspaceRefUpdate(BaseModel):
    """POST /v1/workspaces/refs: add/remove one workspace reference."""

    workspace: str
    path: str
    add: bool


class StatusPayload(BaseModel):
    """Everything a client needs for its status bar / steer bar / prompts."""

    state: State
    session_name: str | None = None
    workspace: str | None = None
    mode: Mode = "write"
    model_name: str = ""
    model_display_name: str = ""
    model_id: str = ""
    cot_strength: str = "medium"
    max_context: int = 128_000
    supports_vision: bool = False
    sandbox_name: str | None = None
    use_container: bool = True
    enable_tracking: bool = True
    filter_list: list[str] = Field(default_factory=list)
    # External reference paths currently bound into the sandbox.
    refs: list[RefInfo] = Field(default_factory=list)
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


class PartialToolCall(BaseModel):
    """One tool call in the round currently being generated/executed."""

    call_id: str
    name: str
    arguments: str = ""
    output: str = ""
    ok: bool | None = None
    parse_error: bool = False


class PartialRound(BaseModel):
    """The live round an attaching client must render on top of the
    snapshot messages: reasoning/content not yet persisted, plus any tool
    calls whose results are not in the history yet."""

    reasoning: str = ""
    content: str = ""
    tool_calls: list[PartialToolCall] = Field(default_factory=list)


class Snapshot(BaseModel):
    """The full state a client receives when it attaches (SSE first event)."""

    # The current compaction epoch's messages — what the model sees.
    messages: list[Message] = Field(default_factory=list)
    # Earlier epochs hidden from the model by compaction. They are still sent
    # so a human resuming the session sees the full transcript, with each
    # summary message rendered as the handoff divider.
    history: list[Message] = Field(default_factory=list)
    status: StatusPayload = Field(default_factory=StatusPayload)
    # Live round state; None when no round is streaming.
    partial_round: PartialRound | None = None


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
    # Unaccepted (pending) change totals vs the session's baseline —
    # hosted (loaded) sessions only, and only when the listing asks for
    # stats (cheap listing omits them).
    additions: int = 0
    deletions: int = 0


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


class ConfigDefaults(BaseModel):
    """PUT /config/defaults: the host owns the config file."""

    default_profile: str | None = None


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
    # False distinguishes "this session has no git tracking" from "tracking
    # is on but nothing changed" (both otherwise look like an empty diff).
    tracking: bool = False
    # Unified diff text — only populated when a specific path is requested
    # (the summary poll passes no path and must stay cheap).
    text: str = ""


# ---------------------------------------------------------------------------
# events

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




class UserMessageAdded(EventBase):
    """A user message landed in the history (sent or steered).

    This is the only way a user message reaches any client — there is no
    local echo, so the sender renders it exactly like everyone else."""

    type: Literal["user_message_added"] = "user_message_added"
    text: str
    images: list[ImageRef] | None = None
    message_id: str | None = None
    send_time: str | None = None


class StatusChanged(EventBase):
    type: Literal["status_changed"] = "status_changed"
    status: StatusPayload = Field(default_factory=StatusPayload)


class Notice(EventBase):
    type: Literal["notice"] = "notice"
    text: str
    level: Level = "info"


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


class DiffChanged(EventBase):
    """The pending-changes summary changed (a mutating tool ran, or the turn
    ended). Clients render this instead of polling ``GET /diff`` after every
    tool result — including read-only ones."""

    type: Literal["diff_changed"] = "diff_changed"
    summary: DiffInfo = Field(default_factory=DiffInfo)


class CompactionDone(EventBase):
    """A compaction finished. The counts and the model's handoff summary are
    structured so clients render the outcome without parsing prose."""

    type: Literal["compaction_done"] = "compaction_done"
    before: int = 0
    after: int = 0
    summary: str = ""
    automatic: bool = False


class RollbackApplied(EventBase):
    """An undo/retry rollback finished (history + file state).

    ``note`` is a machine-readable reason, not a sentence:
    ``reverted`` | ``no_edits`` | ``no_tracking``."""

    type: Literal["rollback_applied"] = "rollback_applied"
    scope: str = "last turn"
    reverted: int = 0
    resend: bool = False
    compaction_undone: bool = False
    note: Literal["reverted", "no_edits", "no_tracking"] = "reverted"


class DriftDetected(EventBase):
    """The session found workspace drift on resume (also carried in status)."""

    type: Literal["drift_detected"] = "drift_detected"
    pending: PendingDrift


class TurnFinished(EventBase):
    type: Literal["turn_finished"] = "turn_finished"
    elapsed: float = 0.0


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
        DiffChanged,
        CompactionDone,
        RollbackApplied,
        DriftDetected,
        TurnFinished,
        TurnEnded,
        SessionDeleted,
    ],
    Field(discriminator="type"),
]
Snapshot.model_rebuild()


EVENT_ADAPTER = TypeAdapter(Event)


# ---------------------------------------------------------------------------
# commands


class CommandBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SendMessage(CommandBase):
    type: Literal["send_message"] = "send_message"
    text: str
    images: list[ImageAttachment] | None = None


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
    mode: Mode


class SwitchModel(CommandBase):
    type: Literal["switch_model"] = "switch_model"
    name: str
    # Also make this the config's default for future sessions. Only an
    # explicit /model asks for this; a scripted switch leaves it alone.
    persist_default: bool = False


class SetSandbox(CommandBase):
    type: Literal["set_sandbox"] = "set_sandbox"
    name: str


class FilterAdd(CommandBase):
    type: Literal["filter_add"] = "filter_add"
    path: str


class FilterRemove(CommandBase):
    type: Literal["filter_remove"] = "filter_remove"
    path: str


class RefAdd(CommandBase):
    type: Literal["ref_add"] = "ref_add"
    path: str


class RefRemove(CommandBase):
    type: Literal["ref_remove"] = "ref_remove"
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
        RefAdd,
        RefRemove,
    ],
    Field(discriminator="type"),
]

class SendMessageResult(BaseModel):
    type: Literal["send_message_result"] = "send_message_result"
    started: bool = False
    queued: bool = False


class CancelResult(BaseModel):
    type: Literal["cancel_result"] = "cancel_result"
    cancelled: bool


class StartedResult(BaseModel):
    type: Literal["started_result"] = "started_result"
    started: bool = True


class DiscardedResult(BaseModel):
    type: Literal["discarded_result"] = "discarded_result"
    discarded: int
    text: str | None = None


class FlushedResult(BaseModel):
    type: Literal["flushed_result"] = "flushed_result"
    flushed: bool


class RollbackResult(BaseModel):
    type: Literal["rollback_result"] = "rollback_result"
    user_text: str | None = None
    images: list[ImageAttachment] | None = None


class ModeResult(BaseModel):
    type: Literal["mode_result"] = "mode_result"
    mode: Mode


class ModelResult(BaseModel):
    type: Literal["model_result"] = "model_result"
    model: str


class SandboxResult(BaseModel):
    type: Literal["sandbox_result"] = "sandbox_result"
    sandbox: str | None


class FilterResult(BaseModel):
    type: Literal["filter_result"] = "filter_result"
    filter_list: list[str]


class RefsResult(BaseModel):
    type: Literal["refs_result"] = "refs_result"
    refs: list[RefInfo]


class DoneResult(BaseModel):
    type: Literal["done"] = "done"


class CloseResult(BaseModel):
    closed: bool


class DeleteResult(BaseModel):
    deleted: bool


CommandPayload = Annotated[
    Union[
        SendMessageResult,
        CancelResult,
        StartedResult,
        DiscardedResult,
        FlushedResult,
        RollbackResult,
        ModeResult,
        ModelResult,
        SandboxResult,
        FilterResult,
        RefsResult,
        DoneResult,
    ],
    Field(discriminator="type"),
]


class CommandResult(BaseModel):
    """Reply to one command, delivered only to its issuer."""

    cmd_id: str = ""
    ok: bool
    error: str | None = None
    payload: CommandPayload | None = None


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


