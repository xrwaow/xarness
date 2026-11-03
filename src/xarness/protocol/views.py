"""Read models: snapshots, status, listings, config and diffs.

Pure views over session state — nothing here mutates anything. They import
the conversation models directly (Usage, Message, ToolCall), so a message
is persisted, converted to a provider request and shown to a client as one
object, with no mirrored copy to keep in sync.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_serializer, model_validator

from ..conversation import (
    Message,
    ToolStatus,
    Usage,
)
from ..images import ImageRef

State = Literal["idle", "running", "compacting"]
Mode = Literal["plan", "write"]


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
    the canonical container ``mount`` the model addresses it by (normally
    ``/tmp/refs/<alias>``; nothing is created there in the worktree)."""

    path: str
    mount: str


class WorkspaceRefUpdate(BaseModel):
    """POST /v1/workspaces/refs: add/remove one workspace reference."""

    workspace: str
    path: str
    add: bool


class SessionConfigView(BaseModel):
    """The slow-changing half of the session's client-facing state: what
    model/sandbox/mode/refs the session runs with. Emitted only when a
    setting actually changes (``ConfigChanged``), never per tool or round."""

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
    # Default False: a payload that doesn't know (e.g. the TUI's draft
    # status, which has no session yet) must not claim tracking exists.
    enable_tracking: bool = False
    filter_list: list[str] = Field(default_factory=list)
    # External reference paths currently bound into the sandbox.
    refs: list[RefInfo] = Field(default_factory=list)
    # Tool names exposed to the model (for the /tools command).
    tools: list[str] = Field(default_factory=list)


class Activity(BaseModel):
    """The fast-changing half of the session's state: what it is doing and
    what it needs from the user. This is what ``StatusChanged`` carries, so
    a per-round or per-tool update never re-sends the config view."""

    state: State = "idle"
    # Cumulative session spend and the last round's spend.
    usage_total: Usage = Field(default_factory=Usage)
    last_usage: Usage = Field(default_factory=Usage)
    queued: list[QueuedMessage] = Field(default_factory=list)
    pending_ask: PendingAsk | None = None
    pending_drift: PendingDrift | None = None
    clients_attached: int = 0
    # True when a turn finished while no client was attached; cleared on
    # attach (the status event carries it once).
    finished_while_away: bool = False


class StatusPayload(BaseModel):
    """Config view + activity in one object: the full state a client gets
    on attach (and what the TUI synthesizes for a draft chat). The two
    halves travel separately on the live stream; here they compose — and
    serialize flat, so the attach shape is one flat object either way.

    (The OpenAPI schema still shows the nested ``config``/``activity``
    fields; the wire JSON is the flat merge — see :meth:`_flatten`. The
    two field declarations exist for construction and typing only.)"""

    config: SessionConfigView = Field(default_factory=SessionConfigView)
    activity: Activity = Field(default_factory=lambda: Activity(state="idle"))

    @model_validator(mode="before")
    @classmethod
    def _split_flat(cls, data: Any) -> Any:
        """Accept the flat wire shape (and the old flat kwargs): unknown
        nesting is not needed — every field belongs to exactly one half."""
        if (isinstance(data, dict) and data
                and "config" not in data and "activity" not in data):
            return {
                "config": {
                    k: v for k, v in data.items()
                    if k in SessionConfigView.model_fields
                },
                "activity": {
                    k: v for k, v in data.items()
                    if k in Activity.model_fields
                },
            }
        return data

    @model_serializer
    def _flatten(self) -> dict[str, Any]:
        return {**self.config.model_dump(), **self.activity.model_dump()}

    # Flat passthroughs: view code reads status.mode / status.state / …
    # without naming the half it lives in.
    @property
    def state(self) -> State:
        return self.activity.state

    @property
    def session_name(self) -> str | None:
        return self.config.session_name

    @property
    def workspace(self) -> str | None:
        return self.config.workspace

    @property
    def mode(self) -> Mode:
        return self.config.mode

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def model_display_name(self) -> str:
        return self.config.model_display_name

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def cot_strength(self) -> str:
        return self.config.cot_strength

    @property
    def max_context(self) -> int:
        return self.config.max_context

    @property
    def supports_vision(self) -> bool:
        return self.config.supports_vision

    @property
    def sandbox_name(self) -> str | None:
        return self.config.sandbox_name

    @property
    def use_container(self) -> bool:
        return self.config.use_container

    @property
    def enable_tracking(self) -> bool:
        return self.config.enable_tracking

    @property
    def filter_list(self) -> list[str]:
        return self.config.filter_list

    @property
    def refs(self) -> list[RefInfo]:
        return self.config.refs

    @property
    def tools(self) -> list[str]:
        return self.config.tools

    @property
    def usage_total(self) -> Usage:
        return self.activity.usage_total

    @property
    def last_usage(self) -> Usage:
        return self.activity.last_usage

    @property
    def queued(self) -> list[QueuedMessage]:
        return self.activity.queued

    @property
    def pending_ask(self) -> PendingAsk | None:
        return self.activity.pending_ask

    @property
    def pending_drift(self) -> PendingDrift | None:
        return self.activity.pending_drift

    @property
    def clients_attached(self) -> int:
        return self.activity.clients_attached

    @property
    def finished_while_away(self) -> bool:
        return self.activity.finished_while_away


class PartialToolCall(BaseModel):
    """One tool call in the round currently being generated/executed."""

    call_id: str
    name: str
    arguments: str = ""
    content: str = ""
    status: ToolStatus | None = None


class PartialRound(BaseModel):
    """The live round an attaching client must render on top of the
    snapshot messages: reasoning/content not yet persisted, plus any tool
    calls whose results are not in the history yet.

    ONE accumulator: the controller fills the model-stream half (reasoning,
    answer, tool-call arguments) and the session folds in the tool-exec
    events it emits itself."""

    reasoning: str = ""
    content: str = ""
    tool_calls: list[PartialToolCall] = Field(default_factory=list)

    def find_call(self, call_id: str) -> PartialToolCall | None:
        return next(
            (call for call in self.tool_calls if call.call_id == call_id), None,
        )

    def __bool__(self) -> bool:
        return bool(self.reasoning or self.content or self.tool_calls)


class Snapshot(BaseModel):
    """The full state a client receives when it attaches (SSE first event)."""

    # The visible slice — what the model sees.
    messages: list[Message] = Field(default_factory=list)
    # Messages the visible slice hides (compacted away). They are still sent
    # so a human resuming the session sees the full transcript, with each
    # summary message rendered as the handoff divider.
    history: list[Message] = Field(default_factory=list)
    status: StatusPayload = Field(default_factory=StatusPayload)
    # Live round state; None when no round is streaming.
    partial_round: PartialRound | None = None


class ServerInfo(BaseModel):
    """GET /health response."""

    version: str
    # No default (it would drift from PROTOCOL_VERSION); the host passes it.
    protocol_version: int
    pid: int
    uptime_seconds: float
    persistent: bool = False
    unix_socket: str | None = None


class SessionSummary(BaseModel):
    """One row of GET /sessions (active or saved)."""

    name: str
    workspace: str | None = None
    # "waiting" and "saved" are host-level states, not conversation states.
    state: Literal["idle", "running", "compacting", "waiting", "saved"] = "saved"
    active: bool = False
    clients_attached: int = 0
    model: str | None = None
    sandbox: str | None = None
    last_user_message_at: str | None = None
    message_count: int = 0
    tracking: bool = False
    additions: int = 0
    deletions: int = 0


class ServerStats(BaseModel):
    """GET /server/stats: the dashboard's data."""

    host: ServerInfo
    started_at: str
    clients_attached: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    sessions: list[SessionSummary] = Field(default_factory=list)


class ModelInfo(BaseModel):
    """One configured model profile (names and non-secret fields only)."""

    name: str
    display_name: str
    cot_strength: str
    max_context: int
    supports_vision: bool
    auto_compact: bool


class SandboxInfo(BaseModel):
    """One configured sandbox profile."""

    name: str


class ConfigInfo(BaseModel):
    """GET /config: names and non-secret fields for the client's pickers."""

    models: list[ModelInfo] = Field(default_factory=list)
    sandboxes: list[SandboxInfo] = Field(default_factory=list)
    presets: list[str] = Field(default_factory=list)
    # What a fresh session starts on (from the default preset), so the
    # client's draft status bar shows real values before any session exists.
    default_model: str | None = None
    default_model_display_name: str | None = None
    default_sandbox: str | None = None


class ConfigDefaults(BaseModel):
    """PUT /config/defaults: the host owns the config file."""

    default_profile: str


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
    source: Literal["agent", "drift"] = "agent"


class DiffInfo(BaseModel):
    files: list[DiffFileInfo] = Field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    tracking: bool = False
    text: str = ""


Snapshot.model_rebuild()
