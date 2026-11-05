"""Wire protocol models shared by the backend, the TUI client and the API.

One source of truth for every event and command that crosses a process
boundary. Pydantic models with a ``type`` discriminator; FastAPI generates
the OpenAPI schema from these directly, so the docs can never drift from
the code.

The package is split by role — :mod:`.events` (the SSE stream),
:mod:`.views` (read models), :mod:`.commands` (write models) — and this
root re-exports all of it, so clients keep their flat
``from xarness.protocol import X`` imports.
"""

from ..conversation import (  # noqa: F401 - re-exported for clients
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
    ToolStatus,
    Usage,
    UserMessage,
    image_refs,
)
from .commands import (  # noqa: F401
    Accept,
    AnswerAsk,
    Cancel,
    Command,
    CommandBase,
    CommandEnvelope,
    CommandResult,
    Compact,
    CreateSessionRequest,
    DeleteResult,
    DiscardQueued,
    Filter,
    FlushQueued,
    Ref,
    Reject,
    ResolveDrift,
    Rollback,
    SendMessage,
    SessionUpdate,
    SetMode,
    SetSandbox,
    SwitchModel,
)
from .events import (  # noqa: F401
    EVENT_ADAPTER,
    PROTOCOL_VERSION,
    AskResolved,
    CheckpointsChanged,
    ContentDelta,
    DiffChanged,
    DriftResolved,
    Event,
    EventBase,
    HistoryReason,
    HistoryReplaced,
    HostEvent,
    Level,
    MessageAdded,
    Notice,
    ProcessingStarted,
    ReasoningDelta,
    RoundComplete,
    SessionDeleted,
    StatusChanged,
    StreamError,
    StreamEvent,
    ToolCallArgumentsDelta,
    ToolCallArgumentsDone,
    ToolCallStarted,
    ToolExecStarted,
    ToolFinished,
    ToolOutputDelta,
    TurnEnded,
)
from .views import (  # noqa: F401
    ConfigDefaults,
    ConfigInfo,
    DiffFileInfo,
    DiffInfo,
    Mode,
    ModelInfo,
    PartialRound,
    PartialToolCall,
    PendingAsk,
    PendingDrift,
    QueuedMessage,
    RefInfo,
    SandboxInfo,
    ServerInfo,
    ServerStats,
    SessionSummary,
    Snapshot,
    State,
    Status,
    WorkspaceInfo,
    WorkspaceRefUpdate,
)
