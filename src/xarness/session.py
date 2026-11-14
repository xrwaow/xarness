"""Headless chat session: all conversation state and logic, no UI.

One :class:`Session` owns the conversation, the provider client, the tool
registry, the sandbox, git tracking, the steering queue and the run state.
It runs in a host process and outlives every client: a turn keeps going
after the TUI that started it closes, and any number of clients attach to
the same event stream.

**One round, one home.** :meth:`_stream_round` streams exactly one model
response and folds it into the history; :meth:`_run_turn` is the loop
around it (execute the tool calls, inject steering, compact, repeat). Both
live here rather than in a separate controller object, so the profile, the
tools and the live-round state have exactly one owner.

**Single writer.** All mutation flows through one per-session command lock:
HTTP handlers ``await session.execute(...)`` — the asyncio.Lock (FIFO)
serializes commands, and the result goes only to the issuer. Pure
read-only queries (snapshots, diffs) bypass the lock; they are safe
because only the session mutates that state.

**Fan-out.** :meth:`_emit` pushes every event to every subscriber queue. A
client attaches through :meth:`subscribe`, which builds the snapshot
synchronously (no await between building it and registering the subscriber,
so no event can slip between the two) and then replays the live stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field, fields
from functools import partial
from itertools import groupby
from pathlib import Path
from typing import Any

from . import compaction, protocol, session_store
from .client import ChatClient, StreamClient
from .compaction import CHARS_PER_TOKEN
from .config import (
    ConfigError,
    McpServerSettings,
    McpSettings,
    ProviderProfile,
    SandboxSettings,
    load_sectioned,
    resolve_api_key,
    save_mcp_settings,
    validate_filter_path,
)
from .errors import RefError, SessionError
from .conversation import (
    AssistantMessage,
    Conversation,
    Message,
    RollbackPlan,
    Timing,
    ToolCall,
    ToolMessage,
    TurnStats,
    Usage,
    user_message,
)
from .gitwork import (
    GitWorktreeError,
    RevertConflict,
    TurnCheckpoint,
    tree_diff_stat,
    write_blob,
)
from .images import ImageRef
from .mcp import McpError, McpManager
from .prompts import system_prompt
from .protocol import (
    Command,
    CommandResult,
    ContentDelta,
    Event,
    HistoryReason,
    Mode,
    PartialRound,
    ReasoningDelta,
    RoundComplete,
    SessionSummary,
    Snapshot,
    State,
    StreamEvent,
    ToolCallArgumentsDelta,
    ToolCallArgumentsDone,
    ToolCallStarted,
)
from .sandbox import (
    SandboxConfig,
    SandboxRuntime,
    SandboxSession,
    SandboxUnavailable,
    ref_infos,
)
from .session_store import StoredSession
from .tools import DEFAULT_TIMEOUT, build_registry
from .util import utcnow_iso as _utcnow
from .workspace import (
    _EXPLICIT_WRITES,
    WorkspaceState,
    change_workspace_ref,
    save_workspace_refs,
)

# Tools that can change the workspace; only these trigger a DiffChanged
# event (read_file/search tools cannot).
MUTATING_TOOLS = frozenset({"write_file", "edit_file", "run_bash"})

# Trailing debounce for DiffChanged after a mutating tool (one in flight).
DIFF_DEBOUNCE_SECONDS = 0.25

# Commands that rewrite conversation/workspace state are rejected while a
# turn or compaction is running.
IDLE_ONLY = (
    protocol.Rollback, protocol.Compact, protocol.SwitchModel,
    protocol.SetSandbox, protocol.Filter, protocol.Ref,
    protocol.SetMcpServerEnabled, protocol.SetMcpToolEnabled,
    protocol.AddMcpServer,
)

# Commands that only touch a flag or a future: they skip the command lock, so
# a slow /accept (up to seconds of git) never delays a cancel or an answer.
# SetViewState lands here too: it only stores a view choice and schedules the
# debounced persist, so nothing may queue behind /accept.
LOCK_FREE = (protocol.Cancel, protocol.AnswerAsk, protocol.SetViewState)


@dataclass(slots=True)
class _QueuedSteer:
    """One steering message waiting for a round boundary."""

    text: str
    images: list[ImageRef] | None
    send_time: str


LOG = logging.getLogger(__name__)


def _alive(task: asyncio.Task[None] | None) -> bool:
    """Whether a session still has a task doing the work it reports."""
    return task is not None and not task.done()


@dataclass(slots=True)
class SessionParts:
    """Everything :class:`Session` needs, resolved by the host.

    Typed instead of a ``**``-splatted dict, so a mismatch with the
    ``Session`` constructor is a type error, not a runtime surprise. Field
    names match Session's keyword parameters, so ``Session(name=...,
    **parts.session_kwargs(name))`` constructs it."""

    workspace: Path
    profile: ProviderProfile
    profile_name: str
    api_key: str | None
    config_path: Path | None
    sandbox: SandboxRuntime
    workspace_state: WorkspaceState | None
    tool_output_limit: int | None
    # The model a subagent of this session runs (a model key), resolved from
    # the preset's `subagent_model`.
    subagent_model: str | None = None
    # MCP servers this session may expose (from the config's `mcp` section).
    mcp: McpSettings = field(default_factory=McpSettings)
    # Workspace reference paths (shared per workspace, managed with /ref).
    workspace_refs: list[str] = field(default_factory=list)
    # Called after a workspace event so every agent in the repo refreshes
    # its diff; injected by the host (None outside it).
    notify_workspace: Callable[[str | None], None] | None = None
    # Starts one subagent session for this one and returns its final message;
    # injected by the host, and None for a subagent itself (no nesting).
    spawn_subagent: Callable[["Session", str, str], Awaitable[str]] | None = None
    # Receives a child session's cumulative usage after each round, so the
    # parent's tool row can show it live.
    report_usage: Callable[[Usage], None] | None = None

    def session_kwargs(self, name: str) -> dict[str, Any]:
        """Keyword arguments for ``Session(name=..., **kwargs)`` (shallow:
        nested dataclasses like the sandbox config must stay objects)."""
        return {"name": name, **{f.name: getattr(self, f.name) for f in fields(self)}}


class Session:
    """One chat session: state, turn loop, tools, git tracking, fan-out."""

    def __init__(
        self,
        *,
        name: str,
        workspace: Path,
        profile: ProviderProfile,
        profile_name: str | None,
        api_key: str | None,
        conversation: Conversation | None = None,
        sandbox: SandboxRuntime | None = None,
        workspace_state: WorkspaceState | None = None,
        mode: Mode = "write",
        tool_output_limit: int | None = None,
        client: StreamClient | None = None,
        config_path: Path | None = None,
        no_save: bool = False,
        parent_id: str | None = None,
        workspace_refs: list[str] | None = None,
        notify_workspace: Callable[[str | None], None] | None = None,
        subagent_model: str | None = None,
        spawn_subagent: Callable[["Session", str, str], Awaitable[str]] | None = None,
        report_usage: Callable[[Usage], None] | None = None,
        view: protocol.ViewState | None = None,
        mcp: McpSettings | None = None,
        mcp_manager: McpManager | None = None,
    ) -> None:
        self.name = name
        # Set only for a subagent's session: the session that spawned it.
        self.parent_id = parent_id
        self.workspace = workspace
        self.profile = profile
        self.profile_name = profile_name
        self.api_key = api_key
        self.config_path = config_path
        self.sandbox = sandbox or SandboxRuntime()
        self.sandbox.set_mode(mode)
        self.workspace_state = workspace_state
        self.mode: Mode = mode
        # The clients' saved view choices for this session (the /diff panel's
        # scope chip); a saved session restores them on load.
        self.view = view or protocol.ViewState()
        # The model a subagent of this session runs (see Host.spawn_subagent).
        self.subagent_model = subagent_model
        # Bound here, not in the host, which builds the parts before this
        # session exists. A subagent has nobody to ask and cannot nest, so it
        # gets neither capability — enforced on this one invariant, because a
        # reloaded child's parts DO carry the host's spawn callable.
        self._spawn_subagent = (
            None if spawn_subagent is None or parent_id is not None
            else lambda prompt, mode_arg: spawn_subagent(self, prompt, mode_arg)
        )
        self._ask_callback = None if parent_id is not None else self._ask_user
        self._report_usage = report_usage
        # MCP servers are host-side subprocesses. A subagent shares its
        # parent's manager (and therefore its already-running processes);
        # only the owner closes them.
        self._owns_mcp = mcp_manager is None
        self.mcp = mcp_manager or McpManager(mcp or McpSettings(), workspace)
        self.tool_registry = build_registry(
            self.sandbox.config, self.sandbox.shell, mode=self.mode,
            ask_callback=self._ask_callback,
            default_timeout=self._tool_timeout(),
            spawn_subagent=self._spawn_subagent,
            mcp=self.mcp,
        )
        if tool_output_limit is not None:
            self.tool_registry.max_output_chars = tool_output_limit

        # The conversation, its provider client and the live-round state are
        # the session's own: one owner, no second copy to keep in step.
        self.conversation = conversation or Conversation()
        self._client = client or ChatClient(profile, api_key)
        # Cumulative session spend, folded together from every round's usage
        # — including compaction's own summarization round and everything an
        # /undo later removes. Monotonic: it counts everything ever spent.
        self.usage_total = Usage(input_tokens=0, output_tokens=0)
        # Usage of the most recent round, kept for the status payload
        # (instead of re-scanning the message list on every status emit).
        self.last_usage = Usage(input_tokens=0, output_tokens=0)
        # The live round currently streaming — ONE accumulator for its
        # reasoning, answer text and tool calls, read by the snapshot, the
        # cancel path and the persist path.
        self.partial_round = PartialRound()
        # The system prompt is derived at request time (never persisted).
        self.system_prompt: str | None = None
        self._refresh_system_prompt()

        # Run state. Nothing stores "running"/"idle": the tasks below are
        # the state (see the property), so a task that dies without running
        # its teardown can never leave a session claiming work it stopped.
        self._compacting = False
        self._turn_task: asyncio.Task[None] | None = None
        self._compact_task: asyncio.Task[None] | None = None
        # The turn task currently inside its end-of-turn teardown. It is
        # NOT cancellable any more: a CancelledError delivered in the
        # teardown would skip the state flip to idle and strand the
        # session reporting a turn that no longer exists.
        self._finishing_turn: asyncio.Task[None] | None = None
        # Steering queue — send_time is when the client sent it, not when it
        # is injected.
        self._queued: list[_QueuedSteer] = []
        # Pending ask-tool prompt. Any client may answer; first answer wins.
        self._pending_ask: protocol.PendingAsk | None = None
        self._pending_ask_future: asyncio.Future[str | None] | None = None
        # Workspace reference paths (kept separate from the sandbox profile's
        # own auto_include_refs; the effective list is computed when the
        # sandbox is built).
        self.workspace_refs: list[str] = list(workspace_refs or [])
        # no_session: the client asked for an unsaved conversation — _persist
        # becomes a no-op and nothing is ever written to the session store.
        self.no_save = no_save
        self._notify_workspace_cb = notify_workspace
        # Content-addressed image blobs for this session's messages.
        self.images = session_store.SessionImageStore(
            self.name, persist=not no_save,
        )

        # Debounced background persistence (see _persist). Appends are the
        # common case; a full rewrite happens after anything mutated the
        # already-persisted prefix (truncate, turn stamping, accept).
        self._persist_dirty = False
        self._persist_task: asyncio.Task[None] | None = None
        self._diff_task: asyncio.Task[None] | None = None
        self._diff_dirty = False
        # The tool call currently executing (if any). A session whose model
        # has handed off to a tool is waiting on it, not generating: /sessions
        # shows that as "waiting" (a subagent call can run for minutes).
        # ``_executing_calls`` counts parallel subagents so the first one to
        # finish does not clear the status while its sibling still runs.
        self._executing_tool: str | None = None
        self._executing_calls = 0

        # Fan-out.
        self._subscribers: list[asyncio.Queue[Event]] = []

        # Command serialization (FIFO).
        self._cmd_lock = asyncio.Lock()
        self._closed = False

    # ------------------------------------------------------------------
    # construction

    @classmethod
    def load(
        cls,
        name: str,
        *,
        parts: SessionParts,
        state: "StoredSession",
    ) -> "Session":
        """Host a saved session: load its conversation and rebind tracking.

        ``state`` is the already-loaded StoredSession (the host loads it in a
        worker thread, off the event loop)."""
        session = cls(
            **parts.session_kwargs(name),
            conversation=state.conversation,
            mode=state.mode or "write",
            parent_id=state.parent_id,
            view=state.view,
        )
        # Restore the persisted token counters so the client's footer shows
        # real in/out/ctx immediately instead of 0 until the next round.
        session.usage_total = state.usage_total
        session.last_usage = state.last_usage
        # A loaded session can never have a running turn: a crash (or a
        # kill) left any open ones behind. Close them all — a turn left
        # "open" forever makes open_turn() return stale results. Tracked
        # turns keep the before tree as their after tree (no edits
        # attributed).
        session.conversation.close_crashed_turns()
        # The loaded history is on disk by definition: the first write must
        # not rewrite it.
        session.conversation.mark_clean()
        # A session saved mid-turn can carry tool calls with no result; give
        # them synthetic answers so the next request is valid.
        session._repair_tool_calls()
        return session

    # ------------------------------------------------------------------
    # MCP

    async def prepare_mcp(self) -> None:
        """Connect the configured MCP servers once, then rebuild the tools.

        Called by the host before a session can be used, and again from
        :meth:`tools_info` so a client that attaches to an old session sees
        live availability rather than an empty listing."""
        if self.mcp.started or self._closed:
            return
        await self.mcp.start()
        self._rebuild_tools()

    async def tools_info(self) -> protocol.ToolsInfo:
        """The effective tools (built-ins plus enabled MCP tools) and every
        configured MCP server, failed and disabled ones included.

        Server processes are started lazily here too, so opening /tools is
        enough to make a newly configured server available."""
        await self.prepare_mcp()
        return protocol.ToolsInfo(
            tools=[
                protocol.ToolInfo(
                    name=tool.name, source=tool.source,
                    description=tool.description,
                )
                for tool in self.tool_registry.tools()
            ],
            servers=self.mcp.statuses(),
        )

    def _mcp_state(self, name: str) -> str:
        """One server's status word ("connected", "unavailable", …, or
        "configured" for a name the manager does not know)."""
        return next(
            (server.status for server in self.mcp.statuses()
             if server.name == name),
            "configured",
        )

    async def _apply_mcp(self, change: Callable[[], Awaitable[None]]) -> None:
        """Run one MCP change: persist it, rebuild the tools, push status.

        The tool set is derived from the connected servers, so every change
        goes through here rather than each handler repeating the three
        steps (and forgetting one)."""
        try:
            await change()
        except McpError as exc:
            raise SessionError(str(exc)) from None
        self._save_mcp_settings()
        self._rebuild_tools()
        self._emit_status()

    async def _cmd_set_mcp_server_enabled(
        self, command: protocol.SetMcpServerEnabled,
    ) -> CommandResult:
        await self._apply_mcp(lambda: self.mcp.set_server_enabled(
            command.name, command.enabled,
        ))
        self._emit_notice(f"mcp {command.name}: {self._mcp_state(command.name)}")
        return CommandResult()

    async def _cmd_set_mcp_tool_enabled(
        self, command: protocol.SetMcpToolEnabled,
    ) -> CommandResult:
        await self._apply_mcp(lambda: self.mcp.set_tool_enabled(
            command.server, command.tool, command.enabled,
        ))
        return CommandResult()

    async def _cmd_add_mcp_server(
        self, command: protocol.AddMcpServer,
    ) -> CommandResult:
        try:
            settings = McpServerSettings(
                **command.model_dump(exclude={"type", "name"}),
            )
        except ValueError as exc:
            raise SessionError(f"invalid MCP server: {exc}") from None
        await self._apply_mcp(lambda: self.mcp.add_server(command.name, settings))
        state = self._mcp_state(command.name)
        self._emit_notice(
            f"mcp {command.name}: {state}" if state == "connected" else
            f"mcp {command.name}: configured ({state})"
        )
        return CommandResult()

    def _save_mcp_settings(self) -> None:
        """Persist toggles/additions in the host's config file.

        The live session keeps working when the file cannot be written; the
        user gets a notice instead of losing the running change."""
        if self.config_path is None:
            return
        try:
            save_mcp_settings(self.config_path, self.mcp.settings)
        except (ConfigError, OSError) as exc:
            self._emit_notice(
                f"mcp settings changed for this session only: {exc}",
                level="warn",
            )

    # ------------------------------------------------------------------
    # lifecycle

    async def stop(self) -> None:
        """Cancel a running turn or compaction and flush pending writes."""
        self._closed = True
        await self._cancel_and_wait(self._turn_task)
        await self._cancel_and_wait(self._compact_task)
        self._compact_task = None
        self._compacting = False
        await self._cancel_and_wait(self._diff_task)
        self._diff_task = None
        await self._flush_persist()
        # Close the sandbox shell here (not only on host eviction) so every
        # path — explicit unload, delete, eviction, shutdown — releases it.
        await self.sandbox.close()
        # Only the owner closes MCP processes; a subagent shares its
        # parent's manager and must leave it running.
        if self._owns_mcp:
            await self.mcp.close()
        # Release the provider connection pool.
        client = self._client
        if client is not None and hasattr(client, "aclose"):
            await client.aclose()

    @property
    def is_active(self) -> bool:
        """True while running, compacting, waiting on an ask, or holding an
        open event stream. Attached clients alone do not keep the host up."""
        return (
            self.state != "idle"
            or self._pending_ask is not None
            or bool(self._subscribers)
        )

    async def shutdown(self, notice: str | None = None) -> None:
        """Close the session for good: cancel a running turn, stop the loops
        and sandbox shell, emit the closing notices, and release every
        subscriber. Used by the host's delete/unload paths."""
        # stop() already flushed the pending persist.
        await self.stop()
        if notice:
            self._emit(protocol.Notice(text=notice, level="info"))
        self._emit(protocol.SessionDeleted(name=self.name))
        for queue in list(self._subscribers):
            self.unsubscribe(queue)

    @property
    def state(self) -> "State":
        """The activity state, derived from the work that causes it: a
        session can never claim a turn nobody is running."""
        if self._compacting:
            return "compacting"
        return "running" if _alive(self._turn_task) else "idle"

    @property
    def pending_ask(self) -> "protocol.PendingAsk | None":
        return self._pending_ask

    def summary(self) -> SessionSummary:
        """The list row for this hosted session."""
        state = self.state
        # A pending ask (the user owes it an answer) and a tool call in
        # flight both block the turn itself, so they read as their own
        # state in every activity state: "ask" for the first, "waiting"
        # for the second (a subagent call can run for minutes).
        if self.pending_ask is not None:
            state = "ask"
        elif self._executing_tool is not None:
            state = "waiting"
        # Active derives from the same predicate the grace loop uses — a
        # hosted but fully idle session (no stream, no ask, no running
        # turn) is not active; it is released and listed as saved.
        active = self.is_active
        return SessionSummary(
            name=self.name, workspace=str(self.workspace),
            state=state if active else "saved",
            active=active, clients_attached=self.clients_attached,
            model=self.profile_name, sandbox=self.sandbox.name,
            last_user_message_at=self.last_user_message_at(),
            message_count=len(self.conversation.visible_messages()),
            parent_id=self.parent_id,
            tracking=self.workspace_state is not None,
        )

    # ------------------------------------------------------------------
    # fan-out

    def _emit(self, event: Event) -> Event:
        """Fold live tool-exec state into the snapshot and push the event."""
        self._track_partial(event)
        for queue in self._subscribers:
            queue.put_nowait(event)
        return event

    def _reset_partial(self) -> None:
        """Drop the live round state (turn start, turn end, round boundary)."""
        self.partial_round = PartialRound()

    def _track_partial(self, event: Event) -> None:
        """Fold the tool-execution events the session itself emits into the
        live round state (the model-stream half is folded in _stream_round).

        This is what lets an attach mid-tool-call render the call and its
        output so far without replaying an unbounded event log."""
        if event.type in ("tool_output_delta", "tool_finished"):
            self.partial_round.apply(event)

    def subscribe(self) -> tuple[Snapshot, asyncio.Queue[Event]]:
        """Build the attach snapshot and register a subscriber.

        Synchronous on purpose: no await between building the snapshot and
        registering the queue, so the stream cannot move in between (no
        gap, no duplicate)."""
        snapshot = self.snapshot()
        queue: asyncio.Queue[Event] = asyncio.Queue()
        self._subscribers.append(queue)
        return snapshot, queue

    def unsubscribe(self, queue: asyncio.Queue[Event]) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)


    @property
    def clients_attached(self) -> int:
        return len(self._subscribers)

    # ------------------------------------------------------------------
    # snapshots and status

    def status(self) -> protocol.Status:
        """The whole client-facing state: activity plus config."""
        return protocol.Status(
            session_name=self.name,
            workspace=str(self.workspace),
            mode=self.mode,
            model_name=self.profile_name or "",
            model_display_name=self.profile.display_name or self.profile.model_id,
            model_id=self.profile.model_id,
            cot_strength=self.profile.cot_strength.value,
            max_context=self.profile.max_context,
            sandbox_name=self.sandbox.name,
            filter_list=(
                list(self.sandbox.settings.filter_list)
                if self.sandbox.settings is not None else []
            ),
            refs=self._ref_infos(),
            use_sandbox=self.sandbox.use_sandbox,
            enable_tracking=self.workspace_state is not None,
            supports_vision=self.profile.supports_vision,
            tools=[tool.name for tool in self.tool_registry.tools()],
            state=self.state,
            usage_total=self.usage_total,
            last_usage=self.last_usage,
            queued=[
                protocol.QueuedMessage(text=item.text, images=item.images)
                for item in self._queued
            ],
            pending_ask=self.pending_ask,
            clients_attached=self.clients_attached,
            view=self.view,
        )

    def quit_verdict(self) -> protocol.QuitVerdict:
        """What a client about to disconnect must do (the quit menu's input).

        Asking is only meaningful when the closing client is the last one
        watching work that is still in flight: otherwise the turn keeps
        streaming to someone else and quitting is just one viewer leaving.
        The requester is assumed to hold one of the attached streams, so one
        is subtracted from the count."""
        others = max(0, self.clients_attached - 1)
        return protocol.QuitVerdict(
            ask=(self.state in ("running", "compacting")
                 or self._pending_ask is not None) and others == 0,
            state=self.state,
            pending_ask=self._pending_ask is not None,
            other_clients=others,
        )

    def snapshot(self) -> Snapshot:
        """Full client-facing state (synchronous; see :meth:`subscribe`).
        Messages are pydantic models, so they go over the wire as-is."""
        conversation = self.conversation
        return Snapshot(
            messages=list(conversation.messages),
            status=self.status(),
            partial_round=self.partial_round or None,
        )

    def last_user_message_at(self) -> str | None:
        """Latest user send_time (see Conversation.last_user_message_at)."""
        return self.conversation.last_user_message_at()

    # ------------------------------------------------------------------
    # commands: serialized by the per-session lock

    async def execute(self, command: Command) -> CommandResult:
        """Run one command under the session lock and return its result.

        A rejected command raises :class:`SessionError` (the API's 400); the
        client only retries on a transport error, where the request was never
        delivered, so there is nothing to dedupe."""
        if isinstance(command, LOCK_FREE):
            # Only touches a flag or a future: no lock, so a slow /accept
            # (seconds of git) can never delay a cancel or an answer.
            return await self._dispatch(command)
        async with self._cmd_lock:
            return await self._dispatch(command)

    async def _dispatch(self, command: Command) -> CommandResult:
        if isinstance(command, IDLE_ONLY) and self.state != "idle":
            raise SessionError(
                "wait for the current turn or compaction to finish first"
            )
        match command:
            case protocol.SendMessage():
                return await self._cmd_send(command)
            case protocol.Cancel():
                return self._cmd_cancel()
            case protocol.AnswerAsk():
                return self._cmd_answer_ask(command)
            case protocol.DiscardQueued():
                return self._cmd_discard_queued()
            case protocol.FlushQueued():
                return self._cmd_flush_queued()
            case protocol.Rollback():
                return await self._cmd_rollback(command)
            case protocol.Compact():
                return await self._cmd_compact()
            case protocol.Accept():
                return await self._cmd_accept()
            case protocol.Reject():
                return await self._cmd_reject()
            case protocol.SetMode():
                return self._cmd_set_mode(command)
            case protocol.SetViewState():
                return self._cmd_set_view_state(command)
            case protocol.SwitchModel():
                return await self._cmd_switch_model(command)
            case protocol.SetSandbox():
                return await self._cmd_set_sandbox(command)
            case protocol.Filter():
                return await self._cmd_filter(command.path, command.add)
            case protocol.Ref():
                return await self._cmd_ref(command.path, command.add)
            case protocol.SetMcpServerEnabled():
                return await self._cmd_set_mcp_server_enabled(command)
            case protocol.SetMcpToolEnabled():
                return await self._cmd_set_mcp_tool_enabled(command)
            case protocol.AddMcpServer():
                return await self._cmd_add_mcp_server(command)
            case _:
                raise SessionError(f"unsupported command: {type(command).__name__}")

    # ------------------------------------------------------------------
    # state

    def _set_compacting(self, compacting: bool) -> None:
        """Enter/leave the compacting state (a flag: the tasks are the rest
        of the state) and push the new status."""
        if self._compacting == compacting:
            return
        self._compacting = compacting
        self._emit_status()

    async def wait_idle(self) -> None:
        """Wait for the running turn and compaction (if any) to finish,
        including their teardown and the queued message it starts."""
        while alive := {
            task for task in (self._turn_task, self._compact_task)
            if _alive(task)
        }:
            await asyncio.wait(alive)

    def _emit_status(self) -> None:
        """Push the current status to every client."""
        self._emit(protocol.StatusChanged(status=self.status()))

    def _emit_notice(self, text: str, level: protocol.Level = "info") -> None:
        self._emit(protocol.Notice(text=text, level=level))

    def _commit_history(
        self, reason: HistoryReason = "rollback", **fields: Any,
    ) -> None:
        """Persist and broadcast the rewritten history in one step."""
        self._persist()
        self._emit(protocol.HistoryReplaced(
            snapshot=self.snapshot(), reason=reason, **fields,
        ))

    def _schedule_diff_changed(self) -> None:
        """Coalesce DiffChanged emissions: a dirty flag plus one loop task —
        a mutation arriving while a diff computes re-arms the loop instead of
        being dropped."""
        self._diff_dirty = True
        if self._diff_task is None or self._diff_task.done():
            self._diff_task = asyncio.get_running_loop().create_task(
                self._diff_loop()
            )

    async def _diff_loop(self) -> None:
        try:
            while self._diff_dirty:
                self._diff_dirty = False
                await asyncio.sleep(DIFF_DEBOUNCE_SECONDS)
                await self._emit_diff_changed()
        except asyncio.CancelledError:
            raise
        finally:
            self._diff_task = None

    async def _emit_diff_changed(self) -> None:
        """Push the pending-changes summary to attached clients.

        Only when someone is attached (the computation snapshots the
        workspace) and git tracking exists. Mutating tools and turn end call
        this; clients no longer poll after every tool result."""
        if self.workspace_state is None or not self._subscribers:
            return
        try:
            summary = await self.diff_info(None)
        except SessionError:
            return
        self._emit(protocol.DiffChanged(summary=summary))

    # ------------------------------------------------------------------
    # commands

    async def _cmd_send(self, command: protocol.SendMessage) -> CommandResult:
        # Images must exist in the store before send: a missing blob would
        # degrade to a text token on the wire, silently losing the image.
        for ref in command.images or []:
            if self.images.load(ref.id) is None:
                raise SessionError(
                    f"image {ref.id[:12]}… ({ref.name!r}) was never uploaded "
                    "— PUT it to /sessions/{name}/images/{id} first"
                )
        if self.state != "idle":
            # Steering: queued until the next round boundary (or turn end).
            self._queued.append(_QueuedSteer(
                command.text, command.images or None, _utcnow(),
            ))
            self._emit_status()
            return CommandResult()
        await self._start_turn(command.text, command.images or None)
        return CommandResult()

    async def _start_turn(self, text: str, images: list[ImageRef] | None) -> None:
        task = asyncio.get_running_loop().create_task(
            self._run_turn(text, images)
        )
        # Set synchronously, BEFORE the task can run: the lock can dispatch
        # the next queued command before the turn task gets its first slot,
        # and an idle state would let a back-to-back send start a second turn
        # instead of steering the first.
        self._turn_task = task
        task.add_done_callback(self._on_turn_done)
        self._emit_status()
        # Yield once so the turn task actually starts: a task cancelled
        # before its first step never runs its cleanup.
        await asyncio.sleep(0)

    def _cancellable(self, task: asyncio.Task[None] | None) -> bool:
        """Whether cancelling ``task`` is safe.

        Not while it is already unwinding (a second ``CancelledError`` lands
        in its cleanup and aborts the state flip), nor while it is in
        ``_close_out_turn`` — which runs after a *normal* turn end and so has
        no pending cancellation to detect."""
        return (
            _alive(task)
            and task is not self._finishing_turn
            and not task.cancelling()
        )

    async def _cancel_and_wait(self, task: asyncio.Task[None] | None) -> None:
        """Cancel a task if it is cancellable, then let it finish unwinding."""
        if self._cancellable(task):
            task.cancel()
        if _alive(task):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    def _cancel_work(self) -> None:
        """Cancel the running turn and/or compaction (if any).

        Their cleanup is what starts the next queued message (or leaves the
        session idle), so both cancel paths are just this."""
        for task in (self._turn_task, self._compact_task):
            if self._cancellable(task):
                task.cancel()

    def _on_turn_done(self, task: asyncio.Task[None]) -> None:
        """Report a turn whose task died inside its own teardown (an
        exception, or a cancel before its first step): nothing else would
        announce an end the clients are still waiting for."""
        if self._turn_task is not task:
            return  # the normal teardown already ended the turn
        self._turn_task = None
        self._reset_partial()
        self._emit_status()
        self._emit(protocol.TurnEnded(outcome="failed"))
        self._emit_notice(
            "the turn ended unexpectedly and was reset to idle", "error",
        )

    def _on_compact_done(self, task: asyncio.Task[None]) -> None:
        """Clear the compacting flag a task that died before its first step
        never cleared — a stuck session is never idle again."""
        if self._compact_task is task:
            self._compact_task = None
            self._set_compacting(False)

    def _cmd_cancel(self) -> CommandResult:
        self._cancel_work()
        return CommandResult()

    def _cmd_answer_ask(self, command: protocol.AnswerAsk) -> CommandResult:
        pending = self._pending_ask
        future = self._pending_ask_future
        if pending is None or future is None or pending.ask_id != command.ask_id:
            raise SessionError("no ask is pending with that id")
        if future.done():
            # First answer wins; later ones are rejected.
            raise SessionError("the ask was already answered")
        future.set_result(command.answers)
        self._emit(protocol.AskResolved(
            ask_id=command.ask_id, answers=command.answers,
        ))
        return CommandResult()

    def _cmd_flush_queued(self) -> CommandResult:
        """Send the oldest queued message now: cancel the running turn or
        compaction; the cleanup starts the queued message immediately."""
        if self.state == "idle" or not self._queued:
            return CommandResult()
        self._cancel_work()
        return CommandResult()

    def _cmd_discard_queued(self) -> CommandResult:
        """Drop every steering message waiting for a round boundary."""
        if self._queued:
            self._queued.clear()
            self._emit_status()
        return CommandResult()

    async def _cmd_rollback(self, command: protocol.Rollback) -> CommandResult:
        """/undo (drop the last turn, put its user message back in the
        input), /retry (drop it and resend it), or click-to-undo to an
        earlier message."""
        label = "retry" if command.resend else "undo"
        if command.message_id is not None:
            message = self.conversation.message_by_id(command.message_id)
            if message is None:
                raise SessionError("no user message with that id in this session")
            plan = self.conversation.rollback_plan_to(message)
            scope = "messages from here on"
        else:
            plan = self.conversation.rollback_plan()
            scope = "last turn"
        if plan is None:
            self._emit_notice(f"nothing to {label}")
            return CommandResult()
        return await self._apply_rollback(plan, resend=command.resend, scope=scope)

    async def _apply_rollback(
        self, plan: RollbackPlan, resend: bool, scope: str = "last turn",
    ) -> protocol.CommandResult:
        """Carry out a rollback plan: rewrite the history, revert the files,
        broadcast HistoryReplaced, and report what happened."""
        if not plan.compaction:
            label = "retry" if resend else "undo"
            try:
                reverted = await self._revert_changes(plan.revert_turns)
            except RevertConflict as exc:
                raise SessionError(
                    f"/{label} conflict: the turn's files changed since it "
                    f"ended — nothing was changed\n{exc}"
                ) from exc
            plan.revert_turns = plan.revert_turns[:reverted]
            # A rollback moves the files too: re-account the workspace before
            # persisting, or the reverted state would look like external drift
            # on the next resume.
            if self.workspace_state is not None:
                try:
                    await self.workspace_state.refresh()
                except GitWorktreeError:
                    pass
        self.conversation.apply_rollback(plan)
        # The truncation invalidated rows on disk from the cut onward; the
        # conversation lowered its persist watermark itself.
        if plan.compaction:
            note = "reverted"
        elif self.workspace_state is None:
            note = "no_tracking"
        elif not plan.revert_turns:
            note = "no_edits"
        else:
            note = "reverted"
        self._commit_history("rollback", scope=scope, reverted=len(plan.revert_turns),
                             resend=resend, note=note)
        if resend and not plan.compaction:
            # A compaction undo has no user message to resend: starting a
            # turn with empty text would send a blank message to the model.
            await self._start_turn(plan.user_text, plan.images)
            return CommandResult()
        return CommandResult(user_text=plan.user_text or None, images=plan.images)

    async def _cmd_compact(self) -> CommandResult:
        """Start a manual compaction as a task: commands stay free (cancel
        works, commands keep flowing); the task drains the steering queue
        when it finishes."""
        self._set_compacting(True)
        task = asyncio.get_running_loop().create_task(self._run_compact())
        self._compact_task = task
        task.add_done_callback(self._on_compact_done)
        return CommandResult()

    async def _run_compact(self) -> None:
        try:
            result = await self.compact()
        except asyncio.CancelledError:
            self._set_compacting(False)
            await self._start_queued()
            raise
        except Exception as exc:
            self._set_compacting(False)
            self._emit_notice(f"compaction failed: {exc}", "error")
            await self._start_queued()
            return
        self._set_compacting(False)
        if result is None:
            # Nothing to compact: a notice, not a history rewrite.
            self._persist()
            self._emit_notice("nothing to compact yet")
        else:
            self._commit_history(
                "compaction", automatic=False,
                before_tokens=result.before_tokens,
                after_tokens=result.after_tokens,
            )
        await self._start_queued()

    async def _start_queued(self) -> bool:
        """Start the oldest queued message unless the session is closed."""
        if not self._queued or self._closed:
            return False
        item = self._queued.pop(0)
        self._emit_status()
        await self._start_turn(item.text, item.images)
        return True

    async def _cmd_accept(self) -> CommandResult:
        """Lock in every pending change, drift included (allowed mid-turn)."""
        ws = self.workspace_state
        if ws is None:
            raise SessionError(
                "/accept: no change tracking for this session (the workspace "
                "isn't a git repo, or tracking setup failed)"
            )
        mid_turn = self.state != "idle"
        try:
            summary = await ws.diff(self.name)
            await ws.accept()
        except GitWorktreeError as exc:
            raise SessionError(f"/accept failed: {exc}") from exc
        self._workspace_changed("accept")
        if mid_turn:
            self._emit_notice(
                "note: the current turn is still generating — it may keep "
                "editing files on top of the accepted state", "warn",
            )
        accepted = summary.files
        foreign = [row for row in accepted if row.kind != "mine"]
        shared = [row for row in accepted if len(row.authors) > 1]
        if not summary.files:
            self._emit_notice(
                "accepted: no pending changes — baseline reset; /diff and /undo "
                "now measure from here"
            )
        else:
            detail = ""
            if foreign:
                detail = (
                    f"; {len(foreign)} of {len(summary.files)} pending "
                    "files weren't made by this agent"
                )
            if shared:
                detail += (
                    f"; {len(shared)} of the pending files were written by "
                    "more than one agent, so the totals count each author's "
                    "own delta"
                )
            self._emit_notice(
                f"accepted: {len(accepted)} file(s) +{summary.additions} "
                f"-{summary.deletions} locked in{detail}; /diff is reset and "
                "/undo can no longer revert them"
            )
        return CommandResult()

    async def _cmd_reject(self) -> CommandResult:
        """Discard every change made since the last /accept (allowed
        mid-turn), including drift."""
        ws = self.workspace_state
        if ws is None:
            raise SessionError(
                "/reject: no change tracking for this session (the workspace "
                "isn't a git repo, or tracking setup failed)"
            )
        mid_turn = self.state != "idle"
        try:
            summary = await ws.diff(self.name)
            unrestorable = await ws.reject()
        except GitWorktreeError as exc:
            raise SessionError(f"/reject failed: {exc}") from exc
        self._workspace_changed("reject")
        if mid_turn:
            self._emit_notice(
                "note: the current turn is still generating — it may keep "
                "editing files on top of the reverted state", "warn",
            )
        if not summary.files and not summary.groups:
            self._emit_notice("rejected: nothing to discard")
        else:
            drift = sum(1 for row in summary.files if row.kind == "drift")
            detail = f" (including {drift} drift file(s))" if drift else ""
            self._emit_notice(
                f"rejected: {len(summary.files)} file(s) "
                f"+{summary.additions} -{summary.deletions} since the last "
                f"accept were reverted{detail}"
            )
        if unrestorable:
            self._emit_notice(
                f"{len(unrestorable)} ignored file(s) edited by agents were "
                "not restored: " + ", ".join(unrestorable[:5]), "warn",
            )
        return CommandResult()

    def _cmd_set_mode(self, command: protocol.SetMode) -> CommandResult:
        self.mode = command.mode
        self._rebuild_tools()
        self._emit_status()
        return CommandResult()

    def _cmd_set_view_state(self, command: protocol.SetViewState) -> CommandResult:
        """Remember the client's view choices with the session.

        View state, so it is allowed mid-turn and takes no command lock; it is
        persisted like any other session field so a reattach (or a later run)
        shows the same thing."""
        self.view = command.view
        self._persist()
        self._emit_status()
        return CommandResult()

    async def _cmd_switch_model(self, command: protocol.SwitchModel) -> CommandResult:
        profile = self._validate_model(command.name)
        api_key = resolve_api_key(profile)
        # Only the endpoint/key change; the conversation carries over.
        switch = getattr(self._client, "switch_profile", None)
        if switch is not None:
            switch(profile, api_key)
        self.profile = profile
        self.api_key = api_key
        self.profile_name = command.name
        self._persist()
        self._emit_status()
        return CommandResult()

    async def _cmd_set_sandbox(self, command: protocol.SetSandbox) -> CommandResult:
        """Switch this session's sandbox profile (idle only): the shell
        restarts, all subscribers get a Notice, and the change is saved in
        the session file — not in the config file."""
        name, settings = self._resolve_sandbox_profile(command.name)
        await self._apply_sandbox_settings(name, settings)
        self._emit_notice(
            f"sandbox switched to {name!r} — the persistent shell restarted"
        )
        return CommandResult()

    def _validate_model(self, name: str) -> "ProviderProfile":
        """Shared validation for /model: the resolved profile
        (with its API key already checked), or SessionError."""
        if self.config_path is None:
            raise SessionError("/model unavailable: no config path known")
        try:
            _, profile = load_sectioned(self.config_path).resolve_model(name)
        except ConfigError as exc:
            raise SessionError(f"/model failed: {exc}") from exc
        try:
            resolve_api_key(profile)
        except ConfigError as exc:
            raise SessionError(f"could not switch model: {exc}") from exc
        return profile

    def _validate_ref(self, path: str) -> str:
        """The normalized /ref path, or SessionError. Existence is checked
        by change_workspace_ref, the one place that also applies it."""
        raw = path.strip()
        if not raw or self.sandbox.settings is None:
            raise SessionError("references need a path and a sandbox")
        return raw

    def _validate_filter(self, path: str, add: bool) -> str:
        """Shared validation for /filter: the normalized
        workspace-relative path, or SessionError."""
        try:
            rel = validate_filter_path(path).strip("/")
        except ValueError as exc:
            raise SessionError(str(exc)) from exc
        if self.workspace_state is not None and (rel == ".git" or rel.startswith(".git/")):
            raise SessionError(
                ".git cannot be filtered while change tracking is on — the "
                "tracking dir must stay mounted"
            )
        if add and not (self.workspace / rel).exists():
            # Nothing to exclude: filtering a nonexistent path would either
            # fail at argv-build time or plant a stray path in the worktree.
            raise SessionError(f"no such path in the workspace: {rel!r}")
        return rel

    async def _cmd_filter(self, path: str, add: bool) -> CommandResult:
        """Add/remove a mount exclusion (idle only). Saved in the session
        file; the shell restarts so the new mounts take effect."""
        rel = self._validate_filter(path, add=add)
        settings = self.sandbox.settings
        if settings is None:
            raise SessionError("this session has no sandbox settings to filter")
        filters = list(settings.filter_list)
        if add:
            if rel not in filters:
                filters.append(rel)
        elif rel in filters:
            filters.remove(rel)
        settings = settings.model_copy(update={"filter_list": filters})
        await self._apply_sandbox_settings(self.sandbox.name, settings)
        verb = "added to" if add else "removed from"
        self._emit_notice(
            f"{rel} {verb} the sandbox mount exclusions — the persistent "
            "shell restarted"
        )
        return CommandResult()

    # ------------------------------------------------------------------
    # External references

    def _ref_infos(self) -> list[protocol.RefInfo]:
        """The refs currently bound, for the client's status payload: the
        profile's own refs plus this workspace's."""
        settings = self.sandbox.settings
        specs = [
            *(settings.auto_include_refs if settings is not None else ()),
            *self.workspace_refs,
        ]
        if self.sandbox.config is not None:
            return ref_infos(self.sandbox.config.external_refs)
        return ref_infos(
            SandboxConfig.resolve_auto_refs(self.workspace, specs)
        )

    async def _cmd_ref(self, path: str, add: bool) -> CommandResult:
        """Add/remove an external reference path for this workspace.

        It is bound read-only under the container-only refs tmpfs at
        ``/tmp/refs/<alias>`` and saved per workspace. Nothing is ever
        created in the worktree (or at any ref path). The add/remove logic
        is the workspace store's (:func:`change_workspace_ref`), shared with
        the host's workspace-level endpoint."""
        raw = self._validate_ref(path)
        try:
            specs, host = change_workspace_ref(
                self.workspace, raw, add, save=False,
            )
        except RefError as exc:
            raise SessionError(str(exc)) from exc
        try:
            if not self.no_save:
                save_workspace_refs(self.workspace, specs)
        except (OSError, sqlite3.Error) as exc:
            # a read-only data dir must not block this session's ref,
            # but the user should know the change won't survive a restart
            LOG.warning("could not save refs for %s: %s", self.workspace, exc)
        # apply_ref_specs owns the assignment (its early return compares
        # the stored list against the new one).
        await self.apply_ref_specs(specs)
        verb = "added to" if add else "removed from"
        self._emit_notice(
            f"{host} {verb} the workspace references"
            + (" — the persistent shell restarted"
               if self.sandbox.shell is not None else "")
        )
        return CommandResult()

    async def apply_ref_specs(self, specs: list[str]) -> None:
        """Apply a workspace's canonical ref list to this live session.

        Settings stay profile-only: the workspace refs ride into the config
        via ``extra_refs``, so removing one here actually unbinds it — the
        previous merge-into-settings approach baked refs into
        ``auto_include_refs`` and could never take one back (and re-merged
        on top of the polluted list on every call)."""
        specs = list(specs)
        if self.workspace_refs == specs:
            return
        self.workspace_refs = specs
        settings = self.sandbox.settings
        if settings is None:
            return
        # Same profile settings; the rebuilt config picks up the new
        # extra_refs (and the shell restarts itself on the mount change).
        await self._apply_sandbox_settings(self.sandbox.name, settings, save=False)

    def _resolve_sandbox_profile(self, name: str | None):
        """Resolve a sandbox profile from the config (with inheritance)."""
        if self.config_path is None:
            raise SessionError("no config path known — cannot resolve sandbox profiles")
        try:
            sectioned = load_sectioned(self.config_path)
        except ConfigError as exc:
            raise SessionError(str(exc)) from exc
        return sectioned.resolve_sandbox(name)

    async def _apply_sandbox_settings(
        self, name: str | None, settings: SandboxSettings, save: bool = True,
    ) -> None:
        """Apply new sandbox settings (idle only). The persistent shell is
        adopted in place: it restarts itself on the next command when its
        argv changed, so only an unsandboxed<->container switch (or a failed
        build) needs a new SandboxSession.

        ``save=False`` for a workspace-refs-only change: the profile's own
        settings stay what the session was saved with, and the workspace refs
        are persisted per workspace, not in the session's sandbox settings."""
        old = self.sandbox
        try:
            config = SandboxConfig.from_settings(
                self.workspace, settings, self.workspace_state,
                extra_refs=self.workspace_refs,
            )
        except SandboxUnavailable as exc:
            config = None
            self._emit_notice(f"warning: filesystem/bash tools disabled: {exc}", "warn")
        if config is not None and old.shell is not None:
            # Same container mode: the persistent shell adopts the new config
            # and restarts itself on the next command if its argv changed.
            old.shell.adopt(config, default_timeout=settings.default_timeout)
            shell = old.shell
        else:
            # Entering or leaving container mode (or a failed build): the old
            # shell cannot follow the config, so start over without one.
            await old.close()
            shell = None if config is None else SandboxSession(
                config, default_timeout=settings.default_timeout,
            )
        self.sandbox = SandboxRuntime(
            config=config, shell=shell, name=name, settings=settings,
        )
        self._rebuild_tools()
        if save:
            self._persist()
        self._emit_status()

    # ------------------------------------------------------------------
    # turn loop

    def _emit_user_message(self, message: Message) -> None:
        """A user message landed in the history. This is the only way any
        client learns about it (no local echo)."""
        if message.timing.send is None:
            message.timing.send = _utcnow()
        # On disk before the turn does any work: a host killed mid-turn (a
        # crash, a stop, an eviction while the turn ran) would otherwise save
        # a session with no trace of what the user asked for, and reopening
        # it shows an empty transcript instead of the interrupted turn.
        self._persist()
        self._emit(protocol.MessageAdded(message=message))

    def _begin_turn(
        self, text: str, images: list[ImageRef] | None, before_tree: str | None,
    ) -> None:
        """Add the user message and open the turn's checkpoint.

        The checkpoint is ALWAYS attached — with ``before_tree=None`` when
        there is no tracking or the snapshot failed — so every turn has an
        entry and a rollback never guesses its scope."""
        message = user_message(text, images, self.images)
        self.conversation.add(message)
        self.conversation.begin_turn(
            message, before_tree,
            self.workspace_state.epoch if self.workspace_state else 0,
        )
        self._emit_user_message(message)

    async def _stream_round(self) -> AsyncIterator[StreamEvent]:
        """Stream one round — one full model response — into the history.

        If the round made tool calls (``RoundComplete.has_tool_calls``),
        :meth:`_run_turn` executes them, records the results and calls this
        again; the loop lives there."""
        partial = self.partial_round = PartialRound()
        done_calls: list[ToolCall] = []
        first_reasoning_at: float | None = None
        # The cot block ends where the answer or the first tool call starts —
        # tool-argument streaming counts as call time, not thinking time.
        first_answer_at: float | None = None

        async for event in self._client.stream(
            self.conversation.to_wire(
                keep_reasoning=self.profile.keep_reasoning,
                system=self.system_prompt,
                image_loader=self._image_url,
            ),
            tools=self.tool_registry.schema(),
        ):
            if isinstance(event, (
                ReasoningDelta, ContentDelta, ToolCallStarted,
                ToolCallArgumentsDelta, ToolCallArgumentsDone,
            )):
                if isinstance(event, ReasoningDelta) and first_reasoning_at is None:
                    first_reasoning_at = time.monotonic()
                if isinstance(event, (ContentDelta, ToolCallStarted)) \
                        and first_answer_at is None:
                    first_answer_at = time.monotonic()
                partial.apply(event)
                if isinstance(event, ToolCallArgumentsDone):
                    # Authoritative once the stream is done parsing; the
                    # client has already assembled the full arguments.
                    done_calls.append(ToolCall(
                        call_id=event.call_id, name=event.name,
                        arguments_json=event.arguments_json,
                    ))
            elif isinstance(event, RoundComplete):
                reasoning = partial.reasoning or None
                content = partial.content
                # The message is in the history now: only the tool calls
                # still executing remain live.
                partial.reasoning = ""
                partial.content = ""
                reasoning_seconds: float | None = None
                if first_reasoning_at is not None:
                    end = first_answer_at or time.monotonic()
                    reasoning_seconds = end - first_reasoning_at
                usage = event.usage or self._estimate_usage(content)
                self._fold_usage(usage)
                self.conversation.add(AssistantMessage(
                    content=content,
                    reasoning=reasoning,
                    tool_calls=done_calls or None,
                    usage=usage,
                    # Measured while the round streamed, never reconstructed
                    # from stamps afterwards.
                    timing=Timing(reasoning_seconds=reasoning_seconds),
                ))
                event = RoundComplete(
                    usage=usage,
                    has_tool_calls=bool(done_calls),
                )
            yield event

    def record_tool_result(
        self, call_id: str, result: Any, usage: Usage | None = None,
    ) -> None:
        """Append a ``tool`` message with the outcome of one call.

        ``content`` is stored as-is — a failing call keeps its real output,
        so failing test output reaches the model. ``usage`` is the token
        spend of a tool that ran its own rounds (a subagent); it stays
        local and is never sent to the provider."""
        self.conversation.add(ToolMessage(
            tool_call_id=call_id, content=result.content,
            status=result.status, usage=usage,
        ))

    def _save_interrupted_round(self) -> bool:
        """Persist the partial round a turn was streaming when it ended early.

        Called both when the user interrupts a turn (cancel, or a forced
        steer that cancels it) and when a round ends without completing (a
        provider or transport error). Keeps the assistant text and the
        partial chain of thought that had arrived — including a
        reasoning-only segment. Returns True when a message was added; a round
        with neither text nor reasoning leaves the history untouched.

        Only the model-stream half is consumed: the tool calls stay live so
        the session can fold the tool-execution events into them, and an
        attach during those tools renders them."""
        partial = self.partial_round
        content = partial.content
        reasoning = partial.reasoning or None
        partial.content = ""
        partial.reasoning = ""
        if not content and not reasoning:
            return False
        self.conversation.add(AssistantMessage(
            content=content, reasoning=reasoning, interrupted=True,
        ))
        return True

    def _image_url(self, ref: ImageRef) -> str | None:
        """A data URL for one stored image (from the blob store's cache);
        None when the blob is gone."""
        return self.images.data_url(ref)

    def _fold_usage(self, usage: Usage) -> None:
        """Add one round's usage to the cumulative session total. Copies, so
        already-emitted status payloads keep their point-in-time values."""
        self.usage_total = self.usage_total + usage
        self.last_usage = usage.model_copy()
        if self._report_usage is not None:
            self._report_usage(self.usage_total)

    def _estimate_usage(self, content: str) -> Usage:
        prompt_chars = sum(
            len(message.content or "")
            for message in self.conversation.visible_messages()
        )
        return Usage(
            input_tokens=prompt_chars // CHARS_PER_TOKEN,
            output_tokens=len(content) // CHARS_PER_TOKEN,
            approximate=True,
        )

    async def compact(self) -> "compaction.CompactionResult | None":
        """Summarize the conversation and shrink the visible slice to the
        summary message (old messages stay in the append-only list and come
        back if the summary is popped). Returns the
        :class:`~xarness.compaction.CompactionResult`, or None when there was
        nothing to do."""
        result = await compaction.compact(
            self.conversation, self._client,
            system=self.system_prompt,
            keep_reasoning=self.profile.keep_reasoning,
            image_loader=self._image_url,
        )
        if result is None:
            return None
        if result.usage is not None:
            self._fold_usage(result.usage)
        return result

    async def _run_turn(self, text: str, images: list[ImageRef] | None) -> None:
        """Drive the full exchange: rounds of (thinking → answer → tools).

        Runs as its own task, started by the send_message handler — so the
        command path stays free to dispatch accept/reject/answer_ask while
        the turn streams. The state was already set to running by
        _start_turn, synchronously before the task existed."""
        self._reset_partial()
        turn_start = time.monotonic()
        # Streaming + tool time, measured as it happens (see _close_out_turn).
        busy = 0.0
        outcome = "finished"
        try:
            # Start-of-turn auto-compaction: the threshold may have been
            # crossed right at the previous turn's end, where compacting
            # would only produce a handoff the model answers redundantly.
            await self._maybe_auto_compact()
            self._begin_turn(text, images, await self._take_checkpoint())
            stream = self._stream_round()
            saw_stream_error = False
            while True:
                round_start = time.monotonic()
                async for event in stream:
                    if isinstance(event, protocol.StreamError):
                        saw_stream_error = True
                    self._emit(event)
                # A round is busy from its request to its last event (the
                # same span the old per-message stamps covered); whatever is
                # left of the turn is queueing and teardown, i.e. idle.
                busy += time.monotonic() - round_start
                # A round that ended without finishing — a provider or
                # transport error, say — still streamed something: keep the
                # partial answer (see _save_interrupted_round).
                if self._save_interrupted_round():
                    self._persist()
                # The round folded its usage into the totals: push a fresh
                # status so clients update in/out/ctx per round while the
                # turn keeps streaming.
                self._emit_status()
                # The provider stamped this round's tool calls onto the
                # assistant message at RoundComplete; read them from there
                # instead of accumulating a second list.
                last = self.conversation.messages[-1] if self.conversation.messages else None
                done_calls = (
                    list(last.tool_calls or [])
                    if last is not None and last.role == "assistant" else []
                )
                if not done_calls:
                    # Final round — no tool calls means the work is done.
                    # Compacting here would hand the model a summary to
                    # answer redundantly; the next turn compacts at its
                    # start instead (see the top of this method).
                    break
                # Execute the calls this round requested, in stream order.
                # Parallel subagents are the one exception (see _execute_calls):
                # a round that spawns several children runs them together.
                tool_start = time.monotonic()
                await self._execute_calls(done_calls)
                busy += time.monotonic() - tool_start
                self._persist()
                # Steering: anything queued while this round streamed is
                # injected here — after the tool answers, before the next
                # LLM call.
                self._inject_queued()
                # Round boundary: everything so far is in the history (the
                # assistant message was stamped, tool results recorded), so
                # the live-round state resets — a client attaching between
                # rounds renders the history only.
                self._reset_partial()
                # Per-round auto-compact: the threshold is checked at every
                # round boundary (tool results recorded, assistant stamped),
                # not only when the whole turn ends — a long tool loop must
                # not overshoot the context window.
                await self._maybe_auto_compact()
                stream = self._stream_round()
            # A provider error ended the last stream: the round produced no
            # final answer, so the turn did not finish.
            if saw_stream_error:
                outcome = "failed"
                self._emit_notice("the provider stream failed", "error")
            # The turn's content is in the history now; clear the live-round
            # state before the turn-end events so an attach during unwinding
            # renders the history once.
            self._reset_partial()
        except asyncio.CancelledError:
            outcome = "cancelled"
            # Keep whatever the model had produced before the interruption.
            if self._save_interrupted_round():
                self._persist()
            self._repair_tool_calls()
            self._emit_notice("interrupted", "error")
        except Exception as exc:
            outcome = "failed"
            if self._save_interrupted_round():
                self._persist()
            self._repair_tool_calls()
            self._emit_notice(f"turn failed: {type(exc).__name__}: {exc}", "error")
        finally:
            # Teardown must not be interrupted: the footer still says
            # "running" until the state flip, so a Cancel can land here
            # while the after-tree snapshot persists a big repo, and a
            # CancelledError would skip the flip and strand "running"
            # (see _cancellable).
            self._finishing_turn = asyncio.current_task()
            try:
                await self._close_out_turn(
                    outcome, time.monotonic() - turn_start, busy,
                )
            finally:
                self._finishing_turn = None

    async def _close_out_turn(
        self, outcome: str, total: float, busy: float,
    ) -> None:
        """End-of-turn teardown: snapshot the workspace, persist, flip to
        idle, announce.

        The snapshot and the disk write are best-effort — a failure there
        must not lose the turn. The state flip is not: a session left
        "running" with no task is rendered as a live turn by every later
        attach and is never evicted."""
        stats: TurnStats | None = None
        try:
            # Record where the turn left the workspace so /undo //retry can
            # reverse exactly this turn's diff later (interruptions
            # included).
            await self._finish_turn()
            # The turn's numbers are the host's to record (see
            # Conversation.stamp_turn_end): stamped onto the turn's last
            # answer before the write, so a reload and a live client read the
            # same figures. A turn cut short reports how long it ran but no
            # busy/idle split — its rounds were not measured to an end.
            stats = self.conversation.stamp_turn_end(
                total, busy if outcome == "finished" else None,
            )
            self._persist()
            await self._flush_persist()
        except asyncio.CancelledError:
            pass  # a cancel slipped past the guards; cleanup still wins
        except Exception:
            LOG.exception(
                "could not close out the turn for session %s", self.name,
            )
        finally:
            self._reset_partial()
            # Clearing the task IS the flip to idle (see the state property):
            # the status goes out before TurnEnded, which describes the end
            # and must not be buffered as a live turn event for the next
            # attach. The numbers are the ones just stamped on the turn's
            # last answer, so a live footer and a replayed one are identical.
            self._turn_task = None
            self._emit_status()
            self._emit(protocol.TurnEnded(outcome=outcome, stats=stats))
        await self._emit_diff_changed()
        # A queued message sent at the exact end of the turn (or a
        # "send queued now" cancel) starts immediately.
        await self._start_queued()

    async def _execute_calls(self, calls: list[ToolCall]) -> None:
        """Run the round's calls in stream order, except that consecutive
        ``subagent`` calls run concurrently (independent children, no shared
        shell). Other tools mutate one workspace and stay ordered."""
        for parallel, group in groupby(calls, lambda c: c.name == "subagent"):
            batch = list(group)
            if not parallel:
                for call in batch:
                    await self._execute_call(call)
                continue
            async with asyncio.TaskGroup() as tasks:
                for call in batch:
                    tasks.create_task(self._execute_call(call))

    async def _execute_call(self, call: ToolCall) -> None:
        """Run one tool call, streaming its output as ToolOutputDelta."""
        self._emit(protocol.ToolExecStarted(call_id=call.call_id))
        call_id = call.call_id
        remaining = self.tool_registry.max_output_chars
        # The last token report a subagent made, stamped onto the finished
        # row so even an instant child shows its spend.
        last_usage: Usage | None = None

        def usage_sink(usage: Usage) -> None:
            nonlocal last_usage
            last_usage = usage
            self._emit(protocol.ToolUsageChanged(
                call_id=call_id, usage=usage,
            ))

        def sink(text: str) -> None:
            nonlocal remaining
            if remaining <= 0:
                return
            chunk = text[:remaining]
            remaining -= len(chunk)
            self._emit(protocol.ToolOutputDelta(call_id=call_id, text=chunk))
            if not remaining:
                self._emit(protocol.ToolOutputDelta(
                    call_id=call_id, text="\n[live output truncated]\n",
                ))

        # ToolRegistry.call already converts any tool exception into a
        # failed ToolResult, so no try/except is needed here.
        ws = self.workspace_state
        record = None if ws is None else await self._observe(ws, call)
        self._executing_tool = call.name or call_id
        self._executing_calls += 1
        try:
            result = await self.tool_registry.call(
                call.name, call.arguments_json, output_sink=sink,
                usage_sink=usage_sink,
            )
        finally:
            self._executing_calls = max(0, self._executing_calls - 1)
            if self._executing_calls == 0:
                self._executing_tool = None
        # A command's change is measured from its before/after snapshots, so
        # it is recorded whatever the exit status: a failing command still
        # leaves its writes on disk. Explicit writes keep the success gate.
        if record is not None and (
            result.status == "ok"
            or (call.name == "run_bash" and result.status != "parse_error")
        ):
            await record()
        self._emit(protocol.ToolFinished(
            call_id=call_id, status=result.status, content=result.content,
        ))
        if call.name in MUTATING_TOOLS:
            # Coalesced: a burst of run_bash calls must not trigger a git
            # snapshot each; one trailing debounce covers them all.
            self._workspace_changed()
        self.record_tool_result(call_id, result, usage=last_usage)

    def _write_target(
        self, ws: WorkspaceState, call: ToolCall,
    ) -> tuple[str, bytes | None] | None:
        """``(repo path, content)`` of an edit_file/write_file target, read
        before the tool runs: the "before" side of the provenance blob pair,
        so the event can attribute this call's own delta instead of the whole
        file's pending diff.

        The tool interprets a relative path against the sandbox cwd
        (``--workspace repo/sub``); events are repo-root-relative. None when
        the arguments do not name a file inside the workspace."""
        try:
            args = json.loads(call.arguments_json)
        except (TypeError, ValueError):
            return None
        path = args.get("path") if isinstance(args, dict) else None
        if not isinstance(path, str) or not path:
            return None
        try:
            rel = (self.workspace / path).resolve().relative_to(
                ws.root).as_posix()
        except ValueError:
            return None
        try:
            return rel, (ws.root / rel).read_bytes()
        except OSError:
            return rel, None

    async def _observe(
        self, ws: WorkspaceState, call: ToolCall,
    ) -> Callable[[], Awaitable[None]] | None:
        """The log step for one tool call, with its "before" state read now,
        so the event describes this call's own change. None for a call that
        never logs (read-only tools, or no file inside the workspace)."""
        if call.name in _EXPLICIT_WRITES:
            target = self._write_target(ws, call)
            if target is None:
                return None
            return partial(self._record_write, ws, call, *target)
        if call.name != "run_bash":
            return None
        # A fresh snapshot before the command (not the cached head).
        before = await ws.observe()
        ignored_before = await ws.ignored()
        return partial(self._record_bash, ws, call, before, ignored_before)

    async def _record_write(
        self, ws: WorkspaceState, call: ToolCall,
        path: str, before: bytes | None,
    ) -> None:
        """Log one explicit write, with the blob shas of its own delta.

        Runs under the workspace lock, so an event lands entirely before or
        entirely after an ``/accept`` or ``/reject``: it can never describe
        pre-rebaseline work to a log the rebaseline just cleared."""
        async with ws.lock:
            try:
                after: bytes | None = (ws.root / path).read_bytes()
            except OSError:
                after = None
            try:
                # A call that left the file as it found it has no provenance:
                # there is no delta to attribute.
                writes = () if before == after else ((
                    path,
                    None if before is None else await write_blob(ws, before),
                    None if after is None else await write_blob(ws, after),
                ),)
            except GitWorktreeError:
                # No provenance this time: /diff falls back to the whole
                # file's pending delta for the path.
                writes = ()
            # Always recorded, even for an ignored path: the tree never holds
            # ignored files, but the event is what makes them agent work
            # rather than drift.
            ws.record(
                session=self.name, tool=call.name,
                turn=self.conversation.open_turn_id,
                paths=(path,), auto=(), writes=writes,
            )

    async def _record_bash(
        self, ws: WorkspaceState, call: ToolCall,
        before: str, ignored_before: tuple[str, ...],
    ) -> None:
        """Log what one command changed, from the snapshots around it."""
        async with ws.lock:
            try:
                after = await ws.snapshot()
                ignored_after = await ws.ignored()
            except GitWorktreeError:
                return
            stat = await tree_diff_stat(ws, before, after)
            ws.record(
                session=self.name, tool=call.name,
                turn=self.conversation.open_turn_id,
                paths=tuple(f.path for f in stat.files),
                auto=tuple(sorted(set(ignored_after) - set(ignored_before))),
            )

    def _inject_queued(self) -> None:
        """Inject the steering queue at a round boundary: the messages land
        in the history (with their original send_time) so the next LLM call
        sees them. No checkpoint of their own: they belong to the turn that
        is already running."""
        if not self._queued:
            return
        for item in self._queued:
            message = user_message(
                item.text, item.images, self.images,
                timing=Timing(send=item.send_time),
            )
            self.conversation.add(message)
            self._emit_user_message(message)
        self._queued.clear()
        self._emit_status()

    def _repair_tool_calls(self) -> None:
        """Answer orphaned tool calls so the next API request is valid;
        persist when anything changed."""
        if self.conversation.repair_orphaned_tool_calls(now=_utcnow()):
            self._persist()

    # ------------------------------------------------------------------
    # ask tool

    async def _ask_user(self, questions: list[protocol.AskQuestion]) -> list[str] | None:
        """Callback for the ask tool: surface pending_ask and wait for an
        ``answer_ask`` command from any attached client. If nobody is
        attached it waits indefinitely — the session is headless and the
        turn keeps its state."""
        ask = protocol.PendingAsk(ask_id=uuid.uuid4().hex, questions=questions)
        future: asyncio.Future[str | None] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending_ask = ask
        self._pending_ask_future = future
        self._emit_status()
        try:
            return await future
        finally:
            self._pending_ask = None
            self._pending_ask_future = None
            self._emit_status()

    # ------------------------------------------------------------------
    # git tracking

    def _workspace_changed(self, reason: str | None = None) -> None:
        """A tool changed the workspace (or its events): refresh the diff in
        every agent sharing it, not only this one. ``reason`` also announces
        an accept/reject rebaseline."""
        if self._notify_workspace_cb is not None:
            self._notify_workspace_cb(reason)
            return
        self._schedule_diff_changed()
        if reason is not None:
            self._emit(protocol.CheckpointsChanged(reason=reason))

    async def _take_checkpoint(self) -> str | None:
        """Snapshot the workspace's current state as a tree and return the
        sha. None when there is no change tracking or the checkpoint failed."""
        if self.workspace_state is None:
            return None
        try:
            return await self.workspace_state.refresh()
        except GitWorktreeError:
            return None

    async def _finish_turn(self) -> None:
        """Record the workspace state after the turn's edits as the open
        turn's after tree, so /undo can reverse exactly this turn's diff
        later, and mark the turn closed. No-op without an open turn; a
        failed snapshot closes the turn without an after tree (its edits
        just become unrevertable, like untracked turns')."""
        turn = self.conversation.open_turn
        if turn is None:
            return
        if self.workspace_state is not None and turn.before_tree:
            try:
                turn.after_tree = await self.workspace_state.refresh()
            except GitWorktreeError:
                turn.after_tree = None
        turn.closed = True

    def _open_turn_before(self) -> str | None:
        """The still-open turn's before tree, or None."""
        turn = self.conversation.open_turn
        return turn.before_tree if turn is not None else None

    def turn_checkpoints(self) -> list[TurnCheckpoint]:
        """The reversible turns in the workspace's current epoch, oldest
        first. A turn that spans an accept/reject is deliberately excluded:
        its before tree no longer describes the baseline."""
        if self.workspace_state is None:
            return []
        return self.conversation.turn_checkpoints(self.workspace_state.epoch)

    async def _revert_changes(self, turn_ids: list[str]) -> int:
        """Reverse the given turns' file edits (newest first) onto the live
        workspace. Returns how many turns were actually reverted.

        Raises :class:`RevertConflict` with the conflicting paths when any
        of a turn's files changed since it ended (the workspace is left
        untouched). No tracking, an empty list, or turns from before an
        accept/reject (their epoch is stale) is a clean no-op."""
        ws = self.workspace_state
        if ws is None or not turn_ids:
            return 0
        turns = [
            (turn_id, turn) for turn_id in turn_ids
            if (turn := self.conversation.turns.get(turn_id)) is not None
            and turn.epoch == ws.epoch
        ]
        if not turns:
            return 0
        conflict = await ws.undo(self.name, turns)
        if conflict:
            raise RevertConflict(", ".join(conflict))
        return len(turns)

    # ------------------------------------------------------------------
    # misc

    def _refresh_system_prompt(self) -> None:
        """Derive the mode-aware system prompt for the next request. It is
        never persisted in the conversation."""
        self.system_prompt = system_prompt(
            self.sandbox.config, self.mode,
            tracked=self.workspace_state is not None,
            git_write=self.sandbox.allow_git_write,
            default_timeout=self._tool_timeout(),
            mcp_servers=self.mcp.connected_server_names(),
        )

    def _tool_timeout(self) -> float:
        """The profile's tool timeout (the one-shot tools default to it; a
        per-command override in the profile still wins for bash)."""
        settings = self.sandbox.settings
        return settings.default_timeout if settings is not None else DEFAULT_TIMEOUT

    def _rebuild_tools(self, output_limit: int | None = None) -> None:
        """Rebuild tools for the current mode/sandbox, preserving the output cap."""
        self.sandbox.set_mode(self.mode)
        limit = output_limit if output_limit is not None else self.tool_registry.max_output_chars
        self.tool_registry = build_registry(
            self.sandbox.config, self.sandbox.shell, mode=self.mode,
            ask_callback=self._ask_callback,
            default_timeout=self._tool_timeout(),
            spawn_subagent=self._spawn_subagent,
            mcp=self.mcp,
        )
        self.tool_registry.max_output_chars = limit
        self._refresh_system_prompt()

    def _persist(self) -> None:
        """Persist the session (profile key, not model_id; atomic write).

        The write itself is debounced and run in a worker thread so a burst
        of tool results never serializes the whole (image-laden) history on
        the event loop and stalls streaming for every client. ``await
        _flush_persist()`` wherever the file must already be on disk (turn
        end, compaction, shutdown)."""
        if self.no_save:
            return
        self._persist_dirty = True
        if self._persist_task is None or self._persist_task.done():
            self._persist_task = asyncio.get_running_loop().create_task(
                self._persist_loop()
            )

    def _persist_state(self) -> tuple[Any, dict[str, Any]]:
        """Serialize into write instructions; runs on the event loop (pydantic
        dumps copy, so the writer thread only ever touches plain data).

        The conversation owns the watermark (the first message index not yet
        safely on disk). Messages are immutable, so a normal persist only
        sends the tail; a rollback or a tool-call repair lowers it.

        ``take_dirty_from`` resets it OPTIMISTICALLY here, at serialize time:
        this is fully synchronous (no awaits), so the captured ``from_seq``
        and the reset describe the same list state. If the write fails, the
        loop restores the watermark to ``from_seq`` so the next persist
        rewrites from there."""
        stored = StoredSession(
            conversation=self.conversation,
            turns=self.conversation.turns,
            mode=self.mode,
            profile=self.profile_name,
            workspace=str(self.workspace),
            parent_id=self.parent_id,
            sandbox_name=self.sandbox.name,
            sandbox_settings=self.sandbox.settings,
            usage_total=self.usage_total,
            last_usage=self.last_usage,
            view=self.view,
        )
        from_seq = self.conversation.take_dirty_from()
        payload, meta = session_store.serialize_state(stored, from_seq)
        return payload, meta

    async def _persist_loop(self) -> None:
        while self._persist_dirty:
            self._persist_dirty = False
            payload, meta = self._persist_state()
            try:
                await asyncio.to_thread(
                    session_store.write_state, self.name, payload, meta,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                # A failed write must not leave the watermark ahead of disk:
                # reset to the serialize-time point so the next _persist()
                # rewrites from there. Swallowing keeps the turn alive — a
                # persist failure must not skip the teardown that reports
                # the end of the turn.
                LOG.exception("persist failed for session %s", self.name)
                self.conversation._mark_dirty(payload["from_seq"])
                return

    async def _flush_persist(self) -> None:
        """Await completion of any pending background write."""
        if self._persist_task is not None:
            try:
                await self._persist_task
            except asyncio.CancelledError:
                pass
            self._persist_task = None

    async def _maybe_auto_compact(self) -> None:
        """Compact when the context nears the window.

        Called at the turn's start and at each round boundary that leads to
        another round (never on the final round: the model would answer the
        handoff redundantly), so a long multi-round turn cannot overshoot
        ``max_context``."""
        if not self.profile.auto_compact:
            return
        # The summary message a compaction appends carries no usage (the
        # summarization round's usage is folded into the totals instead), so
        # right after a compaction the estimate is 0 and this does nothing —
        # no loop guard needed.
        used = compaction.context_tokens(
            self.conversation.visible_messages(), self.profile.keep_reasoning,
        )
        if used < self.profile.auto_compact_threshold * self.profile.max_context:
            return
        # A send arriving during the compaction (or the teardown unwind) is
        # queued, never started mid-teardown.
        self._set_compacting(True)
        try:
            result = await self.compact()
        except RuntimeError as exc:
            self._emit_notice(f"auto-compact failed: {exc}", "warn")
            return
        finally:
            # Mid-turn compaction goes back to streaming; the turn's teardown
            # still owns the transition to idle.
            self._set_compacting(False)
        if result is None:
            # Nothing to compact (e.g. the threshold was crossed between the
            # estimate and the run, or the summarizer refused).
            return
        self._commit_history(
            "compaction", automatic=True,
            before_tokens=result.before_tokens,
            after_tokens=result.after_tokens,
        )
        await self._flush_persist()

    # ------------------------------------------------------------------
    # stats row

    async def diff_info(
        self, path: str | None, author: str | None = None,
    ) -> protocol.DiffInfo:
        """The /diff payload: per-file stat rows (+ the unified diff text for
        a row click-through; the summary poll stays cheap).

        Rows follow the session's saved scope, so a shared file is one
        combined row or one row per author; ``author`` asks for the exact
        patch one of those rows counted."""
        ws = self.workspace_state
        if ws is None:
            return protocol.DiffInfo(tracking=False)
        try:
            # A row click-through needs both the stat row and the text;
            # snapshot the workspace once and share the sha between them.
            current = await ws.observe()
            info = await ws.diff(
                self.name, current=current, path=path,
                scope=self.view.diff_scope,
            )
            if path is not None:
                info.text = await ws.file_diff(
                    path, current=current, author=author,
                )
        except GitWorktreeError as exc:
            # Git failures raise instead of reporting tracking=False: a
            # transient index.lock (concurrent snapshots) must not make
            # clients believe tracking disappeared.
            raise SessionError(f"diff failed: {exc}") from exc
        info.tracking = True
        return info
