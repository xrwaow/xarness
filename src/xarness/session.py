"""Headless chat session: all conversation state and logic, no UI.

One :class:`Session` owns the :class:`~xarness.controller.ChatController`,
the tool registry, the sandbox, git tracking, the steering queue and the
run state. It runs in a host process and outlives every client: a turn
keeps going after the TUI that started it closes, and any number of
clients attach to the same event stream.

**Single writer.** All mutation flows through one per-session command lock:
HTTP handlers ``await session.execute(...)`` — the asyncio.Lock (FIFO)
serializes commands, and the result goes only to the issuer. Pure
read-only queries (snapshots, diffs) bypass the lock; they are safe
because only the session mutates that state.

**Fan-out.** :meth:`_emit` pushes every event to every subscriber queue.
A client attaches through :meth:`subscribe`,
which builds the snapshot synchronously (no await between building it
and registering the subscriber, so no event can slip between the two)
and then replays the live stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from . import compaction, protocol, session_store
from .config import (
    ConfigError,
    ProviderProfile,
    SandboxSettings,
    load_sectioned,
    resolve_api_key,
    save_preferences,
    validate_filter_path,
)
from .controller import ChatController
from .errors import RefError, SessionError
from .conversation import (
    Conversation,
    Message,
    RollbackPlan,
    ToolCall,
    TurnStats,
)
from .gitwork import (
    GitWorktreeError,
    RevertConflict,
    TurnCheckpoint,
    git_diff,
    tree_diff_stat,
)
from .images import ImageRef
from .prompts import system_prompt
from .protocol import (
    Command,
    CommandResult,
    Event,
    HistoryReason,
    Mode,
    SessionSummary,
    Snapshot,
    State,
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
    WorkspaceState,
    change_workspace_ref,
    save_workspace_refs,
)

# Tools that can change the workspace; only these trigger a DiffChanged
# event (read_file/search tools cannot).
MUTATING_TOOLS = frozenset({"write_file", "edit_file", "run_bash"})

# How many recent cmd_ids are remembered for dedupe (client retries must
# not double-send a command).
CMD_DEDUPE_LIMIT = 128

# Trailing debounce for DiffChanged after a mutating tool (one in flight).
DIFF_DEBOUNCE_SECONDS = 0.25

# Commands that rewrite conversation/workspace state are rejected while a
# turn or compaction is running.
IDLE_ONLY = (
    protocol.Rollback, protocol.Compact, protocol.SwitchModel,
    protocol.SetSandbox, protocol.Filter, protocol.Ref,
)

# Commands that only touch a flag or a future: they skip the command lock, so
# a slow /accept (up to seconds of git) never delays a cancel or an answer.
LOCK_FREE = (protocol.Cancel, protocol.AnswerAsk)


@dataclass(slots=True)
class _QueuedSteer:
    """One steering message waiting for a round boundary."""

    text: str
    images: list[ImageRef] | None
    send_time: str


LOG = logging.getLogger(__name__)


def _ok(**fields: Any) -> CommandResult:
    """A successful command result (``cmd_id`` set by the caller)."""
    return CommandResult(cmd_id="", ok=True, **fields)


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
    # Workspace reference paths (shared per workspace, managed with /ref).
    workspace_refs: list[str] = field(default_factory=list)
    # Called after a workspace event so every agent in the repo refreshes
    # its diff; injected by the host (None outside it).
    notify_workspace: Callable[[str | None], None] | None = None
    # Starts one subagent session for this one and returns its final message;
    # injected by the host, and None for a subagent itself (no nesting).
    spawn_subagent: Callable[["Session", str, str], Awaitable[str]] | None = None

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
        client: Any = None,
        config_path: Path | None = None,
        no_save: bool = False,
        parent_id: str | None = None,
        workspace_refs: list[str] | None = None,
        notify_workspace: Callable[[str | None], None] | None = None,
        subagent_model: str | None = None,
        spawn_subagent: Callable[["Session", str, str], Awaitable[str]] | None = None,
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
        self.tool_registry = build_registry(
            self.sandbox.config, self.sandbox.shell, mode=self.mode,
            ask_callback=self._ask_callback,
            default_timeout=self._tool_timeout(),
            spawn_subagent=self._spawn_subagent,
        )
        if tool_output_limit is not None:
            self.tool_registry.max_output_chars = tool_output_limit
        self.controller = ChatController(
            profile, api_key, tool_registry=self.tool_registry,
            client=client,
        )
        self.controller.on_user_message = self._on_user_message
        self._refresh_system_prompt()
        if conversation is not None:
            self.controller.conversation = conversation

        # Run state (replaces the TUI's _turn_busy/_compacting flags). The
        # event flips with it, so wait_idle() needs no polling.
        self._state: State = "idle"
        self._idle = asyncio.Event()
        self._idle.set()
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
        self.controller.image_store = self.images

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
        self._executing_tool: str | None = None

        # Fan-out.
        self._subscribers: list[asyncio.Queue[Event]] = []

        # Command serialization (FIFO) + dedupe of client retries.
        self._cmd_lock = asyncio.Lock()
        self._recent_cmds: OrderedDict[str, CommandResult] = OrderedDict()
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
        )
        # Restore the persisted token counters so the client's footer shows
        # real in/out/ctx immediately instead of 0 until the next round.
        session.controller.usage_total = state.usage_total
        session.controller.last_usage = state.last_usage
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
    # lifecycle

    async def stop(self) -> None:
        """Cancel a running turn or compaction and flush pending writes."""
        self._closed = True
        await self._cancel_and_wait(self._turn_task)
        await self._cancel_and_wait(self._compact_task)
        self._compact_task = None
        await self._cancel_and_wait(self._diff_task)
        self._diff_task = None
        await self._flush_persist()
        # Close the sandbox shell here (not only on host eviction) so every
        # path — explicit unload, delete, eviction, shutdown — releases it.
        await self.sandbox.close()
        # Release the provider connection pool.
        client = getattr(self.controller, "_client", None)
        if client is not None and hasattr(client, "aclose"):
            await client.aclose()

    @property
    def is_active(self) -> bool:
        """True while running, compacting, waiting on an ask, or holding an
        open event stream. Attached clients alone do not keep the host up."""
        return (
            self._state != "idle"
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
        return self._state

    @property
    def pending_ask(self) -> "protocol.PendingAsk | None":
        return self._pending_ask

    def summary(self) -> SessionSummary:
        """The list row for this hosted session."""
        state = self._state
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

    @property
    def conversation(self) -> Conversation:
        return self.controller.conversation

    # ------------------------------------------------------------------
    # fan-out

    def _emit(self, event: Event) -> Event:
        """Fold live tool-exec state into the snapshot and push the event."""
        self._track_partial(event)
        for queue in self._subscribers:
            queue.put_nowait(event)
        return event

    def _reset_partial(self) -> None:
        self.controller.reset_partial()

    def _track_partial(self, event: Event) -> None:
        """Fold the tool-execution events the session itself emits into the
        live round state (the controller owns the model-stream half).

        This is what lets an attach mid-tool-call render the call and its
        output so far without replaying an unbounded event log."""
        partial = self.controller.partial_round
        if isinstance(event, protocol.ToolOutputDelta):
            call = partial.find_call(event.call_id)
            if call is not None:
                call.content += event.text
        elif isinstance(event, protocol.ToolFinished):
            call = partial.find_call(event.call_id)
            if call is not None:
                call.status = event.status
                call.content = event.content

    def subscribe(self) -> tuple[Snapshot, asyncio.Queue[Event]]:
        """Build the attach snapshot and register a subscriber.

        Synchronous on purpose: no await between building the snapshot and
        registering the queue, so the stream cannot move in between (no
        gap, no duplicate)."""
        # Last line of defence before the snapshot: whatever stranded the
        # state (see _heal_stranded), the attaching client must never be
        # shown a live turn nobody is running.
        self._heal_stranded()
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
            state=self._state,
            usage_total=self.controller.usage_total,
            last_usage=self.controller.last_usage,
            queued=[
                protocol.QueuedMessage(text=item.text, images=item.images)
                for item in self._queued
            ],
            pending_ask=self.pending_ask,
            clients_attached=self.clients_attached,
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
            ask=(self._state in ("running", "compacting")
                 or self._pending_ask is not None) and others == 0,
            state=self._state,
            pending_ask=self._pending_ask is not None,
            other_clients=others,
        )

    def snapshot(self) -> Snapshot:
        """Full client-facing state (synchronous; see :meth:`subscribe`).
        Messages are pydantic models, so they go over the wire as-is."""
        conversation = self.conversation
        return Snapshot(
            messages=list(conversation.visible_messages()),
            # The compacted-away prefix is history for the human, not the
            # model: the client renders it so a resumed/compacted session
            # still shows everything that happened before the handoff.
            history=conversation.history_messages(),
            status=self.status(),
            partial_round=self.controller.partial_round or None,
        )

    def last_user_message_at(self) -> str | None:
        """Latest user send_time (see Conversation.last_user_message_at)."""
        return self.conversation.last_user_message_at()

    # ------------------------------------------------------------------
    # commands: serialized by the per-session lock

    async def execute(self, cmd_id: str, command: Command) -> CommandResult:
        """Run one command under the session lock and return its result.

        Duplicate ``cmd_id``s return the cached result, so a client retry
        never double-sends."""
        if (cached := self._recent_cmds.get(cmd_id)) is not None:
            return cached
        if isinstance(command, LOCK_FREE):
            # Only touches a flag or a future: no lock, so a slow /accept
            # (seconds of git) can never delay a cancel or an answer.
            result = await self._run_command(command)
        else:
            async with self._cmd_lock:
                if (cached := self._recent_cmds.get(cmd_id)) is not None:
                    return cached
                result = await self._run_command(command)
        result.cmd_id = cmd_id
        self._recent_cmds[cmd_id] = result
        while len(self._recent_cmds) > CMD_DEDUPE_LIMIT:
            self._recent_cmds.popitem(last=False)
        return result

    async def update_settings(self, update: "protocol.SessionUpdate") -> CommandResult:
        """PATCH /sessions/{name}: the settings commands on one typed body.

        One path for settings: this builds the same commands the slash
        commands use and runs them under the session lock, so a PATCH cannot
        interleave with an /accept or a compaction. Everything is validated
        BEFORE anything is applied, so a bad sandbox name never leaves a
        half-applied request behind (the model switched, the sandbox
        rejected). Errors come back as ``ok=False`` results, like every
        command's."""
        async with self._cmd_lock:
            # Mode flips mid-turn are fine (SetMode allows it too); the
            # history/sandbox-rewriting ones are idle-only.
            if self._state != "idle" and (
                update.model or update.sandbox
                or update.filter_path or update.ref_path
            ):
                return CommandResult(
                    cmd_id="", ok=False,
                    error="wait for the current turn or compaction to finish first",
                )
            # Omitted flags default to ADD: a PATCH body that names a path
            # but leaves the flag out must not silently become a removal.
            filter_add = True if update.filter_add is None else update.filter_add
            ref_add = True if update.ref_add is None else update.ref_add
            # ---- validate everything first ----------------------------
            try:
                if update.model is not None:
                    self._validate_model(update.model)
                if update.sandbox is not None:
                    self._resolve_sandbox_profile(update.sandbox)
                if update.filter_path is not None:
                    self._validate_filter(update.filter_path, add=filter_add)
                if update.ref_path is not None:
                    self._validate_ref(update.ref_path, add=ref_add)
            except SessionError as exc:
                return CommandResult(cmd_id="", ok=False, error=str(exc))
            except Exception as exc:
                return CommandResult(
                    cmd_id="", ok=False, error=f"{type(exc).__name__}: {exc}",
                )
            # ---- apply, stopping at the first failure -----------------
            result = _ok()
            if update.mode is not None:
                result = await self._run_command(
                    protocol.SetMode(mode=update.mode))
            if result.ok and update.model is not None:
                result = await self._run_command(
                    protocol.SwitchModel(name=update.model))
            if result.ok and update.sandbox is not None:
                result = await self._run_command(
                    protocol.SetSandbox(name=update.sandbox))
            if result.ok and update.filter_path is not None:
                result = await self._run_command(
                    protocol.Filter(path=update.filter_path, add=filter_add))
            if result.ok and update.ref_path is not None:
                result = await self._run_command(
                    protocol.Ref(path=update.ref_path, add=ref_add))
            return result

    async def _run_command(self, command: Command) -> CommandResult:
        try:
            return await self._dispatch(command)
        except SessionError as exc:
            return CommandResult(cmd_id="", ok=False, error=str(exc))
        except (GitWorktreeError, RevertConflict) as exc:
            return CommandResult(cmd_id="", ok=False, error=str(exc))
        except Exception as exc:
            return CommandResult(
                cmd_id="", ok=False, error=f"{type(exc).__name__}: {exc}",
            )

    async def _dispatch(self, command: Command) -> CommandResult:
        if isinstance(command, IDLE_ONLY) and self._state != "idle":
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
            case protocol.SwitchModel():
                return await self._cmd_switch_model(command)
            case protocol.SetSandbox():
                return await self._cmd_set_sandbox(command)
            case protocol.Filter():
                return await self._cmd_filter(command.path, command.add)
            case protocol.Ref():
                return await self._cmd_ref(command.path, command.add)
            case _:
                raise SessionError(f"unsupported command: {type(command).__name__}")

    # ------------------------------------------------------------------
    # state

    def _set_state(self, state: State) -> None:
        """Flip the state, wake wait_idle() and push the new status. Purely
        local: the host's reaper polls for inactivity, so nothing here can
        trigger an eviction and there is no callback ordering to respect."""
        if self._state == state:
            return
        self._state = state
        if state == "idle":
            self._idle.set()
        else:
            self._idle.clear()
        self._emit_status()

    async def wait_idle(self) -> None:
        """Wait for the running turn (if any) to finish, including its
        teardown. Returns at once when the session is already idle."""
        await self._idle.wait()

    def _emit_status(self) -> None:
        """Push the current status to every client."""
        self._emit(protocol.StatusChanged(status=self.status()))

    def _emit_notice(self, text: str, level: str = "info") -> None:
        self._emit(protocol.Notice(text=text, level=level))  # type: ignore[arg-type]

    def _history_event(
        self, reason: HistoryReason = "rollback", **fields: Any,
    ) -> protocol.HistoryReplaced:
        return protocol.HistoryReplaced(snapshot=self.snapshot(), reason=reason, **fields)

    def _commit_history(
        self, reason: HistoryReason = "rollback", **fields: Any,
    ) -> None:
        """Persist and broadcast the rewritten history in one step."""
        self._persist()
        self._emit(self._history_event(reason, **fields))

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
        if self._state != "idle":
            # Steering: queued until the next round boundary (or turn end).
            self._queued.append(_QueuedSteer(
                command.text, command.images or None, _utcnow(),
            ))
            self._emit_status()
            return _ok(queued=True)
        await self._start_turn(command.text, command.images or None)
        return _ok(started=True)

    async def _start_turn(self, text: str, images: list[ImageRef] | None) -> None:
        # Set synchronously, BEFORE create_task: the lock can dispatch the
        # next queued command before the turn task gets its first slot, and
        # a still-"idle" state would let a back-to-back send start a second
        # turn instead of steering the first.
        self._set_state("running")
        task = asyncio.get_running_loop().create_task(
            self._run_turn(text, images)
        )
        # Safety net: if the task ever ends without reaching the teardown's
        # state flip (an unexpected exception, a cancellation that slipped
        # past the guards), heal the session instead of reporting a turn
        # nobody is running any more.
        task.add_done_callback(self._on_turn_done)
        self._turn_task = task
        # Yield once so the turn task actually starts: a task cancelled
        # before its first step never runs its cleanup (state would stick
        # at "running" forever).
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
        """Heal a turn whose task ended without reaching the state flip."""
        if self._turn_task is task:
            self._turn_task = None
            self._heal_stranded()

    def _on_compact_done(self, task: asyncio.Task[None]) -> None:
        """Same safety net for a detached compaction."""
        if self._compact_task is task:
            self._compact_task = None
            self._heal_stranded()

    def _heal_stranded(self) -> None:
        """End a "running"/"compacting" state whose task is gone.

        Whatever stranded it — a task killed before its teardown, a path not
        found yet — a session must never claim a turn it is not running:
        every attach would render a live bar that never finishes, and
        ``is_active`` would keep the host from ever evicting it."""
        if self._state == "running" and not _alive(self._turn_task):
            self._turn_task = None
            self._reset_partial()
            self._set_state("idle")
            self._emit(protocol.TurnEnded(outcome="failed", elapsed=0.0))
            self._emit_notice(
                "the turn ended unexpectedly and was reset to idle", "error",
            )
        elif self._state == "compacting" and not _alive(self._compact_task):
            self._compact_task = None
            self._set_state("idle")
            self._emit_notice("compaction ended unexpectedly", "error")

    def _cmd_cancel(self) -> CommandResult:
        if self._state == "idle":
            return _ok(cancelled=False)
        self._cancel_work()
        return _ok(cancelled=True)

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
        return _ok()

    def _cmd_flush_queued(self) -> CommandResult:
        """Send the oldest queued message now: cancel the running turn or
        compaction; the cleanup starts the queued message immediately."""
        if self._state == "idle" or not self._queued:
            return _ok(flushed=False)
        self._cancel_work()
        return _ok(flushed=True)

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
            return _ok()
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
            return _ok()
        return _ok(user_text=plan.user_text or None, images=plan.images)

    async def _cmd_compact(self) -> CommandResult:
        """Start a manual compaction as a task: commands stay free (cancel
        works, commands keep flowing); the task drains the steering queue
        when it finishes."""
        self._set_state("compacting")
        task = asyncio.get_running_loop().create_task(self._run_compact())
        task.add_done_callback(self._on_compact_done)
        self._compact_task = task
        return _ok(started=True)

    async def _run_compact(self) -> None:
        try:
            result = await self.controller.compact()
        except asyncio.CancelledError:
            self._set_state("idle")
            await self._start_queued()
            raise
        except Exception as exc:
            self._set_state("idle")
            self._emit_notice(f"compaction failed: {exc}", "error")
            await self._start_queued()
            return
        self._set_state("idle")
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
        """Lock in the non-drift changes made so far (allowed mid-turn)."""
        ws = self.workspace_state
        if ws is None:
            raise SessionError(
                "/accept: no change tracking for this session (the workspace "
                "isn't a git repo, or tracking setup failed)"
            )
        mid_turn = self._state != "idle"
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
        accepted = [row for row in summary.files if row.kind != "drift"]
        foreign = [row for row in accepted if row.kind != "mine"]
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
            self._emit_notice(
                f"accepted: {len(accepted)} file(s) +{summary.additions} "
                f"-{summary.deletions} locked in{detail}; /diff is reset and "
                "/undo can no longer revert them"
            )
        return _ok()

    async def _cmd_reject(self) -> CommandResult:
        """Discard every change made since the last /accept (allowed
        mid-turn), including drift."""
        ws = self.workspace_state
        if ws is None:
            raise SessionError(
                "/reject: no change tracking for this session (the workspace "
                "isn't a git repo, or tracking setup failed)"
            )
        mid_turn = self._state != "idle"
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
        return _ok()

    def _cmd_set_mode(self, command: protocol.SetMode) -> CommandResult:
        self.mode = command.mode
        self._rebuild_tools()
        self._emit_status()
        return _ok(mode=self.mode)

    async def _cmd_switch_model(self, command: protocol.SwitchModel) -> CommandResult:
        profile = self._validate_model(command.name)
        api_key = resolve_api_key(profile)
        self.controller.switch_profile(profile, api_key)
        self.profile = profile
        self.profile_name = command.name
        self._persist()
        if command.persist_default:
            save_preferences(self.config_path, default_profile=command.name)
        self._emit_status()
        return _ok(model=command.name)

    async def _cmd_set_sandbox(self, command: protocol.SetSandbox) -> CommandResult:
        """Switch this session's sandbox profile (idle only): the shell
        restarts, all subscribers get a Notice, and the change is saved in
        the session file — not in the config file."""
        name, settings = self._resolve_sandbox_profile(command.name)
        await self._apply_sandbox_settings(name, settings)
        self._emit_notice(
            f"sandbox switched to {name!r} — the persistent shell restarted"
        )
        return _ok(sandbox=name)

    def _validate_model(self, name: str) -> "ProviderProfile":
        """Shared validation for /model and PATCH: the resolved profile
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

    def _validate_ref(self, path: str, add: bool) -> str:
        """Shared validation for /ref and PATCH: the normalized path (or
        SessionError). Existence is checked by change_workspace_ref, the one
        place that also applies the change."""
        raw = path.strip()
        if not raw or self.sandbox.settings is None:
            raise SessionError("references need a path and a sandbox")
        return raw

    def _validate_filter(self, path: str, add: bool) -> str:
        """Shared validation for /filter (and PATCH): the normalized
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
        return _ok(filter_list=filters)

    # ------------------------------------------------------------------
    # External references

    def _profile_refs(self) -> list:
        """The sandbox profile's own refs (never touched by /ref)."""
        settings = self.sandbox.settings
        return list(settings.auto_include_refs) if settings is not None else []

    def _ref_specs(self) -> list:
        """The effective ref list: the profile's refs plus this workspace's."""
        return [*self._profile_refs(), *self.workspace_refs]

    def _ref_infos(self) -> list[protocol.RefInfo]:
        """The refs currently bound, for the client's status payload."""
        specs = self._ref_specs()
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
        raw = self._validate_ref(path, add)
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
        return _ok(refs=self._ref_infos())

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
        """Apply new sandbox settings (idle only). The existing runtime is
        mutated in place where possible: the persistent shell restarts
        itself on the next command when the mount key changed, so only an
        unsandboxed<->container switch (or a failed build) needs a new
        SandboxSession.

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
        if config is None or old.config is None:
            # Unavailable (tools disabled) or leaving container mode: a
            # fresh runtime, without a shell when config is None.
            runtime = SandboxRuntime(
                config=config,
                shell=(SandboxSession(
                    config, default_timeout=settings.default_timeout,
                ) if config is not None else None),
                name=name, settings=settings,
            )
            runtime.set_mode(self.mode)
            await old.close()
            self.sandbox = runtime
        else:
            old.config = config
            old.shell.adopt(config, default_timeout=settings.default_timeout)
            old.name = name
            old.settings = settings
            old.set_mode(self.mode)
        self._rebuild_tools()
        if save:
            self._persist()
        self._emit_status()

    # ------------------------------------------------------------------
    # turn loop

    def _on_user_message(self, message: Message) -> None:
        """Controller hook: a user message landed in the history. This is
        the only way any client learns about it (no local echo)."""
        if message.timing.send is None:
            message.timing.send = _utcnow()
        # On disk before the turn does any work: a host killed mid-turn (a
        # crash, a stop, an eviction while the turn ran) would otherwise save
        # a session with no trace of what the user asked for, and reopening
        # it shows an empty transcript instead of the interrupted turn.
        self._persist()
        self._emit(protocol.MessageAdded(message=message))

    async def _run_turn(self, text: str, images: list[ImageRef] | None) -> None:
        """Drive the full exchange: rounds of (thinking → answer → tools).

        Runs as its own task, started by the send_message handler — so the
        command path stays free to dispatch accept/reject/answer_ask while
        the turn streams. The state was already set to running by
        _start_turn, synchronously before the task existed."""
        self._reset_partial()
        turn_start = time.monotonic()
        outcome = "finished"
        try:
            # Start-of-turn auto-compaction: the threshold may have been
            # crossed right at the previous turn's end, where compacting
            # would only produce a handoff the model answers redundantly.
            await self._maybe_auto_compact()
            stream = self.controller.begin_turn(
                text, images, await self._take_checkpoint(),
                self.workspace_state.epoch if self.workspace_state else 0,
            )
            saw_stream_error = False
            while True:
                async for event in stream:
                    if isinstance(event, protocol.StreamError):
                        saw_stream_error = True
                    self._emit(event)
                # A round that ended without finishing — a provider or
                # transport error, say — still streamed something. Keep the
                # partial answer (the controller decides what counts).
                if self.controller.save_interrupted_round():
                    self._persist()
                # The round folded its usage into the totals: push a fresh
                # status so clients update in/out/ctx per round while the
                # turn keeps streaming (the state is unchanged, so
                # _set_state would not emit).
                self._emit_status()
                # The controller stamped this round's tool calls onto the
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
                for call in done_calls:
                    await self._execute_call(call)
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
                stream = self.controller.stream_round()
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
            if self.controller.save_interrupted_round():
                self._persist()
            self._repair_tool_calls()
            self._emit_notice("interrupted", "error")
        except Exception as exc:
            outcome = "failed"
            if self.controller.save_interrupted_round():
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
                await self._close_out_turn(outcome, turn_start)
            finally:
                self._finishing_turn = None

    async def _close_out_turn(self, outcome: str, turn_start: float) -> None:
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
            # same figures. A turn cut short reports how long it ran — it
            # ended too, just not on its own terms.
            stats = self.conversation.stamp_turn_end(
                None if outcome == "finished"
                else time.monotonic() - turn_start,
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
            self._set_state("idle")
            # After the state flip: TurnEnded describes the end and must not
            # be buffered as a live turn event for the next attach. The
            # numbers are the ones just stamped on the turn's last answer, so
            # a live footer and a replayed one are identical.
            stamped = {} if stats is None else {
                "total": stats.total, "busy": stats.busy,
                "idle": stats.idle, "usage": stats.usage,
            }
            self._emit(protocol.TurnEnded(
                outcome=outcome, elapsed=time.monotonic() - turn_start,
                **stamped,
            ))
            self._turn_task = None
        await self._emit_diff_changed()
        # A queued message sent at the exact end of the turn (or a
        # "send queued now" cancel) starts immediately.
        await self._start_queued()

    async def _execute_call(self, call: ToolCall) -> None:
        """Run one tool call, streaming its output as ToolOutputDelta."""
        self._emit(protocol.ToolExecStarted(call_id=call.call_id))
        start = _utcnow()
        call_id = call.call_id
        remaining = self.tool_registry.max_output_chars

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
        before = ignored_before = None
        if ws is not None and call.name == "run_bash":
            # A fresh snapshot before the command (not the cached head), so
            # drift that happened earlier is never credited to the agent.
            before = await ws.observe()
            ignored_before = await ws.ignored()
        self._executing_tool = call.name or call_id
        try:
            result = await self.tool_registry.call(
                call.name, call.arguments_json, output_sink=sink,
            )
        finally:
            self._executing_tool = None
        if ws is not None and result.status == "ok":
            await self._record_event(ws, call, before, ignored_before)
        end = _utcnow()
        self._emit(protocol.ToolFinished(
            call_id=call_id, status=result.status, content=result.content,
        ))
        if call.name in MUTATING_TOOLS:
            # Coalesced: a burst of run_bash calls must not trigger a git
            # snapshot each; one trailing debounce covers them all.
            self._workspace_changed()
        # Refresh the footer's in/out/ctx per message, not just per round.
        self._emit_status()
        self.controller.record_tool_result(
            call_id, result, start_time=start, end_time=end,
        )  # Timing wraps the two stamps inside the controller

    async def _record_event(
        self, ws: WorkspaceState, call: ToolCall,
        before: str | None, ignored_before: tuple[str, ...] | None,
    ) -> None:
        """Record one successful mutating tool call in the workspace log.

        The whole step runs under the workspace lock, so an event lands
        entirely before or entirely after an ``/accept`` or ``/reject``: it
        can never describe pre-rebaseline work to a log the rebaseline just
        cleared.
        """
        if call.name in ("edit_file", "write_file"):
            try:
                args = json.loads(call.arguments_json)
            except (TypeError, ValueError):
                return
            path = args.get("path") if isinstance(args, dict) else None
            if not isinstance(path, str) or not path:
                return
            # The tool interprets a relative path against the sandbox cwd
            # (``--workspace repo/sub``); events are repo-root-relative.
            try:
                path = (self.workspace / path).resolve().relative_to(
                    ws.root).as_posix()
            except ValueError:
                return
            # Always recorded, even for an ignored path: the tree never
            # holds ignored files, but the event is what makes them agent
            # work rather than drift.
            async with ws.lock:
                ws.record(
                    session=self.name, tool=call.name,
                    turn=self.conversation.open_turn_id,
                    paths=(path,), auto=(),
                )
            return
        if call.name != "run_bash" or before is None:
            return
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
                auto=tuple(sorted(
                    set(ignored_after) - set(ignored_before or ())
                )),
            )

    def _inject_queued(self) -> None:
        """Inject the steering queue at a round boundary: the messages land
        in the history (with their original send_time) and fire the
        on_user_message hook, so the next LLM call sees them."""
        if not self._queued:
            return
        for item in self._queued:
            self.controller.inject_user_message(
                item.text, item.images, send_time=item.send_time,
            )
        self._queued.clear()
        self._emit_status()

    def _repair_tool_calls(self) -> None:
        """Answer orphaned tool calls so the next API request is valid;
        persist when anything changed."""
        if self.conversation.repair_orphaned_tool_calls(now=_utcnow()):
            self._persist()

    # ------------------------------------------------------------------
    # ask tool

    async def _ask_user(self, questions: list[str]) -> list[str] | None:
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
        self.controller.system_prompt = system_prompt(
            self.sandbox.config, self.mode,
            tracked=self.workspace_state is not None,
            git_write=self.sandbox.allow_git_write,
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
        )
        self.tool_registry.max_output_chars = limit
        self.controller.tools = self.tool_registry
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
            usage_total=self.controller.usage_total,
            last_usage=self.controller.last_usage,
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
                # persist failure must not skip _set_state("idle")/
                # TurnEnded in _run_turn's finally.
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

    async def _maybe_auto_compact(self) -> bool:
        """Compact when the context nears the window.

        Called at every round boundary (and on the turn's final round), so a
        long multi-round turn cannot overshoot ``max_context``. Returns True
        when a compaction happened — the caller keeps the turn going, so the
        model continues from the handoff summary without a user send.
        """
        if not self.profile.auto_compact:
            return False
        # The summary message a compaction appends carries no usage (the
        # summarization round's usage is folded into the totals instead), so
        # right after a compaction the estimate is 0 and this returns False —
        # no loop guard needed.
        used = compaction.context_tokens(
            self.conversation.visible_messages(), self.profile.keep_reasoning,
        )
        if used < self.profile.auto_compact_threshold * self.profile.max_context:
            return False
        # Stay in a non-idle state: only the turn's finally sets idle, so a
        # send arriving during the compaction (or the finish_turn unwind) is
        # queued, never started mid-teardown.
        self._set_state("compacting")
        try:
            result = await self.controller.compact()
        except RuntimeError as exc:
            self._emit_notice(f"auto-compact failed: {exc}", "warn")
            self._set_state("running")
            return False
        # Mid-turn compaction goes back to streaming; the turn's finally
        # still owns the transition to idle.
        self._set_state("running")
        if result is None:
            # Nothing to compact (e.g. the threshold was crossed between the
            # estimate and the run, or the summarizer refused).
            return False
        self._commit_history(
            "compaction", automatic=True,
            before_tokens=result.before_tokens,
            after_tokens=result.after_tokens,
        )
        await self._flush_persist()
        return True

    # ------------------------------------------------------------------
    # stats row

    async def diff_info(self, path: str | None) -> protocol.DiffInfo:
        """The /diff payload: per-file stat rows (+ the unified diff text
        for a per-file click-through; the summary poll stays cheap)."""
        ws = self.workspace_state
        if ws is None:
            return protocol.DiffInfo(tracking=False)
        try:
            # A per-file click-through needs both the stat rows and the
            # text; snapshot the workspace once and share the sha between
            # them.
            current = await ws.observe()
            info = await ws.diff(self.name, current=current)
            if path is not None:
                info.files = [row for row in info.files if row.path == path]
                info.groups = []
                info.ignored_files = 0
                info.text = await git_diff(ws, path, current=current)
        except GitWorktreeError as exc:
            # Git failures raise instead of reporting tracking=False: a
            # transient index.lock (concurrent snapshots) must not make
            # clients believe tracking disappeared.
            raise SessionError(f"diff failed: {exc}") from exc
        info.additions = sum(row.additions for row in info.files)
        info.deletions = sum(row.deletions for row in info.files)
        info.tracking = True
        return info
