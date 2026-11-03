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
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
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
from .errors import SessionError
from .conversation import (
    Conversation,
    Message,
    RollbackPlan,
    ToolCall,
)
from .diffview import build_diff_info
from .gitwork import (
    GitInfo,
    GitWorktreeError,
    RevertConflict,
    TurnCheckpoint,
    accept_changes,
    check_blocked_git,
    diff_stat,
    revert_to_tree,
    snapshot_state,
    tree_diff_stat,
)
from .images import ImageRef
from .prompts import prompt_facts, system_prompt_for
from .protocol import (
    Command,
    CommandResult,
    Event,
    HistoryReason,
    Mode,
    SessionSummary,
    Snapshot,
    State,
    StatusPayload,
)
from .sandbox import (
    SandboxConfig,
    SandboxRuntime,
    SandboxSession,
    SandboxUnavailable,
)
from .session_store import RefError, StoredSession, change_workspace_ref
from .tools import DEFAULT_TIMEOUT, build_registry
from .util import utcnow_iso as _utcnow

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

# Sentinel for "cached value not computed yet" (distinct from None, which
# is a real "no user message yet" answer).
_UNSET = object()


def _ok(**fields: Any) -> CommandResult:
    """A successful command result (``cmd_id`` set by the caller)."""
    return CommandResult(cmd_id="", ok=True, **fields)


async def _cancel_and_wait(task: asyncio.Task[None] | None) -> None:
    """Cancel a task (if any) and let it unwind, swallowing cancellation."""
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


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
    git_info: GitInfo | None
    tool_output_limit: int | None
    # Workspace reference paths (shared per workspace, managed with /ref).
    workspace_refs: list[str] = field(default_factory=list)

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
        git_info: GitInfo | None = None,
        mode: Mode = "write",
        tool_output_limit: int | None = None,
        client: Any = None,
        config_path: Path | None = None,
        no_save: bool = False,
        workspace_refs: list[str] | None = None,
        idle_callback: "Callable[[Session], None] | None" = None,
    ) -> None:
        self.name = name
        self.workspace = workspace
        self.profile = profile
        self.profile_name = profile_name
        self.api_key = api_key
        self.config_path = config_path
        self.sandbox = sandbox or SandboxRuntime()
        self.sandbox.set_mode(mode)
        self.git_info = git_info
        self.mode: Mode = mode
        self.tool_registry = build_registry(
            self.sandbox.config, self.sandbox.shell, mode=self.mode,
            ask_callback=self._ask_user,
            git_guard=self._make_git_guard(),
            default_timeout=self._tool_timeout(),
        )
        if tool_output_limit is not None:
            self.tool_registry.max_output_chars = tool_output_limit
        self.controller = ChatController(
            profile, api_key, tool_registry=self.tool_registry,
            client=client,
        )
        self.controller.git_info = git_info
        self.controller.on_user_message = self._on_user_message
        self._refresh_system_prompt()
        if conversation is not None:
            self.controller.conversation = conversation

        # Run state (replaces the TUI's _turn_busy/_compacting flags).
        self._state: State = "idle"
        self._turn_task: asyncio.Task[None] | None = None
        self._compact_task: asyncio.Task[None] | None = None
        # Steering queue — send_time is when the client sent it, not when it
        # is injected.
        self._queued: list[_QueuedSteer] = []
        # Pending ask-tool prompt. Any client may answer; first answer wins.
        self._pending_ask: protocol.PendingAsk | None = None
        self._pending_ask_future: asyncio.Future[str | None] | None = None
        # Pending resume-drift prompt.
        self._pending_drift: protocol.PendingDrift | None = None
        # Workspace reference paths (kept separate from the sandbox profile's
        # own auto_include_refs; the effective list is computed when the
        # sandbox is built).
        self.workspace_refs: list[str] = list(workspace_refs or [])
        # Set when a turn finishes with no subscriber attached; surfaced to
        # the next client that attaches ("finished while you were away").
        self._finished_while_away = False
        # Host-provided hook: called whenever the session may have gone
        # fully inactive (idle, no subscribers, no pending ask) so the host
        # can persist and release it.
        self.idle_callback: Callable[["Session"], None] | None = idle_callback
        # Resume drift is detected lazily, when the first client attaches —
        # never at load time (a drifted workspace with nobody attached must
        # not hold the host open or report "waiting").
        self._drift_checked = False
        # no_session: the client asked for an unsaved conversation — _persist
        # becomes a no-op and nothing is ever written to the session store.
        self.no_save = no_save
        # Content-addressed image blobs for this session's messages.
        self.images = session_store.SessionImageStore(
            self.name, persist=not no_save,
        )
        self.controller.image_store = self.images

        # Debounced background persistence (see _persist). Appends are the
        # common case; a full rewrite happens after anything mutated the
        # already-persisted prefix (truncate, turn stamping, accept).
        self._persist_dirty = False
        # Watermark: the first message index not known to be on disk (None
        # until the first write succeeds).
        self._persist_from: int | None = None
        self._persist_task: asyncio.Task[None] | None = None
        self._diff_task: asyncio.Task[None] | None = None
        self._diff_dirty = False
        # Turns closed by Session.load because a crash left them open; the
        # host stamps the real workspace tree over their placeholder.
        self._crash_repaired: list[TurnCheckpoint] = []
        # Cached latest user send_time (see last_user_message_at); _UNSET
        # until first needed, invalidated when the history changes shape.
        self._last_user_at: str | None = _UNSET

        # Fan-out.
        self._subscribers: list[asyncio.Queue[Event]] = []
        # Live round state sent in snapshots (so attach needs no replay log).
        self._partial_round = protocol.PartialRound()

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
        state: "StoredSession | None" = None,
        idle_callback: "Callable[[Session], None] | None" = None,
    ) -> "Session":
        """Host a saved session: load its conversation and rebind tracking.

        ``state`` must carry an already-loaded StoredSession when called
        off-loop (the host loads it in a worker thread first); the direct
        read here is a convenience for sync callers and blocks.
        """
        if state is None:
            state = session_store.load_state(name)
        session = cls(
            **parts.session_kwargs(name),
            conversation=state.conversation,
            mode=state.mode or "write",
            idle_callback=idle_callback,
        )
        # Restore the persisted token counters so the client's footer shows
        # real in/out/ctx immediately instead of 0 until the next round.
        session.controller.usage_total = state.usage_total
        session.controller.last_usage = state.last_usage
        session._finished_while_away = state.finished_while_away
        # Everything in the state row is on disk: the watermark starts at the
        # end, so the next persist only writes the tail.
        session._persist_from = len(session.conversation.messages)
        # A loaded session can never have a running turn: a crash (or a
        # kill) left any open ones behind. Close them all — a turn left
        # "open" forever makes open_turn()/open_turn_before() return stale
        # trees for diff attribution and drift suppression. Tracked turns
        # get the before tree as a placeholder; the host (which can run a
        # snapshot) stamps the real workspace tree over it via
        # _crash_repaired and marks it seen.
        session._crash_repaired = session.conversation.close_crashed_turns()
        # A session saved mid-turn can carry tool calls with no result; give
        # them synthetic answers so the next request is valid.
        session._repair_tool_calls()
        return session

    # ------------------------------------------------------------------
    # lifecycle

    async def stop(self) -> None:
        """Cancel a running turn or compaction and flush pending writes."""
        self._closed = True
        await _cancel_and_wait(self._turn_task)
        await _cancel_and_wait(self._compact_task)
        self._compact_task = None
        await _cancel_and_wait(self._diff_task)
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
        open event stream.

        A pending drift WITHOUT an attached client does not count: nothing
        can resolve it, so it must neither report "waiting" nor keep the
        host alive (the prompt is (re)computed when a client attaches)."""
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

    @property
    def pending_drift(self) -> bool:
        return self._pending_drift is not None

    def summary(self) -> SessionSummary:
        """The list row for this hosted session."""
        state = self._state
        # A pending ask blocks the turn itself, so it reads as waiting in
        # every state. A pending drift prompt does not stop a running turn:
        # it only takes over an otherwise idle row, so the picker keeps
        # reporting running/compacting while the user has not resolved it.
        if self.pending_ask is not None or (
            self.pending_drift is not None
            and self.clients_attached
            and state == "idle"
        ):
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
            tracking=self.git_info is not None,
        )

    @property
    def conversation(self) -> Conversation:
        return self.controller.conversation

    # ------------------------------------------------------------------
    # fan-out

    def _emit(self, event: Event) -> Event:
        """Fold live round state into the snapshot and push the event."""
        self._track_partial(event)
        for queue in self._subscribers:
            queue.put_nowait(event)
        return event

    def _reset_partial(self) -> None:
        self._partial_round = protocol.PartialRound()

    def _track_partial(self, event: Event) -> None:
        """Maintain the compact live-round state clients need on attach.

        This replaces replaying a potentially unbounded event log: the
        snapshot carries the complete partial answer, never a truncated one.
        """
        partial = self._partial_round
        if isinstance(event, protocol.ReasoningDelta):
            partial.reasoning += event.text
        elif isinstance(event, protocol.ContentDelta):
            partial.content += event.text
        elif isinstance(event, protocol.RoundComplete):
            # The assistant message is in the history now; only any tool
            # calls still executing remain live.
            partial.reasoning = ""
            partial.content = ""
        elif isinstance(event, protocol.ToolCallStarted):
            partial.tool_calls.append(protocol.PartialToolCall(
                call_id=event.call_id, name=event.name,
            ))
        elif isinstance(event, protocol.ToolCallArgumentsDelta):
            call = self._partial_call(event.call_id)
            if call is not None:
                call.arguments += event.text
        elif isinstance(event, protocol.ToolCallArgumentsDone):
            call = self._partial_call(event.call_id)
            if call is not None:
                call.arguments = event.arguments_json
        elif isinstance(event, protocol.ToolOutputDelta):
            call = self._partial_call(event.call_id)
            if call is not None:
                call.content += event.text
        elif isinstance(event, protocol.ToolFinished):
            call = self._partial_call(event.call_id)
            if call is not None:
                call.status = event.status
                call.content = event.content
        elif isinstance(event, protocol.StreamError):
            self._reset_partial()

    def _partial_call(self, call_id: str) -> protocol.PartialToolCall | None:
        return next(
            (c for c in self._partial_round.tool_calls if c.call_id == call_id),
            None,
        )

    def subscribe(self) -> tuple[Snapshot, asyncio.Queue[Event]]:
        """Build the attach snapshot and register a subscriber.

        Synchronous on purpose: no await between building the snapshot and
        registering the queue, so the stream cannot move in between (no
        gap, no duplicate)."""
        snapshot = self.snapshot()
        self._finished_while_away = False
        queue: asyncio.Queue[Event] = asyncio.Queue()
        self._subscribers.append(queue)
        return snapshot, queue

    def unsubscribe(self, queue: asyncio.Queue[Event]) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)
        self._notify_idle()

    def _notify_idle(self) -> None:
        """Tell the host the session may have gone fully inactive (the host
        decides; see Host._on_session_idle). Best-effort, never raises."""
        if self.idle_callback is None:
            return
        try:
            self.idle_callback(self)
        except Exception:
            pass

    @property
    def clients_attached(self) -> int:
        return len(self._subscribers)

    # ------------------------------------------------------------------
    # snapshots and status

    def _config_view(self) -> protocol.SessionConfigView:
        """The slow-changing half of the status (see StatusChanged /
        ConfigChanged): only emitted when a setting actually changes."""
        return protocol.SessionConfigView(
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
            use_container=self.sandbox.use_container,
            enable_tracking=self.git_info is not None,
            supports_vision=self.profile.supports_vision,
            tools=[tool.name for tool in self.tool_registry.tools()],
        )

    def _activity(self) -> protocol.Activity:
        """The fast-changing half of the status."""
        return protocol.Activity(
            state=self._state,
            usage_total=self.controller.usage_total,
            last_usage=self.controller.last_usage,
            queued=[
                protocol.QueuedMessage(text=item.text, images=item.images)
                for item in self._queued
            ],
            pending_ask=self.pending_ask,
            pending_drift=self._pending_drift,
            clients_attached=self.clients_attached,
            finished_while_away=self._finished_while_away,
        )

    def status(self) -> StatusPayload:
        return StatusPayload(config=self._config_view(), activity=self._activity())

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
            partial_round=(
                self._partial_round
                if (self._partial_round.reasoning
                    or self._partial_round.content
                    or self._partial_round.tool_calls)
                else None
            ),
        )

    def last_user_message_at(self) -> str | None:
        """Latest user send_time (derived, never stored separately).

        All messages, not just the visible slice: after a compaction with no
        new user message the visible slice starts at the summary, and the
        session would sort by name instead of recency.

        Cached: summary() runs on every listing of every hosted session,
        and a full scan per call is O(history) for O(1) information. The
        cache is dropped whenever the history changes shape (a new user
        message, a rollback)."""
        if self._last_user_at is _UNSET:
            times = [
                m.timing.send for m in self.conversation.messages
                if m.role == "user" and m.timing.send
            ]
            self._last_user_at = max(times) if times else None
        return self._last_user_at

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
            case protocol.ResolveDrift():
                return await self._cmd_resolve_drift(command)
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
        """Flip the state and push it. Does NOT notify the host of idleness:
        the idle notification can trigger eviction (unload -> stop() ->
        cancel this task), so callers notify explicitly at a point where
        cancellation can no longer skip their cleanup — the very end of
        _run_turn's finally, after the turn's cleanup awaits."""
        if self._state == state:
            return
        self._state = state
        self._emit_status()

    def _emit_status(self, config_changed: bool = False) -> None:
        """Push the fast-changing half to every client; a settings change
        also pushes the config view (before the activity, so a client that
        renders the pair is never stale on the config side)."""
        if config_changed:
            self._emit(protocol.ConfigChanged(config=self._config_view()))
        self._emit(protocol.StatusChanged(activity=self._activity()))

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

    def _commit_checkpoints(self, reason: HistoryReason) -> None:
        """Persist and broadcast a checkpoint rebaseline (accept, reject,
        drift): no message changed, so clients only need to refresh the
        diff/pending view — not the whole transcript."""
        self._persist()
        self._emit(protocol.CheckpointsChanged(reason=reason))

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
        if self.git_info is None or not self._subscribers:
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
        self._turn_task = asyncio.get_running_loop().create_task(
            self._run_turn(text, images)
        )
        # Yield once so the turn task actually starts: a task cancelled
        # before its first step never runs its cleanup (state would stick
        # at "running" forever).
        await asyncio.sleep(0)

    def _cmd_cancel(self) -> CommandResult:
        if self._state == "idle":
            return _ok(cancelled=False)
        if self._turn_task is not None:
            self._turn_task.cancel()
        if self._compact_task is not None:
            self._compact_task.cancel()
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
        self._notify_idle()
        return _ok()

    async def _cmd_resolve_drift(self, command: protocol.ResolveDrift) -> CommandResult:
        if self._pending_drift is None:
            raise SessionError("no drift prompt is pending")
        self._pending_drift = None
        info = self.git_info
        if command.accept and info is not None:
            try:
                await accept_changes(info)
            except GitWorktreeError as exc:
                self._emit(protocol.DriftResolved(accepted=False))
                raise SessionError(f"could not accept the drift: {exc}") from exc
            self.conversation.lock_turns(info.baseline_tree)
            self._commit_checkpoints("drift")
            self._emit(protocol.DriftResolved(accepted=True))
            self._emit_notice(
                "drift accepted: the workspace is the new baseline; /diff and "
                "/undo now measure from here"
            )
        else:
            # Acknowledge the drift durably: remember the tree the user just
            # saw so a reload does not re-detect and re-prompt the same
            # changes. /diff still shows them (the baseline is unchanged).
            if info is not None:
                try:
                    info.mark_seen(await snapshot_state(info))
                except GitWorktreeError:
                    pass
                self._persist()
                await self._flush_persist()
            self._emit(protocol.DriftResolved(accepted=False))
            self._emit_notice(
                "keeping tracking against the last known state — the drift "
                "shows up in /diff"
            )
        self._notify_idle()
        return _ok()

    def _cmd_discard_queued(self) -> CommandResult:
        if not self._queued:
            return _ok(discarded=0)
        discarded = self._queued.pop(0)
        self._emit_status()
        return _ok(discarded=1, text=discarded.text)

    def _cmd_flush_queued(self) -> CommandResult:
        """Send the oldest queued message now: cancel the running turn or
        compaction; the cleanup starts the queued message immediately."""
        if self._state == "idle" or not self._queued:
            return _ok(flushed=False)
        if self._turn_task is not None:
            self._turn_task.cancel()
        if self._compact_task is not None:
            self._compact_task.cancel()
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
            plan = self.controller.rollback_plan_to(message)
            scope = "messages from here on"
        else:
            plan = self.controller.rollback_plan()
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
                await self.controller.revert_changes(plan.revert_turns)
            except RevertConflict as exc:
                raise SessionError(
                    f"/{label} conflict: the turn's changes overlap edits made "
                    f"since — nothing was changed\n{exc}"
                ) from exc
            # A rollback moves the files too: re-account the workspace before
            # persisting, or the reverted state would look like external drift
            # on the next resume.
            if self.git_info is not None:
                try:
                    self.git_info.mark_seen(await snapshot_state(self.git_info))
                except GitWorktreeError:
                    pass
        self.controller.apply_rollback(plan)
        self._last_user_at = _UNSET  # the newest user message may be gone
        # The truncation invalidated rows on disk from the cut onward: move
        # the persist watermark back so the next write replaces them. When
        # nothing has been written yet (watermark None), leave it None: the
        # next write starts at seq 0 and covers everything anyway — setting
        # drop_from here would make rows below the cut never be written.
        if self._persist_from is not None:
            self._persist_from = min(self._persist_from, plan.drop_from)
        if plan.compaction:
            note = "reverted"
        elif self.git_info is None:
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
        self._compact_task = asyncio.get_running_loop().create_task(
            self._run_compact()
        )
        return _ok(started=True)

    async def _run_compact(self) -> None:
        try:
            result = await self.controller.compact()
        except asyncio.CancelledError:
            self._set_state("idle")
            await self._start_queued()
            self._notify_idle()
            raise
        except Exception as exc:
            self._set_state("idle")
            self._emit_notice(f"compaction failed: {exc}", "error")
            await self._start_queued()
            self._notify_idle()
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
        self._notify_idle()

    async def _start_queued(self) -> bool:
        """Start the oldest queued message unless the session is closed."""
        if not self._queued or self._closed:
            return False
        item = self._queued.pop(0)
        self._emit_status()
        await self._start_turn(item.text, item.images)
        return True

    async def _cmd_accept(self) -> CommandResult:
        """Lock in the changes made so far (allowed mid-turn)."""
        info = self.git_info
        if info is None:
            raise SessionError(
                "/accept: no change tracking for this session (the workspace "
                "isn't a git repo, or tracking setup failed)"
            )
        mid_turn = self._state != "idle"
        try:
            stat = await diff_stat(info)
        except GitWorktreeError:
            stat = None
        try:
            sha = await accept_changes(info)
        except GitWorktreeError as exc:
            raise SessionError(f"/accept failed: {exc}") from exc
        self.conversation.lock_turns(sha)
        self._commit_checkpoints("accept")
        if mid_turn:
            self._emit_notice(
                "note: the current turn is still generating — it may keep "
                "editing files on top of the accepted state", "warn",
            )
        if stat is None or not stat.files:
            self._emit_notice(
                "accepted: no pending changes — baseline reset; /diff and /undo "
                "now measure from here"
            )
        else:
            self._emit_notice(
                f"accepted: {len(stat.files)} file(s) +{stat.additions} -{stat.deletions} "
                "locked in; /diff is reset and /undo can no longer revert them"
            )
        return _ok()

    async def _cmd_reject(self) -> CommandResult:
        """Discard every change made since the last /accept (allowed mid-turn)."""
        info = self.git_info
        if info is None:
            raise SessionError(
                "/reject: no change tracking for this session (the workspace "
                "isn't a git repo, or tracking setup failed)"
            )
        mid_turn = self._state != "idle"
        try:
            stat = await diff_stat(info)
            await revert_to_tree(info, info.baseline_tree)
        except GitWorktreeError as exc:
            raise SessionError(f"/reject failed: {exc}") from exc
        self.conversation.lock_turns(info.baseline_tree)
        # The workspace is back at the baseline: that is now the last state
        # the session has accounted for, so a resume must not call the
        # discarded changes drift.
        info.reset_to(info.baseline_tree)
        self._commit_checkpoints("reject")
        if mid_turn:
            self._emit_notice(
                "note: the current turn is still generating — it may keep "
                "editing files on top of the reverted state", "warn",
            )
        if not stat.files:
            self._emit_notice("rejected: nothing to discard")
        else:
            self._emit_notice(
                f"rejected: {len(stat.files)} file(s) +{stat.additions} -{stat.deletions} "
                "since the last accept were reverted"
            )
        return _ok()

    def _cmd_set_mode(self, command: protocol.SetMode) -> CommandResult:
        self.mode = command.mode
        self._rebuild_tools()
        self._emit_status(config_changed=True)
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
        self._emit_status(config_changed=True)
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
        SessionError). Existence is checked only when adding."""
        raw = path.strip()
        if not raw or self.sandbox.settings is None:
            raise SessionError("references need a path and a sandbox")
        if add and not self._ref_path(raw).exists():
            raise SessionError(f"no such path: {raw}")
        return raw

    def _validate_filter(self, path: str, add: bool) -> str:
        """Shared validation for /filter (and PATCH): the normalized
        workspace-relative path, or SessionError."""
        try:
            rel = validate_filter_path(path).strip("/")
        except ValueError as exc:
            raise SessionError(str(exc)) from exc
        if self.git_info is not None and (rel == ".git" or rel.startswith(".git/")):
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

    def _ref_path(self, entry) -> Path:
        """Host path of one ref entry (str or RefSpec)."""
        raw = entry if isinstance(entry, str) else entry.path
        path = Path(raw).expanduser()
        return (path if path.is_absolute() else self.workspace / path).resolve()

    def _profile_refs(self) -> list:
        """The sandbox profile's own refs (never touched by /ref)."""
        settings = self.sandbox.settings
        return list(settings.auto_include_refs) if settings is not None else []

    def _ref_specs(self) -> list:
        """The effective ref list: the profile's refs plus this workspace's."""
        return [*self._profile_refs(), *self.workspace_refs]

    def _ref_infos(self) -> list[protocol.RefInfo]:
        """The refs currently bound, for the client's status payload."""
        external = (
            self.sandbox.config.external_refs
            if self.sandbox.config is not None
            else SandboxConfig.resolve_auto_refs(self.workspace, self._ref_specs())
        )
        return [
            protocol.RefInfo(path=str(ref.host), mount=ref.mount)
            for _, ref in sorted(external.items(), key=lambda item: str(item[1].host))
        ]

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
            session_store.save_workspace_refs(str(self.workspace), specs)
        except OSError as exc:
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
                self.workspace, settings, self.git_info,
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
        self._emit_status(config_changed=True)

    # ------------------------------------------------------------------
    # turn loop

    def _on_user_message(self, message: Message) -> None:
        """Controller hook: a user message landed in the history. This is
        the only way any client learns about it (no local echo)."""
        if message.timing.send is None:
            message.timing.send = _utcnow()
        self._last_user_at = message.timing.send  # newest by construction
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
            stream = self.controller.begin_turn(text, images)
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
            # Record where the turn left the workspace so /undo //retry can
            # reverse exactly this turn's diff later (interruptions included).
            await self.controller.finish_turn()
            self._persist()
            await self._flush_persist()
            # The turn is over: its content lives in the history now.
            self._reset_partial()
            self._set_state("idle")
            await self._emit_diff_changed()
            # After the state flip: TurnEnded describes the end and must not
            # be buffered as a live turn event for the next attach.
            self._emit(protocol.TurnEnded(
                outcome=outcome, elapsed=time.monotonic() - turn_start,
            ))
            self._turn_task = None
            if not self._subscribers:
                # Persisted (not just in memory): the host evicts an idle
                # session right after this, and the next attach must still
                # see "finished while you were away".
                self._finished_while_away = True
                self._persist()

            # A queued message sent at the exact end of the turn (or a
            # "send queued now" cancel) starts immediately. The idle
            # notification is LAST: it can trigger eviction (which cancels
            # this task), so nothing may follow it in this finally.
            if not await self._start_queued():
                self._notify_idle()

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
        result = await self.tool_registry.call(
            call.name, call.arguments_json, output_sink=sink,
        )
        end = _utcnow()
        self._emit(protocol.ToolFinished(
            call_id=call_id, status=result.status, content=result.content,
        ))
        if call.name in MUTATING_TOOLS:
            # Coalesced: a burst of run_bash calls must not trigger a git
            # snapshot each; one trailing debounce covers them all.
            self._schedule_diff_changed()
        # Refresh the footer's in/out/ctx per message, not just per round.
        self._emit_status()
        self.controller.record_tool_result(
            call_id, result, start_time=start, end_time=end,
        )  # Timing wraps the two stamps inside the controller

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
        persist when anything changed.

        The synthetic result is INSERTED at the orphan's position, so a
        mid-history repair shifts rows a tail-only write would not rewrite:
        move the watermark back to the lowest insertion index first."""
        low = self.conversation.repair_orphaned_tool_calls(now=_utcnow())
        if low is None:
            return
        if self._persist_from is not None:
            self._persist_from = min(self._persist_from, low)
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

    def _make_git_guard(self) -> Callable[[str], str | None]:
        if self.sandbox.allow_git_write:
            # The git guard is off entirely: the agent manages git itself;
            # tracking guarantees no longer hold.
            return lambda command: None
        if self.sandbox.git_shim_active:
            # In-container mode: the PATH shim blocks bad git subcommands at
            # execution time, statement by statement (see gitwork.py). The
            # string scanner would only veto whole commands the shim now
            # handles better.
            return lambda command: None

        def guard(command: str) -> str | None:
            info = self.git_info
            if info is None:
                return None
            return check_blocked_git(command)
        return guard

    def turn_checkpoints(self) -> list[TurnCheckpoint]:
        return self.controller.turn_checkpoints()

    async def detect_resume_drift(self) -> "protocol.PendingDrift | None":
        """The drift prompt (last known tree, current tree, changed paths)
        when the workspace changed outside this session since it was last
        open, else None."""
        info = self.git_info
        if info is None:
            return None
        # Compare against the last tree this session ever recorded — including
        # accepted (locked) turns. ``turn_checkpoints()`` deliberately omits
        # locked turns (they are not undoable), but they are still the last
        # known workspace state; using it here made a clean /accept look like
        # external drift on the next resume.
        known = self.conversation.known_trees()
        # ``last_seen_tree`` is maintained at every point the workspace state
        # is deliberately recorded (turn end, accept, keep-tracking, rollback)
        # and persisted, so acknowledged drift is not re-prompted on reload.
        # Older session files have no marker: fall back to the turn history.
        last_known = (
            info.last_seen_tree
            or (known[-1] if known else info.baseline_tree)
        )
        try:
            current = await snapshot_state(info)
            if current == last_known:
                return None
            # A turn may be generating right now (a client attached or
            # reconnected mid-turn). Its in-flight edits are not drift: only
            # report a change if something outside the turn's own
            # before→current diff also moved.
            open_before = self.controller.open_turn_before()
            if open_before is not None:
                since_open = {
                    f.path for f in (
                        await tree_diff_stat(info, open_before, current)
                    ).files
                }
                since_last_known = {
                    f.path for f in (
                        await tree_diff_stat(info, last_known, current)
                    ).files
                }
                if since_last_known <= since_open:
                    return None
        except GitWorktreeError:
            return None
        try:
            files = [
                f.path for f in
                (await tree_diff_stat(info, last_known, current)).files
            ]
        except GitWorktreeError:
            files = []
        return protocol.PendingDrift(
            workspace=str(self.workspace), last_known=last_known,
            current=current, files=files,
        )

    async def prompt_resume_drift(self) -> None:
        """Detect drift and surface it as session state (a pending prompt
        any client can resolve).

        Called lazily when the FIRST client attaches — never at load time:
        a pending drift with nobody attached must not report "waiting" nor
        keep the host alive."""
        if self._drift_checked:
            return
        self._drift_checked = True
        drift = await self.detect_resume_drift()
        if drift is not None and self.git_info is not None:
            # The prompt is session state: the StatusChanged that follows
            # carries it (no separate event — state in two places means two
            # code paths in every client).
            self._pending_drift = drift
            self._emit_status()

    # ------------------------------------------------------------------
    # misc

    def _refresh_system_prompt(self) -> None:
        """Derive the mode-aware system prompt for the next request. It is
        never persisted in the conversation."""
        self.controller.system_prompt = system_prompt_for(prompt_facts(
            self.sandbox.config, self.mode,
            tracked=self.git_info is not None,
            git_write=self.sandbox.allow_git_write,
        ))

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
            ask_callback=self._ask_user,
            git_guard=self._make_git_guard(),
            default_timeout=self._tool_timeout(),
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
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No loop (sync callers): write inline.
                self._persist_dirty = False
                payload, meta = self._persist_state()
                session_store.write_state(self.name, payload, meta)
                self._persist_from = (
                    payload["from_seq"] + len(payload["messages"])
                )
                return
            self._persist_task = loop.create_task(self._persist_loop())

    def _persist_state(self) -> tuple[Any, dict[str, Any]]:
        """Serialize into write instructions; runs on the event loop (pydantic
        dumps copy, so the writer thread only ever touches plain data).

        The watermark (``_persist_from``) is the first message index not yet
        safely on disk. Messages are immutable, so a normal persist only
        sends the tail; a rollback moves the watermark back to the cut.

        The watermark is advanced OPTIMISTICALLY here, at serialize time:
        this is fully synchronous (no awaits), so the captured ``from_seq``
        and the optimistic value describe the same list state. If the write
        fails, the loop resets the watermark back to ``from_seq`` so the
        next persist rewrites from there."""
        stored = StoredSession(
            conversation=self.conversation,
            mode=self.mode,
            profile=self.profile_name,
            workspace=str(self.workspace),
            git=self.git_info,
            sandbox_name=self.sandbox.name,
            sandbox_settings=self.sandbox.settings,
            usage_total=self.controller.usage_total,
            last_usage=self.controller.last_usage,
            finished_while_away=self._finished_while_away,
        )
        from_seq = self._persist_from or 0
        payload, meta = session_store.serialize_state(stored, from_seq)
        self._persist_from = len(self.conversation.messages)  # optimistic
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
                self._persist_from = min(
                    self._persist_from if self._persist_from is not None
                    else payload["from_seq"],
                    payload["from_seq"],
                )
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
        # right after a compaction there is no usage-bearing visible message
        # and this returns False — no loop guard needed.
        visible = self.conversation.visible_messages()
        # Only AssistantMessage carries usage; UserMessage/ToolMessage don't
        # have the field at all.
        last_usage_message_index = next(
            (i for i in range(len(visible) - 1, -1, -1)
             if getattr(visible[i], "usage", None)), -1,
        )
        last_usage_message = visible[last_usage_message_index] if last_usage_message_index >= 0 else None
        last_usage = getattr(last_usage_message, "usage", None) if last_usage_message else None
        used = (last_usage.context_tokens(self.profile.keep_reasoning)
                if last_usage else 0) + (
            # Messages appended after the last usage-bearing round (tool
            # results, steering) are not in that round's input; estimate
            # them, or the threshold check undershoots a long tool loop.
            sum(len(m.content or "") for m in visible[last_usage_message_index + 1:])
            // compaction.CHARS_PER_TOKEN
            if last_usage_message is not None else 0
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
        info = self.git_info
        if info is None:
            return protocol.DiffInfo(tracking=False)
        open_before = self.controller.open_turn_before()
        return await build_diff_info(info, self.turn_checkpoints(), path,
                                     open_before=open_before)
