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

**Fan-out.** :meth:`_emit` stamps every event with a monotonically
increasing ``seq``, appends it to a bounded ring buffer and pushes it to
every subscriber queue. A client attaches through :meth:`subscribe`,
which builds the snapshot synchronously (no await between building it
and registering the subscriber, so no event can slip between the two)
and then replays the live stream.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from . import protocol
from .config import (
    ConfigError, ProviderProfile, SandboxSettings, load_sectioned, resolve_api_key,
)
from .conversation import Conversation, Message, ToolCall
from .conversation import turn_time_summary
from .controller import ChatController, RollbackPlan
from .events import Usage
from .gitwork import (
    GitInfo, GitWorktreeError, RevertConflict, accept_changes,
    attributed_diff_stat, check_blocked_git, diff_stat, git_diff, revert_to_tree,
    snapshot_state,
)
from .images import ImageAttachment
from .prompts import GENERAL_SYSTEM_PROMPT, system_prompt_for
from .protocol import (
    Command, CommandResult, Event, Snapshot, StatusPayload,
)
from .sandbox import (
    SandboxConfig, SandboxSession, SandboxUnavailable, make_sandbox,
)
from . import session_store
from .session_store import SessionState, load_state
from .tools import build_registry
from .util import utcnow_iso as _utcnow

# How many recent cmd_ids are remembered for dedupe (client retries must
# not double-send a command).
CMD_DEDUPE_LIMIT = 128

State = Literal["idle", "running", "compacting"]

# Commands that rewrite conversation/workspace state are rejected while a
# turn or compaction is running.
IDLE_ONLY = (
    protocol.Undo, protocol.Retry, protocol.UndoTo, protocol.Compact,
    protocol.SwitchModel, protocol.SetSandbox,
    protocol.FilterAdd, protocol.FilterRemove,
)


class SessionError(Exception):
    """A command was rejected (busy, unknown target, git failure, …)."""


def _ok(payload: dict[str, Any] | None = None) -> CommandResult:
    """A successful command result. ``cmd_id`` is stamped by the inbox."""
    return CommandResult(cmd_id="", ok=True, payload=payload)


async def _cancel_and_wait(task: asyncio.Task[None] | None) -> None:
    """Cancel a task (if any) and let it unwind, swallowing cancellation."""
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:  # noqa: BLE001 - stopping must never raise
        pass


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
        sandbox: SandboxConfig | None = None,
        sandbox_session: SandboxSession | None = None,
        git_info: GitInfo | None = None,
        mode: Literal["plan", "write"] = "write",
        auto_compact: bool = False,
        tool_output_limit: int | None = None,
        client: Any = None,
        sandbox_name: str | None = None,
        sandbox_settings: "SandboxSettings | None" = None,
        config_path: Path | None = None,
        no_save: bool = False,
        idle_callback: "Callable[[Session], None] | None" = None,
    ) -> None:
        self.name = name
        self.workspace = workspace
        self.profile = profile
        self.profile_name = profile_name
        self.api_key = api_key
        self.config_path = config_path
        self.sandbox = sandbox
        self.sandbox_session = sandbox_session
        self.git_info = git_info
        self.mode: Literal["plan", "write"] = mode
        if sandbox is not None:
            sandbox.read_only = mode != "write"
        self.auto_compact = auto_compact
        # Per-session sandbox settings (resolved at creation from the
        # config's sandbox profiles; fixed except via /sandbox or /filter).
        self.sandbox_name = sandbox_name
        self.sandbox_settings = sandbox_settings
        self.allow_git_write = (
            sandbox_settings.allow_git_write if sandbox_settings else False
        )
        self.tool_registry = build_registry(
            sandbox, sandbox_session, mode=self.mode,
            ask_callback=self._ask_user,
            git_guard=self._make_git_guard(),
        )
        if tool_output_limit is not None:
            self.tool_registry.max_output_chars = tool_output_limit
        self.controller = ChatController(
            profile, api_key, tool_registry=self.tool_registry,
            client=client,
        )
        self.controller.git_info = git_info
        self.controller.on_user_message = self._on_user_message
        if conversation is not None:
            self.controller.conversation = conversation

        # Run state (replaces the TUI's _turn_busy/_compacting flags).
        self._state: State = "idle"
        self._turn_task: asyncio.Task[None] | None = None
        self._compact_task: asyncio.Task[None] | None = None
        # Steering queue: (text, images, send_time) — send_time is when the
        # client sent it, not when it is injected.
        self._queued: list[tuple[str, list[ImageAttachment] | None, str]] = []
        # Pending ask: (ask_id, questions, future). Any client may answer;
        # first answer wins.
        self._pending_ask: tuple[str, list[str], asyncio.Future[str | None]] | None = None
        # Pending resume-drift prompt: (last_known_tree, current_tree).
        self._pending_drift: tuple[str, str] | None = None
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

        # Debounced background persistence (see _persist).
        self._persist_dirty = False
        self._persist_task: asyncio.Task[None] | None = None

        # Fan-out.
        self._subscribers: list[asyncio.Queue[Event]] = []
        self._seq = 0
        self._turn_events: list[Event] = []

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
        profile: ProviderProfile,
        profile_name: str | None,
        api_key: str | None,
        config_path: Path | None = None,
        sandbox: SandboxConfig | None = None,
        sandbox_session: SandboxSession | None = None,
        git_info: GitInfo | None = None,
        auto_compact: bool = False,
        tool_output_limit: int | None = None,
        workspace: Path | None = None,
        sandbox_name: str | None = None,
        sandbox_settings: "SandboxSettings | None" = None,
        state: "SessionState | None" = None,
        idle_callback: "Callable[[Session], None] | None" = None,
    ) -> "Session":
        """Host a saved session: load its conversation and rebind tracking.

        ``state`` may carry an already-loaded SessionState (the host loads it
        to resolve the workspace before calling this); it is loaded here only
        when not given."""
        if state is None:
            state = load_state(name)
        if not state.workspace and workspace is None:
            raise SessionError(
                f"saved session {name!r} has no workspace recorded"
            )
        session = cls(
            name=name,
            workspace=Path(state.workspace) if state.workspace else workspace,
            profile=profile, profile_name=profile_name,
            api_key=api_key, config_path=config_path,
            conversation=state.conversation,
            sandbox=sandbox, sandbox_session=sandbox_session,
            git_info=git_info, auto_compact=auto_compact,
            tool_output_limit=tool_output_limit,
            sandbox_name=sandbox_name, sandbox_settings=sandbox_settings,
            idle_callback=idle_callback,
        )
        session.ensure_system_message()
        # Restore the persisted token counters so the client's footer shows
        # real in/out/ctx immediately instead of 0 until the next round.
        session.controller.usage_total = Usage(
            input_tokens=state.total_in, output_tokens=state.total_out,
        )
        session.controller.last_usage = Usage(
            input_tokens=state.last_in, output_tokens=state.last_out,
        )
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
        await self._flush_persist()
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
        if self._pending_ask is None:
            return None
        ask_id, questions, _ = self._pending_ask
        return protocol.PendingAsk(ask_id=ask_id, questions=questions)

    @property
    def pending_drift(self) -> bool:
        return self._pending_drift is not None

    @property
    def conversation(self) -> Conversation:
        return self.controller.conversation

    # ------------------------------------------------------------------
    # fan-out

    def _emit(self, event: Event) -> Event:
        """Stamp ``seq``, buffer, and push to every subscriber."""
        self._seq += 1
        event.seq = self._seq
        if isinstance(event, protocol.HistoryReplaced):
            # A rewrite invalidates every buffered turn event, and the event
            # itself is a full snapshot: it must never sit in turn_events
            # (an attaching client already gets the fresh history in its
            # snapshot — buffering it would render history twice). The
            # buffer simply resets.
            self._turn_events.clear()
        else:
            # UserMessageAdded is never buffered (the message is already in
            # the conversation — buffering it would render it twice on a
            # mid-turn attach); nor is TurnFinished (a client attaching
            # mid-turn has not seen the turn finish — and one attaching
            # during the end-of-turn unwinding would get a stale notice).
            if self._state != "idle" and not isinstance(
                event, (protocol.UserMessageAdded, protocol.TurnFinished)
            ):
                self._turn_events.append(event)
        for queue in self._subscribers:
            queue.put_nowait(event)
        return event

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
        except Exception:  # noqa: BLE001 - a host hook must not kill turns
            pass

    @property
    def clients_attached(self) -> int:
        return len(self._subscribers)

    # ------------------------------------------------------------------
    # snapshots and status

    def status(self) -> StatusPayload:
        last_usage = self.controller.last_usage
        pending_drift = None
        if self._pending_drift is not None:
            last_known, current = self._pending_drift
            pending_drift = protocol.PendingDrift(
                workspace=str(self.workspace),
                detail=(
                    "the workspace changed while this session was closed "
                    f"(last known tree {last_known[:7]}, now {current[:7]}) — "
                    "accept to make the workspace the new baseline, or keep "
                    "tracking against the last known state"
                ),
            )
        return StatusPayload(
            state=self._state,
            session_name=self.name,
            workspace=str(self.workspace),
            mode=self.mode,
            model_name=self.profile_name or "",
            model_display_name=self.profile.display_name,
            model_id=self.profile.model_id,
            cot_strength=self.profile.cot_strength.value,
            max_context=self.profile.max_context,
            sandbox_name=self.sandbox_name,
            filter_list=(
                list(self.sandbox_settings.filter_list)
                if self.sandbox_settings is not None else []
            ),
            use_container=self.sandbox.use_container if self.sandbox else True,
            enable_tracking=self.git_info is not None,
            total_in=self.controller.usage_total.input_tokens,
            total_out=self.controller.usage_total.output_tokens,
            last_in=last_usage.input_tokens,
            last_out=last_usage.output_tokens,
            context_used=last_usage.input_tokens + last_usage.output_tokens,
            queued=[protocol.QueuedMessage(text=text) for text, _, _ in self._queued],
            pending_ask=self.pending_ask,
            pending_drift=pending_drift,
            clients_attached=self.clients_attached,
            tools=[tool.name for tool in self.tool_registry.tools()],
            finished_while_away=self._finished_while_away,
        )

    def snapshot(self) -> Snapshot:
        """Full client-facing state (synchronous; see :meth:`subscribe`).
        Messages are pydantic models, so they go over the wire as-is."""
        return Snapshot(
            seq=self._seq,
            messages=list(self.conversation.messages),
            compact_snapshot=list(self.conversation.compact_snapshot or []),
            turn_start=self.conversation.turn_start,
            status=self.status(),
            turn_events=list(self._turn_events),
        )

    def last_user_message_at(self) -> str | None:
        """Latest user send_time (derived, never stored separately)."""
        times = [
            m.send_time for m in self.conversation.messages
            if m.role == "user" and m.send_time
        ]
        return max(times) if times else None

    # ------------------------------------------------------------------
    # commands: serialized by the per-session lock

    async def execute(self, cmd_id: str, command: Command) -> CommandResult:
        """Run one command under the session lock and return its result.

        Duplicate ``cmd_id``s return the cached result, so a client retry
        never double-sends."""
        if (cached := self._recent_cmds.get(cmd_id)) is not None:
            return cached
        async with self._cmd_lock:
            if (cached := self._recent_cmds.get(cmd_id)) is not None:
                return cached
            try:
                result = await self._dispatch(command)
            except SessionError as exc:
                result = CommandResult(cmd_id=cmd_id, ok=False, error=str(exc))
            except (GitWorktreeError, RevertConflict) as exc:
                result = CommandResult(cmd_id=cmd_id, ok=False, error=str(exc))
            except Exception as exc:  # noqa: BLE001 - never kill the caller
                result = CommandResult(
                    cmd_id=cmd_id, ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
        result.cmd_id = cmd_id
        self._recent_cmds[cmd_id] = result
        while len(self._recent_cmds) > CMD_DEDUPE_LIMIT:
            self._recent_cmds.popitem(last=False)
        return result

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
            case protocol.Undo():
                return await self._cmd_undo(resend=False)
            case protocol.Retry():
                return await self._cmd_undo(resend=True)
            case protocol.UndoTo():
                return await self._cmd_undo_to(command)
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
            case protocol.FilterAdd():
                return await self._cmd_filter(command.path, True)
            case protocol.FilterRemove():
                return await self._cmd_filter(command.path, False)
            case _:
                raise SessionError(f"unsupported command: {type(command).__name__}")

    # ------------------------------------------------------------------
    # state

    def _set_state(self, state: State) -> None:
        if self._state == state:
            return
        self._state = state
        self._emit_status()
        if state == "idle":
            self._notify_idle()

    def _emit_status(self) -> None:
        self._emit(protocol.StatusChanged(status=self.status()))

    def _emit_notice(self, text: str, level: str = "info") -> None:
        self._emit(protocol.Notice(text=text, level=level))  # type: ignore[arg-type]

    def _emit_history(self) -> None:
        self._emit(protocol.HistoryReplaced(snapshot=self.snapshot()))

    # ------------------------------------------------------------------
    # commands

    async def _cmd_send(self, command: protocol.SendMessage) -> CommandResult:
        # SendMessage.images validates straight into ImageAttachment models.
        images = command.images or None
        if self._state != "idle":
            # Steering: queued until the next round boundary (or turn end).
            self._queued.append((command.text, images, _utcnow()))
            self._emit_status()
            return _ok(payload={"queued": True})
        await self._start_turn(command.text, images)
        return _ok(payload={"started": True})

    async def _start_turn(self, text: str, images: list[ImageAttachment] | None) -> None:
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
            return _ok(payload={"cancelled": False})
        if self._turn_task is not None:
            self._turn_task.cancel()
        if self._compact_task is not None:
            self._compact_task.cancel()
        return _ok(payload={"cancelled": True})

    def _cmd_answer_ask(self, command: protocol.AnswerAsk) -> CommandResult:
        pending = self._pending_ask
        if pending is None or pending[0] != command.ask_id:
            raise SessionError("no ask is pending with that id")
        _, _, future = pending
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
        last_known, _current = self._pending_drift
        self._pending_drift = None
        info = self.git_info
        if command.accept and info is not None:
            try:
                await accept_changes(info)
            except GitWorktreeError as exc:
                self._emit(protocol.DriftResolved(accepted=False))
                raise SessionError(f"could not accept the drift: {exc}") from exc
            self._rewrite_checkpoints(info.baseline_tree)
            self._persist()
            self._emit_history()
            self._emit(protocol.DriftResolved(accepted=True))
            self._emit_notice(
                "drift accepted: the workspace is the new baseline; /diff and "
                "/undo now measure from here"
            )
        else:
            self._emit(protocol.DriftResolved(accepted=False))
            self._emit_notice(
                "keeping tracking against the last known state — the drift "
                "shows up in /diff"
            )
        self._notify_idle()
        return _ok()

    def _cmd_discard_queued(self) -> CommandResult:
        if not self._queued:
            return _ok(payload={"discarded": 0})
        text, _, _ = self._queued.pop(0)
        self._emit_status()
        return _ok(payload={"discarded": 1, "text": text})

    def _cmd_flush_queued(self) -> CommandResult:
        """Send the oldest queued message now: cancel the running turn; the
        turn loop's cleanup starts the queued message immediately."""
        if self._state == "idle" or not self._queued:
            return _ok(payload={"flushed": False})
        if self._turn_task is not None:
            self._turn_task.cancel()
        return _ok(payload={"flushed": True})

    async def _cmd_undo(self, resend: bool) -> CommandResult:
        """Shared body of /undo (drop the last turn, put its user message
        back in the input) and /retry (drop it and resend it)."""
        label = "retry" if resend else "undo"
        plan = self.controller.rollback_plan()
        if plan is None:
            self._emit_notice(f"nothing to {label}")
            return _ok(payload={"user_text": None})
        result = await self._apply_rollback(plan, resend=resend)
        return _ok(payload=result)

    async def _cmd_undo_to(self, command: protocol.UndoTo) -> CommandResult:
        message = self._message_by_id(command.message_id)
        if message is None:
            raise SessionError("no user message with that id in this session")
        plan = self.controller.rollback_plan_to(message)
        if plan is None:
            self._emit_notice("nothing to undo there")
            return _ok(payload={"user_text": None})
        result = await self._apply_rollback(plan, resend=False, scope="messages from here on")
        return _ok(payload=result)

    def _message_by_id(self, message_id: str) -> Message | None:
        for message in self.conversation.messages:
            if message.id == message_id:
                return message
        snapshot = self.conversation.compact_snapshot or []
        for message in snapshot:
            if message.id == message_id:
                return message
        return None

    async def _apply_rollback(
        self, plan: RollbackPlan, resend: bool, scope: str = "last turn",
    ) -> dict[str, Any]:
        """Carry out a rollback plan: rewrite the history, revert the files,
        broadcast HistoryReplaced, and report what happened."""
        if plan.compaction_only:
            self.controller.apply_rollback(plan)
            self._persist()
            self._emit_history()
            self._emit_notice(
                "compaction undone: pre-compaction history restored"
                + (" — nothing to retry (the turn is still in the conversation)" if resend else "")
            )
            return {"user_text": None}
        label = "retry" if resend else "undo"
        try:
            await self.controller.revert_changes(plan.revert_turns)
        except RevertConflict as exc:
            raise SessionError(
                f"/{label} conflict: the turn's changes overlap edits made "
                f"since — nothing was changed\n{exc}"
            ) from exc
        self.controller.apply_rollback(plan)
        self._persist()
        self._emit_history()
        if self.git_info is None:
            note = "no git tracking for this session — file edits could not be reverted"
        elif not plan.revert_turns:
            note = "no file edits to revert"
        else:
            note = "file edits reverted"
        if resend:
            self._emit_notice(f"retrying: {scope} rolled back ({note})")
            await self._start_turn(plan.user_text, None)
            return {"user_text": None}
        self._emit_notice(f"undone: {scope} removed ({note})")
        return {"user_text": plan.user_text}

    async def _cmd_compact(self) -> CommandResult:
        """Start a manual compaction as a task: the inbox stays free (cancel
        works, commands keep flowing); the task drains the steering queue
        when it finishes."""
        self._set_state("compacting")
        self._compact_task = asyncio.get_running_loop().create_task(
            self._run_compact()
        )
        return _ok(payload={"started": True})

    async def _run_compact(self) -> None:
        try:
            summary = await self.controller.compact()
        except RuntimeError as exc:
            self._set_state("idle")
            self._emit_notice(f"compaction failed: {exc}", "error")
            return
        except asyncio.CancelledError:
            self._set_state("idle")
            raise
        self._persist()
        self._set_state("idle")
        self._emit_history()
        counts = self._compaction_counts()
        text = f"compacted: {counts}\n{summary}" if counts else summary
        self._emit_notice(text)
        # Drain the steering queue: a message sent during compaction starts
        # now, exactly like at the end of a turn.
        if self._queued:
            queued_text, queued_images, _ = self._queued.pop(0)
            self._emit_status()
            await self._start_turn(queued_text, queued_images)

    def _compaction_counts(self) -> str | None:
        counts = self.controller.last_compaction
        if counts is None:
            return None
        before, after = counts
        return f"{before:,} → {after:,} tokens (freed {max(0, before - after):,})"

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
        self._rewrite_checkpoints(sha)
        self._persist()
        self._emit_history()
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
        self._rewrite_checkpoints(info.baseline_tree)
        self._persist()
        self._emit_history()
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
        self._emit_status()
        return _ok(payload={"mode": self.mode})

    async def _cmd_switch_model(self, command: protocol.SwitchModel) -> CommandResult:
        if self.config_path is None:
            raise SessionError("/model unavailable: no config path known")
        try:
            _, profile = load_sectioned(self.config_path).resolve_model(command.name)
        except ConfigError as exc:
            raise SessionError(f"/model failed: {exc}") from exc
        try:
            api_key = resolve_api_key(profile)
        except ConfigError as exc:
            raise SessionError(f"could not switch model: {exc}") from exc
        self.controller.switch_profile(profile, api_key)
        self.profile = profile
        self.profile_name = command.name
        self.auto_compact = profile.auto_compact
        self._persist()
        self._emit_status()
        return _ok(payload={"model": command.name})

    async def _cmd_set_sandbox(self, command: protocol.SetSandbox) -> CommandResult:
        """Switch this session's sandbox profile (idle only): the shell
        restarts, all subscribers get a Notice, and the change is saved in
        the session file — not in the config file."""
        name, settings = self._resolve_sandbox_profile(command.name)
        await self._apply_sandbox_settings(name, settings)
        self._emit_notice(
            f"sandbox switched to {name!r} — the persistent shell restarted"
        )
        return _ok(payload={"sandbox": name})

    async def _cmd_filter(self, path: str, add: bool) -> CommandResult:
        """Add/remove a mount exclusion (idle only). Saved in the session
        file; the shell restarts so the new mounts take effect."""
        rel = path.strip().strip("/")
        if not rel or ".." in Path(rel).parts:
            raise SessionError(f"invalid filter path: {path!r}")
        if self.git_info is not None and (rel == ".git" or rel.startswith(".git/")):
            raise SessionError(
                ".git cannot be filtered while change tracking is on — the "
                "tracking dir must stay mounted"
            )
        if add and not (self.workspace / rel).exists():
            # Nothing to exclude: filtering a nonexistent path would either
            # fail at argv-build time or plant a stray path in the worktree.
            raise SessionError(f"no such path in the workspace: {rel!r}")
        settings = self.sandbox_settings
        if settings is None:
            raise SessionError("this session has no sandbox settings to filter")
        filters = list(settings.filter_list)
        if add:
            if rel not in filters:
                filters.append(rel)
        elif rel in filters:
            filters.remove(rel)
        settings = settings.model_copy(update={"filter_list": filters})
        await self._apply_sandbox_settings(self.sandbox_name, settings)
        verb = "added to" if add else "removed from"
        self._emit_notice(
            f"{rel} {verb} the sandbox mount exclusions — the persistent "
            "shell restarted"
        )
        return _ok(payload={"filter_list": filters})

    def _resolve_sandbox_profile(self, name: str | None):
        """Resolve a sandbox profile from the config (with inheritance)."""
        if self.config_path is None:
            raise SessionError("no config path known — cannot resolve sandbox profiles")
        try:
            sectioned = load_sectioned(self.config_path)
        except ConfigError as exc:
            raise SessionError(str(exc)) from exc
        return sectioned.resolve_sandbox(name)

    async def _apply_sandbox_settings(self, name: str | None, settings) -> None:
        """Rebuild the sandbox + registry from new settings (idle only)."""
        self.sandbox_name = name
        self.sandbox_settings = settings
        self.allow_git_write = settings.allow_git_write
        old_session = self.sandbox_session
        try:
            sandbox, session_obj = make_sandbox(self.workspace, settings, self.git_info)
        except SandboxUnavailable as exc:
            sandbox = session_obj = None
            self._emit_notice(f"warning: filesystem/bash tools disabled: {exc}", "warn")
        if old_session is not None:
            await old_session.close()
        self.sandbox = sandbox
        self.sandbox_session = session_obj
        if settings.tool_output_limit is not None:
            self.tool_registry.max_output_chars = settings.tool_output_limit
        self._rebuild_tools()
        self._persist()
        self._emit_status()

    # ------------------------------------------------------------------
    # turn loop

    def _on_user_message(self, message: Message) -> None:
        """Controller hook: a user message landed in the history. This is
        the only way any client learns about it (no local echo)."""
        if message.send_time is None:
            message.send_time = _utcnow()
        self._emit(protocol.UserMessageAdded(
            text=message.content,
            images=message.images,
            message_id=message.id,
            send_time=message.send_time,
        ))

    async def _run_turn(self, text: str, images: list[ImageAttachment] | None) -> None:
        """Drive the full exchange: rounds of (thinking → answer → tools).

        Runs as its own task, started by the inbox's send_message handler —
        so the inbox stays free to dispatch accept/reject/answer_ask while
        the turn streams."""
        self._set_state("running")
        self._turn_events.clear()
        turn_start = time.monotonic()
        try:
            stream = self.controller.send(text, images)
            while True:
                done_calls: list[ToolCall] = []
                async for event in stream:
                    if isinstance(event, protocol.ToolCallArgumentsDone):
                        done_calls.append(ToolCall(
                            call_id=event.call_id, name=event.name,
                            arguments_json=event.arguments_json,
                        ))
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
                if not done_calls:
                    break
                # Execute the calls this round requested, in stream order.
                for call in done_calls:
                    await self._execute_call(call)
                # Steering: anything queued while this round streamed is
                # injected here — after the tool answers, before the next
                # LLM call.
                self._inject_queued()
                # Round boundary: everything so far is in the history (the
                # assistant message was stamped, tool results recorded), so
                # the round buffer resets — a client attaching between
                # rounds renders the history only, never a replay of a
                # round the snapshot already contains.
                self._turn_events.clear()
                stream = self.controller.continue_after_tools()
            # The turn's content is in the history now; the buffer must be
            # empty before TurnFinished, or a client attaching during the
            # end-of-turn unwinding would replay the round on top of the
            # snapshot messages (double render).
            self._turn_events.clear()
            summary = turn_time_summary(self.conversation.messages)
            if summary is not None:
                # Stamp the turn's last assistant message: the line persists
                # (reloads render it) and every client derives the same one.
                last_assistant = next(
                    (m for m in reversed(self.conversation.messages)
                     if m.role == "assistant"),
                    None,
                )
                if last_assistant is not None:
                    last_assistant.time_summary = summary
            self._emit(protocol.TurnFinished(
                elapsed=time.monotonic() - turn_start,
                summary=summary or "",
            ))
            # Auto-compact fires at the turn boundary: the history has no
            # pending tool calls to break pairing.
            await self._maybe_auto_compact()
        except asyncio.CancelledError:
            # Keep whatever the model had produced before the interruption.
            if self.controller.save_interrupted_round():
                self._persist()
            self._repair_tool_calls()
            self._emit_notice("interrupted", "error")
        finally:
            # Record where the turn left the workspace so /undo //retry can
            # reverse exactly this turn's diff later (interruptions included).
            await self.controller.finish_turn()
            self._persist()
            await self._flush_persist()
            # The turn is over: its events live in the history now. Leaving
            # them buffered would make the next attach replay the finished
            # turn on top of the snapshot messages (double render).
            self._turn_events.clear()
            self._set_state("idle")
            self._turn_task = None
            if not self._subscribers:
                self._finished_while_away = True
            # A queued message sent at the exact end of the turn (or a
            # "send queued now" cancel) starts immediately.
            if self._queued:
                queued_text, queued_images, _ = self._queued.pop(0)
                self._emit_status()
                await self._start_turn(queued_text, queued_images)
            else:
                self._notify_idle()

    async def _execute_call(self, call: ToolCall) -> None:
        """Run one tool call, streaming its output as ToolOutputDelta."""
        self._emit(protocol.ToolExecStarted(call_id=call.call_id))
        start = _utcnow()
        call_id = call.call_id

        def sink(text: str) -> None:
            self._emit(protocol.ToolOutputDelta(call_id=call_id, text=text))

        # ToolRegistry.call already converts any tool exception into a
        # failed ToolResult, so no try/except is needed here.
        result = await self.tool_registry.call(
            call.name, call.arguments_json, output_sink=sink,
        )
        end = _utcnow()
        self._emit(protocol.ToolFinished(
            call_id=call_id, ok=result.ok, parse_error=result.parse_error,
            output=result.output, error=result.error, header=result.header,
        ))
        # Refresh the footer's in/out/ctx per message, not just per round.
        self._emit_status()
        self.controller.record_tool_result(
            call_id, result, start_time=start, end_time=end,
        )
        self._persist()

    def _inject_queued(self) -> None:
        """Inject the steering queue at a round boundary: the messages land
        in the history (with their original send_time) and fire the
        on_user_message hook, so the next LLM call sees them."""
        if not self._queued:
            return
        for text, images, send_time in self._queued:
            self.controller.inject_user_message(text, images, send_time=send_time)
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
        """Callback for the ask tool: emit AskRequested and wait for an
        ``answer_ask`` command from any attached client. If nobody is
        attached it waits indefinitely — the session is headless and the
        turn keeps its state."""
        ask_id = uuid.uuid4().hex
        future: asyncio.Future[str | None] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending_ask = (ask_id, questions, future)
        self._emit(protocol.AskRequested(ask_id=ask_id, questions=questions))
        self._emit_status()
        try:
            return await future
        finally:
            self._pending_ask = None
            self._emit_status()

    # ------------------------------------------------------------------
    # git tracking

    def _make_git_guard(self) -> Callable[[str], str | None]:
        if self.allow_git_write:
            # The git guard is off entirely: the agent manages git itself;
            # tracking guarantees no longer hold.
            return lambda command: None

        def guard(command: str) -> str | None:
            info = self.git_info
            if info is None:
                return None
            return check_blocked_git(command)
        return guard

    def _rewrite_checkpoints(self, sha: str) -> None:
        """Point every turn's checkpoints at ``sha`` (the accepted tree), so
        /undo //retry can no longer revert file state past an accept."""
        for message in self.conversation.messages:
            if message.role == "user" and message.checkpoint_sha is not None:
                message.checkpoint_sha = sha
                message.after_tree = sha
        if self.conversation.compact_snapshot:
            for message in self.conversation.compact_snapshot:
                if message.role == "user" and message.checkpoint_sha is not None:
                    message.checkpoint_sha = sha
                    message.after_tree = sha

    def turn_checkpoints(self):
        return self.controller.turn_checkpoints()

    async def diff_summary(self) -> protocol.DiffInfo | None:
        """The pending-changes summary (plain drift, no attribution split in
        the wire model — files are listed as they are)."""
        info = self.git_info
        if info is None:
            return None
        try:
            return await attributed_diff_stat(info, self.turn_checkpoints())
        except GitWorktreeError:
            return None

    async def diff_text(self, path: str | None) -> str:
        info = self.git_info
        if info is None:
            raise SessionError(
                "no change tracking for this session — the workspace could "
                "not be tracked with git"
            )
        try:
            return await git_diff(info, path)
        except GitWorktreeError as exc:
            raise SessionError(f"diff failed: {exc}") from exc

    async def detect_resume_drift(self) -> tuple[str, str] | None:
        """(last_known, current) when the workspace changed outside this
        session since it was last open, else None."""
        info = self.git_info
        if info is None:
            return None
        turns = self.turn_checkpoints()
        last_known = turns[-1].after_tree if turns else info.baseline_tree
        try:
            current = await snapshot_state(info)
        except GitWorktreeError:
            return None
        if current == last_known:
            return None
        return last_known, current

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
            self._pending_drift = drift
            self._emit_status()

    # ------------------------------------------------------------------
    # misc

    def ensure_system_message(self) -> None:
        """Make sure the conversation leads with the mode-aware system prompt,
        updated in place (exactly one, at index 0)."""
        conversation = self.conversation
        content = system_prompt_for(GENERAL_SYSTEM_PROMPT, self.mode, self.sandbox)
        if conversation.messages and conversation.messages[0].role == "system":
            conversation.messages[0].content = content
        else:
            conversation.messages.insert(0, Message(role="system", content=content))

    def _rebuild_tools(self) -> None:
        """Apply the current mode/sandbox to the tooling: read-only flag,
        tool registry, and the mode-aware system prompt."""
        if self.sandbox is not None:
            self.sandbox.read_only = self.mode != "write"
        self._rebuild_registry()
        self.ensure_system_message()

    def _rebuild_registry(self) -> None:
        self.tool_registry = build_registry(
            self.sandbox, self.sandbox_session, mode=self.mode,
            ask_callback=self._ask_user,
            git_guard=self._make_git_guard(),
        )
        self.controller.tools = self.tool_registry

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
                self._write_state()
                return
            self._persist_task = loop.create_task(self._persist_loop())

    def _persist_state(self) -> SessionState:
        return SessionState(
            conversation=self.conversation,
            profile=self.profile_name,
            workspace=str(self.workspace),
            git=self.git_info.to_block() if self.git_info else None,
            sandbox_name=self.sandbox_name,
            sandbox_settings=(
                self.sandbox_settings.model_dump()
                if self.sandbox_settings is not None else None
            ),
            total_in=self.controller.usage_total.input_tokens,
            total_out=self.controller.usage_total.output_tokens,
            last_in=self.controller.last_usage.input_tokens,
            last_out=self.controller.last_usage.output_tokens,
        )

    def _write_state(self) -> None:
        session_store.save_state(self.name, self._persist_state())

    async def _persist_loop(self) -> None:
        while self._persist_dirty:
            self._persist_dirty = False
            await asyncio.to_thread(session_store.save_state, self.name, self._persist_state())

    async def _flush_persist(self) -> None:
        """Await completion of any pending background write."""
        if self._persist_task is not None:
            try:
                await self._persist_task
            except asyncio.CancelledError:
                pass
            self._persist_task = None

    async def _maybe_auto_compact(self) -> None:
        if not self.auto_compact:
            return
        last_usage = next(
            (m.usage for m in reversed(self.conversation.messages) if m.usage), None
        )
        used = (last_usage.input_tokens if last_usage else 0) + (
            last_usage.output_tokens if last_usage else 0
        )
        if used < self.profile.auto_compact_threshold * self.profile.max_context:
            return
        # Stay in a non-idle state: only the turn's finally sets idle, so a
        # send arriving during the compaction (or the finish_turn unwind) is
        # queued, never started mid-teardown.
        self._set_state("compacting")
        try:
            await self.controller.compact()
        except RuntimeError as exc:
            self._emit_notice(f"auto-compact failed: {exc}", "warn")
            return
        self._persist()
        await self._flush_persist()
        self._emit_history()
        counts = self._compaction_counts()
        self._emit_notice(f"auto-compacted: {counts}" if counts else "auto-compacted: nothing to compact yet")

    # ------------------------------------------------------------------
    # stats row

    async def diff_info(self, path: str | None) -> protocol.DiffInfo:
        """The /diff payload: per-file stat rows (+ the unified diff text
        for a per-file click-through; the summary poll stays cheap)."""
        info = await self.diff_summary()
        if info is None:
            return protocol.DiffInfo()
        if path is not None:
            files = [f for f in info.files if f.path == path]
            filtered = protocol.DiffInfo(
                files=files,
                additions=sum(f.additions for f in files),
                deletions=sum(f.deletions for f in files),
            )
            filtered.text = await self.diff_text(path)
            return filtered
        return info
