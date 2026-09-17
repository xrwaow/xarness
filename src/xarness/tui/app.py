"""The xarness chat application.

Owns the event loop between :class:`ChatController` (conversation + API) and
the widget tree. All state transitions — processing → thinking → answer
streaming — are driven by stream events, never by timers.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Literal

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static, TextArea
from textual.worker import Worker

from .. import theme
from ..config import ConfigError, ProviderProfile, load_all_profiles, resolve_api_key
from ..conversation import Conversation, Message
from ..controller import ChatController
from ..events import (
    ContentDelta, ReasoningDelta, StreamError, ToolCallArgumentsDone,
    ToolCallStarted, ToolCallStatus, TurnComplete, Usage,
)
from ..file_search import search_files
from ..gitwork import (
    GitInfo, GitWorktreeError, MergeOutcome, check_blocked_git, commit_worktree_changes,
    delete_branch, diff_stat, git_diff, merge_branch, recreate_worktree,
    remove_worktree, worktree_is_valid,
)
from ..prompts import GENERAL_SYSTEM_PROMPT, system_prompt_for
from ..sandbox import SandboxConfig, SandboxSession, SandboxUnavailable
from ..tools import ToolRegistry, build_registry
from .confirm_screen import ConfirmScreen
from .picker_screen import PickerScreen
from .resume_screen import ResumeScreen
from .widgets import (
    AskBar, AssistantMessage, ChatInput, DiffSummary, ErrorLine, NoticeLine,
    PendingIndicator, ShimmerText, StatusBar, SuggestionPopup, ThinkingBlock,
    ToolCallBlock, ToolWritingIndicator, UserMessage, _format_duration,
)


INPUT_PLACEHOLDER = (
    "Send a message…  (Enter: send · Shift+Enter: newline · Ctrl+T: thoughts)"
)
ASK_PLACEHOLDER = "Type your answer…  (Enter: send · Esc: skip)"


SLASH_COMMANDS = [
    ("tools", "list available tools"),
    ("model", "choose what model and reasoning effort to use"),
    ("sessions", "resume a previous session"),
    ("mode", "switch between plan (read-only) and write mode"),
    ("theme", "choose a color theme"),
    ("new", "start a new chat"),
    ("diff", "show pending changes (optionally: /diff <path>)"),
    ("accept", "merge the agent's worktree changes into your branch"),
    ("reject", "discard the agent's worktree changes (asks confirmation)"),
    ("undo", (
        "drop the last turn: revert its file edits, put your message back "
        "in the input (run_bash side effects are not undone)"
    )),
    ("retry", (
        "revert the last turn's file edits and resend your message "
        "(run_bash side effects are not undone)"
    )),
]


class AgentApp(App[None]):
    CSS_PATH = "styles.tcss"
    TITLE = "xarness"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+t", "toggle_thoughts", "Thoughts", priority=True),
        Binding("ctrl+c", "copy_or_quit", "Copy / Quit", priority=True),
        Binding("escape", "interrupt", "Interrupt", priority=True),
    ]

    def __init__(
        self,
        profile: ProviderProfile,
        api_key: str | None,
        controller: ChatController | None = None,
        tool_registry: ToolRegistry | None = None,
        workspace: Path | None = None,
        session_name: str | None = None,
        config_path: Path | None = None,
        profile_name: str | None = None,
        sandbox: SandboxConfig | None = None,
        sandbox_session: SandboxSession | None = None,
        git_info: GitInfo | None = None,
        startup_notices: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.profile = profile
        self.api_key = api_key
        # Kept on the app so /mode can rebuild a registry without re-deriving
        # the sandbox setup.
        self.sandbox = sandbox
        self.sandbox_session = sandbox_session
        self.mode: Literal["plan", "write"] = "write"
        self.tool_registry = tool_registry or build_registry(
            sandbox, sandbox_session, mode=self.mode,
            ask_callback=self._ask_user, compact_callback=self._compact_conversation,
            git_guard=self._make_git_guard(), git_info=git_info,
        )
        self.controller = controller or ChatController(
            profile, api_key, tool_registry=self.tool_registry
        )
        self.controller.git_info = git_info
        self.workspace = workspace
        self.session_name = session_name
        self.config_path = config_path
        self.profile_name = profile_name
        self.git_info = git_info
        self.startup_notices = startup_notices or []
        self.total_in = 0
        self.total_out = 0
        self.last_in = 0
        self._turn_busy = False
        self._queued: list[str] = []
        self._ask_future: asyncio.Future[str | None] | None = None
        self._last_thinking: ThinkingBlock | None = None
        self._worker: Worker | None = None
        self._at_search_timer = None
        self._profiles_cache: dict[str, ProviderProfile] = {}
        self.ensure_system_message()

    def ensure_system_message(self) -> None:
        """Make sure the conversation leads with the mode-aware system prompt.

        Called at startup, after a conversation swap (/resume, cli.py's
        existing-session load), and on /mode. An existing system message is
        updated IN PLACE rather than appended: there is exactly one, at index
        0, and rewriting it means a mode switch applies from the very next
        round onward without duplicating prompts in the history.
        """
        conversation = self.controller.conversation
        content = system_prompt_for(GENERAL_SYSTEM_PROMPT, self.mode)
        if conversation.messages and conversation.messages[0].role == "system":
            conversation.messages[0].content = content
        else:
            conversation.messages.insert(0, Message(role="system", content=content))

    def get_css_variables(self) -> dict[str, str]:
        return {**super().get_css_variables(), **theme.CSS_VARIABLES}

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat-log")
        with Container(id="input-wrap"):
            yield DiffSummary(id="diff-summary")
            yield AskBar(id="ask-bar")
            yield SuggestionPopup(id="suggestion-popup")
            yield ChatInput(
                id="chat-input",
                placeholder=INPUT_PLACEHOLDER,
            )
        yield StatusBar(id="status-bar")

    async def on_mount(self) -> None:
        self.query_one("#chat-input", ChatInput).focus()
        self._refresh_status()
        # Follow new output only while the user is parked at the bottom.
        # Textual's anchor does exactly that: the log stays pinned to the end as
        # content streams, the anchor is released the moment the user scrolls up
        # (so nothing yanks the viewport back), and it re-arms once they scroll
        # down to the bottom again.
        self.query_one("#chat-log", VerticalScroll).anchor()
        # A conversation loaded before mount (e.g. `--session <name>` resume
        # in cli.py) has never been rendered — replay it into the chat log.
        if any(m.role != "system" for m in self.controller.conversation.messages):
            await self._render_history()
        for notice in self.startup_notices:
            self._post_line(NoticeLine(notice))
        # On a resumed session this reconnects the pending-changes summary.
        await self.refresh_diff_summary()

    # ------------------------------------------------------------------
    # Status bar

    def _refresh_status(self) -> None:
        context_used = self.last_in + self.total_out
        self.query_one("#status-bar", StatusBar).update_status(
            shown_name=self.profile.display_name,
            effort=self.profile.cot_strength.value,
            mode=self.mode,
            total_in=self.total_in,
            total_out=self.total_out,
            context_used=context_used,
            max_context=self.profile.max_context,
        )

    # ------------------------------------------------------------------
    # Submission / turn lifecycle

    def on_chat_input_chat_submitted(self, event: ChatInput.ChatSubmitted) -> None:
        # While the ask tool waits for an answer, every submission is the
        # answer — never a chat message or slash command.
        if self._ask_future is not None and not self._ask_future.done():
            self._ask_future.set_result(event.text)
            return
        text = event.text.strip()
        if text.startswith("/"):
            self._handle_slash_command(text)
            return
        self._submit(text)

    def _handle_slash_command(self, text: str) -> None:
        parts = text[1:].split(maxsplit=1)
        cmd = parts[0] if parts else ""
        if cmd == "tools":
            names = ", ".join(t["function"]["name"] for t in self.tool_registry.schema())
            self._post_line(ErrorLine(f"tools: {names}"))
        elif cmd == "sessions":
            self.push_screen(ResumeScreen(), self._on_session_selected)
        elif cmd == "new":
            self._start_new_session()
        elif cmd == "accept":
            self._start_accept()
        elif cmd == "reject":
            self._start_reject()
        elif cmd == "diff":
            arg = parts[1].strip() if len(parts) > 1 else ""
            self._show_diff_command(arg)
        elif cmd == "model":
            if self.config_path is None:
                self._post_line(ErrorLine("/model unavailable: no config path known"))
                return
            try:
                self._profiles_cache = load_all_profiles(self.config_path)
            except ConfigError as exc:
                self._post_line(ErrorLine(f"/model failed: {exc}"))
                return
            self.push_screen(
                PickerScreen(
                    list(self._profiles_cache),
                    current=self.profile_name,
                    title="Switch model",
                ),
                self._on_model_selected,
            )
        elif cmd == "mode":
            new_mode = "plan" if self.mode == "write" else "write"
            self.mode = new_mode
            self.tool_registry = build_registry(
                self.sandbox, self.sandbox_session, mode=new_mode,
                ask_callback=self._ask_user, compact_callback=self._compact_conversation,
                git_guard=self._make_git_guard(), git_info=self.git_info,
            )
            self.controller.tools = self.tool_registry  # rewire to the new registry
            self.ensure_system_message()
            self._refresh_status()
        elif cmd == "undo":
            self._run_undo(resend=False)
        elif cmd == "retry":
            self._run_undo(resend=True)
        elif cmd == "theme":
            arg = parts[1].strip() if len(parts) > 1 else ""
            if arg:
                self._handle_theme_command(arg)
            else:
                # No argument: open the picker, same interaction as /model.
                self.push_screen(
                    PickerScreen(
                        list(theme.THEMES),
                        current=theme.CURRENT_THEME,
                        title="Color theme",
                    ),
                    self._on_theme_selected,
                )
        else:
            self._post_line(ErrorLine(f"unknown command: /{cmd}"))

    # ------------------------------------------------------------------
    # /undo + /retry

    def _sync_totals_from_controller(self) -> None:
        """Mirror the controller's cumulative usage into the status bar."""
        self.total_in = self.controller.usage_total.input_tokens
        self.total_out = self.controller.usage_total.output_tokens
        last = next(
            (m.usage for m in reversed(self.controller.conversation.messages) if m.usage), None
        )
        self.last_in = last.input_tokens if last else 0

    @work(group="undo", exclusive=True)
    async def _run_undo(self, resend: bool) -> None:
        """Shared body of /undo (drop the last turn, put its user message
        back in the input) and /retry (drop it and resend it).

        Only file edits are reverted — the worktree is reset to the turn's
        checkpoint. Anything run_bash did beyond the worktree (installs,
        background jobs, network calls) is not undone."""
        label = "retry" if resend else "undo"
        if self._turn_busy:
            self._post_line(ErrorLine(f"/{label}: wait for the current turn to finish first"))
            return
        plan = self.controller.rollback_plan()
        if plan is None:
            self._post_line(NoticeLine(f"nothing to {label}"))
            return
        sha = self.controller.apply_rollback(plan)
        reverted = await self.controller.revert_worktree(sha)
        await self._render_history()
        self._sync_totals_from_controller()
        self._refresh_status()
        await self.refresh_diff_summary()
        if reverted:
            note = "file edits reverted"
        elif self.git_info is None:
            note = "no git isolation for this session — file edits could not be reverted"
        else:
            note = "file edits could not be reverted (no checkpoint or git failed)"
        if resend:
            self._post_line(NoticeLine(f"retrying: last turn rolled back ({note})"))
            self._submit(plan.user_text)
        else:
            chat_input = self.query_one("#chat-input", ChatInput)
            chat_input.load_text(plan.user_text)
            chat_input.focus()
            self._post_line(NoticeLine(f"undone: last turn removed ({note})"))

    # ------------------------------------------------------------------
    # Git worktree state (diff summary + accept/reject)

    def _make_git_guard(self) -> Callable[[str], str | None]:
        """run_bash veto closure; reads self.git_info at call time so /resume
        rebinding takes effect without rebuilding the guard."""

        def guard(command: str) -> str | None:
            info = self.git_info
            if info is None:
                return None
            return check_blocked_git(command, info.branch, info.agent_workspace)

        return guard

    async def refresh_diff_summary(self) -> None:
        """Recompute the pending-changes bar (hidden when empty or gitless)."""
        try:
            widget = self.query_one("#diff-summary", DiffSummary)
        except NoMatches:
            return  # app shutting down; DOM already pruned
        info = self.git_info
        if info is None:
            await widget.clear()
            return
        try:
            stat = await diff_stat(info.worktree, info.base_ref, info.subtree)
        except GitWorktreeError:
            await widget.clear()  # worktree vanished / git broken: stay quiet
            return
        if not stat.files:
            await widget.clear()
        else:
            await widget.set_summary(stat)

    async def on_diff_summary_diff_requested(self, event: DiffSummary.DiffRequested) -> None:
        info = self.git_info
        if info is None:
            return
        widget = self.query_one("#diff-summary", DiffSummary)
        if widget.active_diff_key == event.key:
            widget.hide_diff()  # same affordance again: collapse the diff
            return
        try:
            if event.key is None:
                text = await git_diff(info.worktree, info.base_ref, subtree=info.subtree)
            else:
                text = await git_diff(
                    info.worktree, info.base_ref, event.key, subtree=info.subtree
                )
        except GitWorktreeError as exc:
            widget.hide_diff()
            self._post_line(ErrorLine(f"diff failed: {exc}"))
            return
        widget.show_diff(event.key, text)

    @work(group="diff-view", exclusive=True)
    async def _show_diff_command(self, arg: str) -> None:
        """Slash /diff: toggle the pending-changes panel — the first /diff
        expands the per-file list, /diff again hides it entirely. With an
        argument, /diff <path> shows that one file's unified diff."""
        info = self.git_info
        if info is None:
            self._post_line(ErrorLine(
                "/diff: no pending-changes tracking for this session — the workspace "
                "could not be isolated with git (auto-init disabled via --no-init-repo, "
                "or repo setup failed), so the agent edits it directly"
            ))
            return
        widget = self.query_one("#diff-summary", DiffSummary)

        if not arg:
            # Toggle: fully shown file list (no diff open) → hide the panel.
            if widget.has_class("visible") and widget.has_class("expanded") and not widget.diff_shown:
                await widget.clear()
                return
            await self.refresh_diff_summary()
            if not widget.has_class("visible"):
                self._post_line(NoticeLine("no pending changes to diff"))
                return
            widget.add_class("expanded")
            return

        try:
            text = await git_diff(info.worktree, info.base_ref, arg, subtree=info.subtree)
        except GitWorktreeError as exc:
            widget.hide_diff()
            self._post_line(ErrorLine(f"/diff failed: {exc}"))
            return
        if not text.strip():
            self._post_line(NoticeLine(f"no pending changes for {arg}"))
            return
        # Rebuild the row list first (a previous /diff toggle may have cleared
        # it) so the active-row highlight has something to attach to.
        await self.refresh_diff_summary()
        widget.show_diff(arg, text)

    async def _rebind_worktree(self, info: GitInfo, chat: VerticalScroll) -> None:
        """Point the sandbox + tools at a (possibly resumed) worktree.

        ``chat`` is kept for signature symmetry with the other worktree
        helpers; failures are posted through _post_line."""
        if self.sandbox_session is not None:
            await self.sandbox_session.close()
            self.sandbox_session = None
        sandbox: SandboxConfig | None = None
        try:
            sandbox = SandboxConfig(
                workspace=info.worktree, subtree=info.subtree, git_dir=info.git_common_dir
            )
        except SandboxUnavailable as exc:
            self._post_line(ErrorLine(f"filesystem tools unavailable: {exc}"))
        self.sandbox = sandbox
        self.sandbox_session = SandboxSession(sandbox) if sandbox is not None else None
        self.git_info = info
        self.controller.git_info = info
        self.workspace = info.agent_workspace
        self.tool_registry = build_registry(
            sandbox, self.sandbox_session, mode=self.mode,
            ask_callback=self._ask_user, compact_callback=self._compact_conversation,
            git_guard=self._make_git_guard(), git_info=info,
        )
        self.controller.tools = self.tool_registry

    def _disable_fs_tools(self) -> None:
        """After accept/reject the worktree is gone; nothing sane to edit."""
        self.git_info = None
        self.controller.git_info = None
        self.sandbox = None
        self.sandbox_session = None
        self.tool_registry = build_registry(
            None, None, mode=self.mode,
            ask_callback=self._ask_user, compact_callback=self._compact_conversation,
        )
        self.controller.tools = self.tool_registry

    def _persist_git_state(self) -> None:
        if self.session_name:
            from ..session_store import save_session
            save_session(
                self.session_name, self.profile.model_id, self.controller.conversation,
                git=self.git_info.to_block() if self.git_info else None,
            )

    def _git_action_preflight(self, label: str) -> tuple[GitInfo, VerticalScroll] | None:
        """Shared /accept //reject guards; mounts an error line if blocked."""
        chat = self.query_one("#chat-log", VerticalScroll)
        info = self.git_info
        if info is None:
            self._post_line(ErrorLine(
                f"/{label}: this session has no git worktree (the workspace isn't "
                "a git repo, or the changes were already resolved)"
            ))
            return None
        if self._turn_busy:
            self._post_line(ErrorLine(f"/{label}: wait for the current turn to finish first"))
            return None
        return info, chat

    async def _finish_git_action(self, message: str) -> None:
        """Post-resolution cleanup shared by accept and reject."""
        self._disable_fs_tools()
        self._persist_git_state()
        await self.refresh_diff_summary()
        self._post_line(NoticeLine(message))
        self._post_line(NoticeLine(
            "filesystem tools disabled — /new starts a fresh session with a new worktree"
        ))

    @work(group="git-action", exclusive=True)
    async def _start_accept(self) -> None:
        """Merge agent/<id> into the user's branch in their real checkout."""
        preflight = self._git_action_preflight("accept")
        if preflight is None:
            return
        info, _chat = preflight
        self._post_line(NoticeLine(f"accepting: committing worktree changes and merging {info.branch}…"))
        try:
            await commit_worktree_changes(info)
            result = await merge_branch(info.original_workspace, info.branch)
        except GitWorktreeError as exc:
            self._post_line(ErrorLine(f"/accept failed: {exc}"))
            return
        if result.outcome is MergeOutcome.IN_PROGRESS:
            self._post_line(ErrorLine(
                f"/accept: a merge is already unresolved in {info.original_workspace}. "
                "Finish it there with normal git tooling, commit, then run /accept again."
            ))
            return
        if result.outcome is MergeOutcome.CONFLICT:
            self._post_line(ErrorLine(
                f"/accept: merge conflicts in {info.original_workspace} — your branch moved on "
                "since the session started. Resolve them there with normal git tooling and "
                "commit the merge; the worktree was left in place so you can retry /accept."
            ))
            return
        if result.outcome is MergeOutcome.ERROR:
            self._post_line(ErrorLine(f"/accept failed: {result.detail}"))
            return
        await remove_worktree(info.original_workspace, info.worktree)
        await self._finish_git_action(
            f"accepted: changes merged into your working directory; worktree removed. "
            f"Branch {info.branch} kept as history (delete with: git branch -D {info.branch})."
        )

    @work(group="git-action", exclusive=True)
    async def _start_reject(self) -> None:
        """Discard everything: force-remove the worktree, delete the branch."""
        preflight = self._git_action_preflight("reject")
        if preflight is None:
            return
        info, _chat = preflight
        confirmed = await self.push_screen(ConfirmScreen(
            title="Reject agent changes?",
            detail=(
                f"Permanently discards the session worktree ({info.worktree}) and deletes "
                f"branch {info.branch}. This cannot be undone. Your own working directory "
                "is never touched by this session, so it is unaffected."
            ),
            confirm_word="reject",
        ), wait_for_dismiss=True)
        if not confirmed:
            return
        try:
            await remove_worktree(info.original_workspace, info.worktree, force=True)
            await delete_branch(info.original_workspace, info.branch)
        except GitWorktreeError as exc:
            self._post_line(ErrorLine(f"/reject failed: {exc}"))
            return
        await self._finish_git_action(
            "rejected: worktree and branch removed; your working directory was never modified"
        )

    def _on_theme_selected(self, name: str | None) -> None:
        if name:
            self._apply_theme(name)

    def _handle_theme_command(self, arg: str) -> None:
        """Apply an explicitly named theme, or report the available ones."""
        if arg not in theme.THEMES:
            self._post_line(ErrorLine(f"unknown theme '{arg}'; available: {', '.join(theme.THEMES)}"))
            return
        self._apply_theme(arg)

    def _apply_theme(self, name: str) -> None:
        """Switch palette live: CSS variables, input caret, shimmers, dots,
        summaries, status bar — anything that baked palette colors at render
        time must be re-rendered here, or the switched theme drifts from the
        same theme set at startup."""
        theme.set_theme(name)
        self.refresh_css()
        self.query_one("#chat-input", ChatInput).apply_input_theme()
        for shimmer in self.query(ShimmerText):
            colors = (
                theme.SHIMMER_PROCESSING if shimmer.id == "pending-shimmer"
                else theme.SHIMMER_THINKING
            )
            shimmer.set_colors(*colors)
        for message in self.query(UserMessage):
            message.apply_palette()
        for block in self.query(ToolCallBlock):
            block.recolor()
        for thinking in self.query(ThinkingBlock):
            thinking.recolor()
        for block in self.query(ToolCallBlock):
            block.recolor()
        self._refresh_status()
        self.query_one("#diff-summary", DiffSummary).recolor()

    def _start_new_session(self) -> None:
        """Slash /new: fresh conversation, fresh session file (unless disabled)."""
        self.controller.conversation = Conversation()
        self.controller.usage_total = Usage(input_tokens=0, output_tokens=0)
        self.controller.last_compaction = None
        if self.session_name is not None:
            from ..session_store import new_session_name
            self.session_name = new_session_name()
        self.ensure_system_message()
        self.total_in = self.total_out = self.last_in = 0
        self._last_thinking = None
        self._queued.clear()
        chat = self.query_one("#chat-log", VerticalScroll)
        chat.remove_children()
        # Fresh, empty log: follow it again from the top of the new session.
        chat.anchor()
        self._refresh_status()

    async def _on_session_selected(self, name: str | None) -> None:
        if not name:
            return
        from ..session_store import load_git_block, load_session
        chat = self.query_one("#chat-log", VerticalScroll)
        self.controller.conversation = load_session(name)
        self.session_name = name
        self.ensure_system_message()
        # Per-message usage is persisted: restore the status-bar totals.
        self._sync_totals_from_controller()
        # Notices mount after _render_history (which clears the chat log).
        notes: list[str] = []
        errors: list[str] = []
        block = load_git_block(name)
        if block is not None:
            info = GitInfo.from_block(block)
            if await worktree_is_valid(info):
                await self._rebind_worktree(info, chat)
                notes.append(f"reconnected to agent worktree {info.worktree} (branch {info.branch})")
            else:
                # Never fall back to editing the user's real directory silently —
                # recreate the worktree, or say loudly what happened.
                try:
                    fresh = await recreate_worktree(info)
                except GitWorktreeError as exc:
                    errors.append(
                        f"this session's agent worktree is gone and could not be recreated ({exc}); "
                        "filesystem tools stay on the previous workspace — use /new for a clean start"
                    )
                else:
                    await self._rebind_worktree(fresh, chat)
                    notes.append(
                        f"previous worktree for this session was removed externally; recreated a fresh "
                        f"one from branch {fresh.branch} — uncommitted agent changes from before are gone"
                    )
        elif self.sandbox is not None:
            # Session predates git tracking. Set up isolation now if possible
            # (same policy as cli.py — including auto-init on a bare directory).
            from .. import gitwork
            try:
                info, notes = await gitwork.setup_isolation(
                    self.workspace or Path.cwd(), name
                )
            except GitWorktreeError as exc:
                errors.append(f"git worktree isolation unavailable: {exc}")
            else:
                if info is not None:
                    await self._rebind_worktree(info, chat)
                    notes.append(
                        f"agent edits go to worktree {info.worktree} (branch {info.branch}); "
                        "/diff shows pending changes, /accept merges, /reject discards"
                    )
        await self._render_history()
        for note in notes:
            self._post_line(NoticeLine(note))
        for error in errors:
            self._post_line(ErrorLine(error))
        await self.refresh_diff_summary()

    def _on_model_selected(self, name: str | None) -> None:
        if not name:
            return
        profile = self._profiles_cache.get(name)
        if profile is None:
            return
        try:
            api_key = resolve_api_key(profile)
        except ConfigError as exc:
            self._post_line(ErrorLine(f"could not switch model: {exc}"))
            return
        # Swap the provider underneath the existing controller: client,
        # conversation, and tools all carry over.
        self.controller.switch_profile(profile, api_key)
        self.profile = profile
        self.profile_name = name
        self._refresh_status()

    async def _render_history(self) -> None:
        """Rebuild #chat-log from self.controller.conversation.messages.

        Used by /resume. Best-effort reconstruction: tool-call status is
        inferred from whether the recorded tool-role content starts with
        "error: " (see ChatController.record_tool_result), since the
        original ToolCallStatus enum value isn't itself persisted.
        """
        chat = self.query_one("#chat-log", VerticalScroll)
        await chat.remove_children()
        messages = self.controller.conversation.messages
        tool_results = {m.tool_call_id: m for m in messages if m.role == "tool" and m.tool_call_id}

        for message in messages:
            if message.role == "user":
                await chat.mount(UserMessage(message.content))
            elif message.role == "assistant":
                if message.reasoning:
                    thinking = ThinkingBlock()
                    await chat.mount(thinking)
                    thinking.append_reasoning(message.reasoning)
                    thinking.finish(message.reasoning_seconds, estimate_if_unknown=False)
                # Tool-call-only rounds render no AssistantMessage body; an
                # empty assistant message with no tool calls replays as
                # "(no output)", matching the live-stream finalize path.
                if message.content or not message.tool_calls:
                    assistant = AssistantMessage()
                    await chat.mount(assistant)
                    if message.content:
                        await assistant.append_delta(message.content)
                    await assistant.finalize()
                for call in message.tool_calls or []:
                    call_id = call.get("id", "")
                    fn = call.get("function", {})
                    block = ToolCallBlock(call_id, fn.get("name", ""))
                    await chat.mount(block)
                    block.append_arguments(fn.get("arguments", ""))
                    result = tool_results.get(call_id)
                    if result is not None:
                        is_error = result.content.startswith("error: ")
                        status = ToolCallStatus.CALL_FAILED if is_error else ToolCallStatus.CALL_SUCCEEDED
                        block.set_result(
                            status,
                            output="" if is_error else result.content,
                            error=result.content[len("error: "):] if is_error else "",
                        )
        chat.scroll_end(animate=False)

    def on_chat_input_slash_query(self, event: ChatInput.SlashQuery) -> None:
        q = event.query.lower()
        # Table-style rows: commands padded to a shared column, descriptions
        # dimmed — same layout language as the session list.
        width = max(len(name) for name, _ in SLASH_COMMANDS)
        matches = [
            (name, f"/{name:<{width}}  [dim]{desc}[/]")
            for name, desc in SLASH_COMMANDS
            if name.startswith(q)
        ]
        # Exact match first (stable sort keeps declaration order otherwise), so
        # typing "/mode" highlights /mode rather than /model.
        matches.sort(key=lambda m: m[0] != q)
        self._show_popup(matches)

    def on_chat_input_at_query(self, event: ChatInput.AtQuery) -> None:
        self._cancel_at_search()
        self._at_search_timer = self.set_timer(0.15, lambda: self._run_at_search(event.query))

    def _cancel_at_search(self) -> None:
        if self._at_search_timer is not None:
            self._at_search_timer.stop()
            self._at_search_timer = None

    @work(exclusive=True, group="at-search")
    async def _run_at_search(self, query: str) -> None:
        chat_input = self.query_one("#chat-input", ChatInput)
        if chat_input.popup_active != "at":
            return  # dismissed (e.g. escape) while the search was pending
        root = self.workspace or Path.cwd()
        results = await search_files(root, query)
        self._show_popup([(r, r) for r in results])

    def on_chat_input_popup_dismiss(self, event: ChatInput.PopupDismiss) -> None:
        self._cancel_at_search()
        self._hide_popup()

    def on_chat_input_popup_nav(self, event: ChatInput.PopupNav) -> None:
        self.query_one("#suggestion-popup", SuggestionPopup).move_highlight(event.direction)

    def on_chat_input_popup_confirm(self, event: ChatInput.PopupConfirm) -> None:
        popup = self.query_one("#suggestion-popup", SuggestionPopup)
        value = popup.selected_value
        chat_input = self.query_one("#chat-input", ChatInput)
        was_slash = chat_input.popup_active == "slash"
        chat_input.popup_active = None
        self._hide_popup()
        chat_input.focus()
        if value is None:
            return
        if was_slash:
            # Slash commands take no arguments: complete and run in one Enter.
            chat_input.load_text("")
            self._handle_slash_command(f"/{value}")
        else:
            chat_input.insert_mention(value)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Mouse click on a popup option: insert it immediately, keep focus on input."""
        event.stop()
        chat_input = self.query_one("#chat-input", ChatInput)
        was_slash = chat_input.popup_active == "slash"
        value = event.option.id
        self._hide_popup()
        chat_input.popup_active = None
        chat_input.focus()
        if value is None:
            return
        if was_slash:
            # Consistent with Enter: clicking a slash command runs it.
            chat_input.load_text("")
            self._handle_slash_command(f"/{value}")
        else:
            chat_input.insert_mention(value)

    def _show_popup(self, items: list[tuple[str, str]]) -> None:
        popup = self.query_one("#suggestion-popup", SuggestionPopup)
        popup.set_items(items)
        if items:
            popup.add_class("visible")
        else:
            popup.remove_class("visible")

    def _hide_popup(self) -> None:
        self.query_one("#suggestion-popup", SuggestionPopup).remove_class("visible")

    def _post_line(self, widget: Static) -> None:
        """Mount a notice/error line into the chat log.

        Whether it lands in view is the chat log's business: it is anchored,
        so it follows the new line only if the user is already at the bottom.
        """
        try:
            chat = self.query_one("#chat-log", VerticalScroll)
        except NoMatches:
            return  # app shutting down; DOM already pruned
        chat.mount(widget)

    def _submit(self, text: str) -> None:
        chat = self.query_one("#chat-log", VerticalScroll)
        user_message = UserMessage(text)
        if self._turn_busy:
            user_message.mark_queued()
            self._queued.append(text)
        chat.mount(user_message)
        if not self._turn_busy:
            self._worker = self._run_turn(text)

    def action_copy_or_quit(self) -> None:
        """ctrl+c: copy the active selection if there is one, quit otherwise."""
        focused = self.focused
        if isinstance(focused, TextArea) and focused.selected_text:
            self.copy_to_clipboard(focused.selected_text)
            return
        selected = self.screen.get_selected_text()
        if selected:
            self.copy_to_clipboard(selected)
            return
        self.exit()

    def action_interrupt(self) -> None:
        # The priority escape binding shadows the modals' own cancel bindings,
        # so cancel the open modal here rather than interrupting the turn.
        if isinstance(self.screen, ModalScreen):
            self.screen.dismiss(None)
            return
        chat_input = self.query_one("#chat-input", ChatInput)
        popup = self.query_one("#suggestion-popup", SuggestionPopup)
        if chat_input.popup_active is not None or popup.has_class("visible"):
            chat_input.popup_active = None
            self._cancel_at_search()
            self._hide_popup()
            return
        if self._ask_future is not None and not self._ask_future.done():
            # Esc while a question is pending: skip the questions, keep the
            # turn running (the ask tool reports the skip to the model).
            self._ask_future.set_result(None)
            return
        diff_summary = self.query_one("#diff-summary", DiffSummary)
        if diff_summary.diff_shown:
            # First esc: minimize the open diff, keeping the file list up and
            # its last-viewed row highlighted for arrow-key navigation.
            diff_summary.minimize_diff()
            return
        if self.focused is diff_summary and diff_summary.has_class("expanded"):
            # Second esc: close the file list too; only the summary remains.
            diff_summary.collapse()
            return
        if self._turn_busy and self._worker is not None:
            self._worker.cancel()

    async def _ask_user(self, questions: list[str]) -> list[str] | None:
        """Callback for the ask tool: show the questions one at a time in the
        AskBar above the input; the user answers through the normal chat
        input (Enter submits, Esc skips all questions).

        Runs inside the turn worker, awaiting a future that the input
        handler resolves — same event loop, so the await is safe.
        """
        ask_bar = self.query_one("#ask-bar", AskBar)
        chat_input = self.query_one("#chat-input", ChatInput)
        answers: list[str] = []
        try:
            for index, question in enumerate(questions):
                ask_bar.show_question(f"Question {index + 1}/{len(questions)}: {question}")
                chat_input.placeholder = ASK_PLACEHOLDER
                chat_input.focus()
                future: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
                self._ask_future = future
                answer = await future
                if answer is None:  # esc: skip the remaining questions
                    return None
                answers.append(answer)
            return answers
        finally:
            self._ask_future = None
            ask_bar.hide()
            chat_input.placeholder = INPUT_PLACEHOLDER

    async def _compact_conversation(self) -> str:
        """Callback for the compact tool: summarize + truncate the history."""
        summary = await self.controller.compact()
        # The summarization round's own spend just entered the controller's
        # cumulative total; mirror it now rather than waiting for the next
        # round's TurnComplete.
        self._sync_totals_from_controller()
        self._refresh_status()
        counts = self.controller.last_compaction
        if counts is not None:
            before, after = counts
            freed = max(0, before - after)
            return f"compacted: {before:,} → {after:,} tokens (freed {freed:,})\n{summary}"
        return summary

    @work(group="turn")
    async def _run_turn(self, text: str) -> None:
        """Drive the full exchange: rounds of (thinking → answer → tool calls).

        Each round streams one model response. If the round requested tool
        calls, they're executed, their blocks settle, results are recorded
        into the conversation, and the next round streams — until a round
        comes back with no tool calls.
        """
        self._turn_busy = True
        chat = self.query_one("#chat-log", VerticalScroll)
        turn_start = time.monotonic()
        turn_had_tools = False
        tool_blocks: dict[str, ToolCallBlock] = {}
        indicator: PendingIndicator | None = None
        indicator_live = False
        writing: ToolWritingIndicator | None = None
        pending_writes = 0
        thinking: ThinkingBlock | None = None
        had_stream_error = False

        stream = self.controller.send(text)
        try:
            while True:
                indicator = PendingIndicator()
                indicator_live = True
                await chat.mount(indicator)

                thinking = None
                assistant = None
                tool_blocks = {}
                writing = None
                pending_writes = 0
                round_has_tools = False

                async for event in stream:
                    if isinstance(event, ReasoningDelta):
                        if thinking is None:
                            await self._dismiss_indicator(indicator)
                            indicator_live = False
                            thinking = ThinkingBlock()
                            self._last_thinking = thinking
                            await chat.mount(thinking)
                        thinking.append_reasoning(event.text)
                    elif isinstance(event, ContentDelta):
                        if assistant is None:
                            if indicator_live:
                                await self._dismiss_indicator(indicator)
                                indicator_live = False
                            if thinking is not None:
                                thinking.finish(duration=None)
                            assistant = AssistantMessage()
                            await chat.mount(assistant)
                        await assistant.append_delta(event.text)
                    elif isinstance(event, ToolCallStarted):
                        if indicator_live:
                            await self._dismiss_indicator(indicator)
                            indicator_live = False
                        if thinking is not None:
                            thinking.finish(duration=None)
                        # One shared "Writing tools" shimmer while arguments
                        # stream; per-call blocks appear once args are complete.
                        pending_writes += 1
                        if writing is None:
                            writing = ToolWritingIndicator()
                            await chat.mount(writing)
                    elif isinstance(event, ToolCallArgumentsDone):
                        block = ToolCallBlock(event.call_id, event.name)
                        block.append_arguments(event.arguments_json)
                        tool_blocks[event.call_id] = block
                        await chat.mount(block)
                        pending_writes = max(0, pending_writes - 1)
                        if pending_writes == 0 and writing is not None:
                            await self._dismiss_indicator(writing)
                            writing = None
                    elif isinstance(event, TurnComplete):
                        round_has_tools = event.has_tool_calls
                        await self._complete_round(event, thinking, assistant)
                    elif isinstance(event, StreamError):
                        if indicator_live:
                            await self._dismiss_indicator(indicator)
                            indicator_live = False
                        await chat.mount(ErrorLine(event.message))
                        had_stream_error = True

                if writing is not None:
                    await self._dismiss_indicator(writing)
                    writing = None
                if indicator_live:
                    await self._dismiss_indicator(indicator)

                if not round_has_tools:
                    if not had_stream_error:
                        worked = _format_duration(time.monotonic() - turn_start)
                        self._post_line(NoticeLine(f"Worked for {worked}"))
                    break

                # Execute the calls this round requested, in stream order.
                turn_had_tools = True
                for call_id, block in tool_blocks.items():
                    result = await self.tool_registry.call(block.tool_name, block.accumulated_arguments)
                    if result.parse_error:
                        status = ToolCallStatus.PARSING_ERROR
                    elif result.ok:
                        status = ToolCallStatus.CALL_SUCCEEDED
                    else:
                        status = ToolCallStatus.CALL_FAILED
                    block.set_result(status, output=result.output, error=result.error)
                    self.controller.record_tool_result(call_id, result)

                stream = self.controller.continue_after_tools()
        except asyncio.CancelledError:
            if writing is not None and writing.is_mounted:
                await writing.remove()
            if indicator is not None and indicator_live:
                await self._dismiss_indicator(indicator)
            if thinking is not None and not thinking.done:
                thinking.finish(duration=None)
            for block in tool_blocks.values():
                if block.status is ToolCallStatus.MAKING_CALL:
                    block.set_result(ToolCallStatus.CALL_FAILED, error="interrupted")
            self._post_line(ErrorLine("interrupted"))
        finally:
            self._turn_busy = False
            self._worker = None
            # Cheapest correct trigger for the diff summary: after any turn
            # that ran tools, recompute once — not per keystroke or per round.
            if turn_had_tools and self.git_info is not None:
                await self.refresh_diff_summary()
            if self._queued:
                self._worker = self._run_turn(self._queued.pop(0))

    async def _dismiss_indicator(self, indicator: PendingIndicator | None) -> None:
        if indicator is not None and indicator.is_mounted:
            await indicator.remove()

    async def _complete_round(
        self,
        event: TurnComplete,
        thinking: ThinkingBlock | None,
        assistant: AssistantMessage | None,
    ) -> None:
        chat = self.query_one("#chat-log", VerticalScroll)
        if assistant is not None:
            await assistant.finalize()
        elif thinking is None and not event.has_tool_calls:
            await chat.mount(ErrorLine("empty response from model"))
        if thinking is not None:
            thinking.finish(event.reasoning_seconds)

        if event.usage is not None:
            self.last_in = event.usage.input_tokens
        # The controller folds every round's usage (including compaction's
        # own summarization round) into its cumulative total; mirror it.
        self.total_in = self.controller.usage_total.input_tokens
        self.total_out = self.controller.usage_total.output_tokens
        self._refresh_status()

        if self.session_name:
            from ..session_store import save_session
            save_session(
                self.session_name, self.profile.model_id, self.controller.conversation,
                git=self.git_info.to_block() if self.git_info else None,
            )

    def action_toggle_thoughts(self) -> None:
        if self._last_thinking is not None:
            self._last_thinking.toggle()
