"""Modal for managing a workspace's external references.

Typing a path into ``add path:`` hands it to ``add_ref``; each row's ``x``
hands the path to ``remove_ref``. Rows come from the session status, so the
dimmed line is the alias the sandbox resolved (``.refs/<alias>``) — nothing
is ever created at that path in the worktree.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from ..protocol import RefInfo


class RefsScreen(ModalScreen[None]):
    """Add/remove reference paths for the current workspace."""

    BINDINGS = [("escape", "dismiss_none", "Done")]

    def __init__(self, refs: list[RefInfo], add_ref, remove_ref) -> None:
        super().__init__()
        self._refs = list(refs)
        self._add_ref = add_ref
        self._remove_ref = remove_ref

    def compose(self) -> ComposeResult:
        with Vertical(id="refs-modal"):
            yield Label("references", id="refs-title")
            yield Label("add path:", id="refs-add-label")
            yield Input(placeholder="file or directory", id="refs-input")
            yield Vertical(id="refs-list")

    async def on_mount(self) -> None:
        await self._render_refs()
        self.query_one("#refs-input", Input).focus()

    async def _render_refs(self) -> None:
        container = self.query_one("#refs-list", Vertical)
        await container.remove_children()
        if not self._refs:
            container.mount(Static("no references yet", id="refs-empty"))
            return
        container.mount_all(
            Vertical(
                Horizontal(
                    Static(ref.path, classes="refs-path", markup=False),
                    Button("x", id=f"refs-remove-{i}", classes="refs-remove"),
                    classes="refs-row-top",
                ),
                Static(ref.mount, classes="refs-mount", markup=False),
                classes="refs-row",
            )
            for i, ref in enumerate(self._refs)
        )

    def set_refs(self, refs: list[RefInfo]) -> None:
        self._refs = list(refs)
        if self.is_mounted:
            self.run_worker(self._render_refs(), group="refs-render", exclusive=True)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if path := event.value.strip():
            event.input.value = ""
            self._add_ref(path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if not button_id.startswith("refs-remove-"):
            return
        event.stop()
        try:
            index = int(button_id.rsplit("-", 1)[1])
        except ValueError:
            return
        if 0 <= index < len(self._refs):
            self._remove_ref(self._refs[index].path)
            self.query_one("#refs-input", Input).focus()

    def action_dismiss_none(self) -> None:
        self.dismiss(None)
