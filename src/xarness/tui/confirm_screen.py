"""Typed-confirmation modal for destructive actions (matches AskScreen's
flat-surface modal style; no buttons exist elsewhere in this TUI, so the
confirmation affordance is the same Input the rest of the modals use)."""

from __future__ import annotations

from textual import events
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label


class ConfirmScreen(ModalScreen[bool]):
    """Asks the user to type a word to confirm; dismisses True on exact match
    (case-insensitive), False on escape or anything else."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, title: str, detail: str, confirm_word: str) -> None:
        super().__init__()
        self._title = title
        self._detail = detail
        self._confirm_word = confirm_word

    def compose(self):
        with Vertical(id="confirm-modal"):
            yield Label(self._title, id="confirm-title")
            yield Label(self._detail, id="confirm-detail")
            yield Input(
                placeholder=f"type '{self._confirm_word}' to confirm", id="confirm-input"
            )
            yield Label("enter: confirm · esc: cancel", classes="ask-hint")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            self.dismiss(False)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.input.value.strip().lower() == self._confirm_word.lower())

    def action_cancel(self) -> None:
        self.dismiss(False)
