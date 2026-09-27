"""Generic yes/no modal for decisions the harness must not make for the user.

Used by the resume flow to surface workspace drift: the harness shows what
changed and lets the user choose, rather than silently reverting or accepting.

Follows the same shape as the other modals (``PickerScreen``/``ResumeScreen``):
a titled surface panel centered on screen, dismissed with the app's modal
cancel binding. The question is the title, an optional plain-text ``detail``
block sits beneath it, and the choice is two buttons — cancel then the
affirmative, so the primary action reads rightmost.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static


class ConfirmScreen(ModalScreen[bool | None]):
    """Ask a yes/no question, dismissing with the user's choice.

    ``True`` is the affirmative; ``False`` and ``None`` (escape, via the app's
    modal-cancel binding) both mean cancel. ``y``/the confirm button choose
    true, ``n``/the cancel button choose false.
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("n", "cancel", "No"),
        ("y", "confirm", "Yes"),
    ]

    def __init__(
        self,
        question: str,
        confirm_label: str = "Confirm",
        cancel_label: str = "Cancel",
        detail: str = "",
    ) -> None:
        super().__init__()
        self._question = question
        self._confirm_label = confirm_label
        self._cancel_label = cancel_label
        self._detail = detail

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-modal"):
            yield Label(self._question, id="confirm-title", markup=False)
            if self._detail:
                yield Static(self._detail, id="confirm-detail", markup=False)
            with Horizontal(id="confirm-buttons"):
                yield Button(self._cancel_label, id="confirm-no")
                yield Button(self._confirm_label, variant="primary", id="confirm-yes")

    def on_mount(self) -> None:
        self.query_one("#confirm-yes", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(event.button.id == "confirm-yes")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)
