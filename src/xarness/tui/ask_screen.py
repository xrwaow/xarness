"""Modal that lets the model ask the user questions mid-turn.

One input per question; Enter advances, Enter on the last one submits.
Dismisses with the list of answers, or None if the user pressed escape.
Styled to match the rest of the TUI: flat surface, no borders.
"""

from __future__ import annotations

from textual import events
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label


class AskScreen(ModalScreen[list[str] | None]):
    """Asks N questions; dismisses with the answers, or None on escape."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, questions: list[str]) -> None:
        super().__init__()
        self._questions = questions

    def compose(self):
        with Vertical(id="ask-modal"):
            for index, question in enumerate(self._questions):
                yield Label(question, classes="ask-question")
                yield Input(id=f"ask-input-{index}")
            yield Label("enter: next · enter on last: submit · esc: skip", classes="ask-hint")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_key(self, event: events.Key) -> None:
        """Up moves back to the previous input (down/enter are Input's own)."""
        if event.key != "up":
            return
        event.stop()
        inputs = self.query(Input).nodes
        try:
            index = inputs.index(self.focused)
        except ValueError:
            return
        if index > 0:
            inputs[index - 1].focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        inputs = list(self.query(Input))
        index = inputs.index(event.input)
        if index + 1 < len(inputs):
            inputs[index + 1].focus()
        else:
            self.dismiss([input.value.strip() for input in inputs])

    def action_cancel(self) -> None:
        self.dismiss(None)