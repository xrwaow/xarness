"""Modal for switching the active provider profile mid-conversation."""

from __future__ import annotations

from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView


class ModelPickerScreen(ModalScreen[str | None]):
    """List of profile names from the config file; dismisses with the chosen name or None."""

    BINDINGS = [("escape", "dismiss_none", "Cancel")]

    def __init__(self, names: list[str], current: str | None = None) -> None:
        super().__init__()
        self._names = names
        self._current = current

    def compose(self):
        with Vertical(id="model-modal"):
            items = [
                ListItem(Label(f"{name}  (current)" if name == self._current else name))
                for name in self._names
            ]
            yield ListView(*items, id="model-list")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is not None and idx < len(self._names):
            self.dismiss(self._names[idx])

    def action_dismiss_none(self) -> None:
        self.dismiss(None)
