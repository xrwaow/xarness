"""Generic modal list picker: choose one name from a list, or cancel.

One class for every "choose an X" flow (/model, /theme, ...): the caller
supplies the entries, which entry is current, and an optional title.
"""

from __future__ import annotations

from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView


class PickerScreen(ModalScreen[str | None]):
    """List of names, the current one marked; dismisses with the choice or None."""

    BINDINGS = [("escape", "dismiss_none", "Cancel")]

    def __init__(
        self,
        names: list[str],
        current: str | None = None,
        title: str | None = None,
    ) -> None:
        super().__init__()
        self._names = names
        self._current = current
        self._title = title

    def compose(self):
        with Vertical(id="picker-modal"):
            if self._title:
                yield Label(self._title, id="picker-title")
            items = [
                ListItem(Label(f"{name}  (current)" if name == self._current else name))
                for name in self._names
            ]
            yield ListView(*items, id="picker-list")

    def on_mount(self) -> None:
        list_view = self.query_one("#picker-list", ListView)
        list_view.focus()
        if list_view.index is None:
            list_view.index = 0

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is not None and idx < len(self._names):
            self.dismiss(self._names[idx])

    def action_dismiss_none(self) -> None:
        self.dismiss(None)