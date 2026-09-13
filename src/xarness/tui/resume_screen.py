"""Modal for selecting a previous chat session to resume."""

from __future__ import annotations

from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView

from .. import session_store


class ResumeScreen(ModalScreen[str | None]):
    """Filterable list of saved sessions; dismisses with the chosen name or None."""

    BINDINGS = [("escape", "dismiss_none", "Cancel")]

    def __init__(self) -> None:
        super().__init__()
        names = session_store.list_sessions()
        self._entries = [(name, session_store.session_meta(name)) for name in names]
        self._entries.sort(key=lambda e: e[1].get("updated_at") or "", reverse=True)
        self._filtered = self._entries

    def compose(self):
        with Vertical(id="resume-modal"):
            yield Input(placeholder="Type to search", id="resume-search")
            yield ListView(*self._build_items(self._entries), id="resume-list")

    def _build_items(self, entries: list[tuple[str, dict]]) -> list[ListItem]:
        items = []
        for name, meta in entries:
            updated = meta.get("updated_at", "?")
            count = meta.get("message_count", 0)
            items.append(ListItem(Label(f"{updated}   {name}   ({count} msgs)")))
        return items

    def on_mount(self) -> None:
        self.query_one("#resume-search", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        query = event.value.lower()
        self._filtered = [(n, m) for n, m in self._entries if query in n.lower()]
        listview = self.query_one("#resume-list", ListView)
        listview.clear()
        for item in self._build_items(self._filtered):
            listview.append(item)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is not None and idx < len(self._filtered):
            self.dismiss(self._filtered[idx][0])

    def action_dismiss_none(self) -> None:
        self.dismiss(None)
