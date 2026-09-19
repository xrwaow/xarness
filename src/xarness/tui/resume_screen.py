"""Modal for selecting a previous chat session to resume."""

from __future__ import annotations

from datetime import datetime, timezone

from rich.markup import escape
from textual import events
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView

from .. import session_store
from .widgets import crop_path


def _relative_time(iso: str | None) -> str:
    """Format an ISO timestamp as age from now: 5m / 3h / 12d."""
    if not iso:
        return "?"
    try:
        updated = datetime.fromisoformat(iso)
    except ValueError:
        return "?"
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    minutes = int((datetime.now(timezone.utc) - updated).total_seconds()) // 60
    if minutes < 1:
        return "<1m"
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    return f"{hours // 24}d"


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
            yield Label("Resume session", id="resume-title")
            yield Input(placeholder="Type to search", id="resume-search")
            yield ListView(*self._build_items(self._entries), id="resume-list")

    def _build_items(self, entries: list[tuple[str, dict]]) -> list[ListItem]:
        items = []
        for name, meta in entries:
            age = _relative_time(meta.get("updated_at"))
            workspace = meta.get("workspace")
            label = f"{escape(name)}  [dim]{age}"
            if workspace:
                label += f" · {escape(crop_path(workspace))}"
            label += "[/]"
            item = ListItem(Label(label))
            item.session_name = name
            items.append(item)
        return items

    def on_mount(self) -> None:
        self.query_one("#resume-search", Input).focus()

    def on_key(self, event: events.Key) -> None:
        """Up/down in the search box moves the list highlight."""
        if event.key not in ("up", "down"):
            return
        event.prevent_default()
        event.stop()
        listview = self.query_one("#resume-list", ListView)
        if event.key == "up":
            listview.action_cursor_up()
        else:
            listview.action_cursor_down()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter in the search box picks the highlighted session."""
        listview = self.query_one("#resume-list", ListView)
        if listview.index is not None:
            listview.action_select_cursor()

    def on_input_changed(self, event: Input.Changed) -> None:
        query = event.value.lower()
        self._filtered = [
            (n, m) for n, m in self._entries
            if query in n.lower() or query in (m.get("workspace") or "").lower()
        ]
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
