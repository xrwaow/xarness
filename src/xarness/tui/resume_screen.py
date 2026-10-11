"""Modal for selecting a chat session to open (attach or load).

Data comes from the host API (``GET /v1/sessions``): a global section first,
listing all **active** sessions across all workspaces, then saved sessions
for the current workspace. Each row shows the session title, the age of its
last user message, and a state dot on the right: waiting=accent,
running/compacting (active)=success green, idle=the plain-message gray,
saved/not loaded=the popup's background color. Both sections sort by
``last_user_message_at``, newest first.
"""

from __future__ import annotations

from datetime import datetime, timezone

from rich.markup import escape
from textual import events
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView
from pathlib import Path

from .. import theme
from ..protocol import SessionSummary

# Row-dot color per session state (theme.PALETTE keys, resolved at render
# time via _DOT_COLORS lookup so /theme recolors reopened popups).
_DOT_COLORS = {
    "waiting": "accent",
    "running": "success",
    "compacting": "success",
    "idle": "muted",
    "saved": "surface",  # the popup's own background: loaded-nowhere dot
}


def _dot_palette_key(state: str) -> str:
    """Theme palette key for a session row's status dot.

    waiting=accent, running/compacting (active)=success green, idle=the
    plain llm-message gray, saved/not loaded (anything else)=background."""
    return _DOT_COLORS.get(state, "bg")


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


def _same_path(a: str, b: Path | None) -> bool:
    """True when saved workspace ``a`` resolves to the same directory as ``b``."""
    if b is None:
        return True  # no workspace filter: everything matches
    try:
        return Path(a).resolve() == b
    except OSError:
        return False


class ResumeScreen(ModalScreen[str | None]):
    """Filterable list of active + saved sessions; dismisses with the chosen
    name or None."""

    BINDINGS = [("escape", "dismiss_none", "Cancel")]

    def __init__(
        self,
        entries: list[SessionSummary],
        workspace: str | None = None,
    ) -> None:
        super().__init__()
        self._workspace = Path(workspace).resolve() if workspace else None
        # Global section: every active session, any workspace. Then saved
        # sessions of the current workspace. Newest user message first.
        active = [s for s in entries if s.active]
        active.sort(key=lambda s: s.last_user_message_at or "", reverse=True)
        saved = [
            s for s in entries
            if not s.active and _same_path(s.workspace or "", self._workspace)
        ]
        saved.sort(key=lambda s: s.last_user_message_at or "", reverse=True)
        self._entries = [(True, s) for s in active] + [(False, s) for s in saved]
        self._filtered = self._entries

    def compose(self):
        with Vertical(id="resume-modal"):
            yield Label("Open session", id="resume-title")
            yield Input(placeholder="Type to search", id="resume-search")
            yield ListView(*self._build_items(self._entries), id="resume-list")

    def _build_items(
        self, entries: list[tuple[bool, SessionSummary]],
    ) -> list[ListItem]:
        items = []
        for is_active, summary in entries:
            age = _relative_time(summary.last_user_message_at)
            # Right-side status dot, colored by session state:
            #   waiting  -> accent (a prompt needs an answer)
            #   running/compacting (active) -> success green
            #   idle -> the same gray as a plain llm message dot
            #   saved / not loaded -> the popup's background color
            dot_color = theme.PALETTE[_dot_palette_key(summary.state)]
            label = f"{escape(summary.name)}  [dim]{age}[/]"
            item = ListItem(Horizontal(
                Label(label, classes="resume-row-text"),
                Label(f"[b {dot_color}]●[/]", classes="resume-row-dot"),
                classes="resume-row",
            ))
            item.session_name = summary.name
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
            (a, s) for a, s in self._entries if query in s.name.lower()
        ]
        listview = self.query_one("#resume-list", ListView)
        listview.clear()
        for item in self._build_items(self._filtered):
            listview.append(item)
        # clear() drops the cursor (and the -highlight class with it); put it
        # back on the first row so the highlight stays visible while filtering.
        if self._filtered:
            listview.index = 0

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is not None and idx < len(self._filtered):
            self.dismiss(self._filtered[idx][1].name)

    def action_dismiss_none(self) -> None:
        self.dismiss(None)
