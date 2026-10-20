"""One modal list picker for every "choose an X" flow.

Covers both plain name lists (/model, /theme, ...) and the filterable session
list (/sessions). Rows are prebuilt ``Option`` objects, so callers own the
presentation and this widget owns filtering, keyboard movement and dismissal.

Filtering rebuilds the OptionList wholesale on every keystroke —
``clear_options()`` + ``add_options()`` is cheap and, unlike mounted widgets,
options carry no mount restrictions.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from rich.text import Text
from textual import events
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, OptionList
from textual.widgets.option_list import Option

from .. import theme
from ..protocol import SessionSummary
from ..util import format_duration

# State word per session state, colored with this palette key; states not
# listed here (saved / not loaded) get no state word.
STATE_WORDS = {
    "waiting": ("waiting", "accent"),
    "running": ("running", "success"),
    "compacting": ("compacting", "success"),
    "idle": ("idle", "muted"),
}

# Right-aligned state column: padded to the longest state word.
STATE_WIDTH = max(len(word) for word, _ in STATE_WORDS.values())


def relative_time(iso: str | None) -> str:
    """Format an ISO timestamp as age from now: 5m / 3h / 12d."""
    if not iso:
        return "?"
    try:
        updated = datetime.fromisoformat(iso)
    except ValueError:
        return "?"
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    seconds = max(0.0, (datetime.now(timezone.utc) - updated).total_seconds())
    days, remainder = divmod(int(seconds), 86400)
    if days:
        return f"{days}d"
    return format_duration(seconds)


def _same_path(a: str, b: Path | None) -> bool:
    """True when saved workspace ``a`` resolves to the same directory as ``b``."""
    if b is None:
        return True  # no workspace filter: everything matches
    try:
        return Path(a).resolve() == b
    except OSError:
        return False


def name_options(names: Sequence[str], current: str | None = None) -> list[Option]:
    """Plain rows for the /model, /theme, ... pickers."""
    return [
        Option(
            f"{name}  (current)" if name == current else name,
            id=name,
        )
        for name in names
    ]


def session_options(
    entries: Sequence[SessionSummary], workspace: str | None,
) -> list[Option]:
    """Global active sessions, then saved sessions of this workspace.

    Each row is one Text: name, dimmed age, right-aligned state word. Both
    sections are newest-user-message first.
    """
    resolved = Path(workspace).resolve() if workspace else None
    active = sorted(
        (s for s in entries if s.active),
        key=lambda s: s.last_user_message_at or "", reverse=True,
    )
    saved = sorted(
        (s for s in entries
         if not s.active and _same_path(s.workspace or "", resolved)),
        key=lambda s: s.last_user_message_at or "", reverse=True,
    )
    rows = []
    for summary in [*active, *saved]:
        row = Text(summary.name)
        row.append(
            f"  {relative_time(summary.last_user_message_at)}",
            style=theme.PALETTE["muted"],
        )
        state_word = STATE_WORDS.get(summary.state)
        if state_word is not None:
            word, color_key = state_word
            row.append("  " + word.rjust(STATE_WIDTH), style=theme.PALETTE[color_key])
        rows.append(Option(row, id=summary.name))
    return rows


class ListPicker(ModalScreen[str | None]):
    """OptionList modal; dismisses with the chosen id or None.

    ``filterable`` adds a search input whose up/down keys move the list
    highlight (so Enter can pick without leaving the box).
    """

    BINDINGS = [("escape", "dismiss_none", "Cancel")]

    def __init__(
        self,
        options: Sequence[Option],
        *,
        title: str | None = None,
        filterable: bool = False,
        placeholder: str = "Type to search",
    ) -> None:
        super().__init__()
        self._options = list(options)
        self._title = title
        self._filterable = filterable
        self._placeholder = placeholder

    def compose(self):
        with Vertical(id="picker-modal", classes="modal-panel"):
            if self._title:
                yield Label(self._title, id="picker-title", classes="modal-title")
            if self._filterable:
                yield Input(
                    placeholder=self._placeholder, id="picker-search",
                    classes="modal-input",
                )
            yield OptionList(id="picker-list")

    def on_mount(self) -> None:
        self._refilter("")
        if self._filterable:
            self.query_one("#picker-search", Input).focus()

    def _refilter(self, query: str) -> None:
        query = query.lower()
        matches = [
            option for option in self._options
            if query in str(option.prompt).lower()
        ]
        option_list = self.query_one("#picker-list", OptionList)
        option_list.clear_options()
        option_list.add_options(matches)
        # Put the cursor on the first row so the highlight (and Enter) lands
        # on a row that is actually shown.
        option_list.highlighted = 0 if matches else None

    def on_key(self, event: events.Key) -> None:
        """Up/down in the search box moves the list highlight."""
        if not self._filterable or event.key not in ("up", "down"):
            return
        event.prevent_default()
        event.stop()
        option_list = self.query_one("#picker-list", OptionList)
        if option_list.option_count == 0:
            return
        current = option_list.highlighted or 0
        option_list.highlighted = (
            (current - 1) % option_list.option_count if event.key == "up"
            else (current + 1) % option_list.option_count
        )

    def on_input_changed(self, event: Input.Changed) -> None:
        self._refilter(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter in the search box picks the highlighted entry."""
        option_list = self.query_one("#picker-list", OptionList)
        if option_list.highlighted is not None:
            option = option_list.get_option_at_index(option_list.highlighted)
            self.dismiss(str(option.id) if option else None)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(str(event.option_id) if event.option_id is not None else None)

    def action_dismiss_none(self) -> None:
        self.dismiss(None)
