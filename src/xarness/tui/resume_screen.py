"""Modal for selecting a chat session to open (attach or load).

Data comes from the host API (``GET /v1/sessions``): a global section first,
listing all **active** sessions across all workspaces, then saved sessions
for the current workspace. Each row shows the session name, the dimmed age
of its last user message and — for loaded sessions only — a state dot:
waiting=accent, running/compacting (active)=success green, idle=the plain
message gray. Saved (not loaded) sessions show no state word. Both sections
sort by ``last_user_message_at``, newest first.

Filtering rebuilds the OptionList wholesale on every keystroke —
``clear_options()`` + ``add_options()`` is cheap and, unlike mounted
widgets, options carry no mount restrictions.
"""

from __future__ import annotations

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
_STATE_WORDS = {
    "waiting": ("waiting", "accent"),
    "running": ("running", "success"),
    "compacting": ("compacting", "success"),
    "idle": ("idle", "muted"),
}

# Right-aligned state column: padded to the longest state word.
_STATE_WIDTH = max(len(word) for word, _ in _STATE_WORDS.values())


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
    seconds = max(0.0, (datetime.now(timezone.utc) - updated).total_seconds())
    return format_duration(seconds)


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

    def compose(self):
        with Vertical(id="resume-modal"):
            yield Label("Open session", id="resume-title")
            yield Input(placeholder="Type to search", id="resume-search")
            yield OptionList(id="resume-list")

    def on_mount(self) -> None:
        self._refilter("")
        self.query_one("#resume-search", Input).focus()

    def _row(self, summary: SessionSummary) -> Option:
        """One row as a single Text: name, dimmed age, right-aligned state word."""
        row = Text(summary.name)
        row.append(f"  {_relative_time(summary.last_user_message_at)}",
                   style=theme.PALETTE["muted"])
        state_word = _STATE_WORDS.get(summary.state)
        if state_word is not None:
            word, color_key = state_word
            row.append("  " + word.rjust(_STATE_WIDTH),
                       style=theme.PALETTE[color_key])
        return Option(row, id=summary.name)

    def _refilter(self, query: str) -> None:
        query = query.lower()
        matches = [
            summary for _, summary in self._entries if query in summary.name.lower()
        ]
        option_list = self.query_one("#resume-list", OptionList)
        option_list.clear_options()
        option_list.add_options(self._row(s) for s in matches)
        # Put the cursor on the first row so the highlight (and Enter) lands
        # on a row that is actually shown.
        option_list.highlighted = 0 if matches else None

    def on_key(self, event: events.Key) -> None:
        """Up/down in the search box moves the list highlight."""
        if event.key not in ("up", "down"):
            return
        event.prevent_default()
        event.stop()
        option_list = self.query_one("#resume-list", OptionList)
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
        """Enter in the search box picks the highlighted session."""
        option_list = self.query_one("#resume-list", OptionList)
        if option_list.highlighted is not None:
            self.dismiss(self._picked(option_list))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(str(event.option_id) if event.option_id is not None else None)

    def _picked(self, option_list: OptionList) -> str | None:
        option = option_list.get_option_at_index(option_list.highlighted)
        return str(option.id) if option else None

    def action_dismiss_none(self) -> None:
        self.dismiss(None)
