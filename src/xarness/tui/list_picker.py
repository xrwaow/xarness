"""One modal list picker for every "choose an X" flow.

Covers both plain name lists (/model, /theme, ...) and the filterable session
list (/sessions). Rows are prebuilt ``Option`` objects, so callers own the
presentation and this widget owns filtering, keyboard movement and dismissal.

Filtering rebuilds the OptionList wholesale on every keystroke —
``clear_options()`` + ``add_options()`` is cheap and, unlike mounted widgets,
options carry no mount restrictions.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from rich.text import Text
from textual import events
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Input, Label, OptionList
from textual.widgets.option_list import Option

from .. import theme
from ..protocol import SessionSummary
from ..util import format_duration, parse_iso

# State word per session state, colored with this palette key; states not
# listed here (saved / not loaded) get no state word. "ask" is a session
# blocked on its ask tool: the user owes it an answer, so it gets its own
# word (and the attention color) rather than being lumped into "waiting".
STATE_WORDS = {
    "ask": ("ask", "warning"),
    "waiting": ("waiting", "accent"),
    "running": ("running", "success"),
    "compacting": ("compacting", "success"),
    "idle": ("idle", "muted"),
}

# Right-aligned state column: padded to the longest state word.
STATE_WIDTH = max(len(word) for word, _ in STATE_WORDS.values())

# Usable row width inside the session picker, derived from the CSS:
# #picker-modal is PICKER_MODAL_WIDTH wide, .modal-panel pads 2 per side
# and each option pads 1 per side — width - 2*2 - 2*1 characters of
# content. Keep in step with styles.tcss. The age, dir and +/- columns are
# fixed; only the name flexes.
PICKER_MODAL_WIDTH = 62
SESSION_ROW_WIDTH = PICKER_MODAL_WIDTH - 6

# A blank between columns; the +/- fields are one blank apart (their signs
# are part of the field).
COLUMN_GAP = 2
DIFF_GAP = 1
NAME_MIN_WIDTH = 8
AGE_WIDTH = 6
# +/- fields: sign glued to a 3-digit number, right-aligned ("  +2",
# "+410" / "  -1", "-431").
PLUS_FIELD = 4
MINUS_FIELD = 4
# Workspace column: the innermost directory, capped at a quarter of the
# row so long names cannot squeeze the title out.
DIR_MAX_WIDTH = SESSION_ROW_WIDTH // 4

# Id of the blank one-line gap between the two /sessions sections (live
# sessions with a state word, then this workspace's saved sessions). It is
# disabled, so the cursor never stops on it, and the search box drops it.
SPACER_ID = "\0spacer"


@dataclass(frozen=True)
class _RowLayout:
    """Column widths shared by every row of one session list."""

    name_width: int
    dir_width: int

    @classmethod
    def fit(
        cls, titles: Iterable[str], directories: Iterable[str],
    ) -> _RowLayout:
        """Fix the title column at the longest title, then widen dir.

        The title width is the max over the current rows (never more than
        the row leaves free), so every row pads its title to the same
        width and the columns that follow line up like a table."""
        dir_width = min(max(map(len, directories), default=0), DIR_MAX_WIDTH)
        fixed = (
            COLUMN_GAP + AGE_WIDTH
            + (COLUMN_GAP + dir_width if dir_width else 0)
            + COLUMN_GAP + PLUS_FIELD
            + DIFF_GAP + MINUS_FIELD
            + COLUMN_GAP + STATE_WIDTH
        )
        available = max(NAME_MIN_WIDTH, SESSION_ROW_WIDTH - fixed)
        name_width = min(max(map(len, titles), default=0), available)
        return cls(max(NAME_MIN_WIDTH, name_width), dir_width)


def relative_time(iso: str | None) -> str:
    """Format an ISO timestamp as age from now: 5m / 3h / 12d."""
    updated = parse_iso(iso)
    if updated is None:
        return "?"
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    seconds = max(0.0, (datetime.now(timezone.utc) - updated).total_seconds())
    days, remainder = divmod(int(seconds), 86400)
    if days:
        return f"{days}d"
    return format_duration(seconds)


def innermost_dir(workspace: str | None) -> str:
    """Last directory of a workspace path (``a/b/c`` -> ``c``)."""
    if not workspace:
        return ""
    return Path(workspace).name or workspace


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
    current: str | None = None,
) -> list[Option]:
    """Live groups, then saved groups — only this workspace's sessions.

    ``workspace`` scopes the whole list: the picker is about the sessions
    of the checkout the TUI runs in, so live sessions hosted for another
    workspace are dropped along with the saved ones. ``None`` means no
    filter (every session matches).

    ``current`` is the session the picker was opened from: a subagent is
    listed only from its own family (its parent, an ancestor or one of its
    children), so a draft chat shows top-level sessions alone. ``None``
    means no current session (a draft chat).

    Each row is one Text: name, dimmed age, innermost workspace directory,
    the session's own unaccepted (``mine``) +/- totals, right-aligned state
    word. A subagent is shown as ``agent-N`` immediately under its parent.
    A whole group moves to the live section when any member is live, with
    the active children above the idle (saved/not loaded) ones. Both
    sections are newest-group first, separated by one blank row of padding
    when both are non-empty.
    """
    resolved = Path(workspace).resolve() if workspace else None
    entries = [
        entry for entry in entries
        if _same_path(entry.workspace or "", resolved)
    ]
    by_name = {entry.name: entry for entry in entries}
    # Without its family a child row would read as a stray ``agent-N`` at
    # the top level, so it is left out of the list entirely.
    family = _family(current, by_name)
    entries = [entry for entry in entries
               if entry.parent_id is None or entry.name in family]
    groups: dict[str, list[SessionSummary]] = {}
    for entry in entries:
        groups.setdefault(_group_root(entry, by_name), []).append(entry)
    # A group is live when any member is hosted: attaching to an idle
    # subagent keeps its parent and siblings above the gap with it.
    ordered_groups = sorted(
        groups.values(), key=_group_activity, reverse=True,
    )
    live_groups = [members for members in ordered_groups
                   if any(entry.active for entry in members)]
    saved_groups = [members for members in ordered_groups
                    if not any(entry.active for entry in members)]
    active = [entry for members in live_groups
              for entry in _flatten(members)]
    saved = [entry for members in saved_groups
             for entry in _flatten(members)]
    # Column widths are derived from the data so every row lines up as a
    # table: the title is padded to the longest title of the list, the age
    # is right-aligned, the workspace dir left-aligned, the session's own
    # unaccepted (``mine``) +/- totals in fixed +/- sub-columns (blank for
    # tracked-but-clean or untracked sessions), state word flush right.
    live_names = {summary.name for summary in active}
    rows_data = []
    for summary in [*active, *saved]:
        tracked = summary.tracking and (summary.additions or summary.deletions)
        # A saved child kept live only because another member of its group
        # is active is idle, not an old on-disk session: show that state.
        state = "idle" if (summary.name in live_names
                           and summary.state == "saved") else summary.state
        rows_data.append((
            summary,
            _title(summary),
            state,
            relative_time(summary.last_user_message_at),
            innermost_dir(summary.workspace),
            f"+{summary.additions}" if tracked and summary.additions else "",
            f"-{summary.deletions}" if tracked and summary.deletions else "",
        ))
    layout = _RowLayout.fit(
        (title for _, title, *_ in rows_data),
        (directory for *_, directory, _, _ in rows_data),
    )

    def row(entry) -> Option:
        summary, title, state, age, directory, plus, minus = entry
        return Option(
            _session_row(
                title, state, age, directory, plus, minus, layout,
            ),
            id=summary.name,
        )

    live = rows_data[:len(active)]
    saved_rows = rows_data[len(active):]
    options = [row(entry) for entry in live]
    if live and saved_rows:
        # One blank row of padding between the live sessions (the rows that
        # carry a state word) and the saved sessions of this workspace.
        options.append(_spacer())
    options.extend(row(entry) for entry in saved_rows)
    return options


def _family(
    current: str | None, by_name: dict[str, SessionSummary],
) -> set[str]:
    """``current`` plus its ancestors and every descendant under it."""
    def ancestry(name: str) -> set[str]:
        chain: set[str] = set()
        entry = by_name.get(name)
        while entry is not None and entry.parent_id not in chain:
            chain.add(entry.parent_id)
            entry = by_name.get(entry.parent_id)
        return chain

    if current not in by_name:
        return set()
    return {name for name in by_name if name == current
            or current in ancestry(name)} | ancestry(current)


def _group_root(
    entry: SessionSummary, by_name: dict[str, SessionSummary],
) -> str:
    """The name of the topmost ancestor present in the listing."""
    seen = {entry.name}
    current = entry
    while (
        current.parent_id
        and current.parent_id in by_name
        and current.parent_id not in seen
    ):
        seen.add(current.parent_id)
        current = by_name[current.parent_id]
    return current.name


def _group_activity(members: Sequence[SessionSummary]) -> str:
    """The group's newest user message, which orders the group as a unit."""
    return max((entry.last_user_message_at or "" for entry in members),
               default="")


def _flatten(members: Sequence[SessionSummary]) -> list[SessionSummary]:
    """Each member once, parent before children, active children first.

    The section order is recency; within one group a parent stays above
    its children, and among siblings the hosted ones come before the idle
    (saved/not loaded) ones."""
    names = {entry.name for entry in members}
    children: dict[str, list[SessionSummary]] = {}
    roots: list[SessionSummary] = []
    for entry in members:
        parent = entry.parent_id
        if parent and parent in names and parent != entry.name:
            children.setdefault(parent, []).append(entry)
        else:
            roots.append(entry)
    roots.sort(key=lambda entry: entry.last_user_message_at or "",
               reverse=True)
    ordered: list[SessionSummary] = []
    seen: set[str] = set()

    def add(entry: SessionSummary) -> None:
        if entry.name in seen:
            return
        seen.add(entry.name)
        ordered.append(entry)
        kids = sorted(
            children.get(entry.name, []),
            key=lambda child: child.last_user_message_at or "", reverse=True,
        )
        # A stable second pass: active siblings first, recency kept inside
        # each of the two runs.
        kids.sort(key=lambda child: not child.active)
        for child in kids:
            add(child)

    for entry in roots:
        add(entry)
    for entry in members:  # cycles or an out-of-order child
        add(entry)
    return ordered


def _spacer() -> Option:
    """A blank, unselectable row: visible padding, never a choice."""
    return Option(Text(" "), id=SPACER_ID, disabled=True)


def _next_enabled(options: Sequence[Option], start: int, step: int) -> int | None:
    """First enabled index at ``start + step, start + 2*step, …`` (wrapping)."""
    count = len(options)
    for offset in range(1, count + 1):
        index = (start + step * offset) % count
        if not options[index].disabled:
            return index
    return None


def _matches(option: Option, query: str) -> bool:
    """Filter test; the blank gap row only shows while nothing is typed."""
    if option.id == SPACER_ID:
        return not query
    return query in str(option.prompt).lower()


def _title(summary: SessionSummary) -> str:
    """The name shown for a session: a child is only its short half.

    Subagents are stored as ``<parent>/agent-N``; under their parent the
    short half is all there is to show (and can repeat across groups)."""
    if summary.parent_id:
        return "↳ " + summary.name.rsplit("/", 1)[-1]
    return summary.name


def _session_row(
    title: str,
    state: str,
    age: str,
    directory: str,
    plus: str,
    minus: str,
    layout: _RowLayout,
) -> Text:
    """One session row: title | age | dir | +/- | ... | state.

    The title is padded (and truncated, if need be) to the fixed title
    column, and the age, dir, +/- and state columns keep their width even
    when a row has nothing to show in them, so every column lines up
    across the list.
    """
    state_word = STATE_WORDS.get(state)
    word, color_key = state_word if state_word else ("", "")
    if len(title) > layout.name_width:
        title = title[:layout.name_width - 1] + "…"
    row = Text(title.ljust(layout.name_width))
    row.append(
        " " * COLUMN_GAP + age.rjust(AGE_WIDTH),
        style=theme.PALETTE["muted"],
    )
    if layout.dir_width:
        if len(directory) > layout.dir_width:
            directory = "…" + directory[-(layout.dir_width - 1):]
        row.append(
            " " * COLUMN_GAP + directory.ljust(layout.dir_width),
            style=theme.PALETTE["muted"],
        )
    # Both fields are always emitted (blank when unused) so the +/-
    # column keeps its width on every row.
    row.append(
        " " * COLUMN_GAP
        + (plus.rjust(PLUS_FIELD) if plus else " " * PLUS_FIELD),
        style=theme.PALETTE["success"],
    )
    row.append(
        " " * DIFF_GAP
        + (minus.rjust(MINUS_FIELD) if minus else " " * MINUS_FIELD),
        style=theme.PALETTE["error"],
    )
    if word:
        pad = SESSION_ROW_WIDTH - row.cell_len - len(word)
        row.append(
            " " * max(pad, COLUMN_GAP) + word,
            style=theme.PALETTE[color_key],
        )
    return row


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
            if _matches(option, query)
        ]
        option_list = self.query_one("#picker-list", OptionList)
        # Keep the cursor on the same entry across a live refresh or a
        # keystroke when it still matches; only fall back to the first row.
        highlighted_id = None
        if option_list.option_count and option_list.highlighted is not None:
            highlighted_id = option_list.get_option_at_index(
                option_list.highlighted,
            ).id
        option_list.clear_options()
        option_list.add_options(matches)
        if not matches:
            option_list.highlighted = None
            return
        # Stay on the highlighted row across a keystroke/refresh; otherwise
        # land on the first real row (the spacer is never highlightable).
        option_list.highlighted = next(
            (i for i, option in enumerate(matches) if option.id == highlighted_id),
            _next_enabled(matches, -1, 1),
        )

    def _query(self) -> str:
        """The current filter text (empty when not filterable/unmounted)."""
        if not self._filterable:
            return ""
        try:
            return self.query_one("#picker-search", Input).value
        except NoMatches:
            return ""

    def update_options(self, options: Sequence[Option]) -> None:
        """Replace the rows in place, keeping the filter and highlight.

        Used by the live /sessions refresh: re-fetching the summaries must
        not move the cursor or drop what the user typed. Safe to call
        before the screen has mounted (the options are kept for compose)."""
        self._options = list(options)
        if not self.is_mounted:
            return
        try:
            self._refilter(self._query())
        except NoMatches:  # raced the mount: on_mount will render them
            pass

    def on_key(self, event: events.Key) -> None:
        """Up/down in the search box moves the list highlight."""
        if not self._filterable or event.key not in ("up", "down"):
            return
        event.prevent_default()
        event.stop()
        option_list = self.query_one("#picker-list", OptionList)
        if option_list.option_count == 0:
            return
        step = -1 if event.key == "up" else 1
        current = option_list.highlighted
        # Skip the blank spacer between the sections: it is not a choice.
        target = _next_enabled(
            list(option_list.options),
            0 if current is None else current,
            step,
        )
        if target is not None:
            option_list.highlighted = target

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
