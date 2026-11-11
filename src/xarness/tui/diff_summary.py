"""The pending-changes summary bar and its inline diff pane.

Collapsed by default: clicking the summary line toggles the per-file list, and
clicking a row (a file, or an ignored group, which reads as files added from
nothing) posts a ``DiffRequested`` the App answers with a git call — the
widget holds no git state itself. Hidden entirely with no pending changes.

Keyboard, once the list has focus (escape minimizes an open diff into this
state): up/down move the highlighted row, enter opens its diff, a further
escape (the app's priority interrupt binding) collapses the list.

The list leads with this session's own rows; below a full-width rule sits one
foldable section per other kind — a dimmed ``agent`` (other sessions, and the
ignored groups agents created), ``auto`` and ``drift`` — so the default view
is the agent's diffs alone.

The summary line carries one click-to-cycle scope chip, always reading
``agent``: with other sessions included the tail dims (``age`` accent, ``nt``
muted), scoped to this session the whole word is accent-colored. The chip is
display state only; the App saves the choice on the session.
"""

from __future__ import annotations

from typing import cast

from rich.text import Text
from textual import events
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Static

from .. import theme
from ..protocol import DiffInfo as DiffStat  # same shape as gitwork.DiffStat
from ..protocol import DiffScope
from .chat_input import ChatInput
from .widgets import _render_diff


def _display_kind(kind: str) -> str:
    """The badge text: own work reads ``agent`` too, only its accent differs."""
    return "agent" if kind == "mine" else kind


def _section_of(kind: str) -> str:
    """The section a kind belongs to: own work folds with ``agent``."""
    return "agent" if kind in ("mine", "agent") else kind


# Expandable sections, in display order. Only ``agent`` can also lead the
# list (see DiffSummary._inline); the other two never do.
_SECTIONS = ("agent", "auto", "drift")
# Fold arrows for the section headers.
TRIANGLE_CLOSED = "▸"
TRIANGLE_OPEN = "▾"

# One row cell: its classes, then its text. The first class keys the cell, so
# a refresh updates it in place instead of remounting the whole row.
Cell = tuple[str, str]


def _chip_text(all_sessions: bool) -> Text:
    """The chip always reads ``agent``: the ``nt`` tail dims while other
    sessions are included. Palette colors are baked in, so ``recolor``
    repaints it."""
    accent = theme.PALETTE.get("accent", "")
    muted = theme.PALETTE.get("muted", "")
    text = Text()
    if all_sessions:
        text.append("age", style=accent)
        text.append("nt", style=muted)
    else:
        text.append("agent", style=accent)
    return text


class DiffText(VerticalScroll):
    """Capped, focusable diff pane. Up/down page by the pane's full height and
    stop at its top/bottom — they never switch files; left/right step to the
    previous/next row's diff."""

    can_focus = True

    def _summary(self) -> "DiffSummary | None":
        node = self.parent
        while node is not None and not isinstance(node, DiffSummary):
            node = node.parent
        return node  # type: ignore[return-value]

    def _step_file(self, delta: int) -> bool:
        summary = self._summary()
        if summary is None:
            return False
        key = summary._active_key
        if key is None:
            return False
        return summary._request_neighbour(key, delta)

    def action_scroll_up(self) -> None:
        self.scroll_page_up(animate=False)

    def action_scroll_down(self) -> None:
        self.scroll_page_down(animate=False)

    def key_left(self) -> None:
        self._step_file(-1)

    def key_right(self) -> None:
        self._step_file(1)


class DiffSummary(Vertical):
    """Live "Edited N files +A -D" bar above the chat input."""

    can_focus = True  # arrow-key navigation over the file list once minimized

    class DiffRequested(Message):
        def __init__(self, key: str) -> None:
            self.key = key
            super().__init__()

    class ScopeChanged(Message):
        """The chip was clicked; the app saves the scope on the session."""

        def __init__(self, scope: DiffScope) -> None:
            self.scope = scope
            super().__init__()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._stat: DiffStat | None = None
        self._active_key: str | None = None
        self._active_text = ""
        # Which sessions lead the list: "all" of them, or "mine" alone (every
        # other session's row then folds into the agent section).
        self._scope: DiffScope = "all"
        # Sections whose rows are unfolded; every section starts folded, so
        # the default view is the agent's diffs alone.
        self._expanded: set[str] = set()

    @property
    def scope(self) -> DiffScope:
        return self._scope

    async def set_scope(self, scope: DiffScope) -> None:
        """Apply a saved scope; a no-op when nothing changes, so a busy status
        stream does not re-render the list."""
        if scope == self._scope:
            return
        self._scope = scope
        if self.is_mounted:
            await self._apply_scope()

    async def _apply_scope(self) -> None:
        """Re-render after the scope changed: other sessions' rows fold into
        (or out of) the agent section. ``_refresh_rows`` closes an open diff
        whose row is no longer listed rather than letting it float orphaned."""
        self._refresh_chip()
        self._refresh_header()
        if self.diff_shown and isinstance(self._active_key, str):
            # Keep an open diff on screen when narrowing the scope moves its
            # row into a section: unfold that section with it.
            section = self._section_for(self._active_key)
            if section is not None:
                self._expanded.add(section)
        await self._refresh_rows()

    async def toggle_scope(self) -> None:
        """The chip's click: flip the scope, re-render, tell the app (which
        saves it on the session)."""
        self._scope = "mine" if self._scope == "all" else "all"
        await self._apply_scope()
        self.post_message(self.ScopeChanged(self._scope))

    def _request_neighbour(self, key: str, delta: int) -> bool:
        """Post a DiffRequested for the row adjacent to ``key`` (delta
        -1/+1); True if a request was posted."""
        if self._stat is None:
            return False
        paths = self._visible_paths()
        try:
            index = paths.index(key)
        except ValueError:
            return False
        neighbor = index + delta
        if not 0 <= neighbor < len(paths):
            return False
        self.post_message(self.DiffRequested(paths[neighbor]))
        return True

    def compose(self):
        with Horizontal(id="diff-summary-row"):
            yield Static("", id="diff-summary-text", markup=False)
            yield Static("", id="diff-filter", classes="diff-filter", markup=False)
            yield Static("", id="diff-summary-add", markup=False)
            yield Static("", id="diff-summary-del", markup=False)
        with Vertical(id="diff-body"):
            yield Vertical(id="diff-files")
            # Focusable scroll container: click it (or tab to it) and the
            # diff scrolls with the keyboard; the wheel scrolls on hover.
            with DiffText(id="diff-content"):
                yield Static("", id="diff-text", markup=False)

    @property
    def active_diff_key(self) -> str | None:
        return self._active_key

    @property
    def diff_shown(self) -> bool:
        return self._active_key is not None

    def toggle(self) -> None:
        if self.has_class("expanded"):
            self.remove_class("expanded")
        else:
            self.add_class("expanded")

    def _reset_diff(self) -> None:
        """Drop the open diff pane: forget the key/text and hide it."""
        self._active_key = None
        self._active_text = ""
        if self.is_mounted:
            self.query_one("#diff-text", Static).update("")
            self.query_one("#diff-content", DiffText).remove_class("show")

    def minimize_diff(self) -> None:
        """Hide the open diff pane but keep the list up with its row still
        highlighted, and take focus so arrows + enter can open another."""
        self._reset_diff()
        self.focus()

    def collapse(self) -> None:
        """Close the expanded file list; only the summary line remains."""
        self.remove_class("expanded")
        self.app.query_one("#chat-input", ChatInput).focus()

    async def on_click(self, event: events.Click) -> None:
        # The chip flips the scope, headers fold their rows, rows request
        # their diff; anything else toggles the list — except inside the open
        # diff pane, which only minimizes it (as esc does).
        node = event.widget
        while node is not None and node is not self:
            if node.id == "diff-filter":
                event.stop()
                await self.toggle_scope()
                return
            diff_section = getattr(node, "diff_section", None)
            if diff_section is not None:
                event.stop()
                await self.toggle_section(cast("str", diff_section))
                return
            diff_key = getattr(node, "diff_key", None)
            if diff_key is not None:
                event.stop()
                self.post_message(self.DiffRequested(cast("str", diff_key)))
                return
            if node.id == "diff-content":
                event.stop()
                if self.diff_shown:
                    self.minimize_diff()
                return
            node = node.parent
        self.toggle()
        event.stop()

    def on_key(self, event: events.Key) -> None:
        """List navigation while focused with no diff open: up/down move the
        highlight, enter opens it. With a diff open the keys belong to the
        pane (scroll/page), so stay out of the way."""
        if self.diff_shown or not self.has_class("expanded") or self._stat is None:
            return
        if event.key in ("up", "down", "enter"):
            event.prevent_default()
            event.stop()
            if event.key == "enter":
                self._open_highlighted()
            else:
                self._move_highlight(-1 if event.key == "up" else 1)

    def _active_row_index(self) -> int | None:
        for index, row in enumerate(self.query(".diff-row")):
            if row.has_class("active"):
                return index
        return None

    def _move_highlight(self, delta: int) -> None:
        rows = list(self.query(".diff-row"))
        if not rows:
            return
        current = self._active_row_index()
        index = 0 if current is None else max(0, min(len(rows) - 1, current + delta))
        self._mark_active_row(_row_path(rows[index]))

    def _open_highlighted(self) -> None:
        rows = list(self.query(".diff-row"))
        index = self._active_row_index()
        if index is None or index >= len(rows):
            return
        key = _row_path(rows[index])
        if key is not None:
            self.post_message(self.DiffRequested(key))

    async def set_summary(self, stat: DiffStat) -> None:
        """Update the summary + rows in place; becomes visible.

        The panel refreshes on every tool finish, so reconcile instead of
        rebuilding: gone rows are removed, new ones mounted, survivors updated
        in place. A teardown/mount flickers and loses the scroll position and
        highlight when only one row changed."""
        self._stat = stat
        self._refresh_chip()
        self._refresh_header()
        await self._refresh_rows()
        self.add_class("visible")

    def _refresh_chip(self) -> None:
        """Repaint the scope chip: every session, or this session alone."""
        if self.is_mounted:
            self.query_one("#diff-filter", Static).update(
                _chip_text(self._scope == "all"))

    def _refresh_header(self) -> None:
        """Paint the label and the +A/-D totals: the count is the whole pending
        diff, the totals follow the scope (the section headers carry their
        own), right-aligned to the row columns so the badges line up."""
        stat = self._stat
        if stat is None:
            return
        count = len(stat.files)
        label = f"Edited {count} file{'s' if count != 1 else ''}"
        if stat.groups:
            label += f" · {stat.ignored_files} ignored"
        shown = [f for f in stat.files if self._inline(f)]
        add_w, del_w = self._widths()
        self.query_one("#diff-summary-text", Static).update(Text(label))
        self.query_one("#diff-summary-add", Static).update(
            Text(f"+{sum(f.additions for f in shown)}".rjust(add_w))
        )
        self.query_one("#diff-summary-del", Static).update(
            Text(f"-{sum(f.deletions for f in shown)}".rjust(del_w))
        )
        self.add_class("visible")

    def _widths(self) -> tuple[int, int]:
        """Uniform widths for the +A/-D columns: the widest string across every
        row and every total, so the badge before the numbers sits at the same
        column everywhere."""
        stat = self._stat
        if stat is None:
            return 1, 1
        totals = [[f for f in stat.files if self._inline(f)]]
        totals += [self._section_entries(section)[0] for section in _SECTIONS]
        return (
            max([len(f"+{f.additions}") for f in stat.files]
                + [len(f"+{sum(f.additions for f in g)}") for g in totals]),
            max([len(f"-{f.deletions}") for f in stat.files]
                + [len(f"-{sum(f.deletions for f in g)}") for g in totals]),
        )

    def _inline(self, f) -> bool:
        """Whether a file row leads the list: this session's work, plus other
        sessions' while the scope includes them."""
        return f.kind == "mine" or (f.kind == "agent" and self._scope == "all")

    def _section_entries(self, section: str) -> tuple[list, list]:
        """The (files, groups) a section holds: what the list does not lead
        with."""
        stat = self._stat
        if stat is None:
            return [], []
        return (
            [
                f for f in stat.files
                if _section_of(f.kind) == section and not self._inline(f)
            ],
            [g for g in stat.groups if _section_of(g.kind) == section],
        )

    def _visible_paths(self) -> list[str]:
        """Row paths in display order — what up/down/enter and the pane's
        left/right step through (folded sections are absent)."""
        return [
            path for path in (
                _row_path(row) for row in self.query(".diff-row")
            ) if path is not None
        ]

    async def _refresh_rows(self) -> None:
        """Reconcile the list against the current stat: own files, then the
        rule and one section per kind (header always present, rows only while
        unfolded). Rows are keyed so a refresh reuses the widget and its
        highlight when only the numbers changed."""
        stat = self._stat
        if stat is None:
            return
        files = self.query_one("#diff-files", Vertical)
        add_w, del_w = self._widths()
        # (key, row classes, cells) in display order; rows after the rule are
        # indented so the fold structure reads.
        wanted: list[tuple[tuple[str, str], str, list[Cell], bool]] = [
            (("file", f.path), "diff-file-row diff-row",
             self._file_cells(f, add_w, del_w), False)
            for f in stat.files if self._inline(f)
        ]
        wanted.append((("divider", ""), "diff-divider", [], False))
        for section in _SECTIONS:
            expanded = section in self._expanded
            entries = self._section_entries(section)
            state = (" expanded" if expanded else "") + (
                "" if any(entries) else " empty")
            wanted.append((
                ("section", section), f"diff-section-row {section}{state}",
                self._section_cells(section, entries, expanded, add_w, del_w),
                False,
            ))
            if not expanded:
                continue
            section_files, section_groups = entries
            wanted += [
                (("file", f.path), "diff-file-row diff-row",
                 self._file_cells(f, add_w, del_w), True)
                for f in section_files
            ]
            wanted += [
                (("group", g.prefix), "diff-group-row diff-row",
                 self._group_cells(g), True)
                for g in section_groups
            ]

        wanted_keys = {key for key, *_ in wanted}
        existing = {
            getattr(row, "diff_row_key", None): row
            for row in list(files.children)
            if getattr(row, "diff_row_key", None) is not None
        }
        for key, row in list(existing.items()):
            if key not in wanted_keys:
                await row.remove()
                existing.pop(key)

        ordered = []
        for key, classes, cells, nested in wanted:
            row = existing.get(key)
            if row is None:
                row = Horizontal(classes=classes) if cells else Static(
                    "", classes=classes, markup=False,
                )
                row.diff_row_key = key  # type: ignore[attr-defined]
                # The semantic handle the click handler navigates by (the
                # keyed one above only identifies the row across refreshes).
                if key[0] == "section":
                    row.diff_section = key[1]  # type: ignore[attr-defined]
                else:
                    # A file's path, or an ignored group's prefix: a click
                    # asks the app for that path's diff (a group's being the
                    # read-only "added" patch of everything under it).
                    row.diff_key = key[1]  # type: ignore[attr-defined]
                await files.mount(row)
            else:
                row.set_classes(classes)
            row.set_class(nested, "nested")
            await _sync_cells(row, cells)
            ordered.append(row)

        # Keep DOM order matching the wanted order without touching rows
        # already in place.
        for index, row in enumerate(ordered):
            if files.children.index(row) != index:
                files.move_child(row, before=index)

        if self.diff_shown:
            active = self._active_key
            if active not in {_row_path(row) for row in ordered}:
                # The row folded away (its section closed or its scope
                # dropped): close the pane instead of floating it orphaned,
                # and keep the keyboard on the list for arrow navigation —
                # unless the user had moved it elsewhere (e.g. the input).
                self._reset_diff()
                focused = self.screen.focused
                if focused is not None and self in focused.ancestors_with_self:
                    self.focus()
            else:
                self._mark_active_row(active if isinstance(active, str) else None)

    def _file_cells(self, f, add_w: int, del_w: int) -> list[Cell]:
        """One per-file row: name, badges, kind, then the counts in their own
        right-hand columns (so they never wrap the name)."""
        cells = [("diff-file-name", f.path)]
        if f.is_new:
            cells.append(("diff-file-new", "new"))
        if len(f.authors) > 1:
            cells.append(("diff-file-shared", str(len(f.authors))))
        cells.append((f"diff-file-source {f.kind}", _display_kind(f.kind)))
        cells.append(("diff-file-add", f"+{f.additions}".rjust(add_w)))
        cells.append(("diff-file-del", f"-{f.deletions}".rjust(del_w)))
        return cells

    @staticmethod
    def _group_cells(group) -> list[Cell]:
        """One ignored-path group row: its prefix (the directory's or just
        the one file) and how many files hide behind it."""
        return [
            ("diff-file-name", group.prefix),
            (f"diff-file-source {group.kind}", _display_kind(group.kind)),
            ("diff-group-count", f"{group.files} files"),
        ]

    @staticmethod
    def _section_cells(
        section: str, entries: tuple[list, list], expanded: bool,
        add_w: int, del_w: int,
    ) -> list[Cell]:
        """One section header: label, fold arrow, then the ignored count and
        the section's line totals."""
        section_files, section_groups = entries
        ignored = sum(g.files for g in section_groups)
        adds = sum(f.additions for f in section_files)
        dels = sum(f.deletions for f in section_files)
        return [
            ("diff-section-name", section),
            ("diff-section-toggle", TRIANGLE_OPEN if expanded else TRIANGLE_CLOSED),
            ("diff-section-spacer", ""),
            ("diff-section-count", f"{ignored} files" if ignored else ""),
            ("diff-file-add", f"+{adds}".rjust(add_w)),
            ("diff-file-del", f"-{dels}".rjust(del_w)),
        ]

    async def toggle_section(self, section: str) -> None:
        """Fold/unfold one section's rows."""
        if section in self._expanded:
            self._expanded.discard(section)
        else:
            self._expanded.add(section)
        await self._refresh_rows()

    async def clear(self) -> None:
        """Hide entirely — no pending changes (or no worktree)."""
        self._stat = None
        self._reset_diff()
        self._expanded.clear()  # a fresh pending set starts folded
        self.remove_class("expanded")
        self.remove_class("visible")
        if self.is_mounted:
            await self.query_one("#diff-files", Vertical).remove_children()

    def show_diff(self, key: str, text: str) -> None:
        """Display one file's (or group's) unified diff inline.

        A section-held row (auto, drift, or another session's work while
        scoped to this one) unfolds its section first, so the highlighted row
        is really on screen — even for ``/diff <path>`` typed directly."""
        self._active_key = key
        self._active_text = text
        self.add_class("visible")
        self.add_class("expanded")
        section = self._section_for(key)
        if section is not None and section not in self._expanded:
            self._expanded.add(section)
            if self.is_mounted:
                self.run_worker(self._refresh_rows())
        content = self.query_one("#diff-content", DiffText)
        self.query_one("#diff-text", Static).update(_render_diff(text))
        content.add_class("show")
        content.focus()  # arrow keys page / step files right away
        # Defer until the container is laid out, or scrolling is a no-op.
        content.call_after_refresh(content.scroll_home, animate=False)
        self._mark_active_row(key)

    def _section_for(self, key: str) -> str | None:
        """The folded section a path belongs to, if any."""
        stat = self._stat
        if stat is None:
            return None
        for f in stat.files:
            if f.path == key and not self._inline(f):
                return _section_of(f.kind)
        for g in stat.groups:
            if g.prefix == key:
                return _section_of(g.kind)
        return None

    def hide_diff(self) -> None:
        self._reset_diff()
        if self.is_mounted:
            self._mark_active_row(None)
            # The diff pane may have held focus; hand it back to the input.
            self.app.query_one("#chat-input", ChatInput).focus()

    def _mark_active_row(self, key: str | None) -> None:
        """Highlight the row whose diff is currently displayed."""
        for row in self.query(".diff-row"):
            if _row_path(row) == key:
                row.add_class("active")
            else:
                row.remove_class("active")

    def recolor(self) -> None:
        """Re-render with the current palette (/theme): CSS variables recolor
        the rows live, but the chip and the diff text bake palette colors into
        rich styles."""
        if not self.is_mounted:
            return
        self._refresh_chip()
        if self._active_key is not None:
            self.query_one("#diff-text", Static).update(_render_diff(self._active_text))


def _row_path(row) -> str | None:
    """The path a row leads to (a file's, an ignored group's prefix), or None
    for a section header."""
    return getattr(row, "diff_key", None)


async def _sync_cells(row, cells: list[Cell]) -> None:
    """Bring one row's cells in line with ``cells``: update the survivors,
    mount the new ones in place and drop the gone ones, so a refresh never
    rebuilds the row."""
    existing = {
        getattr(cell, "diff_cell", None): cell for cell in row.children
    }
    for index, (classes, text) in enumerate(cells):
        name = classes.split()[0]
        cell = existing.pop(name, None)
        if cell is None:
            cell = Static(Text(text), classes=classes, markup=False)
            cell.diff_cell = name  # type: ignore[attr-defined]
            await row.mount(cell, before=index)
        else:
            cell.set_classes(classes)
            cell.update(Text(text))
    for cell in existing.values():
        await cell.remove()
