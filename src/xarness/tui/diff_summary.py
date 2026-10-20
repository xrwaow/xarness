"""The pending-changes summary bar and its inline diff pane.

Collapsed by default, same click-to-expand interaction as ThinkingBlock:
clicking the summary line toggles the per-file list; clicking a file row
posts a ``DiffRequested`` the App fulfills with a git call, since the
widget holds no git state itself. Hidden entirely when there are no
pending changes.

Keyboard, once the list has focus (escape minimizes an open diff into
this state): up/down move the highlighted row, enter opens its diff, and
a further escape (via the app's priority interrupt binding) collapses
the list back to the bare summary line.
"""

from __future__ import annotations

from typing import cast

from rich.text import Text
from textual import events
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Static

from ..protocol import DiffInfo as DiffStat  # same shape as gitwork.DiffStat
from .chat_input import ChatInput
from .widgets import _render_diff


# Sentinel: no diff section currently shown (None is a valid *key* — the full diff).
_NO_DIFF: object = object()


class DiffText(VerticalScroll):
    """Capped, focusable diff pane with its own scrollbar. Up/down page by
    the pane's full visible height and stop at the pane's top/bottom — they
    never switch files. Left/right step to the previous/next file's diff;
    with the full diff open, right opens the first file's diff.
    At the first/last file the arrows do nothing."""

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
        if not isinstance(key, str):
            # Full multi-file diff open: right steps into the first file.
            if (
                key is None and delta == 1 and summary._stat is not None
                and summary._stat.files
            ):
                summary.post_message(summary.DiffRequested(summary._stat.files[0].path))
                return True
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
        def __init__(self, key: str | None) -> None:
            self.key = key  # None = the full multi-file diff
            super().__init__()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._stat: DiffStat | None = None
        self._active_key: str | None | object = _NO_DIFF
        self._active_text = ""

    def _request_neighbour(self, key: str, delta: int) -> bool:
        """Post a DiffRequested for the file adjacent to ``key`` in the file
        list (delta -1/+1); True if a request was posted."""
        if self._stat is None:
            return False
        paths = [f.path for f in self._stat.files]
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
            yield Static("", id="diff-summary-add", markup=False)
            yield Static("", id="diff-summary-del", markup=False)
        with Vertical(id="diff-body"):
            yield Vertical(id="diff-files")
            # Focusable scroll container: click it (or tab to it) and the
            # diff scrolls with the keyboard; the wheel scrolls on hover.
            with DiffText(id="diff-content"):
                yield Static("", id="diff-text", markup=False)

    @property
    def active_diff_key(self) -> str | None | object:
        return self._active_key

    @property
    def diff_shown(self) -> bool:
        return self._active_key is not _NO_DIFF

    def toggle(self) -> None:
        if self.has_class("expanded"):
            self.remove_class("expanded")
        else:
            self.add_class("expanded")

    def _reset_diff(self) -> None:
        """Drop the open diff pane: forget the active key/text, clear the
        Static and hide the scroll container."""
        self._active_key = _NO_DIFF
        self._active_text = ""
        if self.is_mounted:
            self.query_one("#diff-text", Static).update("")
            self.query_one("#diff-content", DiffText).remove_class("show")

    def minimize_diff(self) -> None:
        """Hide the open diff pane but keep the file list up with the last
        viewed file's row still highlighted, and take focus so arrow keys +
        enter can open another file's diff (escape again collapses)."""
        self._reset_diff()
        self.focus()

    def collapse(self) -> None:
        """Close the expanded file list; only the summary line remains."""
        self.remove_class("expanded")
        self.app.query_one("#chat-input", ChatInput).focus()

    def on_click(self, event: events.Click) -> None:
        # Rows carry a ``diff_key`` attribute; clicks on anything else toggle
        # the expanded file list — except clicks inside the open diff pane,
        # which only minimize the diff (the file list and its highlight stay
        # up, same as esc), not collapse the whole panel.
        node = event.widget
        while node is not None and node is not self:
            diff_key = getattr(node, "diff_key", _NO_DIFF)
            if diff_key is not _NO_DIFF:
                event.stop()
                self.post_message(self.DiffRequested(cast("str | None", diff_key)))
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
        """File-list navigation while focused with no diff open: up/down move
        the row highlight, enter opens the highlighted file's diff. With a
        diff open the keys belong to the diff pane (scroll/page), so stay out
        of the way — they only reach here bubbled up from DiffText."""
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
        for index, row in enumerate(self.query(".diff-file-row")):
            if row.has_class("active"):
                return index
        return None

    def _move_highlight(self, delta: int) -> None:
        rows = list(self.query(".diff-file-row"))
        if not rows:
            return
        current = self._active_row_index()
        index = 0 if current is None else max(0, min(len(rows) - 1, current + delta))
        self._mark_active_row(cast("str | None", getattr(rows[index], "diff_key", None)))

    def _open_highlighted(self) -> None:
        rows = list(self.query(".diff-file-row"))
        index = self._active_row_index()
        if index is None or index >= len(rows):
            return
        key = getattr(rows[index], "diff_key", None)
        if key is not None:
            self.post_message(self.DiffRequested(cast("str | None", key)))

    async def set_summary(self, stat: DiffStat) -> None:
        """Update the summary + per-file rows in place; becomes visible.

        The panel refreshes on every tool finish, so a full teardown/mount of
        the row list makes the widget flicker and loses scroll/highlight
        state. Reconcile instead: drop rows whose file disappeared, mount rows
        for new files, and update the existing rows' numbers/source in place
        when only the counts changed."""
        self._stat = stat
        count = len(stat.files)
        # Colors are CSS ($c-* variables) applied to the Statics — keep the
        # Text unstyled so /theme recolors these rows live.
        self.query_one("#diff-summary-text", Static).update(
            Text(f"Edited {count} file{'s' if count != 1 else ''}")
        )
        self.query_one("#diff-summary-add", Static).update(Text(f"+{stat.additions}"))
        self.query_one("#diff-summary-del", Static).update(Text(f"-{stat.deletions}"))

        files = self.query_one("#diff-files", Vertical)
        # Pad the +/- counts to a uniform width across the file list so the
        # right-aligned tail (and the "new" badge) sits at the same column
        # on every row regardless of digit count.
        add_w = max((len(f"+{f.additions}") for f in stat.files), default=1)
        del_w = max((len(f"-{f.deletions}") for f in stat.files), default=1)

        # Reconcile against the rows already mounted (keyed by diff_key).
        rows: dict[str, Horizontal] = {
            row.diff_key: row  # type: ignore[attr-defined]
            for row in files.children
            if isinstance(row, Horizontal) and hasattr(row, "diff_key")
        }
        wanted = {f.path for f in stat.files}
        for path, row in list(rows.items()):
            if path not in wanted:
                await row.remove()
                rows.pop(path)

        ordered: list[Horizontal] = []
        for f in stat.files:
            row = rows.get(f.path)
            if row is None:
                row = await self._mount_file_row(files, f, add_w, del_w)
            else:
                await self._update_file_row(row, f, add_w, del_w)
            ordered.append(row)

        # Keep DOM order matching the stat order (a row may have moved when
        # the diff list is re-sorted), without touching rows already placed.
        for index, row in enumerate(ordered):
            if files.children.index(row) != index:
                files.move_child(row, before=index)

        # Re-apply the open diff's highlight: rows that survived kept it, but
        # if the active file disappeared no row should stay highlighted. With
        # no diff open, leave the existing classes alone.
        if self.diff_shown:
            active = self._active_key
            self._mark_active_row(active if isinstance(active, str) else None)
        self.add_class("visible")

    async def _mount_file_row(
        self, files: Vertical, f, add_w: int, del_w: int,
    ) -> Horizontal:
        """Create and mount one per-file row, returning it."""
        # Row layout: name  [new]  {source}  +N -M — the whole row is
        # clickable and requests that file's unified diff. The "new" badge
        # gets its own column so it never wraps the file name.
        row = Horizontal(classes="diff-file-row")
        row.diff_key = f.path  # type: ignore[attr-defined]
        await files.mount(row)
        await row.mount(Static(Text(f.path), classes="diff-file-name", markup=False))
        if f.is_new:
            await row.mount(Static(
                Text("new"), classes="diff-file-new", markup=False,
            ))
        # Source tag: "agent" for a turn's edits, "drift" for changes made
        # outside the session.
        source_class = "drift" if "drift" in f.source else "agent"
        await row.mount(Static(
            Text(f.source), classes=f"diff-file-source {source_class}", markup=False,
        ))
        await row.mount(Static(
            Text(f"+{f.additions}".rjust(add_w)), classes="diff-file-add", markup=False,
        ))
        await row.mount(Static(
            Text(f"-{f.deletions}".rjust(del_w)), classes="diff-file-del", markup=False,
        ))
        return row

    async def _update_file_row(
        self, row: Horizontal, f, add_w: int, del_w: int,
    ) -> None:
        """Refresh one existing row's values without remounting it."""
        row.query_one(".diff-file-name", Static).update(Text(f.path))
        source = row.query_one(".diff-file-source", Static)
        source.update(Text(f.source))
        source.set_classes(
            "diff-file-source " + ("drift" if "drift" in f.source else "agent")
        )
        row.query_one(".diff-file-add", Static).update(
            Text(f"+{f.additions}".rjust(add_w))
        )
        row.query_one(".diff-file-del", Static).update(
            Text(f"-{f.deletions}".rjust(del_w))
        )
        # The "new" badge can appear/disappear when the baseline moves (e.g.
        # /accept), but the row itself survives: add/remove only the badge.
        badge = next(iter(row.query(".diff-file-new")), None)
        if f.is_new and badge is None:
            await row.mount(
                Static(Text("new"), classes="diff-file-new", markup=False),
                before=source,
            )
        elif not f.is_new and badge is not None:
            await badge.remove()

    async def clear(self) -> None:
        """Hide entirely — no pending changes (or no worktree)."""
        self._stat = None
        self._reset_diff()
        self.remove_class("expanded")
        self.remove_class("visible")
        if self.is_mounted:
            await self.query_one("#diff-files", Vertical).remove_children()

    def show_diff(self, key: str | None, text: str) -> None:
        """Display one file's (or the full) unified diff inline."""
        self._active_key = key
        self._active_text = text
        self.add_class("visible")
        self.add_class("expanded")
        content = self.query_one("#diff-content", DiffText)
        self.query_one("#diff-text", Static).update(_render_diff(text))
        content.add_class("show")
        content.focus()  # arrow keys page / step files right away
        # Defer until the container is laid out, or scrolling is a no-op.
        content.call_after_refresh(content.scroll_home, animate=False)
        self._mark_active_row(key)

    def hide_diff(self) -> None:
        self._reset_diff()
        if self.is_mounted:
            self._mark_active_row(None)
            # The diff pane may have held focus; hand it back to the input.
            self.app.query_one("#chat-input", ChatInput).focus()

    def _mark_active_row(self, key: str | None) -> None:
        """Highlight the row whose diff is currently displayed."""
        for row in self.query(".diff-file-row"):
            if getattr(row, "diff_key", _NO_DIFF) == key:
                row.add_class("active")
            else:
                row.remove_class("active")

    def recolor(self) -> None:
        """Re-render with the current palette (/theme).

        All row/summary/button colors are CSS variables, so the stylesheet
        recolors them live; only the diff text bakes palette colors into
        rich styles and needs a re-render."""
        if not self.is_mounted:
            return
        if self._active_key is not _NO_DIFF:
            self.query_one("#diff-text", Static).update(_render_diff(self._active_text))
