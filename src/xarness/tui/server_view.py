"""The foreground host's live dashboard (``xarness --server`` / ``--web``).

The TUI's palette and vocabulary on the one screen that is about the *host*
rather than a conversation: what is hosted, who is attached, what each
session has spent, and — with the browser listener on — where the web UI is
and what token it wants.

It reads the host object directly, because the host runs on this app's own
event loop: ``on_mount`` starts ``Host.serve`` as a worker, the stop button
ends it through ``Host.request_shutdown``, and there is no HTTP round-trip
and no second copy of the data. Sessions stay saved on disk; the dashboard
only ends the process. Without a terminal the CLI keeps the one-line stats
printer instead (see :func:`xarness.host.run_foreground`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Static

from .. import theme
from ..protocol import ServerStats, SessionSummary
from ..util import format_elapsed, format_tokens
from .confirm_screen import ConfirmScreen
from .list_picker import STATE_WORDS, flatten_sessions, relative_time

# The TUI package never imports the backend (see ImportBoundaryTest); this
# is the slice of Host the dashboard uses, duck-typed.


class _HostView(Protocol):
    sessions: Mapping[str, Any]

    def stats(self) -> ServerStats: ...

    async def serve(
        self, *, web: Any = ..., on_ready: Callable[[], None] | None = ...,
    ) -> None: ...

    async def request_shutdown(self) -> None: ...


# The TUI's /sessions cadence: fast enough that uptime ticks visibly.
REFRESH_SECONDS = 0.5

# The table's fixed columns in display order (the key is the heading; the
# number is content width) and the order a narrow terminal gives them up.
# `session` is the flexible column and never goes.
_COLUMNS = (
    ("state", 12), ("clients", 9), ("in", 7), ("out", 7),
    ("msgs", 6), ("model", 14), ("age", 5),
)
_DROP_ORDER = ("age", "model", "msgs", "clients", "out", "in", "state")
_RIGHT = {"clients", "in", "out", "msgs", "age"}
# What the columns cost beyond their widths: every DataTable cell is padded
# on both sides (a width is content, so a column takes width + 2), the sheet
# pads the table itself `0 1`, and a session name needs at least NAME_MIN
# cells. Leaving either padding out is how a header gets clipped.
NAME_MIN = 10
CELL = 2

# The state words and colors the totals line counts, most interesting first.
_STATES = (
    ("running", "success"), ("waiting", "accent"), ("ask", "warning"),
    ("compacting", "success"), ("idle", "muted"),
)


def _cost(columns: Sequence[tuple[str, int]]) -> int:
    """Cells these fixed columns take: every column's width plus one cell of
    padding on each side of it, the session column's padding, and the
    table's own `padding: 0 1`."""
    return sum(width for _, width in columns) + CELL * (len(columns) + 2)


def _visible(width: int) -> list[tuple[str, int]]:
    """The fixed columns this terminal has room for, least useful out first."""
    keep = list(_COLUMNS)
    for key in _DROP_ORDER:
        if _cost(keep) + NAME_MIN <= width:
            break
        keep = [column for column in keep if column[0] != key]
    return keep


def _name_width(width: int, columns: Sequence[tuple[str, int]]) -> int:
    """The room the fixed columns leave for a session name."""
    return max(NAME_MIN, width - _cost(columns))


def _fit(text: str, width: int) -> str:
    """Truncate with an ellipsis so a long value never pushes columns."""
    return text if len(text) <= width else text[:width - 1] + "…"


def _counted(count: int, noun: str) -> str:
    """``1 session``, ``2 sessions`` — the stop prompt stays grammatical."""
    return f"{count} {noun}" + ("" if count == 1 else "s")


class ServerDashboard(App[None]):
    """Live host stats, with a stop button.

    ``web`` is the browser listener to hand the host (``None`` for the unix
    socket alone); ``urls`` and ``token`` describe it in the header, so the
    token is on screen for the whole run instead of scrolling away at
    startup."""

    # The TUI's stylesheet: the palette variables and the modal chrome the
    # stop confirmation needs. The dashboard's own few rules are `CSS` — an
    # App's `CSS` is applied last, so they beat the widgets' own defaults
    # (DEFAULT_CSS of the app would not: it comes before them).
    CSS_PATH = "styles.tcss"
    TITLE = "xarness host"
    # Textual's own palette and ctrl+p menu would be the only surface not in
    # the TUI's style.
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    #head {
        height: auto; padding: 1 1 0 1; background: $c-bg;
        border-bottom: solid $c-border;
    }
    /* The empty-state line takes the table's place, so the footer stays put. */
    #empty { height: 1fr; padding: 1 2; color: $c-muted; }
    DataTable { height: 1fr; padding: 0 1; background: $c-bg; color: $c-text; }
    DataTable > .datatable--header {
        background: $c-bg; color: $c-muted; text-style: bold;
    }
    /* Stripes one step above the background, like the TUI's surface blocks —
       never a second, competing background color. */
    DataTable > .datatable--odd-row { background: $c-surface; }
    DataTable > .datatable--even-row { background: $c-bg; }
    DataTable > .datatable--hover { background: $c-border; }
    /* The stop target, one row tall in the palette's error color like the
       confirm modal's buttons. Two rows: the rule, then the button. */
    #footer {
        height: 2; padding: 0 1; background: $c-bg;
        border-top: solid $c-border; align-vertical: middle;
    }
    #hint { width: 1fr; height: 1; color: $c-muted; content-align: left middle; }
    #stop-host {
        width: auto; min-width: 12; height: 1; padding: 0 2; border: none;
        background: $c-surface; color: $c-error; text-style: bold;
    }
    #stop-host:hover, #stop-host:focus, #stop-host.-active {
        background: $c-error; color: $c-bg;
    }
    #stop-host:disabled { background: $c-surface; color: $c-muted; }
    """

    BINDINGS = [
        Binding("q", "stop", "Stop host"),
        Binding("ctrl+c", "copy_or_stop", "Copy / stop", priority=True),
        Binding("ctrl+q", "stop", "Stop host", priority=True, system=True),
    ]

    def __init__(
        self,
        host: "_HostView",
        *,
        web: Any = None,
        urls: Sequence[str] = (),
        token: str | None = None,
    ) -> None:
        super().__init__()
        self.host = host
        # Opaque here: the dashboard forwards it to Host.serve, which knows
        # how to bind it.
        self.web = web
        self.urls = tuple(urls)
        self.token = token
        # The last read of the host's stats: the caller prints its final
        # line after the dashboard exits. `error` is why the host exited by
        # itself, when it did (a failed bind, a live socket) — the caller
        # turns it into one clear error line instead of a traceback.
        self.stats: ServerStats | None = None
        self.error: BaseException | None = None
        self._stopping = False
        # Named for the host, not the app: Textual's App already owns
        # `_ready`, and its own bool is not this one.
        self._host_ready = False
        # Per-session token totals for the current frame, from the live
        # Session objects: ``usage_total`` is what the TUI's status bar
        # shows.
        self._usage: dict[str, tuple[int, int]] = {}

    def get_css_variables(self) -> dict[str, str]:
        return {**super().get_css_variables(), **theme.CSS_VARIABLES}

    def compose(self) -> ComposeResult:
        yield Static(id="head")
        # The empty-state line sits where the table starts; the table keeps
        # the flexible height either way, so the footer never moves.
        yield Static(
            "no sessions are hosted right now — start the TUI, or open the "
            "web UI, to give the host something to run",
            id="empty",
        )
        yield DataTable(id="sessions")
        with Horizontal(id="footer"):
            yield Static(id="hint")
            yield Button("stop host", id="stop-host")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        # A display, not a focus target: no cursor, and tab lands on stop.
        table.cursor_type = "none"
        table.show_cursor = False
        table.can_focus = False
        table.zebra_stripes = True
        self.set_interval(REFRESH_SECONDS, self._paint)
        self.run_worker(self._serve(), exclusive=True, name="host-server")
        self._paint()

    async def _serve(self) -> None:
        """Run the host on this app's loop; leave when it exits."""
        try:
            await self.host.serve(web=self.web, on_ready=self._on_ready)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 — one line, no traceback
            self.error = exc
        finally:
            self.exit()

    def _on_ready(self) -> None:
        """Both listeners are up: drop the ``starting`` note and repaint."""
        self._host_ready = True
        self._paint()

    # ------------------------------------------------------------------
    # painting

    def _paint(self) -> None:
        try:
            stats = self.host.stats()
            usage = {
                name: (session.usage_total.input_tokens,
                       session.usage_total.output_tokens)
                for name, session in self.host.sessions.items()
            }
        except Exception:  # noqa: BLE001 — a torn-down host is not an error
            return
        self.stats, self._usage = stats, usage
        self.query_one("#head", Static).update(self._head_text(stats))
        self._paint_table(stats)
        self.query_one("#hint", Static).update(self._hint_text())
        button = self.query_one("#stop-host", Button)
        button.disabled = self._stopping
        button.label = "stopping…" if self._stopping else "stop host"

    def _head_text(self, stats: ServerStats) -> Text:
        """Where the host is, then what it is doing — one block, one rule."""
        p = theme.PALETTE
        info = stats.host
        header = Text.assemble(
            ("xarness", f"bold {p['status']}"), ("  host   ", p["muted"]),
            (f"up {format_elapsed(info.uptime_seconds)}", p["text"]),
            (" · ", p["border"]), (f"pid {info.pid}", p["muted"]),
            (" · ", p["border"]),
            (f"v{info.version} protocol {info.protocol_version}", p["muted"]),
        )
        if not self._host_ready:
            header.append(" · ", style=p["border"])
            header.append("starting…", style=p["warning"])
        lines = [
            header,
            Text.assemble(("socket  ", p["muted"]),
                          (info.unix_socket or "?", p["text"])),
        ]
        # URLs arrive once both listeners are actually bound: one shown
        # earlier would advertise something that could still fail to exist.
        if self.urls and self._host_ready:
            lines.append(Text.assemble(("web     ", p["muted"]))
                         + Text("  ·  ", style=p["border"]).join(
                             Text(url, style=p["status"]) for url in self.urls))
            token = (
                Text.assemble((self.token, f"bold {p['accent2']}"),
                              ("   the browser asks for it once", p["muted"]))
                if self.token else Text(
                    "(loopback only, no token — a phone needs --host 0.0.0.0)",
                    style=p["muted"],
                )
            )
            lines.append(Text.assemble(("token   ", p["muted"])) + token)
        lines.append(self._totals_text(stats))
        return Text("\n").join(lines)

    def _totals_text(self, stats: ServerStats) -> Text:
        p = theme.PALETTE
        counts = stats.by_state()
        text = Text.assemble(
            ("sessions ", p["muted"]), (str(len(stats.sessions)), p["text"]),
        )
        for state, colour in _STATES:
            if counts.get(state):
                text += Text.assemble(
                    (" · ", p["border"]),
                    (f"{counts[state]} {state}", p[colour]),
                )
        text += Text.assemble(
            ("   ·   ", p["border"]), ("clients ", p["muted"]),
            (str(stats.clients_attached),
             p["status" if stats.clients_attached else "muted"]),
            ("   ·   ", p["border"]), ("tokens ", p["muted"]),
            (f"↑ {format_tokens(stats.total_input_tokens)} in", p["text"]),
            (" · ", p["border"]),
            (f"↓ {format_tokens(stats.total_output_tokens)} out", p["text"]),
        )
        return text

    def _hint_text(self) -> Text:
        p = theme.PALETTE
        if self._stopping:
            return Text("stopping the host — sessions stay saved on disk",
                        style=p["warning"])
        return Text.assemble(
            ("q", f"bold {p['accent2']}"), (" stop", p["muted"]),
            ("  ·  ", p["border"]),
            ("ctrl+c", f"bold {p['accent2']}"), (" stop / copy", p["muted"]),
        )

    # ------------------------------------------------------------------
    # session table

    def _paint_table(self, stats: ServerStats) -> None:
        """Refill the whole table each frame. At two frames a second, the
        bookkeeping that would update cells in place is not worth it."""
        table = self.query_one(DataTable)
        columns = _visible(self.size.width)
        layout = [("session", _name_width(self.size.width, columns)), *columns]
        rows = flatten_sessions(stats.sessions)
        table.clear(columns=True)
        for key, width in layout:
            table.add_column(
                Text(key, justify="right" if key in _RIGHT else "left"),
                width=width, key=key,
            )
        for summary in rows:
            table.add_row(
                *(self._cell(summary, key, width) for key, width in layout),
                key=summary.name,
            )
        # No headers floating over nothing: the empty-state line is the
        # message, and the table returns when something is hosted.
        self.query_one("#empty", Static).display = not rows
        table.display = bool(rows)

    def _cell(self, summary: SessionSummary, key: str, width: int) -> Text:
        p = theme.PALETTE
        align = "right" if key in _RIGHT else "left"
        if key == "session":
            if summary.parent_id:  # a subagent, indented under its parent
                return Text(_fit(f"↳ {summary.name.rsplit('/', 1)[-1]}", width),
                            style=p["muted"])
            return Text(_fit(summary.name, width), style=p["text"])
        if key == "state":
            # The host's summary is the truth: a hosted session reads idle,
            # running, waiting, ask or compacting — never saved (that word
            # is for a session that is only on disk, and this dashboard only
            # lists hosted ones).
            word, colour = STATE_WORDS.get(summary.state,
                                           (summary.state, "muted"))
            return Text(word, style=p[colour], justify=align)
        if key in ("in", "out"):
            tokens = self._usage.get(summary.name, (0, 0))
            spent = tokens[1] if key == "out" else tokens[0]
            return Text(_fit(format_tokens(spent), width) if spent else "–",
                        style=p["text" if spent else "muted"], justify=align)
        if key == "clients":
            count = summary.clients_attached
            return Text(str(count), style=p["status" if count else "muted"],
                        justify=align)
        if key == "msgs":
            return Text(str(summary.message_count), style=p["muted"],
                        justify=align)
        if key == "model":
            return Text(_fit(summary.model or "–", width), style=p["muted"])
        return Text(_fit(relative_time(summary.last_user_message_at), width),
                    style=p["muted"], justify=align)

    # ------------------------------------------------------------------
    # stopping

    def action_copy_or_stop(self) -> None:
        """ctrl+c copies a selection first (the token is selectable), then
        means "stop the host" — the only thing leaving this screen can do.
        With the stop confirmation open it cancels that, like escape."""
        selected = self.screen.get_selected_text()
        if selected:
            self.copy_to_clipboard(selected)
            return
        if isinstance(self.screen, ModalScreen):
            self.screen.dismiss(None)
            return
        self.action_stop()

    def action_stop(self) -> None:
        if self._stopping or isinstance(self.screen, ModalScreen):
            return
        stats = self.stats
        if stats is None or not stats.sessions:
            detail = "Nothing is running. Saved sessions stay on disk."
        else:
            detail = f"{_counted(len(stats.sessions), 'session')} hosted"
            if stats.clients_attached:
                detail += f", {_counted(stats.clients_attached, 'client')} attached"
            detail += (". Running turns are cancelled; every session stays "
                       "saved on disk.")
        self.push_screen(
            ConfirmScreen(
                "Stop the host?", confirm_label="Stop host",
                cancel_label="Keep running", detail=detail,
            ),
            self._on_stop_choice,
        )

    def _on_stop_choice(self, choice: bool | None) -> None:
        if not choice or self._stopping:
            return
        self._stopping = True
        self._paint()
        self.run_worker(self.host.request_shutdown(), exclusive=False,
                        name="host-shutdown")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "stop-host":
            event.stop()
            self.action_stop()
