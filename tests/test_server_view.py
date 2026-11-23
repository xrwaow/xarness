"""The foreground host dashboard (``xarness --server`` / ``--web``).

The host itself is faked: these tests are about the view — the session rows,
the per-session token columns, the browser listener's URLs/token and the
stop button's confirmation path.
"""

import asyncio
from types import SimpleNamespace

from textual.screen import Screen
from textual.selection import Selection
from textual.geometry import Offset
from textual.widgets import DataTable, Static

from xarness.conversation import Usage
from xarness.protocol import ServerInfo, ServerStats, SessionSummary
from xarness.tui.confirm_screen import ConfirmScreen
from xarness.tui.server_view import ServerDashboard


class FakeHost:
    """Just the slice of Host the dashboard touches."""

    def __init__(
        self,
        summaries: list[SessionSummary],
        usages: dict[str, Usage] | None = None,
    ) -> None:
        usages = usages or {}
        self.sessions = {
            summary.name: SimpleNamespace(
                usage_total=usages.get(summary.name, Usage()),
            )
            for summary in summaries
        }
        self._summaries = summaries
        self._stop = asyncio.Event()
        self.shutdown_requested = False
        self.ready = False
        self.web = None

    def stats(self) -> ServerStats:
        return ServerStats(
            host=ServerInfo(
                version="0.1.0", protocol_version=23, pid=4242,
                uptime_seconds=135.0, persistent=True,
                unix_socket="/home/x/.local/share/xarness/host.sock",
            ),
            started_at="2026-01-01T00:00:00+00:00",
            clients_attached=sum(s.clients_attached for s in self._summaries),
            total_input_tokens=sum(
                u.input_tokens for u in
                (s.usage_total for s in self.sessions.values())
            ),
            total_output_tokens=sum(
                u.output_tokens for u in
                (s.usage_total for s in self.sessions.values())
            ),
            sessions=list(self._summaries),
        )

    async def serve(self, *, web=None, on_ready=None) -> None:
        self.web = web
        if on_ready is not None:
            on_ready()
        self.ready = True
        await self._stop.wait()

    async def request_shutdown(self) -> None:
        self.shutdown_requested = True
        self._stop.set()


def _summary(name: str, **kwargs) -> SessionSummary:
    fields = {
        "state": "running", "active": True, "model": "glm-flash",
        "message_count": 4,
    }
    fields.update(kwargs)
    return SessionSummary(name=name, **fields)


def _host() -> FakeHost:
    return FakeHost(
        [
            _summary("build-api", clients_attached=2, message_count=18),
            _summary(
                "build-api/agent-1", state="waiting", parent_id="build-api",
                message_count=5,
            ),
            _summary("scratch-notes", state="idle", active=False,
                     model="deepseek", message_count=2),
        ],
        {
            "build-api": Usage(input_tokens=12_345, output_tokens=4_567),
            "build-api/agent-1": Usage(input_tokens=3_100, output_tokens=900),
            "scratch-notes": Usage(input_tokens=120, output_tokens=40),
        },
    )


def _cell(table: DataTable, name: str, column: str) -> str:
    return str(table.get_cell(name, column))


def _text(widget: Static) -> str:
    return str(widget.content)


def test_dashboard_lists_sessions_clients_and_per_session_tokens() -> None:
    async def go() -> None:
        host = _host()
        listener = object()
        app = ServerDashboard(
            host,
            web=listener,
            urls=["http://127.0.0.1:8765/", "http://192.168.1.42:8765/"],
            token="tok-abc123",
        )
        async with app.run_test(size=(110, 32)) as pilot:
            await pilot.pause()
            # The listener is handed to the host, not just drawn: without
            # this the dashboard would advertise a port nothing binds.
            assert host.web is listener
            table = app.query_one(DataTable)
            # Parent above its subagent; newest activity first, then saved.
            assert [key.value for key in table.rows] == [
                "build-api", "build-api/agent-1", "scratch-notes",
            ]
            assert _cell(table, "build-api", "in") == "12.3k"
            assert _cell(table, "build-api", "out") == "4.6k"
            assert _cell(table, "build-api/agent-1", "in") == "3.1k"
            assert _cell(table, "build-api", "clients") == "2"
            assert _cell(table, "build-api", "state") == "running"
            # A hosted-but-inactive session reads idle, never "saved".
            assert _cell(table, "scratch-notes", "state") == "idle"
            # A subagent is shown under its parent, not by its full name.
            assert _cell(table, "build-api/agent-1", "session") == "↳ agent-1"

            totals = _text(app.query_one("#head", Static))
            assert "sessions 3" in totals
            assert "clients 2" in totals
            assert "↑ 15.6k in" in totals
            assert "↓ 5.5k out" in totals

            # Both URLs, the loopback one first, and the token in full.
            assert "http://127.0.0.1:8765/" in totals
            assert "http://192.168.1.42:8765/" in totals
            assert "tok-abc123" in totals
            assert "host.sock" in totals
        # The dashboard only leaves when the host does.
        assert not host.shutdown_requested

    asyncio.run(go())


def test_no_token_means_the_loopback_note() -> None:
    async def go() -> None:
        app = ServerDashboard(
            _host(), urls=["http://127.0.0.1:8765/"], token=None,
        )
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            targets = _text(app.query_one("#head", Static))
            assert "loopback only" in targets
            assert "--host 0.0.0.0" in targets
            app.exit()

    asyncio.run(go())


def test_the_token_can_be_selected_and_survives_a_repaint() -> None:
    """ctrl+c copies a selection before it stops the host — and the head is
    redrawn every half second, so a selection in it must survive that."""

    async def go() -> None:
        app = ServerDashboard(
            _host(), urls=["http://127.0.0.1:8765/"], token="tok-abc123",
        )
        async with app.run_test(size=(100, 20)) as pilot:
            await pilot.pause()
            head = app.query_one("#head", Static)
            lines = str(head.content).split("\n")
            row = next(i for i, line in enumerate(lines) if "token" in line)
            start = lines[row].index("tok-abc123")
            app.screen.selections = {
                head: Selection(Offset(start, row), Offset(start + 10, row)),
            }
            app._paint()
            await pilot.pause()
            assert app.screen.get_selected_text() == "tok-abc123"
            app.exit()

    asyncio.run(go())


def test_stop_button_asks_then_stops_the_host() -> None:
    async def go() -> None:
        host = _host()
        app = ServerDashboard(host)
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            await pilot.click("#stop-host")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            assert not host.shutdown_requested
            await pilot.press("y")
            await pilot.pause()
        assert host.shutdown_requested

    asyncio.run(go())


def test_escape_keeps_the_host_running() -> None:
    async def go() -> None:
        host = _host()
        app = ServerDashboard(host)
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            await pilot.press("q")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmScreen)
            assert app.screen.__class__ is Screen
            assert not host.shutdown_requested
            app.exit()

    asyncio.run(go())


def test_ctrl_c_is_the_same_stop_path() -> None:
    async def go() -> None:
        host = _host()
        app = ServerDashboard(host)
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            await pilot.pause()
        assert host.shutdown_requested

    asyncio.run(go())


def test_empty_host_shows_the_hint_instead_of_table_headers() -> None:
    async def go() -> None:
        app = ServerDashboard(FakeHost([]))
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            assert app.query_one("#empty", Static).display
            assert not app.query_one(DataTable).display
            assert "no sessions are hosted" in _text(
                app.query_one("#empty", Static)
            )
            app.exit()

    asyncio.run(go())


def test_live_updates_repaint_cells_and_add_new_sessions() -> None:
    async def go() -> None:
        host = _host()
        app = ServerDashboard(host)
        async with app.run_test(size=(110, 30)) as pilot:
            await pilot.pause()
            table = app.query_one(DataTable)
            # A turn spends tokens: the same row is updated in place.
            host.sessions["build-api"].usage_total = Usage(
                input_tokens=99_000, output_tokens=1,
            )
            app._paint()
            assert _cell(table, "build-api", "in") == "99.0k"
            # A new session shows up without restarting the dashboard.
            host._summaries.append(_summary("late-session"))
            host.sessions["late-session"] = SimpleNamespace(
                usage_total=Usage(),
            )
            app._paint()
            assert "late-session" in [key.value for key in table.rows]
            app.exit()

    asyncio.run(go())


def test_narrow_terminals_drop_optional_columns() -> None:
    async def go() -> None:
        app = ServerDashboard(_host())
        async with app.run_test(size=(60, 20)) as pilot:
            await pilot.pause()
            table = app.query_one(DataTable)
            assert [column.label.plain for column in table.columns.values()] == [
                "session", "state", "clients", "in", "out",
            ]
            app.exit()

    asyncio.run(go())


def test_columns_never_overflow_the_table_at_any_width() -> None:
    """A column that does not fit is clipped by the one after it (the header
    ran `ag` into `age`, and `18glm-flash` ran together), so the visible set
    must add up to the table's content width — cell padding and the sheet's
    `padding: 0 1` included."""

    async def go() -> None:
        for width in (26, 40, 52, 60, 72, 88, 100, 120, 160, 240):
            app = ServerDashboard(_host())
            async with app.run_test(size=(width, 20)) as pilot:
                await pilot.pause()
                table = app.query_one(DataTable)
                assert table.virtual_size.width <= table.content_region.width, (
                    width, [c.label.plain for c in table.columns.values()]
                )
                app.exit()

    asyncio.run(go())
