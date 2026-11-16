"""The ``/tools`` manager: what the model has, and the MCP sources.

Top half: the effective tool set (built-ins and enabled MCP tools), one
block per tool. Bottom half: every configured MCP server with its
availability and its tools. Clicking a server header or a tool row toggles
it; configuring a new server is one line in the input at the bottom.

The screen is passive: it posts a :class:`Change` carrying the command, and
the app owns the host round-trip. Fresh info arrives as :meth:`update_info`,
exactly like ``RefsScreen``.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Input, Label, OptionList, Static
from textual.widgets.option_list import Option
from rich.text import Text

from .. import theme
from ..mcp import parse_server_spec
from ..protocol import (
    AddMcpServer,
    Command,
    McpServerInfo,
    SetMcpServerScope,
    SetMcpToolEnabled,
    ToolInfo,
    ToolsInfo,
)
from .tool_view import _shorten, mcp_parts

# status -> (dot, palette color, word). Unknown states fall back to muted.
SERVER_LOOK = {
    "connected": ("●", "success", "available"),
    "disabled": ("○", "muted", "off"),
    "starting": ("◌", "warning", "starting"),
    "unavailable": ("✕", "error", "unavailable"),
}
# A server's three states, in click order. "on (global)" is the config
# default; "on (workspace)" starts it here only.
SERVER_SCOPE_NEXT = {"off": "global", "global": "workspace", "workspace": "off"}
SERVER_SCOPE_WORD = {
    "off": "off", "global": "on (global)", "workspace": "on (workspace)",
}


def _tool_block(tool: ToolInfo, *, show_source: bool) -> Vertical:
    """One tool in the effective-tools section: name, then a description
    block indented eight cells and wrapped in full (no ellipsis)."""
    palette = theme.PALETTE
    parts = mcp_parts(tool.name)
    heading = Text(parts[1] if parts else tool.name, style=palette["text"])
    if show_source:
        # The source label names the server already; the tool half suffices.
        builtin = tool.source == "builtin"
        heading.append(
            "  " + ("built-in" if builtin else f"mcp · {tool.source}"),
            style=palette["muted" if builtin else "accent"],
        )
    children = [Static(heading, classes="tools-tool-name", markup=False)]
    if tool.description.strip():
        children.append(Static(
            tool.description.strip(),
            classes="tools-tool-description",
            markup=False,
        ))
    return Vertical(*children, classes="tools-tool")


def _server_header(server: McpServerInfo) -> Text:
    palette = theme.PALETTE
    dot, color, word = SERVER_LOOK.get(server.status, ("○", "muted", server.status))
    if server.status == "connected":
        # The dot says "running"; the word says where the choice lives.
        word = SERVER_SCOPE_WORD[server.scope]
    row = Text()
    row.append(dot + " ", style=palette[color])
    row.append(server.name, style=palette["text"])
    row.append("  " + word, style=(
        palette["accent"] if server.scope == "workspace" else palette[color]
    ))
    if server.available:
        count = len(server.tools)
        row.append(f"  {count} tool" + ("s" if count != 1 else ""),
                   style=palette["muted"])
    row.append("  " + _shorten(server.target, 38), style=palette["muted"])
    return row


def _server_note(server: McpServerInfo) -> Text | None:
    """The dimmed line under a server header: why it is not usable, or how
    to make it usable."""
    if server.error:
        return Text("  " + _shorten(server.error, 74), style=theme.PALETTE["error"])
    if server.status in ("disabled", "unavailable"):
        verb = "enable" if server.status == "disabled" else "retry"
        return Text(f"  click the server name to {verb}",
                    style=theme.PALETTE["muted"])
    return None


def _server_tool(tool: ToolInfo) -> Text:
    palette = theme.PALETTE
    style = palette["text"] if tool.enabled else palette["muted"]
    row = Text("  ")
    row.append("[x] " if tool.enabled else "[ ] ",
               style=palette["success"] if tool.enabled else palette["muted"])
    row.append(_shorten(tool.name, 24).ljust(24), style=style)
    row.append(_shorten(tool.description, 44), style=palette["muted"])
    return row


class ToolsScreen(ModalScreen[None]):
    """Tool overview + MCP server manager."""

    BINDINGS = [("escape", "dismiss_none", "Done")]

    class Change(Message):
        """A command for the app to send (a toggle, or an added server)."""

        def __init__(self, command: Command) -> None:
            self.command = command
            super().__init__()

    def __init__(self, info: ToolsInfo) -> None:
        super().__init__()
        self._info = info
        # option id -> the row it acts on (server, or server and one tool).
        self._rows: dict[str, tuple[McpServerInfo, ToolInfo | None]] = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="tools-modal", classes="modal-panel"):
            yield Label("tools", id="tools-title", classes="modal-title")
            yield Label("available tools", classes="tools-section")
            yield VerticalScroll(id="tools-list")
            yield Static("─" * 72, id="tools-rule")
            yield Label("mcp servers", classes="tools-section")
            yield OptionList(id="mcp-list")
            yield Static(
                "click a server: off → on (global) → on (workspace) "
                "· tools toggle one by one · esc to close",
                id="tools-hint",
            )
            yield Static("", id="tools-error")
            yield Input(
                placeholder="add mcp server", id="tools-add",
                classes="modal-input",
            )
            yield Static(
                "name: command args…  ·  or a URL  ·  JSON for env/cwd/timeout",
                id="tools-add-hint",
            )

    async def on_mount(self) -> None:
        await self._render_rows()
        self.query_one("#tools-add", Input).focus()

    # ------------------------------------------------------------------
    # data

    async def _render_rows(self) -> None:
        info = self._info
        self.query_one("#tools-title", Label).update(
            f"tools · {len(info.tools)} available"
        )
        top = self.query_one("#tools-list", VerticalScroll)
        bottom = self.query_one("#mcp-list", OptionList)
        await top.remove_children()
        if info.tools:
            # The per-tool source tag only earns its space when more than one
            # source contributes (otherwise it repeats the section title).
            served = {tool.source for tool in info.tools} - {"builtin"}
            top.mount_all(
                _tool_block(tool, show_source=bool(served))
                for tool in info.tools
            )
        else:
            top.mount(Static(
                "no tools yet — send a message to start the session",
                classes="tools-empty", markup=False,
            ))

        self._rows = {}
        bottom.clear_options()
        if not info.servers:
            bottom.add_option(Option(
                Text(
                    "no MCP servers configured — add one below",
                    style=theme.PALETTE["muted"],
                ),
                id="mcp-empty", disabled=True,
            ))
            return
        for server in info.servers:
            server_id = f"server:{server.name}"
            self._rows[server_id] = (server, None)
            bottom.add_option(Option(_server_header(server), id=server_id))
            if (note := _server_note(server)) is not None:
                bottom.add_option(Option(
                    note, id=f"note:{server.name}", disabled=True,
                ))
            for tool in server.tools:
                row_id = f"tool:{server.name}:{tool.name}"
                self._rows[row_id] = (server, tool)
                bottom.add_option(Option(
                    _server_tool(tool),
                    id=row_id,
                    disabled=server.status != "connected",
                ))

    def update_info(self, info: ToolsInfo) -> None:
        self._info = info
        if self.is_mounted:
            self.run_worker(
                self._render_rows(), group="tools-render", exclusive=True,
            )

    def show(self, text: str, *, error: bool = False) -> None:
        """One status/error line under the lists (empty text clears it)."""
        if self.is_mounted:
            self.query_one("#tools-error", Static).update(Text(
                text, style=theme.PALETTE["error" if error else "muted"],
            ))

    # ------------------------------------------------------------------
    # interaction

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected,
    ) -> None:
        row = self._rows.get(str(event.option.id or ""))
        if row is None:
            return
        event.stop()
        server, tool = row
        if tool is not None:
            self.post_message(self.Change(SetMcpToolEnabled(
                server=server.name, tool=tool.name, enabled=not tool.enabled,
            )))
            return
        # A failed-but-on server retries on click; otherwise the click moves
        # it to the next scope.
        scope = (
            server.scope if server.status == "unavailable"
            else SERVER_SCOPE_NEXT[server.scope]
        )
        self.post_message(self.Change(SetMcpServerScope(
            name=server.name, scope=scope,
        )))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        try:
            name, settings = parse_server_spec(text)
        except ValueError as exc:
            self.show(str(exc), error=True)
            return
        event.input.value = ""
        self.show(f"starting {name}…")
        # The parsed settings are already the config's model; the command
        # takes the same fields (disabled_tools/enabled are the host's).
        self.post_message(self.Change(AddMcpServer(
            name=name,
            **settings.model_dump(exclude={"enabled", "disabled_tools"}),
        )))
        self.query_one("#tools-add", Input).focus()

    def action_dismiss_none(self) -> None:
        self.dismiss(None)
