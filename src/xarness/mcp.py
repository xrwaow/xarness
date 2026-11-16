"""Model Context Protocol servers, exposed as ordinary tools.

A server is either a long-lived child process speaking newline-delimited
JSON-RPC over stdio, or a remote Streamable HTTP endpoint. This module owns
the connection: the initialize handshake, ``tools/list`` (pagination
included), ``tools/call``, and the lifecycle of start/stop/restart. It
deliberately has no dependency on the MCP SDK: the wire protocol is small,
and keeping it here means the project's dependency set does not grow for one
feature.

Servers run on the host, outside the bubblewrap sandbox. They are trusted
extensions, not sandboxed tools: the UI labels them as such, and their file
writes are not tracked by ``/diff`` or undone by ``/undo``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import shlex
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from .config import McpServerSettings, McpSettings
from .conversation import ToolResult
from .protocol.views import McpScope, McpServerInfo, ToolInfo

LOG = logging.getLogger(__name__)

# Newer servers may return a larger initialize/tools payload than the
# asyncio StreamReader default (64 KiB) allows.
MAX_LINE_BYTES = 16 * 1024 * 1024
# A first npx/uvx run may install a package; still bound how long a broken
# command can hold session creation.
START_TIMEOUT = 60.0
# Protocol version we advertise over stdio. Servers negotiate down; we only
# need the basic tools capability, and stdio is version-agnostic.
PROTOCOL_VERSION = "2024-11-05"
# Streamable HTTP does not exist before 2025-03-26, so the HTTP client has to
# offer a modern revision; the server answers with the one it will speak and
# we echo that on every later request.
HTTP_PROTOCOL_VERSION = "2025-06-18"
# How much stderr is kept for the UI's failure detail.
STDERR_TAIL = 4096


class McpError(Exception):
    """A server failed to start, disappeared, or returned an RPC error.

    ``rpc_code``/``rpc_data`` carry a JSON-RPC error's ``code`` and ``data``
    when the failure came from one — they are how a protocol-version or
    method-not-found error is told apart from a plain failure."""

    def __init__(
        self, message: str, *,
        rpc_code: int | None = None, rpc_data: Any = None,
    ) -> None:
        super().__init__(message)
        self.rpc_code = rpc_code
        self.rpc_data = rpc_data


class _SessionExpired(McpError):
    """The server answered 404 to a request carrying our session id.

    Per the Streamable HTTP spec the session is gone (the server may end it
    at any time) and the client MUST start a new one by re-initializing."""


def exposed_tool_name(server: str, tool: str) -> str:
    """A provider-safe function name for one MCP tool.

    OpenAI-compatible APIs cap names at 64 characters of
    ``[A-Za-z0-9_-]``. The ``mcp__<server>__<tool>`` prefix keeps MCP tools
    from colliding with built-ins, and a digest suffixes an over-long name
    instead of silently truncating it into another tool's name."""
    clean = f"mcp__{_safe(server)}__{_safe(tool)}"
    if len(clean) <= 64:
        return clean
    digest = hashlib.sha1(
        f"{server}\x00{tool}".encode("utf-8")
    ).hexdigest()[:8]
    return f"{clean[:55]}_{digest}"


def _safe(value: str) -> str:
    return _slug(value, "tool", bad="_")


def _slug(value: str, fallback: str, bad: str = "-") -> str:
    """``value`` reduced to ``[A-Za-z0-9_-]``, or ``fallback`` if empty.

    Used for every derived name (tool function names, server names from a
    command or a URL) so they all obey the same rule."""
    cleaned = "".join(
        char if char.isalnum() or char in "_-" else bad for char in value
    ).strip("_-")
    return cleaned or fallback


def parse_server_spec(spec: str) -> tuple[str, McpServerSettings]:
    """Parse the one-line "add MCP server" input.

    Accepted forms::

        filesystem: npx -y @modelcontextprotocol/server-filesystem .
        filesystem = npx -y ...
        npx -y @modelcontextprotocol/server-filesystem .
        https://mcp.exa.ai/mcp
        exa: https://mcp.exa.ai/mcp

    plus a JSON object (``{"name": ..., "command": ...}`` or
    ``{"name": ..., "url": ...}``) for env, cwd or timeout. A bare URL
    names the server from its host (``https://mcp.exa.ai/mcp`` -> ``exa-ai``).
    ``shlex`` is used, so quoted command arguments work.

    Raises ``ValueError`` with a user-readable message.
    """
    text = spec.strip()
    if not text:
        raise ValueError("type a server command or URL first")
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc.msg}") from None
        if not isinstance(payload, dict):
            raise ValueError("JSON must be an object")
        name = str(payload.pop("name", "") or "").strip()
        if not name:
            raise ValueError("JSON needs a \"name\" field")
        try:
            return name, McpServerSettings.model_validate(payload)
        except Exception as exc:
            raise ValueError(str(exc)) from None

    if text.startswith(("http://", "https://")):
        return _url_name(text), McpServerSettings(url=text)

    name = ""
    rest = text
    # Only a single-word prefix counts as a name: ``npx -y https://...`` must
    # not split at the URL's colon.
    match = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9_-]*)\s*[:=]\s*(.+)$", text, re.S)
    if match:
        name, rest = match.group(1), match.group(2).strip()
    elif text.rstrip().endswith((":", "=")):
        raise ValueError("a name needs a command or URL after ':'")
    if rest.startswith(("http://", "https://")):
        return name or _url_name(rest), McpServerSettings(url=rest)
    try:
        parts = shlex.split(rest)
    except ValueError as exc:
        raise ValueError(f"could not parse command: {exc}") from None
    if not parts:
        raise ValueError("type a command to run")
    if not name:
        name = _slug(Path(parts[0]).name or parts[0], "mcp")
    try:
        return name, McpServerSettings(command=parts[0], args=parts[1:])
    except Exception as exc:
        raise ValueError(str(exc)) from None


def _url_name(url: str) -> str:
    """A readable, config-safe name for a bare MCP URL."""
    host = urlparse(url).hostname or "mcp"
    for prefix in ("mcp.", "www."):
        host = host.removeprefix(prefix)
    return _slug(host, "mcp")


@dataclass(frozen=True, slots=True)
class McpToolMeta:
    """One tool advertised by a connected server."""

    server: str
    name: str
    description: str
    input_schema: dict[str, Any]
    exposed_name: str


@dataclass
class _Server:
    settings: McpServerSettings
    client: "_Client | None" = None
    tools: list[McpToolMeta] = field(default_factory=list)
    error: str | None = None
    status: str = "disabled"


class McpManager:
    """One session's set of MCP servers.

    A session owns the manager (a subagent shares its parent's) so toggling a
    server in ``/tools`` changes exactly that conversation's tool set.
    """

    def __init__(
        self,
        settings: McpSettings | None = None,
        workspace: Path | None = None,
        workspace_servers: Iterable[str] | None = None,
    ) -> None:
        self.settings = settings or McpSettings()
        self.workspace = workspace or Path.cwd()
        # Servers switched on for this workspace only ("on (workspace)"):
        # their global setting stays off, so no other workspace gets them.
        self.workspace_servers = set(workspace_servers or ())
        self._servers: dict[str, _Server] = {
            name: _Server(settings=server, status=(
                "starting" if server.enabled or name in self.workspace_servers
                else "disabled"
            ))
            for name, server in self.settings.servers.items()
        }
        self._started = False
        self._closed = False
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # state

    @property
    def started(self) -> bool:
        return self._started

    def statuses(self) -> list[McpServerInfo]:
        """Every configured server (enabled or not) as the UI reads it.

        A server whose process died between calls is downgraded here, before
        a tool call has to discover the same thing the hard way."""
        out: list[McpServerInfo] = []
        for name, server in self._servers.items():
            if (
                server.status == "connected"
                and server.client is not None
                and not server.client.alive
            ):
                server.status = "unavailable"
                server.error = server.error or server.client.exit_detail()
            out.append(McpServerInfo(
                name=name,
                command=server.settings.command or "",
                args=list(server.settings.args),
                url=server.settings.url or "",
                enabled=self._enabled(name),
                scope=self.server_scope(name),
                status=server.status,  # type: ignore[arg-type]
                error=server.error,
                tools=[
                    ToolInfo(
                        name=tool.name,
                        source=name,
                        description=tool.description,
                        enabled=tool.name not in server.settings.disabled_tools,
                    )
                    for tool in server.tools
                ],
            ))
        return out

    def connected_server_names(self) -> list[str]:
        """Servers whose tools the model currently has (for the prompt)."""
        return [
            name for name, server in self._servers.items()
            if self._enabled(name) and server.status == "connected"
        ]

    def tool_specs(self) -> list[McpToolMeta]:
        """Tools currently offered to the model, in configured order."""
        return [
            tool
            for name, server in self._servers.items()
            if self._enabled(name) and server.status == "connected"
            for tool in server.tools
            if tool.name not in server.settings.disabled_tools
        ]

    def _enabled(self, name: str) -> bool:
        """On here: globally on, or kept on only for this workspace."""
        return (
            self._servers[name].settings.enabled
            or name in self.workspace_servers
        )

    def server_scope(self, name: str) -> McpScope:
        """Where this server's on/off choice lives, as /tools shows it."""
        if self._require(name).settings.enabled:
            return "global"
        return "workspace" if name in self.workspace_servers else "off"

    # ------------------------------------------------------------------
    # lifecycle

    async def start(self) -> None:
        """Connect every enabled server once. Idempotent."""
        async with self._lock:
            if self._started or self._closed:
                return
            self._started = True
            await asyncio.gather(
                *(
                    self._start_one(name, server)
                    for name, server in self._servers.items()
                    if self._enabled(name)
                ),
                return_exceptions=True,
            )

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            clients = [
                server.client for server in self._servers.values()
                if server.client is not None
            ]
            for server in self._servers.values():
                server.client = None
                server.status = "disabled"
            await asyncio.gather(
                *(client.close() for client in clients),
                return_exceptions=True,
            )

    async def _start_one(self, name: str, server: _Server) -> None:
        if server.client is not None:
            await server.client.close()
            server.client = None
        server.error = None
        server.status = "starting"
        try:
            if server.settings.url:
                client: _StdioClient | _HttpClient = _HttpClient(
                    name, server.settings,
                )
            else:
                client = _StdioClient(
                    name, server.settings, workspace=self.workspace,
                )
            await client.start()
            tools = await client.list_tools()
        except Exception as exc:  # a broken server must not kill a session
            if not isinstance(exc, (McpError, OSError, ValueError)):
                LOG.debug("mcp server %s failed", name, exc_info=True)
                exc = f"{type(exc).__name__}: {exc}"
            server.status = "unavailable"
            server.error = _one_line(str(exc))
            server.tools = []
            return
        server.client = client
        used: set[str] = set()
        tools_meta: list[McpToolMeta] = []
        for tool in tools:
            if not isinstance(tool, dict) or not tool.get("name"):
                continue
            tool_name = str(tool["name"])
            exposed = exposed_tool_name(name, tool_name)
            if exposed in used:
                # Two original names sanitized to the same function name
                # (``read.me`` / ``read_me``): keep both callable.
                digest = hashlib.sha1(tool_name.encode("utf-8")).hexdigest()[:6]
                exposed = f"{exposed[:57]}_{digest}"
            used.add(exposed)
            tools_meta.append(McpToolMeta(
                server=name,
                name=tool_name,
                description=str(tool.get("description", "") or ""),
                input_schema=(
                    tool.get("inputSchema")
                    if isinstance(tool.get("inputSchema"), dict) else {}
                ),
                exposed_name=exposed,
            ))
        server.tools = tools_meta
        server.status = "connected"
        server.error = None

    async def set_server_enabled(self, name: str, enabled: bool) -> None:
        """Old binary toggle: global on, or off everywhere."""
        await self.set_server_scope(name, "global" if enabled else "off")

    async def set_server_scope(self, name: str, scope: McpScope) -> None:
        """Move one server's choice: off, global on, or on only here.

        "workspace" leaves the global setting off (so other workspaces do
        not start it) and records the override for this one. A server that
        stays on through the move keeps running: only on/off starts or
        stops a process."""
        if scope not in ("off", "global", "workspace"):
            raise McpError(f"unknown server scope {scope!r}")
        server = self._require(name)
        was_on = self._enabled(name)
        if scope == "workspace":
            self.workspace_servers.add(name)
            server.settings.enabled = False
        else:
            self.workspace_servers.discard(name)
            server.settings.enabled = scope == "global"
        if was_on != (scope != "off") or (
            scope != "off" and server.status == "unavailable"
        ):
            await self._set_running(name, scope != "off")

    async def _set_running(self, name: str, running: bool) -> None:
        """Start or stop the process without touching the saved choice."""
        server = self._require(name)
        async with self._lock:
            if not running:
                client, server.client = server.client, None
                server.status = "disabled"
                server.error = None
                if client is not None:
                    await client.close()
                return
            server.status = "starting"
            await self._start_one(name, server)

    async def set_tool_enabled(
        self, server_name: str, tool_name: str, enabled: bool,
    ) -> None:
        server = self._require(server_name)
        disabled = [item for item in server.settings.disabled_tools
                    if item != tool_name]
        if not enabled:
            disabled.append(tool_name)
        server.settings.disabled_tools = disabled

    async def add_server(
        self, name: str, settings: McpServerSettings,
    ) -> None:
        async with self._lock:
            if name in self._servers:
                raise McpError(f"an MCP server named {name!r} already exists")
            self.settings.servers[name] = settings
            server = _Server(
                settings=settings,
                status="starting" if settings.enabled else "disabled",
            )
            self._servers[name] = server
            if self._started and not self._closed and settings.enabled:
                await self._start_one(name, server)

    def _require(self, name: str) -> _Server:
        server = self._servers.get(name)
        if server is None:
            raise McpError(f"no MCP server named {name!r}")
        return server

    # ------------------------------------------------------------------
    # calls

    async def call(
        self, server_name: str, tool_name: str, arguments: dict[str, Any],
    ) -> ToolResult:
        server = self._require(server_name)
        if not self._enabled(server_name):
            return ToolResult(
                status="error",
                content=f"MCP server {server_name!r} is disabled",
            )
        client = server.client
        if client is None or server.status != "connected":
            detail = f": {server.error}" if server.error else ""
            return ToolResult(
                status="error",
                content=(
                    f"MCP server {server_name!r} is not available{detail}"
                ),
            )
        try:
            result = await client.call_tool(
                tool_name, arguments, timeout=server.settings.timeout,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                status="error",
                content=(
                    f"MCP tool {server_name}/{tool_name} timed out after "
                    f"{server.settings.timeout:g}s"
                ),
            )
        except (McpError, OSError) as exc:
            server.status = "unavailable"
            server.error = _one_line(str(exc))
            return ToolResult(
                status="error",
                content=f"MCP tool {server_name}/{tool_name} failed: {exc}",
            )
        return _tool_result(result)


# ----------------------------------------------------------------------
# protocol client


class _Client:
    """What both transports share: the tools the manager calls on top of
    ``request``/``notify``/``close``/``alive``, which a subclass provides."""

    def __init__(self, name: str, settings: McpServerSettings) -> None:
        self.name = name
        self.settings = settings
        self._next_id = 0
        self._closing = False

    @property
    def startup_timeout(self) -> float:
        """The cap on startup/handshake work: a broken server must not hold
        session creation for the full per-call budget."""
        return min(START_TIMEOUT, max(1.0, self.settings.timeout))

    async def request(
        self, method: str, params: dict[str, Any], *, timeout: float,
    ) -> Any:
        raise NotImplementedError

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError

    @property
    def alive(self) -> bool:
        raise NotImplementedError

    def exit_detail(self) -> str:
        """A one-line reason for the UI (the manager reports it)."""
        raise NotImplementedError

    async def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            result = await self.request(
                "tools/list", {"cursor": cursor} if cursor else {},
                timeout=self.startup_timeout,
            )
            if not isinstance(result, dict):
                raise McpError("tools/list returned an invalid payload")
            page = result.get("tools")
            if isinstance(page, list):
                tools.extend(item for item in page if isinstance(item, dict))
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    async def call_tool(
        self, name: str, arguments: dict[str, Any], *, timeout: float,
    ) -> dict[str, Any]:
        result = await self.request(
            "tools/call",
            {"name": name, "arguments": arguments},
            timeout=timeout,
        )
        if not isinstance(result, dict):
            raise McpError("tools/call returned an invalid payload")
        return result


class _StdioClient(_Client):
    """One MCP server process and its JSON-RPC channel."""

    def __init__(
        self, name: str, settings: McpServerSettings, *, workspace: Path,
    ) -> None:
        super().__init__(name, settings)
        self.workspace = workspace
        self._process: asyncio.subprocess.Process | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail = ""

    @property
    def alive(self) -> bool:
        return (
            self._process is not None
            and self._process.returncode is None
            and not self._closing
        )

    async def start(self) -> None:
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in self.settings.env.items()})
        # A relative cwd is relative to the workspace; unset means the
        # workspace itself.
        cwd = Path(self.settings.cwd or ".").expanduser()
        try:
            self._process = await asyncio.create_subprocess_exec(
                self.settings.command,
                *self.settings.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd if cwd.is_absolute() else self.workspace / cwd),
                env=env,
                limit=MAX_LINE_BYTES,
            )
        except OSError as exc:
            detail = (
                "command not found" if isinstance(exc, FileNotFoundError)
                else exc
            )
            raise McpError(
                f"could not start {self.settings.command!r}: {detail}"
            ) from None
        self._reader_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())
        try:
            await self.request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "xarness", "version": "0.1.0"},
                },
                timeout=self.startup_timeout,
            )
            await self.notify("notifications/initialized", {})
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        process = self._process
        for task in (self._reader_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                with contextlib.suppress(Exception):
                    await process.wait()
        self._fail_all("MCP server closed")
        self._process = None

    def _fail_all(self, message: str) -> None:
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(McpError(message))
        self._pending.clear()

    async def _read_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        stream = self._process.stdout
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace").strip()
                if not text:
                    continue
                try:
                    message = json.loads(text)
                except json.JSONDecodeError:
                    LOG.debug("mcp %s: non-JSON stdout: %s", self.name, text[:200])
                    continue
                if not isinstance(message, dict):
                    continue
                message_id = message.get("id")
                if message_id is None:
                    # Server-initiated notification; nothing needs it today.
                    continue
                future = self._pending.pop(int(message_id), None)
                if future is None or future.done():
                    continue
                if "error" in message:
                    error = message.get("error") or {}
                    future.set_exception(McpError(
                        f"{error.get('message') or error or 'RPC error'}"
                    ))
                else:
                    future.set_result(message.get("result"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOG.debug("mcp %s: stdout reader stopped", self.name, exc_info=True)
            self._fail_all(f"connection lost: {exc}")
        else:
            self._fail_all(self.exit_detail())

    async def _read_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        stream = self._process.stderr
        try:
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    return
                self._stderr_tail = (
                    self._stderr_tail + chunk.decode("utf-8", "replace")
                )[-STDERR_TAIL:]
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    def exit_detail(self) -> str:
        """A one-line reason the server is gone (the manager reports it)."""
        tail = " ".join(self._stderr_tail.split())
        if tail:
            return _one_line(tail)
        code = self._process.returncode if self._process is not None else None
        if code is not None:
            return f"server exited with status {code}"
        return "server closed its output"

    async def _write(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.returncode is not None:
            raise McpError(self.exit_detail())
        data = json.dumps(payload, ensure_ascii=False) + "\n"
        process.stdin.write(data.encode("utf-8"))
        await process.stdin.drain()

    async def request(
        self, method: str, params: dict[str, Any], *, timeout: float,
    ) -> Any:
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._write({
                "jsonrpc": "2.0", "id": request_id,
                "method": method, "params": params,
            })
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            # A timeout or a cancelled tool call must not leave the future
            # in the table forever; a late response is simply dropped.
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})


class _HttpClient(_Client):
    """One remote MCP server speaking Streamable HTTP.

    Each JSON-RPC message is a POST to the endpoint. A server may answer with
    ``application/json`` or with an SSE stream carrying the same message; both
    are accepted. The session id the server hands out on ``initialize`` is
    echoed on every later request (some servers require it), together with the
    negotiated protocol version."""

    def __init__(
        self, name: str, settings: McpServerSettings,
        *, transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(name, settings)
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._session_id: str | None = None
        self._protocol_version = PROTOCOL_VERSION
        self._last_error = "HTTP endpoint closed"

    @property
    def alive(self) -> bool:
        # HTTP has no connection to watch. Once a request fails the manager
        # marks the server unavailable, and a click retries it.
        return self._client is not None and not self._closing

    async def start(self) -> None:
        if not self.settings.url:
            raise McpError("no URL configured")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=self.startup_timeout,
                read=max(1.0, self.settings.timeout),
                write=30.0,
                pool=15.0,
            ),
            follow_redirects=True,
            transport=self._transport,
        )
        try:
            await self._handshake()
        except Exception:
            await self.close()
            raise

    async def _handshake(self) -> None:
        """initialize + notifications/initialized, minting a new session."""
        self._session_id = None
        try:
            result = await self._request_once(
                "initialize",
                {
                    "protocolVersion": HTTP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "xarness", "version": "0.1.0"},
                },
                timeout=self.startup_timeout,
            )
        except McpError as exc:
            era = _era_problem(exc)
            if era is not None:
                raise McpError(f"{exc} — {era}") from None
            # The old HTTP+SSE transport (2024-11-05) answers the POST with
            # 400/404/405 and expects a GET to /sse instead; say so rather
            # than leaving a bare status code.
            text = str(exc)
            if any(f"HTTP {code}" in text for code in (400, 404, 405)):
                raise McpError(
                    f"{text} — not a Streamable HTTP endpoint (the legacy "
                    "HTTP+SSE transport is not supported)"
                ) from None
            raise
        # The server picks the version; every later request must announce the
        # negotiated one, not the one we offered.
        if isinstance(result, dict) and result.get("protocolVersion"):
            self._protocol_version = str(result["protocolVersion"])
        with contextlib.suppress(_SessionExpired, McpError):
            # A stateless server may 400/404 a notification it did not ask
            # for; the handshake itself already succeeded, so carry on.
            await self.notify("notifications/initialized", {})

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        client, self._client = self._client, None
        session_id, self._session_id = self._session_id, None
        if client is not None:
            if session_id:
                # Explicitly end the session (spec: SHOULD); a server that
                # does not allow it answers 405, which is fine.
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(
                        client.delete(
                            self.settings.url or "",
                            headers={
                                "Mcp-Session-Id": session_id,
                                "MCP-Protocol-Version": self._protocol_version,
                            },
                        ),
                        timeout=5.0,
                    )
            with contextlib.suppress(Exception):
                await client.aclose()

    def exit_detail(self) -> str:
        """A one-line reason for the UI (public: the manager reports it)."""
        return _one_line(self._last_error)

    async def _post(
        self,
        payload: dict[str, Any],
        *,
        timeout: float,
    ) -> list[dict[str, Any]]:
        client = self._client
        if client is None or self._closing:
            raise McpError(self._last_error)
        # Whether *this* request carried a session id: a 404 means the
        # session is gone only if we sent one.
        sent_session = self._session_id
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
            headers["MCP-Protocol-Version"] = self._protocol_version
        try:
            response = await asyncio.wait_for(
                client.post(
                    self.settings.url or "", json=payload, headers=headers,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            self._last_error = f"timed out after {timeout:g}s"
            raise
        except httpx.HTTPError as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise McpError(self._last_error) from None
        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self._session_id = session_id
        if response.status_code == 404 and sent_session:
            # The server ended the session; the caller re-initializes.
            self._last_error = "session expired (HTTP 404)"
            raise _SessionExpired(self._last_error)
        if response.status_code >= 400:
            body = _one_line(response.text)[:240]
            self._last_error = (
                f"HTTP {response.status_code}"
                + (f": {body}" if body else "")
            )
            # An error response may still be a JSON-RPC error (a modern
            # server answers `initialize` with 404 + method-not-found); keep
            # its code so the handshake can name the protocol era.
            rpc = _rpc_error(response)
            raise McpError(
                self._last_error,
                rpc_code=rpc.get("code") if rpc else None,
                rpc_data=rpc.get("data") if rpc else None,
            )
        if response.status_code == 202 or not response.content:
            return []
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return _sse_messages(response.text)
        try:
            data = response.json()
        except ValueError:
            body = _one_line(response.text)[:240]
            self._last_error = f"invalid response: {body}"
            raise McpError(self._last_error) from None
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            return [data]
        return []

    async def _request_once(
        self, method: str, params: dict[str, Any], *, timeout: float,
    ) -> Any:
        self._next_id += 1
        request_id = self._next_id
        messages = await self._post(
            {
                "jsonrpc": "2.0", "id": request_id,
                "method": method, "params": params,
            },
            timeout=timeout,
        )
        for message in messages:
            if str(message.get("id")) != str(request_id):
                continue  # a server-initiated notification or a stale reply
            error = message.get("error")
            if error:
                if isinstance(error, dict):
                    raise McpError(
                        str(error.get("message") or error),
                        rpc_code=error.get("code"),
                        rpc_data=error.get("data"),
                    )
                raise McpError(f"{error}")
            return message.get("result")
        self._last_error = f"{method}: no response"
        raise McpError(self._last_error)

    async def request(
        self, method: str, params: dict[str, Any], *, timeout: float,
    ) -> Any:
        try:
            return await self._request_once(method, params, timeout=timeout)
        except _SessionExpired:
            # The server forgot the session (restart, expiry). Start a new
            # one and replay the request once, as the spec requires.
            LOG.debug("mcp %s: session expired, re-initializing", self.name)
            await self._handshake()
            return await self._request_once(method, params, timeout=timeout)

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        # The reply to a notification is normally 202 with an empty body, but
        # a strict server may still answer with an error status; `_post`
        # turns that into an McpError, which start() surfaces.
        await self._post(
            {"jsonrpc": "2.0", "method": method, "params": params},
            timeout=self.startup_timeout,
        )


def _rpc_error(response: httpx.Response) -> dict[str, Any] | None:
    """The JSON-RPC error object in an HTTP error response, if any."""
    try:
        body = response.json()
    except Exception:
        return None
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    return error if isinstance(error, dict) else None


def _era_problem(exc: McpError) -> str | None:
    """A readable diagnosis when a server speaks the other protocol era.

    ``2026-07-28`` dropped the ``initialize`` handshake and the concept of a
    session: every request declares its own version in ``_meta``, and
    servers answer ``initialize`` with ``UnsupportedProtocolVersionError``
    (``-32022``) or ``method not found`` (``-32601``). That is a protocol
    generation gap, not a broken server, and the message should say so."""
    if exc.rpc_code == -32022:
        data = exc.rpc_data if isinstance(exc.rpc_data, dict) else {}
        supported = [str(item) for item in data.get("supported") or ()]
        offered = ", ".join(supported) or "newer revisions"
        return (
            f"the server speaks {offered}; xarness speaks the 2025-11-25 "
            "(handshake) era, and the stateless 2026-07-28 revision is not "
            "supported yet"
        )
    if exc.rpc_code == -32601:
        return (
            "the server does not implement 'initialize': it speaks the "
            "stateless 2026-07-28 revision, which is not supported yet"
        )
    return None


def _sse_messages(body: str) -> list[dict[str, Any]]:
    """The JSON-RPC messages carried by an SSE response body.

    Streamable HTTP puts each message on its own ``data:`` line (it may span
    several, which are joined with newlines); comment and event-name lines are
    ignored, and anything that is not JSON is skipped."""
    messages: list[dict[str, Any]] = []
    data: list[str] = []

    def flush() -> None:
        if not data:
            return
        payload = "\n".join(data)
        data.clear()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            LOG.debug("mcp: non-JSON SSE data: %s", payload[:200])
            return
        for item in parsed if isinstance(parsed, list) else [parsed]:
            if isinstance(item, dict):
                messages.append(item)

    for raw in body.splitlines():
        line = raw.rstrip("\r")
        if not line:
            flush()
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    flush()
    return messages


def _tool_result(result: dict[str, Any]) -> ToolResult:
    """Flatten MCP content blocks into the one text field a ToolResult has.

    The conversation model is text-only, so images/resources are described
    rather than attached; a text-heavy tool (the common case) is unchanged."""
    def block_text(block: dict[str, Any]) -> str:
        kind = block.get("type")
        if kind in ("image", "audio"):
            return f"[{kind}: {block.get('mimeType', 'unknown')}]"
        if kind == "resource":
            resource = block.get("resource")
            if not isinstance(resource, dict):
                return "[resource]"
            return str(
                resource.get("text") or f"[resource: {resource.get('uri', '')}]"
            )
        if kind == "resource_link":
            return f"[resource: {block.get('uri', '')}]"
        if kind == "text":
            return str(block.get("text", ""))
        return json.dumps(block, ensure_ascii=False)

    blocks = result.get("content")
    parts = [
        block_text(block)
        for block in (blocks if isinstance(blocks, list) else [])
        if isinstance(block, dict)
    ]
    if not parts and result.get("structuredContent") is not None:
        parts.append(json.dumps(
            result["structuredContent"], ensure_ascii=False, indent=2,
        ))
    content = "\n".join(part for part in parts if part)
    if result.get("isError"):
        return ToolResult(
            status="error", content=content or "MCP tool returned an error",
        )
    return ToolResult(status="ok", content=content)


def _one_line(text: str) -> str:
    return " ".join(text.split())[:500]
