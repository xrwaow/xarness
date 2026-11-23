"""MCP config, the stdio client, and registry integration."""

import asyncio
import functools
import json
import sys
from pathlib import Path

import httpx
import pytest

from xarness import session_store
from xarness.config import (
    McpServerSettings,
    McpSettings,
    ProviderProfile,
    SectionedConfig,
    load_sectioned,
    save_mcp_settings,
)
from xarness.mcp import (
    McpManager,
    _HttpClient,
    exposed_tool_name,
    parse_server_spec,
)
from xarness.protocol import (
    AddMcpServer,
    SetMcpServerScope,
    SetMcpToolEnabled,
)
from xarness.sandbox import SandboxRuntime
from xarness.session import Session, SessionError
from xarness.tools import ToolRegistry, build_registry, mcp_tools
from xarness.workspace import (
    load_workspace_mcp_servers,
    load_workspace_refs,
    save_workspace_refs,
)

FAKE_SERVER = r'''
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    if method is None:
        continue
    mid = msg.get("id")
    if method == "initialize":
        res = {"protocolVersion": "2024-11-05",
               "capabilities": {"tools": {}},
               "serverInfo": {"name": "fake", "version": "1"}}
    elif method == "tools/list":
        res = {"tools": [
            {"name": "echo", "description": "Echo text",
             "inputSchema": {"type": "object",
                             "properties": {"text": {"type": "string"}},
                             "required": ["text"]}},
            {"name": "boom", "description": "Fail",
             "inputSchema": {"type": "object", "properties": {}}},
            {"name": "read.me", "description": "",
             "inputSchema": {"type": "object", "properties": {}}},
        ]}
    elif method == "tools/call":
        p = msg.get("params") or {}
        if p.get("name") == "boom":
            res = {"isError": True,
                   "content": [{"type": "text", "text": "kaboom"}]}
        else:
            res = {"content": [
                {"type": "text", "text": "echo: " + str((p.get("arguments") or {}).get("text"))},
                {"type": "image", "mimeType": "image/png"},
            ]}
    else:
        continue
    if mid is not None:
        print(json.dumps({"jsonrpc": "2.0", "id": mid, "result": res}), flush=True)
'''


@pytest.fixture()
def fake_server(tmp_path: Path) -> Path:
    path = tmp_path / "fake_mcp.py"
    path.write_text(FAKE_SERVER, encoding="utf-8")
    return path


def _manager(path: Path, *, workspace: Path | None = None) -> McpManager:
    return McpManager(
        McpSettings(servers={"fake": McpServerSettings(
            command=sys.executable, args=[str(path)],
        )}),
        workspace or path.parent,
    )


def test_parse_server_spec_forms() -> None:
    name, server = parse_server_spec("fs: npx -y @scope/pkg .")
    assert name == "fs"
    assert server.command == "npx"
    assert server.args == ["-y", "@scope/pkg", "."]

    name, server = parse_server_spec("fs = npx -y pkg")
    assert name == "fs" and server.command == "npx"

    # No explicit name: the command's basename becomes the server name.
    name, server = parse_server_spec("npx -y https://example.test/x")
    assert name == "npx"
    assert server.args == ["-y", "https://example.test/x"]

    # JSON carries env/cwd/timeout.
    name, server = parse_server_spec(json.dumps({
        "name": "db", "command": "uvx", "args": ["db"],
        "env": {"TOKEN": "x"}, "cwd": "sub",
    }))
    assert name == "db" and server.env["TOKEN"] == "x" and server.cwd == "sub"


@pytest.mark.parametrize("spec", ["", "   ", "name:", "name: 'unclosed"])
def test_parse_server_spec_errors(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_server_spec(spec)


def test_config_mcp_section_and_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "models": {"a": {"base_url": "http://x/v1", "model_id": "m"}},
        "mcp": {"servers": {"fs": {
            "command": "npx", "args": ["pkg"], "env": {"A": "b"},
        }}},
    }), encoding="utf-8")
    config = load_sectioned(path)
    assert config.mcp.servers["fs"].command == "npx"
    assert config.mcp.servers["fs"].env == {"A": "b"}

    settings = McpSettings(servers={
        "fs": McpServerSettings(
            command="npx", args=["pkg"], disabled_tools=["read"],
            enabled=False,
        ),
        "other": McpServerSettings(command="uvx", args=["srv"]),
    })
    save_mcp_settings(path, settings)
    raw = json.loads(path.read_text())
    assert raw["models"]["a"]["model_id"] == "m"  # other sections untouched
    assert raw["mcp"]["servers"]["fs"]["enabled"] is False
    assert load_sectioned(path).mcp.servers["fs"].disabled_tools == ["read"]


def test_invalid_server_name_rejected() -> None:
    with pytest.raises(ValueError):
        McpSettings.model_validate({"servers": {"bad name": {
            "command": "x",
        }}})


def test_sectioned_config_has_no_mcp_by_default() -> None:
    config = SectionedConfig()
    assert config.mcp.servers == {}


def test_exposed_names_are_provider_safe() -> None:
    assert exposed_tool_name("fs", "read_file") == "mcp__fs__read_file"
    long = exposed_tool_name("server" * 10, "tool" * 20)
    assert len(long) <= 64
    assert all(char.isalnum() or char in "_-" for char in long)
    assert exposed_tool_name("fs", "read.me") == "mcp__fs__read_me"


def test_manager_starts_lists_calls_and_toggles(fake_server: Path) -> None:
    async def go() -> None:
        manager = _manager(fake_server)
        await manager.start()
        statuses = manager.statuses()
        assert statuses[0].status == "connected"
        assert statuses[0].available
        assert [tool.name for tool in statuses[0].tools] == ["echo", "boom", "read.me"]
        assert [t.exposed_name for t in manager.tool_specs()] == [
            "mcp__fake__echo", "mcp__fake__boom", "mcp__fake__read_me",
        ]

        ok = await manager.call("fake", "echo", {"text": "hi"})
        assert ok.ok and "echo: hi" in ok.content
        assert "[image: image/png]" in ok.content

        failed = await manager.call("fake", "boom", {})
        assert not failed.ok and failed.content == "kaboom"

        await manager.set_tool_enabled("fake", "echo", False)
        assert [t.name for t in manager.tool_specs()] == ["boom", "read.me"]
        await manager.set_tool_enabled("fake", "echo", True)
        assert "echo" in [t.name for t in manager.tool_specs()]

        await manager.set_server_enabled("fake", False)
        status = manager.statuses()[0]
        assert status.status == "disabled" and not status.available
        disabled_call = await manager.call("fake", "echo", {})
        assert not disabled_call.ok and "disabled" in disabled_call.content

        await manager.set_server_enabled("fake", True)
        assert manager.statuses()[0].status == "connected"
        await manager.close()
        assert manager.statuses()[0].status == "disabled"

    asyncio.run(go())


def test_manager_reports_a_missing_command(tmp_path: Path) -> None:
    async def go() -> None:
        manager = McpManager(McpSettings(servers={"gone": McpServerSettings(
            command="definitely-not-a-real-command-xyz",
        )}), tmp_path)
        await manager.start()
        status = manager.statuses()[0]
        assert status.status == "unavailable"
        assert "not found" in (status.error or "")
        assert manager.tool_specs() == []
        call = await manager.call("gone", "x", {})
        assert not call.ok and "not available" in call.content

    asyncio.run(go())


def test_server_scope_cycles_and_workspace_override_persists(
    fake_server: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """off → on (global) → on (workspace): the workspace choice keeps the
    global setting off and rides in the workspace's shared row."""
    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path / "sessions")
    save_workspace_refs(tmp_path, ["/tmp/ref"])  # same shared row
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    async def go() -> None:
        session = Session(
            name="scope-test",
            workspace=tmp_path,
            profile=ProviderProfile(
                base_url="http://127.0.0.1:1/v1", model_id="test",
            ),
            profile_name="test",
            api_key=None,
            config_path=config_path,
            sandbox=SandboxRuntime(),
            mcp=McpSettings(servers={"fake": McpServerSettings(
                command=sys.executable, args=[str(fake_server)],
            )}),
        )
        await session.prepare_mcp()

        def server() -> object:
            return session.mcp.statuses()[0]

        assert server().scope == "global" and server().enabled

        await session.execute(SetMcpServerScope(name="fake", scope="off"))
        assert server().scope == "off" and not server().enabled
        assert server().status == "disabled"

        await session.execute(SetMcpServerScope(
            name="fake", scope="workspace",
        ))
        assert server().scope == "workspace" and server().enabled
        assert server().status == "connected"
        # Global stays off for every other workspace…
        assert load_sectioned(config_path).mcp.servers["fake"].enabled is False
        # …and this workspace remembers the override, without disturbing
        # the other per-workspace columns.
        assert load_workspace_mcp_servers(tmp_path) == {"fake"}
        assert load_workspace_refs(tmp_path) == ["/tmp/ref"]

        await session.execute(SetMcpServerScope(
            name="fake", scope="global",
        ))
        assert server().scope == "global" and server().enabled
        assert load_sectioned(config_path).mcp.servers["fake"].enabled is True
        assert load_workspace_mcp_servers(tmp_path) == set()
        await session.mcp.close()

    asyncio.run(go())


def test_registry_exposes_mcp_tools_and_keeps_schemas(fake_server: Path) -> None:
    async def go() -> None:
        manager = _manager(fake_server)
        await manager.start()
        tools = mcp_tools(manager)
        assert [tool.name for tool in tools] == [
            "mcp__fake__echo", "mcp__fake__boom", "mcp__fake__read_me",
        ]
        assert all(tool.source == "fake" for tool in tools)
        echo = tools[0]
        assert echo.schema["properties"]["text"] == {"type": "string"}
        registry = ToolRegistry()
        for tool in tools:
            registry.register(tool)
        result = await registry.call(
            "mcp__fake__echo", '{"text": "through the registry"}',
        )
        assert result.ok and "through the registry" in result.content
        await manager.close()

    asyncio.run(go())


def test_build_registry_includes_mcp_only_when_injected(fake_server: Path) -> None:
    async def go() -> None:
        manager = _manager(fake_server)
        await manager.start()
        assert build_registry(None, None).tools() == []
        names = [tool.name for tool in build_registry(None, None, mcp=manager).tools()]
        assert "mcp__fake__echo" in names
        await manager.close()

    asyncio.run(go())


def test_session_tools_info_and_toggles_persist(
    fake_server: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    async def go() -> None:
        session = Session(
            name="mcp-test",
            workspace=tmp_path,
            profile=ProviderProfile(
                base_url="http://127.0.0.1:1/v1", model_id="test",
            ),
            profile_name="test",
            api_key=None,
            config_path=config_path,
            sandbox=SandboxRuntime(),
            mcp=McpSettings(servers={"fake": McpServerSettings(
                command=sys.executable, args=[str(fake_server)],
            )}),
        )
        await session.prepare_mcp()
        info = await session.tools_info()
        assert [tool.name for tool in info.tools if tool.source == "fake"] == [
            "mcp__fake__echo", "mcp__fake__boom", "mcp__fake__read_me",
        ]
        assert info.servers[0].status == "connected"
        assert info.servers[0].available

        await session.execute(SetMcpToolEnabled(
            server="fake", tool="boom", enabled=False,
        ))
        info = await session.tools_info()
        assert "mcp__fake__boom" not in [t.name for t in info.tools]
        assert any(
            t.name == "boom" and not t.enabled for t in info.servers[0].tools
        )

        await session.execute(SetMcpServerScope(name="fake", scope="off"))
        info = await session.tools_info()
        assert info.servers[0].status == "disabled"
        assert not any(tool.source == "fake" for tool in info.tools)

        await session.execute(SetMcpServerScope(name="fake", scope="global"))
        info = await session.tools_info()
        assert info.servers[0].status == "connected"

        with pytest.raises(SessionError):
            await session.execute(AddMcpServer(
                name="fake", command="npx",
            ))
        await session.execute(AddMcpServer(
            name="second", command=sys.executable, args=[str(fake_server)],
        ))
        info = await session.tools_info()
        assert [server.name for server in info.servers] == ["fake", "second"]
        saved = load_sectioned(config_path).mcp
        assert set(saved.servers) == {"fake", "second"}
        await session.mcp.close()

    asyncio.run(go())


# ----------------------------------------------------------------------
# remote (Streamable HTTP) servers


URL = "https://mcp.example.test/mcp"


def _http_transport(
    requests: list[httpx.Request],
    *,
    sse: bool = False,
    status: int | None = None,
) -> httpx.MockTransport:
    """A scripted Streamable-HTTP MCP endpoint (one tool, ``search``)."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        message = json.loads(request.content)
        if status is not None:
            return httpx.Response(status, text="server said no")
        if message.get("id") is None:  # a notification: 202, no body
            return httpx.Response(202)
        method = message.get("method")
        headers = {"mcp-session-id": "sess-1"} if method == "initialize" else {}
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-http", "version": "1"},
            }
        elif method == "tools/list":
            result = {"tools": [{
                "name": "search",
                "description": "Search the web",
                "inputSchema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            }]}
        elif method == "tools/call":
            params = message.get("params") or {}
            if params.get("name") != "search":
                return httpx.Response(200, json={
                    "jsonrpc": "2.0", "id": message["id"],
                    "error": {"code": -32602, "message": "unknown tool"},
                })
            result = {"content": [{
                "type": "text",
                "text": "hit " + str((params.get("arguments") or {}).get("q")),
            }]}
        else:
            return httpx.Response(400, text="unknown method")
        payload = {"jsonrpc": "2.0", "id": message["id"], "result": result}
        if sse:
            return httpx.Response(
                200,
                text="event: message\ndata: " + json.dumps(payload) + "\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=payload, headers=headers)

    return httpx.MockTransport(handler)


def _http(
    requests: list[httpx.Request] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    **kwargs,
) -> _HttpClient:
    """A Streamable-HTTP client wired to the scripted endpoint above."""
    return _HttpClient(
        "rem", McpServerSettings(url=URL),
        transport=transport or _http_transport(
            [] if requests is None else requests, **kwargs,
        ),
    )


def _install_http(monkeypatch: pytest.MonkeyPatch, transport) -> None:
    """Make every client the manager builds use this transport."""
    monkeypatch.setattr(
        "xarness.mcp._HttpClient",
        functools.partial(_HttpClient, transport=transport),
    )


def test_ensure_mcp_is_one_shared_background_task(
    fake_server: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attaching never waits on a handshake: ensure_mcp returns the task
    instead of awaiting it, and every caller shares the one task."""
    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    async def go() -> None:
        session = Session(
            name="mcp-test",
            workspace=tmp_path,
            profile=ProviderProfile(
                base_url="http://127.0.0.1:1/v1", model_id="test",
            ),
            profile_name="test",
            api_key=None,
            config_path=config_path,
            sandbox=SandboxRuntime(),
            mcp=McpSettings(servers={"fake": McpServerSettings(
                command=sys.executable, args=[str(fake_server)],
            )}),
        )
        started = asyncio.Event()
        start = session.mcp.start

        async def slow_start() -> None:
            started.set()
            await asyncio.sleep(0.05)
            await start()

        session.mcp.start = slow_start  # type: ignore[method-assign]
        task = session.ensure_mcp()
        assert task is not None
        assert session.ensure_mcp() is task  # one task, not one per caller
        await asyncio.wait_for(started.wait(), 1)
        # Returned, not waited for: the caller got a live task.
        assert not task.done()
        await task
        assert session.mcp.started
        await session.mcp.close()

    asyncio.run(go())


def test_a_session_without_servers_has_nothing_to_wait_for(tmp_path: Path) -> None:
    """No server configured: ensure_mcp returns None (no task to yield on,
    which matters because turn startup is synchronous), and prepare_mcp
    leaves a replaced tool registry alone."""
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    async def go() -> None:
        session = Session(
            name="no-mcp",
            workspace=tmp_path,
            profile=ProviderProfile(
                base_url="http://127.0.0.1:1/v1", model_id="test",
            ),
            profile_name="test",
            api_key=None,
            config_path=config_path,
            sandbox=SandboxRuntime(),
            mcp=McpSettings(),
        )
        marker = ToolRegistry()
        session.tool_registry = marker
        assert session.ensure_mcp() is None
        await session.prepare_mcp()
        assert session.tool_registry is marker

    asyncio.run(go())


def test_parse_server_spec_urls() -> None:
    name, server = parse_server_spec(URL)
    assert name == "example-test"  # from the host, "mcp." dropped
    assert server.url == URL and server.command is None

    name, server = parse_server_spec("exa: https://mcp.exa.ai/mcp")
    assert name == "exa" and server.url == "https://mcp.exa.ai/mcp"

    # A bare URL of a real service keeps a readable, config-safe name.
    name, server = parse_server_spec("https://mcp.exa.ai/mcp")
    assert name == "exa-ai" and server.url == "https://mcp.exa.ai/mcp"

    assert parse_server_spec("https://mcp.exa.ai/mcp")[1].command is None


@pytest.mark.parametrize("payload", [{}, {"command": "npx", "url": URL}])
def test_config_requires_exactly_one_transport(payload: dict) -> None:
    with pytest.raises(ValueError):
        McpServerSettings.model_validate(payload)


def test_config_rejects_a_non_http_url() -> None:
    with pytest.raises(ValueError):
        McpServerSettings.model_validate({"url": "ftp://example.test/x"})


def test_http_client_roundtrip() -> None:
    async def go() -> None:
        requests: list[httpx.Request] = []
        client = _http(requests)
        await client.start()
        assert client.alive
        tools = await client.list_tools()
        assert [tool["name"] for tool in tools] == ["search"]
        result = await client.call_tool("search", {"q": "hi"}, timeout=5)
        assert result["content"][0]["text"] == "hit hi"

        later = next(
            request for request in requests
            if json.loads(request.content).get("method") == "tools/list"
        )
        # The initialize handshake hands out a session id; every later
        # request echoes it with the negotiated protocol version.
        assert later.headers["mcp-session-id"] == "sess-1"
        assert later.headers["mcp-protocol-version"] == "2024-11-05"
        assert "text/event-stream" in later.headers["accept"]

        with pytest.raises(Exception) as excinfo:
            await client.call_tool("nope", {}, timeout=5)
        assert "unknown tool" in str(excinfo.value)
        await client.close()
        assert not client.alive

    asyncio.run(go())


def test_http_client_reads_sse_responses() -> None:
    async def go() -> None:
        client = _http(sse=True)
        await client.start()
        assert [tool["name"] for tool in await client.list_tools()] == ["search"]
        result = await client.call_tool("search", {"q": "sse"}, timeout=5)
        assert result["content"][0]["text"] == "hit sse"
        await client.close()

    asyncio.run(go())


def test_http_client_reports_http_status() -> None:
    async def go() -> None:
        client = _http(status=503)
        with pytest.raises(Exception) as excinfo:
            await client.start()
        assert "HTTP 503" in str(excinfo.value)
        assert client.exit_detail().startswith("HTTP 503")
        await client.close()

    asyncio.run(go())


def test_manager_starts_an_http_server(monkeypatch: pytest.MonkeyPatch) -> None:
    async def go() -> None:
        requests: list[httpx.Request] = []
        _install_http(monkeypatch, _http_transport(requests))
        manager = McpManager(McpSettings(servers={
            "rem": McpServerSettings(url=URL),
        }))
        await manager.start()
        status = manager.statuses()[0]
        assert status.status == "connected" and status.available
        assert status.url == URL
        assert status.command == "" and status.args == []
        assert status.target == URL
        assert [tool.exposed_name for tool in manager.tool_specs()] == [
            "mcp__rem__search",
        ]
        call = await manager.call("rem", "search", {"q": "x"})
        assert call.ok and "hit x" in call.content
        await manager.close()

    asyncio.run(go())


def test_manager_reports_an_unreachable_http_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def go() -> None:
        _install_http(monkeypatch, _http_transport([], status=500))
        manager = McpManager(McpSettings(servers={
            "rem": McpServerSettings(url=URL),
        }))
        await manager.start()
        status = manager.statuses()[0]
        assert status.status == "unavailable"
        assert "HTTP 500" in (status.error or "")
        assert manager.tool_specs() == []

    asyncio.run(go())


def test_session_adds_and_persists_an_http_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    async def go() -> None:
        _install_http(monkeypatch, _http_transport([]))
        session = Session(
            name="http-test",
            workspace=tmp_path,
            profile=ProviderProfile(
                base_url="http://127.0.0.1:1/v1", model_id="test",
            ),
            profile_name="test",
            api_key=None,
            config_path=config_path,
            sandbox=SandboxRuntime(),
            mcp=McpSettings(),
        )
        await session.execute(AddMcpServer(name="rem", url=URL))
        info = await session.tools_info()
        assert info.servers[0].url == URL
        assert info.servers[0].target == URL
        assert info.servers[0].status == "connected"
        assert [tool.name for tool in info.tools if tool.source == "rem"] == [
            "mcp__rem__search",
        ]

        saved = json.loads(config_path.read_text())["mcp"]["servers"]["rem"]
        assert saved["url"] == URL
        assert "command" not in saved  # exclude_none keeps the file clean
        await session.mcp.close()

    asyncio.run(go())


def test_http_client_offers_a_streamable_http_version() -> None:
    """2024-11-05 predates Streamable HTTP; the HTTP client must offer a
    modern revision and then echo the one the server negotiates."""
    async def go() -> None:
        requests: list[httpx.Request] = []
        client = _http(requests)
        await client.start()
        first = json.loads(requests[0].content)
        assert first["method"] == "initialize"
        assert first["params"]["protocolVersion"] >= "2025-03-26"
        await client.list_tools()
        later = requests[-1].headers["mcp-protocol-version"]
        assert later == "2024-11-05"  # what the fake server negotiated
        await client.close()

    asyncio.run(go())


def test_http_client_reinitializes_an_expired_session() -> None:
    """A 404 carrying our session id means the server forgot the session:
    the spec says start a new one and replay the request."""
    async def go() -> None:
        initializes: list[int] = []
        expired = {"done": False}

        def handler(request: httpx.Request) -> httpx.Response:
            message = json.loads(request.content)
            method = message.get("method")
            if method == "initialize":
                initializes.append(1)
                return httpx.Response(
                    200, json={"jsonrpc": "2.0", "id": message["id"],
                               "result": {"protocolVersion": "2025-06-18"}},
                    headers={"mcp-session-id": f"sess-{len(initializes)}"},
                )
            if message.get("id") is None:
                return httpx.Response(202)
            if not expired["done"]:
                expired["done"] = True
                return httpx.Response(404, text="no such session")
            assert request.headers["mcp-session-id"] == "sess-2"
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": message["id"],
                "result": {"tools": [{"name": "search"}]},
            })

        client = _http(transport=httpx.MockTransport(handler))
        await client.start()
        tools = await client.list_tools()  # 404s once, then succeeds
        assert [tool["name"] for tool in tools] == ["search"]
        assert len(initializes) == 2
        await client.close()

    asyncio.run(go())


def test_http_client_deletes_the_session_on_close() -> None:
    """Leaving the client should end the session, not leak it server-side."""
    async def go() -> None:
        deleted: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "DELETE":
                deleted.append(request)
                return httpx.Response(204)
            message = json.loads(request.content)
            if message.get("id") is None:
                return httpx.Response(202)
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": message["id"],
                           "result": {"tools": []}},
                headers={"mcp-session-id": "sess-9"},
            )

        client = _http(transport=httpx.MockTransport(handler))
        await client.start()
        await client.close()
        assert len(deleted) == 1
        assert deleted[0].headers["mcp-session-id"] == "sess-9"

    asyncio.run(go())


def test_http_client_names_the_legacy_transport() -> None:
    """A 405 on initialize is the old HTTP+SSE endpoint answering; the error
    should say that instead of a bare status code."""
    async def go() -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(405, text="use GET /sse"),
        )
        client = _http(transport=transport)
        with pytest.raises(Exception) as excinfo:
            await client.start()
        message = str(excinfo.value)
        assert "HTTP 405" in message
        assert "HTTP+SSE" in message
        await client.close()

    asyncio.run(go())


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        # A modern server rejects the handshake: unsupported version...
        (400, {"jsonrpc": "2.0", "id": 1, "error": {
            "code": -32022, "message": "Unsupported protocol version",
            "data": {"supported": ["2026-07-28"], "requested": "2025-06-18"},
        }}, "the server speaks 2026-07-28"),
        # ...or, if it only knows modern RPCs, "method not found".
        (404, {"jsonrpc": "2.0", "id": 1, "error": {
            "code": -32601, "message": "Method not found",
        }}, "stateless 2026-07-28 revision"),
    ],
)
def test_http_client_names_the_protocol_era_gap(
    status: int, body: dict, expected: str,
) -> None:
    """A 2026-07-28 server is a generation gap, not a broken URL: the error
    has to say which era the server speaks instead of the raw JSON."""
    async def go() -> None:
        client = _http(transport=httpx.MockTransport(
            lambda request: httpx.Response(status, json=body),
        ))
        with pytest.raises(Exception) as excinfo:
            await client.start()
        message = str(excinfo.value)
        assert expected in message
        assert "not supported yet" in message
        await client.close()

    asyncio.run(go())
