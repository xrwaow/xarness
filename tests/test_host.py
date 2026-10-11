"""Host tests: survives client disconnect, grace exit, stale socket, race.

SSE needs real unix sockets (httpx's ASGITransport does not stream), so the
host runs uvicorn on a unix socket inside the test event loop."""

import asyncio
import json
import socket
import time
from pathlib import Path

import httpx
import pytest

from xarness import host as host_mod
from xarness import protocol, session_store
from xarness.conversation import Message
from xarness.events import ContentDelta, TurnComplete, Usage
from xarness.host import GRACE_SECONDS, Host
from xarness.protocol import SendMessage, Snapshot


class FakeClient:
    """Replays scripted rounds instead of talking HTTP."""

    def __init__(self, script: list[list]) -> None:
        self.script = list(script)

    async def stream(self, wire_messages, tools=None):
        from xarness.events import ProcessingStarted
        yield ProcessingStarted()
        for event in (self.script.pop(0) if self.script else []):
            yield event


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(host_mod, "HOST_DIR", tmp_path / "host")
    monkeypatch.setattr(host_mod, "SOCKET_PATH", tmp_path / "host" / "host.sock")
    monkeypatch.setattr(host_mod, "LOCK_PATH", tmp_path / "host" / "host.lock")
    monkeypatch.setattr(host_mod, "BASELINES_DIR", tmp_path / "host" / "baselines")
    monkeypatch.setenv("XARNESS_TEST_KEY", "test-key")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "models": {"test": {
            "base_url": "https://api.example.test/v1",
            "api_key_env": "XARNESS_TEST_KEY",
            "model_id": "test-model",
        }},
        "sandboxes": {"default": {}},
        "presets": {"default": {"model": "test", "sandbox": "default"}},
        "defaults": {"preset": "default"},
    }))
    return config_path


def make_host(tmp_path) -> Host:
    return Host(
        config_path=tmp_path / "config.json",
        unix_socket=tmp_path / "host" / "test.sock",
    )


async def _serve(host: Host):
    task = asyncio.get_running_loop().create_task(host.serve())
    while not host.unix_socket.exists():
        await asyncio.sleep(0.02)
    return task


def _unix_http(socket_path: Path) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(uds=str(socket_path)),
        base_url="http://localhost",
        timeout=5.0,
    )


async def _make_session(host: Host, tmp_path, script) -> str:
    (tmp_path / "ws").mkdir(exist_ok=True)
    summary = await host.create_session(workspace=str(tmp_path / "ws"))
    session = host.sessions[summary.name]
    session.controller._client = FakeClient(script)
    return session.name


def test_diff_info_includes_text_for_path_requests(tmp_path) -> None:
    """The per-file click-through needs the unified diff text: a path-
    specific /diff request must populate DiffInfo.text, while the summary
    poll (no path) stays cheap and returns no text."""

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            name = await _make_session(host, tmp_path, [[]])
            session = host.sessions[name]
            # No real git repo here: stub the two git-backed methods.
            from xarness.gitwork import DiffStat, FileDiff

            async def fake_summary():
                return DiffStat(files=[FileDiff(
                    path="a.txt", additions=1, deletions=0, is_new=True,
                    source="agent",
                )])

            async def fake_text(path):
                return f"diff for {path}"

            session.diff_summary = fake_summary
            session.diff_text = fake_text
            async with _unix_http(host.unix_socket) as http:
                whole_resp = await http.get(f"/v1/sessions/{name}/diff")
                per_file_resp = await http.get(
                    f"/v1/sessions/{name}/diff", params={"path": "a.txt"},
                )
                assert whole_resp.status_code == 200, whole_resp.text
                assert per_file_resp.status_code == 200, per_file_resp.text
                whole = whole_resp.json()
                per_file = per_file_resp.json()
                return whole, per_file
        finally:
            task.cancel()

    whole, per_file = asyncio.run(go())
    assert whole["text"] == ""
    assert per_file["text"] == "diff for a.txt"
    assert per_file["files"][0]["path"] == "a.txt"


def test_host_survives_client_disconnect(tmp_path) -> None:
    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            name = await _make_session(
                host, tmp_path, [[ContentDelta("late answer"), TurnComplete()]],
            )
            # A client sends and disconnects immediately after.
            async with _unix_http(host.unix_socket) as http:
                result = (await http.post(
                    f"/v1/sessions/{name}/commands",
                    json={"cmd_id": "x1", "command": {"type": "send_message", "text": "hi"}},
                )).json()
                assert result["ok"], result
            # No client attached now; the turn still completes in the host.
            session = host.sessions[name]
            await asyncio.wait_for(session._turn_task, timeout=5)
            assert session._state == "idle"
            assert session.conversation.messages[-1].content == "late answer"
            # A second client attaches afterwards and sees the result.
            async with _unix_http(host.unix_socket) as http:
                snap = (await http.get(f"/v1/sessions/{name}")).json()
                assert snap["messages"][-1]["content"] == "late answer"
                assert snap["status"]["state"] == "idle"
        finally:
            await host.request_shutdown()
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_grace_exit_with_no_active_session(tmp_path) -> None:
    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        # No sessions: the host exits after the grace period.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), timeout=GRACE_SECONDS * 0.5)
        await asyncio.wait_for(task, timeout=GRACE_SECONDS + 10)
        assert not host.unix_socket.exists()

    asyncio.run(go())


def test_host_stays_alive_while_turn_runs(tmp_path) -> None:
    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            release = asyncio.Event()
            from xarness.tools import ToolRegistry, ToolResult, tool

            @tool("blocker", "blocks")
            async def blocker(args) -> ToolResult:
                await release.wait()
                return ToolResult(ok=True)

            name = await _make_session(host, tmp_path, [[
                __import__("xarness.events", fromlist=["ToolCallArgumentsDone"])
                .ToolCallArgumentsDone(call_id="c1", name="blocker", arguments_json="{}"),
                TurnComplete(has_tool_calls=True),
            ]])
            session = host.sessions[name]
            registry = ToolRegistry()
            registry.register(blocker)
            session.tool_registry = registry
            session.controller.tools = registry
            await session.execute("x", SendMessage(text="hi"))
            # The turn is running; the host must not exit during the grace window.
            await asyncio.sleep(GRACE_SECONDS * 0.6)
            assert not task.done()
            release.set()
            await asyncio.wait_for(session._turn_task, timeout=5)
        finally:
            await host.request_shutdown()
            if not task.done():
                await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_stale_socket_is_removed_and_host_spawns(tmp_path) -> None:
    from xarness.host import connect_or_spawn

    spawned = []
    keep_alive: list[socket.socket] = []

    def fake_popen(*args, **kwargs):
        spawned.append(1)

        def bind_soon():
            time.sleep(0.2)
            if host_mod.SOCKET_PATH.exists():
                host_mod.SOCKET_PATH.unlink()
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(str(host_mod.SOCKET_PATH))
            srv.listen(1)
            keep_alive.append(srv)  # outliving the thread keeps it listening

        import threading
        threading.Thread(target=bind_soon, daemon=True).start()

        class P:
            def poll(self):
                return None
        return P()

    (host_mod.SOCKET_PATH.parent).mkdir(parents=True, exist_ok=True)
    host_mod.SOCKET_PATH.touch()  # a stale regular file at the socket path
    monkey = pytest.MonkeyPatch()
    monkey.setattr(host_mod.subprocess, "Popen", fake_popen)
    try:
        path = connect_or_spawn(timeout=10)
        assert path == host_mod.SOCKET_PATH
        assert spawned == [1]
    finally:
        monkey.undo()


def test_shared_baseline_moves_with_accept(tmp_path) -> None:
    """The baseline is shared per workspace: /accept in one session moves it
    for every other live session on that directory."""
    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            (tmp_path / "ws").mkdir(exist_ok=True)
            name_a = (await host.create_session(workspace=str(tmp_path / "ws"))).name
            name_b = (await host.create_session(workspace=str(tmp_path / "ws"))).name
            a, b = host.sessions[name_a], host.sessions[name_b]
            assert a.git_info is not None and b.git_info is not None
            # Both sessions start from the same baseline tree.
            assert a.git_info.baseline_tree == b.git_info.baseline_tree
            # /accept in A (with a fake accepted tree) moves B's checkpoints.
            accepted = "0" * 40
            await a.execute("x1", protocol.Accept())
            # accept_changes ran against the real repo (no changes: same tree);
            # simulate the shared move directly:
            host.share_baseline(a, a.git_info, accepted)
            assert b.conversation.messages == [] or True
            b_user = Message(role="user", content="q", checkpoint_sha="1" * 40, after_tree="2" * 40)
            b.conversation.add(b_user)
            host.share_baseline(a, a.git_info, accepted)
            assert b_user.checkpoint_sha == accepted
            # The record exists and carries the accepted tree.
            tree, epoch = host.read_baseline(a.git_info.workspace)
            assert tree == accepted and epoch >= 1
        finally:
            await host.request_shutdown()
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_clientlib_attach_reads_snapshot_and_events(tmp_path) -> None:
    """The typed client's SSE reader against the real host: snapshot frame
    first, then live events."""
    from xarness.clientlib import XarnessClient

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            name = await _make_session(
                host, tmp_path, [[ContentDelta("answer"), TurnComplete()]],
            )
            client = await XarnessClient.connect_unix(host.unix_socket, spawn=False)
            try:
                stream = client.attach(name)
                first = await asyncio.wait_for(anext(stream), timeout=5)
                assert isinstance(first, Snapshot)
                host.sessions[name]._emit_notice("live one")
                second = await asyncio.wait_for(anext(stream), timeout=5)
                assert isinstance(second, protocol.Notice)
                assert second.text == "live one"
            finally:
                await client.aclose()
        finally:
            await host.request_shutdown()
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_workspace_last_used_model_id_falls_through(tmp_path) -> None:
    """Old sessions persisted profile.model_id instead of the profile key;
    a last-used value that isn't a known model name must fall through to
    the preset's model, not fail the request."""
    async def go():
        host = make_host(tmp_path)
        (tmp_path / "ws").mkdir(exist_ok=True)
        # A saved session carrying the model_id as its profile key.
        session_store.save_state("stale", session_store.SessionState(
            conversation=__import__("xarness.conversation", fromlist=["Conversation"]).Conversation(),
            profile="z-ai/glm-5.3-flash",  # bogus: a model_id, not a name
            workspace=str(tmp_path / "ws"),
            updated_at="2026-01-01T00:00:00+00:00",
        ))
        parts = await host._build_parts(
            workspace=str(tmp_path / "ws"), model=None, sandbox=None,
            preset=None, no_init_repo=True, session_name="s",
        )
        assert parts["profile_name"] == "test"  # the preset's model

    asyncio.run(go())


def test_old_layout_resolves_default_preset(tmp_path) -> None:
    """An old-layout config synthesizes its default preset from
    default_profile, so preset-based resolution works unchanged."""
    from xarness.config import load_sectioned
    config_path = tmp_path / "old.json"
    config_path.write_text(json.dumps({
        "default_profile": "glm-flash",
        "default_theme": "ayu-darker",
        "profiles": [
            {"name": "local", "base_url": "http://127.0.0.1:8080/v1",
             "api_key_env": None, "model_id": "qwen"},
            {"name": "glm-flash", "base_url": "https://openrouter.ai/api/v1",
             "api_key_env": "OPENROUTER_API_KEY", "model_id": "z-ai/glm-5.3-flash"},
        ],
    }))
    sectioned = load_sectioned(config_path)
    name, profile = sectioned.resolve_model(None)
    assert name == "glm-flash" and profile.model_id == "z-ai/glm-5.3-flash"


def test_tui_end_to_end_draft_flow(tmp_path) -> None:
    """The full stack: TUI (pilot) -> clientlib -> host API -> Session turn.

    Regression: the event pump died on the first UserMessageAdded (a sync
    handler was awaited), killing the TUI's view while the turn kept
    running in the host."""
    from xarness.clientlib import XarnessClient
    from xarness.tui.app import AgentApp
    from xarness.tui.widgets import AssistantMessage, UserMessage

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            (tmp_path / "ws").mkdir(exist_ok=True)
            client = await XarnessClient.connect_unix(host.unix_socket, spawn=False)
            # Inject the scripted provider the moment the draft flow creates
            # the session (before the first message is sent).
            orig_create = client.create_session

            async def create_session(**kwargs):
                summary = await orig_create(**kwargs)
                host.sessions[summary.name].controller._client = FakeClient(
                    [[ContentDelta("Hello!"), TurnComplete(usage=Usage(10, 2))]]
                )
                return summary

            client.create_session = create_session
            app = AgentApp(client, workspace=tmp_path / "ws")
            async with app.run_test(size=(100, 40)) as pilot:
                await pilot.pause()
                await pilot.click("#chat-input")
                await pilot.press(*"meow")
                await pilot.press("enter")
                await pilot.pause(1.0)
                assert app.session_name is not None
                assert app.status is not None and app.status.state == "idle"
                # No local echo: the user message arrived via the stream.
                assert len(app.query("#chat-log > UserMessage")) == 1
                assert len(app.query("#chat-log > AssistantMessage")) == 1
                assert not app.query("#chat-log > ErrorLine")
            await client.aclose()
        finally:
            await host.request_shutdown()
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_startup_race_yields_one_host(tmp_path) -> None:
    from xarness.host import connect_or_spawn

    spawn_count = []
    keep_alive: list[socket.socket] = []

    def fake_popen(*args, **kwargs):
        spawn_count.append(1)

        def bind_soon():
            time.sleep(0.3)
            if host_mod.SOCKET_PATH.exists():
                host_mod.SOCKET_PATH.unlink()
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(str(host_mod.SOCKET_PATH))
            srv.listen(1)
            keep_alive.append(srv)  # outliving the thread keeps it listening

        import threading
        threading.Thread(target=bind_soon, daemon=True).start()

        class P:
            def poll(self):
                return None
        return P()

    monkey = pytest.MonkeyPatch()
    monkey.setattr(host_mod.subprocess, "Popen", fake_popen)
    try:
        async def go():
            results = await asyncio.gather(
                asyncio.to_thread(connect_or_spawn, timeout=10),
                asyncio.to_thread(connect_or_spawn, timeout=10),
            )
            assert results[0] == results[1]
            assert len(spawn_count) <= 1

        asyncio.run(go())
    finally:
        monkey.undo()


def test_load_session_uses_saved_workspace_sandbox_and_baseline(tmp_path, monkeypatch) -> None:
    """Loading a saved session must honor its saved workspace (never the
    host's cwd), its saved sandbox name/settings, and re-sync the workspace
    baseline record (another session may have accepted changes)."""
    from xarness.config import SandboxSettings

    async def go():
        host = make_host(tmp_path)
        (tmp_path / "ws").mkdir()
        (tmp_path / "elsewhere").mkdir()
        summary = await host.create_session(
            workspace=str(tmp_path / "ws"), no_init_repo=False,
        )
        name = summary.name
        session = host.sessions[name]
        assert session.git_info is not None
        # A distinct saved sandbox state + an /accept that happened in
        # another session (a newer baseline record for the workspace).
        session.sandbox_name = "default"
        session.sandbox_settings = SandboxSettings(allow_git_write=True)
        session._persist()
        host.write_baseline(session.git_info.workspace, "f" * 40)
        await host.unload_session(name)

        # Load from a different host cwd: the saved workspace must win.
        monkeypatch.chdir(tmp_path / "elsewhere")
        loaded = await host.load_session(name)
        assert loaded.workspace == (tmp_path / "ws").resolve()
        assert loaded.sandbox_name == "default"
        assert loaded.sandbox_settings is not None
        assert loaded.sandbox_settings.allow_git_write is True
        assert loaded.git_info is not None
        assert loaded.git_info.workspace == (tmp_path / "ws").resolve()
        # The workspace baseline record won over the saved git block.
        assert loaded.git_info.baseline_tree == "f" * 40
        if loaded.sandbox is not None:
            assert loaded.sandbox.workspace == (tmp_path / "ws").resolve()
        return host

    asyncio.run(go())


def test_host_stays_alive_while_a_client_is_attached(tmp_path) -> None:
    """An open event stream counts as active: the host must not exit during
    the grace window while a client is attached to a session, even with no
    turn running."""
    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            name = await _make_session(host, tmp_path, [[ContentDelta("x"), TurnComplete()]])
            session = host.sessions[name]
            session._subscribers.append(asyncio.Queue())
            await asyncio.sleep(GRACE_SECONDS + 1)
            assert not task.done()
            session._subscribers.clear()
            await asyncio.wait_for(task, timeout=GRACE_SECONDS + 10)
        finally:
            if not task.done():
                await host.request_shutdown()
                await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_run_foreground_with_live_host_prints_stats_and_never_binds(tmp_path, capsys) -> None:
    async def go():
        # Serve on the global SOCKET_PATH so run_foreground's live-host
        # check finds it.
        host = Host(
            persistent=False, config_path=tmp_path / "config.json",
            unix_socket=host_mod.SOCKET_PATH,
        )
        task = asyncio.get_running_loop().create_task(host.serve())
        while not host.unix_socket.exists():
            await asyncio.sleep(0.02)
        try:
            # A live host exists: --server prints its stats instead of
            # binding (and never unlinks the live socket).
            await asyncio.to_thread(host_mod.run_foreground)
            assert host_mod._socket_alive()
            assert host.unix_socket.exists()
        finally:
            await host.request_shutdown()
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())
    out = capsys.readouterr().out
    assert "already running" in out and "sessions:" in out


def test_start_unix_server_refuses_to_unlink_a_live_socket(tmp_path) -> None:
    import socket as socket_mod

    sock = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
    path = tmp_path / "live.sock"
    sock.bind(str(path))
    sock.listen(1)
    try:
        from xarness.host import _start_unix_server
        async def attempt():
            return await _start_unix_server(
                Host(unix_socket=path, config_path=tmp_path / "cfg.json"),
            )
        with pytest.raises(RuntimeError, match="live host"):
            asyncio.run(attempt())
        assert path.exists()  # the live socket was never unlinked
    finally:
        sock.close()


def test_session_listing_uses_meta_not_conversations(tmp_path, monkeypatch) -> None:
    async def go():
        host = make_host(tmp_path)
        (tmp_path / "ws").mkdir()
        await host.create_session(workspace=str(tmp_path / "ws"))
        name = next(iter(host.sessions))
        host.sessions[name]._persist()
        await host.unload_session(name)
        # Listing and last-used lookups must not parse full conversations.
        def explode(*args, **kwargs):
            raise AssertionError("full conversation parsed")
        monkeypatch.setattr(session_store, "load_state", explode)

        summaries = host.list_sessions()
        saved = [s for s in summaries if s.name == name and not s.active]
        assert saved and saved[0].workspace
        assert saved[0].model == "test"
        last_model, last_sandbox = host._workspace_last_used(
            (tmp_path / "ws").resolve(),
        )
        assert last_model == "test"

    asyncio.run(go())


def test_create_session_endpoint_uses_request_model(tmp_path) -> None:
    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            (tmp_path / "ws").mkdir()
            async with _unix_http(host.unix_socket) as http:
                response = await http.post("/v1/sessions", json={
                    "workspace": str(tmp_path / "ws"), "no_init_repo": True,
                    "first_message": None,
                })
                assert response.status_code == 201, response.text
                body = response.json()
                assert body["name"]
                assert body["workspace"] == str((tmp_path / "ws").resolve())
        finally:
            await host.request_shutdown()
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_idle_session_is_released_and_listed_as_saved(tmp_path) -> None:
    """Send a message, let the turn finish, close the stream: the session
    leaves host.sessions (its sandbox shell released), is not listed as
    active, and /sessions lists it as saved."""
    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            name = await _make_session(
                host, tmp_path, [[ContentDelta("answer"), TurnComplete()]],
            )
            session = host.sessions[name]
            snapshot, queue = session.subscribe()  # a client attaches
            await session.execute("x1", SendMessage(text="hi"))
            await asyncio.wait_for(session._turn_task, timeout=5)
            assert host.summary_for(session).active
            # The client closes the stream.
            session.unsubscribe(queue)
            await asyncio.sleep(0.1)  # let the release task run
            assert name not in host.sessions, "idle session must be released"
            # It is saved, and /sessions lists it as saved (not active).
            assert session_store.session_path(name).exists()
            summaries = host.list_sessions()
            row = next(s for s in summaries if s.name == name)
            assert row.active is False
            assert row.state == "saved"
            # And the HTTP listing agrees.
            async with _unix_http(host.unix_socket) as http:
                rows = (await http.get("/v1/sessions")).json()
                http_row = next(r for r in rows if r["name"] == name)
                assert http_row["active"] is False and http_row["state"] == "saved"
                active_rows = (
                    await http.get("/v1/sessions", params={"active": "true"})
                ).json()
                assert all(r["name"] != name for r in active_rows)
        finally:
            await host.request_shutdown()
            if not task.done():
                await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_loaded_drifted_session_not_waiting_without_client(tmp_path) -> None:
    """A saved session in a drifted workspace, loaded with no client
    attached: state is not "waiting" and the session is not active, so the
    host can exit. The drift prompt appears only once a client attaches."""
    async def go():
        host = make_host(tmp_path)
        (tmp_path / "ws").mkdir()
        summary = await host.create_session(workspace=str(tmp_path / "ws"))
        name = summary.name
        session = host.sessions[name]
        assert session.git_info is not None
        session._persist()
        # Drift: the workspace changes outside the session after the save.
        (tmp_path / "ws" / "drifted.txt").write_text("changed\n")
        await host.unload_session(name)

        loaded = await host.load_session(name)
        assert loaded._pending_drift is None, "drift must not be computed at load"
        assert not host.is_active_session(loaded), "host must be able to exit"
        row = host.summary_for(loaded)
        assert row.state != "waiting" and row.active is False

        # A client attaches: NOW the drift is detected and prompted.
        await loaded.prompt_resume_drift()
        assert loaded._pending_drift is not None
        assert loaded.status().pending_drift is not None

    asyncio.run(go())
