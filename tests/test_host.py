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
from xarness.conversation import user_message
from xarness.host import Host
from xarness.protocol import ContentDelta, SendMessage, Snapshot, RoundComplete, Usage
from xarness.session import SessionError
from xarness.workspace import WorkspaceState


class FakeClient:
    """Replays scripted rounds instead of talking HTTP."""

    def __init__(self, script: list[list]) -> None:
        self.script = list(script)

    async def stream(self, wire_messages, tools=None):
        from xarness.protocol import ProcessingStarted
        yield ProcessingStarted()
        for event in (self.script.pop(0) if self.script else []):
            yield event


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(host_mod, "HOST_DIR", tmp_path / "host")
    monkeypatch.setattr(host_mod, "SOCKET_PATH", tmp_path / "host" / "host.sock")
    monkeypatch.setattr(host_mod, "LOCK_PATH", tmp_path / "host" / "host.lock")
    # Short lifecycle windows: these tests only care that the host waits past
    # the grace period, not how long it is in production.
    monkeypatch.setattr(host_mod, "GRACE_SECONDS", 0.3)
    monkeypatch.setattr(host_mod, "REAP_INTERVAL_SECONDS", 0.02)
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
    summary = await host.create_session(protocol.CreateSessionRequest(workspace=str(tmp_path / "ws")))
    session = host.sessions[summary.name]
    session._client = FakeClient(script)
    return session.name


def test_create_session_creates_missing_workspace(tmp_path) -> None:
    """A workspace path that doesn't exist yet is created (mkdir -p), not
    refused — starting a session on a brand-new directory is normal."""

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            missing = tmp_path / "data" / "new_ws"
            summary = await host.create_session(protocol.CreateSessionRequest(workspace=str(missing)))
            assert missing.is_dir()
            assert host.sessions[summary.name].workspace == missing.resolve()
            # An uncreatable path still fails with a readable error.
            blocked = tmp_path / "file.txt" / "under"
            (tmp_path / "file.txt").write_text("x")
            with pytest.raises(SessionError, match="could not be created"):
                await host.create_session(protocol.CreateSessionRequest(workspace=str(blocked)))
        finally:
            task.cancel()

    asyncio.run(go())


def test_workspace_diff_needs_no_session(tmp_path, monkeypatch) -> None:
    """A draft chat's /diff: tracking is per repo root, so the workspace
    endpoint answers with the same payload a session's /diff would — and
    the rows are labelled for no viewer (nothing is ``mine`` before a
    session exists)."""

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            ws = tmp_path / "ws"
            ws.mkdir()
            state = WorkspaceState(
                root=ws, git_dir=ws / ".git", baseline_tree="t" * 40,
            )
            seen_scopes: list[str] = []

            async def fake_open(root, *, allow_init=True, persist=True):
                return state

            async def fake_observe():
                return "h" * 40

            async def fake_diff(viewer=None, current=None, path=None, scope="all"):
                assert viewer is None, "a draft has no session to label rows for"
                seen_scopes.append(scope)
                return protocol.DiffInfo(
                    files=[protocol.DiffFileInfo(
                        path="a.txt", additions=1, deletions=0, is_new=True,
                        kind="agent",
                    )],
                    additions=1,
                )

            async def fake_text(path, current=None, author=None):
                return f"diff for {path}"

            state.observe = fake_observe
            state.diff = fake_diff
            state.file_diff = fake_text
            monkeypatch.setattr(host_mod, "open_workspace", fake_open)
            async with _unix_http(host.unix_socket) as http:
                whole = await http.get(
                    "/v1/workspaces/diff", params={"workspace": str(ws)},
                )
                per_file = await http.get(
                    "/v1/workspaces/diff",
                    params={"workspace": str(ws), "path": "a.txt"},
                )
                scoped = await http.get(
                    "/v1/workspaces/diff",
                    params={"workspace": str(ws), "scope": "mine"},
                )
                missing = await http.get(
                    "/v1/workspaces/diff",
                    params={"workspace": str(tmp_path / "nope")},
                )
            assert whole.status_code == 200, whole.text
            assert whole.json()["tracking"] is True
            assert whole.json()["files"][0]["path"] == "a.txt"
            assert whole.json()["text"] == ""
            assert per_file.status_code == 200, per_file.text
            assert per_file.json()["text"] == "diff for a.txt"
            # The chip's scope rides the query, exactly as it does on a
            # session's /diff.
            assert scoped.status_code == 200, scoped.text
            assert seen_scopes == ["all", "all", "mine"]
            # A workspace that isn't there is no error: just nothing tracked.
            assert missing.status_code == 200, missing.text
            assert missing.json()["tracking"] is False
            # Nothing was created or hosted: the panel needs no session.
            assert host.sessions == {}
        finally:
            task.cancel()

    asyncio.run(go())


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
            # No real git repo here: stub the git-backed pieces of the
            # diff and give the session a tracked workspace.
            from xarness.protocol import DiffFileInfo, DiffInfo

            async def fake_diff(viewer=None, current=None, path=None, scope="all"):
                return DiffInfo(files=[DiffFileInfo(
                    path="a.txt", additions=1, deletions=0, is_new=True,
                    kind="agent",
                )], additions=1, deletions=0)

            async def fake_text(path, current=None, author=None):
                return f"diff for {path}"

            session.workspace_state = WorkspaceState(
                root=tmp_path, git_dir=tmp_path / ".git",
                baseline_tree="t" * 40,
            )
            monkey = pytest.MonkeyPatch()
            async def fake_snapshot():
                return "t" * 40

            session.workspace_state.observe = fake_snapshot
            session.workspace_state.diff = fake_diff
            session.workspace_state.file_diff = fake_text
            try:
                async with _unix_http(host.unix_socket) as http:
                    whole_resp = await http.get(f"/v1/sessions/{name}/diff")
                    per_file_resp = await http.get(
                        f"/v1/sessions/{name}/diff", params={"path": "a.txt"},
                    )
                    assert whole_resp.status_code == 200, whole_resp.text
                    assert per_file_resp.status_code == 200, per_file_resp.text
                    return whole_resp.json(), per_file_resp.json()
            finally:
                monkey.undo()
        finally:
            task.cancel()

    whole, per_file = asyncio.run(go())
    assert whole["text"] == ""
    assert per_file["text"] == "diff for a.txt"
    assert per_file["files"][0]["path"] == "a.txt"


def test_saved_session_reads_load_and_host_the_session(tmp_path) -> None:
    """GET /sessions/{name} and /diff use the one load path: an unloaded
    session is hosted on demand (its state and saved tracking reused), and
    the reaper releases it again when it goes idle."""

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            name = await _make_session(host, tmp_path, [[]])
            await host.unload_session(name)
            assert name not in host.sessions

            async with _unix_http(host.unix_socket) as http:
                snap_resp = await http.get(f"/v1/sessions/{name}")
                diff_resp = await http.get(f"/v1/sessions/{name}/diff")
                assert snap_resp.status_code == 200, snap_resp.text
                assert diff_resp.status_code == 200, diff_resp.text
                snap = snap_resp.json()
                assert snap["status"]["session_name"] == name
                assert snap["status"]["state"] == "idle"
                # Same saved git block: the baseline survived the reload.
                assert diff_resp.json()["tracking"] is True
            assert name in host.sessions
        finally:
            await host.request_shutdown()
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_view_state_round_trips_through_the_session(tmp_path) -> None:
    """The /diff scope travels with the session: it can be set at creation,
    changed by a command over HTTP, and it is still there after the host
    reloads the session from disk."""

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            (tmp_path / "ws").mkdir(exist_ok=True)
            summary = await host.create_session(protocol.CreateSessionRequest(
                workspace=str(tmp_path / "ws"),
                view=protocol.ViewState(diff_scope="mine"),
            ))
            name = summary.name
            assert host.sessions[name].status().view.diff_scope == "mine"
            async with _unix_http(host.unix_socket) as http:
                await http.post(
                    f"/v1/sessions/{name}/commands",
                    json={
                        "type": "set_view_state",
                        "view": {"diff_scope": "all"},
                    },
                )
                snap = (await http.get(f"/v1/sessions/{name}")).json()
                assert snap["status"]["view"]["diff_scope"] == "all"

            # Reload from disk: the saved filter comes back on attach.
            await host.unload_session(name)
            assert name not in host.sessions
            async with _unix_http(host.unix_socket) as http:
                snap = (await http.get(f"/v1/sessions/{name}")).json()
                assert snap["status"]["view"]["diff_scope"] == "all"
        finally:
            await host.request_shutdown()
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_host_survives_client_disconnect(tmp_path) -> None:
    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            name = await _make_session(
                host, tmp_path, [[ContentDelta(text="late answer"), RoundComplete()]],
            )
            # A client sends and disconnects immediately after.
            async with _unix_http(host.unix_socket) as http:
                await http.post(
                    f"/v1/sessions/{name}/commands",
                    json={"type": "send_message", "text": "hi"},
                )
            # No client attached now; the turn still completes in the host.
            session = host.sessions[name]
            await asyncio.wait_for(session._turn_task, timeout=5)
            assert session.state == "idle"
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
            await asyncio.wait_for(asyncio.shield(task), timeout=host_mod.GRACE_SECONDS * 0.5)
        await asyncio.wait_for(task, timeout=host_mod.GRACE_SECONDS + 10)
        assert not host.unix_socket.exists()

    asyncio.run(go())


def test_host_stays_alive_while_turn_runs(tmp_path) -> None:
    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            release = asyncio.Event()
            from pydantic import BaseModel

            from xarness.tools import ToolRegistry, ToolResult, tool

            class _NoArgs(BaseModel):
                pass

            @tool("blocker", "blocks", _NoArgs)
            async def blocker(args) -> ToolResult:
                await release.wait()
                return ToolResult()

            name = await _make_session(host, tmp_path, [[
                __import__("xarness.protocol", fromlist=["ToolCallArgumentsDone"])
                .ToolCallArgumentsDone(call_id="c1", name="blocker", arguments_json="{}"),
                RoundComplete(has_tool_calls=True),
            ]])
            session = host.sessions[name]
            registry = ToolRegistry()
            registry.register(blocker)
            session.tool_registry = registry
            session.tool_registry = registry
            await session.execute(SendMessage(text="hi"))
            # The turn is running; the host must not exit during the grace window.
            await asyncio.sleep(host_mod.GRACE_SECONDS * 0.6)
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


def test_load_session_restores_saved_baseline(tmp_path) -> None:
    """The regression the shared-baseline experiment caused: a session that
    made changes and exited WITHOUT /accept must reopen with its saved
    baseline and turn checkpoints intact — no drift prompt, /undo working."""

    async def go():
        host = make_host(tmp_path)
        (tmp_path / "ws").mkdir()
        summary = await host.create_session(protocol.CreateSessionRequest(workspace=str(tmp_path / "ws")))
        name = summary.name
        session = host.sessions[name]
        assert session.workspace_state is not None
        # Simulate a finished turn that changed the workspace (never accepted).
        session.workspace_state.baseline_tree = "b" * 40
        user = session.conversation.add(user_message("do things"))
        session.conversation.begin_turn(user, "b" * 40)
        session.conversation.turns[user.id].after_tree = "c" * 40
        session._persist()
        await host.unload_session(name)

        loaded = await host.load_session(name)
        info = loaded.workspace_state
        assert info is not None
        # The saved baseline survived the round-trip — it was NOT reset to
        # the workspace's current state by the fresh setup_tracking().
        assert info.baseline_tree == "b" * 40
        turns = loaded.turn_checkpoints()
        assert turns and turns[-1].after_tree == "c" * 40
        # /undo still sees the turn's tree pair (not flattened to one sha).
        assert turns[-1].before_tree != turns[-1].after_tree
        # Drift compares current state against the last known tree — with the
        # saved (real) baseline restored, no drift is reported after a
        # clean exit. (Fake shas above only test persistence.)
        assert loaded.workspace_state is not None
    asyncio.run(go())


def test_no_false_drift_after_clean_exit(tmp_path) -> None:
    """The reported symptom: make changes in a session, exit WITHOUT
    /accept, reopen — no "workspace changed" prompt may appear (the saved
    baseline, not the workspace's current state, must be restored)."""
    async def go():
        host = make_host(tmp_path)
        (tmp_path / "ws").mkdir()
        (tmp_path / "ws" / "file.txt").write_text("hello")
        summary = await host.create_session(protocol.CreateSessionRequest(workspace=str(tmp_path / "ws")))
        name = summary.name
        session = host.sessions[name]
        assert session.workspace_state is not None
        current = await session.workspace_state.refresh()
        user = session.conversation.add(user_message("edit"))
        session.conversation.begin_turn(user, current)
        session.conversation.turns[user.id].after_tree = current
        session._persist()
        await host.unload_session(name)

        loaded = await host.load_session(name)
        assert loaded.workspace_state is not None
        # The saved baseline survived: the reopen did not re-baseline.
        assert loaded.workspace_state.baseline_tree == current
    asyncio.run(go())


def test_load_session_finalizes_turn_left_open_by_a_crash(tmp_path) -> None:
    """Quitting (or a host crash) mid-turn leaves the last user turn without
    an after-tree. Reopening must finalize it against the workspace as it is
    now — not leave it dangling, which made every reopen report drift."""
    async def go():
        host = make_host(tmp_path)
        (tmp_path / "ws").mkdir()
        (tmp_path / "ws" / "file.txt").write_text("hello")
        summary = await host.create_session(protocol.CreateSessionRequest(workspace=str(tmp_path / "ws")))
        name = summary.name
        session = host.sessions[name]
        assert session.workspace_state is not None
        current = await session.workspace_state.refresh()
        # A turn with a checkpoint but no after-tree: finish_turn never ran.
        user = session.conversation.add(user_message("still working"))
        session.conversation.begin_turn(user, current)
        session._persist()
        await host.unload_session(name)

        loaded = await host.load_session(name)
        assert loaded.workspace_state is not None
        # The crashed turn is closed and stamped with the live tree (no
        # edits since the crash, so it has nothing to revert).
        (turn,) = loaded.conversation.turns.values()
        assert turn.closed and turn.after_tree
        assert loaded.turn_checkpoints() == []
    asyncio.run(go())


def test_load_session_without_git_block_falls_back_to_fresh_tracking(tmp_path) -> None:
    """A session with no saved git block (pre-tracking) still gets fresh
    tracking on load."""
    async def go():
        host = make_host(tmp_path)
        (tmp_path / "ws").mkdir()
        summary = await host.create_session(protocol.CreateSessionRequest(workspace=str(tmp_path / "ws")))
        name = summary.name
        session = host.sessions[name]
        assert session.workspace_state is not None
        session._persist()
        await host.unload_session(name)

        # Drop the saved git block: loading must fall back to fresh
        # tracking instead of failing.
        with session_store.connection() as db:
            (state_json,) = db.execute(
                "SELECT state FROM sessions WHERE name = ?", (name,),
            ).fetchone()
            state = json.loads(state_json)
            state.pop("git", None)
            db.execute(
                "UPDATE sessions SET state = ? WHERE name = ?",
                (json.dumps(state), name),
            )

        loaded = await host.load_session(name)
        assert loaded.workspace_state is not None
        assert loaded.workspace_state.root == (tmp_path / "ws").resolve()
        assert loaded.workspace_state.baseline_tree  # a real tree sha
    asyncio.run(go())


def test_protocol_models_tolerate_unknown_fields() -> None:
    """A newer host's extra fields must parse, not crash with extra_forbidden."""
    raw = {
        "seq": 7,
        "status": {"state": "idle"},
        "messages": [
            {
                "role": "assistant",
                "content": "hi",
                "time_summary": "22m 45s | 8m 57s 11m 48s",
                "some_future_field": {"nested": True},
            },
        ],
    }
    snap = protocol.Snapshot.model_validate(raw)
    assert snap.messages[0].content == "hi"
    assert not hasattr(snap.messages[0].timing, "summary")
    # Events are equally tolerant.
    event = protocol.Notice.model_validate({"seq": 1, "text": "x", "new": 1})
    assert event.text == "x"
    # Commands stay strict: the host must reject unknown commands' fields.
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        protocol.SendMessage.model_validate({"text": "x", "bogus": 1})


def test_clientlib_attach_reads_snapshot_and_events(tmp_path) -> None:
    """The typed client's SSE reader against the real host: snapshot frame
    first, then live events."""
    from xarness.clientlib import XarnessClient

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            name = await _make_session(
                host, tmp_path, [[ContentDelta(text="answer"), RoundComplete()]],
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


def test_attach_window_and_message_pages_cover_the_history(tmp_path) -> None:
    """Attaching sends only the tail; pages fetched on demand add up to the
    whole history exactly once (both on the SSE and the plain route)."""
    from xarness.clientlib import XarnessClient
    from xarness.conversation import AssistantMessage

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            name = await _make_session(host, tmp_path, [])
            conversation = host.sessions[name].conversation
            for index in range(30):
                conversation.add(user_message(f"q{index}"))
                conversation.add(AssistantMessage(content=f"a{index}"))
            full = list(conversation.messages)

            client = await XarnessClient.connect_unix(host.unix_socket, spawn=False)
            try:
                stream = client.attach(name, window="[-8:]")
                first = await asyncio.wait_for(anext(stream), timeout=5)
                await stream.aclose()
                assert isinstance(first, Snapshot)
                assert first.has_more and 0 < first.message_offset
                assert first.messages == full[first.message_offset:]
                assert first.message_total == len(full) > len(first.messages)

                collected = list(first.messages)
                window: str | None = (
                    f"[{max(0, first.message_offset - 8)}:{first.message_offset}]"
                    if first.has_more else None
                )
                while window:
                    page = await client.message_page(name, window=window)
                    collected = [*page.messages, *collected]
                    window = (
                        f"[{max(0, page.start - 8)}:{page.start}]"
                        if page.has_more else None
                    )
                assert page.start == 0 and collected == full
            finally:
                await client.aclose()

            # The plain snapshot route takes the same window parameter.
            async with _unix_http(host.unix_socket) as http:
                response = await http.get(f"/v1/sessions/{name}", params={"range": "[-8:]"})
                page = await http.get(
                    f"/v1/sessions/{name}/messages",
                    params={"range": "[-8:]", "slim": "true"},
                )
                body = await http.get(
                    f"/v1/sessions/{name}/messages/1", params={"content": "false"},
                )
            window = Snapshot.model_validate(response.json())
            assert window.has_more
            assert window.messages == full[window.message_offset:]
            assert page.json()["slim"] is True
            assert body.json()["messages"][0]["role"] == "assistant"
        finally:
            await host.request_shutdown()
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_clientlib_refuses_a_host_that_speaks_another_protocol(
    tmp_path, monkeypatch,
) -> None:
    """The version handshake is the client's /v1/health GET: a host built
    from a different protocol version is replaced when it has no work, and
    refused (never killed) while it still hosts active sessions."""
    from xarness.clientlib import ClientError, XarnessClient

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            name = await _make_session(
                host, tmp_path, [[ContentDelta(text="x"), RoundComplete()]],
            )
            session = host.sessions[name]
            session.subscribe()  # an attached client makes it active work
            real_info = host.server_info
            monkeypatch.setattr(
                type(host), "server_info",
                lambda self: real_info().model_copy(
                    update={"protocol_version": protocol.PROTOCOL_VERSION + 1},
                ),
            )
            with pytest.raises(ClientError, match="host speaks v"):
                await XarnessClient.connect_unix(host.unix_socket, spawn=False)
        finally:
            await host.request_shutdown()
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_kill_sessions_stops_the_host_too(tmp_path, monkeypatch, capsys) -> None:
    """``--kill`` must not leave the old host running: a reset that only
    unloaded sessions would reconnect to the code that host imported."""
    from xarness import cli
    from xarness.clientlib import XarnessClient

    stopped: list[int] = []

    class FakeClient:
        async def health(self):
            return type("Info", (), {"pid": 4242})()

        async def sessions(self, active=False):
            return [type("S", (), {"name": "one"})(),
                    type("S", (), {"name": "two"})()]

        async def unload_session(self, name):
            return True

        async def aclose(self):
            pass

    async def fake_connect(cls, **kwargs):
        return FakeClient()

    def fake_stop(pid, path):
        stopped.append(pid)

    monkeypatch.setattr(XarnessClient, "connect_unix", classmethod(fake_connect))
    monkeypatch.setattr(host_mod, "stop_host", fake_stop)

    cli._kill_sessions()

    assert stopped == [4242]
    out = capsys.readouterr().out
    assert "closed one" in out and "closed two" in out
    assert "host stopped" in out


def test_clientlib_replaces_a_stale_host_with_no_work(
    tmp_path, monkeypatch,
) -> None:
    """The other half of the handshake: a mismatched host with no active
    session is shut down (and the client reconnects, respawning it) rather
    than left running."""
    from xarness import host as host_mod
    from xarness.clientlib import XarnessClient

    class Replaced(Exception):
        """Stops the test where the respawn would start."""

    stopped: list[tuple[int, Path]] = []

    def fake_stop(pid: int, path: Path) -> None:
        stopped.append((pid, path))
        raise Replaced

    monkeypatch.setattr(host_mod, "stop_host", fake_stop)

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            real_info = host.server_info
            monkeypatch.setattr(
                type(host), "server_info",
                lambda self: real_info().model_copy(update={"protocol_version": 0}),
            )
            with pytest.raises(Replaced):
                await XarnessClient.connect_unix(host.unix_socket, spawn=False)
            assert stopped == [(real_info().pid, host.unix_socket)]
        finally:
            await host.request_shutdown()
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_quit_verdict_route_counts_the_other_streams(tmp_path) -> None:
    """A client about to disconnect asks the host (read-only route, no
    command lock): alone + work in flight is the only case that needs the
    continue-headless menu."""
    from xarness.clientlib import XarnessClient

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            name = await _make_session(host, tmp_path, [])
            session = host.sessions[name]
            client = await XarnessClient.connect_unix(host.unix_socket, spawn=False)
            try:
                stream = client.attach(name)
                assert isinstance(
                    await asyncio.wait_for(anext(stream), timeout=5), Snapshot,
                )
                assert (await client.quit_verdict(name)).ask is False  # idle

                session._pending_ask = protocol.PendingAsk(
                    ask_id="a1", questions=["pick one"],
                )
                alone = await client.quit_verdict(name)
                assert alone.ask is True and alone.other_clients == 0
                assert alone.pending_ask is True

                # A second stream attaches: the prompt stays up for it, so
                # closing the first client needs no answer.
                second = client.attach(name)
                assert isinstance(
                    await asyncio.wait_for(anext(second), timeout=5), Snapshot,
                )
                shared = await client.quit_verdict(name)
                assert shared.ask is False and shared.other_clients == 1

                await second.aclose()
                await stream.aclose()
            finally:
                await client.aclose()
        finally:
            await host.request_shutdown()
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_headless_turn_runs_like_any_other_and_reattach_sees_it_done(tmp_path) -> None:
    """A turn is the host's, not the client's: it runs with zero subscribers,
    and a later attach sees the finished transcript and an idle status."""
    from xarness.clientlib import XarnessClient

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            name = await _make_session(
                host, tmp_path, [[ContentDelta(text="headless answer"), RoundComplete()]],
            )
            session = host.sessions[name]
            assert session.clients_attached == 0
            await session.execute(SendMessage(text="hi"))
            await asyncio.wait_for(session._turn_task, timeout=5)
            assert session.state == "idle"

            client = await XarnessClient.connect_unix(host.unix_socket, spawn=False)
            try:
                first = await asyncio.wait_for(
                    anext(client.attach(name)), timeout=5,
                )
            finally:
                await client.aclose()
            assert isinstance(first, Snapshot)
            assert first.status.state == "idle"
            assert [m.role for m in first.messages] == ["user", "assistant"]
            assert first.messages[1].content == "headless answer"
        finally:
            await host.request_shutdown()
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_default_preset_resolves_model(tmp_path) -> None:
    """The default preset picks the model used when none is requested."""
    from xarness.config import load_sectioned
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "models": {
            "local": {"base_url": "http://127.0.0.1:8080/v1",
                      "api_key_env": None, "model_id": "qwen"},
            "glm-flash": {"base_url": "https://openrouter.ai/api/v1",
                          "api_key_env": "OPENROUTER_API_KEY",
                          "model_id": "z-ai/glm-5.3-flash"},
        },
        "presets": {"default": {"model": "glm-flash"}},
    }))
    sectioned = load_sectioned(config_path)
    name, profile = sectioned.resolve_model(None)
    assert name == "glm-flash" and profile.model_id == "z-ai/glm-5.3-flash"


def test_tui_end_to_end_draft_flow(tmp_path) -> None:
    """The full stack: TUI (pilot) -> clientlib -> host API -> Session turn."""
    from xarness.clientlib import XarnessClient
    from xarness.tui.app import AgentApp

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            (tmp_path / "ws").mkdir(exist_ok=True)
            client = await XarnessClient.connect_unix(host.unix_socket, spawn=False)
            # Inject the scripted provider the moment the draft flow creates
            # the session (before the first message is sent).
            orig_create = client.create_session

            async def create_session(request):
                summary = await orig_create(request)
                host.sessions[summary.name]._client = FakeClient(
                    [[ContentDelta(text="Hello!"), RoundComplete(usage=Usage(input_tokens=10, output_tokens=2))]]
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
                assert app.view_status.state == "idle"
                # No local echo: the user message arrived via the stream.
                from xarness.tui import flow as tflow
                users = [e for e in app._flow_model.entries
                         if isinstance(e.item, tflow.UserItem)]
                answers = [e for e in app._flow_model.entries
                           if isinstance(e.item, tflow.AssistantItem)]
                assert len(users) == 1
                assert len(answers) == 1
                assert not [e for e in app._flow_model.entries
                            if isinstance(e.item, tflow.LineItem)
                            and e.item.kind == "error"]
            await client.aclose()
        finally:
            await host.request_shutdown()
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_tui_sessions_picker_follows_the_pushed_list(tmp_path) -> None:
    """End to end: an open /sessions picker is fed by GET /v1/events, so a
    session created by anyone (another client, a subagent) shows up in the
    modal as it happens, with no listing poll."""
    from textual.widgets import OptionList

    from xarness.clientlib import XarnessClient
    from xarness.tui.app import AgentApp

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            await _make_session(host, tmp_path, [[RoundComplete()]])
            client = await XarnessClient.connect_unix(
                host.unix_socket, spawn=False,
            )
            app = AgentApp(client, workspace=tmp_path / "ws")
            async with app.run_test(size=(100, 40)) as pilot:
                await pilot.pause()
                await app._open_sessions_async()
                await pilot.pause()
                picker = app.screen
                option_list = picker.query_one("#picker-list", OptionList)

                def rows() -> list[str]:
                    return [
                        str(option_list.get_option_at_index(index).prompt)
                        for index in range(option_list.option_count)
                    ]

                # The picker opened on the listing fetch; its stream then
                # re-snapshots the same one row.
                self_rows = [row for row in rows() if row.strip()]
                assert len(self_rows) == 1, rows()
                assert "idle" in self_rows[0]

                # A second session appears in the same workspace: the host
                # pushes its summary, the open modal gains the row.
                await host.create_session(protocol.CreateSessionRequest(
                    workspace=str(tmp_path / "ws")))
                for _ in range(50):
                    await asyncio.sleep(0.02)
                    await pilot.pause()
                    if len([row for row in rows() if row.strip()]) == 2:
                        break
                assert len([row for row in rows() if row.strip()]) == 2, rows()
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


def test_cli_overrides_beat_the_sandbox_profile(tmp_path) -> None:
    """--sandbox true/false, --network, --gpu and --enable_tracking are
    tri-state overrides: a given flag wins over the profile, an absent one
    (None) leaves the profile's value alone."""

    async def go():
        (tmp_path / "ws").mkdir()
        host = make_host(tmp_path)

        session = await host.create_session(protocol.CreateSessionRequest(
            workspace=str(tmp_path / "ws"),
            use_sandbox=False, network=False, gpu=True, enable_tracking=False,
        ))
        settings = session.sandbox.settings
        assert settings is not None
        assert settings.use_sandbox is False
        assert settings.allow_network is False
        assert settings.gpu_access is True
        assert settings.enable_tracking is False
        # Tracking off means no git block at all (no checkpoints/undo).
        assert session.workspace_state is None
        assert session.status().use_sandbox is False
        assert session.status().enable_tracking is False

        # No overrides: the profile's values (the config defaults) apply.
        plain = await host.create_session(protocol.CreateSessionRequest(
            workspace=str(tmp_path / "ws"),
        ))
        plain_settings = plain.sandbox.settings
        assert plain_settings is not None
        assert plain_settings.use_sandbox is True
        assert plain_settings.allow_network is True
        assert plain_settings.gpu_access is False
        assert plain_settings.enable_tracking is True

    asyncio.run(go())


def test_load_session_uses_saved_workspace_sandbox_and_baseline(tmp_path, monkeypatch) -> None:
    """Loading a saved session must honor its saved workspace (never the
    host's cwd) and its saved sandbox name/settings."""
    from xarness.config import SandboxSettings

    async def go():
        host = make_host(tmp_path)
        (tmp_path / "ws").mkdir()
        (tmp_path / "elsewhere").mkdir()
        summary = await host.create_session(protocol.CreateSessionRequest(
            workspace=str(tmp_path / "ws"), no_init_repo=False,
        ))
        name = summary.name
        session = host.sessions[name]
        assert session.workspace_state is not None
        saved_tree = session.workspace_state.baseline_tree
        # A distinct saved sandbox state.
        session.sandbox.name = "default"
        session.sandbox.settings = SandboxSettings(allow_git_write=True)
        session._persist()
        await host.unload_session(name)

        # Load from a different host cwd: the saved workspace must win.
        monkeypatch.chdir(tmp_path / "elsewhere")
        loaded = await host.load_session(name)
        assert loaded.workspace == (tmp_path / "ws").resolve()
        assert loaded.sandbox.name == "default"
        assert loaded.sandbox.settings is not None
        assert loaded.sandbox.settings.allow_git_write is True
        assert loaded.workspace_state is not None
        assert loaded.workspace_state.root == (tmp_path / "ws").resolve()
        # The SAVED git block won: the baseline is the session's own, not a
        # fresh snapshot of the (possibly changed) workspace.
        assert loaded.workspace_state.baseline_tree == saved_tree
        if loaded.sandbox.config is not None:
            assert loaded.sandbox.config.workspace == (tmp_path / "ws").resolve()
        return host

    asyncio.run(go())


def test_host_stays_alive_while_a_client_is_attached(tmp_path, monkeypatch) -> None:
    """An open event stream counts as active: the host must not exit during
    the grace window while a client is attached to a session, even with no
    turn running."""
    monkeypatch.setenv("XARNESS_IDLE_TTL_SECONDS", "0")

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            name = await _make_session(host, tmp_path, [[ContentDelta(text="x"), RoundComplete()]])
            session = host.sessions[name]
            session._subscribers.append(asyncio.Queue())
            await asyncio.sleep(host_mod.GRACE_SECONDS + 0.1)
            assert not task.done()
            session._subscribers.clear()
            await asyncio.wait_for(task, timeout=host_mod.GRACE_SECONDS + 10)
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


def test_web_listen_generates_a_token_only_off_loopback() -> None:
    """A listener reachable from the LAN must demand a token: one is made
    when none was given, and an explicit one is kept as-is."""
    assert host_mod.web_listen().token is None
    generated = host_mod.web_listen(bind="0.0.0.0", port=9000)
    assert generated.token and generated.port == 9000
    assert host_mod.web_listen(bind="0.0.0.0", token="s3cret").token == "s3cret"
    assert host_mod.web_listen(bind="127.0.0.1", token="s3cret").token == "s3cret"


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
        await host.create_session(protocol.CreateSessionRequest(workspace=str(tmp_path / "ws")))
        name = next(iter(host.sessions))
        host.sessions[name]._persist()
        await host.unload_session(name)
        # Listing and last-used lookups must not parse full conversations.
        def explode(*args, **kwargs):
            raise AssertionError("full conversation parsed")
        monkeypatch.setattr(session_store, "load_state", explode)
        monkeypatch.setattr(session_store, "load_meta", explode)

        summaries = await host.list_sessions()
        saved = [s for s in summaries if s.name == name and not s.active]
        assert saved and saved[0].workspace
        assert saved[0].model == "test"
        last_model, last_sandbox = host._workspace_last_used(
            (tmp_path / "ws").resolve(),
        )
        assert last_model == "test"
        assert any(
            info.session_count == 1
            for info in host.workspaces()
            if info.path == str((tmp_path / "ws").resolve())
        )

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


def test_config_defaults_endpoint_is_host_owned(tmp_path) -> None:
    """The client asks the host to persist defaults; it never writes the
    config file itself."""
    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            async with _unix_http(host.unix_socket) as http:
                response = await http.put(
                    "/v1/config/defaults", json={"default_profile": "test"},
                )
                assert response.status_code == 200, response.text
                assert response.json()["default_model"] == "test"
            saved = json.loads(host.config_path.read_text())
            assert saved["presets"]["default"]["model"] == "test"
        finally:
            await host.request_shutdown()
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_idle_session_is_released_and_listed_as_saved(tmp_path, monkeypatch) -> None:
    """Send a message, let the turn finish, close the stream: the session
    leaves host.sessions and /sessions lists it as saved."""
    monkeypatch.setenv("XARNESS_IDLE_TTL_SECONDS", "0")

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            name = await _make_session(
                host, tmp_path, [[ContentDelta(text="answer"), RoundComplete()]],
            )
            session = host.sessions[name]
            snapshot, queue = session.subscribe()  # a client attaches
            await session.execute(SendMessage(text="hi"))
            await asyncio.wait_for(session._turn_task, timeout=5)
            assert session.summary().active
            # The client closes the stream.
            session.unsubscribe(queue)

            async def released() -> None:
                for _ in range(100):
                    if name not in host.sessions:
                        return
                    await asyncio.sleep(0.05)

            await released()
            assert name not in host.sessions, "idle session must be released"
            # It is saved, and /sessions lists it as saved (not active).
            assert name in session_store.load_metas()
            summaries = await host.list_sessions()
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


def test_session_listing_takes_a_range(tmp_path) -> None:
    """``range`` slices the recency-sorted listing; a malformed window is a
    400 naming the syntax instead of a traceback."""
    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            async with _unix_http(host.unix_socket) as http:
                bad = await http.get("/v1/sessions", params={"range": "nope"})
                assert bad.status_code == 400
                assert "range" in bad.json()["detail"]
                rows = await http.get("/v1/sessions", params={"range": "[:]"})
                assert rows.status_code == 200 and rows.json() == []
        finally:
            await host.request_shutdown()
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_concurrent_load_yields_one_session(tmp_path) -> None:
    """Two simultaneous loads of the same saved session must share one
    Session object (no duplicate sandbox / tracking)."""
    async def go():
        host = make_host(tmp_path)
        (tmp_path / "ws").mkdir()
        summary = await host.create_session(protocol.CreateSessionRequest(workspace=str(tmp_path / "ws")))
        name = summary.name
        await host.unload_session(name)

        first, second = await asyncio.gather(
            host.load_session(name), host.load_session(name),
        )
        assert first is second
        assert host.sessions[name] is first
    asyncio.run(go())


def test_unload_closes_sandbox_session(tmp_path) -> None:
    """Unloading/evicting a session must close its sandbox shell."""
    async def go():
        host = make_host(tmp_path)
        (tmp_path / "ws").mkdir()
        summary = await host.create_session(protocol.CreateSessionRequest(workspace=str(tmp_path / "ws")))
        session = host.sessions[summary.name]

        class FakeShell:
            closed = False
            async def close(self) -> None:
                self.closed = True

        shell = FakeShell()
        session.sandbox.shell = shell
        await host.unload_session(summary.name)
        assert shell.closed
    asyncio.run(go())


def test_loading_a_session_reuses_the_saved_baseline(tmp_path) -> None:
    """Resume must reuse the workspace row: re-baselining on load would call
    everything saved "drift" and pay a full scan on every resume."""
    from unittest import mock

    async def go():
        host = make_host(tmp_path)
        (tmp_path / "ws").mkdir()
        (tmp_path / "ws" / "file.txt").write_text("hello\n")
        summary = await host.create_session(protocol.CreateSessionRequest(workspace=str(tmp_path / "ws")))
        name = summary.name
        await host.unload_session(name)

        baseline = host._workspace_states[(tmp_path / "ws").resolve()].baseline_tree
        with mock.patch.object(
            host_mod, "open_workspace",
            side_effect=AssertionError("resume re-opened the workspace"),
        ):
            loaded = await host.load_session(name)
        assert loaded.workspace_state is not None
        assert loaded.workspace_state.baseline_tree == baseline
    asyncio.run(go())


def test_subagent_names_are_parent_scoped(tmp_path, monkeypatch) -> None:
    """Children are ``<parent>/agent-N``: the same short name can appear
    under different parents, and a restarted host continues the count."""
    host = make_host(tmp_path)
    parent = "550e8400e29b41d4a716446655440000"
    other = "6ba7b8109dad11d180b400c04fd430c8"
    assert host._next_agent_name(parent) == f"{parent}/agent-1"
    assert host._next_agent_name(parent) == f"{parent}/agent-2"
    assert host._next_agent_name(other) == f"{other}/agent-1"

    restarted = make_host(tmp_path)
    monkeypatch.setattr(
        session_store, "load_metas", lambda: {f"{parent}/agent-2": {}},
    )
    assert restarted._next_agent_name(parent) == f"{parent}/agent-3"


def test_daemon_import_does_not_pull_in_textual() -> None:
    """Importing the host (the daemon) must not import the TUI stack."""
    import subprocess
    import sys

    code = (
        "import sys, xarness.host; "
        "assert 'textual' not in sys.modules, sorted(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_session_listing_counts_a_subagents_work_as_its_parents() -> None:
    """The /sessions +/- totals follow the parent link: a child's pending
    work is part of the row for the session that spawned it."""
    from xarness.host import Host

    children = {"parent": ["child"], "child": ["grandchild"]}
    assert Host._subtree("parent", children) == {
        "parent", "child", "grandchild",
    }
    assert Host._subtree("child", children) == {"child", "grandchild"}
    assert Host._subtree("other", children) == {"other"}


def test_session_listing_stats_fill_pending_diff_totals(tmp_path, monkeypatch) -> None:
    """A listing row's totals are its own ``mine`` delta, not the workspace's,
    and zero rather than an error when a stat cannot be computed."""

    from xarness import gitwork
    from xarness.gitwork import DiffFile, DiffStat
    from xarness.workspace import WorkspaceState

    async def fake_observe():
        return "current"

    async def fake_ignored():
        return ()

    def fake_state(name: str):
        state = WorkspaceState(
            root=tmp_path, git_dir=tmp_path / ".git", baseline_tree="x",
            persist=False,
        )
        state.observe = fake_observe
        state.ignored = fake_ignored
        state.record(
            session=name, tool="write_file", turn="t",
            paths=("mine.py",), auto=(),
        )
        state.record(
            session="other-session", tool="write_file", turn="t",
            paths=("other.py",), auto=(),
        )
        state.record(
            session=name, tool="run_bash", turn="t",
            paths=("bash.py",), auto=(),
        )
        state.record(
            session=name, tool="edit_file", turn="t",
            paths=("mine2.py",), auto=(),
        )
        return state

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            name = await _make_session(host, tmp_path, [[RoundComplete()]])
            session = host.sessions[name]

            # No git info: totals stay zero, listing still works.
            session.workspace_state = None
            await host.refresh_stats()
            rows = await host.list_sessions(stats=True)
            row = next(r for r in rows if r.name == name)
            assert (row.additions, row.deletions, row.tracking) == (0, 0, False)

            # With a workspace, only the session's own explicit writes count:
            # another session's write, this session's bash change and
            # uncategorised drift are all excluded.
            files = [
                DiffFile(path="mine.py", additions=7, deletions=2),
                DiffFile(path="mine2.py", additions=3, deletions=1),
                DiffFile(path="other.py", additions=100, deletions=100),
                DiffFile(path="bash.py", additions=100, deletions=100),
                DiffFile(path="drift.py", additions=100, deletions=100),
            ]

            async def fake_diff_stat(repo, base, other):
                return DiffStat(
                    files=list(files),
                    additions=sum(f.additions for f in files),
                    deletions=sum(f.deletions for f in files),
                )

            monkeypatch.setattr(gitwork, "tree_diff_stat", fake_diff_stat)
            session.workspace_state = fake_state(name)
            await host.refresh_stats()
            rows = await host.list_sessions(stats=True)
            row = next(r for r in rows if r.name == name)
            assert (row.additions, row.deletions) == (10, 3)
            assert row.tracking is True

            # A failing stat keeps the last cached totals and never breaks
            # the listing.
            async def broken_diff_stat(repo, base, other):
                raise RuntimeError("git blew up")

            monkeypatch.setattr(gitwork, "tree_diff_stat", broken_diff_stat)
            await host.refresh_stats()
            rows = await host.list_sessions(stats=True)
            row = next(r for r in rows if r.name == name)
            assert (row.additions, row.deletions) == (10, 3)

            # Saved (not hosted) sessions get no totals at all: the
            # persisted numbers would be stale, so they stay blank.
            monkeypatch.setattr(gitwork, "tree_diff_stat", fake_diff_stat)
            session.workspace_state = fake_state(name)
            session._persist()
            await host.unload_session(name)
            assert name not in host.sessions
            rows = await host.list_sessions(stats=True)
            row = next(r for r in rows if r.name == name)
            assert (row.additions, row.deletions) == (0, 0)
            # ...and load_state was never called for the listing.
            monkeypatch.setattr(
                session_store, "load_state",
                lambda *a, **kw: (_ for _ in ()).throw(
                    AssertionError("full conversation parsed")),
            )
            assert await host.list_sessions(stats=True)
        finally:
            await host.request_shutdown()
            if not task.done():
                task.cancel()

    asyncio.run(go())


def test_hosting_a_session_schedules_its_totals_refresh(
    tmp_path, monkeypatch,
) -> None:
    """A session entering the host has no computed totals yet, so the list
    is told to recompute at once: a row that showed numbers as saved must not
    drop to +0 -0 the moment it is opened."""

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            monkeypatch.setattr(host_mod, "STATS_DEBOUNCE_SECONDS", 0.01)
            refreshed: list[str] = []

            async def fake_refresh(state):
                refreshed.append(str(state.root))

            monkeypatch.setattr(host, "_refresh_stats", fake_refresh)
            host.watch_sessions()  # someone is looking at the list
            (tmp_path / "ws").mkdir()
            summary = await host.create_session(protocol.CreateSessionRequest(
                workspace=str(tmp_path / "ws"), enable_tracking=True,
            ))
            await asyncio.sleep(0.1)
            assert refreshed, "creating a session did not schedule a refresh"

            refreshed.clear()
            await host.unload_session(summary.name)
            await host.load_session(summary.name)
            await asyncio.sleep(0.1)
            assert refreshed, "loading a session did not schedule a refresh"
        finally:
            await host.request_shutdown()
            if not task.done():
                await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_the_listing_never_runs_git(tmp_path) -> None:
    """GET /v1/sessions is a pure read: hosted summaries come from memory,
    saved rows from the metas, and the +/- totals from the cache. A poll
    used to run one worktree snapshot and N diffs."""

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            name = await _make_session(host, tmp_path, [[RoundComplete()]])
            session = host.sessions[name]

            async def no_git(*args, **kwargs):
                raise AssertionError("the listing ran git")

            state = WorkspaceState(
                root=tmp_path, git_dir=tmp_path / ".git", baseline_tree="x",
                persist=False,
            )
            state.observe = no_git
            state.diff = no_git
            session.workspace_state = state
            session.pending_additions, session.pending_deletions = 4, 2

            rows = await host.list_sessions(stats=True)
            row = next(r for r in rows if r.name == name)
            assert row.tracking is True
            assert (row.additions, row.deletions) == (4, 2)
            assert row.hosted is True
        finally:
            await host.request_shutdown()
            if not task.done():
                await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_a_saved_sessions_row_is_computed_without_hosting_it(
    tmp_path, monkeypatch,
) -> None:
    """A saved row's own totals come from the workspace's saved baseline.

    The listing never runs git, so a client that wants the numbers for one
    saved row asks for that row: the host computes one diff scoped to this
    session (and its saved subagents) instead of hosting it. Hovering must
    never fill the server view with idle sessions nobody opened."""

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            name = await _make_session(host, tmp_path, [[RoundComplete()]])
            await host.unload_session(name)
            assert name not in host.sessions

            calls: list[str] = []

            async def fake_observe() -> str:
                return "current"

            async def fake_diff(
                viewer=None, current=None, path=None, scope="all",
            ):
                calls.append(scope)
                return protocol.DiffInfo(tracking=True, files=[
                    protocol.DiffFileInfo(
                        path="mine.py", additions=7, deletions=2, author=name,
                    ),
                    protocol.DiffFileInfo(
                        path="other.py", additions=100, deletions=100,
                        author="someone-else",
                    ),
                ])

            state = WorkspaceState(
                root=tmp_path / "ws", git_dir=tmp_path / ".git",
                baseline_tree="x", persist=False,
            )
            state.observe = fake_observe
            state.diff = fake_diff
            opened: list[Path] = []

            async def fake_open(workspace, **kwargs):
                opened.append(workspace)
                return state

            monkeypatch.setattr(host, "open_workspace", fake_open)
            monkeypatch.setattr(
                host_mod, "has_workspace_state", lambda workspace: True,
            )

            row = await host.session_summary(name)
            assert row is not None
            assert row.hosted is False and row.state == "saved"
            assert row.tracking is True
            # Only this session's rows, not the rest of the workspace's.
            assert (row.additions, row.deletions) == (7, 2)
            assert calls == ["mine"]
            assert opened  # the saved baseline was read…
            assert name not in host.sessions  # …and nothing was hosted

            # A second peek within the TTL reuses the diff: no new snapshot.
            calls.clear()
            again = await host.session_summary(name)
            assert (again.additions, again.deletions) == (7, 2)
            assert calls == []

            # A hosted session answers from memory (its totals are pushed).
            hosted = await host.load_session(name)
            hosted.pending_additions, hosted.pending_deletions = 3, 1
            row = await host.session_summary(name)
            assert row.hosted is True
            assert (row.additions, row.deletions) == (3, 1)

            assert await host.session_summary("nope") is None
        finally:
            await host.request_shutdown()
            if not task.done():
                await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_session_summary_route_does_not_load_the_session(tmp_path) -> None:
    """GET /v1/sessions/{name}/summary is the read-only row a hover makes:
    it answers for a saved session and leaves it saved."""

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            name = await _make_session(host, tmp_path, [[RoundComplete()]])
            await host.unload_session(name)
            async with _unix_http(host.unix_socket) as http:
                response = await http.get(f"/v1/sessions/{name}/summary")
                assert response.status_code == 200, response.text
                row = response.json()
                assert row["name"] == name
                assert row["hosted"] is False and row["state"] == "saved"
                assert name not in host.sessions
                missing = await http.get("/v1/sessions/nope/summary")
                assert missing.status_code == 404
        finally:
            await host.request_shutdown()
            if not task.done():
                await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_watch_sessions_pushes_the_list_snapshot_then_deltas(tmp_path) -> None:
    """GET /v1/events: one snapshot, then the rows that changed — no poll.

    A create, an attach (which changes ``clients_attached``) and an unload
    each reach the client as they happen."""
    from xarness.clientlib import XarnessClient

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            client = await XarnessClient.connect_unix(
                host.unix_socket, spawn=False,
            )
            try:
                stream = client.watch_sessions()
                first = await asyncio.wait_for(anext(stream), timeout=5)
                assert isinstance(first, protocol.SessionsSnapshot)
                assert first.sessions == []

                name = await _make_session(host, tmp_path, [[RoundComplete()]])
                added = await asyncio.wait_for(anext(stream), timeout=5)
                assert isinstance(added, protocol.SessionSummaryChanged)
                assert added.summary.name == name
                assert added.summary.hosted is True

                # Attaching is a state change too (clients_attached).
                session = host.sessions[name]
                queue = session.subscribe()[1]
                changed = await asyncio.wait_for(anext(stream), timeout=5)
                assert isinstance(changed, protocol.SessionSummaryChanged)
                assert changed.summary.clients_attached == 1
                session.unsubscribe(queue)
                left = await asyncio.wait_for(anext(stream), timeout=5)
                assert isinstance(left, protocol.SessionSummaryChanged)
                assert left.summary.clients_attached == 0

                await host.unload_session(name)
                released = await asyncio.wait_for(anext(stream), timeout=5)
                assert isinstance(released, protocol.SessionSummaryChanged)
                assert released.summary.name == name
                assert released.summary.hosted is False
                assert released.summary.state == "saved"
                # It is still on disk, so it is still listed as a saved row.
                listed = await host.list_sessions()
                assert [(r.name, r.hosted) for r in listed] == [(name, False)]

                # Deleting the saved file is the one thing that removes the
                # row for good.
                assert (await host.delete_session(name)).deleted
                gone = await asyncio.wait_for(anext(stream), timeout=5)
                assert isinstance(gone, protocol.SessionsRemoved)
                assert gone.names == [name]
                assert await host.list_sessions() == []
            finally:
                await client.aclose()
        finally:
            await host.request_shutdown()
            if not task.done():
                await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_creating_and_loading_a_session_never_waits_for_mcp(
    tmp_path, monkeypatch,
) -> None:
    """A broken MCP server can take its whole startup timeout: neither the
    create/load response nor the attach snapshot may wait for it."""
    from xarness.mcp import McpManager
    from xarness.session import Session

    release = asyncio.Event()

    async def handshake(self) -> None:
        await release.wait()

    monkeypatch.setattr(Session, "prepare_mcp", handshake)
    monkeypatch.setattr(
        McpManager, "has_enabled_servers", property(lambda self: True),
    )

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            summary = await asyncio.wait_for(
                host.create_session(protocol.CreateSessionRequest(
                    workspace=str(tmp_path / "ws"))),
                timeout=5,
            )
            session = host.sessions[summary.name]
            assert session._mcp_task is not None
            assert not session._mcp_task.done()

            name = summary.name
            await asyncio.wait_for(host.unload_session(name), timeout=5)
            loaded = await asyncio.wait_for(host.load_session(name), timeout=5)
            assert loaded.name == name
            assert loaded._mcp_task is not None
            assert not loaded._mcp_task.done()
        finally:
            await host.request_shutdown()
            if not task.done():
                await asyncio.wait_for(task, timeout=5)

    asyncio.run(go())


def test_system_prompt_matches_tracking_status(tmp_path) -> None:
    """End-to-end: the system prompt states exactly the facts the status
    payload claims — the snapshot clause, git read-only, the PLAN label."""

    async def go():
        host = make_host(tmp_path)
        task = await _serve(host)
        try:
            (tmp_path / "ws").mkdir()
            session = await host.create_session(protocol.CreateSessionRequest(
                workspace=str(tmp_path / "ws")))
            status = session.status()
            prompt = session.system_prompt
            assert status.enable_tracking is (session.workspace_state is not None)
            if status.enable_tracking:
                assert "Changes are snapshotted" in prompt
                assert "git in the shell is read-only" in prompt
            else:
                assert "Changes are snapshotted" not in prompt
            assert "Mode: WRITE (read/write)" in prompt
            # A fresh session reports tracking on (the workspace is auto-inited).
            assert status.enable_tracking is True
        finally:
            task.cancel()

    asyncio.run(go())


def test_tracking_failure_does_not_block_session_start(tmp_path) -> None:
    """A workspace that cannot be snapshotted must not block session start."""
    from unittest import mock

    async def go():
        host = make_host(tmp_path)
        (tmp_path / "ws").mkdir()
        with mock.patch.object(
            host_mod, "open_workspace",
            side_effect=OSError(7, "Argument list too long"),
        ):
            session = await host.create_session(
                protocol.CreateSessionRequest(workspace=str(tmp_path / "ws")),
            )
        assert session.workspace_state is None
        assert session.status().enable_tracking is False

    asyncio.run(go())


def test_unexpected_errors_name_themselves() -> None:
    """An unhandled exception must name itself, not a bare 500."""
    from xarness.api import create_app

    class ExplodingHost:
        async def create_session(self, request):
            raise MemoryError("out of memory")

    app = create_app(ExplodingHost())

    async def go():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test",
        ) as client:
            return await client.post("/v1/sessions", json={})

    response = asyncio.run(go())
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "MemoryError" in detail
    assert "out of memory" in detail
