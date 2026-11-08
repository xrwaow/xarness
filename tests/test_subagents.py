"""Subagent tests: the tool, the child session, cancellation and the model key.

The child is a real session in a real (temporary) workspace; only the
provider is faked, by replacing the client the controller would build.
"""

import asyncio
import json
from pathlib import Path

import pytest

from xarness import controller as controller_mod
from xarness import host as host_mod
from xarness import protocol, session_store
from xarness.config import ConfigError, load_sectioned
from xarness.errors import SessionError
from xarness.host import Host
from xarness.protocol import (
    ContentDelta,
    ProcessingStarted,
    RoundComplete,
    ToolCallArgumentsDone,
    ToolCallStarted,
    Usage,
)


class FakeClient:
    """Replays scripted rounds; records whether it was closed."""

    def __init__(self, script: list[list]) -> None:
        self.script = list(script)
        self.closed = False

    async def stream(self, wire_messages, tools=None):
        yield ProcessingStarted()
        for event in (self.script.pop(0) if self.script else []):
            yield event

    async def aclose(self) -> None:
        self.closed = True


class GatedClient(FakeClient):
    """A first round that blocks until the test releases it, so the child
    session can be inspected while it is still running."""

    def __init__(self, script: list[list], gate: asyncio.Event) -> None:
        super().__init__(script)
        self.gate = gate

    async def stream(self, wire_messages, tools=None):
        await self.gate.wait()
        async for event in super().stream(wire_messages, tools):
            yield event


def _patch_clients(monkeypatch, *scripts) -> list[FakeClient]:
    """Hand the next scripted client to every controller built from now on
    (the parent's first, then the child's)."""
    queue = [FakeClient(script) for script in scripts]

    def factory(profile, api_key):
        return queue.pop(0) if queue else FakeClient([])

    monkeypatch.setattr(controller_mod, "ChatClient", factory)
    return queue


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(host_mod, "HOST_DIR", tmp_path / "host")
    monkeypatch.setenv("XARNESS_TEST_KEY", "test-key")
    return _write_config(tmp_path)


def _write_config(tmp_path: Path, **preset) -> Path:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "models": {
            "test": {
                "base_url": "https://api.example.test/v1",
                "api_key_env": "XARNESS_TEST_KEY",
                "model_id": "test-model",
            },
            "small": {
                "base_url": "https://api.example.test/v1",
                "api_key_env": "XARNESS_TEST_KEY",
                "model_id": "small-model",
            },
        },
        "sandboxes": {"default": {"use_sandbox": False}},
        "presets": {
            "default": {"model": "test", "sandbox": "default", **preset},
        },
        "defaults": {"preset": "default"},
    }))
    return config_path


def _host(tmp_path: Path) -> Host:
    return Host(
        config_path=tmp_path / "config.json",
        unix_socket=tmp_path / "host" / "test.sock",
    )


def _round(text: str = "ok") -> list:
    return [ContentDelta(text=text),
            RoundComplete(usage=Usage(input_tokens=1, output_tokens=1))]


def _call_round(name: str, arguments: dict, call_id: str = "c1") -> list:
    return [
        ToolCallStarted(call_id=call_id, name=name),
        ToolCallArgumentsDone(
            call_id=call_id, name=name, arguments_json=json.dumps(arguments),
        ),
        RoundComplete(has_tool_calls=True),
    ]


def _subagent_round(prompt="do the thing", mode="write") -> list:
    return _call_round("subagent", {"prompt": prompt, "mode": mode})


async def _until(predicate, timeout: float = 5.0) -> None:
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never became true")


async def _parent(host: Host, tmp_path: Path, **request):
    (tmp_path / "ws").mkdir(exist_ok=True)
    return await host.create_session(protocol.CreateSessionRequest(
        workspace=str(tmp_path / "ws"), **request,
    ))


def _tool_result(session, call_id: str = "c1"):
    return next(
        m for m in session.conversation.messages
        if m.role == "tool" and m.tool_call_id == call_id
    )


def test_write_parent_gets_the_childs_last_non_empty_reply(
    tmp_path, monkeypatch,
) -> None:
    """The tool returns the final assistant message, not the tool-call-only
    one before it; the child's edit is recorded as that child's event."""
    _patch_clients(
        monkeypatch,
        # parent: spawn, then answer
        [_subagent_round(), _round("parent done")],
        # child: one tool round (empty text) + the real answer
        [_call_round("write_file", {"path": "a.txt", "content": "hi"}),
         _round("child answer")],
    )

    async def go():
        host = _host(tmp_path)
        parent = await _parent(host, tmp_path)
        await parent.execute("s1", protocol.SendMessage(text="go"))
        await parent.wait_idle()
        assert _tool_result(parent).status == "ok"
        assert _tool_result(parent).content == "child answer"
        # The child closed, and its edit is a pending event of its own
        # session — the parent sees it as another agent's work.
        assert (tmp_path / "ws" / "a.txt").read_text() == "hi"
        stat = await parent.diff_info(None)
        assert {f.path: f.kind for f in stat.files} == {"a.txt": "agent"}
        events = parent.workspace_state.events
        assert [(event.tool, event.paths) for event in events] \
            == [("write_file", ("a.txt",))]
        assert events[0].session != parent.name

    asyncio.run(go())


def test_child_session_shape(tmp_path, monkeypatch) -> None:
    """The live child: parent_id set, same workspace state, no subagent tool
    and no ask_user, and the requested mode."""
    gate = asyncio.Event()
    # Replace the child's client with a gated one (the parent takes the
    # first script).
    queue = [FakeClient([_subagent_round(mode="plan")]), GatedClient([], gate)]
    monkeypatch.setattr(
        controller_mod, "ChatClient",
        lambda profile, api_key: queue.pop(0) if queue else FakeClient([]),
    )

    async def go():
        host = _host(tmp_path)
        parent = await _parent(host, tmp_path)
        await parent.execute("s1", protocol.SendMessage(text="go"))
        await _until(lambda: len(host.sessions) == 2)
        child = next(s for s in host.sessions.values() if s is not parent)
        assert child.name == f"{parent.name}/agent-1"
        assert child.parent_id == parent.name
        assert child.workspace_state is parent.workspace_state
        assert child.workspace == parent.workspace
        assert child.mode == "plan"
        names = [t.name for t in child.tool_registry.tools()]
        assert "subagent" not in names, names
        assert "ask_user" not in names, names
        assert "read_file" in names, names
        # The parent, by contrast, has both.
        parent_names = [t.name for t in parent.tool_registry.tools()]
        assert "subagent" in parent_names and "ask_user" in parent_names
        gate.set()
        await parent.wait_idle()
        assert [s.name for s in host.sessions.values()] == [parent.name]

    asyncio.run(go())


def test_plan_parent_cannot_spawn_a_write_child(tmp_path, monkeypatch) -> None:
    """A plan-mode agent is read-only itself: a write child is refused, and
    no child session is created."""
    _patch_clients(
        monkeypatch,
        [_subagent_round(mode="write"), _round("parent done")],
    )

    async def go():
        host = _host(tmp_path)
        parent = await _parent(host, tmp_path)
        await parent.execute("m1", protocol.SetMode(mode="plan"))
        await parent.execute("s1", protocol.SendMessage(text="go"))
        await parent.wait_idle()
        result = _tool_result(parent)
        assert result.status == "parse_error"
        assert "only spawn plan subagents" in result.content
        assert [s.name for s in host.sessions.values()] == [parent.name]
        # And the tool offers no write children at all: its description never
        # mentions one and a plan child is the default.
        tool = next(t for t in parent.tool_registry.tools() if t.name == "subagent")
        assert "mode='write'" not in tool.description
        assert tool.schema["properties"]["mode"]["default"] == "plan"

    asyncio.run(go())


def test_child_inherits_no_session(tmp_path, monkeypatch) -> None:
    """--no-session: neither the parent nor its child writes to the store, so
    the child vanishes for good when it finishes."""
    gate = asyncio.Event()
    queue = [
        FakeClient([_subagent_round(), _round("parent done")]),
        GatedClient([_round("child answer")], gate),
    ]
    monkeypatch.setattr(
        controller_mod, "ChatClient",
        lambda profile, api_key: queue.pop(0) if queue else FakeClient([]),
    )

    async def go():
        host = _host(tmp_path)
        parent = await _parent(host, tmp_path, no_session=True)
        await parent.execute("s1", protocol.SendMessage(text="go"))
        await _until(lambda: len(host.sessions) == 2)
        child_name = next(
            name for name in host.sessions if name != parent.name
        )
        gate.set()
        await parent.wait_idle()
        assert _tool_result(parent).content == "child answer"
        assert parent.no_save
        # Nothing was saved for either the parent or its child — and the
        # store's directory was not even created.
        assert session_store.load_metas() == {}
        assert list(session_store.SESSIONS_DIR.glob("*")) == []
        # ...so the child is gone for good: it is not listed and cannot be
        # reopened (its state lived in this host process only).
        assert child_name not in [
            summary.name for summary in await host.list_sessions()
        ]
        with pytest.raises(SessionError, match="no saved session"):
            await host.load_session(child_name)

    asyncio.run(go())


def test_a_message_to_a_running_child_is_steering(
    tmp_path, monkeypatch,
) -> None:
    """A user message to a running child is queued steering, exactly like a
    normal session: it is injected at the next round boundary, so the child
    answers it inside the same turn and the parent's tool call returns the
    answer that came after it."""
    gate = asyncio.Event()
    child_client = GatedClient(
        [_call_round("write_file", {"path": "a.txt", "content": "hi"}),
         _round("after steer")],
        gate,
    )
    queue = [FakeClient([_subagent_round()]), child_client]
    monkeypatch.setattr(
        controller_mod, "ChatClient",
        lambda profile, api_key: queue.pop(0) if queue else FakeClient([]),
    )

    async def go():
        host = _host(tmp_path)
        parent = await _parent(host, tmp_path)
        await parent.execute("s1", protocol.SendMessage(text="go"))
        await _until(lambda: len(host.sessions) == 2)
        child = next(s for s in host.sessions.values() if s is not parent)
        sent = await child.execute(
            "u1", protocol.SendMessage(text="also this"),
        )
        assert sent.ok and sent.queued
        assert child.state == "running"
        gate.set()
        await parent.wait_idle()
        assert "also this" in [
            m.content for m in child.conversation.messages if m.role == "user"
        ]
        assert _tool_result(parent).content == "after steer"

    asyncio.run(go())


def test_a_finished_child_reopens_as_an_ordinary_session(
    tmp_path, monkeypatch,
) -> None:
    """A finished child is dropped from the host but stays saved: the user can
    reopen it from /sessions and drive it like any session. It keeps its
    parent link and its own-work label, and it does NOT get nesting or
    ask_user back."""
    _patch_clients(
        monkeypatch,
        [_subagent_round(), _round("parent done")],
        [_call_round("write_file", {"path": "a.txt", "content": "hi"}),
         _round("child answer")],
        [_round("driven by the user")],
    )

    async def go():
        host = _host(tmp_path)
        parent = await _parent(host, tmp_path)
        await parent.execute("s1", protocol.SendMessage(text="go"))
        await parent.wait_idle()
        assert [s.name for s in host.sessions.values()] == [parent.name]
        child_name = next(
            summary.name for summary in await host.list_sessions()
            if summary.name != parent.name
        )

        reopened = await host.load_session(child_name)
        assert reopened.parent_id == parent.name
        assert reopened.workspace_state is parent.workspace_state
        names = [t.name for t in reopened.tool_registry.tools()]
        assert "subagent" not in names, names
        assert "ask_user" not in names, names
        # /sessions still nests it under its parent.
        summaries = {s.name: s for s in await host.list_sessions()}
        assert summaries[child_name].parent_id == parent.name
        # Its own write is still its own: ``mine`` for this viewer.
        diff = await reopened.diff_info(None)
        assert {f.path: f.kind for f in diff.files} == {"a.txt": "mine"}
        # And it can be driven: a turn the user sends happens normally.
        await reopened.execute("u1", protocol.SendMessage(text="continue"))
        await reopened.wait_idle()
        assert reopened.conversation.messages[-1].content \
            == "driven by the user"

    asyncio.run(go())


def test_cancelling_the_parent_cancels_and_closes_the_child(
    tmp_path, monkeypatch,
) -> None:
    """The tool call is cancelled with the parent's turn, so the child (whose
    round never returns) is cancelled, closed and dropped from the host."""
    gate = asyncio.Event()
    child_client = GatedClient([], gate)
    queue = [FakeClient([_subagent_round()]), child_client]
    monkeypatch.setattr(
        controller_mod, "ChatClient",
        lambda profile, api_key: queue.pop(0) if queue else FakeClient([]),
    )

    async def go():
        host = _host(tmp_path)
        parent = await _parent(host, tmp_path)
        await parent.execute("s1", protocol.SendMessage(text="go"))
        await _until(lambda: len(host.sessions) == 2)
        child = next(s for s in host.sessions.values() if s is not parent)
        await parent.execute("c1", protocol.Cancel())
        await parent.wait_idle()
        assert child.state == "idle"
        assert child_client.closed, "the child's provider client was not closed"
        assert [s.name for s in host.sessions.values()] == [parent.name]
        # The parent's tool call was answered as interrupted, so its next
        # request stays valid.
        assert _tool_result(parent).status == "interrupted"

    asyncio.run(go())


@pytest.mark.parametrize(
    ("preset", "request_model", "expected"),
    [
        ({}, None, "test"),            # unset: the preset's own model
        ({}, "small", "test"),         # unset ignores a --model override
        ({"subagent_model": "same"}, "small", "small"),
        ({"subagent_model": "same"}, None, "test"),
        ({"subagent_model": "small"}, "test", "small"),
    ],
)
def test_subagent_model_resolution(
    tmp_path, monkeypatch, preset, request_model, expected,
) -> None:
    _write_config(tmp_path, **preset)
    _patch_clients(monkeypatch, [])

    async def go():
        host = _host(tmp_path)
        parent = await _parent(host, tmp_path, model=request_model)
        assert parent.subagent_model == expected
        await parent.shutdown()

    asyncio.run(go())


def test_invalid_subagent_model_fails_at_config_load(tmp_path) -> None:
    path = _write_config(tmp_path, subagent_model="not-a-model")
    with pytest.raises(ConfigError, match="subagent_model"):
        load_sectioned(path)


def test_subagent_uses_the_configured_model(tmp_path, monkeypatch) -> None:
    """The child is built with the preset's subagent model, not the parent's."""
    _write_config(tmp_path, subagent_model="small")
    gate = asyncio.Event()
    child_client = GatedClient([], gate)
    queue = [FakeClient([_subagent_round()]), child_client]
    monkeypatch.setattr(
        controller_mod, "ChatClient",
        lambda profile, api_key: queue.pop(0) if queue else FakeClient([]),
    )

    async def go():
        host = _host(tmp_path)
        parent = await _parent(host, tmp_path)
        await parent.execute("s1", protocol.SendMessage(text="go"))
        await _until(lambda: len(host.sessions) == 2)
        child = next(s for s in host.sessions.values() if s is not parent)
        assert child.profile_name == "small"
        assert child.profile.model_id == "small-model"
        assert parent.profile.model_id == "test-model"
        gate.set()
        await parent.wait_idle()

    asyncio.run(go())
