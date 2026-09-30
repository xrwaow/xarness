"""Tests for run_bash_host: the approval flow and the saved prefix rules."""

import asyncio
import json
import types

import pytest

from xarness.permissions import (
    Decision, HostBashRequest, add_prefix, is_compound, load_prefixes,
    prefix_matches, rules_path, tokens,
)
from xarness.tools import ToolRegistry, _make_run_bash_host_tool, build_registry


@pytest.fixture(autouse=True)
def _config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    return tmp_path / "cfg" / "xarness" / "host_bash.json"


def _sandbox(tmp_path) -> types.SimpleNamespace:
    # run_bash_host only reads workspace/subtree; no bwrap needed (nothing
    # here runs inside the sandbox).
    return types.SimpleNamespace(workspace=tmp_path, subtree="")


def _registry_for(tmp_path, decisions):
    """A registry whose approve callback replays ``decisions`` in order."""
    queue = list(decisions)

    async def approve(req: HostBashRequest) -> Decision:
        assert queue, "unexpected prompt"
        decision = queue.pop(0)
        if isinstance(decision, Exception):
            raise decision
        return decision

    registry = ToolRegistry()
    registry.register(_make_run_bash_host_tool(_sandbox(tmp_path), approve))
    return registry


def call(registry, args: str, sink=None):
    return asyncio.run(registry.call("run_bash_host", args, output_sink=sink))


def test_rules_path_respects_xdg_config_home(_config_home) -> None:
    assert rules_path() == _config_home


def test_add_prefix_persists_and_dedupes(_config_home) -> None:
    add_prefix(("uv", "run"))
    add_prefix(("uv", "run"))
    add_prefix(("cargo", "fetch"))
    assert load_prefixes() == (("uv", "run"), ("cargo", "fetch"))
    data = json.loads(_config_home.read_text())
    assert data["prefixes"] == [["uv", "run"], ["cargo", "fetch"]]


def test_load_prefixes_tolerates_missing_or_broken_file(tmp_path) -> None:
    (tmp_path / "xarness").mkdir()
    (tmp_path / "xarness" / "host_bash.json").write_text("not json{")
    assert load_prefixes() == ()


def test_prefix_match_is_token_based() -> None:
    prefixes = (("uv", "run"),)
    assert prefix_matches("uv run pytest", prefixes)
    assert not prefix_matches("uv runaway", prefixes)
    assert not prefix_matches("uv", prefixes)


def test_tokens_none_when_unparseable() -> None:
    assert tokens("echo 'unterminated") is None


def test_compound_detection() -> None:
    assert is_compound("ls | wc")
    assert is_compound("echo $(id)")
    assert is_compound("echo `id`")
    assert is_compound("echo hi > out")
    assert is_compound("echo hi < in")
    assert is_compound("echo a; echo b")
    assert is_compound("echo a & echo b")
    assert is_compound("echo hi\n")
    assert not is_compound("cargo fetch")


def test_host_tool_registered_only_in_write_mode_with_approve(tmp_path) -> None:
    from xarness.sandbox import SandboxConfig, SandboxSession

    try:
        sandbox = SandboxConfig(workspace=tmp_path)
    except Exception:  # pragma: no cover - bwrap missing; the test is moot
        pytest.skip("bwrap not available")
    session = SandboxSession(sandbox)  # not started; construction is cheap

    async def approve(req):
        raise AssertionError("must not prompt during registration tests")

    names = lambda reg: {t["function"]["name"] for t in reg.schema()}
    assert "run_bash_host" not in names(build_registry(sandbox, session, mode="plan"))
    assert "run_bash_host" not in names(build_registry(sandbox, session, mode="write"))
    assert "run_bash_host" in names(
        build_registry(sandbox, session, mode="write", approve_callback=approve)
    )


# ---------------------------------------------------------------------------
# the tool itself


def test_command_and_reason_required(tmp_path) -> None:
    registry = _registry_for(tmp_path, [])
    result = call(registry, '{"reason": "why"}')
    assert not result.ok and "'command'" in result.error
    result = call(registry, '{"command": "ls"}')
    assert not result.ok and "'reason'" in result.error


def test_deny(tmp_path) -> None:
    registry = _registry_for(tmp_path, [Decision(kind="deny", deny_reason="not now")])
    result = call(registry, '{"command": "curl example.com", "reason": "need network"}')
    assert not result.ok
    assert result.error == "User denied this command. Reason: not now"
    assert result.header == "curl example.com"
    assert result.output == ""


def test_once_runs_the_command(tmp_path) -> None:
    registry = _registry_for(tmp_path, [Decision(kind="once")])
    result = call(registry, '{"command": "echo hi", "reason": "r"}')
    assert result.ok
    assert result.output == "hi\n"
    assert result.error == ""
    assert result.header == "echo hi"


def test_session_allows_exact_command_without_prompting(tmp_path) -> None:
    registry = _registry_for(tmp_path, [Decision(kind="session"), Decision(kind="once")])
    assert call(registry, '{"command": "echo a", "reason": "r"}').ok
    # Same command: allowed, no second prompt consumed.
    result = call(registry, '{"command": "echo a", "reason": "r"}')
    assert result.ok
    # A different command still needs the (now queued) approval.
    assert call(registry, '{"command": "echo b", "reason": "r"}').ok


def test_saved_prefix_skips_the_prompt(tmp_path) -> None:
    add_prefix(("echo",))
    registry = _registry_for(tmp_path, [])  # any prompt would fail the test
    assert call(registry, '{"command": "echo hi there", "reason": "r"}').ok


def test_prefix_decision_is_saved(tmp_path) -> None:
    registry = _registry_for(tmp_path, [Decision(kind="prefix", prefix=("echo", "hi"))])
    result = call(registry, '{"command": "echo hi", "reason": "testing"}')
    assert result.ok
    assert load_prefixes() == (("echo", "hi"),)
    # And the saved rule covers the next call without prompting.
    registry = _registry_for(tmp_path, [])
    assert call(registry, '{"command": "echo hi there", "reason": "r"}').ok


def test_compound_commands_always_prompt(tmp_path) -> None:
    add_prefix(("echo",))
    registry = _registry_for(tmp_path, [Decision(kind="deny")])
    result = call(registry, '{"command": "echo hi | tee /tmp/x", "reason": "r"}')
    assert not result.ok
    assert "denied" in result.error


def test_git_guard_runs_before_approval(tmp_path) -> None:
    async def approve(req):
        raise AssertionError("must not prompt")

    registry = ToolRegistry()
    registry.register(_make_run_bash_host_tool(
        _sandbox(tmp_path), approve, git_guard=lambda c: "no ref rewrites"
    ))
    result = call(registry, '{"command": "git checkout main", "reason": "r"}')
    assert not result.ok
    assert result.error == "no ref rewrites"
    assert result.header == "git checkout main"


def test_nonzero_exit_and_merged_output(tmp_path) -> None:
    registry = _registry_for(tmp_path, [])
    cmd = ("python3 -c \"import sys; print('out'); "
           "print('err', file=sys.stderr); raise SystemExit(3)\"")
    result = call(registry, json.dumps({"command": cmd, "reason": "r"}))
    assert not result.ok
    assert sorted(result.output.splitlines()) == ["err", "out"]
    assert result.error == "exit code 3"


def test_output_is_streamed_to_the_sink(tmp_path) -> None:
    registry = _registry_for(tmp_path, [])
    lines: list[str] = []
    result = call(
        registry, '{"command": "printf \'one\\ntwo\\n\'", "reason": "r"}', sink=lines.append
    )
    assert lines == ["one\n", "two\n"]
    assert result.output == "one\ntwo\n"


def test_timeout_kills_the_process_group(tmp_path) -> None:
    # The command is compound (`&`, `;`) so it prompts first; allow it, then
    # verify the timeout kills the group (both sleeps) instead of hanging.
    registry = _registry_for(
        tmp_path, [Decision(kind="once")]
    )
    result = call(registry, '{"command": "sleep 30 & sleep 30; echo done", '
                            '"reason": "r", "timeout_seconds": 1}')
    assert not result.ok
    assert result.error == "command timed out after 1s"
    assert result.output == ""
