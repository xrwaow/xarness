"""Tests for the rendered system prompt.

``system_prompt_for`` is the only place the model's instructions are built.
It should carry environment facts the model cannot discover on its own —
mode, sandbox, git/undo, network, toolchains — and nothing else: mechanics
belong in the tool descriptions, and behavioural policy belongs to the
model. A prompt that claims something false (edit tools in plan mode,
`.refs` in unsandboxed mode, `.venv` in a Rust project) is worse than a
shorter one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xarness.prompts import (
    CONTAINER_HEADER,
    NETWORK_OFF,
    NETWORK_ON,
    prompt_facts,
    system_prompt_for,
)
from xarness.sandbox import SandboxConfig, bind_ref


def _facts(config: SandboxConfig | None, mode: str = "write", **kw):
    return prompt_facts(config, mode, **kw)


@pytest.fixture
def sandbox(tmp_path: Path) -> SandboxConfig:
    return SandboxConfig(workspace=tmp_path, allow_network=True)


@pytest.fixture
def offline(tmp_path: Path) -> SandboxConfig:
    return SandboxConfig(workspace=tmp_path, allow_network=False)


@pytest.fixture
def python_project(tmp_path: Path) -> SandboxConfig:
    (tmp_path / "pyproject.toml").write_text("")
    return SandboxConfig(workspace=tmp_path, allow_network=False)


def test_base_comes_first_and_mode_clause_matches(sandbox):
    plan = system_prompt_for(_facts(sandbox, "plan"))
    write = system_prompt_for(_facts(sandbox, "write"))
    for prompt in (plan, write):
        assert prompt.startswith(CONTAINER_HEADER)
    assert "Mode: PLAN (read-only)" in plan
    assert "Mode: WRITE (read/write)" in write
    assert "Mode: PLAN" not in write and "Mode: WRITE" not in plan
    assert plan != write


def test_plan_mode_does_not_describe_write_tools(sandbox):
    prompt = system_prompt_for(_facts(sandbox, "plan"))
    assert "edit_file" not in prompt
    assert "write_file" not in prompt
    # No shell in unsandboxed plan mode → no shell section either.
    host_prompt = system_prompt_for(
        _facts(SandboxConfig(workspace=sandbox.workspace, use_container=False), "plan"))
    assert "run_bash" not in host_prompt


def test_read_mode_omits_write_only_facts(sandbox):
    prompt = system_prompt_for(_facts(sandbox, "plan", tracked=True))
    assert "snapshotted" not in prompt
    assert "git in the shell" not in prompt


def test_prompt_has_no_methodology_or_style(sandbox):
    """How to work is the model's call; only environment facts belong here."""
    prompt = system_prompt_for(_facts(sandbox, "write"))
    for policy in (
        "all-or-nothing",
        "Batch every change",
        "Finish with",
        "Verification",
        "confirm with ask_user",
        "ask_user",
        "exit code",
        "/bin/sh",
        "| head",
    ):
        assert policy not in prompt


def test_unsandboxed_prompt_has_no_container_claims(sandbox):
    host = SandboxConfig(workspace=sandbox.workspace, use_container=False)
    prompt = system_prompt_for(_facts(host, "write"))
    assert "/workspace" not in prompt
    assert ".refs" not in prompt
    assert "tmpfs" not in prompt
    assert "No container" in prompt
    # The real workspace path is named instead.
    assert str(sandbox.workspace) in prompt


def test_container_prompt_names_workspace_and_ephemeral_tmp(sandbox):
    prompt = system_prompt_for(_facts(sandbox, "write"))
    assert "/workspace" in prompt
    assert "tmpfs" in prompt


def test_refs_mentioned_only_when_mounted(sandbox, tmp_path):
    """Ref paths are a session fact, so they belong in the prompt — but only
    when the refs area is actually mounted. /diff and /undo are the user's
    commands and need not be named here."""
    ref = tmp_path / "notes.md"
    ref.write_text("x")
    host = SandboxConfig(workspace=sandbox.workspace)
    bind_ref(host.workspace, host.external_refs, ref, alias="notes")
    with_refs = system_prompt_for(_facts(host, "write", tracked=True))
    assert "/tmp/refs" in with_refs
    assert "/diff" not in with_refs and "/undo" not in with_refs
    # Untracked / git-write sessions and ref-less sessions still omit them.
    bare = system_prompt_for(_facts(sandbox, "write", git_write=True))
    assert "/tmp/refs" not in bare
    assert "/diff" not in bare and "/undo" not in bare
    assert ".refs" not in bare


def test_network_claim_follows_the_sandbox(sandbox, offline):
    online = system_prompt_for(_facts(sandbox))
    assert NETWORK_ON in online
    assert NETWORK_OFF not in online
    assert "UV_OFFLINE" not in online

    off = system_prompt_for(_facts(offline))
    assert NETWORK_OFF in off
    assert "UV_OFFLINE" in off
    assert NETWORK_ON not in off


def test_python_clause_only_for_python_projects(tmp_path, python_project):
    py = system_prompt_for(_facts(python_project))
    assert "`.venv`" in py
    assert "uv run --frozen" in py
    assert "uv.lock" in py
    # A directory with no project files gets no toolchain claim.
    plain = tmp_path / "plain"
    plain.mkdir()
    other = system_prompt_for(_facts(SandboxConfig(workspace=plain, allow_network=True)))
    assert ".venv" not in other
    assert "uv run" not in other


def test_cargo_project_gets_cargo_facts_without_test_instructions(tmp_path):
    (tmp_path / "Cargo.toml").write_text("")
    config = SandboxConfig(workspace=tmp_path, allow_network=True)
    prompt = system_prompt_for(_facts(config))
    assert "target-sandbox" in prompt
    assert "cargo test" not in prompt
    assert ".venv" not in prompt
    assert "pytest" not in prompt


def test_tracking_clause_is_opt_out(sandbox):
    with_tracking = system_prompt_for(_facts(sandbox, tracked=True))
    without = system_prompt_for(_facts(sandbox, tracked=False))
    assert "review and undo" in with_tracking
    assert "except gitignored files" in with_tracking
    assert "review and undo" not in without


def test_git_policy_follows_allow_git_write(sandbox):
    readonly = system_prompt_for(_facts(sandbox, tracked=True))
    assert "git in the shell is read-only" in readonly
    writable = system_prompt_for(_facts(sandbox, git_write=True))
    assert "git is read-write in the shell" in writable
    assert "inspection only" not in writable


def test_no_sandbox_means_no_environment_claims():
    prompt = system_prompt_for(_facts(None))
    assert "Environment" not in prompt
    assert "Sandbox" not in prompt
    assert ".venv" not in prompt
    assert "git" not in prompt
    assert "Mode: WRITE" in prompt


def test_same_facts_render_byte_identically(sandbox):
    """The prompt is part of the cached prefix: the same session must render
    the same bytes (no set iteration, no timestamps)."""
    facts = _facts(sandbox, tracked=True)
    assert system_prompt_for(facts) == system_prompt_for(facts)
