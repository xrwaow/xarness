"""Tests for plan-mode run_bash against a read-only workspace mount.

Plan mode = read_file + run_bash, with the whole workspace (including any
scoped subtree) bound read-only; the persistent shell restarts when the
read-only flag flips on a /mode switch. Uses a real temp git repo and real
bwrap (both are hard dependencies of the feature).
"""

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from xarness.sandbox import SandboxConfig, SandboxSession
from xarness.tools import build_registry


def _git(cwd: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    if check:
        assert result.returncode == 0, f"{' '.join(args)}: {result.stderr}"
    return result.stdout


def _make_repo(base: Path) -> Path:
    repo = base / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main", ".")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "app.py").write_text("print('hi')\n")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "controller.py").write_text("class Controller:\n    pass\n")
    (repo / "pkg" / "util.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def _sandbox(workspace: Path) -> SandboxConfig:
    if shutil.which("bwrap") is None:
        pytest.skip("bwrap not available")
    return SandboxConfig(workspace=workspace)


def _call(registry, name: str, args: str):
    return asyncio.run(registry.call(name, args))


def test_plan_mode_binds_workspace_read_only(tmp_path) -> None:
    sandbox = _sandbox(_make_repo(tmp_path))
    session = SandboxSession(sandbox)
    sandbox.read_only = True  # the caller (app) owns the sandbox's mode
    registry = build_registry(sandbox, session, mode="plan")
    names = {t["function"]["name"] for t in registry.schema()}
    assert names == {"read_file", "run_bash"}

    reads = _call(registry, "run_bash", '{"command": "cat app.py"}')
    assert reads.ok
    assert "print('hi')" in reads.content

    # The read-only mount blocks writes from the shell.
    write = _call(registry, "run_bash", '{"command": "touch blocked.txt"}')
    assert not write.ok
    assert not (sandbox.workspace / "blocked.txt").exists()


def test_write_mode_mount_is_writable(tmp_path) -> None:
    sandbox = _sandbox(_make_repo(tmp_path))
    session = SandboxSession(sandbox)
    registry = build_registry(sandbox, session, mode="write")
    write = _call(registry, "run_bash", '{"command": "echo ok > made.txt"}')
    assert write.ok
    assert (sandbox.workspace / "made.txt").read_text().strip() == "ok"


def test_mode_switch_restarts_shell_under_new_mounts(tmp_path) -> None:
    """The persistent shell must pick up a read_only flip: a session started
    in write mode cannot stay writable after switching to plan."""
    sandbox = _sandbox(_make_repo(tmp_path))
    session = SandboxSession(sandbox)
    write_registry = build_registry(sandbox, session, mode="write")
    _call(write_registry, "run_bash", '{"command": "echo state > s.txt"}')
    assert sandbox.read_only is False

    # The app flips the sandbox's mode itself (build_registry no longer does).
    sandbox.read_only = True
    plan_registry = build_registry(sandbox, session, mode="plan")
    assert sandbox.read_only is True
    blocked = _call(plan_registry, "run_bash", '{"command": "touch nope.txt; cat s.txt"}')
    assert not blocked.ok
    assert not (sandbox.workspace / "nope.txt").exists()
    # The restarted shell lost write-mode state? cwd/env yes, files no —
    # s.txt still exists (it was written to the host workspace).
    assert (sandbox.workspace / "s.txt").exists()


def test_plan_mode_without_container_has_no_run_bash(tmp_path) -> None:
    """Plan mode with use_container=False: there is no read-only mount to
    enforce the mode with, so run_bash must not be available at all
    (read_file stays). No bwrap needed — registry construction only."""
    sandbox = SandboxConfig(workspace=_make_repo(tmp_path), use_container=False)
    session = SandboxSession(sandbox)
    sandbox.read_only = True
    registry = build_registry(sandbox, session, mode="plan")
    names = {t["function"]["name"] for t in registry.schema()}
    assert names == {"read_file"}
    # Write mode keeps run_bash (unsandboxed, but that is the mode's contract).
    write_registry = build_registry(sandbox, session, mode="write")
    write_names = {t["function"]["name"] for t in write_registry.schema()}
    assert "run_bash" in write_names and "read_file" in write_names


def _tracked_sandbox(workspace: Path) -> SandboxConfig:
    """A container sandbox built the way from_settings builds it for a
    tracked workspace with allow_git_write off: the read-only git shim is
    installed first on PATH."""
    from xarness.config import SandboxSettings
    from xarness.gitwork import GitInfo
    if shutil.which("bwrap") is None:
        pytest.skip("bwrap not available")
    git_dir = workspace / ".git"
    info = GitInfo(
        git_dir=git_dir, external=False, last_seen_tree=None,
        workspace=workspace, checkpoints=[],
    ) if hasattr(GitInfo, "checkpoints") else mock_git_info(git_dir)
    return SandboxConfig.from_settings(workspace, SandboxSettings(), info)


def mock_git_info(git_dir: Path):
    from unittest.mock import Mock
    info = Mock()
    info.git_dir = git_dir
    info.external = False
    return info


def test_git_shim_blocks_in_place(tmp_path) -> None:
    """The PATH shim enforces read-only git at execution time: statements
    around a blocked git call still run, `&&` stops at the block, and
    commands that merely mention git are unaffected."""
    repo = _make_repo(tmp_path)
    sandbox = _tracked_sandbox(repo)
    assert sandbox.git_shim
    session = SandboxSession(sandbox)
    registry = build_registry(sandbox, session, mode="write")

    async def go():
        results = []
        for command in (
            "echo before; git commit -m x; echo after",
            "echo before && git add -A && echo after",
            "git status --short",
            "grep git README || echo gone",
        ):
            results.append(await registry.call(
                "run_bash", json.dumps({"command": command})))
        return results

    semi, amp, inspect_, mention = asyncio.run(go())
    # The blocked call fails in place; the shell keeps going and the last
    # statement's status (echo after) is the command's own.
    assert semi.ok
    assert "before" in semi.output and "after" in semi.content
    assert "blocked, read-only here" in semi.content

    assert "before" in amp.output and "after" not in amp.content

    assert inspect_.ok  # inspection subcommands pass the shim
    assert mention.ok  # mentioning git is not running git


def test_git_shim_writes_into_the_sandbox_cache(tmp_path) -> None:
    """The shim lives in the shared sandbox cache (already bound at the
    /cache anchor) and is on PATH ahead of the real bin dirs."""
    from xarness.sandbox import sandbox_cache_root
    repo = _make_repo(tmp_path)
    sandbox = _tracked_sandbox(repo)
    shim = sandbox_cache_root() / "git-shim" / "git"
    assert shim.is_file() and shim.stat().st_mode & 0o111
    argv = sandbox.build_argv(["/bin/sh"])
    path = next(argv[i + 2] for i, a in enumerate(argv)
                if a == "--setenv" and argv[i + 1] == "PATH")
    assert path.startswith("/cache/git-shim:")
