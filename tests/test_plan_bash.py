"""Tests for plan-mode run_bash against a read-only workspace mount.

Plan mode = read_file + run_bash, with the whole workspace (including any
scoped subtree) bound read-only; the persistent shell restarts when the
read-only flag flips on a /mode switch. Uses a real temp git repo and real
bwrap (both are hard dependencies of the feature).
"""

import asyncio
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
    assert "print('hi')" in reads.output

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


def test_subtree_scoped_plan_mode_keeps_subtree_read_only(tmp_path) -> None:
    from xarness.gitwork import setup_tracking

    repo = _make_repo(tmp_path)
    info, _ = asyncio.run(setup_tracking(repo / "pkg", "sess-ro"))
    sandbox = SandboxConfig(
        workspace=info.workspace, subtree=info.subtree, git_dir=info.git_dir,
    )
    session = SandboxSession(sandbox)
    sandbox.read_only = True
    registry = build_registry(sandbox, session, mode="plan")
    read = _call(registry, "read_file", '{"path": "util.py"}')
    assert read.ok
    write = _call(registry, "run_bash", '{"command": "touch nope.txt"}')
    assert not write.ok
    assert not (repo / "pkg" / "nope.txt").exists()
