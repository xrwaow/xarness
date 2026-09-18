"""Tests for the read-only exploration tools (ls/glob/grep) and the
read/write scoping split in validate_relpath.

Uses a real temp git repo and real bwrap (git and bwrap are hard
dependencies of the feature).
"""

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from xarness.gitwork import setup_tracking
from xarness.sandbox import SandboxConfig
from xarness.tools import build_registry


def _git(cwd: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    if check:
        assert result.returncode == 0, f"{' '.join(args)}: {result.stderr}"
    return result.stdout


def _make_repo(base: Path) -> Path:
    """Committed files + a committed .gitignore, plus untracked files in
    every ignore category the tools must distinguish."""
    repo = base / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main", ".")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / ".gitignore").write_text("secret.txt\nignored_dir/\n")
    (repo / "app.py").write_text("print('hi')\n")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "controller.py").write_text("class Controller:\n    pass\n")
    (repo / "pkg" / "util.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    # layer a: gitignored (untracked, matched by the committed .gitignore)
    (repo / "secret.txt").write_text("hush\n")
    (repo / "ignored_dir").mkdir()
    (repo / "ignored_dir" / "controller.py").write_text("class Ignored:\n    pass\n")
    # layer b: default-ignored dirs, independent of any .gitignore
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "app.cpython-311.pyc").write_text("bin")
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "lib.py").write_text("class Nope:\n    pass\n")
    # untracked but NOT ignored — must stay visible everywhere
    (repo / "notes.txt").write_text("scratch\n")
    return repo


def _registry_for(
    workspace: Path,
    subtree: str = "",
    git_dir: Path | None = None,
    mode: str = "write",
):
    if shutil.which("bwrap") is None:
        pytest.skip("bwrap not available")
    sandbox = SandboxConfig(workspace=workspace, subtree=subtree, git_dir=git_dir)
    return build_registry(sandbox, None, mode=mode)


def _call(registry, name: str, args: str):
    return asyncio.run(registry.call(name, args))


# ---------------------------------------------------------------------------
# ls


def test_ls_filters_gitignored_and_default_ignored(tmp_path) -> None:
    repo = _make_repo(tmp_path)
    result = _call(_registry_for(repo), "ls", "{}")
    assert result.ok
    names = result.output.splitlines()
    assert "app.py" in names
    assert "pkg/" in names
    assert "notes.txt" in names          # untracked but not ignored
    assert ".git/" not in names          # layer b
    assert "__pycache__/" not in names   # layer b
    assert "node_modules/" not in names  # layer b
    assert "secret.txt" not in names     # layer a (committed .gitignore)
    # directories first, then alphabetical
    assert names == sorted(names, key=lambda n: (not n.endswith("/"), n.rstrip("/")))


def test_ls_lists_immediate_entries_only(tmp_path) -> None:
    repo = _make_repo(tmp_path)
    result = _call(_registry_for(repo), "ls", '{"path": "pkg"}')
    assert result.ok
    assert result.output.splitlines() == ["controller.py", "util.py"]


def test_ls_errors_on_missing_path_and_file(tmp_path) -> None:
    repo = _make_repo(tmp_path)
    missing = _call(_registry_for(repo), "ls", '{"path": "nope"}')
    assert not missing.ok
    assert "does not exist" in missing.error
    not_dir = _call(_registry_for(repo), "ls", '{"path": "app.py"}')
    assert not not_dir.ok
    assert "not a directory" in not_dir.error


# ---------------------------------------------------------------------------
# glob


def test_glob_finds_nested_controller_and_respects_ignores(tmp_path) -> None:
    repo = _make_repo(tmp_path)
    result = _call(_registry_for(repo), "glob", '{"glob": "**/controller.py"}')
    assert result.ok
    assert result.output.splitlines() == ["pkg/controller.py"]


def test_glob_scopes_by_path(tmp_path) -> None:
    repo = _make_repo(tmp_path)
    result = _call(_registry_for(repo), "glob", '{"glob": "*.py", "path": "pkg"}')
    assert result.ok
    assert sorted(result.output.splitlines()) == ["controller.py", "util.py"]


def test_glob_no_matches_is_ok(tmp_path) -> None:
    repo = _make_repo(tmp_path)
    result = _call(_registry_for(repo), "glob", '{"glob": "**/nothere.py"}')
    assert result.ok
    assert result.output == "(no matches)"


# ---------------------------------------------------------------------------
# grep


def test_grep_rg_present_and_absent_return_equivalent_results(tmp_path, monkeypatch) -> None:
    repo = _make_repo(tmp_path)
    query = '{"regex": "def |class ", "include_pattern": "**/*.py"}'
    with_rg = _call(_registry_for(repo), "grep", query)
    assert with_rg.ok
    assert "pkg/controller.py:1:class Controller:" in with_rg.output
    # node_modules is not gitignored — only the default-ignore layer hides it
    assert "node_modules" not in with_rg.output
    assert "ignored_dir" not in with_rg.output  # layer a

    real_which = shutil.which
    monkeypatch.setattr(
        shutil, "which",
        lambda name, *a, **k: None if name == "rg" else real_which(name, *a, **k),
    )
    without_rg = _call(_registry_for(repo), "grep", query)
    assert without_rg.ok
    # Backends may traverse in different orders; the match set must be equal.
    assert sorted(without_rg.output.splitlines()) == sorted(with_rg.output.splitlines())


def test_grep_no_matches_is_ok_not_error(tmp_path) -> None:
    repo = _make_repo(tmp_path)
    result = _call(_registry_for(repo), "grep", '{"regex": "zzz_nothing_zzz"}')
    assert result.ok
    assert result.output == "(no matches)"


# ---------------------------------------------------------------------------
# subtree scoping: the agent's workspace IS the --workspace dir


def _scoped_registry(tmp_path):
    repo = _make_repo(tmp_path)
    info, _ = asyncio.run(setup_tracking(repo / "pkg", "sess-scoped"))
    return repo, _registry_for(
        info.workspace, subtree=info.subtree, git_dir=info.git_dir,
    )


def test_scoped_session_tools_root_at_the_workspace(tmp_path) -> None:
    """With --workspace inside a bigger repo, tool paths are relative to that
    directory: ls lists its entries, and writes land inside it on the host."""
    repo, registry = _scoped_registry(tmp_path)
    # ls lists the workspace dir's own entries, not the repo root's.
    ls = _call(registry, "ls", "{}")
    assert ls.ok
    assert "controller.py" in ls.output.splitlines()
    assert "app.py" not in ls.output.splitlines()  # repo root, outside

    # Reads and writes are workspace-relative.
    read = _call(registry, "read_file", '{"path": "util.py"}')
    assert read.ok
    assert read.output == "x = 1\n"

    write = _call(registry, "write_file", '{"path": "new.py", "content": "ok = 1\\n"}')
    assert write.ok
    assert (repo / "pkg" / "new.py").read_text() == "ok = 1\n"
    # Nothing leaked outside the workspace dir.
    assert not (repo / "new.py").exists()


def test_scoped_session_explore_tools(tmp_path) -> None:
    _repo, registry = _scoped_registry(tmp_path)
    glob = _call(registry, "glob", '{"glob": "**/*.py"}')
    assert glob.ok
    assert sorted(glob.output.splitlines()) == ["controller.py", "util.py"]
    grep = _call(registry, "grep", '{"regex": "class Controller"}')
    assert grep.ok
    assert "controller.py:1:class Controller:" in grep.output


# ---------------------------------------------------------------------------
# truncation messaging (small caps, not the production constants)


def test_glob_truncation_message(tmp_path, monkeypatch) -> None:
    import xarness.tools as tools_mod

    repo = _make_repo(tmp_path)
    for i in range(5):
        (repo / f"data_{i}.txt").write_text("x\n")
    monkeypatch.setattr(tools_mod, "_MAX_GLOB_RESULTS", 2)
    result = _call(_registry_for(repo), "glob", '{"glob": "data_*.txt"}')
    assert result.ok
    matches = [l for l in result.output.splitlines() if l.startswith("data_")]
    assert len(matches) == 2
    assert "showing 2 of 5" in result.output


def test_grep_total_cap_message(tmp_path, monkeypatch) -> None:
    import xarness.tools as tools_mod

    repo = _make_repo(tmp_path)
    (repo / "hot.py").write_text("".join(f"needle {i}\n" for i in range(10)))
    monkeypatch.setattr(tools_mod, "_MAX_GREP_MATCHES", 3)
    result = _call(_registry_for(repo), "grep", '{"regex": "needle"}')
    assert result.ok
    hits = [l for l in result.output.splitlines() if "needle" in l]
    assert len(hits) == 3
    assert "stopped at 3 matches" in result.output


def test_grep_per_file_cap_message(tmp_path, monkeypatch) -> None:
    import xarness.tools as tools_mod

    repo = _make_repo(tmp_path)
    (repo / "hot.py").write_text("".join(f"needle {i}\n" for i in range(10)))
    monkeypatch.setattr(tools_mod, "_MAX_GREP_PER_FILE", 2)
    result = _call(
        _registry_for(repo), "grep", '{"regex": "needle|class Controller"}'
    )
    assert result.ok
    assert "pkg/controller.py" in result.output  # not crowded out by hot.py
    assert "2-matches-per-file cap" in result.output
