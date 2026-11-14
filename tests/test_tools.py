"""Tests for the tool registry: dispatch, failure containment, schema."""

import asyncio
import json
import shutil
from types import SimpleNamespace

import pytest

from xarness.sandbox import SandboxConfig, bind_ref
from xarness.tools import (
    Tool,
    ToolRegistry,
    ToolResult,
    _truncate_output,
    build_registry,
)
from pydantic import BaseModel


class _EmptyArgs(BaseModel):
    pass


class _TwoArgs(BaseModel):
    a: int = 0
    b: int = 0


async def _ok(args: _TwoArgs) -> ToolResult:
    return ToolResult(content=f"ran with a={args.a} b={args.b}")


async def _boom(args: _EmptyArgs) -> ToolResult:
    raise RuntimeError("kaboom")


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool(name="ok_tool", description="", args_model=_TwoArgs, handler=_ok))
    registry.register(Tool(name="boom_tool", description="", args_model=_EmptyArgs, handler=_boom))
    return registry


def _sandbox(tmp_path) -> SandboxConfig:
    if shutil.which("bwrap") is None:
        pytest.skip("bwrap not available")
    return SandboxConfig(workspace=tmp_path)


def _registry_for(tmp_path, mode: str = "write") -> ToolRegistry:
    return build_registry(_sandbox(tmp_path), None, mode=mode)


def _fake_session(sandbox: SandboxConfig) -> SimpleNamespace:
    """Stand-in for a SandboxSession: build_registry reads the config and the
    per-session tool timeout only."""
    return SimpleNamespace(config=sandbox, default_timeout=120.0)


def test_schema_matches_openai_tools_format() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="my_tool",
            description="does things",
            args_model=_EmptyArgs,
            handler=_ok,
        )
    )

    assert registry.schema() == [
        {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "does things",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_call_success_parses_arguments() -> None:
    result = asyncio.run(ToolRegistry.call(_registry(), "ok_tool", '{"b": 1, "a": 2}'))
    assert result.ok
    assert result.content == "ran with a=2 b=1"


def test_empty_arguments_become_empty_dict() -> None:
    result = asyncio.run(ToolRegistry.call(_registry(), "ok_tool", ""))
    assert result.ok


def test_invalid_json_is_a_parse_error() -> None:
    result = asyncio.run(ToolRegistry.call(_registry(), "ok_tool", '{"a": '))
    assert not result.ok
    assert result.status == "parse_error"
    assert "invalid arguments JSON" in result.content


def test_non_object_arguments_are_a_parse_error() -> None:
    result = asyncio.run(ToolRegistry.call(_registry(), "ok_tool", "[1, 2]"))
    assert not result.ok
    assert result.status == "parse_error"


def test_unknown_tool_fails_without_raising() -> None:
    result = asyncio.run(ToolRegistry.call(_registry(), "nope", "{}"))
    assert not result.ok
    assert result.status != "parse_error"
    assert "unknown tool" in result.content


def test_handler_exception_is_contained() -> None:
    result = asyncio.run(ToolRegistry.call(_registry(), "boom_tool", "{}"))
    assert not result.ok
    assert "RuntimeError: kaboom" in result.content


def test_plan_registry_is_read_only(tmp_path) -> None:
    names = {t["function"]["name"] for t in _registry_for(tmp_path, mode="plan").schema()}
    assert names == {"read_file"}  # no sandbox session → no run_bash in plan


def test_write_registry_exposes_write_tools(tmp_path) -> None:
    names = {t["function"]["name"] for t in _registry_for(tmp_path, mode="write").schema()}
    assert names == {
        "read_file", "write_file", "edit_file",
    }


def _descriptions(registry: ToolRegistry) -> dict[str, str]:
    return {
        t["function"]["name"]: t["function"]["description"]
        for t in registry.schema()
    }


def _schema_of(registry: ToolRegistry, name: str) -> dict:
    return next(
        t["function"]["parameters"] for t in registry.schema()
        if t["function"]["name"] == name
    )


def test_refs_do_not_vary_the_tool_descriptions(tmp_path) -> None:
    """Tool descriptions are stable regardless of refs (a stable schema keeps
    the provider's prompt cache warm); refs are described in the system
    prompt instead."""
    sandbox = _sandbox(tmp_path)
    notes = tmp_path / "notes.md"
    notes.write_text("x")

    before = _descriptions(build_registry(
        sandbox, _fake_session(sandbox), mode="write"))
    assert "/tmp/refs" not in before["read_file"]
    assert "/tmp/refs" not in before["run_bash"]
    assert "git in the shell is read-only" not in before["run_bash"]

    bind_ref(sandbox.workspace, sandbox.external_refs, notes, alias="notes")
    after = _descriptions(build_registry(
        sandbox, _fake_session(sandbox), mode="write"))
    assert after == before


def test_descriptions_split_on_host_vs_container(tmp_path) -> None:
    """A tool whose text names the sandbox names the real one: an
    unsandboxed session gets the host wording, and a tool that reads the same
    either way keeps one description."""
    container = _sandbox(tmp_path)
    host = SandboxConfig(workspace=tmp_path, use_sandbox=False)

    in_container = _descriptions(build_registry(
        container, _fake_session(container), mode="write"))
    on_host = _descriptions(build_registry(
        host, _fake_session(host), mode="write"))

    assert set(in_container) == set(on_host)
    assert "inside the sandboxed workspace" in in_container["run_bash"]
    assert "inside the sandboxed workspace" not in on_host["run_bash"]
    assert "on the host" in on_host["run_bash"]
    assert "Absolute paths are accepted" in in_container["read_file"]
    assert "Absolute paths are accepted" in on_host["read_file"]

    # Each description states only its own environment: no "when the session
    # runs in a container", no "there is no container" hedge.
    for name in ("read_file", "run_bash"):
        assert "container" not in on_host[name]
        assert "host" not in in_container[name]

    # Only the environment-naming tools differ; every other description is
    # the same text, byte for byte (a stable schema keeps the prompt cache
    # warm).
    assert set(in_container) - {"read_file", "run_bash"} == {
        "write_file", "edit_file",
    }
    for name in ("write_file", "edit_file"):
        assert in_container[name] == on_host[name]


def test_run_bash_description_states_the_real_timeout(tmp_path) -> None:
    """The timeout default is quoted as a number: the model cannot see the
    session's value anywhere else."""
    sandbox = _sandbox(tmp_path)
    session = SimpleNamespace(config=sandbox, default_timeout=45.0)
    registry = build_registry(sandbox, session, mode="write")
    description = _descriptions(registry)["run_bash"]
    assert "45s" in description
    assert "120s" not in description


def test_run_bash_description_states_the_starting_cwd(tmp_path) -> None:
    """The description names the starting cwd, so no command needs a ``cd``."""
    container = _sandbox(tmp_path)
    in_container = _descriptions(build_registry(
        container, _fake_session(container), mode="write"))["run_bash"]
    assert "starts in /workspace and keeps its cwd between calls" in in_container

    host = SandboxConfig(workspace=tmp_path, use_sandbox=False)
    on_host = _descriptions(build_registry(
        host, _fake_session(host), mode="write"))["run_bash"]
    assert f"starts in {tmp_path} and keeps its cwd between calls" in on_host


# ---------------------------------------------------------------------------
# read_file


def test_read_small_file_returns_full_contents(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("one\ntwo\n")
    result = asyncio.run(
        _registry_for(tmp_path).call("read_file", '{"path": "a.txt"}')
    )
    assert result.ok
    assert result.content == "one\ntwo\n"


def test_read_accepts_absolute_container_path(tmp_path) -> None:
    """read_file also takes absolute (container) paths, e.g. files in the
    shared cache anchor; only reads — writes stay workspace-relative."""
    (tmp_path / "a.txt").write_text("one\ntwo\n")
    result = asyncio.run(_registry_for(tmp_path).call(
        "read_file", '{"path": "/workspace/a.txt"}'
    ))
    assert result.ok
    assert result.content == "one\ntwo\n"


def test_read_accepts_absolute_path_outside_the_workspace(tmp_path) -> None:
    """Unsandboxed: an absolute path is a real path on the machine, so a file
    outside the workspace reads like any other — run_bash could already cat
    it, and nothing else in that mode is jailed either."""
    outside = tmp_path.parent / f"outside-{tmp_path.name}.txt"
    outside.write_text("one\ntwo\n")
    workspace = tmp_path / "project"
    workspace.mkdir()
    sandbox = SandboxConfig(workspace=workspace, use_sandbox=False)

    registry = build_registry(sandbox, None, mode="write")
    result = asyncio.run(registry.call(
        "read_file", json.dumps({"path": str(outside)})
    ))
    assert result.ok
    assert result.content == "one\ntwo\n"


def test_read_absolute_path_outline_keeps_path_intact(tmp_path) -> None:
    _big_python_file(tmp_path)
    result = asyncio.run(_registry_for(tmp_path).call(
        "read_file", '{"path": "/workspace/big.py"}'
    ))
    assert result.ok
    assert "# File outline for /workspace/big.py" in result.content


def _big_python_file(tmp_path) -> None:
    lines = [
        "import os",
        "",
        "@decorator",
        "class Big:",
        '    """doc"""',
        "",
        "    def method(self):",
        "        return 1",
        "",
        "    async def amethod(self):",
        "        return 2",
        "",
        "def top():",
        "    pass",
        "",
    ]
    lines += [f"# filler {i}" for i in range(600)]
    (tmp_path / "big.py").write_text("\n".join(lines) + "\n")


def test_read_large_python_file_returns_outline(tmp_path) -> None:
    _big_python_file(tmp_path)
    result = asyncio.run(
        _registry_for(tmp_path).call("read_file", '{"path": "big.py"}')
    )
    assert result.ok
    assert result.content.startswith(
        "File outline retrieved. This file is too large to read all at once"
    )
    assert "Do NOT retry this call without line numbers" in result.content
    # The outline names the path the caller passed, never the host path.
    assert "# File outline for big.py" in result.content
    assert str(tmp_path) not in result.content
    # Ranges are 1-based, include the whole body, exclude decorators, and
    # methods are indented under their class.
    assert "class Big [L4-11]" in result.content
    assert " def method [L7-8]" in result.content
    assert " async def amethod [L10-11]" in result.content
    assert "def top [L13-14]" in result.content
    assert result.content.rstrip().endswith("use start_line: 100 and end_line: 150.")


def test_read_large_file_with_range_returns_requested_lines(tmp_path) -> None:
    _big_python_file(tmp_path)
    result = asyncio.run(_registry_for(tmp_path).call(
        "read_file", '{"path": "big.py", "start_line": 4, "end_line": 5}'
    ))
    assert result.ok
    assert result.content.startswith('class Big:\n    """doc"""\n')
    assert "[showing lines 4-5 of " in result.content


def test_read_range_defaults_end_to_eof(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n")
    result = asyncio.run(_registry_for(tmp_path).call(
        "read_file", '{"path": "a.txt", "start_line": 2}'
    ))
    assert result.ok
    assert result.content == "two\nthree\n"


def test_read_non_python_large_file_falls_back_to_preview(tmp_path) -> None:
    (tmp_path / "big.txt").write_text(
        "".join(f"line {i}\n" for i in range(1, 601))
    )
    result = asyncio.run(
        _registry_for(tmp_path).call("read_file", '{"path": "big.txt"}')
    )
    assert result.ok
    assert "start_line/end_line" in result.content
    assert "line 1\n" in result.content
    assert "line 201" not in result.content


NUMBERED_1000 = "".join(f"line {i}\n" for i in range(1, 1001))


def _capped_read(tmp_path, text: str, limit: int, **args) -> ToolResult:
    """read_file with the registry's output cap forced to ``limit``."""
    (tmp_path / "a.txt").write_text(text)
    registry = _registry_for(tmp_path)
    registry.max_output_chars = limit
    return asyncio.run(registry.call(
        "read_file", json.dumps({"path": "a.txt", **args})
    ))


def _shown(result: ToolResult) -> tuple[list[str], str]:
    """(body lines, footer) of a read_file result."""
    body, _, footer = result.content.partition("\n[showing lines ")
    return body.splitlines(), footer


def test_read_range_is_contiguous_under_the_output_cap(tmp_path) -> None:
    """The cap must not splice a range (head + "[... elided ...]" + tail)
    while the footer keeps claiming every requested line."""
    result = _capped_read(tmp_path, NUMBERED_1000, 2000, end_line=1000)
    assert result.ok and "chars elided" not in result.content
    shown, footer = _shown(result)
    assert shown == [f"line {i}" for i in range(1, len(shown) + 1)]  # no gaps
    assert footer.startswith(f"1-{len(shown)} of 1000; output limit reached")
    assert f"continue from line {len(shown) + 1}]" in footer
    assert len(result.content) <= 2000


def test_read_resumes_where_a_capped_range_ended(tmp_path) -> None:
    first = _capped_read(tmp_path, NUMBERED_1000, 600, start_line=100)
    shown, footer = _shown(first)
    last = 100 + len(shown) - 1
    assert footer.startswith(f"100-{last} of 1000; output limit reached")
    rest = _capped_read(tmp_path, NUMBERED_1000, 600, start_line=last + 1)
    assert rest.content.splitlines()[0] == f"line {last + 1}"


def test_read_full_file_is_contiguous_under_the_output_cap(tmp_path) -> None:
    text = "".join(f"line {i} {'x' * 40}\n" for i in range(1, 301))
    result = _capped_read(tmp_path, text, 1000)
    shown, footer = _shown(result)
    assert shown == [f"line {i} {'x' * 40}" for i in range(1, len(shown) + 1)]
    assert footer.startswith(f"1-{len(shown)} of 300; output limit reached")
    assert len(result.content) <= 1000


def test_read_under_the_cap_is_unchanged(tmp_path) -> None:
    text = "".join(f"line {i}\n" for i in range(1, 101))
    ranged = _capped_read(tmp_path, text, 32768, start_line=10, end_line=20)
    assert ranged.content == "".join(
        f"line {i}\n" for i in range(10, 21)
    ) + "\n[showing lines 10-20 of 100]"
    assert _capped_read(tmp_path, text, 32768).content == text


def test_read_rejects_inverted_range(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("one\ntwo\n")
    result = asyncio.run(_registry_for(tmp_path).call(
        "read_file", '{"path": "a.txt", "start_line": 2, "end_line": 1}'
    ))
    assert not result.ok
    assert result.status == "parse_error"


# ---------------------------------------------------------------------------
# line counting (no trailing newline)


def test_read_counts_lines_without_trailing_newline(tmp_path) -> None:
    """wc -l counts newlines; a 3-line file without one must still report 3."""
    (tmp_path / "a.txt").write_text("one\ntwo\nthree")
    result = asyncio.run(
        _registry_for(tmp_path).call("read_file", '{"path": "a.txt"}')
    )
    assert result.ok
    # Full read: no truncation annotation, and the last line is included.
    assert result.content == "one\ntwo\nthree"

    ranged = asyncio.run(_registry_for(tmp_path).call(
        "read_file", '{"path": "a.txt", "start_line": 2, "end_line": 2}'
    ))
    assert ranged.ok
    assert "[showing lines 2-2 of 3" in ranged.content


def test_read_outline_threshold_ignores_missing_trailing_newline(tmp_path) -> None:
    """A 501-line Python file with no trailing newline must cross the outline
    threshold (wc -l would count 500 and read it whole instead)."""
    lines = ["import os", *(["# filler"] * 498), "def top():", "    pass"]
    assert len(lines) == 501
    (tmp_path / "big.py").write_text("\n".join(lines))
    result = asyncio.run(
        _registry_for(tmp_path).call("read_file", '{"path": "big.py"}')
    )
    assert result.ok
    assert result.content.startswith("File outline retrieved.")
    assert "def top [L500-501]" in result.content


def test_write_file_creates_missing_parent_directories(tmp_path) -> None:
    result = asyncio.run(_registry_for(tmp_path).call(
        "write_file", '{"path": "a/b/c.txt", "content": "hi\\n"}'
    ))
    assert result.ok, result.content
    assert (tmp_path / "a" / "b" / "c.txt").read_text() == "hi\n"


def test_write_file_reports_bytes_not_characters(tmp_path) -> None:
    result = asyncio.run(_registry_for(tmp_path).call(
        "write_file", json.dumps({"path": "u.txt", "content": "héllo"})
    ))
    assert result.ok
    assert "(6 bytes)" in result.content


def test_write_tools_reject_the_read_only_refs_area(tmp_path) -> None:
    """The canonical refs path is absolute; writes there must say read-only,
    not "use a workspace-relative path". Dot segments cannot escape it."""
    for path in ("/tmp/refs/x", "/tmp/refs/./x", "/tmp/refs/../refs/x"):
        result = asyncio.run(_registry_for(tmp_path).call(
            "write_file", json.dumps({"path": path, "content": "no"})
        ))
        assert not result.ok, path
        assert "read-only" in result.content, path


def test_string_integers_are_coerced(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n")
    result = asyncio.run(_registry_for(tmp_path).call(
        "read_file", '{"path": "a.txt", "start_line": "2"}'
    ))
    assert result.ok, result.content
    assert result.content == "two\nthree\n"


def test_missing_required_argument_is_a_parse_error(tmp_path) -> None:
    result = asyncio.run(_registry_for(tmp_path).call("read_file", "{}"))
    assert result.status == "parse_error"
    assert "path: Field required" in result.content


def test_read_start_past_eof_names_the_length(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("one\ntwo\n")
    result = asyncio.run(_registry_for(tmp_path).call(
        "read_file", '{"path": "a.txt", "start_line": 9}'
    ))
    assert not result.ok
    assert "2 lines" in result.content


# ---------------------------------------------------------------------------
# write_file / edit_file


def test_write_file_creates_then_overwrites(tmp_path) -> None:
    registry = _registry_for(tmp_path)
    first = asyncio.run(registry.call(
        "write_file", '{"path": "new.txt", "content": "hello\\n"}'
    ))
    assert first.ok
    assert first.content.startswith("created ")
    second = asyncio.run(registry.call(
        "write_file", '{"path": "new.txt", "content": "bye\\n"}'
    ))
    assert second.ok
    assert second.content.startswith("overwrote ")
    assert (tmp_path / "new.txt").read_text() == "bye\n"


def test_edit_file_replaces_unique_occurrence(tmp_path) -> None:
    (tmp_path / "f.txt").write_text("alpha\nbeta\nalpha\n")
    result = asyncio.run(_registry_for(tmp_path).call(
        "edit_file",
        '{"path": "f.txt", "edits": [{"old_string": "beta", "new_string": "BETA"}]}',
    ))
    assert result.ok
    assert (tmp_path / "f.txt").read_text() == "alpha\nBETA\nalpha\n"
    # The result carries the changed code, not just an acknowledgement.
    assert result.content.startswith("applied edit to f.txt")
    assert "-beta" in result.content
    assert "+BETA" in result.content


def test_edit_file_newline_only_change_is_reported(tmp_path) -> None:
    (tmp_path / "f.txt").write_text("alpha")
    args = {"path": "f.txt", "edits": [{"old_string": "alpha", "new_string": "alpha\n"}]}
    result = asyncio.run(_registry_for(tmp_path).call("edit_file", json.dumps(args)))
    assert result.ok and "(no textual change)" not in result.content
    assert (tmp_path / "f.txt").read_text() == "alpha\n"


def test_truncate_output_never_exceeds_the_limit() -> None:
    assert all(
        len(_truncate_output("x" * 1000, limit)) <= limit
        for limit in (1, 10, 64, 100)
    )


def test_edit_file_rejects_bare_top_level_pair(tmp_path) -> None:
    (tmp_path / "f.txt").write_text("alpha\nbeta\n")
    result = asyncio.run(_registry_for(tmp_path).call(
        "edit_file", '{"path": "f.txt", "old_string": "beta", "new_string": "BETA"}'
    ))
    assert not result.ok
    assert "edits" in result.content


def test_edit_file_applies_multiple_edits_in_one_call(tmp_path) -> None:
    (tmp_path / "f.txt").write_text("alpha\nbeta\ngamma\n")
    result = asyncio.run(_registry_for(tmp_path).call(
        "edit_file",
        json.dumps({"path": "f.txt", "edits": [
            {"old_string": "alpha", "new_string": "ALPHA"},
            {"old_string": "gamma", "new_string": "GAMMA"},
        ]}),
    ))
    assert result.ok
    assert (tmp_path / "f.txt").read_text() == "ALPHA\nbeta\nGAMMA\n"
    assert result.content.startswith("applied 2 edits to f.txt")
    assert "diff --git a/f.txt b/f.txt" in result.content
    for line in ("-alpha", "+ALPHA", "-gamma", "+GAMMA"):
        assert line in result.content
    # The diff must be line-structured: the TUI parser keys off "---",
    # "+++" and "@@" starting their own lines.
    lines = result.content.splitlines()
    assert "--- a/f.txt" in lines
    assert "+++ b/f.txt" in lines
    assert any(line.startswith("@@") for line in lines)


def test_edit_file_edits_apply_in_order(tmp_path) -> None:
    """A later edit matches against the text the earlier one produced."""
    (tmp_path / "f.txt").write_text("one\ntwo\n")
    result = asyncio.run(_registry_for(tmp_path).call(
        "edit_file",
        json.dumps({"path": "f.txt", "edits": [
            {"old_string": "one", "new_string": "uno"},
            {"old_string": "uno", "new_string": "UNO"},
        ]}),
    ))
    assert result.ok
    assert (tmp_path / "f.txt").read_text() == "UNO\ntwo\n"


def test_edit_file_is_atomic_on_a_failed_batch(tmp_path) -> None:
    (tmp_path / "f.txt").write_text("alpha\nbeta\n")
    result = asyncio.run(_registry_for(tmp_path).call(
        "edit_file",
        json.dumps({"path": "f.txt", "edits": [
            {"old_string": "alpha", "new_string": "ALPHA"},
            {"old_string": "missing", "new_string": "X"},
        ]}),
    ))
    assert not result.ok
    assert "edit 2" in result.content
    assert "not found" in result.content
    # Nothing was written: a failed batch never leaves a half-edited file.
    assert (tmp_path / "f.txt").read_text() == "alpha\nbeta\n"
    assert "ALPHA" not in result.content


def test_edit_file_reports_which_edit_was_ambiguous(tmp_path) -> None:
    (tmp_path / "f.txt").write_text("alpha\nbeta\nalpha\n")
    result = asyncio.run(_registry_for(tmp_path).call(
        "edit_file",
        json.dumps({"path": "f.txt", "edits": [
            {"old_string": "beta", "new_string": "BETA"},
            {"old_string": "alpha", "new_string": "ALPHA"},
        ]}),
    ))
    assert not result.ok
    assert "edit 2" in result.content
    assert "not unique" in result.content
    assert "replace_all" in result.content
    # Atomic: the ambiguous edit means edit 1 is not applied either.
    assert (tmp_path / "f.txt").read_text() == "alpha\nbeta\nalpha\n"


def test_edit_file_replace_all(tmp_path) -> None:
    (tmp_path / "f.txt").write_text("old\nold\n")
    result = asyncio.run(_registry_for(tmp_path).call(
        "edit_file",
        json.dumps({"path": "f.txt", "edits": [
            {"old_string": "old", "new_string": "new", "replace_all": True},
        ]}),
    ))
    assert result.ok
    assert (tmp_path / "f.txt").read_text() == "new\nnew\n"


def test_edit_file_rejects_empty_edits_list(tmp_path) -> None:
    (tmp_path / "f.txt").write_text("alpha\n")
    result = asyncio.run(_registry_for(tmp_path).call(
        "edit_file", '{"path": "f.txt", "edits": []}'
    ))
    assert not result.ok
    assert result.status == "parse_error"
    assert "non-empty list" in result.content


def test_edit_file_rejects_malformed_edit_item(tmp_path) -> None:
    (tmp_path / "f.txt").write_text("alpha\n")
    result = asyncio.run(_registry_for(tmp_path).call(
        "edit_file", '{"path": "f.txt", "edits": [{"new_string": "x"}]}'
    ))
    assert not result.ok
    assert result.status == "parse_error"
    assert "old_string" in result.content
    assert (tmp_path / "f.txt").read_text() == "alpha\n"


def test_edit_file_rejects_ambiguous_match(tmp_path) -> None:
    (tmp_path / "f.txt").write_text("alpha\nalpha\n")
    result = asyncio.run(_registry_for(tmp_path).call(
        "edit_file",
        '{"path": "f.txt", "edits": [{"old_string": "alpha", "new_string": "BETA"}]}',
    ))
    assert not result.ok
    assert "not unique" in result.content


def test_edit_file_requires_old_string(tmp_path) -> None:
    (tmp_path / "f.txt").write_text("alpha\n")
    result = asyncio.run(_registry_for(tmp_path).call(
        "edit_file",
        '{"path": "f.txt", "edits": [{"new_string": "BETA"}]}',
    ))
    assert not result.ok
    assert result.status == "parse_error"
    assert "old_string" in result.content

# ---------------------------------------------------------------------------
# ask_user


def _ask_registry(tmp_path, answer: list[str] | None):
    """A registry whose ask callback records the questions it was handed and
    replies with ``answer`` (None = the user skipped)."""
    seen: dict = {}

    async def ask(questions):
        seen["questions"] = questions
        return answer

    sandbox = SandboxConfig(workspace=tmp_path, use_sandbox=False)
    return build_registry(
        sandbox, _fake_session(sandbox), mode="write", ask_callback=ask,
    ), seen


def _ask_payload(*questions) -> str:
    return json.dumps({"questions": list(questions)})


def test_ask_accepts_plain_string_questions(tmp_path) -> None:
    """A bare string parses as a free-text question: an older model is not
    punished for the pre-``options`` shape."""
    registry, seen = _ask_registry(tmp_path, ["ship it"])
    result = asyncio.run(registry.call("ask_user", _ask_payload("what next?")))
    assert result.ok
    assert result.content == "Q: what next?\nA: ship it"
    assert [q.question for q in seen["questions"]] == ["what next?"]


def test_ask_resolves_a_picked_number_and_keeps_a_write_in(tmp_path) -> None:
    """The client sends the key typed; the tool returns the choice's label."""
    payload = _ask_payload(
        {"question": "Which storage?", "options": ["sqlite", "postgres"]},
    )
    registry, seen = _ask_registry(tmp_path, ["2"])
    result = asyncio.run(registry.call("ask_user", payload))
    assert result.content == "Q: Which storage?\nA: postgres"
    assert seen["questions"][0].options == ["sqlite", "postgres"]

    for typed in ("2", "2)", "2.", "2:"):
        registry, _ = _ask_registry(tmp_path, [typed])
        result = asyncio.run(registry.call("ask_user", payload))
        assert result.content.endswith("A: postgres"), typed

    for typed in ("duckdb", "7", "2 but keep the old one"):
        registry, _ = _ask_registry(tmp_path, [typed])
        result = asyncio.run(registry.call("ask_user", payload))
        assert result.content.endswith(f"A: {typed}")


def test_ask_reports_unanswered_questions_instead_of_dropping_them(tmp_path) -> None:
    """A client that answers fewer questions than it was asked must not make
    the tail vanish: the model would read a short pairing as "all answered"."""
    registry, _ = _ask_registry(tmp_path, ["sqlite"])
    result = asyncio.run(registry.call("ask_user", _ask_payload(
        {"question": "Which storage?", "options": ["sqlite", "postgres"]},
        {"question": "Project name?"},
    )))
    assert "A: sqlite" in result.content
    assert "Q: Project name?\nA: (no answer given)" in result.content


def test_ask_skip_and_blank_questions_do_not_error(tmp_path) -> None:
    """Skipping is a notice, not a failure; a blank question never reaches
    the user."""
    registry, _ = _ask_registry(tmp_path, None)
    skipped = asyncio.run(registry.call("ask_user", _ask_payload("what next?")))
    assert skipped.ok and "user skipped" in skipped.content

    registry, _ = _ask_registry(tmp_path, ["x"])
    blank = asyncio.run(registry.call("ask_user", _ask_payload("   ")))
    assert blank.status == "parse_error" and "non-empty" in blank.content


def test_ask_schema_offers_options_per_question(tmp_path) -> None:
    """The choices live inside each question: a separate flat list is what
    made models send options as if they were questions."""
    registry, _ = _ask_registry(tmp_path, None)
    items = _schema_of(registry, "ask_user")["properties"]["questions"]["items"]
    assert set(items["properties"]) == {"question", "options"}
    assert "choices" in items["properties"]["options"]["description"]
