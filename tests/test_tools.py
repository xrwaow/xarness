"""Tests for the tool registry: dispatch, failure containment, schema."""

import asyncio
import shutil

import pytest

from xarness.sandbox import SandboxConfig
from xarness.tools import Tool, ToolRegistry, ToolResult, build_registry


async def _ok(args: dict) -> ToolResult:
    return ToolResult(ok=True, output=f"ran with {sorted(args)}")


async def _boom(args: dict) -> ToolResult:
    raise RuntimeError("kaboom")


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool(name="ok_tool", description="", parameters_schema={}, handler=_ok))
    registry.register(Tool(name="boom_tool", description="", parameters_schema={}, handler=_boom))
    return registry


def _sandbox(tmp_path) -> SandboxConfig:
    if shutil.which("bwrap") is None:
        pytest.skip("bwrap not available")
    return SandboxConfig(workspace=tmp_path)


def _registry_for(tmp_path, mode: str = "write") -> ToolRegistry:
    return build_registry(_sandbox(tmp_path), None, mode=mode)


def test_schema_matches_openai_tools_format() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="my_tool",
            description="does things",
            parameters_schema={"type": "object", "properties": {}},
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
    assert result.output == "ran with ['a', 'b']"


def test_empty_arguments_become_empty_dict() -> None:
    result = asyncio.run(ToolRegistry.call(_registry(), "ok_tool", ""))
    assert result.ok


def test_invalid_json_is_a_parse_error() -> None:
    result = asyncio.run(ToolRegistry.call(_registry(), "ok_tool", '{"a": '))
    assert not result.ok
    assert result.parse_error is True
    assert "invalid arguments JSON" in result.error


def test_non_object_arguments_are_a_parse_error() -> None:
    result = asyncio.run(ToolRegistry.call(_registry(), "ok_tool", "[1, 2]"))
    assert not result.ok
    assert result.parse_error is True


def test_unknown_tool_fails_without_raising() -> None:
    result = asyncio.run(ToolRegistry.call(_registry(), "nope", "{}"))
    assert not result.ok
    assert result.parse_error is False
    assert "unknown tool" in result.error


def test_handler_exception_is_contained() -> None:
    result = asyncio.run(ToolRegistry.call(_registry(), "boom_tool", "{}"))
    assert not result.ok
    assert "RuntimeError: kaboom" in result.error


def test_plan_registry_is_read_only(tmp_path) -> None:
    names = {t["function"]["name"] for t in _registry_for(tmp_path, mode="plan").schema()}
    assert names == {"web_search", "read_file"}


def test_write_registry_exposes_write_tools(tmp_path) -> None:
    names = {t["function"]["name"] for t in _registry_for(tmp_path, mode="write").schema()}
    assert names == {"web_search", "read_file", "write_file", "edit_file"}


# ---------------------------------------------------------------------------
# read_file


def test_read_small_file_returns_full_contents(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("one\ntwo\n")
    result = asyncio.run(
        _registry_for(tmp_path).call("read_file", '{"path": "a.txt"}')
    )
    assert result.ok
    assert result.output == "one\ntwo\n"


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
    assert result.output.startswith(
        "File outline retrieved. This file is too large to read all at once"
    )
    assert "Do NOT retry this call without line numbers" in result.output
    assert f"# File outline for {tmp_path / 'big.py'}" in result.output
    # Ranges are 1-based, include the whole body, exclude decorators, and
    # methods are indented under their class.
    assert "class Big [L4-11]" in result.output
    assert " def method [L7-8]" in result.output
    assert " async def amethod [L10-11]" in result.output
    assert "def top [L13-14]" in result.output
    assert result.output.rstrip().endswith("use start_line: 100 and end_line: 150.")


def test_read_large_file_with_range_returns_requested_lines(tmp_path) -> None:
    _big_python_file(tmp_path)
    result = asyncio.run(_registry_for(tmp_path).call(
        "read_file", '{"path": "big.py", "start_line": 4, "end_line": 5}'
    ))
    assert result.ok
    assert result.output.startswith('class Big:\n    """doc"""\n')
    assert "[showing lines 4-5 of " in result.output


def test_read_range_defaults_end_to_eof(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n")
    result = asyncio.run(_registry_for(tmp_path).call(
        "read_file", '{"path": "a.txt", "start_line": 2}'
    ))
    assert result.ok
    assert result.output == "two\nthree\n"


def test_read_non_python_large_file_falls_back_to_preview(tmp_path) -> None:
    (tmp_path / "big.txt").write_text(
        "".join(f"line {i}\n" for i in range(1, 601))
    )
    result = asyncio.run(
        _registry_for(tmp_path).call("read_file", '{"path": "big.txt"}')
    )
    assert result.ok
    assert "start_line/end_line" in result.output
    assert "line 1\n" in result.output
    assert "line 201" not in result.output


def test_read_rejects_inverted_range(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("one\ntwo\n")
    result = asyncio.run(_registry_for(tmp_path).call(
        "read_file", '{"path": "a.txt", "start_line": 2, "end_line": 1}'
    ))
    assert not result.ok
    assert result.parse_error is True


# ---------------------------------------------------------------------------
# write_file / edit_file


def test_write_file_creates_then_overwrites(tmp_path) -> None:
    registry = _registry_for(tmp_path)
    first = asyncio.run(registry.call(
        "write_file", '{"path": "new.txt", "content": "hello\\n"}'
    ))
    assert first.ok
    assert first.output.startswith("created ")
    second = asyncio.run(registry.call(
        "write_file", '{"path": "new.txt", "content": "bye\\n"}'
    ))
    assert second.ok
    assert second.output.startswith("overwrote ")
    assert (tmp_path / "new.txt").read_text() == "bye\n"


def test_write_file_rejects_refs(tmp_path) -> None:
    result = asyncio.run(_registry_for(tmp_path).call(
        "write_file", '{"path": ".refs/x", "content": "nope"}'
    ))
    assert not result.ok
    assert "read-only" in result.error


def test_edit_file_replaces_unique_occurrence(tmp_path) -> None:
    (tmp_path / "f.txt").write_text("alpha\nbeta\nalpha\n")
    result = asyncio.run(_registry_for(tmp_path).call(
        "edit_file", '{"path": "f.txt", "old_string": "beta", "new_string": "BETA"}'
    ))
    assert result.ok
    assert (tmp_path / "f.txt").read_text() == "alpha\nBETA\nalpha\n"


def test_edit_file_rejects_ambiguous_match(tmp_path) -> None:
    (tmp_path / "f.txt").write_text("alpha\nalpha\n")
    result = asyncio.run(_registry_for(tmp_path).call(
        "edit_file", '{"path": "f.txt", "old_string": "alpha", "new_string": "BETA"}'
    ))
    assert not result.ok
    assert "not unique" in result.error


def test_edit_file_requires_old_string(tmp_path) -> None:
    (tmp_path / "f.txt").write_text("alpha\n")
    result = asyncio.run(_registry_for(tmp_path).call(
        "edit_file", '{"path": "f.txt", "new_string": "BETA"}'
    ))
    assert not result.ok
    assert result.parse_error is True
    assert "old_string" in result.error