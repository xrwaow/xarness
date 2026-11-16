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


def _host_plan_registry(tmp_path) -> ToolRegistry:
    sandbox = SandboxConfig(workspace=tmp_path, use_sandbox=False)
    return build_registry(sandbox, None, mode="plan")


def test_plan_registry_on_the_host_swaps_the_shell_for_search(tmp_path) -> None:
    """No container means no read-only mount to enforce run_bash with, so the
    bounded read-only search tools take its place."""
    names = {t["function"]["name"] for t in _host_plan_registry(tmp_path).schema()}
    assert names == {"read_file", "glob", "grep"}


def test_host_search_tools_find_files_and_lines(tmp_path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("class A:\n    pass\n")
    (tmp_path / "notes.txt").write_text("class notes\n")
    registry = _host_plan_registry(tmp_path)

    globbed = asyncio.run(registry.call("glob", '{"pattern": "*.py"}'))
    assert globbed.content.splitlines() == ["pkg/a.py"]
    grepped = asyncio.run(
        registry.call("grep", '{"pattern": "class", "glob": "*.py"}'))
    assert grepped.content.splitlines() == ["pkg/a.py:1: class A:"]


def test_host_search_tools_are_read_only_and_confined(tmp_path) -> None:
    registry = _host_plan_registry(tmp_path)
    outside = asyncio.run(registry.call("glob", '{"pattern": "*", "path": "/etc"}'))
    assert not outside.ok and "outside the workspace" in outside.content
    traversal = asyncio.run(registry.call("grep", '{"pattern": "x", "path": ".."}'))
    assert not traversal.ok
    invalid = asyncio.run(registry.call("grep", '{"pattern": "(unclosed"}'))
    assert invalid.status == "parse_error"


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


def _every_text_tool(registry: ToolRegistry, path: str) -> list[ToolResult]:
    """read_file, write_file and edit_file called on one path: what they
    answer must agree, so every refusal is asserted once per tool here."""
    return [
        asyncio.run(registry.call(tool, json.dumps(args)))
        for tool, args in (
            ("read_file", {"path": path}),
            ("write_file", {"path": path, "content": "x\n"}),
            ("edit_file", {"path": path,
                           "edits": [{"old_string": "a", "new_string": "b"}]}),
        )
    ]


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
    shared cache anchor; writes take them only inside the workspace."""
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
    not "use a workspace-relative path"."""
    for path in ("/tmp/refs", "/tmp/refs/x", "/tmp/refs/./x"):
        result = asyncio.run(_registry_for(tmp_path).call(
            "write_file", json.dumps({"path": path, "content": "no"})
        ))
        assert not result.ok, path
        assert "read-only" in result.content, path
    # No dot segment escapes the area — nor is one a way in: it is refused
    # for the same reason as any other path that names a walk.
    result = asyncio.run(_registry_for(tmp_path).call(
        "write_file", json.dumps({"path": "/tmp/refs/../refs/x", "content": "no"})
    ))
    assert not result.ok
    assert "must not contain '..'" in result.content


def test_write_file_accepts_an_absolute_path_inside_the_workspace(tmp_path) -> None:
    """The workspace's container mount is a path like any other: naming a file
    by it writes that file, and the result names it workspace-relative."""
    result = asyncio.run(_registry_for(tmp_path).call(
        "write_file",
        json.dumps({"path": "/workspace/sub/a.txt", "content": "hi\n"}),
    ))
    assert result.ok, result.content
    assert (tmp_path / "sub" / "a.txt").read_text() == "hi\n"
    assert result.content.startswith("created sub/a.txt")


def test_edit_file_accepts_an_absolute_path_inside_the_workspace(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("one\ntwo\n")
    result = asyncio.run(_registry_for(tmp_path).call(
        "edit_file",
        json.dumps({
            "path": "/workspace/./a.txt",
            "edits": [{"old_string": "two", "new_string": "three"}],
        }),
    ))
    assert result.ok, result.content
    assert (tmp_path / "a.txt").read_text() == "one\nthree\n"
    assert "applied edit to a.txt" in result.content


def test_write_tools_reject_an_absolute_path_outside_the_workspace(tmp_path) -> None:
    for path in ("/etc/passwd", "/tmp/x.txt"):
        result = asyncio.run(_registry_for(tmp_path).call(
            "write_file", json.dumps({"path": path, "content": "no"})
        ))
        assert not result.ok, path
        assert "outside the workspace" in result.content, path
        assert "run_bash" in result.content, path
    assert not (tmp_path.parent / "x.txt").exists()


def test_unsandboxed_write_takes_the_real_absolute_path(tmp_path) -> None:
    """No container: the workspace is the real directory, so that is the
    absolute form that works — /workspace names nothing on the host."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    sandbox = SandboxConfig(workspace=workspace, use_sandbox=False)
    registry = build_registry(sandbox, None, mode="write")

    result = asyncio.run(registry.call(
        "write_file",
        json.dumps({"path": str(workspace / "a.txt"), "content": "hi\n"}),
    ))
    assert result.ok, result.content
    assert (workspace / "a.txt").read_text() == "hi\n"

    reject = asyncio.run(registry.call(
        "write_file", json.dumps({"path": "/workspace/a.txt", "content": "no"})
    ))
    assert not reject.ok
    assert "outside the workspace" in reject.content


def test_the_file_tools_share_one_path_check(tmp_path) -> None:
    """read_file, write_file and edit_file answer a bad path identically: one
    validator, one message, and dotted/doubled segments normalize the same"""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_text("one\ntwo\n")
    registry = _registry_for(tmp_path)
    for result in _every_text_tool(registry, "./sub/../a.txt"):
        assert result.status == "error"
        assert result.content == "path must not contain '..', got './sub/../a.txt'"

    read = asyncio.run(registry.call(
        "read_file", json.dumps({"path": "sub//./a.txt"})))
    assert read.ok and read.content == "one\ntwo\n"
    write = asyncio.run(registry.call(
        "write_file", json.dumps({"path": "sub//./b.txt", "content": "hi\n"})))
    assert write.content.startswith("created sub/b.txt")
    edit = asyncio.run(registry.call("edit_file", json.dumps({
        "path": "./sub/a.txt",
        "edits": [{"old_string": "two", "new_string": "2"}],
    })))
    assert edit.content.startswith("applied edit to sub/a.txt")


def test_a_binary_file_is_refused_by_all_three(tmp_path) -> None:
    """Bytes the text tools cannot carry are refused with the way out,
    instead of being handed back as replacement characters.

    "Binary" is gitwork's probe (magic prefix or a NUL in the head) — the
    same test that keeps the file out of /diff, so the tools and the change
    tracking agree on what a binary is."""
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)
    before = (tmp_path / "logo.png").read_bytes()
    (tmp_path / "nul.dat").write_bytes(b"a\x00b")
    registry = _registry_for(tmp_path)

    for path in ("logo.png", "nul.dat", "/workspace/logo.png"):
        for result in _every_text_tool(registry, path):
            assert result.status == "error", path
            assert "is a binary file" in result.content, path
            assert "run_bash" in result.content, path
            assert "xxd" in result.content, path
    assert (tmp_path / "logo.png").read_bytes() == before  # nothing written

    named = asyncio.run(registry.call(
        "read_file", json.dumps({"path": "logo.png"})))
    assert named.content.startswith("logo.png is a binary file (48 bytes):")


def test_non_utf8_text_is_refused_by_all_three(tmp_path) -> None:
    """latin-1/CP1252 source is text, but not UTF-8: the model would be shown
    U+FFFD where bytes were. All three refuse, so nothing is lost silently."""
    (tmp_path / "latin1.c").write_bytes(b"/* caf\xe9 */\nint x;\n")
    before = (tmp_path / "latin1.c").read_bytes()

    for result in _every_text_tool(_registry_for(tmp_path), "latin1.c"):
        assert result.status == "error"
        assert "not valid UTF-8" in result.content
        assert "run_bash" in result.content
    assert (tmp_path / "latin1.c").read_bytes() == before


def test_utf16_is_refused_as_binary(tmp_path) -> None:
    """UTF-16 is text, but not text these tools can carry: the probe exempts
    the BOM (so git keeps tracking the file as text), so the tools refuse it
    separately."""
    (tmp_path / "u16.txt").write_bytes("hi\n".encode("utf-16"))
    for result in _every_text_tool(_registry_for(tmp_path), "u16.txt"):
        assert result.status == "error"
        assert "UTF-16" in result.content
        assert "run_bash" in result.content


def test_a_valid_utf8_file_with_a_replacement_char_is_not_refused(tmp_path) -> None:
    """The check is on the file's bytes, not on the character: a file that
    really contains U+FFFD is valid UTF-8 and reads normally."""
    (tmp_path / "u.txt").write_text("a\ufffdb\n")
    result = asyncio.run(_registry_for(tmp_path).call(
        "read_file", json.dumps({"path": "u.txt"})))
    assert result.ok, result.content
    assert result.content == "a\ufffdb\n"


def test_a_directory_is_refused_by_all_three(tmp_path) -> None:
    """A path that names a directory is refused up front, not by the shell
    (whose message names the binary that ran, not the mistake)."""
    (tmp_path / "dir").mkdir()
    for result in _every_text_tool(_registry_for(tmp_path), "dir"):
        assert result.status == "error"
        assert result.content == "dir is a directory, not a file"


def test_dotdot_is_refused_for_reads_too(tmp_path) -> None:
    """One rule for every tool: no path is read as a walk — so
    /workspace/../etc/passwd is refused like ../etc/passwd, even though
    /etc/passwd itself is readable."""
    registry = _registry_for(tmp_path)
    for path in ("../up.txt", "/workspace/../etc/passwd", "/tmp/refs/../x"):
        result = asyncio.run(registry.call("read_file", json.dumps({"path": path})))
        assert not result.ok, path
        assert result.status == "error", path
        assert "must not contain '..'" in result.content, path
    assert asyncio.run(registry.call(
        "read_file", json.dumps({"path": "/etc/passwd"}))).ok


def test_write_still_creates_a_new_file(tmp_path) -> None:
    """The probe looks at the file that is there; a new file (even one with
    a binary-sounding name) is written as usual."""
    result = asyncio.run(_registry_for(tmp_path).call(
        "write_file", json.dumps({"path": "data.bin", "content": "text\n"})
    ))
    assert result.ok, result.content
    assert (tmp_path / "data.bin").read_text() == "text\n"


def test_unsandboxed_refuses_a_binary_host_path(tmp_path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)
    sandbox = SandboxConfig(workspace=workspace, use_sandbox=False)
    registry = build_registry(sandbox, None, mode="write")

    for path in ("logo.png", str(workspace / "logo.png")):
        result = asyncio.run(registry.call("read_file", json.dumps({"path": path})))
        assert not result.ok, path
        assert "is a binary file" in result.content, path


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
