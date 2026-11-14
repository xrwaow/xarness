"""Tool registry and the built-in tools.

read_file/write_file/edit_file/run_bash execute inside a bwrap sandbox (see
sandbox.py) scoped to one workspace directory. run_bash uses a persistent
SandboxSession so shell state survives across calls within one chat.

Plan mode exposes read_file plus run_bash against a read-only workspace
mount; write mode exposes read_file plus the editing tools (write_file,
edit_file). ask_user is available in both modes, and subagent in both modes
when the caller can start one.

A tool whose text names the machine it runs on is described by the factory
that builds it, which is handed the sandbox the tool will run against — so
the model is never told about a container that is not there, or the other way
around (see :func:`_make_run_bash_tool`).

Every tool declares a pydantic args model: the JSON schema for the provider
is generated from it, and :meth:`ToolRegistry.call` validates the call's
arguments with it in one place — so handlers read ``args.x`` (typed, with
defaults applied) instead of re-checking types.
"""

from __future__ import annotations

import copy
import difflib
import json
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .conversation import AskQuestion, ToolResult, Usage
from .mcp import McpManager, McpToolMeta
from .sandbox import (
    DEFAULT_TIMEOUT,
    REFS_ANCHOR,
    SandboxConfig,
    SandboxSession,
    is_refs_path,
    run_in_sandbox,
)


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    args_model: "type[BaseModel]"
    handler: "Callable[[BaseModel], Awaitable[ToolResult]]"
    # The provider-facing JSON schema, computed once at construction (it is
    # immutable; regenerating it per tool per round was pure waste).
    schema: dict[str, Any] = field(default_factory=dict)
    # Where the tool came from: "builtin" for the harness's own tools, or
    # the MCP server's name. The UI groups by it; the model never sees it.
    source: str = "builtin"

    def __post_init__(self) -> None:
        """Derive the provider schema from the args model.

        Here (not in the registry) so every Tool has one — the decorator and
        a hand-built Tool produce the same shape."""
        if not self.schema:
            self.schema = ToolRegistry._schema_of(self)


def tool(name: str, description: str, args_model: "type[BaseModel]") -> Callable:
    """Decorator: pair an async handler with its pydantic args model.

    The provider-facing JSON schema is generated from the model, and the
    registry validates each call's arguments with it, so the handler
    receives typed, defaulted fields instead of a raw dict."""

    def deco(handler):
        return Tool(
            name=name, description=description,
            args_model=args_model, handler=handler,
        )

    return deco


# Per-tool cap on the text sent back to the model (config key
# "tool_output_limit"). Applies to every tool via ToolRegistry.call.
DEFAULT_OUTPUT_LIMIT = 32768


def _truncate_output(text: str, limit: int) -> str:
    """Head + tail with the middle elided: for a failing test run the tail
    is the interesting part, for a big listing the head is."""
    if len(text) <= limit:
        return text
    marker = (
        f"\n[... {len(text) - limit} chars elided (limit {limit}); narrow the "
        "command or read a range ...]\n"
    )
    if len(marker) >= limit:
        return marker[:limit]
    keep = limit - len(marker)
    head = keep * 2 // 3
    return text[:head] + marker + text[-keep + head:]


def _ok(content: str) -> ToolResult:
    return ToolResult(status="ok", content=content)


def _bad(error: str) -> ToolResult:
    """An argument error: the model should retry with valid arguments."""
    return ToolResult(status="parse_error", content=error)


# ----------------------------------------------------------------------
# per-tool argument models (schema + validation in one place)


class _ToolArgs(BaseModel):
    """Base for tool arguments: unknown keys are IGNORED (a model emitting a
    stray key must not fail the call — the schema does not advertise
    ``additionalProperties: false`` either), and models emitting digit
    strings for integers are handled by pydantic's lax coercion."""

    model_config = ConfigDict(extra="ignore")


class ReadFileArgs(_ToolArgs):
    path: str
    start_line: int | None = Field(
        default=None,
        description="1-based first line to read (omit to start at line 1)")
    end_line: int | None = Field(
        default=None,
        description="1-based last line to read, inclusive (omit to read to EOF)")


class WriteFileArgs(_ToolArgs):
    path: str
    content: str = Field(description="the full new file contents")


class EditSpec(_ToolArgs):
    old_string: str = Field(description="the exact text to replace")
    new_string: str = Field(description="the replacement text")
    replace_all: bool = Field(
        default=False,
        description="replace every occurrence (default: the old_string must be unique)")


class EditFileArgs(_ToolArgs):
    path: str
    edits: list[EditSpec] = Field(
        description="replacements to apply, in order (all or nothing)")


class RunBashArgs(_ToolArgs):
    command: str
    timeout_seconds: int | None = Field(
        default=None, ge=1,
        description="kill the command after this many seconds; on timeout the "
                    "shell is restarted and the output so far is returned")


class AskUserArgs(_ToolArgs):
    questions: list[AskQuestion] = Field(description="the questions to ask")


class SubagentArgs(_ToolArgs):
    prompt: str = Field(
        description="the whole task, as one self-contained message; the subagent "
                    "has no access to this conversation")
    mode: Literal["plan", "write"] = Field(
        default="write",
        description="'plan' explores read-only, 'write' may edit files in the "
                    "shared workspace")


class PlanSubagentArgs(SubagentArgs):
    """A plan-mode agent's variant: defaulting to a plan child, since it can
    never start a write one (the handler refuses, see _make_subagent_tool)."""

    mode: Literal["plan", "write"] = Field(
        default="plan", description="'plan' explores read-only")


# Output sink for the call in flight: registry.call sets it around the
# handler await, run_bash forwards streamed shell lines to it so the TUI can
# show live output while the command runs.
_output_sink: ContextVar[Callable[[str], None] | None] = ContextVar(
    "tool_output_sink", default=None
)

# The registry's output cap during the call in flight. read_file sizes its
# line window to it, so a long range is not later spliced head/tail by
# _truncate_output (which would contradict its own "[showing lines …]" footer).
_output_limit: ContextVar[int | None] = ContextVar(
    "tool_output_limit", default=None
)

# Usage sink for the call in flight (a subagent reports its child's spend).
_usage_sink: ContextVar[Callable[[Usage], None] | None] = ContextVar(
    "tool_usage_sink", default=None
)


def report_tool_usage(usage: Usage) -> None:
    """Report spend for the tool call running in this context (a no-op outside)."""
    if (sink := _usage_sink.get()) is not None:
        sink(usage)


def _exec_error(result, fallback: str) -> ToolResult:
    return ToolResult(status="error", content=result.stderr.strip() or fallback)


# ----------------------------------------------------------------------
# argument validation


@dataclass(slots=True)
class ToolRegistry:
    """Named tool set exposed to the model.

    ``max_output_chars`` caps the text of every tool result before it goes
    on the wire (see _truncate_output); set from the config's
    "tool_output_limit". It is also published to the running handler as
    ``_output_limit`` so a tool that returns structured line ranges
    (read_file) can size itself to the cap instead of being cut mid-way.
    """

    max_output_chars: int = DEFAULT_OUTPUT_LIMIT
    _tools: dict[str, Tool] = field(default_factory=dict)

    @staticmethod
    def _flatten_schema(node: Any, defs: dict[str, Any]) -> Any:
        """Make a JSON schema palatable to small local servers.

        llama.cpp and older vLLM reject ``anyOf: [..., null]``, ``$defs``
        and ``$ref``. Optional fields become their plain type (null is
        optional, not a type), and ``$ref``s are inlined — tool argument
        models are tiny, so duplication is harmless. ``title``s are dropped
        at every level: noise for the model, and otherwise two identical
        nested shapes differ by the pydantic class name."""
        if isinstance(node, list):
            return [ToolRegistry._flatten_schema(item, defs) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            ref = node.pop("$ref")
            target = defs.get(ref.rsplit("/", 1)[-1], {})
            merged = {
                k: ToolRegistry._flatten_schema(v, defs)
                for k, v in {**target, **node}.items()
            }
            merged.pop("title", None)
            return merged
        if "anyOf" in node and len(node["anyOf"]) > 1:
            variants = [v for v in node["anyOf"] if v.get("type") != "null"]
            rest = {k: v for k, v in node.items() if k != "anyOf"}
            if len(variants) == 1:
                # Optional single type: flatten to that type, keeping the
                # other keys (default, description, …).
                return ToolRegistry._flatten_schema({**variants[0], **rest}, defs)
            return ToolRegistry._flatten_schema({**node, "anyOf": variants}, defs)
        return {
            k: ToolRegistry._flatten_schema(v, defs)
            for k, v in node.items() if k not in ("$defs", "title")
        }

    @staticmethod
    def _schema_of(tool: Tool) -> dict[str, Any]:
        """The provider-facing ``parameters`` object for one tool.

        Generated from the args model and flattened for local servers (see
        :meth:`_flatten_schema`, which also drops every ``title``)."""
        schema = tool.args_model.model_json_schema()
        defs = schema.pop("$defs", {})
        return ToolRegistry._flatten_schema(schema, defs)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def tools(self) -> list[Tool]:
        """The registered tools, in registration order."""
        return list(self._tools.values())

    def schema(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema,
                },
            }
            for tool in self._tools.values()
        ]

    async def call(
        self,
        name: str,
        arguments_json: str,
        output_sink: Callable[[str], None] | None = None,
        usage_sink: Callable[[Usage], None] | None = None,
    ) -> ToolResult:
        """Execute one call. ``output_sink`` streams live output (run_bash);
        ``usage_sink`` gets spend reports from a self-running tool (subagent)."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(status="error", content=f"unknown tool '{name}'")
        try:
            args = json.loads(arguments_json) if arguments_json.strip() else {}
        except json.JSONDecodeError as exc:
            return ToolResult(status="parse_error", content=f"invalid arguments JSON: {exc}")
        if not isinstance(args, dict):
            return _bad(
                f"invalid arguments JSON: expected an object, got {type(args).__name__}"
            )
        try:
            args = tool.args_model.model_validate(args)
        except ValidationError as exc:
            return _bad("; ".join(_validation_error(e) for e in exc.errors()))
        try:
            token = _output_sink.set(output_sink)
            limit_token = _output_limit.set(self.max_output_chars)
            usage_token = _usage_sink.set(usage_sink)
            try:
                result = await tool.handler(args)
            finally:
                _usage_sink.reset(usage_token)
                _output_limit.reset(limit_token)
                _output_sink.reset(token)
        except Exception as exc:
            result = ToolResult(status="error", content=f"{type(exc).__name__}: {exc}")
        result = result._replace(
            content=_truncate_output(result.content, self.max_output_chars),
        )
        return result


def _validation_error(error: dict[str, Any]) -> str:
    """One pydantic ValidationError entry as a short, model-readable line:
    ``path.start_line: Input should be a valid integer``."""
    loc = ".".join(str(part) for part in error.get("loc", ()))
    message = error.get("msg", "invalid value")
    return f"{loc}: {message}" if loc else message


# ----------------------------------------------------------------------
# read_file

# Files larger than this return a structural outline instead of contents
# (unless the call passes explicit line numbers).
OUTLINE_THRESHOLD = 500
# Upper bound on lines returned by one ranged read.
MAX_READ_LINES = 2000
# Upper bound on bytes fetched for a non-ranged read (outline / whole file).
# Well above any real source file; a larger one is read by range instead.
READ_CAP = 4 * 1024 * 1024
# Lines of a large, non-Python file shown when no range is given.
PREVIEW_LINES = 200


def _python_outline(source: str) -> str | None:
    """One line per class/def with its 1-based line range, methods indented.

    Returns None for non-Python (or symbol-less) files, so the caller can
    fall back to a plain truncated preview.
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    lines: list[str] = []

    def walk(node: ast.AST, depth: int) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                lines.append(f"{' ' * depth}class {child.name} [L{child.lineno}-{child.end_lineno}]")
                walk(child, depth + 1)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                keyword = "async def" if isinstance(child, ast.AsyncFunctionDef) else "def"
                lines.append(
                    f"{' ' * depth}{keyword} {child.name} [L{child.lineno}-{child.end_lineno}]"
                )
                walk(child, depth + 1)

    walk(tree, 0)
    return "\n".join(lines) if lines else None


def _check_path(sandbox: SandboxConfig, path: str) -> ToolResult | None:
    error = sandbox.validate_relpath(path)
    return None if error is None else _bad(error)


def _check_write_path(sandbox: SandboxConfig, path: str) -> ToolResult | None:
    """Workspace-relative path checks shared by the writing tools: no
    absolute paths except the read-only refs area, no traversal."""
    if sandbox.use_sandbox and is_refs_path(path):
        return ToolResult(status="error", content=f"'{REFS_ANCHOR}/' is read-only")
    return _check_path(sandbox, path)


def _line_window(
    lines: list[str], start: int, end: int, total: int, limit: int | None,
) -> str:
    """Join a file's ``lines`` (its ``start..end`` slice, 1-based), keeping
    the longest whole-line prefix that fits ``limit`` — contiguously, so
    read_file never hands back a head/tail splice with a hole in it — and
    say in the footer where to resume."""
    def render(kept: int) -> str:
        last = start + kept - 1
        if not kept or last >= total:  # nothing left over: plain contents
            return "".join(lines[:kept])
        note = "" if last >= end else f"; output limit reached — continue from line {last + 1}"
        return "".join(lines[:kept]) + f"\n[showing lines {start}-{last} of {total}{note}]"

    text = render(len(lines))
    if limit is None or len(text) <= limit:
        return text
    # The longest whole-line prefix of the contents that fits the budget:
    # cut at the last newline inside it (never mid-line), then count the
    # lines kept.
    contents = "".join(lines)
    kept = contents[:contents.rfind("\n", 0, limit) + 1].count("\n")
    if kept <= 0:
        # A lone line longer than the whole budget: let _truncate_output cut it.
        return render(1)
    # The footer the cut introduces must fit too.
    while kept > 1 and len(render(kept)) > limit:
        kept -= 1
    return render(kept)


def _make_read_tool(sandbox: SandboxConfig, default_timeout: float) -> Tool:
    # What the workspace root is, and what an absolute path can reach, differ
    # per environment; the rest of the description holds either way.
    root, reach = (
        (" (/workspace)", "e.g. /cache/...") if sandbox.use_sandbox
        else (", which is your project directory", "e.g. the user's home directory")
    )

    @tool(
        "read_file",
        "Read a file's contents. Paths are relative to the workspace root"
        f"{root}. Absolute paths are accepted for reads outside it ({reach}). "
        f"With no range, files over {OUTLINE_THRESHOLD} lines return an outline "
        "(class/def signatures with line ranges for Python, the first "
        f"{PREVIEW_LINES} lines otherwise) instead of the full contents; pass "
        "start_line and/or end_line to read a section of any file, however "
        "large.",
        ReadFileArgs,
    )
    async def read_file(args: ReadFileArgs) -> ToolResult:
        path = args.path
        if Path(path).is_absolute():
            # Taken as given: the environment bounds it, not this check. The
            # container mounts the workspace, the caches and the refs and
            # nothing else; the host has no jail, exactly like run_bash there.
            target = path
        elif err := _check_path(sandbox, path):
            return err
        else:
            target = sandbox.tool_path(path)

        start = args.start_line
        end = args.end_line
        ranged = start is not None or end is not None
        start = start or 1
        if start < 1:
            return _bad("start_line is 1-based")

        if ranged:
            # One shell: total line count (awk counts a last line without a
            # trailing newline) followed by the slice.
            command = [
                "sh", "-c",
                'n=$(awk \'END{print NR}\' "$1"); printf "%d\\n" "$n"; '
                'sed -n "${2},${3}p" -- "$1"',
                "read", target, str(start), str(end if end is not None else 10**9),
            ]
        else:
            command = ["head", "-c", str(READ_CAP), "--", target]
        result = await run_in_sandbox(sandbox, command, timeout=default_timeout)
        if result.exit_code != 0:
            return _exec_error(result, "read failed")
        if ranged:
            first_nl = result.stdout.find("\n")
            try:
                total_lines = int(result.stdout[:first_nl].strip())
            except ValueError:
                return ToolResult(status="error", content=f"could not count lines in {path}")
            contents = result.stdout[first_nl + 1:]
            if end is None or end > total_lines:
                end = total_lines
            if start > total_lines:
                return _bad(f"start_line {start} is past the end of the file ({total_lines} lines)")
            if end < start:
                return _bad(f"end_line ({end}) is before start_line ({start})")
            if end - start + 1 > MAX_READ_LINES:
                end = start + MAX_READ_LINES - 1
            # keepends: preserve the exact trailing-newline behavior of sed.
            return _ok(_line_window(
                contents.splitlines(keepends=True)[: end - start + 1],
                start, end, total_lines, _output_limit.get(),
            ))

        contents = result.stdout
        truncated = len(contents.encode(errors="replace")) >= READ_CAP
        total_lines = contents.count("\n") + (0 if not contents or contents.endswith("\n") else 1)
        if not truncated and total_lines <= OUTLINE_THRESHOLD:
            # A full read is just the 1..total_lines range: under the output
            # cap it must still come back contiguous, not spliced.
            return _ok(_line_window(
                contents.splitlines(keepends=True),
                1, total_lines, total_lines, _output_limit.get(),
            ))
        return _outline_result(path, total_lines, contents, truncated)

    return read_file


def _outline_result(path: str, total_lines: int, contents: str, truncated: bool) -> ToolResult:
    """Outline (Python) or preview (anything else) of a large file."""
    if truncated:
        return _ok((
            f"[this file is larger than {READ_CAP // (1024 * 1024)} MiB, so only "
            "its first part was read; pass start_line/end_line to read specific "
            "sections]\n"
            f"{_preview(contents)}"
        ))
    outline = _python_outline(contents) if path.lower().endswith(".py") else None
    if outline is None:
        return _ok((
            f"[showing lines 1-{min(PREVIEW_LINES, total_lines)} of {total_lines}; this "
            "file is too large to read all at once and has no outline — pass "
            "start_line/end_line to read specific sections]\n"
            f"{_preview(contents)}"
        ))
    return _ok((
        "File outline retrieved. This file is too large to read all at once, so "
        "the outline below shows the file's structure with line numbers.\n"
        "\n"
        "IMPORTANT: Do NOT retry this call without line numbers - you will get "
        "the same outline.\n"
        "Instead, use the line numbers below to read specific sections by calling "
        "this tool again with start_line and end_line parameters.\n"
        "\n"
        f"# File outline for {path}\n"
        "\n"
        f"{outline}\n"
        "\n"
        "NEXT STEPS: To read a specific symbol's implementation, call read_file "
        "with the same path plus start_line and end_line from the outline above.\n"
        "For example, to read a function shown as [L100-150], use start_line: 100 "
        "and end_line: 150."
    ))


def _preview(contents: str) -> str:
    lines = contents.splitlines()[:PREVIEW_LINES]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# write_file


def _make_write_tool(sandbox: SandboxConfig, default_timeout: float) -> Tool:
    @tool(
        "write_file",
        "Create a new file or overwrite an existing one with completely new "
        "contents (parent directories are created as needed). Prefer "
        "edit_file for changing part of an existing file.",
        WriteFileArgs,
    )
    async def write_file(args: WriteFileArgs) -> ToolResult:
        path = args.path
        content = args.content
        if err := _check_write_path(sandbox, path):
            return err

        target = sandbox.tool_path(path)
        # One bwrap spawn, not two: report whether the file existed before
        # ("created" vs "overwrote") and write it in the same shell. The
        # content travels on stdin (tee), so no quoting issues.
        result = await run_in_sandbox(
            sandbox,
            ["sh", "-c",
             'mkdir -p -- "$(dirname -- "$1")" && '
             '{ if [ -e "$1" ]; then echo existing; else echo new; fi; '
             'tee -- "$1" >/dev/null; }',
             "write_file", target],
            timeout=default_timeout,
            input_bytes=content.encode(),
        )
        if result.exit_code != 0:
            return _exec_error(result, "write failed")
        verb = "overwrote" if result.stdout.startswith("existing") else "created"
        return _ok(f"{verb} {path} ({len(content.encode())} bytes)")

    return write_file


# ----------------------------------------------------------------------
# edit_file

# Upper bound on the diff lines echoed back by edit_file.
_MAX_EDIT_DIFF_LINES = 400


class _EditError(Exception):
    """One edit in a batch did not match; nothing was written."""


def _apply_edits(current: str, edits: "list[EditSpec]") -> str:
    """Apply every edit in order, or raise: the batch is all-or-nothing.

    Each edit is matched against the text the previous ones produced (so a
    later edit can target an earlier edit's result), and must match exactly
    once unless ``replace_all`` is set.
    """
    updated = current
    for index, edit in enumerate(edits, 1):
        where = f"edit {index}" if len(edits) > 1 else "old_string"
        occurrences = updated.count(edit.old_string)
        if occurrences == 0:
            raise _EditError(f"{where}: old_string not found in file")
        if occurrences > 1 and not edit.replace_all:
            raise _EditError(
                f"{where}: old_string is not unique ({occurrences} matches); "
                "include more surrounding context, or set replace_all"
            )
        updated = updated.replace(
            edit.old_string, edit.new_string,
            occurrences if edit.replace_all else 1,
        )
    return updated


def _format_edit_diff(path: str, before: str, after: str) -> str:
    """Git-style unified diff of one edit call's changes.

    Returned as the tool's output so both the model (to verify what it
    actually changed) and the TUI (expanding the tool block) can see the
    changed lines rather than just an "applied" acknowledgement.
    """
    # keepends distinguishes "x" from "x\n" (newline-only edits).
    lines = list(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
            lineterm="",
        )
    )
    if not lines:
        return "(no textual change)"
    if len(lines) > _MAX_EDIT_DIFF_LINES:
        lines = lines[:_MAX_EDIT_DIFF_LINES] + [
            f"\n[... diff truncated at {_MAX_EDIT_DIFF_LINES} lines]"
        ]
    # ``lineterm=""`` keeps content line endings intact, but leaves the
    # ``---``/``+++``/``@@`` headers without one. Normalise every element to
    # its own line, or the parser cannot see the hunk at all (and renders
    # an empty body for edit_file).
    diff = "\n".join(line.rstrip("\n") for line in lines)
    return f"diff --git a/{path} b/{path}\n" + diff.rstrip("\n") + "\n"


def _make_edit_tool(sandbox: SandboxConfig, default_timeout: float) -> Tool:
    @tool(
        "edit_file",
        "Edit an existing file by replacing exact strings with new text. Pass "
        "'edits': a list of {old_string, new_string} replacements, applied in "
        "order — batch every change you want to make to the same file into one "
        "call. Each old_string must appear exactly once (add 'replace_all': "
        "true to replace every occurrence, e.g. for a rename), and must match "
        "the file exactly: include a few surrounding lines to disambiguate. "
        "The batch is all-or-nothing — if any edit fails, nothing is written "
        "and the error identifies the offending edit (e.g. `edit 2: old_string "
        "not found in file`). The result includes a diff of what "
        "changed. Use write_file to create a file or replace its whole "
        "contents; use run_bash on non-UTF-8 files.",
        EditFileArgs,
    )
    async def edit_file(args: EditFileArgs) -> ToolResult:
        path = args.path
        if err := _check_write_path(sandbox, path):
            return err

        edits = args.edits
        if not edits:
            return _bad("'edits' must be a non-empty list")
        for index, edit in enumerate(edits, 1):
            if not edit.old_string:
                return _bad(f"edit {index} needs a non-empty 'old_string'")

        read_result = await run_in_sandbox(
            sandbox, ["cat", "--", sandbox.tool_path(path)], timeout=default_timeout)
        if read_result.exit_code != 0:
            return _exec_error(read_result, "read failed")
        current = read_result.stdout
        if "\ufffd" in current:
            # run_in_sandbox decodes with errors="replace": writing this back
            # would bake U+FFFD into the user's file.
            return ToolResult(
                status="error",
                content=f"{path} is not valid UTF-8; edit it with run_bash instead",
            )

        try:
            updated = _apply_edits(current, edits)
        except _EditError as exc:
            return ToolResult(status="error", content=str(exc))

        write_result = await run_in_sandbox(
            sandbox,
            ["sh", "-c", 'tee -- "$1" >/dev/null', "edit_file", sandbox.tool_path(path)],
            input_bytes=updated.encode(), timeout=default_timeout,
        )
        if write_result.exit_code != 0:
            return _exec_error(write_result, "write failed")
        plural = "edit" if len(edits) == 1 else f"{len(edits)} edits"
        return _ok(f"applied {plural} to {path}\n{_format_edit_diff(path, current, updated)}")

    return edit_file


# ----------------------------------------------------------------------
# run_bash


def _make_run_bash_tool(session: SandboxSession) -> Tool:
    # Which machine the command lands on is the one thing the model must not
    # get wrong; the rest of the description holds in both environments.
    where = (
        "inside the sandboxed workspace, as /bin/sh. "
        if session.config.use_sandbox
        else "on the host, as /bin/sh: it runs directly in the user's project, "
        "on the user's machine, with the user's own permissions. "
    )

    # Where the shell comes up, and that it stays there across calls: an
    # agent cannot see its cwd, so it `cd`s to the root before every command
    # unless the description says it is already there.
    root = session.config.tool_root

    @tool(
        "run_bash",
        "Run a shell command " + where +
        f"The shell starts in {root} and keeps its cwd between calls — a "
        "timeout or a mode switch restarts it. Exported variables stick the same way. "
        f"Commands are killed after {int(session.default_timeout)}s unless "
        "timeout_seconds says otherwise. "
        "The command's stdin is /dev/null, so interactive commands read EOF "
        "instead of hanging. The command runs as one shell string, so the exit "
        "code is the last stage's alone — a `grep` with no match or a `test "
        "-f` on a missing file fails despite earlier steps succeeding, and "
        "piping into `tail` or `head` reports 0 even when the command before "
        "it failed. Every result ends with an `[exit N]` status line.",
        RunBashArgs,
    )
    async def run_bash(args: RunBashArgs) -> ToolResult:
        command = args.command
        if not command.strip():
            # Whitespace-only would make the persistent-shell group below an
            # empty `{ }`, a syntax error that kills the shell.
            return _bad("'command' is required")
        timeout = args.timeout_seconds
        result = await session.run(command, on_output=_output_sink.get(), timeout=timeout)
        if result.timed_out:
            label = f" after {timeout}s" if timeout is not None else ""
            return ToolResult(
                status="error",
                content=(
                    f"{result.stdout}\n[command timed out{label} (shell "
                    "restarted); the rest of the command did not run; the "
                    "output above is what it printed before the timeout]"
                ),
            )
        # Status line on every result: empty output plus a non-zero exit is
        # otherwise indistinguishable from success with no output. A failing
        # call keeps its output — the model must see what the command printed.
        output = result.stdout
        if output and not output.endswith("\n"):
            output += "\n"
        return ToolResult(
            status="ok" if result.exit_code == 0 else "error",
            content=f"{output}[exit {result.exit_code}]",
        )

    return run_bash


# ----------------------------------------------------------------------
# ask_user


def _answer_text(question: AskQuestion, answers: list[str], index: int) -> str:
    """One answer as text: a number picks the choice it names (a client sends
    the key the user typed, and people type ``4)`` and ``4.`` as readily as
    ``4``), and a missing answer reads as unanswered rather than vanishing
    from the pairing."""
    answer = answers[index].strip() if index < len(answers) else ""
    key = answer.rstrip(").:-").strip()  # a trailing separator is still a pick
    if key.isdigit() and 1 <= int(key) <= len(question.options):
        return question.options[int(key) - 1]
    return answer or "(no answer given)"


def _make_ask_user_tool(
    ask_callback: Callable[[list[AskQuestion]], Awaitable[list[str] | None]],
) -> Tool:
    """ask_user: hand questions to the user and return their answers as the result."""

    @tool(
        "ask_user",
        "Ask the user questions and wait for their answers. Use when you "
        "need a decision, a missing detail, or confirmation. Each item in "
        "'questions' is one self-contained question. Put its choices in that "
        "question's 'options' field, not as separate items; omit 'options' "
        "for an open question. The user may always write their own answer "
        "instead of picking one.",
        AskUserArgs,
    )
    async def ask_user(args: AskUserArgs) -> ToolResult:
        questions = [q for q in args.questions if q.question.strip()]
        if not questions:
            return _bad("'questions' must be a non-empty list")
        answers = await ask_callback(questions)
        if answers is None:
            return _ok("(user skipped the questions — no answers given)")
        pairs = [
            f"Q: {question.question.strip()}\n"
            f"A: {_answer_text(question, answers, index)}"
            for index, question in enumerate(questions)
        ]
        return _ok("\n\n".join(pairs))

    return ask_user


# ----------------------------------------------------------------------
# subagent


def _make_subagent_tool(
    mode: str, spawn: Callable[[str, str], Awaitable[str]],
) -> Tool:
    """subagent: run one fresh agent in the shared workspace, return its answer.

    ``mode`` is the spawning agent's mode: a plan-mode agent is read-only
    itself, so its args model defaults to plan and a write child is refused."""
    description = (
        "Run a fresh agent on one self-contained task in the same workspace "
        "and return its final answer. It does not see this conversation and "
        "cannot ask the user, so put everything it needs in the prompt. Use "
        "it for a bounded side task (a search, a review, a refactor of one "
        "area) whose back and forth should stay out of this conversation."
    )
    if mode != "plan":
        description += (
            " mode='plan' explores with read-only tools; mode='write' may edit "
            "files in the workspace."
        )

    @tool(
        "subagent", description,
        SubagentArgs if mode != "plan" else PlanSubagentArgs,
    )
    async def subagent(args: SubagentArgs) -> ToolResult:
        prompt = args.prompt.strip()
        if not prompt:
            return _bad("'prompt' must be a non-empty string")
        if mode == "plan" and args.mode != "plan":
            return _bad(
                "a plan-mode agent is read-only: it can only spawn plan "
                "subagents (mode='plan')"
            )
        reply = await spawn(prompt, args.mode)
        if not reply:
            return ToolResult(
                status="error", content="the subagent finished without a reply",
            )
        return _ok(reply)

    return subagent


def build_registry(
    sandbox: SandboxConfig | None,
    session: SandboxSession | None,
    mode: str = "write",
    ask_callback: Callable[[list[AskQuestion]], Awaitable[list[str] | None]] | None = None,
    default_timeout: float = DEFAULT_TIMEOUT,
    spawn_subagent: Callable[[str, str], Awaitable[str]] | None = None,
    mcp: McpManager | None = None,
) -> ToolRegistry:
    """Build the tool set for one session.

    Plan mode is read-only: read_file plus run_bash against a read-only
    workspace mount. Write mode adds the editing tools: write_file,
    edit_file. ``ask_user`` is available in both modes when ``ask_callback``
    is given, and ``subagent`` when the caller can start one (``spawn_subagent``;
    a subagent itself gets neither). The read-only-git policy is enforced by
    the PATH shim (gitwork.git_shim_script), not here.

    ``mcp`` adds the tools of connected MCP servers, namespaced and tagged
    with their server as ``source``. They run on the host, outside the
    sandbox, so they are available in plan mode too (the server itself
    cannot be made read-only).

    Does NOT touch ``sandbox.read_only`` — the caller owns the sandbox's mode
    (set it to ``mode != "write"`` before calling; a persistent
    SandboxSession restarts its shell to pick up the new mounts).
    """
    registry = ToolRegistry()
    if ask_callback is not None:
        registry.register(_make_ask_user_tool(ask_callback))
    if spawn_subagent is not None:
        registry.register(_make_subagent_tool(mode, spawn_subagent))
    if sandbox is not None:
        registry.register(_make_read_tool(sandbox, default_timeout))  # read_file always available
        # Plan mode is fully read-only: the workspace is mounted read-only
        # (sandbox.read_only) and the persistent shell restarts under those
        # mounts, so run_bash cannot change files. Write mode drops the
        # read-only mount and adds the editing tools.
        #
        # EXCEPT when the sandbox is unsandboxed (use_sandbox=False):
        # there is no read-only mount to enforce plan mode with — commands
        # would run unsandboxed on the host — so run_bash is NOT available
        # in plan mode at all (read_file still is). filter_list,
        # allow_network and gpu_access are likewise ignored in that mode
        # (see SandboxConfig.use_sandbox).
        if session is not None and (sandbox.use_sandbox or mode != "plan"):
            registry.register(_make_run_bash_tool(session))  # run_bash
        if mode == "write":
            registry.register(_make_write_tool(sandbox, default_timeout))  # write_file
            registry.register(_make_edit_tool(sandbox, default_timeout))  # edit_file
    if mcp is not None:
        for tool in mcp_tools(mcp):
            registry.register(tool)
    return registry


class _McpArgs(BaseModel):
    """Permissive argument carrier for MCP tools.

    The server owns its real input schema, so the harness advertises that
    schema verbatim (``Tool.schema``) and does not second-guess the shape
    here: extra keys survive into ``model_dump()`` and the server validates.
    """

    model_config = ConfigDict(extra="allow")


def mcp_tools(manager: McpManager) -> list[Tool]:
    """Wrap every enabled tool of every connected MCP server as a Tool."""
    def tool_for(meta: McpToolMeta) -> Tool:
        async def handler(args: BaseModel) -> ToolResult:
            return await manager.call(meta.server, meta.name, args.model_dump())

        description = meta.description.strip() or meta.name
        return Tool(
            name=meta.exposed_name,
            description=(
                description if description.lower().startswith("mcp")
                else f"MCP ({meta.server}): {description}"
            ),
            args_model=_McpArgs,
            handler=handler,
            schema=_mcp_schema(meta.input_schema),
            source=meta.server,
        )

    return [tool_for(meta) for meta in manager.tool_specs()]


def _mcp_schema(input_schema: dict[str, Any]) -> dict[str, Any]:
    """Make a server's input schema palatable to the provider's tool API.

    The registry's flattener already exists for pydantic schemas (local
    servers reject ``$ref``/``$defs``), so reuse it rather than writing a
    second dialect. Anything that is not a usable JSON object schema falls
    back to an empty object schema (the server still validates the call).
    """
    empty: dict[str, Any] = {"type": "object", "properties": {}}
    if not isinstance(input_schema, dict) or not input_schema:
        return empty
    schema = copy.deepcopy(input_schema)
    defs = schema.pop("$defs", {}) or schema.pop("definitions", {})
    try:
        flat = ToolRegistry._flatten_schema(schema, defs)
    except Exception:
        return empty
    if not isinstance(flat, dict):
        return empty
    return {"type": "object", "properties": {}, **flat}
