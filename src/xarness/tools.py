"""Tool registry and the built-in tools.

read_file/write_file/edit_file/run_bash execute inside a bwrap sandbox (see
sandbox.py) scoped to one workspace directory. run_bash uses a persistent
SandboxSession so shell state survives across calls within one chat.

Plan mode exposes read_file plus run_bash against a read-only workspace
mount; write mode exposes read_file plus the editing tools (write_file,
edit_file). ask_user is available in both modes.

Every call is validated against the tool's own ``parameters_schema`` in one
place (:meth:`ToolRegistry.call`), so handlers can read ``args["x"]``
directly instead of re-checking types.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .sandbox import (
    REFS_ANCHOR, SandboxConfig, SandboxSession, has_ref_mounts, is_refs_path,
    run_in_sandbox,
)


@dataclass(slots=True)
class ToolResult:
    ok: bool
    output: str = ""
    error: str = ""
    parse_error: bool = False


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    parameters_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[ToolResult]]


def tool(name: str, description: str, required: tuple[str, ...] = (), **params: Any) -> Callable:
    """Decorator: turn an async handler into a Tool with a generated schema.

    Each keyword argument after ``required`` declares one parameter. Its
    value is the JSON-schema type (``path="string"``), a ``(type,
    description)`` pair (``start_line=("integer", "1-based …")``), or a full
    schema dict for structured params (arrays, items, …).
    """
    properties = {
        pname: spec if isinstance(spec, dict)
        else {"type": spec[0], "description": spec[1]} if isinstance(spec, tuple)
        else {"type": spec}
        for pname, spec in params.items()
    }

    def deco(handler):
        return Tool(
            name=name,
            description=description,
            parameters_schema={
                "type": "object",
                "properties": properties,
                "required": list(required),
            },
            handler=handler,
        )

    return deco


# Per-tool cap on the text sent back to the model (config key
# "container"."tool_output_limit"). Applies to every tool via
# ToolRegistry.call.
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


def _bad(error: str) -> ToolResult:
    return ToolResult(ok=False, error=error, parse_error=True)


# Output sink for the call in flight: registry.call sets it around the
# handler await, run_bash forwards streamed shell lines to it so the TUI can
# show live output while the command runs.
_output_sink: ContextVar[Callable[[str], None] | None] = ContextVar(
    "tool_output_sink", default=None
)


def _exec_error(result, fallback: str) -> ToolResult:
    return ToolResult(ok=False, error=result.stderr.strip() or fallback)


# ----------------------------------------------------------------------
# argument validation


def _coerce(path: str, value: Any, schema: dict[str, Any]) -> tuple[Any, str | None]:
    """Check one value against its schema; returns (value, error).

    Only the types the built-in tools actually declare are handled: string,
    integer (accepting digit strings, since models emit ``"50"``), array,
    object. Unknown schemas pass through untouched.
    """
    kind = schema.get("type")
    if kind == "integer":
        if isinstance(value, bool):
            return value, f"'{path}' must be an integer"
        if isinstance(value, int):
            return value, None
        if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
            return int(value.strip()), None
        return value, f"'{path}' must be an integer"
    if kind == "string":
        return (value, None) if isinstance(value, str) else (value, f"'{path}' must be a string")
    if kind == "array":
        if not isinstance(value, list):
            return value, f"'{path}' must be a list"
        items = schema.get("items") or {}
        if items:
            coerced = []
            for index, item in enumerate(value):
                new, error = _coerce(f"{path}[{index}]", item, items)
                if error:
                    return value, error
                coerced.append(new)
            return coerced, None
        return value, None
    if kind == "object":
        if not isinstance(value, dict):
            return value, f"'{path}' must be an object"
        return _validate(value, schema, path)
    return value, None


def _validate(args: dict[str, Any], schema: dict[str, Any], where: str = "") -> tuple[Any, str | None]:
    """Coerce/validate ``args`` against a tool's object schema."""
    properties = schema.get("properties") or {}
    for name in schema.get("required") or []:
        if name not in args:
            prefix = f"{where}." if where else ""
            return args, f"missing required argument '{prefix}{name}'"
    for name, value in list(args.items()):
        sub = properties.get(name)
        if sub is None:
            continue  # undeclared keys are the handler's business
        new, error = _coerce(name, value, sub)
        if error:
            return args, error
        args[name] = new
    return args, None


@dataclass(slots=True)
class ToolRegistry:
    """Named tool set exposed to the model.

    ``max_output_chars`` caps the text of every tool result before it goes
    on the wire (see _truncate_output); set from the config's
    "container"."tool_output_limit".
    """

    max_output_chars: int = DEFAULT_OUTPUT_LIMIT
    _tools: dict[str, Tool] = field(default_factory=dict)

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
                    "parameters": tool.parameters_schema,
                },
            }
            for tool in self._tools.values()
        ]

    async def call(
        self,
        name: str,
        arguments_json: str,
        output_sink: Callable[[str], None] | None = None,
    ) -> ToolResult:
        """Execute one call. ``output_sink``, when given, receives streamed
        output lines while the call runs (run_bash) so a UI can render them
        live; the final ToolResult is unchanged either way."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, error=f"unknown tool '{name}'")
        try:
            args = json.loads(arguments_json) if arguments_json.strip() else {}
        except json.JSONDecodeError as exc:
            return ToolResult(ok=False, error=f"invalid arguments JSON: {exc}", parse_error=True)
        if not isinstance(args, dict):
            return _bad(
                f"invalid arguments JSON: expected an object, got {type(args).__name__}"
            )
        args, error = _validate(args, tool.parameters_schema)
        if error:
            return _bad(error)
        try:
            token = _output_sink.set(output_sink)
            try:
                result = await tool.handler(args)
            finally:
                _output_sink.reset(token)
        except Exception as exc:  # noqa: BLE001
            result = ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        result.output = _truncate_output(result.output, self.max_output_chars)
        result.error = _truncate_output(result.error, self.max_output_chars)
        return result


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
    if sandbox.use_container and is_refs_path(path):
        return ToolResult(ok=False, error=f"'{REFS_ANCHOR}/' is read-only")
    return _check_path(sandbox, path)


def _make_read_tool(sandbox: SandboxConfig) -> Tool:
    absolute = (
        f"{REFS_ANCHOR}/<alias>, /cache/..."
        if has_ref_mounts(sandbox)
        else "/cache/..."
    )
    absolute_clause = (
        f"Absolute container paths (e.g. {absolute}) are also accepted for reads. "
        if sandbox.use_container
        else ""
    )

    @tool(
        "read_file",
        "Read a file's contents. Paths are relative to the workspace root. "
        f"{absolute_clause}"
        f"Files over {OUTLINE_THRESHOLD} lines return a structural outline with line "
        "numbers instead of contents; read specific sections of those by "
        "passing start_line and end_line (1-based, inclusive).",
        required=("path",),
        path="string",
        start_line=("integer", "1-based line to start reading from (default 1)"),
        end_line=("integer", "1-based last line to read (default: end of file)"),
    )
    async def read_file(args: dict[str, Any]) -> ToolResult:
        path = args["path"]
        if Path(path).is_absolute():
            if not sandbox.use_container:
                return _bad(
                    "absolute paths only exist inside the sandbox container; "
                    "use a workspace-relative path"
                )
            target = path
        elif err := _check_path(sandbox, path):
            return err
        else:
            target = sandbox.tool_path(path)

        start = args.get("start_line")
        end = args.get("end_line")
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
        result = await run_in_sandbox(sandbox, command)
        if result.exit_code != 0:
            return _exec_error(result, "read failed")
        if ranged:
            first_nl = result.stdout.find("\n")
            try:
                total_lines = int(result.stdout[:first_nl].strip())
            except ValueError:
                return ToolResult(ok=False, error=f"could not count lines in {path}")
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
            output = "".join(contents.splitlines(keepends=True)[: end - start + 1])
            if end < total_lines:
                output += f"\n[showing lines {start}-{end} of {total_lines}]"
            return ToolResult(ok=True, output=output)

        contents = result.stdout
        truncated = len(contents.encode(errors="replace")) >= READ_CAP
        total_lines = contents.count("\n") + (0 if not contents or contents.endswith("\n") else 1)
        if not truncated and total_lines <= OUTLINE_THRESHOLD:
            return ToolResult(ok=True, output=contents)
        return _outline_result(path, total_lines, contents, truncated)

    return read_file


def _outline_result(path: str, total_lines: int, contents: str, truncated: bool) -> ToolResult:
    """Outline (Python) or preview (anything else) of a large file."""
    if truncated:
        return ToolResult(ok=True, output=(
            f"[this file is larger than {READ_CAP // (1024 * 1024)} MiB, so only "
            "its first part was read; pass start_line/end_line to read specific "
            "sections]\n"
            f"{_preview(contents)}"
        ))
    outline = _python_outline(contents) if path.lower().endswith(".py") else None
    if outline is None:
        return ToolResult(ok=True, output=(
            f"[showing lines 1-{min(PREVIEW_LINES, total_lines)} of {total_lines}; this "
            "file is too large to read all at once and has no outline — pass "
            "start_line/end_line to read specific sections]\n"
            f"{_preview(contents)}"
        ))
    return ToolResult(ok=True, output=(
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


def _make_write_tool(sandbox: SandboxConfig) -> Tool:
    @tool(
        "write_file",
        "Create a new file or overwrite an existing one with completely new "
        "contents (parent directories are created as needed). Prefer "
        "edit_file for changing part of an existing file.",
        required=("path", "content"),
        path="string",
        content=("string", "the full new file contents"),
    )
    async def write_file(args: dict[str, Any]) -> ToolResult:
        path = args["path"]
        content = args["content"]
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
            input_bytes=content.encode(),
        )
        if result.exit_code != 0:
            return _exec_error(result, "write failed")
        verb = "overwrote" if result.stdout.startswith("existing") else "created"
        return ToolResult(
            ok=True,
            output=f"{verb} {path} ({len(content.encode())} bytes)",
        )

    return write_file


# ----------------------------------------------------------------------
# edit_file

# Upper bound on the diff lines echoed back by edit_file.
_MAX_EDIT_DIFF_LINES = 400


class _EditError(Exception):
    """One edit in a batch did not match; nothing was written."""


def _apply_edits(current: str, edits: list[dict[str, Any]]) -> str:
    """Apply every edit in order, or raise: the batch is all-or-nothing.

    Each edit is matched against the text the previous ones produced (so a
    later edit can target an earlier edit's result), and must match exactly
    once unless ``replace_all`` is set.
    """
    updated = current
    for index, edit in enumerate(edits, 1):
        old, new = edit["old_string"], edit["new_string"]
        where = f"edit {index}" if len(edits) > 1 else "old_string"
        occurrences = updated.count(old)
        if occurrences == 0:
            raise _EditError(f"{where}: old_string not found in file")
        if occurrences > 1 and not edit.get("replace_all"):
            raise _EditError(
                f"{where}: old_string is not unique ({occurrences} matches); "
                "include more surrounding context, or set replace_all"
            )
        updated = updated.replace(old, new, occurrences if edit.get("replace_all") else 1)
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


def _make_edit_tool(sandbox: SandboxConfig) -> Tool:
    @tool(
        "edit_file",
        "Edit an existing file by replacing exact strings with new text. Pass "
        "'edits': a list of {old_string, new_string} replacements, applied in "
        "order — batch every change you want to make to the same file into one "
        "call. Each old_string must appear exactly once (add 'replace_all': "
        "true to replace every occurrence, e.g. for a rename), and must match "
        "the file exactly: include a few surrounding lines to disambiguate. "
        "The batch is all-or-nothing — if any edit fails, nothing is written "
        "and the error names the edit. The result includes a diff of what "
        "changed. Use write_file to create a file or replace its whole "
        "contents; use run_bash on non-UTF-8 files.",
        required=("path", "edits"),
        path="string",
        edits={
            "type": "array",
            "description": "replacements to apply, in order (all or nothing)",
            "items": {
                "type": "object",
                "properties": {
                    "old_string": {
                        "type": "string",
                        "description": "the exact text to replace",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "the replacement text",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "replace every occurrence (default: the "
                                       "old_string must be unique)",
                    },
                },
                "required": ["old_string", "new_string"],
            },
        },
    )
    async def edit_file(args: dict[str, Any]) -> ToolResult:
        path = args["path"]
        if err := _check_write_path(sandbox, path):
            return err

        edits = args["edits"]
        if not edits:
            return _bad("'edits' must be a non-empty list")
        for index, edit in enumerate(edits, 1):
            if not edit.get("old_string"):
                return _bad(f"edit {index} needs a non-empty 'old_string'")

        read_result = await run_in_sandbox(sandbox, ["cat", "--", sandbox.tool_path(path)])
        if read_result.exit_code != 0:
            return _exec_error(read_result, "read failed")
        current = read_result.stdout
        if "\ufffd" in current:
            # run_in_sandbox decodes with errors="replace": writing this back
            # would bake U+FFFD into the user's file.
            return ToolResult(
                ok=False,
                error=f"{path} is not valid UTF-8; edit it with run_bash instead",
            )

        try:
            updated = _apply_edits(current, edits)
        except _EditError as exc:
            return ToolResult(ok=False, error=str(exc))

        write_result = await run_in_sandbox(
            sandbox,
            ["sh", "-c", 'tee -- "$1" >/dev/null', "edit_file", sandbox.tool_path(path)],
            input_bytes=updated.encode(),
        )
        if write_result.exit_code != 0:
            return _exec_error(write_result, "write failed")
        plural = "edit" if len(edits) == 1 else f"{len(edits)} edits"
        return ToolResult(
            ok=True,
            output=f"applied {plural} to {path}\n{_format_edit_diff(path, current, updated)}",
        )

    return edit_file


# ----------------------------------------------------------------------
# run_bash


def _make_run_bash_tool(
    session: SandboxSession,
    git_guard: Callable[[str], str | None] | None = None,
) -> Tool:
    config = session.config
    description = ["Run a shell command inside the sandboxed workspace, as /bin/sh."]
    description.append(
        "The shell persists across calls within this chat — cd and exported "
        "variables carry over until it restarts (a timeout, or a mode switch). "
        "The command's stdin is /dev/null, so interactive commands read EOF "
        "instead of hanging."
    )
    description.append(
        "The command runs as one shell string: the reported exit code is the "
        "last statement's, so a chain whose final command finds nothing "
        "(grep, test -f) reads as a failure even when everything before it "
        "worked — and a pipe reports the last stage's status. Every result "
        "ends with an `[exit N]` status line."
    )

    @tool(
        "run_bash",
        " ".join(description),
        required=("command",),
        command="string",
        timeout_seconds=(
            "integer",
            (f"kill the command after this many seconds (default: the session "
             f"timeout, {config.timeout_seconds:g}); on timeout the shell is "
             "restarted and the output so far is returned"),
        ),
    )
    async def run_bash(args: dict[str, Any]) -> ToolResult:
        command = args["command"]
        if not command.strip():
            # Whitespace-only would make the persistent-shell group below an
            # empty `{ }`, a syntax error that kills the shell.
            return _bad("'command' is required")
        timeout = args.get("timeout_seconds")
        if timeout is not None and timeout < 1:
            return _bad("'timeout_seconds' must be >= 1")
        if git_guard is not None:
            block_reason = git_guard(command)
            if block_reason:
                return ToolResult(ok=False, error=block_reason)
        result = await session.run(command, on_output=_output_sink.get(), timeout=timeout)
        if result.timed_out:
            label = f" after {timeout}s" if timeout is not None else ""
            return ToolResult(
                ok=False,
                output=result.stdout,
                error=f"command timed out{label} (shell restarted); the rest "
                      "of the command did not run; output above is what it "
                      "printed before the timeout",
            )
        # Status line on every result: empty output plus a non-zero exit is
        # otherwise indistinguishable from success with no output.
        output = result.stdout
        if output and not output.endswith("\n"):
            output += "\n"
        return ToolResult(
            ok=result.exit_code == 0,
            output=f"{output}[exit {result.exit_code}]",
            error="" if result.exit_code == 0 else f"exit code {result.exit_code}",
        )

    return run_bash


# ----------------------------------------------------------------------
# ask_user


def _make_ask_user_tool(
    ask_callback: Callable[[list[str]], Awaitable[list[str] | None]],
) -> Tool:
    """ask_user: hand questions to the user and return their answers as the result."""

    @tool(
        "ask_user",
        "Ask the user one or more questions and wait for their answers. "
        "Use when you need a decision, a missing detail, or confirmation "
        "before acting. Keep questions short and self-contained.",
        required=("questions",),
        questions={
            "type": "array",
            "items": {"type": "string"},
            "description": "the questions to ask (one or more)",
        },
    )
    async def ask_user(args: dict[str, Any]) -> ToolResult:
        raw = args["questions"]
        questions = [q.strip() for q in raw if q.strip()]
        if not questions:
            return _bad("'questions' must be a non-empty list")
        answers = await ask_callback(questions)
        if answers is None:
            return ToolResult(ok=True, output="(user skipped the questions — no answers given)")
        pairs = [f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)]
        return ToolResult(ok=True, output="\n\n".join(pairs))

    return ask_user


def build_registry(
    sandbox: SandboxConfig | None,
    session: SandboxSession | None,
    mode: str = "write",
    ask_callback: Callable[[list[str]], Awaitable[list[str] | None]] | None = None,
    git_guard: Callable[[str], str | None] | None = None,
) -> ToolRegistry:
    """Build the tool set for one session.

    Plan mode is read-only: read_file plus run_bash against a read-only
    workspace mount. Write mode adds the editing tools: write_file,
    edit_file. ``ask_user`` is available in both modes when ``ask_callback``
    is given. ``git_guard`` optionally vetoes run_bash commands that would
    rewrite the user's branch/refs (see gitwork.py).

    Does NOT touch ``sandbox.read_only`` — the caller owns the sandbox's mode
    (set it to ``mode != "write"`` before calling; a persistent
    SandboxSession restarts its shell to pick up the new mounts).
    """
    registry = ToolRegistry()
    if ask_callback is not None:
        registry.register(_make_ask_user_tool(ask_callback))
    if sandbox is not None:
        registry.register(_make_read_tool(sandbox))  # read_file always available
        # Plan mode is fully read-only: the workspace is mounted read-only
        # (sandbox.read_only) and the persistent shell restarts under those
        # mounts, so run_bash cannot change files. Write mode drops the
        # read-only mount and adds the editing tools.
        #
        # EXCEPT when the sandbox is unsandboxed (use_container=False):
        # there is no read-only mount to enforce plan mode with — commands
        # would run unsandboxed on the host — so run_bash is NOT available
        # in plan mode at all (read_file still is). filter_list,
        # allow_network and gpu_access are likewise ignored in that mode
        # (see SandboxConfig.use_container).
        if session is not None and (sandbox.use_container or mode != "plan"):
            registry.register(_make_run_bash_tool(session, git_guard))  # run_bash
        if mode == "write":
            registry.register(_make_write_tool(sandbox))  # write_file
            registry.register(_make_edit_tool(sandbox))  # edit_file
    return registry
