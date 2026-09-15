"""Tool registry and the built-in tools.

read_file/write_file/edit_file/run_bash execute inside a bwrap sandbox (see
sandbox.py) scoped to one workspace directory, with no network access —
run_bash uses a persistent SandboxSession so shell state survives across
calls within one chat. web_search is the one tool that runs outside the
sandbox, since it's the only one that needs a real network path.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .sandbox import SandboxConfig, SandboxSession, run_in_sandbox


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


@dataclass(slots=True)
class ToolRegistry:
    """Named tool set exposed to the model.

    ``max_calls_per_turn`` is reserved for per-turn call limiting in a later
    phase; it is stored but not yet enforced.
    """

    max_calls_per_turn: int | None = None
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

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

    async def call(self, name: str, arguments_json: str) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, error=f"unknown tool '{name}'")
        try:
            args = json.loads(arguments_json) if arguments_json.strip() else {}
        except json.JSONDecodeError as exc:
            return ToolResult(ok=False, error=f"invalid arguments JSON: {exc}", parse_error=True)
        if not isinstance(args, dict):
            return ToolResult(
                ok=False,
                error=f"invalid arguments JSON: expected an object, got {type(args).__name__}",
                parse_error=True,
            )
        try:
            return await tool.handler(args)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")


async def _web_search_handler(args: dict[str, Any]) -> ToolResult:
    """Runs OUTSIDE the sandbox — the only tool with real network access."""
    import httpx

    query = args.get("query", "")
    if not query:
        return ToolResult(ok=False, error="'query' is required", parse_error=True)

    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        return ToolResult(ok=False, error="BRAVE_API_KEY environment variable is not set")

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query},
            headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

    results = data.get("web", {}).get("results", [])[:5]
    if not results:
        return ToolResult(ok=True, output="no results")
    lines = [
        f"{r.get('title', '')}\n{r.get('url', '')}\n{r.get('description', '')}"
        for r in results
    ]
    return ToolResult(ok=True, output="\n\n".join(lines))


web_search_tool = Tool(
    name="web_search",
    description="Search the web and return results as text.",
    parameters_schema={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "search query"}},
        "required": ["query"],
    },
    handler=_web_search_handler,
)


# Files larger than this return a structural outline instead of contents
# (unless the call passes explicit line numbers).
OUTLINE_THRESHOLD = 500
# Upper bound on lines returned by one ranged read.
MAX_READ_LINES = 2000


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


def _make_read_tool(sandbox: SandboxConfig) -> Tool:
    async def _read_file(args: dict[str, Any]) -> ToolResult:
        path = args.get("path", "")
        error = sandbox.validate_relpath(path)
        if error:
            return ToolResult(ok=False, error=error, parse_error=True)

        try:
            start = int(args["start_line"]) if args.get("start_line") is not None else 1
            end = int(args["end_line"]) if args.get("end_line") is not None else None
        except (TypeError, ValueError):
            return ToolResult(
                ok=False, error="start_line/end_line must be integers", parse_error=True
            )
        if start < 1:
            return ToolResult(ok=False, error="start_line is 1-based", parse_error=True)

        count_result = await run_in_sandbox(sandbox, ["wc", "-l", path])
        if count_result.exit_code != 0:
            return ToolResult(ok=False, error=count_result.stderr.strip() or "read failed")
        try:
            total_lines = int(count_result.stdout.split()[0])
        except (IndexError, ValueError):
            return ToolResult(ok=False, error=f"could not count lines in {path}")

        if args.get("start_line") is None and args.get("end_line") is None:
            if total_lines <= OUTLINE_THRESHOLD:
                start, end = 1, max(total_lines, 1)
            else:
                return await _outline_result(sandbox, path, total_lines)
        else:
            end = total_lines if end is None else min(end, total_lines)
            if end < start:
                return ToolResult(
                    ok=False,
                    error=f"end_line ({end}) is before start_line ({start})",
                    parse_error=True,
                )
            if end - start + 1 > MAX_READ_LINES:
                end = start + MAX_READ_LINES - 1

        result = await run_in_sandbox(sandbox, ["sed", "-n", f"{start},{end}p", path])
        if result.exit_code != 0:
            return ToolResult(ok=False, error=result.stderr.strip() or "read failed")

        output = result.stdout
        if end < total_lines:
            output += (
                f"\n[showing lines {start}-{end} of {total_lines}; pass "
                "start_line/end_line for more]"
            )
        return ToolResult(ok=True, output=output)

    async def _outline_result(sandbox: SandboxConfig, path: str, total_lines: int) -> ToolResult:
        cat_result = await run_in_sandbox(sandbox, ["cat", path])
        if cat_result.exit_code != 0:
            return ToolResult(ok=False, error=cat_result.stderr.strip() or "read failed")
        outline = _python_outline(cat_result.stdout)
        if outline is None:
            # Not a Python file (or no defs/classes): fall back to a preview
            # chunk, pointing at start_line/end_line for the rest.
            end = min(200, total_lines)
            preview = "\n".join(cat_result.stdout.splitlines()[:end]) + "\n"
            return ToolResult(ok=True, output=(
                f"[showing lines 1-{end} of {total_lines}; this file is too large "
                "to read all at once and has no outline — pass start_line/end_line "
                "to read specific sections]\n"
                f"{preview}"
            ))
        abs_path = sandbox.workspace / path
        return ToolResult(ok=True, output=(
            "File outline retrieved. This file is too large to read all at once, so "
            "the outline below shows the file's structure with line numbers.\n"
            "\n"
            "IMPORTANT: Do NOT retry this call without line numbers - you will get "
            "the same outline.\n"
            "Instead, use the line numbers below to read specific sections by calling "
            "this tool again with start_line and end_line parameters.\n"
            "\n"
            f"# File outline for {abs_path}\n"
            "\n"
            f"{outline}\n"
            "\n"
            "NEXT STEPS: To read a specific symbol's implementation, call read_file "
            "with the same path plus start_line and end_line from the outline above.\n"
            "For example, to read a function shown as [L100-150], use start_line: 100 "
            "and end_line: 150."
        ))

    return Tool(
        name="read_file",
        description=(
            "Read a file's contents. Paths are relative to the workspace root, "
            "or '.refs/<alias>' for externally referenced files. Files over "
            f"{OUTLINE_THRESHOLD} lines return a structural outline with line "
            "numbers instead of contents; read specific sections of those by "
            "passing start_line and end_line (1-based, inclusive)."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {
                    "type": "integer",
                    "description": "1-based line to start reading from (default 1)",
                },
                "end_line": {
                    "type": "integer",
                    "description": "1-based last line to read (default: end of file)",
                },
            },
            "required": ["path"],
        },
        handler=_read_file,
    )


def _make_write_tool(sandbox: SandboxConfig) -> Tool:
    async def _write_file(args: dict[str, Any]) -> ToolResult:
        path = args.get("path", "")
        content = args.get("content", "")
        error = sandbox.validate_relpath(path)
        if error:
            return ToolResult(ok=False, error=error, parse_error=True)
        if path.startswith((".refs/", "./.refs/")):
            return ToolResult(ok=False, error="'.refs/' is read-only")

        exists = await run_in_sandbox(sandbox, ["test", "-e", path])
        created = exists.exit_code != 0
        write_result = await run_in_sandbox(
            sandbox, ["tee", path], input_bytes=content.encode()
        )
        if write_result.exit_code != 0:
            return ToolResult(ok=False, error=write_result.stderr.strip() or "write failed")
        verb = "created" if created else "overwrote"
        return ToolResult(ok=True, output=f"{verb} {path} ({len(content)} bytes)")

    return Tool(
        name="write_file",
        description=(
            "Create a new file or overwrite an existing one with completely new "
            "contents. Prefer edit_file for changing part of an existing file. "
            "Cannot write under '.refs/', which is read-only."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string", "description": "the full new file contents"},
            },
            "required": ["path", "content"],
        },
        handler=_write_file,
    )


def _make_edit_tool(sandbox: SandboxConfig) -> Tool:
    async def _edit_file(args: dict[str, Any]) -> ToolResult:
        path = args.get("path", "")
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        error = sandbox.validate_relpath(path)
        if error:
            return ToolResult(ok=False, error=error, parse_error=True)
        if path.startswith((".refs/", "./.refs/")):
            return ToolResult(ok=False, error="'.refs/' is read-only")
        if not old_string:
            return ToolResult(
                ok=False,
                error="'old_string' is required (use write_file to replace a whole file)",
                parse_error=True,
            )

        read_result = await run_in_sandbox(sandbox, ["cat", path])
        if read_result.exit_code != 0:
            return ToolResult(ok=False, error=read_result.stderr.strip() or "read failed")

        current = read_result.stdout
        occurrences = current.count(old_string)
        if occurrences == 0:
            return ToolResult(ok=False, error="old_string not found in file")
        if occurrences > 1:
            return ToolResult(
                ok=False,
                error=f"old_string is not unique ({occurrences} matches); "
                "include more surrounding context to disambiguate",
            )

        updated = current.replace(old_string, new_string, 1)
        write_result = await run_in_sandbox(
            sandbox, ["tee", path], input_bytes=updated.encode()
        )
        if write_result.exit_code != 0:
            return ToolResult(ok=False, error=write_result.stderr.strip() or "write failed")
        return ToolResult(ok=True, output=f"applied edit to {path}")

    return Tool(
        name="edit_file",
        description=(
            "Edit an existing file by replacing one exact, unique occurrence of "
            "old_string with new_string — include enough surrounding context (a "
            "few lines) to disambiguate if the snippet could appear more than "
            "once. Use write_file to create a file or replace its whole contents. "
            "Cannot write under '.refs/', which is read-only."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {
                    "type": "string",
                    "description": "the exact text to replace (must match once)",
                },
                "new_string": {"type": "string", "description": "the replacement text"},
            },
            "required": ["path", "old_string", "new_string"],
        },
        handler=_edit_file,
    )


def _make_run_bash_tool(
    session: SandboxSession,
    git_guard: Callable[[str], str | None] | None = None,
) -> Tool:
    async def _run_bash(args: dict[str, Any]) -> ToolResult:
        command = args.get("command", "")
        if not command:
            return ToolResult(ok=False, error="'command' is required", parse_error=True)
        if git_guard is not None:
            block_reason = git_guard(command)
            if block_reason:
                return ToolResult(ok=False, error=block_reason)
        result = await session.run(command)
        if result.timed_out:
            return ToolResult(ok=False, error="command timed out (shell restarted)")
        return ToolResult(
            ok=result.exit_code == 0,
            output=result.stdout,
            error="" if result.exit_code == 0 else f"exit code {result.exit_code}",
        )

    return Tool(
        name="run_bash",
        description=(
            "Run a shell command inside the sandboxed workspace. No network access. "
            "The shell persists across calls within this chat — cwd and exported "
            "variables carry over. Branch/worktree git operations (checkout <ref>, "
            "switch, worktree, branch -d/-D, reset --hard, rebase) are managed by "
            "the harness and rejected; status/diff/log/show/blame/add/commit work."
        ),
        parameters_schema={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        handler=_run_bash,
    )


def _make_ask_tool(ask_callback: Callable[[list[str]], Awaitable[list[str] | None]]) -> Tool:
    """ask: hand questions to the user and return their answers as the result."""

    async def _ask_user(args: dict[str, Any]) -> ToolResult:
        raw = args.get("questions")
        if isinstance(raw, str):
            raw = [raw]
        questions = [q.strip() for q in raw or [] if isinstance(q, str) and q.strip()]
        if not questions:
            return ToolResult(ok=False, error="'questions' must be a non-empty list", parse_error=True)
        answers = await ask_callback(questions)
        if answers is None:
            return ToolResult(ok=True, output="(user skipped the questions — no answers given)")
        pairs = [f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)]
        return ToolResult(ok=True, output="\n\n".join(pairs))

    return Tool(
        name="ask",
        description=(
            "Ask the user one or more questions and wait for their answers. "
            "Use when you need a decision, a missing detail, or confirmation "
            "before acting. Keep questions short and self-contained."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "the questions to ask (one or more)",
                }
            },
            "required": ["questions"],
        },
        handler=_ask_user,
    )


def _make_compact_tool(compact_callback: Callable[[], Awaitable[str]]) -> Tool:
    """compact: summarize the conversation so far to free context window."""

    async def _compact(args: dict[str, Any]) -> ToolResult:
        summary = await compact_callback()
        return ToolResult(ok=True, output=f"conversation compacted. {summary}")

    return Tool(
        name="compact",
        description=(
            "Compact the conversation: everything before this turn is replaced "
            "with a short summary, freeing context window. Use when the chat is "
            "long and earlier details no longer need to be verbatim. The current "
            "turn is not affected."
        ),
        parameters_schema={"type": "object", "properties": {}},
        handler=_compact,
    )


def build_registry(
    sandbox: SandboxConfig | None,
    session: SandboxSession | None,
    mode: str = "write",
    allow_subagent: bool = True,
    max_calls_per_turn: int | None = None,
    ask_callback: Callable[[list[str]], Awaitable[list[str] | None]] | None = None,
    compact_callback: Callable[[], Awaitable[str]] | None = None,
    git_guard: Callable[[str], str | None] | None = None,
) -> ToolRegistry:
    """Build the tool set for one session.

    plan mode exposes read_file (plus the always-on web_search); write mode
    additionally exposes write_file, edit_file, and run_bash. ``ask_callback``
    enables the ask tool (prompts the user in the TUI); ``compact_callback``
    enables the compact tool (summarizes + truncates the conversation).
    ``git_guard`` optionally vetoes run_bash commands that would interfere
    with harness-managed worktree/branch lifecycle (see gitwork.py).
    ``allow_subagent`` is reserved for subagent registration in a later phase.
    """
    registry = ToolRegistry(max_calls_per_turn=max_calls_per_turn)
    registry.register(web_search_tool)
    if ask_callback is not None:
        registry.register(_make_ask_tool(ask_callback))
    if compact_callback is not None:
        registry.register(_make_compact_tool(compact_callback))
    if sandbox is not None:
        registry.register(_make_read_tool(sandbox))  # read_file always available
        if mode == "write":
            registry.register(_make_write_tool(sandbox))  # write_file
            registry.register(_make_edit_tool(sandbox))  # edit_file
            if session is not None:
                registry.register(_make_run_bash_tool(session, git_guard))  # run_bash
    return registry
