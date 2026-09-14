"""Tool registry and the built-in tools.

test_tool has no filesystem or process access. read_file/edit_file/run_bash
execute inside a bwrap sandbox (see sandbox.py) scoped to one workspace
directory, with no network access — run_bash uses a persistent SandboxSession
so shell state survives across calls within one chat. web_search is the one
tool that runs outside the sandbox, since it's the only one that needs a
real network path.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .sandbox import SandboxConfig, SandboxSession, run_in_sandbox, validate_relpath


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


async def _test_tool_handler(args: dict[str, Any]) -> ToolResult:
    return ToolResult(ok=True, output="success!")


test_tool = Tool(
    name="test_tool",
    description="A no-op test tool that always succeeds, for wiring verification.",
    parameters_schema={"type": "object", "properties": {}},
    handler=_test_tool_handler,
)


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


def _make_read_tool(sandbox: SandboxConfig) -> Tool:
    async def _read_file(args: dict[str, Any]) -> ToolResult:
        path = args.get("path", "")
        error = validate_relpath(path)
        if error:
            return ToolResult(ok=False, error=error, parse_error=True)

        count_result = await run_in_sandbox(sandbox, ["wc", "-l", path])
        if count_result.exit_code != 0:
            return ToolResult(ok=False, error=count_result.stderr.strip() or "read failed")
        try:
            total_lines = int(count_result.stdout.split()[0])
        except (IndexError, ValueError):
            return ToolResult(ok=False, error=f"could not count lines in {path}")

        if "offset" in args or "limit" in args:
            start = int(args.get("offset") or 1)
            count = int(args.get("limit") or 2000)
        elif total_lines > 500:
            start, count = 1, 200
        else:
            start, count = 1, total_lines or 1

        end = start + count - 1
        result = await run_in_sandbox(sandbox, ["sed", "-n", f"{start},{end}p", path])
        if result.exit_code != 0:
            return ToolResult(ok=False, error=result.stderr.strip() or "read failed")

        output = result.stdout
        if end < total_lines:
            output += (
                f"\n[showing lines {start}-{end} of {total_lines}; "
                "pass offset/limit for more]"
            )
        return ToolResult(ok=True, output=output)

    return Tool(
        name="read_file",
        description=(
            "Read a file's contents. Paths are relative to the workspace root, "
            "or '.refs/<alias>' for externally referenced files. Large files "
            "(over 500 lines) return only the first 200 lines by default; pass "
            "offset/limit to read further chunks."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {
                    "type": "integer",
                    "description": "1-based line to start reading from (default 1)",
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "maximum lines to return (default 200 on large files, "
                        "2000 otherwise)"
                    ),
                },
            },
            "required": ["path"],
        },
        handler=_read_file,
    )


def _make_edit_tool(sandbox: SandboxConfig) -> Tool:
    async def _edit_file(args: dict[str, Any]) -> ToolResult:
        path = args.get("path", "")
        mode = args.get("mode", "")
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        error = validate_relpath(path)
        if error:
            return ToolResult(ok=False, error=error, parse_error=True)
        if path.startswith((".refs/", "./.refs/")):
            return ToolResult(ok=False, error="'.refs/' is read-only")

        if mode == "create":
            exists = await run_in_sandbox(sandbox, ["test", "-e", path])
            if exists.exit_code == 0:
                return ToolResult(
                    ok=False,
                    error=f"{path} already exists; use mode='replace' to modify it",
                )
            write_result = await run_in_sandbox(
                sandbox, ["tee", path], input_bytes=new_string.encode()
            )
            if write_result.exit_code != 0:
                return ToolResult(ok=False, error=write_result.stderr.strip() or "write failed")
            return ToolResult(ok=True, output=f"created {path} ({len(new_string)} bytes)")

        if mode == "replace":
            if not old_string:
                return ToolResult(
                    ok=False,
                    error="'old_string' is required when mode='replace'",
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

        return ToolResult(
            ok=False,
            error="'mode' must be 'replace' or 'create'",
            parse_error=True,
        )

    return Tool(
        name="edit_file",
        description=(
            "Create a new file or edit an existing one. With mode='create', "
            "write new_string to a path that must not already exist. With "
            "mode='replace', replace one exact, unique occurrence of "
            "old_string with new_string — include enough surrounding context "
            "(a few lines) to disambiguate if the snippet could appear more "
            "than once. Cannot write under '.refs/', which is read-only."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "mode": {"type": "string", "enum": ["replace", "create"]},
                "old_string": {
                    "type": "string",
                    "description": "required when mode='replace'",
                },
                "new_string": {"type": "string"},
            },
            "required": ["path", "mode", "new_string"],
        },
        handler=_edit_file,
    )


def _make_run_bash_tool(session: SandboxSession) -> Tool:
    async def _run_bash(args: dict[str, Any]) -> ToolResult:
        command = args.get("command", "")
        if not command:
            return ToolResult(ok=False, error="'command' is required", parse_error=True)
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
            "variables carry over."
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
) -> ToolRegistry:
    """Build the tool set for one session.

    plan mode exposes read_file (plus the always-on test_tool/web_search);
    write mode additionally exposes edit_file and run_bash. ``ask_callback``
    enables the ask tool (prompts the user in the TUI); ``compact_callback``
    enables the compact tool (summarizes + truncates the conversation).
    ``allow_subagent`` is reserved for subagent registration in a later phase.
    """
    registry = ToolRegistry(max_calls_per_turn=max_calls_per_turn)
    registry.register(test_tool)
    registry.register(web_search_tool)
    if ask_callback is not None:
        registry.register(_make_ask_tool(ask_callback))
    if compact_callback is not None:
        registry.register(_make_compact_tool(compact_callback))
    if sandbox is not None:
        registry.register(_make_read_tool(sandbox))  # read_file always available
        if mode == "write":
            registry.register(_make_edit_tool(sandbox))  # edit_file
            if session is not None:
                registry.register(_make_run_bash_tool(session))  # run_bash
    return registry
