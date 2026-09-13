"""Tool registry and the built-in tools.

test_tool has no filesystem or process access. read_file/write_file/run_bash
execute inside a bwrap sandbox (see sandbox.py) scoped to one workspace
directory, with no network access. web_search is the one tool that runs
outside the sandbox, in this process, since it's the only one that needs a
real network path — keeping that boundary narrow is the point.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .sandbox import SandboxConfig, run_in_sandbox, validate_relpath


@dataclass(slots=True)
class ToolResult:
    """Outcome of one tool execution.

    ``parse_error`` distinguishes "the model's argument JSON was invalid"
    (rendered red as a parsing error) from "the tool ran and failed".
    """

    ok: bool
    output: str = ""
    error: str = ""
    parse_error: bool = False


@dataclass(slots=True)
class Tool:
    """One callable tool, described in JSON schema for the API's tools param."""

    name: str
    description: str
    parameters_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[ToolResult]]


@dataclass(slots=True)
class ToolRegistry:
    """Name-keyed tool set; produces the OpenAI-compatible ``tools`` payload."""

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
        """Parse arguments, run the handler, and contain every failure.

        Never raises: unknown tools, bad JSON, and handler exceptions all come
        back as a failed :class:`ToolResult` so the caller can just render it.
        """
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
        except Exception as exc:  # noqa: BLE001 — tool handlers are contained here
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


def _make_fs_tools(sandbox: SandboxConfig) -> list[Tool]:
    async def _read_file(args: dict[str, Any]) -> ToolResult:
        path = args.get("path", "")
        error = validate_relpath(path)
        if error:
            return ToolResult(ok=False, error=error, parse_error=True)
        result = await run_in_sandbox(sandbox, ["cat", path])
        if result.exit_code != 0:
            return ToolResult(ok=False, error=result.stderr.strip() or "read failed")
        return ToolResult(ok=True, output=result.stdout)

    async def _write_file(args: dict[str, Any]) -> ToolResult:
        path = args.get("path", "")
        content = args.get("content", "")
        error = validate_relpath(path)
        if error:
            return ToolResult(ok=False, error=error, parse_error=True)
        if path.startswith(".refs/") or path.startswith("./.refs/"):
            return ToolResult(ok=False, error="'.refs/' is read-only")
        result = await run_in_sandbox(sandbox, ["tee", path], input_bytes=content.encode())
        if result.exit_code != 0:
            return ToolResult(ok=False, error=result.stderr.strip() or "write failed")
        return ToolResult(ok=True, output=f"wrote {len(content)} bytes to {path}")

    async def _run_bash(args: dict[str, Any]) -> ToolResult:
        command = args.get("command", "")
        if not command:
            return ToolResult(ok=False, error="'command' is required", parse_error=True)
        result = await run_in_sandbox(sandbox, ["/bin/sh", "-c", command])
        output = result.stdout + (("\n" + result.stderr) if result.stderr else "")
        if result.timed_out:
            return ToolResult(ok=False, error="command timed out")
        return ToolResult(
            ok=result.exit_code == 0,
            output=output,
            error="" if result.exit_code == 0 else (result.stderr.strip() or f"exit code {result.exit_code}"),
        )

    return [
        Tool(
            name="read_file",
            description=(
                "Read a file's contents. Paths are relative to the workspace root, "
                "or '.refs/<alias>' for externally referenced files."
            ),
            parameters_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            handler=_read_file,
        ),
        Tool(
            name="write_file",
            description="Write content to a file in the workspace. Cannot write under '.refs/', which is read-only.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            handler=_write_file,
        ),
        Tool(
            name="run_bash",
            description="Run a shell command inside the sandboxed workspace. No network access.",
            parameters_schema={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
            handler=_run_bash,
        ),
    ]


def default_registry(sandbox: SandboxConfig | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(test_tool)
    registry.register(web_search_tool)
    if sandbox is not None:
        for tool in _make_fs_tools(sandbox):
            registry.register(tool)
    return registry
