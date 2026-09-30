"""Tool registry and the built-in tools.

read_file/write_file/edit_file/ls/glob/grep/run_bash execute inside a bwrap
sandbox (see sandbox.py) scoped to one workspace directory, with no network
access — run_bash uses a persistent SandboxSession so shell state survives
across calls within one chat.

Plan mode exposes read_file plus the read-only exploration tools (ls, glob,
grep); write mode exposes read_file plus the editing tools (write_file,
edit_file, run_bash). ask_user is available in both modes. web_search is
disabled for now (see build_registry).
"""

from __future__ import annotations

import difflib
import json
import os
import shlex
import shutil
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from .ignore import glob_to_regex
from .sandbox import SandboxConfig, SandboxSession, run_in_sandbox


@dataclass(slots=True)
class ToolResult:
    ok: bool
    output: str = ""
    error: str = ""
    parse_error: bool = False
    # One-line summary for the TUI's settled tool-call header (the dimmed
    # detail after "Ran <tool>"). Computed by the tool itself so it can be
    # persisted with the conversation and replayed on /resume.
    header: str = ""


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


# Per-tool cap on the text sent back to the model (adjustable with
# /output_limit in the TUI). Applies to every tool via ToolRegistry.call.
DEFAULT_OUTPUT_LIMIT = 32768


def _truncate_output(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    note = f"\n[output truncated at {limit} chars — raise the cap with /output_limit <chars>]"
    return text[: max(limit - len(note), 0)] + note


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


def _check_path(sandbox: SandboxConfig, path: str, mode: str | None = None) -> ToolResult | None:
    error = sandbox.validate_relpath(path, mode=mode)
    return None if error is None else _bad(error)


def _refs_guard(path: str) -> ToolResult | None:
    if path.startswith((".refs/", "./.refs/")):
        return ToolResult(ok=False, error="'.refs/' is read-only")
    return None


@dataclass(slots=True)
class ToolRegistry:
    """Named tool set exposed to the model.

    ``max_output_chars`` caps the text of every tool result before it goes
    on the wire (see _truncate_output); the TUI's /output_limit command
    mutates it live. ``max_calls_per_turn`` is reserved for per-turn call
    limiting in a later phase; it is stored but not yet enforced.
    """

    max_calls_per_turn: int | None = None
    max_output_chars: int = DEFAULT_OUTPUT_LIMIT
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


# web_search is disabled for now: the implementation isn't good enough to
# ship, so it is not registered in build_registry. Kept here for reference.
# web_search_tool = Tool(
#     name="web_search",
#     description="Search the web and return results as text.",
#     parameters_schema={
#         "type": "object",
#         "properties": {"query": {"type": "string", "description": "search query"}},
#         "required": ["query"],
#     },
#     handler=_web_search_handler,
# )


# Files larger than this return a structural outline instead of contents
# (unless the call passes explicit line numbers).
OUTLINE_THRESHOLD = 500
# Upper bound on lines returned by one ranged read.
MAX_READ_LINES = 2000

# Result caps for the read-only exploration tools (ls/glob/grep).
_MAX_GLOB_RESULTS = 200
_MAX_GREP_MATCHES = 200
_MAX_GREP_PER_FILE = 20


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
    @tool(
        "read_file",
        "Read a file's contents. Paths are relative to the workspace root, "
        "or '.refs/<alias>' for externally referenced files. Files over "
        f"{OUTLINE_THRESHOLD} lines return a structural outline with line "
        "numbers instead of contents; read specific sections of those by "
        "passing start_line and end_line (1-based, inclusive).",
        required=("path",),
        path="string",
        start_line=("integer", "1-based line to start reading from (default 1)"),
        end_line=("integer", "1-based last line to read (default: end of file)"),
    )
    async def read_file(args: dict[str, Any]) -> ToolResult:
        path = args.get("path", "")
        if err := _check_path(sandbox, path, mode="read"):
            return err

        try:
            start = int(args["start_line"]) if args.get("start_line") is not None else 1
            end = int(args["end_line"]) if args.get("end_line") is not None else None
        except (TypeError, ValueError):
            return ToolResult(
                ok=False, error="start_line/end_line must be integers", parse_error=True
            )
        if start < 1:
            return ToolResult(ok=False, error="start_line is 1-based", parse_error=True)

        target = sandbox.tool_path(path)
        # awk's NR counts lines regardless of a trailing newline (wc -l counts
        # newlines, undercounting files that don't end with one).
        count_result = await run_in_sandbox(sandbox, ["awk", "END{print NR}", target])
        if count_result.exit_code != 0:
            return _exec_error(count_result, "read failed")
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

        result = await run_in_sandbox(sandbox, ["sed", "-n", f"{start},{end}p", target])
        if result.exit_code != 0:
            return _exec_error(result, "read failed")

        output = result.stdout
        if end < total_lines:
            output += (
                f"\n[showing lines {start}-{end} of {total_lines}; pass "
                "start_line/end_line for more]"
            )
        return ToolResult(ok=True, output=output, header=path)

    async def _outline_result(sandbox: SandboxConfig, path: str, total_lines: int) -> ToolResult:
        cat_result = await run_in_sandbox(sandbox, ["cat", sandbox.tool_path(path)])
        if cat_result.exit_code != 0:
            return _exec_error(cat_result, "read failed")
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
            "and end_line: 150."),
            header=path,
        )

    return read_file


def _make_write_tool(sandbox: SandboxConfig) -> Tool:
    @tool(
        "write_file",
        "Create a new file or overwrite an existing one with completely new "
        "contents. Prefer edit_file for changing part of an existing file. "
        "Cannot write under '.refs/', which is read-only.",
        required=("path", "content"),
        path="string",
        content=("string", "the full new file contents"),
    )
    async def write_file(args: dict[str, Any]) -> ToolResult:
        path = args.get("path", "")
        content = args.get("content", "")
        if err := _check_path(sandbox, path) or _refs_guard(path):
            return err

        target = sandbox.tool_path(path)
        exists = await run_in_sandbox(sandbox, ["test", "-e", target])
        created = exists.exit_code != 0
        write_result = await run_in_sandbox(
            sandbox, ["tee", target], input_bytes=content.encode()
        )
        if write_result.exit_code != 0:
            return _exec_error(write_result, "write failed")
        verb = "created" if created else "overwrote"
        return ToolResult(ok=True, output=f"{verb} {path} ({len(content)} bytes)", header=path)

    return write_file


# Upper bound on the diff lines echoed back by edit_file.
_MAX_EDIT_DIFF_LINES = 400


def _collect_edits(
    args: dict[str, Any],
) -> tuple[list[tuple[str, str]], ToolResult | None]:
    """Normalize the two accepted argument shapes into ``(old, new)`` pairs.

    Preferred is an ``edits`` array (every change to the file in one call);
    a bare top-level ``old_string``/``new_string`` pair is still accepted so
    older transcripts and habits keep working.
    """
    raw = args.get("edits")
    if raw is None:
        old = args.get("old_string", "")
        if not old:
            return [], ToolResult(
                ok=False,
                error="provide 'edits' (a list of {old_string, new_string}) or a "
                "single 'old_string'/'new_string' pair",
                parse_error=True,
            )
        return [(old, args.get("new_string", ""))], None
    if not isinstance(raw, list) or not raw:
        return [], ToolResult(
            ok=False, error="'edits' must be a non-empty list", parse_error=True
        )
    edits: list[tuple[str, str]] = []
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            return [], ToolResult(
                ok=False, error=f"edit {index} must be an object", parse_error=True
            )
        old, new = item.get("old_string", ""), item.get("new_string", "")
        if not isinstance(old, str) or not isinstance(new, str) or not old:
            return [], ToolResult(
                ok=False,
                error=f"edit {index} needs a non-empty 'old_string' and a 'new_string'",
                parse_error=True,
            )
        edits.append((old, new))
    return edits, None


def _apply_edits(current: str, edits: list[tuple[str, str]]) -> tuple[str, str | None]:
    """Apply every edit in order; return ``(new_text, error)``.

    All-or-nothing: the first edit that does not match exactly once aborts
    the whole call, so a partially edited file is never written.
    """
    updated = current
    for index, (old, new) in enumerate(edits, 1):
        where = f"edit {index}" if len(edits) > 1 else "old_string"
        occurrences = updated.count(old)
        if occurrences == 0:
            return updated, f"{where}: old_string not found in file"
        if occurrences > 1:
            return updated, (
                f"{where}: old_string is not unique ({occurrences} matches); "
                "include more surrounding context to disambiguate"
            )
        updated = updated.replace(old, new, 1)
    return updated, None


def _format_edit_diff(path: str, before: str, after: str) -> str:
    """Git-style unified diff of one edit call's changes.

    Returned as the tool's output so both the model (to verify what it
    actually changed) and the TUI (expanding the tool block) can see the
    changed lines rather than just an "applied" acknowledgement.
    """
    lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
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
            f"[... diff truncated at {_MAX_EDIT_DIFF_LINES} lines]"
        ]
    return f"diff --git a/{path} b/{path}\n" + "\n".join(lines)


def _make_edit_tool(sandbox: SandboxConfig) -> Tool:
    @tool(
        "edit_file",
        "Edit an existing file by replacing exact, unique strings with new "
        "text. Pass 'edits': a list of {old_string, new_string} replacements, "
        "applied in order — batch every change you want to make to the same "
        "file into one call rather than calling this tool repeatedly. Each "
        "old_string must match the file exactly and uniquely; include enough "
        "surrounding context (a few lines) to disambiguate if the snippet "
        "could appear more than once. Edits are all-or-nothing: if any one "
        "fails to match, nothing is written. The result includes a diff of "
        "what changed. Use write_file to create a file or replace its whole "
        "contents. Cannot write under '.refs/', which is read-only.",
        required=("path",),
        path="string",
        edits={
            "type": "array",
            "description": (
                "replacements to apply, in order (prefer this over the "
                "single-edit old_string/new_string pair)"
            ),
            "items": {
                "type": "object",
                "properties": {
                    "old_string": {
                        "type": "string",
                        "description": "the exact text to replace (must match once)",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "the replacement text",
                    },
                },
                "required": ["old_string", "new_string"],
            },
        },
        old_string=(
            "string",
            "single-edit form: the exact text to replace (must match once)",
        ),
        new_string=("string", "single-edit form: the replacement text"),
    )
    async def edit_file(args: dict[str, Any]) -> ToolResult:
        path = args.get("path", "")
        if err := _check_path(sandbox, path) or _refs_guard(path):
            return err

        edits, parse_error = _collect_edits(args)
        if parse_error is not None:
            return parse_error

        read_result = await run_in_sandbox(sandbox, ["cat", sandbox.tool_path(path)])
        if read_result.exit_code != 0:
            return _exec_error(read_result, "read failed")

        current = read_result.stdout
        updated, failure = _apply_edits(current, edits)
        if failure is not None:
            return ToolResult(ok=False, error=failure)

        write_result = await run_in_sandbox(
            sandbox, ["tee", sandbox.tool_path(path)], input_bytes=updated.encode()
        )
        if write_result.exit_code != 0:
            return _exec_error(write_result, "write failed")
        plural = "edit" if len(edits) == 1 else f"{len(edits)} edits"
        diff = _format_edit_diff(path, current, updated)
        adds = sum(
            1 for line in diff.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        dels = sum(
            1 for line in diff.splitlines()
            if line.startswith("-") and not line.startswith("---")
        )
        return ToolResult(
            ok=True,
            output=f"applied {plural} to {path}\n{diff}",
            header=f"+{adds} -{dels} {path}",
        )

    return edit_file


def _make_run_bash_tool(
    session: SandboxSession,
    git_guard: Callable[[str], str | None] | None = None,
) -> Tool:
    @tool(
        "run_bash",
        "Run a shell command inside the sandboxed workspace. No network access. "
        "The shell persists across calls within this chat — cwd and exported "
        "variables carry over. Branch/ref git operations (checkout <ref>, "
        "switch, worktree, branch -d/-D, reset --hard, rebase) are rejected; "
        "status/diff/log/show/blame/add/commit work.",
        required=("command",),
        command="string",
    )
    async def run_bash(args: dict[str, Any]) -> ToolResult:
        command = args.get("command", "")
        if not command:
            return _bad("'command' is required")
        if git_guard is not None:
            block_reason = git_guard(command)
            if block_reason:
                return ToolResult(ok=False, error=block_reason, header=command)
        result = await session.run(command, on_output=_output_sink.get())
        if result.timed_out:
            return ToolResult(ok=False, error="command timed out (shell restarted)", header=command)
        return ToolResult(
            ok=result.exit_code == 0,
            output=result.stdout,
            error="" if result.exit_code == 0 else f"exit code {result.exit_code}",
            header=command,
        )

    return run_bash


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
        raw = args.get("questions")
        if isinstance(raw, str):
            raw = [raw]
        questions = [q.strip() for q in raw or [] if isinstance(q, str) and q.strip()]
        if not questions:
            return _bad("'questions' must be a non-empty list")
        answers = await ask_callback(questions)
        if answers is None:
            return ToolResult(ok=True, output="(user skipped the questions — no answers given)")
        pairs = [f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)]
        return ToolResult(ok=True, output="\n\n".join(pairs))

    return ask_user


# ---------------------------------------------------------------------------
# read-only exploration: ls / glob / grep (plan mode only)
# Ignore rules are NOT applied here: ignored paths (gitignored + default
# noise) are shadowed out of the sandbox at mount time, so every tool —
# bash included — simply cannot see them.

def _make_ls_tool(sandbox: SandboxConfig) -> Tool:
    @tool(
        "ls",
        "List a directory's immediate entries (not recursive); directories "
        "end with '/'. Ignored files (gitignored, caches, ...) are shadowed "
        "out of the sandbox and never appear. Paths are relative "
        "to the workspace root.",
        path=("string", "directory to list (default '.')"),
    )
    async def ls(args: dict[str, Any]) -> ToolResult:
        path = args.get("path") or "."
        if not isinstance(path, str):
            return _bad("'path' must be a string")
        if err := _check_path(sandbox, path, mode="read"):
            return err
        target = sandbox.tool_path(path)
        is_dir = await run_in_sandbox(sandbox, ["test", "-d", target])
        if is_dir.exit_code != 0:
            exists = await run_in_sandbox(sandbox, ["test", "-e", target])
            if exists.exit_code == 0:
                return _bad(f"'{path}' is not a directory")
            return _bad(f"path does not exist: {path}")

        # -p suffixes directories with '/', which doubles as the dir marker.
        listing = await run_in_sandbox(sandbox, ["ls", "-A", "-p", "--", target])
        if listing.exit_code != 0:
            return _exec_error(listing, "ls failed")
        entries = [e for e in listing.stdout.splitlines() if e]
        # A shadowed directory is empty but its name still lists; drop the
        # names of anything hidden under the listed path (shadowed files
        # read as empty but also shouldn't be offered).
        base = "" if path in (".", "") else path.strip("/") + "/"
        skip: set[str] = set()
        for h in sandbox.hidden_paths:
            if base and not h.startswith(base):
                continue
            first = h[len(base):].split("/", 1)[0]
            skip.update((first, first + "/"))
        entries = [e for e in entries if e not in skip]
        if not entries:
            return ToolResult(ok=True, output="(empty directory)")
        entries.sort(key=lambda e: (not e.endswith("/"), e.rstrip("/")))
        return ToolResult(ok=True, output="\n".join(entries), header=path)

    return ls


async def _list_files(sandbox: SandboxConfig, base: str) -> list[str]:
    """Files under ``base`` (agent-workspace-relative, "" = root) as
    agent-workspace-relative posix paths. No ignore logic: hidden paths are
    shadowed out of the container, so find simply never sees them."""
    target = sandbox.tool_root if base in ("", ".") else sandbox.tool_path(base)
    res = await run_in_sandbox(
        sandbox,
        ["find", target, "-name", ".git", "-prune",
         "-o", "-type", "f", "-print"],
    )
    if res.exit_code != 0:
        raise RuntimeError(res.stderr.strip() or f"could not list files under {base or '.'}")
    prefix = sandbox.tool_root + "/"
    out = []
    for line in res.stdout.splitlines():
        out.append(line[len(prefix):] if line.startswith(prefix) else line)
    return out


def _make_glob_tool(sandbox: SandboxConfig) -> Tool:
    @tool(
        "glob",
        "Find files by glob pattern (e.g. '**/controller.py', 'src/*.py'). "
        "Matches paths relative to 'path' (default workspace root); "
        "ignored files (gitignored, caches, ...) are shadowed out of the "
        "sandbox and never matched. "
        "Results are capped and sorted shortest-path-first.",
        required=("glob",),
        glob=("string", "glob pattern ('**/' = any depth)"),
        path=("string", "base directory to search (default '.')"),
    )
    async def glob(args: dict[str, Any]) -> ToolResult:
        pattern = args.get("glob", "")
        if not pattern:
            return _bad("'glob' is required")
        path = args.get("path") or "."
        if not isinstance(pattern, str) or not isinstance(path, str):
            return _bad("'glob'/'path' must be strings")
        if err := _check_path(sandbox, path, mode="read"):
            return err
        base = "" if path in (".", "") else path.strip("/")
        if base:
            is_dir = await run_in_sandbox(sandbox, ["test", "-d", sandbox.tool_path(base)])
            if is_dir.exit_code != 0:
                return ToolResult(ok=False, error=f"path is not a directory: {path}", parse_error=True)

        files = await _list_files(sandbox, base)
        if base:
            prefix = base + "/"
            files = [f[len(prefix):] for f in files if f.startswith(prefix)]

        matcher = glob_to_regex(pattern)
        matches = [f for f in files if matcher.fullmatch(f)]
        matches.sort(key=len)
        cap = _MAX_GLOB_RESULTS
        if len(matches) > cap:
            output = "\n".join(matches[:cap])
            output += f"\n[showing {cap} of {len(matches)} matches — narrow the glob or 'path']"
            return ToolResult(ok=True, output=output, header=_search_header(pattern, path))
        if not matches:
            return ToolResult(ok=True, output="(no matches)", header=_search_header(pattern, path))
        return ToolResult(ok=True, output="\n".join(matches), header=_search_header(pattern, path))

    return glob


def _rg_argv(regex: str, include: str, path: str, root: str) -> list[str]:
    script = f"cd {root} && exec rg -n --no-heading --color=never"
    if include:
        script += f" -g {shlex.quote(include)}"
    script += f" -- {shlex.quote(regex)} {shlex.quote(path)}"
    return ["sh", "-c", script]


def _git_grep_argv(regex: str, include: str, path: str, root: str) -> list[str]:
    if include:
        spec = include if path in (".", "") else f":(glob){path}/**/{include}"
    else:
        spec = path
    return ["git", "-C", root, "grep", "-n", "--untracked",
            "-E", "-e", regex, "--", spec]


def _plain_grep_argv(regex: str, include: str, path: str, root: str) -> list[str]:
    script = f"cd {root} && exec grep -rnE --color=never"
    if include:
        script += f" --include={shlex.quote(include)}"
    script += f" -- {shlex.quote(regex)} {shlex.quote(path)}"
    return ["sh", "-c", script]


def _search_header(query: str, path: str) -> str:
    """Shared header shape for grep/glob: ``query, scope`` (scope dropped
    when it's the workspace root)."""
    return query if path in (".", "") else f"{query}, {path}"


def _make_grep_tool(sandbox: SandboxConfig) -> Tool:
    @tool(
        "grep",
        "Search file contents with a regex; returns 'path:line:content' "
        "per match. Optional 'include_pattern' scopes the search (a single "
        "path or a glob like '**/*.py'); optional 'path' sets the base "
        "directory. Ignored files (gitignored, caches, ...) are shadowed "
        "out of the sandbox and never match. "
        "Matches are capped per file and in total.",
        required=("regex",),
        regex="string",
        include_pattern=("string", "single path or glob scoping the search"),
        path=("string", "base directory (default '.')"),
    )
    async def grep(args: dict[str, Any]) -> ToolResult:
        regex = args.get("regex", "")
        if not regex:
            return _bad("'regex' is required")
        include = args.get("include_pattern") or ""
        path = args.get("path") or "."
        if not isinstance(regex, str) or not isinstance(include, str) or not isinstance(path, str):
            return _bad("'regex'/'include_pattern'/'path' must be strings")
        if err := _check_path(sandbox, path, mode="read"):
            return err
        if path not in (".", ""):
            exists = await run_in_sandbox(sandbox, ["test", "-e", sandbox.tool_path(path)])
            if exists.exit_code != 0:
                return _bad(f"path does not exist: {path}")

        # Prefer rg; git grep is the fallback and always exists in this
        # harness. Exit 127 means the binary wasn't actually reachable
        # inside the sandbox — try the next backend.
        backends = []
        root = sandbox.tool_root
        if shutil.which("rg") is not None:
            backends.append(_rg_argv(regex, include, path, root))
        backends.append(_git_grep_argv(regex, include, path, root))
        res = await run_in_sandbox(sandbox, backends[0])
        for argv in backends[1:]:
            if res.exit_code != 127:
                break
            res = await run_in_sandbox(sandbox, argv)

        if res.exit_code not in (0, 1) and not res.stdout:
            err = res.stderr.strip()
            if res.exit_code not in (0, 1) and not res.stdout:
                if "not a git repository" in err:
                    # No repo to lean on (isolation disabled): plain grep.
                    res = await run_in_sandbox(
                        sandbox, _plain_grep_argv(regex, include, path, root)
                    )
                if res.exit_code not in (0, 1):
                    return ToolResult(
                        ok=False,
                        error=res.stderr.strip() or f"grep failed (exit {res.exit_code})",
                    )

        counts: dict[str, int] = {}
        lines: list[str] = []
        per_file_truncated = False
        total_truncated = False
        for raw in res.stdout.splitlines():
            parts = raw.split(":", 2)
            if len(parts) < 3:
                continue
            fpath = parts[0]
            if fpath.startswith("./"):
                # git grep prints './'-prefixed paths when the pathspec is
                # '.'; rg strips it. Normalize so backends agree.
                fpath = fpath[2:]
                raw = f"{fpath}:{parts[1]}:{parts[2]}"
            seen = counts.get(fpath, 0)
            if seen >= _MAX_GREP_PER_FILE:
                per_file_truncated = True
                continue
            if len(lines) >= _MAX_GREP_MATCHES:
                total_truncated = True
                break
            counts[fpath] = seen + 1
            lines.append(raw)

        if not lines:
            return ToolResult(ok=True, output="(no matches)")
        output = "\n".join(lines)
        notes = []
        if total_truncated:
            notes.append(
                f"stopped at {_MAX_GREP_MATCHES} matches — narrow "
                "'include_pattern' or the regex to see more"
            )
        elif per_file_truncated:
            notes.append(
                f"some files hit the {_MAX_GREP_PER_FILE}-matches-per-file cap "
                "— narrow 'include_pattern' or the regex"
            )
        if notes:
            output += "\n[" + "; ".join(notes) + "]"
        return ToolResult(
            ok=True,
            output=output,
            header=_search_header(regex, include or path),
        )

    return grep


def build_registry(
    sandbox: SandboxConfig | None,
    session: SandboxSession | None,
    mode: str = "write",
    allow_subagent: bool = True,
    max_calls_per_turn: int | None = None,
    ask_callback: Callable[[list[str]], Awaitable[list[str] | None]] | None = None,
    git_guard: Callable[[str], str | None] | None = None,
) -> ToolRegistry:
    """Build the tool set for one session.

    Plan mode is read-only: read_file plus the exploration tools (ls, glob,
    grep). Write mode trades those for the editing tools: read_file,
    write_file, edit_file, and run_bash. ``ask_user`` is available in both
    modes when ``ask_callback`` is given. ``git_guard`` optionally vetoes
    run_bash commands that would rewrite the user's branch/refs (see
    gitwork.py). ``allow_subagent`` is reserved for subagent registration in a
    later phase.
    """
    registry = ToolRegistry(max_calls_per_turn=max_calls_per_turn)
    # web_search is disabled for now — not implemented well enough to ship.
    # registry.register(web_search_tool)
    if ask_callback is not None:
        registry.register(_make_ask_user_tool(ask_callback))
    if sandbox is not None:
        registry.register(_make_read_tool(sandbox))  # read_file always available
        if mode == "write":
            registry.register(_make_write_tool(sandbox))  # write_file
            registry.register(_make_edit_tool(sandbox))  # edit_file
            if session is not None:
                registry.register(_make_run_bash_tool(session, git_guard))  # run_bash
        else:
            registry.register(_make_ls_tool(sandbox))
            registry.register(_make_glob_tool(sandbox))
            registry.register(_make_grep_tool(sandbox))
    return registry
