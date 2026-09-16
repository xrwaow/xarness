"""Tool registry and the built-in tools.

read_file/write_file/edit_file/ls/glob/grep/run_bash execute inside a bwrap
sandbox (see sandbox.py) scoped to one workspace directory, with no network
access — run_bash uses a persistent SandboxSession so shell state survives
across calls within one chat. web_search is the one tool that runs outside
the sandbox, since it's the only one that needs a real network path.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .gitwork import GitInfo, GitWorktreeError, sync_gitignore
from .ignore import DEFAULT_IGNORE_DIRS, glob_to_regex, is_ignored
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

# Result caps for the read-only exploration tools (ls/glob/grep).
_MAX_GLOB_RESULTS = 200
_MAX_GREP_MATCHES = 200
_MAX_GREP_PER_FILE = 20


def _sandbox_path(path: str) -> str:
    """Anchor a validated workspace-relative path at the sandbox mount point.

    Commands must not rely on the process cwd: with a subtree session bwrap
    chdirs into the subtree, while tool paths are workspace-root-relative
    (see validate_relpath)."""
    return f"/workspace/{path}"


async def _sync_gitignore_best_effort(git_info: GitInfo | None) -> None:
    """Pull the original workspace's live .gitignore into the worktree before
    ignore-aware exploration, so uncommitted user edits are respected. Best
    effort: no git isolation (or a failed copy) just means the worktree's
    committed copy is used as-is."""
    if git_info is None:
        return
    try:
        await sync_gitignore(git_info)
    except (GitWorktreeError, OSError) as exc:
        logging.getLogger(__name__).debug("gitignore re-sync skipped: %s", exc)


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
        error = sandbox.validate_relpath(path, mode="read")
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

        target = _sandbox_path(path)
        # awk's NR counts lines regardless of a trailing newline (wc -l counts
        # newlines, undercounting files that don't end with one).
        count_result = await run_in_sandbox(sandbox, ["awk", "END{print NR}", target])
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

        result = await run_in_sandbox(sandbox, ["sed", "-n", f"{start},{end}p", target])
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
        cat_result = await run_in_sandbox(sandbox, ["cat", _sandbox_path(path)])
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

        target = _sandbox_path(path)
        exists = await run_in_sandbox(sandbox, ["test", "-e", target])
        created = exists.exit_code != 0
        write_result = await run_in_sandbox(
            sandbox, ["tee", target], input_bytes=content.encode()
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

        read_result = await run_in_sandbox(sandbox, ["cat", _sandbox_path(path)])
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
            sandbox, ["tee", _sandbox_path(path)], input_bytes=updated.encode()
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
        # The callback (controller.compact via the app) returns the summary
        # prefixed with the before/after token counts on the first line.
        summary = await compact_callback()
        return ToolResult(ok=True, output=summary)

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


# ---------------------------------------------------------------------------
# read-only exploration: ls / glob / grep (available in plan and write mode)


async def _git_ignored_set(sandbox: SandboxConfig, rel_paths: list[str]) -> set[str]:
    """Workspace-relative paths the repo's ignore rules flag (ignore layer a).
    ``--no-index`` makes this purely rule-based, matching rg's behavior —
    without it git refuses to report tracked files, and a tracked file that
    matches an ignore rule would then be visible in ls/glob but hidden in
    grep. Empty when git can't answer (no repo, git failed) — the caller
    then keeps only the default-ignore filtering."""
    if not rel_paths:
        return set()
    res = await run_in_sandbox(
        sandbox,
        ["git", "-C", "/workspace", "check-ignore", "--stdin", "--no-index"],
        input_bytes=("\n".join(rel_paths) + "\n").encode(),
    )
    if res.exit_code not in (0, 1):
        return set()
    return {line for line in res.stdout.splitlines() if line}


def _make_ls_tool(sandbox: SandboxConfig, git_info: GitInfo | None) -> Tool:
    async def _ls(args: dict[str, Any]) -> ToolResult:
        path = args.get("path") or "."
        if not isinstance(path, str):
            return ToolResult(ok=False, error="'path' must be a string", parse_error=True)
        error = sandbox.validate_relpath(path, mode="read")
        if error:
            return ToolResult(ok=False, error=error, parse_error=True)
        await _sync_gitignore_best_effort(git_info)

        target = _sandbox_path(path)
        is_dir = await run_in_sandbox(sandbox, ["test", "-d", target])
        if is_dir.exit_code != 0:
            exists = await run_in_sandbox(sandbox, ["test", "-e", target])
            if exists.exit_code == 0:
                return ToolResult(ok=False, error=f"'{path}' is not a directory", parse_error=True)
            return ToolResult(ok=False, error=f"path does not exist: {path}", parse_error=True)

        # -p suffixes directories with '/', which doubles as the dir marker.
        listing = await run_in_sandbox(sandbox, ["ls", "-A", "-p", "--", target])
        if listing.exit_code != 0:
            return ToolResult(ok=False, error=listing.stderr.strip() or "ls failed")
        entries = [e for e in listing.stdout.splitlines() if e]
        if not entries:
            return ToolResult(ok=True, output="(empty directory)")

        base = "" if path in (".", "") else path.rstrip("/") + "/"
        git_ignored = await _git_ignored_set(
            sandbox, [base + e.rstrip("/") for e in entries]
        )
        kept = [
            e for e in entries
            if base + e.rstrip("/") not in git_ignored  # layer a
            and not is_ignored(base + e.rstrip("/"))    # layer b
        ]
        if not kept:
            return ToolResult(ok=True, output="(nothing left after ignore filtering)")
        kept.sort(key=lambda e: (not e.endswith("/"), e.rstrip("/")))
        return ToolResult(ok=True, output="\n".join(kept))

    return Tool(
        name="ls",
        description=(
            "List a directory's immediate entries (not recursive); directories "
            "end with '/'. Gitignored files and default-ignored dirs (.git, "
            "node_modules, __pycache__, ...) are hidden. Paths are relative "
            "to the workspace root."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "directory to list (default '.')"},
            },
        },
        handler=_ls,
    )


async def _list_files_respecting_gitignore(sandbox: SandboxConfig, base: str) -> list[str]:
    """Files under ``base`` (workspace-root-relative, "" = root) as
    workspace-root-relative posix paths, honoring the repo's ignore rules
    (layer a) via git. Falls back to ``find`` when the workspace isn't a git
    repo; default-ignored dirs are pruned either way by the caller (layer b)."""
    argv = ["git", "-C", "/workspace", "ls-files", "--cached", "--others",
            "--exclude-standard", "--"]
    if base:
        argv.append(base)
    res = await run_in_sandbox(sandbox, argv)
    if res.exit_code == 0:
        return [line for line in res.stdout.splitlines() if line]

    target = _sandbox_path(base)
    prune: list[str] = ["("]
    for i, name in enumerate(sorted(DEFAULT_IGNORE_DIRS)):
        if i:
            prune.append("-o")
        prune += ["-name", name]
    prune += [")", "-prune", "-o", "-type", "f", "-print"]
    res = await run_in_sandbox(sandbox, ["find", target, *prune])
    if res.exit_code != 0:
        raise RuntimeError(res.stderr.strip() or f"could not list files under {base or '.'}")
    prefix = "/workspace/"
    return [line[len(prefix):] for line in res.stdout.splitlines() if line.startswith(prefix)]


def _make_glob_tool(sandbox: SandboxConfig, git_info: GitInfo | None) -> Tool:
    async def _glob(args: dict[str, Any]) -> ToolResult:
        pattern = args.get("glob", "")
        if not pattern:
            return ToolResult(ok=False, error="'glob' is required", parse_error=True)
        path = args.get("path") or "."
        if not isinstance(pattern, str) or not isinstance(path, str):
            return ToolResult(ok=False, error="'glob'/'path' must be strings", parse_error=True)
        error = sandbox.validate_relpath(path, mode="read")
        if error:
            return ToolResult(ok=False, error=error, parse_error=True)
        await _sync_gitignore_best_effort(git_info)

        base = "" if path in (".", "") else path.strip("/")
        if base:
            is_dir = await run_in_sandbox(sandbox, ["test", "-d", _sandbox_path(base)])
            if is_dir.exit_code != 0:
                return ToolResult(ok=False, error=f"path is not a directory: {path}", parse_error=True)

        files = await _list_files_respecting_gitignore(sandbox, base)
        # Layer a, rule-based (covers tracked files matching ignore rules,
        # which ls-files --cached would otherwise always list).
        git_ignored = await _git_ignored_set(sandbox, files)
        files = [f for f in files if f not in git_ignored]
        if base:
            prefix = base + "/"
            files = [f[len(prefix):] for f in files if f.startswith(prefix)]
        files = [f for f in files if not is_ignored(f"{base}/{f}" if base else f)]  # layer b

        matcher = glob_to_regex(pattern)
        matches = [f for f in files if matcher.fullmatch(f)]
        matches.sort(key=len)
        cap = _MAX_GLOB_RESULTS
        if len(matches) > cap:
            output = "\n".join(matches[:cap])
            output += f"\n[showing {cap} of {len(matches)} matches — narrow the glob or 'path']"
            return ToolResult(ok=True, output=output)
        if not matches:
            return ToolResult(ok=True, output="(no matches)")
        return ToolResult(ok=True, output="\n".join(matches))

    return Tool(
        name="glob",
        description=(
            "Find files by glob pattern (e.g. '**/controller.py', 'src/*.py'). "
            "Matches paths relative to 'path' (default workspace root); "
            "gitignored files and default-ignored dirs are never matched. "
            "Results are capped and sorted shortest-path-first."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "glob": {"type": "string", "description": "glob pattern ('**/' = any depth)"},
                "path": {"type": "string", "description": "base directory to search (default '.')"},
            },
            "required": ["glob"],
        },
        handler=_glob,
    )


def _rg_argv(regex: str, include: str, path: str) -> list[str]:
    script = "cd /workspace && exec rg -n --no-heading --color=never"
    if include:
        script += f" -g {shlex.quote(include)}"
    script += f" -- {shlex.quote(regex)} {shlex.quote(path)}"
    return ["sh", "-c", script]


def _git_grep_argv(regex: str, include: str, path: str) -> list[str]:
    if include:
        spec = include if path in (".", "") else f":(glob){path}/**/{include}"
    else:
        spec = path
    return ["git", "-C", "/workspace", "grep", "-n", "--untracked",
            "--exclude-standard", "-E", "-e", regex, "--", spec]


def _plain_grep_argv(regex: str, include: str, path: str) -> list[str]:
    script = "cd /workspace && exec grep -rnE --color=never"
    if include:
        script += f" --include={shlex.quote(include)}"
    script += f" -- {shlex.quote(regex)} {shlex.quote(path)}"
    return ["sh", "-c", script]


def _make_grep_tool(sandbox: SandboxConfig, git_info: GitInfo | None) -> Tool:
    async def _grep(args: dict[str, Any]) -> ToolResult:
        regex = args.get("regex", "")
        if not regex:
            return ToolResult(ok=False, error="'regex' is required", parse_error=True)
        include = args.get("include_pattern") or ""
        path = args.get("path") or "."
        if not isinstance(regex, str) or not isinstance(include, str) or not isinstance(path, str):
            return ToolResult(
                ok=False, error="'regex'/'include_pattern'/'path' must be strings", parse_error=True
            )
        error = sandbox.validate_relpath(path, mode="read")
        if error:
            return ToolResult(ok=False, error=error, parse_error=True)
        await _sync_gitignore_best_effort(git_info)

        if path not in (".", ""):
            exists = await run_in_sandbox(sandbox, ["test", "-e", _sandbox_path(path)])
            if exists.exit_code != 0:
                return ToolResult(ok=False, error=f"path does not exist: {path}", parse_error=True)

        # Prefer rg (reads .gitignore natively); git grep is the fallback and
        # always exists in this harness. Exit 127 means the binary wasn't
        # actually reachable inside the sandbox — try the next backend.
        backends = []
        if shutil.which("rg") is not None:
            backends.append(_rg_argv(regex, include, path))
        backends.append(_git_grep_argv(regex, include, path))
        res = await run_in_sandbox(sandbox, backends[0])
        for argv in backends[1:]:
            if res.exit_code != 127:
                break
            res = await run_in_sandbox(sandbox, argv)

        if res.exit_code not in (0, 1) and not res.stdout:
            err = res.stderr.strip()
            if "cannot be used for tracked contents" in err:
                # A tracked file matches an ignore rule; git refuses the
                # combination outright. Retry without --exclude-standard —
                # layer b still filters, we just lose git's layer a here.
                retry = [a for a in backends[-1] if a != "--exclude-standard"]
                res = await run_in_sandbox(sandbox, retry)
            err = res.stderr.strip()
            if res.exit_code not in (0, 1) and not res.stdout:
                if "not a git repository" in err:
                    # No repo to lean on (isolation disabled): plain grep,
                    # with only the default-ignore filtering on top.
                    res = await run_in_sandbox(
                        sandbox, _plain_grep_argv(regex, include, path)
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
            if is_ignored(fpath):  # layer b — neither -g nor pathspec knows it
                continue
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
        return ToolResult(ok=True, output=output)

    return Tool(
        name="grep",
        description=(
            "Search file contents with a regex; returns 'path:line:content' "
            "per match. Optional 'include_pattern' scopes the search (a single "
            "path or a glob like '**/*.py'); optional 'path' sets the base "
            "directory. Gitignored files and default-ignored dirs are skipped. "
            "Matches are capped per file and in total."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "regex": {"type": "string"},
                "include_pattern": {
                    "type": "string",
                    "description": "single path or glob scoping the search",
                },
                "path": {"type": "string", "description": "base directory (default '.')"},
            },
            "required": ["regex"],
        },
        handler=_grep,
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
    git_info: GitInfo | None = None,
) -> ToolRegistry:
    """Build the tool set for one session.

    Both modes expose read_file plus the read-only exploration tools (ls,
    glob, grep); write mode additionally exposes write_file, edit_file, and
    run_bash. ``ask_callback`` enables the ask tool (prompts the user in the
    TUI); ``compact_callback`` enables the compact tool (summarizes +
    truncates the conversation). ``git_guard`` optionally vetoes run_bash
    commands that would interfere with harness-managed worktree/branch
    lifecycle (see gitwork.py). ``git_info`` (when the session is git-
    isolated) lets the exploration tools re-sync the original workspace's
    live .gitignore before filtering. ``allow_subagent`` is reserved for
    subagent registration in a later phase.
    """
    registry = ToolRegistry(max_calls_per_turn=max_calls_per_turn)
    registry.register(web_search_tool)
    if ask_callback is not None:
        registry.register(_make_ask_tool(ask_callback))
    if compact_callback is not None:
        registry.register(_make_compact_tool(compact_callback))
    if sandbox is not None:
        registry.register(_make_read_tool(sandbox))  # read_file always available
        registry.register(_make_ls_tool(sandbox, git_info))
        registry.register(_make_glob_tool(sandbox, git_info))
        registry.register(_make_grep_tool(sandbox, git_info))
        if mode == "write":
            registry.register(_make_write_tool(sandbox))  # write_file
            registry.register(_make_edit_tool(sandbox))  # edit_file
            if session is not None:
                registry.register(_make_run_bash_tool(session, git_guard))  # run_bash
    return registry
