"""System prompts for the chat agent.

The top-level conversation always leads with a system message built from
``GENERAL_SYSTEM_PROMPT`` plus the mode-specific clause for the active mode
("plan" or "write"). ``system_prompt_for`` is the single place that assembles
it, so the prompt the model sees always matches the tool set it was given.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only; avoids an import cycle at runtime
    from .sandbox import SandboxConfig

GENERAL_SYSTEM_PROMPT = (
    "You are a coding agent working inside a sandboxed workspace."
    "Ground rules:\n"
    "- File paths in tool calls are relative to the workspace root.\n"
    "- Files under '.refs/' are read-only external references.\n"
    "- read_file returns a structural outline for large files; read specific "
    "sections of those with start_line/end_line instead of guessing at "
    "contents.\n"
    "- Use ask_user when you need a decision or a missing detail.\n"
    "- Verify your changes when you can (run tests, build, or search) before "
    "reporting success, and say plainly when something didn't work.\n"
    "- Be concise: answer the question or make the change, then summarize what "
    "you did — don't narrate every step."
)

MODE_CLAUSE = {
    "plan": (
        "You are in PLAN mode: read-only. The workspace is mounted read-only, "
        "so run_bash cannot change files — use it to explore (ls, find, rg, "
        "cat) alongside read_file. Pipe search output through '| head -n 100' "
        "(or count with 'wc -l') to keep output sane, and search paths first "
        "before reading whole files. You cannot edit files; produce a plan or "
        "answer, not changes."
    ),
    "write": (
        "You are in WRITE mode: read files with read_file, create new files "
        "with write_file, make targeted changes with edit_file, and run shell "
        "commands with run_bash. Pipe search output (rg/find/grep) through "
        "'| head -n 100' so results stay sane. Prefer to put every change to "
        "the same file in one edit_file call."
    ),
}


def system_prompt_for(base: str, mode: str, sandbox: "SandboxConfig | None" = None) -> str:
    """Assemble the full system prompt for the given mode.

    With a sandbox, append the working-directory facts: where the sandboxed
    tools are rooted, and the same directory's host path (the cwd
    run_bash_host starts in), so host commands don't need to guess paths.
    """
    prompt = f"{base}\n\n{MODE_CLAUSE[mode]}"
    if sandbox is not None:
        host_dir = Path(sandbox.workspace) / sandbox.subtree
        prompt += (
            "\n\nWorking directory (one directory, two paths):\n"
            f"- relative path with run_bash and the file tools - resolves under {sandbox.tool_root}\n"
            f"- relative path with run_bash_host - resolves under {host_dir}\n"
            "Both are the same files, so edits made in the sandbox are already "
            "on the host.\n\n"
            "Running tests and project tooling:\n"
            "The sandbox has no project toolchain or environment (no venv, "
            "compiler, or installed dependencies). Run tests, builds, and "
            "anything that needs the project's toolchain with run_bash_host."
        )
    return prompt
