"""System prompts for the chat agent.

The top-level conversation always leads with a system message built from
``GENERAL_SYSTEM_PROMPT`` plus the mode-specific clause for the active mode
("plan" or "write"). ``system_prompt_for`` is the single place that assembles
it, so the prompt the model sees always matches the tool set it was given.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only; avoids an import cycle at runtime
    from .sandbox import SandboxConfig

GENERAL_SYSTEM_PROMPT = """You are a coding agent working inside a sandboxed workspace.
Ground rules:
- File paths in tool calls are relative to the workspace root.
- External references are read-only: address them as '.refs/<alias>' in tool calls; in shell commands they live at /tmp/refs/<alias>. Nothing is ever created in the worktree for them.
- read_file returns a structural outline for large files; read specific sections of those with start_line/end_line instead of guessing at contents.
- Use ask_user when you need a decision that cannot be infered from context.
- Pipe search output (rg/find/grep) through '| head -n' so results stay sane.
"""

MODE_CLAUSE = {
    "plan": """You are in PLAN mode: the workspace is mounted read-only. Produce a plan or answer, not changes.""",
    "write": """You are in WRITE mode:
- Verify your changes before reporting success, scaling the effort to the change: a quick tarteged check is enough for small changes, and save full test runs for larger or riskier ones.
- Prefer to put every change to the same file in one edit_file call.
""",
}


GITIGNORE_CLAUSE = """- Gitignored files (and paths in .git/info/exclude) show up as 0-byte empty files.
"""


def system_prompt_for(base: str, mode: str, sandbox: "SandboxConfig | None" = None) -> str:
    """Assemble the full system prompt for the given mode.

    With a sandbox, append the working-directory facts: where the sandboxed
    tools are rooted. When ignore rules are respected, mention that
    gitignored files appear as 0-byte placeholders.
    """
    prompt = f"{base}\n\n{MODE_CLAUSE[mode]}"
    if sandbox is not None and sandbox.respect_gitignore:
        prompt += f"\n\n{GITIGNORE_CLAUSE.rstrip()}"
    if sandbox is not None:
        prompt += (
            "\n\nRunning tests and project tooling:\n"
            "The sandbox has writable, persistent caches for uv, pip, cargo "
            "and npm (UV_CACHE_DIR, PIP_CACHE_DIR, CARGO_HOME, "
            "npm_config_cache), so ``uv sync``/``uv run`` and similar work "
            "inside run_bash without network. Run tests and builds with "
            "run_bash."
        )
    return prompt
