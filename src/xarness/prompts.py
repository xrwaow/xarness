"""System prompt for the chat agent.

The prompt carries only facts about the session that is actually running —
facts the model cannot get from the tool descriptions or by looking around:
the mode, what the sandbox is, where the workspace lives, what git does,
whether there is a network, and which toolchains are first-class here.
Behavioral policy (how to edit, when to verify, how to report) is
deliberately absent: the tool descriptions state what a tool does, and
anything beyond that is the model's judgement, not the harness's.

:func:`system_prompt` is the single place that turns a session's sandbox into
text, so the prompt always matches the tool set it was given.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .sandbox import REFS_ANCHOR, has_ref_mounts

if TYPE_CHECKING:  # annotation-only; avoids an import cycle at runtime
    from .sandbox import SandboxConfig


def system_prompt(
    sandbox: "SandboxConfig | None",
    mode: str,
    *,
    tracked: bool = False,
    git_write: bool = False,
) -> str:
    """Render the whole system prompt for one session.

    ``tracked`` and ``git_write`` are session facts the sandbox config does
    not carry (tracking can fail after the config was built)."""
    modename = "PLAN" if mode == "plan" else "WRITE"
    if sandbox is None:
        # No tools at all: the plain fallback, mode still named.
        return (
            "You are a coding agent working in the user's workspace.\n"
            f"Mode: {modename}\n"
        )
    container = sandbox.use_container
    # Same rule as tools.build_registry: plan mode has no shell unless the
    # sandbox is a container (where the read-only mount enforces it).
    shell = container or mode != "plan"
    lines = [
        (
            "You are a coding agent working in a container with the user's "
            "project mounted at /workspace."
        )
        if container else (
            "You are a coding agent working directly in the user's project "
            f"at {sandbox.tool_root}."
        ),
        f"Mode: {modename}" + (" (read-only)" if mode == "plan" else " (read/write)"),
        "",
        "Environment",
    ]
    if container:
        lines.append(
            "- /tmp is a tmpfs wiped when the shell restarts; the host home "
            "and the rest of the host filesystem are not mounted."
        )
    else:
        lines.append("- No container: tools run directly on the host.")
    lines.append("- Tool paths are relative to the workspace root.")
    if container:
        # Unsandboxed mode has no network isolation, so no claim either way.
        if sandbox.allow_network:
            lines.append("- Network access is available.")
        else:
            lines.append(
                "- There is no network (uv runs offline, UV_OFFLINE): only "
                "cached packages and index data are available."
            )
    if sandbox.uses_project_venv():
        if container:
            lines.append(
                "- Python: `.venv` is the sandbox's uv-managed environment "
                "(not the host's) and `uv.lock` is read-only; use "
                "`uv run --frozen <cmd>` or `.venv/bin/python`."
            )
        else:
            lines.append(
                "- Python: `.venv` is the project's uv environment; use "
                "`uv run <cmd>` or `.venv/bin/python`."
            )
    if container and (sandbox.workspace / "Cargo.toml").is_file():
        lines.append(
            "- Cargo: builds use `target-sandbox/`, which is not tracked; "
            "the user's `target/` is untouched."
        )
    # Nothing can change in READ mode, so git/undo facts only matter when
    # editing is possible.
    if shell and mode != "plan":
        if git_write:
            lines.append(
                "- git is read-write in the shell: staging, commits, branches "
                "and switches work."
            )
        else:
            lines.append("- git in the shell is read-only (inspection only).")
    # With allow_git_write the agent manages git itself and the snapshot
    # guarantees no longer hold — say nothing rather than over-promise.
    if tracked and not git_write and mode != "plan":
        lines.append(
            "- Changes are snapshotted so the user can review and undo them, "
            "except gitignored files, binaries and media, files over 1 MB, "
            "and Cargo build trees."
        )
    if has_ref_mounts(sandbox):
        lines.append(
            f"- Externally referenced paths are mounted read-only under "
            f"{REFS_ANCHOR}/."
        )
    return "\n".join(lines) + "\n"


# Appended as one extra user message to the live conversation, so the whole
# prefix (system prompt + history) stays byte-identical to the round that just
# ran and the provider's prompt cache still covers it. The instruction itself
# carries the "no tools" notice on purpose: changing the system prompt or the
# tool list would invalidate that cached prefix.
COMPACTION_PROMPT = (
    "The conversation above is being handed off to a fresh context. This "
    "round runs without tools: do not call a tool, do not attempt to make "
    "changes, and do not continue the task. Instead write the handoff the "
    "next assistant instance needs to continue the work seamlessly. Cover, "
    "in this order:\n"
    "1. The task: what the user is trying to accomplish, in their terms.\n"
    "2. What's known so far: key decisions, file paths, relevant commands "
    "and their results, constraints discovered.\n"
    "3. What's next: the current state of any work in progress and the "
    "immediate next steps.\n"
    "Be concise — a few short paragraphs at most. Reply with the handoff "
    "only."
)
