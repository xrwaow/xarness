"""System prompt for the chat agent.

The prompt carries only facts about the session that is actually running —
facts the model cannot get from the tool descriptions or by looking around:
the mode, what the sandbox is, where the workspace lives, what git does,
whether there is a network, and which toolchains are first-class here.
Behavioral policy (how to edit, when to verify, how to report) is
deliberately absent: the tool descriptions state what a tool does, and
anything beyond that is the model's judgement, not the harness's.

``prompt_facts`` collects those facts from a :class:`~xarness.sandbox.SandboxConfig`;
``system_prompt_for`` is the single place that turns them into text, so the
prompt always matches the tool set it was given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .sandbox import REFS_ANCHOR, has_ref_mounts

if TYPE_CHECKING:  # annotation-only; avoids an import cycle at runtime
    from .sandbox import SandboxConfig

CONTAINER_HEADER = (
    "You are a coding agent working in a container with the user's project "
    "mounted at /workspace."
)
HOST_HEADER = (
    "You are a coding agent working directly in the user's project at {root}."
)
PLAIN_HEADER = "You are a coding agent working in the user's workspace."

WRITE_MODE = "Mode: WRITE (read/write)"
WRITE_MODE_NO_TOOLS = "Mode: WRITE"
READ_MODE = "Mode: PLAN (read-only)"
READ_MODE_NO_TOOLS = "Mode: PLAN"

CONTAINER_LINE = (
    "- /tmp is a tmpfs wiped when the shell restarts; the host home and the "
    "rest of the host filesystem are not mounted."
)
HOST_LINE = "- No container: tools run directly on the host."
TOOL_PATH_LINE = "- Tool paths are relative to the workspace root."
NETWORK_ON = "- Network access is available."
NETWORK_OFF = (
    "- There is no network (uv runs offline, UV_OFFLINE): only cached "
    "packages and index data are available."
)

PYTHON_SANDBOXED = (
    "- Python: `.venv` is the sandbox's uv-managed environment (not the "
    "host's) and `uv.lock` is read-only; use `uv run --frozen <cmd>` or "
    "`.venv/bin/python`."
)
PYTHON_HOST = (
    "- Python: `.venv` is the project's uv environment; use `uv run <cmd>` "
    "or `.venv/bin/python`."
)
CARGO_LINE = (
    "- Cargo: builds use `target-sandbox/`, which is not tracked; the "
    "user's `target/` is untouched."
)

GIT_READONLY = "- git in the shell is read-only (inspection only)."
GIT_WRITABLE = (
    "- git is read-write in the shell: staging, commits, branches and "
    "switches work."
)
TRACKING_CLAUSE = (
    "- Changes are snapshotted so the user can review and undo them, except "
    "gitignored files, binaries and media, files over 1 MB, and Cargo build "
    "trees."
)


@dataclass(frozen=True, slots=True)
class PromptFacts:
    """Everything the prompt says about *this* session."""

    mode: str  # "plan" | "write"
    root: str = "/workspace"
    container: bool = True
    # run_bash is registered (in plan mode it is not, unsandboxed).
    shell: bool = True
    network: bool = False
    git_write: bool = False
    tracked: bool = False
    # The canonical read-only external-references area is mounted.
    refs: bool = False
    python: bool = False
    cargo: bool = False
    # A sandbox exists at all (tools are enabled).
    paths: bool = True


def prompt_facts(
    sandbox: "SandboxConfig | None",
    mode: str,
    *,
    tracked: bool = False,
    git_write: bool = False,
) -> PromptFacts:
    """Collect the facts a prompt is rendered from."""
    if sandbox is None:
        return PromptFacts(mode=mode, container=False, shell=False, paths=False)
    container = sandbox.use_container
    # Same rule as tools.build_registry: plan mode has no shell unless the
    # sandbox is a container (where the read-only mount enforces it).
    shell = container or mode != "plan"
    return PromptFacts(
        mode=mode,
        root=sandbox.tool_root,
        container=container,
        shell=shell,
        network=sandbox.allow_network,
        git_write=git_write,
        tracked=tracked,
        refs=has_ref_mounts(sandbox),
        python=sandbox.uses_project_venv(),
        cargo=(sandbox.workspace / "Cargo.toml").is_file(),
    )


def _mode_line(facts: PromptFacts) -> str:
    if facts.mode == "plan":
        return READ_MODE if facts.paths else READ_MODE_NO_TOOLS
    return WRITE_MODE if facts.paths else WRITE_MODE_NO_TOOLS


def _toolchain(facts: PromptFacts) -> tuple[str, ...]:
    lines: list[str] = []
    if facts.python:
        lines.append(PYTHON_SANDBOXED if facts.container else PYTHON_HOST)
    if facts.cargo and facts.container:
        lines.append(CARGO_LINE)
    return tuple(lines)


def _environment(facts: PromptFacts) -> list[str]:
    lines = [CONTAINER_LINE if facts.container else HOST_LINE]
    lines.append(TOOL_PATH_LINE)
    if facts.container:
        # Unsandboxed mode has no network isolation, so no claim either way.
        lines.append(NETWORK_ON if facts.network else NETWORK_OFF)
    lines.extend(_toolchain(facts))
    # Nothing can change in READ mode, so git/undo facts only matter when
    # editing is possible.
    if facts.mode != "plan":
        if facts.shell:
            lines.append(GIT_WRITABLE if facts.git_write else GIT_READONLY)
        # With allow_git_write the agent manages git itself and the snapshot
        # guarantees no longer hold — say nothing rather than over-promise.
        if facts.tracked and not facts.git_write:
            lines.append(TRACKING_CLAUSE)
    if facts.refs:
        lines.append(
            f"- Externally referenced paths are mounted read-only under "
            f"{REFS_ANCHOR}/."
        )
    return lines


def system_prompt_for(facts: PromptFacts) -> str:
    """Render the whole system prompt for one session."""
    mode = _mode_line(facts)
    if not facts.paths:
        return f"{PLAIN_HEADER}\n{mode}\n"
    header = (
        CONTAINER_HEADER if facts.container
        else HOST_HEADER.format(root=facts.root)
    )
    return "\n".join([
        header,
        mode,
        "",
        "Environment",
        *(_environment(facts)),
    ]) + "\n"


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
