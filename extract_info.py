#!/usr/bin/env python3
"""Print the system prompt and tool definitions a fresh ``xarness`` session
sends to the model.

With no arguments this reproduces what a default ``xarness`` start (current
directory as the workspace, the default config/preset/sandbox, write mode)
puts in the request. The CLI flags that change that prompt are accepted too,
with the same spellings and semantics as ``xarness`` itself: ``--workspace``,
``--sandbox``, ``--preset``, ``--network``, ``--gpu``, ``--enable_tracking``
and ``--no-init-repo``. ``--mode`` selects the plan/write prompt that the TUI
would toggle to with Tab.

Output is plain text: a ``=== SYSTEM PROMPT ===`` section followed by the
provider-facing ``=== TOOL DEFINITIONS ===`` JSON array (exactly what
``ToolRegistry.schema()`` returns). ``--json`` emits both as one JSON object
instead.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace


# Running from a checkout: use the local package even if it is not installed.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from xarness import gitwork  # noqa: E402
from xarness.cli import build_parser  # noqa: E402
from xarness.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    ConfigError,
    ensure_config,
    load_sectioned,
)
from xarness.prompts import system_prompt  # noqa: E402
from xarness.sandbox import SandboxRuntime, SandboxUnavailable  # noqa: E402
from xarness.session_store import load_metas  # noqa: E402
from xarness.tools import build_registry  # noqa: E402
from xarness.workspace import load_workspace_refs  # noqa: E402


async def _ask_user(questions: list[str]) -> list[str] | None:
    """Stand-in for the session's ask callback: its presence is what puts
    ``ask_user`` in the tool set; extraction never calls it."""
    return None


async def _spawn_subagent(prompt: str, mode: str) -> str:
    """Stand-in for the host's spawner: its presence is what puts
    ``subagent`` in the tool set; extraction never calls it."""
    return ""


def _workspace_last_sandbox(workspace: Path) -> str | None:
    """The sandbox profile the workspace last used, as ``xarness`` resolves it
    from saved sessions. Read-only: no database file means no sessions."""
    best: tuple[str, str] | None = None
    for meta in load_metas().values():
        saved_ws = meta.get("workspace")
        if not saved_ws:
            continue
        try:
            if Path(saved_ws).resolve() != workspace.resolve():
                continue
        except OSError:
            continue
        stamp = meta.get("updated_at") or ""
        if best is None or stamp > best[0]:
            best = (stamp, meta.get("sandbox_name"))
    return best[1] if best else None


def _detect_repo(workspace: Path):
    """The workspace's git repo, if any. Used only to decide whether change
    tracking is on (and thus whether the prompt promises snapshots)."""
    try:
        return asyncio.run(gitwork.detect_repo(workspace))
    except Exception:  # noqa: BLE001 - tracking is best-effort, like startup's
        return None


def _add_info_flags(parser: argparse.ArgumentParser) -> None:
    # Same flag spellings as xarness.cli, so anything that changes the prompt
    # can be asked about the same way. --model/--session/--server/... are
    # inherited but do not affect the prompt.
    parser.add_argument(
        "--mode", choices=("write", "plan"), default="write",
        help="Session mode: write (default, read/write) or plan (read-only). "
        "This is the TUI's Tab toggle, not an xarness startup flag.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit one JSON object ({'system_prompt': ..., 'tools': [...]}) "
        "instead of the two text sections.",
    )


def _render(args: argparse.Namespace) -> tuple[str, list[dict]]:
    """Build the prompt and tool schema for the requested session settings."""
    # Same startup sequence as the TUI: create the starter config on first
    # run, then load it.
    notices = ensure_config(DEFAULT_CONFIG_PATH)
    for notice in notices:
        print(f"extract_info: {notice}", file=sys.stderr)
    sectioned = load_sectioned(DEFAULT_CONFIG_PATH)

    workspace = (args.workspace or Path.cwd()).expanduser().resolve()
    if not workspace.is_dir():
        # xarness creates a missing workspace at session start; this tool has
        # nothing to inspect there and says so instead of making directories.
        raise ConfigError(f"workspace directory does not exist: {workspace}")

    # Model/sandbox precedence, as Host._build_parts: explicit request >
    # preset > the workspace's most recent saved choice.
    preset_name = args.preset or sectioned.default_preset
    preset_settings = sectioned.presets.get(preset_name)
    last_sandbox = _workspace_last_sandbox(workspace)
    sandbox_name = args.sandbox or (
        preset_settings.sandbox if preset_settings else None
    ) or last_sandbox
    sandbox_key, settings = sectioned.resolve_sandbox(sandbox_name)

    # Tri-state CLI overrides: a given flag wins over the sandbox profile.
    overrides = {
        "allow_network": args.network,
        "gpu_access": args.gpu,
        "use_sandbox": args.use_sandbox,
        "enable_tracking": args.enable_tracking,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if overrides:
        settings = settings.model_copy(update=overrides)

    # Tracking is on unless disabled, the workspace has no repo and
    # --no-init-repo forbids creating the private snapshot store, or the
    # repo cannot be opened. (A first run without --no-init-repo would
    # initialize the store, so tracking is on.)
    repo = None
    tracked = False
    if settings.enable_tracking:
        repo = _detect_repo(workspace)
        tracked = repo is not None or not args.no_init_repo

    # SandboxConfig only needs the repo's git dir and external flag from it.
    repo_state = (
        SimpleNamespace(
            root=repo.root, git_dir=repo.git_common_dir, external=repo.external,
        )
        if repo is not None else None
    )
    try:
        runtime = SandboxRuntime.make(
            workspace, settings, repo_state, name=sandbox_key,
            extra_refs=load_workspace_refs(workspace),
        )
    except SandboxUnavailable as exc:
        # Same fallback as Host._build_parts: chat works, tools are disabled.
        print(f"extract_info: {exc}; filesystem/shell tools are disabled",
              file=sys.stderr)
        runtime = SandboxRuntime(name=sandbox_key, settings=settings)
    runtime.set_mode(args.mode)

    registry = build_registry(
        runtime.config, runtime.shell, mode=args.mode,
        ask_callback=_ask_user,
        default_timeout=settings.default_timeout,
        spawn_subagent=_spawn_subagent,
    )
    prompt = system_prompt(
        runtime.config, args.mode,
        tracked=tracked,
        git_write=runtime.allow_git_write,
    )
    return prompt, registry.schema()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.prog = "extract_info.py"
    parser.description = (
        "Print the system prompt and tool definitions an LLM receives from a "
        "default xarness session, with the same flags that adjust them."
    )
    _add_info_flags(parser)
    args = parser.parse_args(argv)

    try:
        prompt, tools = _render(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(
            {"system_prompt": prompt, "tools": tools},
            indent=2, ensure_ascii=False,
        ))
    else:
        print("=== SYSTEM PROMPT ===")
        print(prompt, end="" if prompt.endswith("\n") else "\n")
        print()
        print(f"=== TOOL DEFINITIONS ({len(tools)}) ===")
        print(json.dumps(tools, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
