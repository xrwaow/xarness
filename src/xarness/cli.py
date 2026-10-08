"""Command-line entrypoint: ``xarness`` starts the interactive chat TUI."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from . import __version__
from .config import (
    DEFAULT_CONFIG_PATH, ConfigError, LoadedConfig, ensure_config,
    load_config, resolve_api_key,
)
from .gitwork import GitInfo
from .sandbox import SandboxConfig, SandboxSession, SandboxUnavailable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xarness",
        description="Terminal chat client for OpenAI-compatible LLM APIs.",
    )
    parser.add_argument("--version", action="version", version=f"xarness {__version__}")
    parser.add_argument(
        "--profile", default=None,
        help="Named profile from the config's 'profiles:' section "
        "(default: default_profile, or the only profile if just one is defined)",
    )
    parser.add_argument(
        "--workspace", type=Path, default=None,
        help="Directory the read_file/edit_file/run_bash tools are sandboxed to "
        "(default: current directory).",
    )
    parser.add_argument(
        "--no-init-repo",
        action="store_true",
        help="Don't auto-initialize a git repo when the workspace has none "
        "(default: xarness runs `git init` itself so change tracking works "
        "without setup; your files are never modified by this).",
    )
    parser.add_argument(
        "--no-session", action="store_true",
        help="Don't save this conversation as a resumable session.",
    )
    keep = parser.add_mutually_exclusive_group()
    keep.add_argument(
        "--keep-reasoning", action="store_true", default=None,
        help="Send assistant reasoning back to the model on later rounds "
        "(default: the profile's keep_reasoning setting, true unless overridden).",
    )
    keep.add_argument(
        "--no-keep-reasoning", action="store_false", default=None,
        dest="keep_reasoning",
        help="Drop reasoning at end-of-turn, even if the profile enables it.",
    )
    parser.add_argument(
        "--network", action="append", default=[], metavar="BOOL",
        help="Network access for sandboxed tool commands, true or false "
        "(default: true). The last occurrence wins.",
    )
    parser.add_argument(
        "--gpu", action="append", default=[], metavar="BOOL",
        help="Expose the host's GPU device nodes to sandboxed tool commands, "
        "true or false (default: true). The last occurrence wins.",
    )
    return parser


def _flag_value(values: list[str]) -> bool:
    """Collapse a list of "true"/"false" flag values into one bool.

    Empty list (flag not given) defaults to True; the last occurrence wins;
    anything but "false"/"0"/"no" is treated as true."""
    if not values:
        return True
    return values[-1].strip().lower() not in {"false", "0", "no"}


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(argv)
    _run_chat(args)


def _prepare_git(workspace: Path, allow_init: bool) -> tuple[GitInfo | None, list[str]]:
    """Set up change tracking before the TUI boots.

    The agent always edits the workspace directly; git tracking (baseline
    tree snapshot) is what makes /diff and /undo work. Returns (git_info,
    startup notices); git_info is None when the workspace can't be tracked.
    """
    from . import gitwork, session_store

    async def _setup() -> tuple[GitInfo | None, list[str]]:
        try:
            info, notes = await gitwork.setup_tracking(
                workspace,
                session_store.new_session_name(),
                allow_init=allow_init,
            )
        except gitwork.GitWorktreeError as exc:
            return None, [
                f"warning: git change tracking unavailable ({exc}); the agent "
                f"still edits {workspace} directly, but /diff and /undo cannot "
                "revert file changes"
            ]
        return info, notes

    return asyncio.run(_setup())


def _run_chat(args: argparse.Namespace) -> None:
    loaded: LoadedConfig
    try:
        # One startup check: autogenerate the config on first run; on later
        # runs, note any optional values the file leaves at their defaults.
        config_notices = ensure_config(DEFAULT_CONFIG_PATH)
        loaded = load_config(DEFAULT_CONFIG_PATH, args.profile)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    profile = loaded.profile
    if args.keep_reasoning is not None:
        profile = profile.model_copy(update={"keep_reasoning": args.keep_reasoning})
    api_key = resolve_api_key(profile)

    from . import theme
    try:
        theme.set_theme(loaded.default_theme)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    workspace = args.workspace or Path.cwd()

    from . import session_store

    session_name = None if args.no_session else session_store.new_session_name()

    git_info: GitInfo | None = None
    git_notices: list[str] = config_notices
    git_info, git_git_notices = _prepare_git(workspace, allow_init=not args.no_init_repo)
    git_notices.extend(git_git_notices)

    allow_network = _flag_value(args.network)
    gpu_access = _flag_value(args.gpu)

    sandbox: SandboxConfig | None = None
    session: SandboxSession | None = None
    effective_workspace = workspace
    try:
        # The agent's workspace is exactly the directory the user pointed at,
        # even when it sits inside a bigger git repo. (gitwork still scopes
        # snapshots/diffs/reverts to the repo-relative subtree of that
        # directory — the rest of the repo just isn't mounted in-container.)
        sandbox = SandboxConfig(
            workspace=workspace,
            subtree="",
            allow_network=allow_network,
            gpu_access=gpu_access,
            external_refs=SandboxConfig.resolve_auto_refs(
                workspace, loaded.container.auto_include_refs,
            ),
            git_dir=git_info.git_dir if git_info is not None else None,
        )
        session = SandboxSession(sandbox)
        effective_workspace = workspace
        if gpu_access:
            from . import sandbox as sandbox_mod
            probe = asyncio.run(sandbox_mod.probe_gpu_access(sandbox))
            if probe is not None:
                level, message = probe
                git_notices.append(
                    message if level == "info" else f"warning: {message}"
                )
    except SandboxUnavailable as exc:
        print(f"warning: filesystem/bash tools disabled: {exc}", file=sys.stderr)
        sandbox = session = None

    from .tools import build_registry  # noqa: F401  (registry built by AgentApp)
    from .tui.app import AgentApp

    app = AgentApp(
        profile,
        api_key,
        workspace=effective_workspace,
        session_name=session_name,
        config_path=DEFAULT_CONFIG_PATH,
        profile_name=loaded.profile_name,
        sandbox=sandbox,
        sandbox_session=session,
        tool_output_limit=loaded.container.tool_output_limit,
        git_info=git_info,
        startup_notices=git_notices,
    )

    try:
        app.run()
    finally:
        if session is not None:
            asyncio.run(session.close())


if __name__ == "__main__":
    main()
