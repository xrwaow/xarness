"""Command-line entrypoint: ``xarness chat`` / ``xarness sessions``."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from . import __version__
from .config import (
    DEFAULT_CONFIG_PATH, ConfigError, list_profile_names, load_config, resolve_api_key, resolve_profile_name
)
from .gitwork import GitInfo
from .sandbox import SandboxConfig, SandboxSession, SandboxUnavailable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xarness",
        description="Terminal chat client for OpenAI-compatible LLM APIs.",
    )
    parser.add_argument("--version", action="version", version=f"xarness {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=False)

    chat = subparsers.add_parser("chat", help="Start the interactive chat TUI")
    chat.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help=f"Path to the YAML config file (default: {DEFAULT_CONFIG_PATH})",
    )
    chat.add_argument(
        "--profile", default=None,
        help="Named profile from the config's 'profiles:' section "
        "(default: default_profile, or the only profile if just one is defined)",
    )
    chat.add_argument(
        "--workspace", type=Path, default=None,
        help="Directory the read_file/edit_file/run_bash tools are sandboxed to "
        "(default: current directory).",
    )
    chat.add_argument(
        "--no-fs-tools", action="store_true",
        help="Disable read_file/edit_file/run_bash even if bwrap is available.",
    )
    chat.add_argument(
        "--ref", action="append", default=[], metavar="ALIAS=PATH",
        help="Expose an external file/dir read-only at .refs/ALIAS inside the workspace. Repeatable.",
    )
    chat.add_argument(
        "--session", default=None,
        help="Name to save/resume this conversation under. "
        "(Default: an auto-generated timestamped name — sessions autosave.)",
    )
    chat.add_argument(
        "--no-init-repo",
        action="store_true",
        help="Don't auto-initialize a git repo when the workspace has none "
        "(default: xarness runs `git init` itself so change tracking works "
        "without setup; your files are never modified by this).",
    )
    chat.add_argument(
        "--no-session", action="store_true",
        help="Don't save this conversation as a resumable session.",
    )

    sessions = subparsers.add_parser("sessions", help="List or delete saved chat sessions")
    sessions.add_argument("action", choices=["list", "delete"], nargs="?", default="list")
    sessions.add_argument("name", nargs="?", default=None, help="Session name (for delete)")

    return parser


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(argv)

    if args.command is None:
        args = build_parser().parse_args(["chat", *argv])

    if args.command == "chat":
        _run_chat(args)
    elif args.command == "sessions":
        _run_sessions(args)


def _parse_refs(raw_refs: list[str]) -> dict[str, Path]:
    refs: dict[str, Path] = {}
    for entry in raw_refs:
        if "=" not in entry:
            print(f"error: --ref must be ALIAS=PATH, got {entry!r}", file=sys.stderr)
            raise SystemExit(2)
        alias, _, raw_path = entry.partition("=")
        alias = alias.strip()
        path = Path(raw_path.strip()).expanduser()
        if not alias:
            print(f"error: empty alias in --ref {entry!r}", file=sys.stderr)
            raise SystemExit(2)
        if not path.exists():
            print(f"error: --ref path does not exist: {path}", file=sys.stderr)
            raise SystemExit(2)
        refs[alias] = path
    return refs


def _prepare_git(
    args: argparse.Namespace, workspace: Path, session_name: str | None
) -> tuple[GitInfo | None, list[str]]:
    """Set up change tracking before the TUI boots.

    The agent always edits the workspace directly; git tracking (baseline
    tree snapshot) is what makes /diff and /undo work. Returns (git_info,
    startup notices); git_info is None when the workspace can't be tracked.
    """
    from . import gitwork, session_store

    async def _setup() -> tuple[GitInfo | None, list[str]]:
        notices: list[str] = []

        # Resume: reconnect to the persisted tracking state if there is one.
        if args.session and session_store.session_path(args.session).exists():
            block = session_store.load_git_block(args.session)
            if block is not None:
                try:
                    return gitwork.GitInfo.from_block(block), notices
                except (KeyError, TypeError, ValueError):
                    # Session predates direct-write tracking (old worktree
                    # block). Set up tracking now — the session shouldn't be
                    # penalized forever for when it was first created.
                    notices.append(
                        "note: this session used the old worktree isolation; "
                        "switched to direct edits with fresh change tracking"
                    )
        return await _fresh_tracking(notices)

    async def _fresh_tracking(notices: list[str]) -> tuple[GitInfo | None, list[str]]:
        try:
            info, notes = await gitwork.setup_tracking(
                workspace,
                session_name or session_store.new_session_name(),
                allow_init=not args.no_init_repo,
            )
        except gitwork.GitWorktreeError as exc:
            notices.append(
                f"warning: git change tracking unavailable ({exc}); the agent "
                f"still edits {workspace} directly, but /diff and /undo cannot "
                "revert file changes"
            )
            return None, notices
        return info, notices + notes

    return asyncio.run(_setup())


def _run_chat(args: argparse.Namespace) -> None:
    try:
        profile = load_config(args.config, args.profile)
        api_key = resolve_api_key(profile)
        profile_names = list_profile_names(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    refs = _parse_refs(args.ref)
    workspace = args.workspace or Path.cwd()

    from . import session_store

    session_name = None if args.no_session else (args.session or session_store.new_session_name())

    fs_tools_enabled = not args.no_fs_tools
    git_info: GitInfo | None = None
    git_notices: list[str] = []
    if fs_tools_enabled:
        git_info, git_notices = _prepare_git(args, workspace, session_name)

    sandbox: SandboxConfig | None = None
    session: SandboxSession | None = None
    if fs_tools_enabled:
        effective_workspace = git_info.agent_workspace if git_info is not None else workspace
        try:
            sandbox = SandboxConfig(
                workspace=git_info.workspace if git_info is not None else workspace,
                subtree=git_info.subtree if git_info is not None else "",
                external_refs=refs,
                git_dir=git_info.git_dir if git_info is not None else None,
            )
            session = SandboxSession(sandbox)
        except SandboxUnavailable as exc:
            print(f"warning: filesystem/bash tools disabled: {exc}", file=sys.stderr)
            sandbox = session = None
    else:
        effective_workspace = workspace

    from .tools import build_registry  # noqa: F401  (registry built by AgentApp)
    from .tui.app import AgentApp

    conversation = None
    if args.session:
        if session_store.session_path(args.session).exists():
            conversation = session_store.load_session(args.session)

    try:
        profile_name = resolve_profile_name(args.config, args.profile)
    except ConfigError:
        profile_name = None

    app = AgentApp(
        profile,
        api_key,
        workspace=effective_workspace,
        session_name=session_name,
        config_path=args.config,
        profile_name=profile_name,
        sandbox=sandbox,
        sandbox_session=session,
        git_info=git_info,
        startup_notices=git_notices,
    )
    if conversation is not None:
        app.controller.conversation = conversation
        app.ensure_system_message()

    try:
        app.run()
    finally:
        if session is not None:
            asyncio.run(session.close())


def _run_sessions(args: argparse.Namespace) -> None:
    from . import session_store

    if args.action == "list":
        names = session_store.list_sessions()
        if not names:
            print("no saved sessions")
            return
        for name in names:
            meta = session_store.session_meta(name)
            workspace = meta.get("workspace") or "?"
            print(
                f"{name}\t{meta.get('updated_at', '?')}\t"
                f"{meta.get('message_count', 0)} messages\t{workspace}"
            )
    elif args.action == "delete":
        if not args.name:
            print("error: 'delete' requires a session name", file=sys.stderr)
            raise SystemExit(2)
        if not session_store.delete_session(args.name):
            print(f"error: no session named {args.name!r}", file=sys.stderr)
            raise SystemExit(1)
        print(f"deleted {args.name!r}")


if __name__ == "__main__":
    main()
