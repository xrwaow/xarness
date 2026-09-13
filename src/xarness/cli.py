"""Command-line entrypoint: ``xarness chat`` / ``xarness sessions``."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from . import __version__
from .config import DEFAULT_CONFIG_PATH, ConfigError, load_config, resolve_api_key
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
        help="Directory the read_file/write_file/run_bash tools are sandboxed to "
        "(default: current directory).",
    )
    chat.add_argument(
        "--no-fs-tools", action="store_true",
        help="Disable read_file/write_file/run_bash even if bwrap is available.",
    )
    chat.add_argument(
        "--ref", action="append", default=[], metavar="ALIAS=PATH",
        help="Expose an external file/dir read-only at .refs/ALIAS inside the workspace. Repeatable.",
    )
    chat.add_argument(
        "--session", default=None,
        help="Name to save/resume this conversation under.",
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


def _run_chat(args: argparse.Namespace) -> None:
    try:
        profile = load_config(args.config, args.profile)
        api_key = resolve_api_key(profile)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    refs = _parse_refs(args.ref)
    workspace = args.workspace or Path.cwd()

    sandbox: SandboxConfig | None = None
    session: SandboxSession | None = None
    if not args.no_fs_tools:
        try:
            sandbox = SandboxConfig(workspace=workspace, external_refs=refs)
            session = SandboxSession(sandbox)
        except SandboxUnavailable as exc:
            print(f"warning: filesystem/bash tools disabled: {exc}", file=sys.stderr)
            sandbox = session = None

    from .tools import default_registry
    from .tui.app import AgentApp

    registry = default_registry(sandbox, session)

    conversation = None
    if args.session:
        from . import session_store
        if session_store.session_path(args.session).exists():
            conversation = session_store.load_session(args.session)

    app = AgentApp(
        profile,
        api_key,
        tool_registry=registry,
        workspace=workspace,
        session_name=args.session,
    )
    if conversation is not None:
        app.controller.conversation = conversation

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
            print(f"{name}\t{meta.get('updated_at', '?')}\t{meta.get('message_count', 0)} messages")
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
