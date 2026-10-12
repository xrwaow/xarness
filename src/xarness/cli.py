"""Command-line entrypoint for the xarness chat client.

``xarness`` opens a fresh chat (a draft — no session exists on the server
until the first message is sent). ``xarness --session <name>`` attaches to
an active session or loads a saved one; ``xarness --sessions`` prints the
active sessions and exits; ``xarness --server`` runs the host in the
foreground (persistent) and prints the stats.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from . import __version__
from .config import DEFAULT_CONFIG_PATH, ConfigError, load_sectioned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xarness",
        description="Terminal chat client for OpenAI-compatible LLM APIs.",
    )
    parser.add_argument("--version", action="version", version=f"xarness {__version__}")
    parser.add_argument(
        "--workspace", type=Path, default=None,
        help="Directory the read_file/edit_file/run_bash tools are sandboxed to "
        "(default: current directory).",
    )
    parser.add_argument(
        "--model", "--profile", dest="model", default=None,
        help="Named model profile from the config's 'models' section "
        "(default: the preset's model, or the workspace's most recently "
        "used choice, or the default preset).",
    )
    parser.add_argument(
        "--sandbox", default=None,
        help="Named sandbox profile from the config's 'sandboxes' section "
        "(default: the preset's sandbox, or the workspace's most recently "
        "used choice, or 'default').",
    )
    parser.add_argument(
        "--preset", default=None,
        help="Named preset from the config's 'presets' section "
        "(default: the 'defaults' section's preset).",
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
    parser.add_argument(
        "--session", default=None, metavar="NAME",
        help="Attach to an active session or load a saved one.",
    )
    parser.add_argument(
        "--sessions", action="store_true",
        help="Print the active sessions (name, workspace, state, clients, "
        "last user message) and exit.",
    )
    parser.add_argument(
        "--server", action="store_true",
        help="Run the host in the foreground, persistent, printing the stats.",
    )
    parser.add_argument(
        "--network", action=argparse.BooleanOptionalAction, default=None,
        help="Force network access for sandboxed tool commands on/off "
        "(tri-state: --network = on, --no-network = off; not given = the "
        "sandbox profile's setting).",
    )
    parser.add_argument(
        "--gpu", action=argparse.BooleanOptionalAction, default=None,
        help="Force GPU device access for sandboxed tool commands on/off "
        "(tri-state, like --network).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(argv)
    if args.server:
        from .host import run_foreground
        run_foreground()
        return
    if args.sessions:
        _print_sessions()
        return
    _run_chat(args)


def _connect():
    """Connect to the host, auto-starting it (detached) if needed."""
    from .clientlib import ClientError, XarnessClient

    async def _go():
        return await XarnessClient.connect_unix()

    try:
        return asyncio.run(_go())
    except ClientError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


def _print_sessions() -> None:
    client = _connect()

    async def _go():
        try:
            return await client.sessions(active=True)
        finally:
            await client.aclose()

    sessions = asyncio.run(_go())
    if not sessions:
        print("no active sessions")
        return
    for s in sessions:
        state = f"{s.state}{'*' if s.clients_attached else ''}"
        workspace = s.workspace or "?"
        last = s.last_user_message_at or "?"
        print(f"{s.name}  {state:<10}  clients={s.clients_attached}  {workspace}  {last}")


def _run_chat(args: argparse.Namespace) -> None:
    from . import theme

    try:
        sectioned = load_sectioned(DEFAULT_CONFIG_PATH)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    try:
        theme.set_theme(sectioned.theme)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    workspace = args.workspace or Path.cwd()

    from .tui.app import AgentApp

    # client=None: the app connects on mount, inside Textual's event loop —
    # a client created under its own asyncio.run() would carry a closed loop.
    app = AgentApp(
        None,
        workspace=workspace,
        session_name=args.session,
        config_path=DEFAULT_CONFIG_PATH,
        model=args.model,
        sandbox=args.sandbox,
        preset=args.preset,
        network=args.network,
        gpu=args.gpu,
        no_init_repo=args.no_init_repo,
        no_session=args.no_session,
    )

    app.run()


if __name__ == "__main__":
    main()
