"""Command-line entrypoint: ``agentcli chat``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import DEFAULT_CONFIG_PATH, ConfigError, load_config, resolve_api_key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentcli",
        description="Terminal chat client for OpenAI-compatible LLM APIs.",
    )
    parser.add_argument("--version", action="version", version=f"agentcli {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=False)

    chat = subparsers.add_parser("chat", help="Start the interactive chat TUI")
    chat.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the YAML config file (default: {DEFAULT_CONFIG_PATH})",
    )
    chat.add_argument(
        "--profile",
        default=None,
        help="Named profile from the config's 'profiles:' section "
        "(default: default_profile, or the only profile if just one is defined)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(argv)

    if args.command is None:  # bare `agentcli` behaves like `agentcli chat`
        args = build_parser().parse_args(["chat", *argv])

    if args.command == "chat":
        _run_chat(args)


def _run_chat(args: argparse.Namespace) -> None:
    try:
        profile = load_config(args.config, args.profile)
        api_key = resolve_api_key(profile)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    # Imported lazily so config errors print before any TUI initialization.
    from .tui.app import AgentApp

    AgentApp(profile, api_key).run()


if __name__ == "__main__":
    main()
