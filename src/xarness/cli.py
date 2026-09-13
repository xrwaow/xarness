"""Command-line entrypoint: ``xarness chat``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import DEFAULT_CONFIG_PATH, ConfigError, load_config, resolve_api_key
from .sandbox import SandboxConfig, SandboxUnavailable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xarness",
        description="Terminal chat client for OpenAI-compatible LLM APIs.",
    )
    parser.add_argument("--version", action="version", version=f"xarness {__version__}")
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
    chat.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Directory the read_file/write_file/run_bash tools are sandboxed to "
        "(default: current directory). Omit entirely to disable those tools.",
    )
    chat.add_argument(
        "--no-fs-tools",
        action="store_true",
        help="Disable read_file/write_file/run_bash even if bwrap is available.",
    )
    chat.add_argument(
        "--ref",
        action="append",
        default=[],
        metavar="ALIAS=PATH",
        help="Expose an external file/dir read-only at .refs/ALIAS inside the "
        "workspace. Repeatable.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(argv)

    if args.command is None:  # bare `xarness` behaves like `xarness chat`
        args = build_parser().parse_args(["chat", *argv])

    if args.command == "chat":
        _run_chat(args)


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

    sandbox: SandboxConfig | None = None
    if not args.no_fs_tools:
        workspace = args.workspace or Path.cwd()
        try:
            sandbox = SandboxConfig(workspace=workspace, external_refs=refs)
        except SandboxUnavailable as exc:
            print(
                f"warning: filesystem/bash tools disabled: {exc}",
                file=sys.stderr,
            )
            sandbox = None

    # Imported lazily so config errors print before any TUI initialization.
    from .tools import default_registry
    from .tui.app import AgentApp

    registry = default_registry(sandbox)
    AgentApp(profile, api_key, tool_registry=registry).run()


if __name__ == "__main__":
    main()
