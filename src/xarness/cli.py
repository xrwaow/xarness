"""Command-line entrypoint for the xarness chat client.

``xarness`` opens a fresh chat (a draft — no session exists on the server
until the first message is sent). ``xarness --session <name>`` attaches to
an active session or loads a saved one; ``xarness --sessions`` prints the
active sessions and exits; ``xarness --kill`` closes every active session
(each stays saved on disk); ``xarness --server`` runs the host in the
foreground (persistent) and prints the stats.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from . import __version__
from .config import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    ensure_config,
    load_sectioned,
)

# Spellings accepted by every CLI boolean flag (--sandbox/--network/--gpu/
# --enable_tracking): argparse has no bool type that rejects junk, and this
# app wants `--network false`, not a bare `--network` that silently means on.
_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
_FALSE_WORDS = frozenset({"0", "false", "no", "off"})


def _bool_arg(value: str) -> bool:
    """argparse ``type`` for an explicit true/false (or 1/0, yes/no, on/off)."""
    word = value.strip().lower()
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    raise argparse.ArgumentTypeError(
        f"expected true or false, got {value!r}"
    )


class _SandboxArg(argparse.Action):
    """``--sandbox <name|true|false>``: pick a sandbox profile by name, or
    turn sandboxing on/off without naming one.

    Sets two namespace attributes: ``sandbox`` (the profile name, None when
    the value was a boolean — resolution then falls back to the preset or
    the workspace's last-used choice) and ``use_sandbox`` (the tri-state
    override, None for a named profile)."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        word = str(values).strip()
        try:
            enabled = _bool_arg(word)
        except argparse.ArgumentTypeError:
            if not word:
                parser.error(f"{option_string} needs a sandbox name or true/false")
            namespace.sandbox = word
            namespace.use_sandbox = None
            return
        namespace.sandbox = None
        namespace.use_sandbox = enabled


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
        "--sandbox", default=None, metavar="NAME|true|false", action=_SandboxArg,
        help="Sandbox to run the tools in: a named sandbox profile from the "
        "config's 'sandboxes' section, or true/false to keep the profile "
        "resolution (preset, then the workspace's most recent choice, then "
        "'default') and just turn sandboxing on/off. A bare --sandbox is an "
        "error: it needs a name or a boolean.",
    )
    parser.add_argument(
        "--preset", default=None,
        help="Named preset from the config's 'presets' section "
        "(default: the 'defaults' section's preset).",
    )
    parser.add_argument(
        "--no-init-repo",
        action="store_true",
        help="Don't track changes when the workspace has no git repo "
        "(default: xarness keeps a private snapshot store outside the "
        "workspace, so tracking works without setup and no .git is "
        "created; your files are never modified by this).",
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
        "--kill", action="store_true",
        help="Close every active session (turns are cancelled; each session "
        "stays saved on disk) and exit. Works even against a host speaking "
        "an older protocol version.",
    )
    parser.add_argument(
        "--server", action="store_true",
        help="Run the host in the foreground, persistent, printing the stats.",
    )
    parser.add_argument(
        "--network", type=_bool_arg, default=None, metavar="true|false",
        help="Force network access for sandboxed tool commands on or off "
        "(--network false = offline: UV_OFFLINE and no net namespace); not "
        "given = the sandbox profile's allow_network.",
    )
    parser.add_argument(
        "--gpu", type=_bool_arg, default=None, metavar="true|false",
        help="Force GPU device access for sandboxed tool commands on or off; "
        "not given = the sandbox profile's gpu_access.",
    )
    parser.add_argument(
        "--enable_tracking", type=_bool_arg, default=None, metavar="true|false",
        help="Force change tracking (checkpoints, diff bar, undo/revert) on "
        "or off; not given = the sandbox profile's enable_tracking.",
    )
    # Set by the --sandbox action only when the value is a boolean; the
    # default must exist even when the flag is absent.
    parser.set_defaults(use_sandbox=None)
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
    if args.kill:
        _kill_sessions()
        return
    _run_chat(args)


def _print_sessions() -> None:
    """Connect, fetch, print and close — all in one asyncio.run, so the
    httpx client never outlives its event loop."""
    from .clientlib import ClientError, XarnessClient

    async def _go() -> list:
        client = await XarnessClient.connect_unix()
        try:
            return await client.sessions(active=True)
        finally:
            await client.aclose()

    try:
        sessions = asyncio.run(_go())
    except ClientError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    if not sessions:
        print("no active sessions")
        return
    for s in sessions:
        state = f"{s.state}{'*' if s.clients_attached else ''}"
        workspace = s.workspace or "?"
        last = s.last_user_message_at or "?"
        print(f"{s.name}  {state:<10}  clients={s.clients_attached}  {workspace}  {last}")


def _kill_sessions() -> None:
    """Unload every active session, then exit.

    Connects with the protocol-version check disabled: a host from an older
    build is exactly the situation this flag exists for."""
    from .clientlib import ClientError, XarnessClient

    async def _go() -> list[str]:
        client = await XarnessClient.connect_unix(check_version=False)
        try:
            names = [s.name for s in await client.sessions(active=True)]
            closed: list[str] = []
            for name in names:
                try:
                    if await client.unload_session(name):
                        closed.append(name)
                except ClientError as exc:
                    # An older host has no /unload route (404). It cannot
                    # release them one by one, so stop it outright.
                    if exc.status != 404:
                        raise
                    from .host import SOCKET_PATH, stop_host
                    pid = (await client.health()).pid
                    await asyncio.to_thread(stop_host, pid, SOCKET_PATH)
                    return names
            return closed
        finally:
            await client.aclose()

    try:
        closed = asyncio.run(_go())
    except ClientError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    if not closed:
        print("no active sessions")
        return
    for name in closed:
        print(f"closed {name}")


def _run_chat(args: argparse.Namespace) -> None:
    from . import theme

    try:
        # First run: create the starter config before loading it. Its
        # notices are surfaced in the TUI instead of being silently lost.
        startup_notices = ensure_config(DEFAULT_CONFIG_PATH)
        sectioned = load_sectioned(DEFAULT_CONFIG_PATH)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    try:
        theme.set_theme(sectioned.theme)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    # Resolve client-side: the host may be a persistent daemon whose cwd is
    # not this shell's, so a relative path would land somewhere else there.
    workspace = (args.workspace or Path.cwd()).expanduser().resolve()

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
        use_sandbox=args.use_sandbox,
        preset=args.preset,
        network=args.network,
        gpu=args.gpu,
        enable_tracking=args.enable_tracking,
        no_init_repo=args.no_init_repo,
        no_session=args.no_session,
        startup_notices=startup_notices,
    )

    app.run()


if __name__ == "__main__":
    main()
