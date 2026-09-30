"""Saved prefix rules for the run_bash_host tool.

Prefix rules let the user permanently allow a command *shape* — "always
allow `cargo fetch`" — without allowing everything. They live in
``~/.config/xarness/host_bash.json`` (XDG_CONFIG_HOME respected), outside
the workspace on purpose: the sandboxed agent can write into the workspace,
so a rules file stored there could be edited by the tool it governs.

Prefixes are token tuples, compared against ``shlex.split`` of the command,
so ``uv run`` matches ``uv run pytest`` but not ``uv runaway``. Compound
commands (shells metacharacters) never match a rule — they always prompt.
"""

from __future__ import annotations

import json
import os
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Commands containing any of these skip the rules entirely and always
# prompt: a prefix match on the first tokens says nothing about what the
# rest of the line does once shells/redirects are involved.
_COMPOUND_CHARS = ";&|<>\n`"
_COMPOUND_SUBSTRINGS = "$("

HARNESS_DIR = "xarness"
RULES_FILENAME = "host_bash.json"


@dataclass(slots=True)
class HostBashRequest:
    """What the user sees on the run_bash_host approval prompt."""

    command: str
    reason: str
    cwd: str
    # Default edit text for an "always allow" rule (first two tokens).
    suggested_prefix: tuple[str, ...]
    # Compound commands (shells metacharacters) never offer prefix saving.
    is_compound: bool


@dataclass(slots=True)
class Decision:
    """The user's choice from the approval prompt.

    ``kind="prefix"`` saves ``prefix`` (already edited by the user);
    ``kind="deny"`` may carry a one-line ``deny_reason`` shown to the model.
    """

    kind: Literal["once", "session", "prefix", "deny"]
    prefix: tuple[str, ...] | None = None
    deny_reason: str = ""


def rules_path() -> Path:
    """Config file location; re-derived on every call so a changed
    XDG_CONFIG_HOME mid-process is respected (tests rely on this)."""
    root = os.environ.get("XDG_CONFIG_HOME")
    base = Path(root).expanduser() if root else Path("~/.config").expanduser()
    return base / HARNESS_DIR / RULES_FILENAME


def load_prefixes() -> tuple[tuple[str, ...], ...]:
    """Read the saved prefixes fresh each call (another process — or the
    user's editor — may have changed the file since the last tool call)."""
    path = rules_path()
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return ()
    raw = data.get("prefixes", []) if isinstance(data, dict) else []
    if not isinstance(raw, list):
        return ()
    out = []
    for entry in raw:
        if isinstance(entry, list) and entry and all(isinstance(t, str) for t in entry):
            out.append(tuple(entry))
    return tuple(out)


def add_prefix(prefix: tuple[str, ...]) -> None:
    """Persist one prefix rule, atomically (write temp file, rename)."""
    if not prefix:
        return
    existing = list(load_prefixes())
    if prefix not in existing:
        existing.append(prefix)
    path = rules_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".host_bash-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump({"prefixes": [list(p) for p in existing]}, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def is_compound(command: str) -> bool:
    """True when the command uses shells metacharacters: prefix rules are
    skipped and the user is always asked."""
    return (
        any(c in command for c in _COMPOUND_CHARS)
        or any(s in command for s in _COMPOUND_SUBSTRINGS)
    )


def tokens(command: str) -> tuple[str, ...] | None:
    """Tokenize like the rules do; None when shlex can't parse the line."""
    try:
        return tuple(shlex.split(command))
    except ValueError:
        return None


def prefix_matches(command: str, prefixes: tuple[tuple[str, ...], ...]) -> bool:
    """True when any saved prefix is a leading-token match of the command."""
    toks = tokens(command)
    if toks is None:
        return False
    return any(len(toks) >= len(p) and toks[: len(p)] == p for p in prefixes)
