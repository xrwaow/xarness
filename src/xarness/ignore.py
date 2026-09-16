"""Shared ignore rules for file exploration (ls/glob/grep, @-mention
autocomplete).

This is deliberately NOT a .gitignore parser. Ignore handling is layered:
git's own tools (``ls-files --exclude-standard``, ``check-ignore``, ``rg``)
apply whatever the worktree's .gitignore says; the defaults here catch the
noise repos commonly don't bother ignoring, independent of any .gitignore.
"""

from __future__ import annotations

import fnmatch
import re
from functools import lru_cache

DEFAULT_IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
                        "dist", "build", ".cache", ".pytest_cache", ".mypy_cache",
                        ".ruff_cache", ".idea", ".vscode"}
DEFAULT_IGNORE_FILE_GLOBS = ("*.pyc", "*.egg-info", ".DS_Store")


def is_ignored(rel_path: str) -> bool:
    """True if any path component is a default-ignored directory name or
    matches a default-ignored file glob (e.g. ``pkg.egg-info`` dirs)."""
    for part in rel_path.split("/"):
        if part in DEFAULT_IGNORE_DIRS:
            return True
        if any(fnmatch.fnmatchcase(part, glob) for glob in DEFAULT_IGNORE_FILE_GLOBS):
            return True
    return False


@lru_cache(maxsize=256)
def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a glob into an anchored regex (match with ``fullmatch``).

    ``**/`` matches zero or more whole directory segments, a bare ``**``
    matches anything, ``*`` anything but ``/``, ``?`` one non-``/`` char.
    Hand-rolled because ``pathlib.PurePath.match``'s ``**`` support is only
    solid from 3.13+ and xarness targets 3.11.
    """
    out: list[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*":
            if pattern[i:i + 3] == "**/":
                out.append("(?:[^/]+/)*")
                i += 3
            elif pattern[i + 1:i + 2] == "*":
                out.append(".*")
                i += 2
            else:
                out.append("[^/]*")
                i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("".join(out))
