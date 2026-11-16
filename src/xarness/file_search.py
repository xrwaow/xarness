"""Bounded filename and content search.

`search_files` backs @-mention autocomplete; `glob_files`/`grep_files` back
the read-only `glob`/`grep` tools, which plan mode gets on the host where
there is no shell. All three share one candidate index: when the workspace
is tracked by git it is ``git ls-files -co --exclude-standard`` — tracked
plus untracked-not-ignored files, so ignore rules apply for free and
node_modules/.venv-style trees never enter the set. Otherwise (no repo, or
git unavailable) a bounded walk caps scanned entries and skips
dot-directories, so a huge or accidentally-scoped workspace ($HOME) can't
freeze the UI. The content search additionally skips binaries and oversized
files, and all result counts are capped.
"""

from __future__ import annotations

import asyncio
import heapq
import os
import re
import stat
import subprocess
import time
from functools import lru_cache
from pathlib import Path

_MAX_WALK = 20_000
_MAX_RESULTS = 50
# Ranking is cheap, probing is not: over-fetch this many matches and drop the
# binaries after ranking, so a directory full of images cannot empty the list.
_FILTER_FETCH = _MAX_RESULTS * 3
_GIT_TIMEOUT = 5.0
_CANDIDATE_TTL = 2.0

# Caps for the grep/glob tools (a plan-mode exploration is a look, not an
# export; the footer tells the model how to narrow the next call).
MAX_GLOB_RESULTS = 200
MAX_GREP_MATCHES = 100
_GREP_FILES = 5_000
_GREP_FILE_BYTES = 2 << 20  # larger files are skipped, not read into memory
_GREP_LINE_CHARS = 300      # one long line cannot eat the output budget


def _git_candidates(root: Path) -> list[str] | None:
    """Workspace-relative candidate paths from git, or None when the
    workspace is not tracked (or git is unavailable).

    ``-z`` keeps non-ASCII names verbatim (without it git quotes and
    octal-escapes them, and the quoted string is not a path on disk).
    Tracked-but-deleted files are dropped: they are in the index but there
    is nothing to open."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard", "-z"],
            cwd=root, capture_output=True, timeout=_GIT_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    paths = proc.stdout.decode("utf-8", "surrogateescape").split("\0")
    return [path for path in paths if path and (root / path).exists()]


def _walk_bounded(root: Path) -> list[str]:
    results: list[str] = []
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            results.append(os.path.relpath(os.path.join(dirpath, name), root))
            scanned += 1
            if scanned >= _MAX_WALK:
                return results
    return results


@lru_cache(maxsize=32)
def _candidates(root: Path, _bucket: int) -> list[str]:
    """The workspace's candidate paths, cached for one TTL bucket."""
    tracked = _git_candidates(root)
    return tracked if tracked is not None else _walk_bounded(root)


def _cached_candidates(root: Path) -> list[str]:
    return _candidates(
        Path(root).resolve(), int(time.monotonic() // _CANDIDATE_TTL),
    )


def _best(paths: list[str], query: str, limit: int = _MAX_RESULTS) -> list[str]:
    """The ``limit`` shortest paths matching ``query`` (shortest first — a
    bare-bones but effective relevance order for @-mention)."""
    if not query:
        return heapq.nsmallest(limit, paths, key=len)
    q = query.lower()
    matching = (p for p in paths if q in p.lower())
    return heapq.nsmallest(limit, matching, key=len)


@lru_cache(maxsize=4096)
def _binary(path: str, _size: int, _mtime_ns: int) -> bool:
    """Content sniff, cached by the stat signature. Git's excludes no longer
    hide media files, so the ranked matches must be filtered here."""
    from .gitwork import probe_binary_path
    return probe_binary_path(Path(path))


def _visible(root: Path, matches: list[str]) -> list[str]:
    """``matches`` minus the binaries, capped at :data:`_MAX_RESULTS`."""
    out: list[str] = []
    for rel in matches:
        try:
            info = os.lstat(root / rel)
        except OSError:
            continue
        if not _binary(str(root / rel), info.st_size, info.st_mtime_ns):
            out.append(rel)
            if len(out) == _MAX_RESULTS:
                break
    return out


def _search(root: Path, query: str) -> list[str]:
    """The whole blocking pipeline (git ls-files + ranking + probing), run in
    one executor job — nothing here may touch the event loop."""
    ranked = _best(_cached_candidates(root), query, _FILTER_FETCH)
    return _visible(root, ranked)


async def search_files(root: Path, query: str) -> list[str]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _search, root, query)


# ----------------------------------------------------------------------
# glob / grep


@lru_cache(maxsize=256)
def _glob_regex(pattern: str) -> re.Pattern[str]:
    """Glob -> regex over '/'-separated paths: '**' crosses directories while
    '*', '?' and '[...]' stay inside one segment."""
    out: list[str] = []
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "*":
            if pattern.startswith("**", i):
                i += 2
                if pattern.startswith("/", i):
                    i += 1
                    out.append("(?:.*/)?")
                else:
                    out.append(".*")
                continue
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        elif char == "[" and (close := pattern.find("]", i + 1)) != -1:
            body = pattern[i + 1:close]
            out.append("[" + ("^" + body[1:] if body.startswith("!") else body) + "]")
            i = close
        else:
            out.append(re.escape(char))
        i += 1
    return re.compile("(?s:" + "".join(out) + ")")


def _match(path: str, pattern: str) -> bool:
    """A pattern without '/' matches a file name at any depth ('*.py'); one
    with '/' matches the whole path relative to the search root."""
    name = path if "/" in pattern else path.rsplit("/", 1)[-1]
    return _glob_regex(pattern).fullmatch(name) is not None


def _under(root: Path, base: str) -> list[tuple[str, str]]:
    """(workspace-relative, base-relative) files under ``base``. A base that
    names a file returns just it, even when gitignored: it was asked for."""
    if base != "." and (root / base).is_file():
        return [(base, Path(base).name)]
    prefix = "" if base == "." else base + "/"
    return [
        (rel, rel[len(prefix):]) for rel in _cached_candidates(root)
        if rel.startswith(prefix)
    ]


def _glob(root: Path, base: str, pattern: str, limit: int) -> tuple[list[str], int]:
    matched = sorted(rel for rel, sub in _under(root, base) if _match(sub, pattern))
    return matched[:limit], len(matched)


async def glob_files(
    root: Path, base: str, pattern: str, limit: int = MAX_GLOB_RESULTS,
) -> tuple[list[str], int]:
    """Sorted workspace-relative matches under ``base`` and the total count."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _glob, root, base, pattern, limit)


def _clip(line: str) -> str:
    return line if len(line) <= _GREP_LINE_CHARS else line[: _GREP_LINE_CHARS - 1] + "…"


def _grep(
    root: Path, base: str, pattern: str,
    file_glob: str | None, case_sensitive: bool, limit: int,
) -> tuple[list[str], bool]:
    """'path:line: text' matches and whether a cap stopped the scan early."""
    from .gitwork import TEXT_PROBE_BYTES, looks_binary

    regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
    matches: list[str] = []
    scanned = 0
    truncated = False
    for rel, sub in sorted(_under(root, base)):
        if file_glob is not None and not _match(sub, file_glob):
            continue
        if scanned >= _GREP_FILES:
            truncated = True
            break
        scanned += 1
        try:
            info = (root / rel).stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > _GREP_FILE_BYTES:
                continue
            data = (root / rel).read_bytes()
        except OSError:
            continue
        if looks_binary(data[:TEXT_PROBE_BYTES]):
            continue
        for lineno, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            if regex.search(line):
                matches.append(f"{rel}:{lineno}: {_clip(line)}")
                if len(matches) >= limit:
                    truncated = True
                    break
        if truncated:
            break
    return matches, truncated


async def grep_files(
    root: Path, base: str, pattern: str, file_glob: str | None = None,
    case_sensitive: bool = True, limit: int = MAX_GREP_MATCHES,
) -> tuple[list[str], bool]:
    """Async wrapper around :func:`_grep` (runs the scan off-loop)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _grep, root, base, pattern, file_glob, case_sensitive, limit,
    )
