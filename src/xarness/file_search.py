"""Fast, bounded filename search for @-mention autocomplete.

When the workspace is tracked by git, candidates come from
``git ls-files -co --exclude-standard`` — tracked plus untracked-not-ignored
files, so ignore rules apply for free and node_modules/.venv-style trees
never enter the candidate set. Otherwise (no repo, or git unavailable) it
falls back to a bounded walk that caps scanned entries and skips
dot-directories, so a huge or accidentally-scoped workspace ($HOME) can't
freeze the UI.
"""

from __future__ import annotations

import asyncio
import heapq
import os
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
