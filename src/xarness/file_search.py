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
from pathlib import Path

_MAX_WALK = 20_000
_MAX_RESULTS = 50
_GIT_TIMEOUT = 5.0


def _git_candidates(root: Path) -> list[str] | None:
    """Workspace-relative candidate paths from git, or None when the
    workspace is not tracked (or git is unavailable)."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard"],
            cwd=root, capture_output=True, text=True, timeout=_GIT_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.splitlines()


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


def _candidates(root: Path) -> list[str]:
    tracked = _git_candidates(root)
    if tracked is not None:
        return tracked
    return _walk_bounded(root)


def _best(paths: list[str], query: str) -> list[str]:
    """The _MAX_RESULTS shortest paths matching ``query`` (shortest first —
    a bare-bones but effective relevance order for @-mention)."""
    if not query:
        return heapq.nsmallest(_MAX_RESULTS, paths, key=len)
    q = query.lower()
    matching = (p for p in paths if q in p.lower())
    return heapq.nsmallest(_MAX_RESULTS, matching, key=len)


async def search_files(root: Path, query: str) -> list[str]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _best, _candidates(root), query)
