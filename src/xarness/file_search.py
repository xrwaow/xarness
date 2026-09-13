"""Fast, bounded filename search for @-mention autocomplete.

Caps scanned entries and skips heavy directories so a huge repo (or an
accidentally-scoped workspace like $HOME) can't freeze the UI.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".cache"}
_MAX_SCAN = 20_000
_MAX_RESULTS = 50


def _walk_bounded(root: Path) -> list[str]:
    results: list[str] = []
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            results.append(rel)
            scanned += 1
            if scanned >= _MAX_SCAN:
                return results
    return results


async def search_files(root: Path, query: str) -> list[str]:
    loop = asyncio.get_running_loop()
    all_paths = await loop.run_in_executor(None, _walk_bounded, root)
    if not query:
        return sorted(all_paths, key=len)[:_MAX_RESULTS]
    q = query.lower()
    matches = [p for p in all_paths if q in p.lower()]
    matches.sort(key=len)
    return matches[:_MAX_RESULTS]
