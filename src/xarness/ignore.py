"""Glob-to-regex translation for file matching (the glob tool).

No ignore rules live here anymore: ignored paths (whatever the user's
``.gitignore``/``.git/info/exclude`` say) are shadowed out of the sandbox at
mount time — see ``sandbox._hidden_paths`` — so no tool-side filtering is
needed.
"""

from __future__ import annotations

import re
from functools import lru_cache


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
