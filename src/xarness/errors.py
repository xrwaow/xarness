"""Small error types shared across modules.

``SessionError`` used to live in ``session`` and ``RefError`` in
``session_store``; lower-level modules (``diffview``) need them too, and a
local import from ``session`` there would be a cycle.
"""

from __future__ import annotations


class SessionError(Exception):
    """A user-facing command failure (reported as ``ok=False``/HTTP 400)."""


class RefError(Exception):
    """An invalid external-reference path or operation."""
