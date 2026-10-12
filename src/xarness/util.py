"""Small shared helpers."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone


def format_duration(seconds: float) -> str:
    """Human duration, compact (no inner spaces): 5.3s · 42s · 2m15s · 1h04m."""
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def utcnow_iso() -> str:
    """Wall-clock UTC timestamp (ISO 8601)."""
    return datetime.now(timezone.utc).isoformat()


def utcnow_iso_offset(seconds: float) -> str:
    """Wall-clock UTC ISO stamp shifted by ``seconds``."""
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def mono_to_iso(monotonic: float | None) -> str | None:
    """A time.monotonic() reading converted to a wall-clock ISO stamp, or None.

    Monotonic readings have no absolute epoch; the conversion anchors on
    'now', which is accurate to within the seconds since the reading was
    taken — fine for second-resolution timestamps."""
    if monotonic is None:
        return None
    return utcnow_iso_offset(-(time.monotonic() - monotonic))
