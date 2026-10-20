"""Small shared helpers."""

from __future__ import annotations

from datetime import datetime, timezone


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
