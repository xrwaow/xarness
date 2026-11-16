"""Small shared helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # a leaf module: no runtime dependency on the models
    from .conversation import Usage


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


def format_elapsed(seconds: float) -> str:
    """Whole-second duration for running timers: 1s · 45s · 2m15s.

    A timer that counts seconds must show the same digits it has counted
    (no tenths that the user never saw tick)."""
    total = max(0, int(seconds))
    return f"{total}s" if total < 60 else format_duration(total)


def format_tokens(count: int) -> str:
    """Compact token count: 999 · 1.2k · 1.23M."""
    if count < 1_000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1_000:.1f}k"
    return f"{count / 1_000_000:.2f}M"


def format_usage(usage: Usage) -> str:
    """Compact ``~↑input ↓output`` spend (``~`` marks an estimate)."""
    return (f"{'~' if usage.approximate else ''}↑"
            f"{format_tokens(usage.input_tokens)} "
            f"↓{format_tokens(usage.output_tokens)}")


def utcnow_iso() -> str:
    """Wall-clock UTC timestamp (ISO 8601)."""
    return datetime.now(timezone.utc).isoformat()


def parse_iso(stamp: str | None) -> datetime | None:
    """Parse :func:`utcnow_iso` output; None when missing or malformed."""
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp)
    except ValueError:
        return None
