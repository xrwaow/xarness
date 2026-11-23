"""Small shared helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
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


class RangeError(ValueError):
    """A malformed ``range`` query value (the API answers 400)."""


def same_path(left: object, right: object) -> bool:
    """Whether two workspace spellings name the same directory.

    Saved metas may record a symlink or a relative ``--workspace``; the
    clients and the host must agree on which checkout that is. ``None`` on
    either side means "no filter" (match) and "no workspace" (no match).
    """
    if not right:
        return True
    if not left:
        return False
    try:
        return Path(str(left)).resolve() == Path(str(right)).resolve()
    except OSError:
        return False


def parse_range(spec: str | None, total: int) -> tuple[int, int]:
    """Resolve a Python-style ``range=[start:stop]`` against ``total``.

    ``None``/``""``/``"[:]"`` is the whole collection; a negative bound
    counts from the end (``[-10:]`` is the newest ten), exactly like a
    slice. Bounds clamp to ``[0, total]`` and an inverted window is empty,
    so a client can walk ``[-N:]``, then ``[start-N:start]``, … without
    booking offsets.
    """
    text = (spec or "").strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return 0, total
    if text.count(":") != 1:
        raise RangeError(
            f"bad range {spec!r}: expected [start:stop], e.g. [-10:]"
        )

    def bound(part: str, default: int) -> int:
        if not part.strip():
            return default
        try:
            value = int(part)
        except ValueError:
            raise RangeError(
                f"bad range {spec!r}: {part.strip()!r} is not an integer"
            ) from None
        return value + total if value < 0 else value

    start, stop = text.split(":")
    start = min(max(bound(start, 0), 0), total)
    return start, min(max(bound(stop, total), start), total)
