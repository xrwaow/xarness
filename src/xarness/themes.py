"""Pure theme palette data (no Textual / Pygments imports).

Kept separate from :mod:`xarness.theme` so the daemon can advertise the
theme names without pulling the TUI stack into its process.
"""

from __future__ import annotations

# Semantic keys every theme must define:
#   bg, surface, border, text, muted, user, reasoning, accent, accent2, status,
#   error, success, warning, highlight, cursor,
#   diff_add, diff_add_bg, diff_del, diff_del_bg, diff_hunk, diff_meta
# The diff_* keys style unified diffs: added/removed line foreground plus a
# full-line background tint (GitHub-style), hunk headers, and gutter/metadata text.
THEMES: dict[str, dict[str, str]] = {
    "ayu-darker": {
        "bg": "#121212",
        "surface": "#1d1f21",
        "border": "#2d2f34",
        "text": "#bfbdb6",
        "muted": "#8a8986",
        "user": "#aad84c",
        "reasoning": "#8a8986",
        "accent": "#d2a6fe",
        "accent2": "#feb454",
        "status": "#5ac1fe",
        "error": "#ef7177",
        "success": "#aad84c",
        "warning": "#feb454",
        "highlight": "#bfbdb6",
        "cursor": "#d2a6fe",
        "diff_add": "#aad84c",
        "diff_add_bg": "#182e1c",
        "diff_del": "#ef7177",
        "diff_del_bg": "#2f1a1d",
        "diff_hunk": "#5ac1fe",
        "diff_meta": "#8a8986",
    },
    "one-light": {
        "bg": "#fafafa",
        "surface": "#eaeaeb",
        "border": "#d3d3d3",
        "text": "#000000",
        "muted": "#a0a1a7",
        "user": "#50a14f",
        "reasoning": "#a0a1a7",
        "accent": "#7c4dff",
        "accent2": "#c18401",
        "status": "#7c4dff",
        "error": "#e45649",
        "success": "#50a14f",
        "warning": "#c18401",
        "highlight": "#000000",
        "cursor": "#000000",
        "diff_add": "#1a7f37",
        "diff_add_bg": "#e6ffed",
        "diff_del": "#cf222e",
        "diff_del_bg": "#ffebe9",
        "diff_hunk": "#7c4dff",
        "diff_meta": "#a0a1a7",
    },
}


def blend_hex(fg: str, bg: str, alpha: float) -> str:
    """Blend two #rrggbb colors (alpha = weight of fg)."""
    f, b = fg.lstrip("#"), bg.lstrip("#")
    return "#" + "".join(
        f"{round(int(f[i:i+2], 16) * alpha + int(b[i:i+2], 16) * (1 - alpha)):02x}"
        for i in (0, 2, 4)
    )
