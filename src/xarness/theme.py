"""Theme palettes for the TUI.

Call ``set_theme(name)`` once at startup (before AgentApp is constructed) to
select a palette. Everything else reads ``theme.PALETTE`` / ``theme.CSS_VARIABLES``
/ ``theme.SHIMMER_*`` at call-time, so setting the theme before the app starts
is sufficient — no threading a theme object through every widget.
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
        "bg": "#121212",              # background
        "surface": "#1d1f21",         # status_bar.background / elevated_surface.background
        "border": "#2d2f34",          # border.variant
        "text": "#bfbdb6",            # text
        "muted": "#8a8986",           # text.muted
        "user": "#aad84c",            # terminal.ansi.green / success
        "reasoning": "#8a8986",       # text.muted (was wrongly accent2-tinted before)
        "accent": "#d2a6fe",          # terminal.ansi.bright_magenta family / punctuation.special
        "accent2": "#feb454",         # terminal.ansi.yellow / warning
        "status": "#5ac1fe",          # text.accent
        "error": "#ef7177",           # error
        "success": "#aad84c",         # green status dot / tool call succeeded
        "warning": "#feb454",         # amber status dot / call in flight
        "highlight": "#bfbdb6",       # generic UI emphasis (hover, focus)
        "cursor": "#d2a6fe",          # input caret block
        "diff_add": "#aad84c",        # added line text
        "diff_add_bg": "#182e1c",     # added line background tint
        "diff_del": "#ef7177",        # removed line text
        "diff_del_bg": "#2f1a1d",     # removed line background tint
        "diff_hunk": "#5ac1fe",       # @@ hunk headers
        "diff_meta": "#8a8986",       # line-number gutter / metadata
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

DEFAULT_THEME = "ayu-darker"

PALETTE: dict[str, str] = {}
CSS_VARIABLES: dict[str, str] = {}
CURRENT_THEME: str = DEFAULT_THEME
SHIMMER_PROCESSING: tuple[str, str] = ("", "")
SHIMMER_THINKING: tuple[str, str] = ("", "")

SHIMMER_FPS: int = 20
SHIMMER_BAND_WIDTH: int = 6
SHIMMER_SPEED: float = 8.0


def set_theme(name: str) -> None:
    """Select the active palette. Call once, before AgentApp() is built
    (or from /theme, followed by App.refresh_css())."""
    global PALETTE, CSS_VARIABLES, CURRENT_THEME, SHIMMER_PROCESSING, SHIMMER_THINKING
    if name not in THEMES:
        raise ValueError(f"unknown theme '{name}', options: {list(THEMES)}")
    PALETTE = THEMES[name]
    CURRENT_THEME = name
    CSS_VARIABLES = {f"c-{k}": v for k, v in PALETTE.items()}
    # Textual's built-in themes ship blue scrollbars; re-point the scrollbar
    # variables at the palette so every scrollbar follows the active theme.
    # The track uses `surface` (not `bg`) so it stays visible against the
    # pane background — an invisible track makes the bar read as broken
    # floating slivers.
    CSS_VARIABLES.update({
        "scrollbar": PALETTE["border"],
        "scrollbar-hover": PALETTE["muted"],
        "scrollbar-active": PALETTE["muted"],
        "scrollbar-background": PALETTE["surface"],
        "scrollbar-background-hover": PALETTE["surface"],
        "scrollbar-background-active": PALETTE["surface"],
        "scrollbar-corner-color": PALETTE["surface"],
    })
    SHIMMER_PROCESSING = (PALETTE["muted"], PALETTE["text"])
    SHIMMER_THINKING = (PALETTE["muted"], PALETTE["surface"])


set_theme(DEFAULT_THEME)
