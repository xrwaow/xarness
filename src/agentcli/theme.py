"""Theme palettes for the TUI.

Call ``set_theme(name)`` once at startup (before AgentApp is constructed) to
select a palette. Everything else reads ``theme.PALETTE`` / ``theme.CSS_VARIABLES``
/ ``theme.SHIMMER_*`` at call-time, so setting the theme before the app starts
is sufficient — no threading a theme object through every widget.
"""

from __future__ import annotations

# Semantic keys every theme must define:
#   bg, surface, border, text, muted, user, reasoning, accent, accent2, status, error
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
    },
    "one-light": {
        "bg": "#fafafa",
        "surface": "#eaeaeb",
        "border": "#d3d3d3",
        "text": "#383a42",
        "muted": "#a0a1a7",
        "user": "#50a14f",
        "reasoning": "#a0a1a7",
        "accent": "#4078f2",
        "accent2": "#c18401",
        "status": "#4078f2",
        "error": "#e45649",
    },
}

DEFAULT_THEME = "ayu-darker"

PALETTE: dict[str, str] = {}
CSS_VARIABLES: dict[str, str] = {}
SHIMMER_PROCESSING: tuple[str, str] = ("", "")
SHIMMER_THINKING: tuple[str, str] = ("", "")

SHIMMER_FPS: int = 20
SHIMMER_BAND_WIDTH: int = 6
SHIMMER_SPEED: float = 8.0


def set_theme(name: str) -> None:
    """Select the active palette. Call once, before AgentApp() is built."""
    global PALETTE, CSS_VARIABLES, SHIMMER_PROCESSING, SHIMMER_THINKING
    if name not in THEMES:
        raise ValueError(f"unknown theme '{name}', options: {list(THEMES)}")
    PALETTE = THEMES[name]
    CSS_VARIABLES = {f"c-{k}": v for k, v in PALETTE.items()}
    SHIMMER_PROCESSING = (PALETTE["muted"], PALETTE["text"])
    SHIMMER_THINKING = (PALETTE["muted"], PALETTE["surface"])


set_theme(DEFAULT_THEME)
