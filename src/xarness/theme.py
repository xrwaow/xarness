"""Theme palettes for the TUI.

Call ``set_theme(name)`` once at startup (before AgentApp is constructed) to
select a palette. Everything else reads ``theme.PALETTE`` / ``theme.CSS_VARIABLES``
/ ``theme.SHIMMER_*`` at call-time, so setting the theme before the app starts
is sufficient — no threading a theme object through every widget.
"""

from __future__ import annotations

from pygments.token import Token
from textual.highlight import HighlightTheme, TokenType

from .config import DEFAULT_THEME

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

PALETTE: dict[str, str] = {}
CSS_VARIABLES: dict[str, str] = {}
CURRENT_THEME: str = DEFAULT_THEME
SHIMMER_PROCESSING: tuple[str, str] = ("", "")
SHIMMER_THINKING: tuple[str, str] = ("", "")
SHIMMER_BAND_WIDTH: int = 6
SHIMMER_SPEED: float = 8.0


def rich_markdown_theme() -> "object":
    """A rich Theme mapping rich.markdown's ``markdown.*`` style names onto
    the active palette.

    The transcript presenter renders prose with rich.markdown, whose built-in
    styles hardcode cyan/magenta-on-black — they ignore both the terminal
    palette and the app theme. Pushing this theme onto the presenter's
    console makes rich resolve those names to palette colors instead.
    """
    import rich.theme as rich_theme

    p = PALETTE
    return rich_theme.Theme({
        "markdown.paragraph": "",
        "markdown.h1": f"bold {p['status']}",
        "markdown.h2": f"bold {p['status']}",
        "markdown.h3": f"bold {p['status']}",
        "markdown.h4": f"italic bold {p['status']}",
        "markdown.h5": f"italic {p['status']}",
        "markdown.h6": f"{p['muted']}",
        "markdown.h1.border": p["status"],
        "markdown.code": f"bold {blend_hex(p['warning'], p['bg'], 0.95)}",
        "markdown.code_block": p["text"],
        "markdown.item": "",
        "markdown.item.bullet": "",
        "markdown.item.number": "",
        "markdown.link": p["status"],
        "markdown.link_url": f"bold {p['status']}",
        "markdown.block_quote": f"italic {p['muted']}",
        "markdown.hr": p["border"],
        "markdown.s": "strike",
        "markdown.table.border": p["border"],
        "markdown.table.header": f"bold {p['text']}",
        "markdown.kbd": f"bold {p['accent2']}",
    })


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


class CodeHighlightTheme(HighlightTheme):
    """Syntax colors for code blocks, driven by the active xarness palette.

    Textual's built-in HighlightTheme styles code with its own theme's
    variables ($text-warning etc.), which always resolve to the dark
    "textual-dark" palette here — unreadable on light backgrounds. These
    styles use our $c-* variables instead, resolved at render time against
    the app's CSS variables, so code follows the active palette and live
    /theme switches.
    """

    STYLES: dict[TokenType, str] = {
        **dict.fromkeys((Token.Error, Token.Generic.Error, Token.Name.Constant), "$c-error"),
        **dict.fromkeys((Token.Keyword, Token.Keyword.Namespace, Token.Name.Builtin), "$c-accent"),
        **dict.fromkeys((Token.Keyword.Constant, Token.Keyword.Type, Token.Name.Attribute,
                         Token.Name.Class), "bold $c-accent2"),
        **dict.fromkeys((Token.Literal.Number, Token.Number), "$c-accent2"),
        **dict.fromkeys((Token.String, Token.Literal.String), "$c-success"),
        **dict.fromkeys((Token.Name, Token.Name.Variable, Token.Operator,
                         Token.Whitespace), "$c-text"),
        **dict.fromkeys((Token.Name.Decorator, Token.Name.Function,
                         Token.Name.Function.Magic), "$c-status"),
        Token.Comment: "italic $c-muted",
        Token.Generic.Heading: "bold $c-status",
        Token.Generic.Subheading: "$c-status",
        Token.Operator.Word: "bold $c-accent",
        Token.Literal.String.Backtick: "$c-muted",
        Token.Literal.String.Doc: "italic $c-success 80%",
        Token.Name.Builtin.Pseudo: "italic $c-accent",
        Token.Name.Tag: "bold $c-accent",
    }
