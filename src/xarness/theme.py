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
from .themes import THEMES, blend_hex as _blend_hex

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
        "markdown.code": f"bold {_blend_hex(p['warning'], p['bg'], 0.95)}",
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
