"""The persistent multi-line chat input bar.

Detects '/' (slash commands, must start the input) and '@' (file mentions,
anywhere) and posts query messages so App can drive a SuggestionPopup. When
a popup is active, arrow/enter/tab/escape are redirected to popup navigation
instead of normal TextArea behavior.

Also owns the image-attachment machinery: pasted/mentioned images become
atomic ``[🖼 name W×H]`` tokens in the text, deletable as one unit.
"""

from __future__ import annotations

import re
from typing import ClassVar

from rich.style import Style
from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea
from textual.widgets.text_area import Selection, TextAreaTheme

from .. import images
from .. import theme
from ..images import ImageAttachment, ImageError, token_text

_SLASH_RE = re.compile(r"^/(\w*)$")
_AT_RE = re.compile(r"(?:^|\s)@(\S*)$")


class ChatInput(TextArea):
    """Persistent multi-line input bar: Enter submits, Shift+Enter newlines."""

    class ChatSubmitted(Message):
        def __init__(self, text: str, images: list[ImageAttachment] | None = None) -> None:
            self.text = text
            # Attached images in the order their tokens appear in the text.
            self.images = images or []
            super().__init__()

    class SlashQuery(Message):
        def __init__(self, query: str) -> None:
            self.query = query
            super().__init__()

    class AtQuery(Message):
        def __init__(self, query: str) -> None:
            self.query = query
            super().__init__()

    class PopupNav(Message):
        def __init__(self, direction: int) -> None:
            self.direction = direction
            super().__init__()

    class PopupConfirm(Message):
        pass

    class PopupDismiss(Message):
        pass

    class ModeToggle(Message):
        """Tab pressed with no popup active — toggle plan/write mode."""

    class ImageRefused(Message):
        """The active model does not accept image input."""

        def __init__(self, name: str) -> None:
            self.name = name
            super().__init__()

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "submit", "Send", priority=True),
        Binding("shift+enter", "newline", "Newline", priority=True, show=False),
        Binding("alt+enter", "newline", "Newline", priority=True, show=False),
        # priority beats Widget's inherited tab→focus_next; _on_key still
        # claims tab first while a popup is open (popup confirm).
        Binding("tab", "toggle_mode", "Mode", priority=True, show=False),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.popup_active: str | None = None
        self.supports_vision = True
        # Attached images, kept in token order (the editor's image segments).
        # Synced against the document on every edit: deleting a token drops
        # its attachment.
        self.attachments: list[ImageAttachment] = []

    def on_mount(self) -> None:
        self.apply_input_theme()
        self._refresh_token_highlights()

    def _refresh_token_highlights(self) -> None:
        """Paint image tokens with the ``image-token`` style.

        TextArea draws per-range highlights from its internal map (normally
        fed by tree-sitter); with no language set we own that map. Ranges
        are byte offsets into the line (the renderer maps them back to
        codepoints), which matters because 🖼 is multi-byte. Tokens never
        span lines, so per-line scanning is enough."""
        self._highlights.clear()
        for row in range(self.document.line_count):
            line = self.document.get_line(row)
            for match in images.IMAGE_TOKEN_RE.finditer(line):
                self._highlights[row].append(
                    (
                        len(line[: match.start()].encode("utf-8")),
                        len(line[: match.end()].encode("utf-8")),
                        "image-token",
                    )
                )

    def apply_input_theme(self) -> None:
        """(Re)build the TextArea theme from the current palette.

        Called on mount and again when /theme switches palettes live; the
        cursor and selection colors come from the theme config.
        """
        base = TextAreaTheme.get_builtin_theme("css")
        no_line_highlight = TextAreaTheme(
            name="xarness-input",
            base_style=base.base_style,
            gutter_style=base.gutter_style,
            cursor_style=Style(bgcolor=theme.PALETTE["cursor"], color=theme.PALETTE["bg"]),
            cursor_line_style=Style(),
            cursor_line_gutter_style=base.cursor_line_gutter_style,
            bracket_matching_style=base.bracket_matching_style,
            selection_style=Style(bgcolor=theme.PALETTE["highlight"], color=theme.PALETTE["bg"]),
            syntax_styles={
                **base.syntax_styles,
                # Image attachment tokens ([🖼 name W×H]) get a dimmed chip
                # look, matching .user-image-chip in the transcript.
                "image-token": Style(bgcolor=theme.PALETTE["border"], color=theme.PALETTE["muted"]),
            },
        )
        self.register_theme(no_line_highlight)
        # Assigning the same name wouldn't re-run the theme watcher, so bounce
        # through the builtin to force the new colors to be applied.
        self.theme = "css"
        self.theme = "xarness-input"

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._check_triggers()
        self._sync_attachments()
        self._refresh_token_highlights()

    def _check_triggers(self) -> None:
        text = self.text
        slash_match = _SLASH_RE.match(text) if "\n" not in text else None
        if slash_match:
            self.popup_active = "slash"
            self.post_message(self.SlashQuery(slash_match.group(1)))
            return

        row, col = self.cursor_location
        line = self.document.get_line(row)
        prefix = line[:col]
        at_match = _AT_RE.search(prefix)
        if at_match:
            self.popup_active = "at"
            self.post_message(self.AtQuery(at_match.group(1)))
            return

        if self.popup_active is not None:
            self.popup_active = None
            self.post_message(self.PopupDismiss())

    def _on_key(self, event: events.Key) -> None:
        if self.popup_active is not None and event.key in ("up", "down", "enter", "tab", "escape"):
            event.prevent_default()
            event.stop()
            if event.key == "up":
                self.post_message(self.PopupNav(-1))
            elif event.key == "down":
                self.post_message(self.PopupNav(1))
            elif event.key in ("enter", "tab"):
                self.post_message(self.PopupConfirm())
            elif event.key == "escape":
                self.popup_active = None
                self.post_message(self.PopupDismiss())
            return
        # NOTE: no super()._on_key() here — Textual already dispatches
        # TextArea._on_key via the MRO; calling it manually creates an
        # un-awaited coroutine (RuntimeWarning) with no extra effect.

    def action_newline(self) -> None:
        self.insert("\n")

    def action_toggle_mode(self) -> None:
        self.post_message(self.ModeToggle())

    def _on_paste(self, event: events.Paste) -> None:
        """A pasted path or ``file://`` URI naming an image attaches it.

        Terminals (kitty, iTerm2, ...) paste a clipboard image as a file
        URI or path; the image becomes a ``[🖼 name W×H]`` token in the text
        (deletable as one unit). Raw image bytes cannot cross a plain
        terminal, so there is nothing to intercept when the paste event
        never fires.
        """
        if "\n" in event.text:
            return
        path = images.parse_image_path(event.text)
        if path is None or not images.is_image_path(path) or not path.is_file():
            return
        try:
            attachment = images.load_image(path)
        except ImageError:
            return  # not a decodable image: paste the text as-is
        event.prevent_default()
        event.stop()
        self.insert_image(attachment)

    def insert_image(self, attachment: ImageAttachment) -> bool:
        """Attach an image; false (with a notice) when vision is unsupported."""
        if not self.supports_vision:
            self.post_message(self.ImageRefused(attachment.name))
            return False
        if attachment not in self.attachments:
            self.attachments.append(attachment)
        self.insert(token_text(attachment))
        return True

    def restore(self, text: str, images: list[ImageAttachment] | None = None) -> None:
        """Put an undone message (text and attachments) back in the input."""
        self.load_text(text)
        self.attachments = list(images or [])
        self._refresh_token_highlights()

    def _sync_attachments(self) -> None:
        """Drop attachments whose token is no longer in the document."""
        text = self.text
        self.attachments = [
            attachment for attachment in self.attachments
            if token_text(attachment) in text
        ]

    def _ordered_attachments(self) -> list[ImageAttachment]:
        """Attachments in the order their tokens appear in the text."""
        by_token: dict[str, ImageAttachment] = {}
        for attachment in self.attachments:
            by_token.setdefault(token_text(attachment), attachment)
        return [
            by_token[token] for token in images.IMAGE_TOKEN_RE.findall(self.text)
            if token in by_token
        ]

    # -- atomic image tokens -------------------------------------------

    def _delete_token_at_cursor(self, direction: int) -> bool:
        """Backspace/Delete at an image token's edge: remove the whole token
        (and its attachment) as if it were one character."""
        if self.selected_text:
            return False
        offset = self.document.get_index_from_location(self.cursor_location)
        for match in images.IMAGE_TOKEN_RE.finditer(self.text):
            start, end = match.span()
            hit = start < offset <= end if direction < 0 else start <= offset < end
            if not hit:
                continue
            start_loc = self.document.get_location_from_index(start)
            end_loc = self.document.get_location_from_index(end)
            self.delete(start_loc, end_loc, maintain_selection_offset=False)
            self.selection = Selection(start_loc, start_loc)
            self._sync_attachments()
            return True
        return False

    def action_delete_left(self) -> None:
        if self._delete_token_at_cursor(-1):
            return
        super().action_delete_left()

    def action_delete_right(self) -> None:
        if self._delete_token_at_cursor(1):
            return
        super().action_delete_right()

    def action_submit(self) -> None:
        if self.popup_active is not None:
            self.post_message(self.PopupConfirm())
            return
        text = self.text.strip()
        if not text:
            # Empty Enter is still an event: the app uses it to "send now"
            # a message that is queued behind a running turn.
            self.post_message(self.ChatSubmitted(""))
            return
        images_out = self._ordered_attachments()
        self.load_text("")
        self.attachments = []
        self.post_message(self.ChatSubmitted(text, images_out))

    def insert_mention(self, value: str) -> None:
        """Complete an @-mention. Image files are attached instead: their
        token replaces the typed ``@query``."""
        attachment = self._try_load_mention_image(value)
        if attachment is not None:
            self.popup_active = None
            self.insert_image(attachment)
            return
        row, col = self.cursor_location
        line = self.document.get_line(row)
        prefix = line[:col]
        at_col = prefix.rfind("@")
        if at_col == -1:
            return
        self.replace(f"@{value} ", (row, at_col), (row, col))
        self.popup_active = None

    def _try_load_mention_image(self, value: str) -> ImageAttachment | None:
        path = images.parse_image_path(value)
        if path is None or not images.is_image_path(path) or not path.is_file():
            return None
        try:
            attachment = images.load_image(path)
        except ImageError:
            return None
        if attachment in self.attachments:
            return None  # already attached: let the plain @-mention go out
        return attachment
