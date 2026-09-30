"""Inline approval bar for run_bash_host: one-shot commands on the host.

Shown above the chat input (like ``AskBar``) while a host command waits for
approval: the exact command, the model's reason, the working directory, and
clickable buttons — allow once, this session, a saved prefix rule (editable;
not offered for compound commands), or deny with an optional reason. The
same choices are bound to y / s / p / n while the bar holds focus, and
escape denies, so an accidental keypress never runs a host command silently.

The bar resolves the future ``app._host_bash_future`` with a ``Decision``
(see xarness/permissions.py); the app's ``_approve_host_bash`` awaits it and
refocuses the chat input afterwards.
"""

from __future__ import annotations

import shlex

from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Button, Input, Label, Static

from rich.text import Text

from .. import theme
from ..permissions import Decision


class HostBashBar(Widget):
    """Approval prompt for one host command, docked above the chat input."""

    BINDINGS = [
        ("escape", "escape", "Deny / back out"),
        ("y", "once", "Allow once"),
        ("s", "session", "Allow this session"),
        ("p", "prefix", "Always allow prefix"),
        ("n", "deny", "Deny"),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._req = None
        self._input_mode = ""

    def compose(self):
        with Vertical(id="host-bash-body"):
            yield Label("", id="host-bash-title", markup=False)
            yield Static("", id="host-bash-command", markup=False)
            yield Label("", id="host-bash-reason", markup=False)
            with Horizontal(id="host-bash-buttons"):
                yield Button("y · once", id="host-bash-once")
                yield Button("s · this session", id="host-bash-session")
                yield Button("n · deny", id="host-bash-deny", variant="error")
            yield Input(id="host-bash-input")

    # ------------------------------------------------------------------
    # show / hide

    def _render_reason(self) -> None:
        """"reason:" stays dimmed; the reason itself reads as normal text.
        Re-rendered on /theme so the baked styles follow the palette."""
        if self._req is None:
            return
        reason = Text("reason: ", style=theme.PALETTE["muted"])
        reason.append(self._req.reason, style=theme.PALETTE["text"])
        self.query_one("#host-bash-reason", Label).update(reason)

    def recolor(self) -> None:
        """Re-render palette-baked text after a /theme switch."""
        if self.has_class("visible"):
            self._render_reason()

    def show(self, req) -> None:
        self._req = req
        self._input_mode = ""
        self.query_one("#host-bash-title", Label).update(
            "run on host — allow this command?"
        )
        self.query_one("#host-bash-command", Static).update(req.command)
        self._render_reason()
        buttons = self.query_one("#host-bash-buttons", Horizontal)
        # The prefix button doubles as the hint: the default (first two
        # tokens) is shown inline so the user knows what `p` would save.
        existing = self.query("#host-bash-buttons > Button.host-bash-prefix")
        for button in existing:
            button.remove()
        if not req.is_compound and req.suggested_prefix:
            prefix = " ".join(req.suggested_prefix)
            buttons.mount(Button(
                f"p · always allow `{prefix}`",
                id="host-bash-prefix-button",
                classes="host-bash-prefix",
            ), before=self.query_one("#host-bash-deny", Button))
        inp = self.query_one("#host-bash-input", Input)
        inp.display = False
        inp.value = ""
        self.add_class("visible")
        self.query_one("#host-bash-once", Button).focus()

    def hide(self) -> None:
        self.remove_class("visible")

    # ------------------------------------------------------------------
    # choices

    def _decide(self, decision: Decision) -> None:
        fut = getattr(self.app, "_host_bash_future", None)
        if fut is not None and not fut.done():
            fut.set_result(decision)

    def action_once(self) -> None:
        if self._input_mode:
            return
        self._decide(Decision(kind="once"))

    def action_session(self) -> None:
        if self._input_mode:
            return
        self._decide(Decision(kind="session"))

    def action_prefix(self) -> None:
        if self._input_mode or self._req is None \
                or self._req.is_compound or not self._req.suggested_prefix:
            return
        self._open_input(
            " ".join(self._req.suggested_prefix),
            "prefix to always allow (edit me, enter to save)",
            mode="prefix",
        )

    def action_deny(self) -> None:
        if self._input_mode:
            return
        self._open_input("", "why deny (optional — enter to deny)", mode="deny")

    def action_escape(self) -> None:
        """Escape: back out of an open input, else deny."""
        if self._input_mode:
            self._close_input()
            return
        self._decide(Decision(kind="deny"))

    # ------------------------------------------------------------------
    # the input line (prefix edit / deny reason)

    def _open_input(self, value: str, placeholder: str, mode: str) -> None:
        self._input_mode = mode
        inp = self.query_one("#host-bash-input", Input)
        inp.value = value
        inp.placeholder = placeholder
        inp.display = True
        inp.focus()

    def _close_input(self) -> None:
        self._input_mode = ""
        inp = self.query_one("#host-bash-input", Input)
        inp.display = False
        inp.value = ""
        self.query_one("#host-bash-once", Button).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "host-bash-input" or not self._input_mode:
            return
        text = event.value.strip()
        if self._input_mode == "prefix":
            self._decide(Decision(kind="prefix", prefix=tuple(shlex.split(text))))
        else:
            self._decide(Decision(kind="deny", deny_reason=text))
        event.stop()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button = event.button
        if self._input_mode:  # choice keys are suspended while editing
            return
        if button.id == "host-bash-once":
            self.action_once()
        elif button.id == "host-bash-session":
            self.action_session()
        elif button.id in ("host-bash-prefix-button", "host-bash-prefix"):
            self.action_prefix()
        elif button.id == "host-bash-deny":
            self.action_deny()
