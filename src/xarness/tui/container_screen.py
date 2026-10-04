"""Container settings popup, two tabs:

- **Session** (default): live-mutates the running SandboxConfig — network
  access, .gitignore shadowing, GPU access, external references. The
  persistent shell restarts on its next command via the sandbox's mount key;
  changes die with the chat session.
- **Global**: edits the same settings as they will be written to the
  config file's top-level ``container`` section. Every change is persisted
  automatically; it becomes the startup default for new sessions.

Ref rows show the full host path plus mount point, a read-only/writable
switch, and a ✕ button to remove.
"""

from __future__ import annotations

import json
from pathlib import Path

from textual import on
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Switch, TabbedContent, TabPane

from ..config import ConfigError, ContainerSettings, RefSpec, save_container_settings
from ..sandbox import SandboxConfig, SandboxUnavailable


class ContainerSettingsScreen(ModalScreen[None]):
    """Sandbox policy and external references, per session and as defaults."""

    BINDINGS = [
        ("escape", "cancel", "Close"),
        ("delete", "remove_ref", "Remove selected ref"),
    ]

    def __init__(
        self,
        sandbox: SandboxConfig | None,
        config_path: Path | None = None,
    ) -> None:
        super().__init__()
        self._sandbox = sandbox
        self._config_path = config_path
        self._defaults = self._load_defaults()

    def _load_defaults(self) -> ContainerSettings:
        if self._config_path is not None and self._config_path.exists():
            try:
                data = json.loads(self._config_path.read_text(encoding="utf-8"))
                if isinstance(data.get("container"), dict):
                    return ContainerSettings.model_validate(data["container"])
            except (OSError, ValueError):
                pass  # unreadable/broken config: edit from defaults
        return ContainerSettings()

    # ------------------------------------------------------------------
    # Layout

    def compose(self):
        with Vertical(id="container-modal"):
            yield Label("Container settings", id="container-title")
            with TabbedContent(initial="session-tab", id="container-tabs"):
                with TabPane("Session", id="session-tab"):
                    yield from self._pane(self._sandbox is not None)
                with TabPane("Global", id="defaults-tab"):
                    yield from self._pane_defaults()

    def _switch_row(self, label: str, value: bool, switch_id: str):
        with Horizontal(classes="container-row"):
            yield Label(label, classes="container-label")
            yield self._toggle(value, switch_id)

    @staticmethod
    def _toggle(value: bool, button_id: str) -> Button:
        """A toggle button instead of a slider switch: dimmed when off,
        accent when on."""
        button = Button("on" if value else "off", id=button_id, classes="container-toggle")
        if value:
            button.add_class("is-on")
        return button

    def _pane(self, session: bool):
        yield from self._switch_row("Network access", session and self._sandbox.allow_network, "container-net")
        yield from self._switch_row(
            "Respect .gitignore", session and self._sandbox.respect_gitignore, "container-gitignore",
        )
        yield from self._switch_row("GPU access", session and self._sandbox.gpu_access, "container-gpu")
        yield Label(
            "External references — ✕ removes, switch flips read-only/writable",
            id="container-refs-title",
            markup=False,
        )
        yield ListView(id="container-refs")
        yield Input(
            placeholder="add: <path> [mount] — enter to add",
            id="container-ref-input",
        )

    def _pane_defaults(self):
        yield from self._switch_row("Network access", self._defaults.network_access, "defaults-net")
        yield from self._switch_row(
            "Respect .gitignore", self._defaults.respect_gitignore, "defaults-gitignore",
        )
        yield from self._switch_row("GPU access", self._defaults.gpu_access, "defaults-gpu")
        yield Label(
            "External references — ✕ removes, switch flips read-only/writable",
            id="defaults-refs-title",
            markup=False,
        )
        yield ListView(id="defaults-refs")
        yield Input(
            placeholder="add: <path> [mount] — enter to add",
            id="defaults-ref-input",
        )

    def on_mount(self) -> None:
        self._refresh_refs()
        self.query_one("#container-ref-input", Input).focus()

    # ------------------------------------------------------------------
    # Active tab / state plumbing

    @property
    def _active_defaults(self) -> bool:
        return self.query_one("#container-tabs", TabbedContent).active == "defaults-tab"

    def _refs_widget(self) -> ListView:
        return self.query_one(
            "#defaults-refs" if self._active_defaults else "#container-refs", ListView,
        )

    def _ref_input(self) -> Input:
        return self.query_one(
            "#defaults-ref-input" if self._active_defaults else "#container-ref-input", Input,
        )

    # ------------------------------------------------------------------
    # Refs list

    def _session_ref_rows(self) -> list[tuple[Path, str, bool]]:
        return [
            (ref.host, ref.mount, ref.read_only)
            for ref in self._sandbox.external_refs.values()
        ]

    def _default_ref_rows(self) -> list[tuple[str, str, bool]]:
        rows = []
        for entry in self._defaults.auto_include_refs:
            path = entry if isinstance(entry, str) else entry.path
            if isinstance(entry, str):
                mount = f".refs/{Path(path).expanduser().name}"
            else:
                mount = entry.mount or f".refs/{Path(path).expanduser().name}"
            read_only = True if isinstance(entry, str) else entry.read_only
            rows.append((path, mount, read_only))
        return rows

    def _refresh_refs(self) -> None:
        list_view = self._refs_widget()
        list_view.clear()
        prefix = "d" if self._active_defaults else "s"
        rows = self._default_ref_rows() if self._active_defaults else self._session_ref_rows()
        for idx, (path, mount, read_only) in enumerate(rows):
            flag = "writable" if not read_only else "read-only"
            list_view.append(ListItem(
                Vertical(
                    Label(str(path), classes="container-ref-path"),
                    Horizontal(
                        Label(f"→ {mount}  ({flag})", classes="container-ref-mount"),
                        Switch(
                            not read_only, id=f"{prefix}-rw-{idx}",
                            classes="container-ref-rw",
                        ),
                        Button("✕", id=f"{prefix}-rm-{idx}", classes="container-rm"),
                        classes="container-ref-controls",
                    ),
                    classes="container-ref-item",
                ),
            ))

    @on(TabbedContent.TabActivated)
    def _tab_changed(self) -> None:
        self._refresh_refs()
        self._ref_input().focus()

    # ------------------------------------------------------------------
    # Policy switches

    @on(Button.Pressed, "#container-net")
    def _net_changed(self, event: Button.Pressed) -> None:
        if self._sandbox is not None:
            self._sandbox.set_network_access(not self._sandbox.allow_network)
            self._sync_toggle(event.button, self._sandbox.allow_network)
            self._persist()

    @on(Button.Pressed, "#container-gitignore")
    def _gitignore_changed(self, event: Button.Pressed) -> None:
        if self._sandbox is not None:
            self._sandbox.set_respect_gitignore(not self._sandbox.respect_gitignore)
            self._sync_toggle(event.button, self._sandbox.respect_gitignore)
            self._persist()

    @on(Button.Pressed, "#container-gpu")
    def _gpu_changed(self, event: Button.Pressed) -> None:
        if self._sandbox is not None:
            self._sandbox.set_gpu_access(not self._sandbox.gpu_access)
            self._sync_toggle(event.button, self._sandbox.gpu_access)
            if self._sandbox.gpu_access:
                self.run_worker(self._gpu_probe(), exclusive=True)
            self._persist()

    async def _gpu_probe(self) -> None:
        from ..sandbox import probe_gpu_access

        probe = await probe_gpu_access(self._sandbox)
        if probe is None:
            return
        level, message = probe
        self.app.notify(message, severity="warning" if level == "warn" else "information",
                        title="GPU access")

    @on(Button.Pressed, "#defaults-net")
    def _defaults_net(self, event: Button.Pressed) -> None:
        self._defaults.network_access = not self._defaults.network_access
        self._sync_toggle(event.button, self._defaults.network_access)
        self._persist()

    @on(Button.Pressed, "#defaults-gitignore")
    def _defaults_gitignore(self, event: Button.Pressed) -> None:
        self._defaults.respect_gitignore = not self._defaults.respect_gitignore
        self._sync_toggle(event.button, self._defaults.respect_gitignore)
        self._persist()

    @on(Button.Pressed, "#defaults-gpu")
    def _defaults_gpu(self, event: Button.Pressed) -> None:
        self._defaults.gpu_access = not self._defaults.gpu_access
        self._sync_toggle(event.button, self._defaults.gpu_access)
        self._persist()

    @staticmethod
    def _sync_toggle(button: Button, value: bool) -> None:
        button.label = "on" if value else "off"
        button.set_class(value, "is-on")

    # ------------------------------------------------------------------
    # Ref rows: writable switch + remove button

    @on(Switch.Changed, ".container-ref-rw")
    def _ref_rw_changed(self, event: Switch.Changed) -> None:
        switch = event.switch
        idx = int((switch.id or "").rsplit("-", 1)[1])
        writable = event.value
        if self._active_defaults:
            rows = self._default_ref_rows()
            path, mount, _ = rows[idx]
            self._defaults.auto_include_refs[idx] = RefSpec(
                path=path, mount=None if mount.startswith(".refs/") else mount,
                read_only=not writable,
            )
        else:
            ref = list(self._sandbox.external_refs.values())[idx]
            ref.read_only = not writable
        self._refresh_refs()

    @on(Button.Pressed, ".container-rm")
    def _remove_ref(self, event: Button.Pressed) -> None:
        idx = int((event.button.id or "").rsplit("-", 1)[1])
        if self._active_defaults:
            del self._defaults.auto_include_refs[idx]
        else:
            alias = list(self._sandbox.external_refs)[idx]
            self._sandbox.remove_ref(alias)
        self._refresh_refs()
        self._persist()

    # ------------------------------------------------------------------
    # Add / remove via keyboard

    @on(Input.Submitted, "#container-ref-input")
    @on(Input.Submitted, "#defaults-ref-input")
    def _add_ref(self) -> None:
        input_widget = self._ref_input()
        try:
            path, mount, read_only = _parse_ref_input(input_widget.value)
        except (ValueError, FileNotFoundError) as exc:
            self.app.notify(f"ref not added: {exc}", severity="warning")
            return
        if self._active_defaults:
            self._defaults.auto_include_refs.append(
                RefSpec(path=path, mount=mount, read_only=read_only)
            )
        else:
            try:
                self._sandbox.add_ref(path, mount=mount, read_only=read_only)
            except (FileNotFoundError, SandboxUnavailable) as exc:
                self.app.notify(f"ref not added: {exc}", severity="warning")
                return
        input_widget.value = ""
        self._refresh_refs()
        self._persist()
        self.app.notify(f"mounted {path} at {mount or '.refs/<alias>'}")

    def action_remove_ref(self) -> None:
        list_view = self._refs_widget()
        idx = list_view.index
        rows = self._default_ref_rows() if self._active_defaults else self._session_ref_rows()
        if idx is None or idx >= len(rows):
            return
        if self._active_defaults:
            del self._defaults.auto_include_refs[idx]
        else:
            alias = list(self._sandbox.external_refs)[idx]
            self._sandbox.remove_ref(alias)
        self._refresh_refs()
        self._persist()

    # ------------------------------------------------------------------
    # Footer

    def action_cancel(self) -> None:
        """Escape closes the popup (see BINDINGS)."""
        self.dismiss(None)

    def _persist(self) -> None:
        """Auto-save: any change writes the active tab's settings to the
        config file immediately (session tab snapshots the sandbox as the
        new defaults; defaults tab writes what it edits)."""
        if self._active_defaults:
            settings = self._defaults
        elif self._sandbox is not None:
            settings = ContainerSettings(
                network_access=self._sandbox.allow_network,
                respect_gitignore=self._sandbox.respect_gitignore,
                gpu_access=self._sandbox.gpu_access,
                auto_include_refs=[
                    RefSpec(path=str(ref.host), mount=ref.mount, read_only=ref.read_only)
                    for ref in self._sandbox.external_refs.values()
                ],
            )
        else:
            self.app.notify("no container running; nothing to save", severity="warning")
            return
        if self._config_path is None:
            self.app.notify("no config path known; cannot save defaults", severity="warning")
            return
        try:
            save_container_settings(self._config_path, settings)
        except ConfigError as exc:
            self.app.notify(f"save failed: {exc}", severity="error")


def _parse_ref_input(text: str) -> tuple[str, str | None, bool]:
    """Parse the add-ref input: ``<path> [mount]``.

    The first token is the host path (absolute, ~-expanded, or
    workspace-relative); an optional second is the mount point relative to
    the workspace. New refs default to read-only — flip the row's switch to
    make one writable. (``--rw`` is still accepted for backwards
    compatibility.)"""
    tokens = text.split()
    if not tokens:
        raise ValueError("give a path")
    read_only = True
    mount: str | None = None
    rest = tokens[1:]
    if rest and rest[-1] == "--rw":
        read_only = False
        rest = rest[:-1]
    if rest:
        if len(rest) > 1:
            raise ValueError("expected at most: <path> [mount]")
        mount = rest[0]
    return tokens[0], mount, read_only
