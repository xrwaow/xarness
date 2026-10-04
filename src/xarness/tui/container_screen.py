"""Container settings popup, two tabs:

- **Workspace** (default): live-mutates the running SandboxConfig — network
  access, .gitignore shadowing, GPU access, external references — plus the
  auto-compact flag and per-tool output cap. Changes are persisted per
  *workspace* (not per session), so every session launched in the same
  workspace starts with them; the persistent shell restarts on its next
  command via the sandbox's mount key.
- **Global**: edits the same settings as they will be written to the config
  file's top-level ``container`` section (plus the profile's auto-compact
  default). Every change is persisted automatically; it becomes the startup
  default for new sessions.

Ref rows show the full host path plus mount point, a read-only/writable
switch, and a ✕ button to remove.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from textual import on
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Switch, TabbedContent, TabPane

from ..config import ConfigError, ContainerSettings, RefSpec, save_container_settings, save_preferences
from ..sandbox import SandboxConfig, SandboxUnavailable
from ..tools import DEFAULT_OUTPUT_LIMIT


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
        *,
        session_auto_compact: bool = False,
        session_output_limit: int = DEFAULT_OUTPUT_LIMIT,
        default_auto_compact: bool = False,
        profile_name: str | None = None,
        on_auto_compact: Callable[[bool], None] | None = None,
        on_output_limit: Callable[[int], None] | None = None,
        workspace: str | None = None,
    ) -> None:
        super().__init__()
        self._sandbox = sandbox
        self._config_path = config_path
        self._workspace = workspace
        self._session_auto_compact = session_auto_compact
        self._session_output_limit = session_output_limit
        self._default_auto_compact = default_auto_compact
        self._profile_name = profile_name
        self._on_auto_compact = on_auto_compact
        self._on_output_limit = on_output_limit
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
                with TabPane("Workspace", id="session-tab"):
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
        yield from self._switch_row(
            "Auto-compact", self._session_auto_compact, "container-autocompact",
        )
        yield from self._limit_row("container-output-limit", self._session_output_limit)
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

    @staticmethod
    def _limit_row(input_id: str, value: int):
        with Horizontal(classes="container-row"):
            yield Label("Tool output limit", classes="container-label")
            yield Input(str(value), id=input_id, classes="container-limit-input")

    def _pane_defaults(self):
        yield from self._switch_row("Network access", self._defaults.network_access, "defaults-net")
        yield from self._switch_row(
            "Respect .gitignore", self._defaults.respect_gitignore, "defaults-gitignore",
        )
        yield from self._switch_row("GPU access", self._defaults.gpu_access, "defaults-gpu")
        yield from self._switch_row(
            "Auto-compact", self._default_auto_compact, "defaults-autocompact",
        )
        yield from self._limit_row(
            "defaults-output-limit", self._defaults.tool_output_limit or DEFAULT_OUTPUT_LIMIT,
        )
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
            # The system prompt mentions 0-byte gitignored placeholders only
            # when ignore rules are respected — refresh it for the next turn.
            app = self.app
            if hasattr(app, "ensure_system_message"):
                app.ensure_system_message()
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

    @on(Button.Pressed, "#container-autocompact, #defaults-autocompact")
    def _autocompact_changed(self, event: Button.Pressed) -> None:
        defaults = self._active_defaults
        value = not (self._default_auto_compact if defaults else self._session_auto_compact)
        if defaults:
            self._default_auto_compact = value
        else:
            self._session_auto_compact = value
        self._sync_toggle(event.button, value)
        if defaults:
            if self._config_path is None:
                self.app.notify("no config path known; cannot save default", severity="warning")
                return
            try:
                save_preferences(
                    self._config_path, auto_compact=value, profile=self._profile_name,
                )
            except ConfigError as exc:
                self.app.notify(f"save failed: {exc}", severity="error")
        elif self._on_auto_compact is not None:
            self._on_auto_compact(value)

    @on(Input.Submitted, "#container-output-limit, #defaults-output-limit")
    def _output_limit_changed(self, event: Input.Submitted) -> None:
        limit = _parse_output_limit(event.input.value)
        if isinstance(limit, str):
            self.app.notify(f"output limit not set: {limit}", severity="warning")
            return
        event.input.value = str(limit)
        if self._active_defaults:
            self._defaults.tool_output_limit = limit
            self._persist()
            self.app.notify(f"tool output limit default set to {limit:,} chars")
        else:
            self._session_output_limit = limit
            if self._on_output_limit is not None:
                self._on_output_limit(limit)
            self.app.notify(f"tool output limit set to {limit:,} chars")

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
        """Auto-save the active tab's settings immediately.

        Workspace-tab changes are saved per-workspace (shared by every
        session launched in that workspace) without leaking into the global
        config defaults; defaults-tab changes write the config file."""
        if self._active_defaults:
            settings = self._defaults
        elif self._sandbox is not None:
            if self._workspace is None:
                # No workspace (e.g. started with --no-workspace): the change
                # is in-memory only for this run.
                return
            from ..session_store import save_workspace_container

            save_workspace_container(self._workspace, self._sandbox.session_settings(
                tool_output_limit=self._session_output_limit,
                auto_compact=self._session_auto_compact,
            ))
            return
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


def _parse_output_limit(text: str) -> int | str:
    """Parse the output-limit input; returns the limit or an error message."""
    try:
        limit = int(text.strip())
    except ValueError:
        return f"not a number: {text.strip()}"
    if limit < 256:
        return "must be at least 256"
    return limit


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
