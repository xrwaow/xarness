"""Config file (JSON, default ``~/.config/xarness/config.json``).

Sectioned layout:

    {
      "models":    {"<name>": {ProviderProfile fields}},
      "sandboxes": {"<name>": {SandboxSettings fields}},
      "presets":   {"<name>": {"model": ..., "sandbox": ...}},
      "ui":        {"theme": "..."},
      "defaults":  {"preset": "default"},
      "server":    {"port": 0}
    }

A model's ``name`` is implied by its key in ``models``. The API key is never
stored here; ``api_key_env`` names the env var that holds it (null = no key
needed).
"""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

DEFAULT_CONFIG_PATH = Path("~/.config/xarness/config.json").expanduser()

DEFAULT_THEME = "ayu-darker"

# Config written on first run (no file at DEFAULT_CONFIG_PATH yet): one
# OpenAI-shaped model using the built-in defaults for everything else.
STARTER_CONFIG: dict[str, Any] = {
    "models": {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "model_id": "gpt-4.1",
        }
    }
}


class ConfigError(Exception):
    """Raised when the config file is missing, malformed, or invalid."""


class CotStrength(str, Enum):
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ProviderProfile(BaseModel):
    """A single provider/model configuration."""

    model_config = ConfigDict(extra="forbid")

    base_url: str
    api_key_env: str | None = "OPENAI_API_KEY"
    model_id: str
    # The key in the config's "models" section; set when parsing.
    name: str | None = None
    shown_name: str | None = None
    max_context: int = Field(default=128_000, gt=0)
    cot_strength: CotStrength = CotStrength.MEDIUM
    # Send assistant reasoning back to the model on later rounds.
    keep_reasoning: bool = True
    # The model accepts image content parts (vision). Off by default; when
    # False the TUI refuses to attach images to messages.
    supports_vision: bool = False
    # Auto-compact after each turn once the context estimate passes
    # auto_compact_threshold of max_context (0.9 = 90% full).
    auto_compact: bool = False
    auto_compact_threshold: float = Field(default=0.9, gt=0.0, le=1.0)
    # OpenRouter-style provider routing, sent verbatim as the request's
    # "provider" field; ignored by endpoints that don't support it.
    provider: dict[str, Any] | None = None

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("must be an http(s) URL, e.g. https://api.openai.com/v1")
        return value

    @field_validator("api_key_env")
    @classmethod
    def _validate_api_key_env(cls, value: str | None) -> str | None:
        """Treat null/blank as "no key needed" (local models), not an error."""
        return value if value is None or value.strip() else None

    @property
    def display_name(self) -> str:
        """Name shown in the UI, independent of the wire ``model_id``."""
        return self.shown_name or self.model_id


class RefSpec(BaseModel):
    """One ``auto_include_refs`` entry with an explicit mount point.

    Plain string entries mean "bind at ``.refs/<alias>``"; a RefSpec can
    instead mount the host path anywhere workspace-relative (e.g. ``mount:
    ".venv"`` to drop a host-built venv into the project) and mark it
    writable."""

    model_config = ConfigDict(extra="forbid")

    path: str
    # Container path of the mount ("" = the container-only refs tmpfs,
    # addressed as ``.refs/<alias>`` with the alias derived from the path's
    # basename).
    mount: str | None = None
    read_only: bool = True

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("path must be a non-empty host path")
        if ".." in Path(value).parts:
            raise ValueError(f"path must not contain '..': {value!r}")
        return value

    @field_validator("mount")
    @classmethod
    def _validate_mount(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().strip("/")
        if not value:
            return None
        if ".." in Path(value).parts or value.startswith(".git"):
            raise ValueError(f"mount must be a relative workspace path without '..': {value!r}")
        return value


class SandboxSettings(BaseModel):
    """One named sandbox profile (the "sandboxes" config section).

    ``sandboxes.default`` is the global profile; other profiles inherit from
    it and list only the keys they override. Resolved **per session** at
    creation, saved in the session file, and fixed for the session's
    lifetime except via /sandbox or /filter."""

    model_config = ConfigDict(extra="forbid")

    allow_network: bool = True
    gpu_access: bool = True
    # Off by default; on disables the git guard entirely (the agent manages
    # git itself) — tracking guarantees no longer hold.
    allow_git_write: bool = False
    auto_include_refs: list[str | RefSpec] = Field(default_factory=list)
    # Paths not mounted into the container (a mount exclusion, not a
    # gitignore). Part of SandboxConfig.mount_key, so the persistent shell
    # restarts when it changes.
    filter_list: list[str] = Field(default_factory=list)
    # False = tools run as a user-level stateful bash on the host (an
    # unsandboxed mode; the TUI shows a visible marker).
    use_container: bool = True
    tool_output_limit: int | None = Field(default=None, ge=256)
    # Replaces the hardcoded 120 s; a per-command timeout_seconds overrides.
    default_timeout: float = Field(default=120.0, gt=0)
    # False: no checkpoints, diff bar, accept/reject or file revert; undo is
    # conversation-only; no auto-init of a repo.
    enable_tracking: bool = True

    @field_validator("auto_include_refs", mode="before")
    @classmethod
    def _coerce_refs(cls, value: Any) -> Any:
        return coerce_refs(value)

    def merged_over(self, base: "SandboxSettings") -> "SandboxSettings":
        """Inheritance: keys this profile sets explicitly win over ``base``."""
        merged = base.model_copy()
        for name in type(self).model_fields:
            if name in self.model_fields_set:
                setattr(merged, name, getattr(self, name))
        return merged


class PresetSettings(BaseModel):
    """One named preset: a model + sandbox profile."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    sandbox: str | None = None


def coerce_refs(value: Any) -> Any:
    """Validator body shared by every ``auto_include_refs`` field: reject
    empty/``..`` string entries, pass dicts and RefSpecs through."""
    if not isinstance(value, list):
        return value
    out: list[str | dict[str, Any]] = []
    for entry in value:
        if isinstance(entry, str):
            if not entry.strip():
                raise ValueError("auto_include_refs entries must be non-empty paths")
            if ".." in Path(entry.strip()).parts:
                raise ValueError(f"auto_include_refs entry must not contain '..': {entry!r}")
            out.append(entry.strip())
        else:
            out.append(entry)
    return out


class SectionedConfig(BaseModel):
    """The parsed config file."""

    models: dict[str, ProviderProfile] = Field(default_factory=dict)
    sandboxes: dict[str, SandboxSettings] = Field(default_factory=dict)
    presets: dict[str, PresetSettings] = Field(default_factory=dict)
    theme: str = DEFAULT_THEME
    default_preset: str = "default"

    def resolve_sandbox(self, name: str | None) -> tuple[str, SandboxSettings]:
        """Resolve a sandbox profile by name ("default" when None), with
        inheritance from ``sandboxes.default``."""
        key = name or "default"
        if key not in self.sandboxes:
            raise ConfigError(
                f"no sandbox profile named {key!r}; available: "
                + (", ".join(self.sandboxes) or "none")
            )
        settings = self.sandboxes[key]
        if key != "default" and "default" in self.sandboxes:
            settings = settings.merged_over(self.sandboxes["default"])
        return key, settings

    def resolve_model(self, name: str | None = None) -> tuple[str, ProviderProfile]:
        """Resolve a model profile by name, defaulting to the default
        preset's model."""
        key = name or self.default_preset_model()
        if key not in self.models:
            raise ConfigError(
                f"no model named {key!r}; available: " + (", ".join(self.models) or "none")
            )
        return key, self.models[key]

    def default_preset_model(self) -> str | None:
        preset = self.presets.get(self.default_preset)
        if preset and preset.model:
            return preset.model
        # No preset carrying a model: a single defined model is unambiguous.
        if len(self.models) == 1:
            return next(iter(self.models))
        return None

    def default_preset_sandbox(self) -> str | None:
        preset = self.presets.get(self.default_preset)
        return preset.sandbox if preset else None


def _validate(path: Path, model: Any, data: Any) -> Any:
    try:
        return TypeAdapter(model).validate_python(data)
    except ValidationError as exc:
        lines = [
            f"  - {'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']} (got {repr(e.get('input'))[:60]})"
            for e in exc.errors()
        ]
        raise ConfigError(f"invalid config in {path}:\n" + "\n".join(lines)) from None


def _read(path: Path) -> dict[str, Any]:
    """Read and parse the config file into a top-level object."""
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"could not parse {path} as JSON:\n{exc}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a JSON object at the top level")
    return data


_SECTIONED_CACHE: dict[tuple[str, int], SectionedConfig] = {}
# SectionedConfig is immutable in practice (callers only read it); cache by
# (path, mtime_ns) so per-request loads (config_info, sandbox resolution,
# workspace last-used) don't re-read and re-validate the file.


def load_sectioned(path: Path) -> SectionedConfig:
    """Load and validate the config file (cached on path + mtime)."""
    path = Path(path)
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except OSError:
        key = None
    if key is not None:
        cached = _SECTIONED_CACHE.get(key)
        if cached is not None:
            return cached
    data = _read(path)
    models = {
        name: _validate(path, ProviderProfile, {"name": name, **fields})
        for name, fields in (data.get("models") or {}).items()
    }
    sandboxes = {
        name: _validate(path, SandboxSettings, fields)
        for name, fields in (data.get("sandboxes") or {}).items()
    }
    sandboxes.setdefault("default", SandboxSettings())
    presets = {
        name: _validate(path, PresetSettings, fields)
        for name, fields in (data.get("presets") or {}).items()
    }
    ui = data.get("ui") or {}
    defaults = data.get("defaults") or {}
    sectioned = SectionedConfig(
        models=models, sandboxes=sandboxes, presets=presets,
        theme=ui.get("theme") or DEFAULT_THEME,
        default_preset=defaults.get("preset") or "default",
    )
    if key is not None:
        _SECTIONED_CACHE.clear()  # keep at most the newest entry per file
        _SECTIONED_CACHE[key] = sectioned
    return sectioned


def ensure_config(path: Path = DEFAULT_CONFIG_PATH) -> list[str]:
    """Single startup check for the config file.

    Missing file: create one from the starter config. Anything else is left
    alone — load_sectioned reports malformed/invalid files properly."""
    if path.exists():
        return []
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(STARTER_CONFIG, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not create {path}: {exc}") from exc
    return [
        f"created starter config at {path} — edit it to point at your provider"
    ]


def save_preferences(
    path: Path,
    *,
    default_theme: str | None = None,
    default_profile: str | None = None,
    auto_compact: bool | None = None,
    profile: str | None = None,
) -> None:
    """Persist UI preferences (chosen theme / profile) into the config file so
    they carry over to the next session.

    ``default_profile`` lands where the loader reads it back: as
    ``defaults.preset`` (when a preset already carries that model) or as the
    default preset's model. ``auto_compact`` is written onto the named model
    (``profile``, defaulting to the file's default)."""
    data = _read(path)
    if default_theme is not None:
        data.setdefault("ui", {})["theme"] = default_theme
    if default_profile is not None:
        presets = data.setdefault("presets", {})
        defaults = data.setdefault("defaults", {})
        current = defaults.get("preset") or "default"
        matching = [
            name for name, fields in presets.items()
            if isinstance(fields, dict) and fields.get("model") == default_profile
        ]
        if matching:
            defaults["preset"] = matching[0]
        else:
            presets.setdefault(current, {})["model"] = default_profile
    if auto_compact is not None:
        target = profile or default_profile
        if target is None:
            # Fall back to the default preset's model.
            sectioned = load_sectioned(path)
            target = sectioned.default_preset_model()
        if target is None:
            raise ConfigError("no model named to write auto_compact onto")
        data.setdefault("models", {}).setdefault(target, {})["auto_compact"] = auto_compact
    try:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not write {path}: {exc}") from exc


def resolve_api_key(
    profile: ProviderProfile, env: dict[str, str] | None = None,
) -> str | None:
    """Read the API key from the env var named by ``api_key_env``.

    ``env`` defaults to this process's environment; a host process resolves
    keys against the *client's* environment instead (its own is stale — the
    host is long-lived). The mapping is used only for the lookup and is
    never persisted or logged."""
    if profile.api_key_env is None:
        return None
    source = os.environ if env is None else env
    key = source.get(profile.api_key_env)
    if not key:
        raise ConfigError(
            f"environment variable {profile.api_key_env!r} (api_key_env from config) is not set; "
            "the API key is read from the environment, never from the config file"
        )
    return key
