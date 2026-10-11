"""Config file (JSON, default ``~/.config/xarness/config.json``).

Two shapes: named profiles (``{"default_profile", "default_theme",
"profiles": [{...}]}``, each profile needs a unique ``name``) or a flat
single profile (a ``"provider"`` object, or profile fields at the top
level). The API key is never stored here; ``api_key_env`` names the env
var that holds it (null = no key needed).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

DEFAULT_CONFIG_PATH = Path("~/.config/xarness/config.json").expanduser()

DEFAULT_THEME = "ayu-darker"

# Config written on first run (no file at DEFAULT_CONFIG_PATH yet): one
# OpenAI-shaped profile using the built-in defaults for everything else.
STARTER_CONFIG: dict[str, Any] = {
    "default_profile": "openai",
    "profiles": [
        {
            "name": "openai",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "model_id": "gpt-4.1",
        }
    ],
}

# Optional per-profile keys the starter config leaves at their defaults.
PROFILE_OPTIONAL_DEFAULTS: dict[str, Any] = {
    "api_key_env": "OPENAI_API_KEY",
    "shown_name": None,
    "max_context": 128_000,
    "cot_strength": "medium",
    "keep_reasoning": True,
    "supports_vision": False,
    "auto_compact": False,
    "auto_compact_threshold": 0.9,
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
    # Required (and unique) for profiles in the "profiles" list; the flat
    # single-profile form has no name.
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


class ContainerSettings(BaseModel):
    """Sandbox/container knobs, top-level "container" section of the config.

    Applies to the one sandbox all tools run in (see ``sandbox.py``);
    independent of which profile is selected.

    Network and GPU access are launch-time CLI flags (``--network``,
    ``--gpu``), not config settings."""

    model_config = ConfigDict(extra="forbid")

    # Host paths bound into the container at runtime, so the model can
    # consult (or, when writable, reuse) files that live outside the
    # workspace: specs, notes, other checkouts, prebuilt toolchains. Entries
    # are plain paths (bound read-only under the container-only refs tmpfs,
    # addressed as ``.refs/<alias>``) or RefSpec
    # objects with an explicit ``mount`` and ``read_only`` flag. The alias is
    # derived from the path (its basename, disambiguated with parent segments
    # when two paths share a basename).
    auto_include_refs: list[str | RefSpec] = Field(default_factory=list)

    # Per-tool output cap (chars) the model receives. None = the built-in
    # default (tools.DEFAULT_OUTPUT_LIMIT).
    tool_output_limit: int | None = Field(default=None, ge=256)

    @field_validator("auto_include_refs", mode="before")
    @classmethod
    def _coerce_refs(cls, value: Any) -> Any:
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
            elif isinstance(entry, dict):
                out.append(entry)
            else:
                out.append(entry)
        return out


@dataclass(frozen=True)
class LoadedConfig:
    """Everything load_config derives from one config file read."""

    profile: ProviderProfile
    # Name of the selected profile ("default" for the flat form).
    profile_name: str | None
    # All profile names in the file ([] for the flat form).
    profile_names: list[str]
    default_theme: str
    # Container settings (sandbox policy); defaults when the config omits
    # the "container" section.
    container: ContainerSettings = Field(default_factory=ContainerSettings)


# ----------------------------------------------------------------------
# sectioned layout (models / sandboxes / presets / ui / defaults / server)


class SandboxSettings(BaseModel):
    """One named sandbox profile (the "sandboxes" config section).

    ``sandboxes.default`` is the global profile; other profiles inherit from
    it and list only the keys they override. Resolved **per session** at
    creation, saved in the session file, and fixed for the session's
    lifetime except via /sandbox or /filter."""

    model_config = ConfigDict(extra="forbid")

    allow_network: bool = False
    gpu_access: bool = False
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

    def merged_over(self, base: "SandboxSettings") -> "SandboxSettings":
        """Inheritance: keys this profile sets explicitly win over ``base``."""
        merged = base.model_copy()
        for name in type(self).model_fields:
            if name in self.model_fields_set:
                setattr(merged, name, getattr(self, name))
        return merged


class PresetSettings(BaseModel):
    """One named preset: a model + sandbox (+ subagent, parsed but not yet
    used — the subagent feature comes later)."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    sandbox: str | None = None
    subagent: str | None = None


class ServerSettings(BaseModel):
    """The "server" section: TCP port for ``xarness --server`` (0 = pick a
    free loopback port). The unix socket is always served."""

    model_config = ConfigDict(extra="forbid")

    port: int = Field(default=0, ge=0, le=65535)


@dataclass(frozen=True)
class SectionedConfig:
    """The sectioned view of one config file (either layout)."""

    models: dict[str, ProviderProfile]
    sandboxes: dict[str, SandboxSettings]
    presets: dict[str, PresetSettings]
    theme: str
    default_preset: str
    server: ServerSettings
    # Notices (e.g. old-layout deprecation) for the caller to surface.
    notices: list[str]

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

    def resolve_model(self, name: str | None) -> tuple[str, ProviderProfile]:
        key = name or self.default_preset_model()
        if key not in self.models:
            raise ConfigError(
                f"no model named {key!r}; available: " + ", ".join(self.models)
            )
        return key, self.models[key]

    def default_preset_model(self) -> str | None:
        preset = self.presets.get(self.default_preset)
        return preset.model if preset else None

    def default_preset_sandbox(self) -> str | None:
        preset = self.presets.get(self.default_preset)
        return preset.sandbox if preset else None


def _sectioned(path: Path, data: dict[str, Any]) -> SectionedConfig:
    """Parse either layout into a SectionedConfig.

    The old layout (``profiles`` list + ``container`` + ``default_theme`` +
    ``default_profile``) keeps working, with a deprecation notice; new
    writes go to the sectioned layout."""
    notices: list[str] = []
    if isinstance(data.get("models"), dict):
        # Sectioned layout.
        models = {
            name: _validate(path, ProviderProfile, {"name": name, **fields})
            for name, fields in data["models"].items()
        }
        sandboxes = {
            name: _validate(path, SandboxSettings, fields)
            for name, fields in (data.get("sandboxes") or {"default": {}}).items()
        }
        sandboxes.setdefault("default", SandboxSettings())
        presets = {
            name: _validate(path, PresetSettings, fields)
            for name, fields in (data.get("presets") or {}).items()
        }
        ui = data.get("ui") or {}
        defaults = data.get("defaults") or {}
        server = _validate(path, ServerSettings, data.get("server") or {})
        return SectionedConfig(
            models=models, sandboxes=sandboxes, presets=presets,
            theme=ui.get("theme") or DEFAULT_THEME,
            default_preset=defaults.get("preset") or "default",
            server=server, notices=notices,
        )
    # Old layout: normalize to the sectioned form in memory.
    notices.append(
        "config uses the old layout (profiles/container sections); it still "
        "works — rewrite it with named sections (models/sandboxes/presets) "
        "when you next edit it"
    )
    normalized = _normalize(path, data)
    profiles = _validate(path, list[ProviderProfile], normalized.get("profiles"))
    models = {p.name or "default": p for p in profiles}
    container = _validate(path, ContainerSettings | None, data.get("container")) or ContainerSettings()
    sandboxes = {"default": SandboxSettings(
        auto_include_refs=list(container.auto_include_refs),
        tool_output_limit=container.tool_output_limit,
    )}
    default_profile = normalized.get("default_profile")
    # Synthesize the default preset from default_profile, so the usual
    # preset-based resolution works unchanged for old-layout files.
    presets = {"default": PresetSettings(model=default_profile, sandbox="default")}
    return SectionedConfig(
        models=models, sandboxes=sandboxes, presets=presets,
        theme=data.get("default_theme") or DEFAULT_THEME,
        default_preset="default",
        server=ServerSettings(), notices=notices,
    )


def load_sectioned(path: Path) -> SectionedConfig:
    """Load the config file as a SectionedConfig (either layout)."""
    return _sectioned(path, _read(path))


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


def ensure_config(path: Path = DEFAULT_CONFIG_PATH) -> list[str]:
    """Single startup check for the config file.

    Missing file: create one from the starter config. Existing file: report
    (as returned notices) the optional values it omits so the user knows the
    built-in defaults are in effect. Malformed/invalid files are left alone —
    load_config reports them properly."""
    if not path.exists():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(STARTER_CONFIG, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"could not create {path}: {exc}") from exc
        return [
            f"created starter config at {path} — edit it to point at your provider"
        ]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []  # load_config reports the problem
    if not isinstance(data, dict):
        return []
    missing: list[str] = []
    if "default_theme" not in data:
        missing.append("default_theme (= ayu-darker)")
    if "container" not in data:
        missing.append("container (= built-in defaults)")
    profiles = data.get("profiles")
    profiles = profiles if isinstance(profiles, list) else [
        data.get("provider") if isinstance(data.get("provider"), dict) else data,
    ]
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        name = profile.get("name") or "default"
        absent = [key for key in PROFILE_OPTIONAL_DEFAULTS if key not in profile]
        if absent:
            missing.append(f"{name}: {', '.join(absent)} (= built-in defaults)")
    if not missing:
        return []
    return [
        f"{path} is missing some optional values; the built-in defaults apply: "
        + "; ".join(missing)
    ]


def _validate(path: Path, model: Any, data: Any) -> Any:
    try:
        return TypeAdapter(model).validate_python(data)
    except ValidationError as exc:
        lines = [
            f"  - {'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']} (got {repr(e.get('input'))[:60]})"
            for e in exc.errors()
        ]
        raise ConfigError(f"invalid config in {path}:\n" + "\n".join(lines)) from None


def _named_profiles(path: Path, data: dict[str, Any]) -> list[ProviderProfile]:
    """Validate the named-profile form, returning its profiles."""
    stray = [key for key in data
             if key not in ("default_profile", "default_theme", "profiles", "container")]
    if stray:
        raise ConfigError(
            f"{path} mixes a 'profiles' section with top-level fields {stray}; use one style or the other"
        )
    profiles = _validate(path, list[ProviderProfile], data.get("profiles"))
    names = [p.name for p in profiles]
    if any(not n for n in names):
        raise ConfigError(f"every profile in {path} needs a non-empty 'name'")
    if len(set(names)) != len(names):
        raise ConfigError(f"duplicate profile names in {path}")
    return profiles


def _normalize(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    """Rewrite either config shape into the named-profiles form.

    The flat single-profile form becomes a one-entry ``profiles`` list named
    "default" (in memory only — the file keeps its original style), so every
    consumer works with one shape instead of branching per function."""
    if isinstance(data.get("profiles"), list):
        _named_profiles(path, data)  # validate the named form up front
        return data
    if "default_profile" in data:
        raise ConfigError(f"'default_profile' in {path} is only meaningful alongside a 'profiles' section")
    fields = dict(data["provider"]) if "provider" in data else {
        key: value for key, value in data.items()
        if key not in ("default_profile", "default_theme", "profiles", "provider", "container")
    }
    return {**data, "profiles": [{"name": "default", **fields}], "default_profile": "default"}


def load_config(path: Path, profile_name: str | None = None) -> LoadedConfig:
    """Load and validate the config file.

    Profile selection order: explicit ``--profile`` > ``default_profile`` >
    the only profile when exactly one is defined.
    """
    data = _read(path)
    if not isinstance(data.get("profiles"), list) and profile_name is not None:
        raise ConfigError(
            f"{path} defines a single unnamed profile, so --profile {profile_name!r} has nothing to "
            "select; add a top-level 'profiles' section to use named profiles"
        )
    normalized = _normalize(path, data)
    profiles = _validate(path, list[ProviderProfile], normalized.get("profiles"))
    names = [p.name for p in profiles]
    selected = profile_name or normalized.get("default_profile") or (names[0] if len(names) == 1 else None)
    if selected is None:
        raise ConfigError(
            f"{path} defines multiple profiles ({', '.join(names)}) but no 'default_profile'; "
            "pass --profile to choose one"
        )
    if selected not in names:
        raise ConfigError(
            f"no profile named {selected!r} in {path}; available profiles: {', '.join(names)}"
        )
    by_name = {p.name: p for p in profiles}
    container = _validate(path, ContainerSettings | None, data.get("container"))
    return LoadedConfig(
        by_name[selected], selected, names if isinstance(data.get("profiles"), list) else [],
        data.get("default_theme") or DEFAULT_THEME,
        container or ContainerSettings(),
    )


def list_profile_names(path: Path) -> list[str]:
    """Profile names in the file ([] for the flat single-profile form)."""
    data = _read(path)
    if not isinstance(data.get("profiles"), list):
        return []
    return [p.name for p in _validate(path, list[ProviderProfile], _normalize(path, data)["profiles"])]


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

    ``default_profile`` is only written when the file uses the named-profiles
    form — the flat form has nothing to select, and a stray ``default_profile``
    there would make the config invalid.

    ``auto_compact`` is written onto the named profile (``profile``, defaulting
    to the file's default); in the flat form it goes to the top level (or into
    the ``provider`` section when the config uses one).
    """
    data = _read(path)
    sectioned = isinstance(data.get("models"), dict)
    if default_theme is not None:
        if sectioned:
            data.setdefault("ui", {})["theme"] = default_theme
        else:
            data["default_theme"] = default_theme
    # default_profile only fits the named-profiles form; a stray one would
    # make a flat config invalid, so it is not written there. In the
    # sectioned form it must land where _sectioned reads it back: either as
    # ``defaults.preset`` (when a preset carries that model) or as the
    # default preset's model.
    if default_profile is not None and (
        sectioned or isinstance(data.get("profiles"), list)
    ):
        if sectioned:
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
        else:
            data["default_profile"] = default_profile
    if auto_compact is not None:
        if sectioned:
            target = profile or default_profile
            if target is None:
                raise ConfigError("no model named to write auto_compact onto")
            data.setdefault("models", {}).setdefault(target, {})["auto_compact"] = auto_compact
        elif isinstance(data.get("profiles"), list):
            target = profile or data.get("default_profile")
            entries = [p for p in data["profiles"] if p.get("name") == target]
            if not entries:
                raise ConfigError(f"no profile named {target!r} in {path}")
            entries[0]["auto_compact"] = auto_compact
        elif "provider" in data:
            data["provider"]["auto_compact"] = auto_compact
        else:
            data["auto_compact"] = auto_compact
    try:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not write {path}: {exc}") from exc


def load_all_profiles(path: Path) -> dict[str, ProviderProfile]:
    """Every profile in the file, keyed by name ("default" for the flat form)."""
    data = _normalize(path, _read(path))
    return {p.name: p for p in _validate(path, list[ProviderProfile], data["profiles"])}


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
