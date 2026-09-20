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

DEFAULT_THEME = "carbonfox"


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


@dataclass(frozen=True)
class LoadedConfig:
    """Everything load_config derives from one config file read."""

    profile: ProviderProfile
    # Name of the selected profile ("default" for the flat form).
    profile_name: str | None
    # All profile names in the file ([] for the flat form).
    profile_names: list[str]
    default_theme: str


def _read(path: Path) -> dict[str, Any]:
    """Read and parse the config file into a top-level object."""
    if not path.exists():
        raise ConfigError(
            f"config file not found: {path}\n"
            "  copy config.example.json from the repo there to get started"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"could not parse {path} as JSON:\n{exc}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a JSON object at the top level")
    return data


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
    stray = [key for key in data if key not in ("default_profile", "default_theme", "profiles")]
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


def _flat_profile(path: Path, data: dict[str, Any]) -> ProviderProfile:
    """Validate the flat single-profile form."""
    fields = data["provider"] if "provider" in data else {
        key: value for key, value in data.items()
        if key not in ("default_profile", "default_theme", "profiles", "provider")
    }
    if "default_profile" in data:
        raise ConfigError(f"'default_profile' in {path} is only meaningful alongside a 'profiles' section")
    return _validate(path, ProviderProfile, fields)


def load_config(path: Path, profile_name: str | None = None) -> LoadedConfig:
    """Load and validate the config file.

    Profile selection order: explicit ``--profile`` > ``default_profile`` >
    the only profile when exactly one is defined.
    """
    data = _read(path)

    if isinstance(data.get("profiles"), list):
        profiles = _named_profiles(path, data)
        names = [p.name for p in profiles]
        selected = profile_name or data.get("default_profile") or (names[0] if len(names) == 1 else None)
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
        return LoadedConfig(by_name[selected], selected, names, data.get("default_theme") or DEFAULT_THEME)

    if profile_name is not None:
        raise ConfigError(
            f"{path} defines a single unnamed profile, so --profile {profile_name!r} has nothing to "
            "select; add a top-level 'profiles' section to use named profiles"
        )
    return LoadedConfig(
        _flat_profile(path, data), "default", [], data.get("default_theme") or DEFAULT_THEME
    )


def list_profile_names(path: Path) -> list[str]:
    """Profile names in the file ([] for the flat single-profile form)."""
    data = _read(path)
    if not isinstance(data.get("profiles"), list):
        return []
    return [p.name for p in _named_profiles(path, data)]


def load_all_profiles(path: Path) -> dict[str, ProviderProfile]:
    """Every profile in the file, keyed by name ("default" for the flat form)."""
    data = _read(path)
    if isinstance(data.get("profiles"), list):
        return {p.name: p for p in _named_profiles(path, data)}
    return {"default": _flat_profile(path, data)}


def resolve_api_key(profile: ProviderProfile) -> str | None:
    """Read the API key from the env var named by ``api_key_env``."""
    if profile.api_key_env is None:
        return None
    key = os.environ.get(profile.api_key_env)
    if not key:
        raise ConfigError(
            f"environment variable {profile.api_key_env!r} (api_key_env from config) is not set; "
            "the API key is read from the environment, never from the config file"
        )
    return key
