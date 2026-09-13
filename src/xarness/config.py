"""Configuration loading and validation.

Config file layout (YAML), default path ``~/.config/xarness/config.yaml``:

Named profiles (switch with ``--profile``)::

    default_profile: openai
    profiles:
      openai:
        base_url: https://api.openai.com/v1
        api_key_env: OPENAI_API_KEY
        model_id: gpt-4.1
        shown_name: GPT-4.1
        max_context: 128000
        cot_strength: medium

Flat single-profile form (no ``profiles:`` key)::

    provider:
      base_url: https://api.openai.com/v1
      ...

The API key itself is never stored in the config; ``api_key_env`` names the
environment variable that holds it. Set ``api_key_env`` to null (or "") for
providers that need no key (e.g. local models).
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DEFAULT_CONFIG_PATH = Path("~/.config/xarness/config.yaml").expanduser()


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
    shown_name: str | None = None
    max_context: int = Field(default=128_000, gt=0)
    cot_strength: CotStrength = CotStrength.MEDIUM

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
        if value is not None and not value.strip():
            return None
        return value

    @field_validator("model_id", "shown_name")
    @classmethod
    def _validate_non_empty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be empty")
        return value

    @property
    def display_name(self) -> str:
        """Name shown in the UI, independent of the wire ``model_id``."""
        return self.shown_name or self.model_id


class _NamedConfig(BaseModel):
    """Top-level shape when the file uses named profiles."""

    model_config = ConfigDict(extra="forbid")

    default_profile: str | None = None
    profiles: dict[str, ProviderProfile]


def _read_config_data(path: Path) -> dict[str, Any]:
    """Read and parse the config file into a top-level mapping."""
    if not path.exists():
        raise ConfigError(
            f"config file not found: {path}\n"
            "  copy config.example.yaml from the repo there to get started"
        )
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {path} as YAML:\n{exc}") from exc

    if data is None:
        raise ConfigError(f"{path} is empty")
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return data


def load_config(path: Path, profile_name: str | None = None) -> ProviderProfile:
    """Load and validate the config file, returning the selected profile.

    Selection order: explicit ``--profile`` > ``default_profile`` > the only
    profile when exactly one is defined.
    """
    data = _read_config_data(path)

    if isinstance(data.get("profiles"), dict):
        return _from_named_profiles(data, path, profile_name)
    return _from_flat(data, path, profile_name)


def list_profile_names(path: Path) -> list[str]:
    """List the profile names defined in the config file, without resolving one.

    Returns an empty list for the flat single-profile form (nothing to switch
    between). Raises :class:`ConfigError` for a missing/malformed file, and for
    a named-profiles section that fails validation.
    """
    data = _read_config_data(path)
    if not isinstance(data.get("profiles"), dict):
        return []
    try:
        config = _NamedConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid config in {path}:\n{_format_validation_error(exc)}") from exc
    return list(config.profiles)


def _from_named_profiles(data: dict[str, Any], path: Path, profile_name: str | None) -> ProviderProfile:
    stray = [key for key in data if key not in ("default_profile", "profiles")]
    if stray:
        raise ConfigError(
            f"{path} mixes a 'profiles:' section with top-level fields {stray}; use one style or the other"
        )
    try:
        config = _NamedConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid config in {path}:\n{_format_validation_error(exc)}") from exc

    names = list(config.profiles)
    selected = profile_name or config.default_profile
    if selected is None:
        if len(names) == 1:
            selected = names[0]
        else:
            raise ConfigError(
                f"{path} defines multiple profiles ({', '.join(names)}) but no 'default_profile'; "
                "pass --profile to choose one"
            )
    if selected not in config.profiles:
        raise ConfigError(
            f"no profile named {selected!r} in {path}; available profiles: {', '.join(names)}"
        )
    return config.profiles[selected]


def _from_flat(data: dict[str, Any], path: Path, profile_name: str | None) -> ProviderProfile:
    if profile_name is not None:
        raise ConfigError(
            f"{path} defines a single unnamed profile, so --profile {profile_name!r} has nothing to "
            "select; add a top-level 'profiles:' section to use named profiles"
        )
    if "default_profile" in data:
        raise ConfigError(f"'default_profile' in {path} is only meaningful alongside a 'profiles:' section")

    inner: Any = data.get("provider", data)
    if not isinstance(inner, dict):
        raise ConfigError(f"'provider' in {path} must be a mapping of profile fields")

    try:
        return ProviderProfile.model_validate(inner)
    except ValidationError as exc:
        raise ConfigError(f"invalid config in {path}:\n{_format_validation_error(exc)}") from exc


def _format_validation_error(exc: ValidationError) -> str:
    """Render pydantic errors as one specific line per problem, no traceback."""
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        value_repr = repr(error.get("input"))
        if len(value_repr) > 60:
            value_repr = value_repr[:57] + "..."
        lines.append(f"  - {location}: {error['msg']} (got {value_repr})")
    return "\n".join(lines)


def resolve_api_key(profile: ProviderProfile) -> str | None:
    """Read the API key from the env var named by ``api_key_env``.

    Returns None when the profile needs no key (``api_key_env`` is null).
    """
    if profile.api_key_env is None:
        return None
    key = os.environ.get(profile.api_key_env)
    if not key:
        raise ConfigError(
            f"environment variable {profile.api_key_env!r} (api_key_env from config) is not set; "
            "the API key is read from the environment, never from the config file"
        )
    return key

def resolve_profile_name(path: Path, profile_name: str | None = None) -> str:
    """Return the name of the profile load_config would select, without full
    validation — used to label the active profile in the /model picker."""
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    if isinstance(data.get("profiles"), dict):
        names = list(data["profiles"])
        selected = profile_name or data.get("default_profile")
        if selected is None:
            if len(names) == 1:
                return names[0]
            raise ConfigError(f"{path} defines multiple profiles but no 'default_profile'")
        return selected
    return "default"


def load_all_profiles(path: Path) -> dict[str, ProviderProfile]:
    """Return every profile in the config file, keyed by name.

    A flat single-profile config is keyed "default".
    """
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {path} as YAML:\n{exc}") from exc
    if data is None:
        raise ConfigError(f"{path} is empty")
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")

    if isinstance(data.get("profiles"), dict):
        try:
            config = _NamedConfig.model_validate(data)
        except ValidationError as exc:
            raise ConfigError(f"invalid config in {path}:\n{_format_validation_error(exc)}") from exc
        return dict(config.profiles)

    inner: Any = data.get("provider", data)
    if not isinstance(inner, dict):
        raise ConfigError(f"'provider' in {path} must be a mapping of profile fields")
    try:
        return {"default": ProviderProfile.model_validate(inner)}
    except ValidationError as exc:
        raise ConfigError(f"invalid config in {path}:\n{_format_validation_error(exc)}") from exc
