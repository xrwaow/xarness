"""Tests for config loading and validation."""

from pathlib import Path

import pytest

from agentcli.config import ConfigError, CotStrength, load_config


def write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_flat_provider_form(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
provider:
  base_url: "https://api.openai.com/v1/"
  api_key_env: "MY_KEY"
  model_id: "gpt-4.1"
  shown_name: "GPT-4.1"
  max_context: 128000
  cot_strength: "medium"
""",
    )
    profile = load_config(path)
    assert profile.base_url == "https://api.openai.com/v1"  # trailing slash trimmed
    assert profile.api_key_env == "MY_KEY"
    assert profile.model_id == "gpt-4.1"
    assert profile.display_name == "GPT-4.1"
    assert profile.max_context == 128_000
    assert profile.cot_strength is CotStrength.MEDIUM


def test_bare_flat_form_without_provider_key(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
base_url: "https://api.openai.com/v1"
model_id: "gpt-4.1"
""",
    )
    profile = load_config(path)
    assert profile.model_id == "gpt-4.1"
    assert profile.cot_strength is CotStrength.MEDIUM  # default


def test_named_profiles_with_default(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
default_profile: b
profiles:
  a:
    base_url: "https://a.example/v1"
    model_id: "model-a"
  b:
    base_url: "https://b.example/v1"
    model_id: "model-b"
""",
    )
    assert load_config(path).model_id == "model-b"
    assert load_config(path, profile_name="a").model_id == "model-a"


def test_single_named_profile_selected_without_default(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
profiles:
  only:
    base_url: "https://x.example/v1"
    model_id: "model-x"
""",
    )
    assert load_config(path).model_id == "model-x"


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(tmp_path / "nope.yaml")


def test_malformed_yaml_reports_file(tmp_path: Path) -> None:
    path = write(tmp_path, "provider: [unclosed")
    with pytest.raises(ConfigError, match="could not parse"):
        load_config(path)


def test_invalid_field_names_the_field(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
provider:
  base_url: "https://api.openai.com/v1"
  model_id: "gpt-4.1"
  cot_strength: "maximum"
""",
    )
    with pytest.raises(ConfigError, match="cot_strength") as excinfo:
        load_config(path)
    # No raw pydantic traceback text in the message.
    assert "Traceback" not in str(excinfo.value)


def test_missing_required_field_is_specific(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
provider:
  base_url: "https://api.openai.com/v1"
""",
    )
    with pytest.raises(ConfigError, match="model_id"):
        load_config(path)


def test_unknown_profile_lists_available(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
profiles:
  a:
    base_url: "https://a.example/v1"
    model_id: "model-a"
""",
    )
    with pytest.raises(ConfigError, match="available profiles: a"):
        load_config(path, profile_name="nope")


def test_profile_flag_on_flat_config_is_an_error(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
provider:
  base_url: "https://api.openai.com/v1"
  model_id: "gpt-4.1"
""",
    )
    with pytest.raises(ConfigError, match="single unnamed profile"):
        load_config(path, profile_name="anything")


def test_mixed_styles_rejected(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
provider:
  base_url: "https://api.openai.com/v1"
  model_id: "gpt-4.1"
profiles:
  a:
    base_url: "https://a.example/v1"
    model_id: "model-a"
""",
    )
    with pytest.raises(ConfigError, match="one style or the other"):
        load_config(path)


def test_negative_max_context_rejected(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
provider:
  base_url: "https://api.openai.com/v1"
  model_id: "gpt-4.1"
  max_context: -5
""",
    )
    with pytest.raises(ConfigError, match="max_context"):
        load_config(path)


def test_typo_field_is_reported(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
provider:
  base_url: "https://api.openai.com/v1"
  model: "gpt-4.1"  # typo: should be model_id
""",
    )
    with pytest.raises(ConfigError, match="model"):
        load_config(path)
