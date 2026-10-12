"""Tests for config loading and validation."""

import json
from pathlib import Path

import pytest

from xarness.config import (
    ConfigError, CotStrength, RefSpec, SectionedConfig, SandboxSettings,
    load_sectioned, resolve_api_key, save_preferences,
)


def write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def minimal_model() -> dict:
    return {"base_url": "https://api.openai.com/v1", "model_id": "gpt-4.1"}


def test_profile_fields_parsed(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        {"models": {"a": {
            "base_url": "https://api.openai.com/v1/",
            "api_key_env": "MY_KEY",
            "model_id": "gpt-4.1",
            "shown_name": "GPT-4.1",
            "max_context": 128000,
            "cot_strength": "medium",
        }}},
    )
    profile = load_sectioned(path).models["a"]
    assert profile.base_url == "https://api.openai.com/v1"  # trailing slash trimmed
    assert profile.api_key_env == "MY_KEY"
    assert profile.model_id == "gpt-4.1"
    assert profile.display_name == "GPT-4.1"
    assert profile.max_context == 128_000
    assert profile.cot_strength is CotStrength.MEDIUM
    assert profile.name == "a"  # implied by the section key


def test_null_api_key_env_needs_no_key(tmp_path: Path) -> None:
    path = write(tmp_path, {"models": {"a": {
        "base_url": "http://127.0.0.1:8080/v1",
        "api_key_env": None,
        "model_id": "qwen",
    }}})
    profile = load_sectioned(path).models["a"]
    assert profile.api_key_env is None
    assert resolve_api_key(profile) is None


def test_empty_api_key_env_is_normalized_to_none(tmp_path: Path) -> None:
    path = write(tmp_path, {"models": {"a": {
        "base_url": "http://127.0.0.1:8080/v1",
        "api_key_env": "",
        "model_id": "qwen",
    }}})
    assert load_sectioned(path).models["a"].api_key_env is None


def test_resolve_api_key_reads_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write(tmp_path, {"models": {"a": {**minimal_model(), "api_key_env": "MY_KEY"}}})
    profile = load_sectioned(path).models["a"]
    monkeypatch.setenv("MY_KEY", "sk-test")
    assert resolve_api_key(profile) == "sk-test"
    monkeypatch.delenv("MY_KEY")
    with pytest.raises(ConfigError, match="MY_KEY"):
        resolve_api_key(profile)


def test_resolve_model_by_default_preset(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        {
            "models": {
                "a": {"base_url": "https://a.example/v1", "model_id": "model-a"},
                "b": {"base_url": "https://b.example/v1", "model_id": "model-b"},
            },
            "presets": {"default": {"model": "b"}},
        },
    )
    sectioned = load_sectioned(path)
    assert sectioned.resolve_model(None)[0] == "b"
    assert sectioned.resolve_model("a")[0] == "a"


def test_resolve_model_errors_list_available(tmp_path: Path) -> None:
    path = write(tmp_path, {"models": {
        "a": {"base_url": "https://a.example/v1", "model_id": "model-a"},
    }})
    with pytest.raises(ConfigError, match="available: a"):
        load_sectioned(path).resolve_model("nope")


def test_resolve_model_no_models_says_none(tmp_path: Path) -> None:
    path = write(tmp_path, {})
    with pytest.raises(ConfigError, match="available: none"):
        load_sectioned(path).resolve_model(None)


def test_default_theme(tmp_path: Path) -> None:
    path = write(tmp_path, {"ui": {"theme": "one-light"}})
    assert load_sectioned(path).theme == "one-light"
    path2 = write(tmp_path, {})
    assert load_sectioned(path2).theme == "ayu-darker"  # default


def test_save_preferences(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        {
            "models": {
                "a": {"base_url": "https://x.test/v1", "model_id": "m"},
                "b": {"base_url": "https://y.test/v1", "model_id": "m2"},
            },
            "presets": {"default": {"model": "a"}},
        },
    )
    save_preferences(path, default_theme="one-light", default_profile="b")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["ui"]["theme"] == "one-light"
    assert load_sectioned(path).resolve_model(None)[0] == "b"


def test_save_preferences_prefers_matching_preset(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        {
            "models": {
                "a": {"base_url": "https://api.example.test/v1", "model_id": "a"},
                "b": {"base_url": "https://api.example.test/v1", "model_id": "b"},
            },
            "presets": {"default": {"model": "a"}, "fast": {"model": "b"}},
            "defaults": {"preset": "default"},
        },
    )
    save_preferences(path, default_profile="b")
    data = json.loads(path.read_text())
    assert data["defaults"]["preset"] == "fast"
    assert load_sectioned(path).default_preset_model() == "b"


def test_save_preferences_auto_compact(tmp_path: Path) -> None:
    path = write(tmp_path, {"models": {
        "a": {"base_url": "https://x.test/v1", "model_id": "m"},
    }})
    save_preferences(path, auto_compact=True)
    data = json.loads(path.read_text())
    assert data["models"]["a"]["auto_compact"] is True


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_sectioned(tmp_path / "nope.json")


def test_malformed_json_reports_file(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="could not parse"):
        load_sectioned(path)


def test_invalid_field_names_the_field(tmp_path: Path) -> None:
    path = write(tmp_path, {"models": {"a": {**minimal_model(), "cot_strength": "maximum"}}})
    with pytest.raises(ConfigError, match="cot_strength") as excinfo:
        load_sectioned(path)
    # No raw pydantic traceback text in the message.
    assert "Traceback" not in str(excinfo.value)


def test_missing_required_field_is_specific(tmp_path: Path) -> None:
    path = write(tmp_path, {"models": {"a": {"base_url": "https://api.openai.com/v1"}}})
    with pytest.raises(ConfigError, match="model_id"):
        load_sectioned(path)


def test_negative_max_context_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, {"models": {"a": {**minimal_model(), "max_context": -5}}})
    with pytest.raises(ConfigError, match="max_context"):
        load_sectioned(path)


def test_typo_field_is_reported(tmp_path: Path) -> None:
    path = write(tmp_path, {"models": {"a": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4.1",  # typo: should be model_id
    }}})
    with pytest.raises(ConfigError, match="model"):
        load_sectioned(path)


def test_auto_compact_defaults(tmp_path: Path) -> None:
    path = write(tmp_path, {"models": {"a": minimal_model()}})
    profile = load_sectioned(path).models["a"]
    assert profile.auto_compact is False
    assert profile.auto_compact_threshold == 0.9


def test_auto_compact_options_honored(tmp_path: Path) -> None:
    path = write(tmp_path, {"models": {"a": {
        **minimal_model(), "auto_compact": True, "auto_compact_threshold": 0.75,
    }}})
    profile = load_sectioned(path).models["a"]
    assert profile.auto_compact is True
    assert profile.auto_compact_threshold == 0.75


@pytest.mark.parametrize("threshold", [0, -0.5, 1.5])
def test_bad_auto_compact_threshold_rejected(tmp_path: Path, threshold: float) -> None:
    path = write(tmp_path, {"models": {"a": {
        **minimal_model(), "auto_compact_threshold": threshold,
    }}})
    with pytest.raises(ConfigError, match="auto_compact_threshold"):
        load_sectioned(path)


def test_sandbox_settings_defaults() -> None:
    settings = SandboxSettings()
    assert settings.allow_network is True
    assert settings.gpu_access is True
    assert settings.allow_git_write is False  # secure by default
    assert settings.use_container is True
    assert settings.enable_tracking is True
    assert settings.auto_include_refs == []


def test_sandbox_section_parsed(tmp_path: Path) -> None:
    path = write(tmp_path, {
        "sandboxes": {"default": {"tool_output_limit": 4096}},
    })
    _, settings = load_sectioned(path).resolve_sandbox(None)
    assert settings.tool_output_limit == 4096


def test_sandbox_inheritance_over_default(tmp_path: Path) -> None:
    path = write(tmp_path, {
        "sandboxes": {
            "default": {"allow_network": False},
            "fast": {"gpu_access": False},
        },
    })
    _, settings = load_sectioned(path).resolve_sandbox("fast")
    assert settings.allow_network is False  # inherited
    assert settings.gpu_access is False  # overridden


def test_sandbox_ref_with_dotdot_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, {
        "sandboxes": {"default": {"auto_include_refs": ["../escape"]}},
    })
    with pytest.raises(ConfigError, match=r"\.\."):
        load_sectioned(path)


def test_ref_spec_entries(tmp_path: Path) -> None:
    path = write(tmp_path, {
        "sandboxes": {"default": {"auto_include_refs": [
            "~/notes/api.md",
            {"path": "docs/spec.md", "mount": ".venv", "read_only": False},
        ]}},
    })
    refs = load_sectioned(path).resolve_sandbox(None)[1].auto_include_refs
    assert refs[0] == RefSpec(path="~/notes/api.md")
    spec = refs[1]
    assert spec.path == "docs/spec.md"
    assert spec.mount == ".venv"
    assert spec.read_only is False


def test_ref_spec_bad_mount_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, {
        "sandboxes": {"default": {"auto_include_refs": [
            {"path": "x", "mount": "../escape"},
        ]}},
    })
    with pytest.raises(ConfigError, match="mount"):
        load_sectioned(path)


def test_sandbox_unknown_key_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, {
        "sandboxes": {"default": {"netwrok": True}},
    })
    with pytest.raises(ConfigError, match="netwrok"):
        load_sectioned(path)


def test_resolve_sandbox_error_says_none_when_no_profiles() -> None:
    sectioned = SectionedConfig()
    with pytest.raises(ConfigError) as exc:
        sectioned.resolve_sandbox(None)
    assert "available: none" in str(exc.value)


def test_unknown_server_section_is_ignored(tmp_path: Path) -> None:
    """The old "server" section no longer exists; unknown sections don't
    break config loading."""
    path = write(tmp_path, {"server": {"port": 8080}})
    assert load_sectioned(path).models == {}
