"""CLI flag parsing: model/sandbox/preset selection and the boolean
overrides (--sandbox/--network/--gpu/--enable_tracking)."""

import pytest

from xarness import cli
from xarness.cli import build_parser


def test_model_sandbox_preset_flags() -> None:
    args = build_parser().parse_args(
        ["--model", "fast", "--sandbox", "gpu", "--preset", "ml"]
    )
    assert args.model == "fast"
    assert args.sandbox == "gpu"
    assert args.use_sandbox is None  # a profile name names the container mode
    assert args.preset == "ml"


def test_sandbox_boolean_overrides_without_a_profile_name() -> None:
    args = build_parser().parse_args(["--sandbox", "true"])
    assert args.sandbox is None  # default profile resolution still applies
    assert args.use_sandbox is True
    args = build_parser().parse_args(["--sandbox", "false"])
    assert args.sandbox is None
    assert args.use_sandbox is False


def test_bare_sandbox_is_an_error() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--sandbox"])


def test_profile_is_an_alias_for_model() -> None:
    args = build_parser().parse_args(["--profile", "fast"])
    assert args.model == "fast"


def test_boolean_overrides_default_to_none() -> None:
    args = build_parser().parse_args([])
    assert args.network is None
    assert args.gpu is None
    assert args.enable_tracking is None
    assert args.sandbox is None
    assert args.use_sandbox is None


@pytest.mark.parametrize("word,expected", [
    ("true", True), ("TRUE", True), ("1", True), ("yes", True), ("on", True),
    ("false", False), ("False", False), ("0", False), ("no", False), ("off", False),
])
def test_boolean_flags_take_an_explicit_value(word: str, expected: bool) -> None:
    args = build_parser().parse_args(
        ["--network", word, "--gpu", word, "--enable_tracking", word]
    )
    assert args.network is expected
    assert args.gpu is expected
    assert args.enable_tracking is expected


def test_boolean_flags_reject_junk_and_need_a_value() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--network", "maybe"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--gpu"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--enable_tracking"])


def test_session_and_server_flags() -> None:
    args = build_parser().parse_args(["--session", "2026-01-01_x"])
    assert args.session == "2026-01-01_x"
    assert args.sessions is False
    assert args.server is False
    args = build_parser().parse_args(["--sessions"])
    assert args.sessions is True
    args = build_parser().parse_args(["--server"])
    assert args.server is True


def test_chat_startup_creates_a_starter_config(tmp_path, monkeypatch) -> None:
    from xarness.tui import app as tui_app

    config_path = tmp_path / ".config" / "xarness" / "config.json"
    captured: dict = {}

    class FakeApp:
        def __init__(self, *args, **kwargs) -> None:
            captured.update(kwargs)

        def run(self) -> None:
            pass

    monkeypatch.setattr(cli, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(tui_app, "AgentApp", FakeApp)
    cli._run_chat(build_parser().parse_args(["--workspace", str(tmp_path / "ws")]))
    assert config_path.exists() and captured["startup_notices"]
