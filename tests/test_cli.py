"""CLI flag parsing: model/sandbox/preset selection and tri-state overrides."""

from xarness import cli
from xarness.cli import build_parser


def test_model_sandbox_preset_flags() -> None:
    args = build_parser().parse_args(
        ["--model", "fast", "--sandbox", "gpu", "--preset", "ml"]
    )
    assert args.model == "fast"
    assert args.sandbox == "gpu"
    assert args.preset == "ml"


def test_profile_is_an_alias_for_model() -> None:
    args = build_parser().parse_args(["--profile", "fast"])
    assert args.model == "fast"


def test_network_gpu_default_none() -> None:
    args = build_parser().parse_args([])
    assert args.network is None
    assert args.gpu is None


def test_network_gpu_tri_state() -> None:
    args = build_parser().parse_args(["--network", "--no-gpu"])
    assert args.network is True
    assert args.gpu is False


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
