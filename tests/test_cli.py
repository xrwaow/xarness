"""CLI flag parsing: model/sandbox/preset selection and tri-state overrides."""

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
