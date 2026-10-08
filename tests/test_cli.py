"""CLI flag parsing: --network / --gpu launch-time booleans (default true)."""

from xarness.cli import _flag_value, build_parser


def test_network_and_gpu_default_true() -> None:
    args = build_parser().parse_args([])
    assert _flag_value(args.network) is True
    assert _flag_value(args.gpu) is True


def test_flags_accept_false() -> None:
    args = build_parser().parse_args(["--network", "false", "--gpu", "false"])
    assert _flag_value(args.network) is False
    assert _flag_value(args.gpu) is False


def test_flags_collect_lists_last_wins() -> None:
    args = build_parser().parse_args(
        ["--network", "false", "--network", "true", "--gpu", "false"],
    )
    assert args.network == ["false", "true"]
    assert _flag_value(args.network) is True
    assert args.gpu == ["false"]
    assert _flag_value(args.gpu) is False


def test_flag_value_nonstandard_strings() -> None:
    assert _flag_value(["TRUE"]) is True
    assert _flag_value(["0"]) is False
    assert _flag_value(["no"]) is False
    assert _flag_value([]) is True
