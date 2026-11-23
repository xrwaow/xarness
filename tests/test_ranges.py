"""The ``range`` window syntax shared by the list routes."""

from __future__ import annotations

import pytest

from xarness.conversation import AssistantMessage, ToolMessage, user_message
from xarness.util import RangeError, parse_range
from xarness.session import history_page


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (None, (0, 10)),
        ("", (0, 10)),
        ("[:]", (0, 10)),
        ("[-3:]", (7, 10)),
        ("[2:]", (2, 10)),
        ("[:4]", (0, 4)),
        ("[2:8]", (2, 8)),
        ("[-4:-1]", (6, 9)),
        ("[-4:8]", (6, 8)),
        (" [ 2 : 8 ] ", (2, 8)),
        # Out-of-range bounds clamp instead of failing.
        ("[-99:]", (0, 10)),
        ("[99:]", (10, 10)),
        ("[:99]", (0, 10)),
        ("[8:2]", (8, 8)),
        ("[3:3]", (3, 3)),
    ],
)
def test_parse_range_follows_python_slice_semantics(
    spec: str | None, expected: tuple[int, int],
) -> None:
    assert parse_range(spec, 10) == expected


@pytest.mark.parametrize("spec", ["[1:2:3]", "[", "[1]", "[a:]", "[1:b]"])
def test_parse_range_rejects_a_malformed_window(spec: str) -> None:
    with pytest.raises(RangeError):
        parse_range(spec, 10)


def test_history_page_window_starts_at_the_turn_boundary() -> None:
    """A window that lands mid-turn grows back to the user message that
    opened it, so a tool call keeps its result."""
    messages = [
        *[user_message("q0"), AssistantMessage(content="a0")],
        user_message("q1"),
        AssistantMessage(content="a1"),
        ToolMessage(tool_call_id="c1", content="ok"),
    ]
    # The last three messages are one turn; asking for its tail returns all
    # of it.
    start, end = history_page(messages, window="[-1:]")
    assert (start, end) == (2, 5)
    # An explicit empty window stays empty (no snap-back).
    assert history_page(messages, window="[5:5]") == (5, 5)
    assert history_page(messages, window="[:]") == (0, 5)
