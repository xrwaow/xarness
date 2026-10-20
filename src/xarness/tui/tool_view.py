"""Per-tool presentation: live label, settled header detail and body.

Each tool name needing special display is described once in ``TOOL_VIEWS``;
the transcript item and the live labels look the result up instead of
branching on the tool name in several places.
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from rich.console import RenderableType
from rich.text import Text

from .. import theme
from .widgets import (
    _render_diff, _render_read_file, _split_tool_diff, highlight_code,
)


def _shorten(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def json_tool_args(args_text: str) -> dict:
    """Best-effort parse of a tool call's streamed arguments JSON."""
    try:
        parsed = json.loads(args_text)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _partial_strings(args_text: str, key: str, unescape: bool = False) -> list[str]:
    """Best-effort extraction of every still-streaming ``key`` string value.

    Used for the live edit summary: a partially streamed ``edits`` array is
    not valid JSON yet, but its old/new strings can already be counted."""
    values: list[str] = []
    for match in re.finditer(rf'"{re.escape(key)}"\s*:\s*"', args_text):
        rest, i = args_text[match.end():], 0
        while i < len(rest):
            if rest[i] == "\\":
                i += 2
                continue
            if rest[i] == '"':
                rest = rest[:i]
                break
            i += 1
        rest = rest.rstrip("\\")
        if unescape:
            rest = rest.replace('\\"', '"').replace("\\\\", "\\").replace("\\n", "\n")
        values.append(rest)
    return values


def _partial_string(args_text: str, key: str, unescape: bool = False) -> str | None:
    """Best-effort extraction of a still-streaming string argument."""
    values = _partial_strings(args_text, key, unescape=unescape)
    return values[0] if values else None


def _lines(text: str) -> int:
    return text.count("\n") + (0 if text.endswith("\n") else 1) if text else 0


def _path(args_text: str) -> str:
    value = json_tool_args(args_text).get("path")
    if isinstance(value, str):
        return value
    return _partial_string(args_text, "path") or ""


def _command(args_text: str, unescape: bool = True) -> str:
    args = json_tool_args(args_text)
    value = args.get("command")
    if isinstance(value, str):
        return value
    return _partial_string(args_text, "command", unescape=unescape) or ""


def _muted(value: str) -> Text:
    return Text(value, style=theme.PALETTE["muted"])


def _write_loc(args_text: str) -> int | None:
    content = json_tool_args(args_text).get("content")
    if isinstance(content, str):
        return _lines(content)
    raw = _partial_string(args_text, "content")
    return None if raw is None else _lines(raw.replace("\\\\", "").replace("\\n", "\n"))


def _edit_counts(args_text: str) -> str | None:
    args = json_tool_args(args_text)
    edits = args.get("edits")
    if not isinstance(edits, list):
        old, new = args.get("old_string"), args.get("new_string")
        if not isinstance(old, str) or not old:
            # Still streaming: JSON.parse can't see the array yet. Salvage
            # every old/new string so the live label counts as it grows.
            olds = _partial_strings(args_text, "old_string", unescape=True)
            if not olds:
                return None
            news = _partial_strings(args_text, "new_string", unescape=True)
            edits = [
                {"old_string": old, "new_string": news[i] if i < len(news) else ""}
                for i, old in enumerate(olds)
            ]
        else:
            edits = [{
                "old_string": old,
                "new_string": new if isinstance(new, str) else "",
            }]
    adds = dels = 0
    for edit in edits:
        if not isinstance(edit, dict) or not isinstance(edit.get("old_string"), str):
            continue
        old, new = edit["old_string"], edit.get("new_string", "")
        if not old:
            continue
        diff = list(difflib.unified_diff(
            old.splitlines(), new.splitlines() if isinstance(new, str) else [],
            n=0, lineterm="",
        ))
        adds += sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
        dels += sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
    return f"+{adds} -{dels}" if adds or dels else None


@dataclass(frozen=True)
class ToolCallView:
    """The data one tool presentation needs (no widget/entry state)."""

    name: str
    args: str
    output: str = ""
    header: str = ""
    status: object = None
    expanded: bool = False


# live(args) -> (tool name, static detail); header(stored header) and
# settled(args) -> dimmed detail Text; body(call) -> expanded renderable.
ToolView = tuple[
    Callable[[str], tuple[str, str]],
    Callable[[str], Text | None],
    Callable[[str], Text | None],
    Callable[[ToolCallView], RenderableType],
]


# -- shared pieces ---------------------------------------------------------

_COUNTS = {
    "edit_file": re.compile(r"^(\+\d+) (-\d+) (.+)$"),
    "write_file": re.compile(r"^(\+\d+) (.+)$"),
}


def _count_header(tool_name: str, header: str) -> Text | None:
    pattern = _COUNTS.get(tool_name)
    match = pattern.match(header) if pattern else None
    if match is None:
        return None
    *counts, path = match.groups()
    detail = Text(" ")
    for value, key in zip(counts, ("diff_add", "diff_del")):
        detail.append(value, style=theme.PALETTE[key])
        detail.append(" ")
    detail.append(path, style=theme.PALETTE["muted"])
    return detail


def _path_settled(args_text: str) -> Text | None:
    path = _path(args_text)
    return None if not path else _muted(" " + _shorten(path))


def _with_output(body: str, output: str) -> RenderableType:
    if output:
        body = f"{body}\n{output}" if body else output
    split = _split_tool_diff(body)
    if split is None:
        return Text(body)
    prose, diff = split
    rendered = Text(prose.rstrip("\n"))
    if rendered:
        rendered.append("\n")
    rendered.append_text(_render_diff(diff))
    return rendered


# -- per-tool views --------------------------------------------------------

def _write_live(args_text: str) -> tuple[str, str]:
    loc = _write_loc(args_text)
    return "write_file", " ".join(
        p for p in (f"+{loc}" if loc is not None else "", _path(args_text)) if p
    )


def _edit_live(args_text: str) -> tuple[str, str]:
    return "edit_file", " ".join(
        p for p in (_edit_counts(args_text) or "", _path(args_text)) if p
    )


def _read_live(args_text: str) -> tuple[str, str]:
    return "read_file", _path(args_text)


def _read_settled(args_text: str) -> Text | None:
    path = _path(args_text)
    if not path:
        return None
    args, detail = json_tool_args(args_text), Text(" ")
    detail.append(_shorten(path), style=theme.PALETTE["muted"])
    ranges = [
        f"{key}={args[key]}" for key in ("start_line", "end_line")
        if args.get(key) is not None
    ]
    if ranges:
        detail.append(f" [{', '.join(ranges)}]", style=theme.PALETTE["muted"])
    return detail


def _read_body(call: ToolCallView) -> RenderableType:
    path = _path(call.args)
    if path and getattr(call.status, "name", "") == "CALL_SUCCEEDED":
        return _render_read_file(path, call.output)
    return _plain_body(call)


def _write_body(call: ToolCallView) -> RenderableType:
    args = json_tool_args(call.args)
    path, content = args.get("path"), args.get("content")
    if isinstance(path, str) and path and isinstance(content, str):
        body = highlight_code(content, path=path)
        if call.output:
            body.append("\n" + call.output)
        return body
    return _plain_body(call)


def _edit_body(call: ToolCallView) -> RenderableType:
    split = _split_tool_diff(call.output)
    return _render_diff(split[1], show_headers=False) if split else Text(call.output)


def _bash_body(call: ToolCallView) -> RenderableType:
    return _with_output(_command(call.args, unescape=False), call.output)


def _plain_body(call: ToolCallView) -> RenderableType:
    body = _command(call.args) if call.name == "run_bash" else call.args
    return _with_output(body, call.output)


def _bash_settled(args_text: str) -> Text | None:
    command = _command(args_text)
    return _muted(" " + _shorten(command.splitlines()[0], 80)) if command else None


def _plain_live(args_text: str) -> tuple[str, str]:
    return "tool", ""


def _bash_live(args_text: str) -> tuple[str, str]:
    command = _command(args_text)
    return "run_bash", command.splitlines()[0] if command else ""


_DEFAULT: ToolView = (_plain_live, _muted, lambda a: None, _plain_body)
TOOL_VIEWS: dict[str, ToolView] = {
    "write_file": (
        _write_live, lambda h: _count_header("write_file", h),
        _path_settled, _write_body,
    ),
    "edit_file": (
        _edit_live, lambda h: _count_header("edit_file", h),
        _path_settled, _edit_body,
    ),
    "read_file": (
        _read_live, lambda h: _muted(" " + _shorten(h)),
        _read_settled, _read_body,
    ),
    "run_bash": (
        _bash_live, lambda h: _muted(" " + _shorten(h, 80)),
        _bash_settled, _bash_body,
    ),
}


def live_parts(tool_name: str, args_text: str) -> tuple[str, str]:
    """The shimmering tool name and the static argument-derived detail."""
    name, detail = TOOL_VIEWS.get(tool_name, _DEFAULT)[0](args_text)
    return name, _shorten(detail) if detail else detail


def summary_text(call: ToolCallView) -> Text:
    """Settled one-line summary after the status verb."""
    if getattr(call.status, "name", "") != "CALL_SUCCEEDED":
        return Text(call.name)
    _, header, settled, _ = TOOL_VIEWS.get(call.name, _DEFAULT)
    detail = header(call.header) if call.header else None
    if detail is None:
        detail = settled(call.args)
    text = Text(call.name)
    if detail is not None:
        text.append_text(detail)
    return text


def body_renderable(call: ToolCallView) -> RenderableType:
    return TOOL_VIEWS.get(call.name, _DEFAULT)[3](call)
