"""Tests for the tool registry: dispatch, failure containment, schema."""

import asyncio

from xarness.tools import Tool, ToolRegistry, ToolResult, build_registry


async def _ok(args: dict) -> ToolResult:
    return ToolResult(ok=True, output=f"ran with {sorted(args)}")


async def _boom(args: dict) -> ToolResult:
    raise RuntimeError("kaboom")


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool(name="ok_tool", description="", parameters_schema={}, handler=_ok))
    registry.register(Tool(name="boom_tool", description="", parameters_schema={}, handler=_boom))
    return registry


def test_schema_matches_openai_tools_format() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="test_tool",
            description="does things",
            parameters_schema={"type": "object", "properties": {}},
            handler=_ok,
        )
    )

    assert registry.schema() == [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "does things",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_call_success_parses_arguments() -> None:
    result = asyncio.run(ToolRegistry.call(_registry(), "ok_tool", '{"b": 1, "a": 2}'))
    assert result.ok
    assert result.output == "ran with ['a', 'b']"


def test_empty_arguments_become_empty_dict() -> None:
    result = asyncio.run(ToolRegistry.call(_registry(), "ok_tool", ""))
    assert result.ok


def test_invalid_json_is_a_parse_error() -> None:
    result = asyncio.run(ToolRegistry.call(_registry(), "ok_tool", '{"a": '))
    assert not result.ok
    assert result.parse_error is True
    assert "invalid arguments JSON" in result.error


def test_non_object_arguments_are_a_parse_error() -> None:
    result = asyncio.run(ToolRegistry.call(_registry(), "ok_tool", "[1, 2]"))
    assert not result.ok
    assert result.parse_error is True


def test_unknown_tool_fails_without_raising() -> None:
    result = asyncio.run(ToolRegistry.call(_registry(), "nope", "{}"))
    assert not result.ok
    assert result.parse_error is False
    assert "unknown tool" in result.error


def test_handler_exception_is_contained() -> None:
    result = asyncio.run(ToolRegistry.call(_registry(), "boom_tool", "{}"))
    assert not result.ok
    assert "RuntimeError: kaboom" in result.error


def test_write_registry_has_test_tool() -> None:
    registry = build_registry(None, None, mode="write")
    result = asyncio.run(registry.call("test_tool", "{}"))
    assert result.ok
    assert result.output == "success!"