"""Unit adaptation of pi's anthropic-tool-name-normalization.test.ts.

pi's suite runs the round-trips against the live API with an OAuth token; the
normalization contract it documents is deterministic, so pidrei pins it at the
function and params-builder level instead:

1. Tool names matching Claude Code tools (case-insensitive) convert to CC
   casing outbound and back to the caller's casing inbound.
2. It's a case-insensitive *lookup*, never a mapping of different names
   (`find` must NOT become `Glob` — the old broken behavior).
"""

import pytest

from pidrei_ai.api.anthropic_messages import (
    AnthropicOptions,
    _build_params,
    _from_claude_code_name,
    _to_claude_code_name,
)
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, Tool, UserMessage
from tests.anthropic_helpers import now_ms


def make_tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"The {name} tool",
        parameters={"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]},
    )


def test_round_trips_user_tool_matching_cc_name():
    # todowrite -> TodoWrite -> todowrite
    assert _to_claude_code_name("todowrite") == "TodoWrite"
    assert _from_claude_code_name("TodoWrite", [make_tool("todowrite")]) == "todowrite"


def test_round_trips_pi_builtin_tools():
    for name, canonical in (("read", "Read"), ("write", "Write"), ("edit", "Edit"), ("bash", "Bash")):
        assert _to_claude_code_name(name) == canonical
        assert _from_claude_code_name(canonical, [make_tool(name)]) == name


def test_does_not_map_find_to_glob():
    # `find` is not a CC tool name; the old find -> Glob mapping broke the
    # round-trip because no tool named "glob" exists in context.tools.
    assert _to_claude_code_name("find") == "find"
    assert _from_claude_code_name("find", [make_tool("find")]) == "find"


def test_custom_tools_pass_through_unchanged():
    assert _to_claude_code_name("my_custom_tool") == "my_custom_tool"
    assert _from_claude_code_name("my_custom_tool", [make_tool("my_custom_tool")]) == "my_custom_tool"


def test_inbound_names_without_matching_tool_pass_through():
    assert _from_claude_code_name("Glob", [make_tool("find")]) == "Glob"


@pytest.mark.tonio
async def test_oauth_params_use_cc_casing_and_claude_code_identity():
    model = get_builtin_model("anthropic", "claude-sonnet-4-6")
    context = Context(
        system_prompt="Custom system prompt.",
        messages=[UserMessage(content="Add a todo.", timestamp=now_ms())],
        tools=[make_tool("todowrite")],
    )

    params = _build_params(model, context, True, AnthropicOptions())

    assert params["tools"][0]["name"] == "TodoWrite"
    assert params["system"][0]["text"] == "You are Claude Code, Anthropic's official CLI for Claude."
    assert params["system"][1]["text"] == "Custom system prompt."

    non_oauth = _build_params(model, context, False, AnthropicOptions())
    assert non_oauth["tools"][0]["name"] == "todowrite"
    assert non_oauth["system"][0]["text"] == "Custom system prompt."
