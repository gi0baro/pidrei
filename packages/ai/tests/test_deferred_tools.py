"""Partial mirror of pi ai/test/deferred-tools.test.ts.

Holds the 0.84.2 `additional_tools` cases (e47b8e37); the rest of the pi suite
(Kimi message-anchored tools, Anthropic/legacy flows) is a recorded parity gap
in scripts/upstream_diff.py. pi drives streamSimple against catalog models with
an onPayload capture; here the request builders are called directly and the
models are hand-built with explicit compat, so the assertions do not depend on
a catalog regen.
"""

from pidrei_ai.api.openai_codex_responses import build_request_body
from pidrei_ai.api.openai_responses import OpenAIResponsesOptions, build_params
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    OpenAIResponsesCompat,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def make_tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"The {name} tool",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
    )


def make_user_message(timestamp: int) -> UserMessage:
    return UserMessage(content="Hello", timestamp=timestamp)


def make_assistant_tool_call() -> AssistantMessage:
    return AssistantMessage(
        content=[ToolCall(id="call_1", name="base_tool", arguments={})],
        api="anthropic-messages",
        provider="anthropic",
        model="claude-opus-4-6",
        usage=Usage(),
        stop_reason="toolUse",
        timestamp=2,
    )


def make_tool_result(added_tool_names: list[str]) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id="call_1",
        tool_name="base_tool",
        content=[TextContent(text="done")],
        added_tool_names=added_tool_names,
        is_error=False,
        timestamp=3,
    )


def make_context(tools: list[Tool], added_tool_names: list[str] | None = None) -> Context:
    return Context(
        messages=[
            make_user_message(1),
            make_assistant_tool_call(),
            make_tool_result(added_tool_names if added_tool_names is not None else ["late_tool"]),
            make_user_message(4),
        ],
        tools=tools,
    )


def make_responses_model(compat: OpenAIResponsesCompat | None) -> Model:
    return Model(
        id="gpt-5.4",
        name="GPT-5.4",
        api="openai-responses",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=True,
        input=["text"],
        cost=ModelCost(),
        context_window=400_000,
        max_tokens=128_000,
        compat=compat,
    )


def make_codex_model(model_id: str, compat: OpenAIResponsesCompat | None) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="openai-codex-responses",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        reasoning=True,
        input=["text"],
        cost=ModelCost(),
        context_window=400_000,
        max_tokens=128_000,
        compat=compat,
    )


def openai_tool_names(payload: dict) -> list[str]:
    return [tool.get("name") for tool in payload.get("tools") or []]


def responses_payload(compat: OpenAIResponsesCompat | None, context: Context) -> dict:
    return build_params(make_responses_model(compat), context, OpenAIResponsesOptions())


def codex_payload(model: Model, context: Context) -> dict:
    return build_request_body(model, context, None, None)


def test_loads_an_openai_responses_tool_through_additional_tools():
    context = make_context([make_tool("base_tool"), make_tool("late_tool")])
    payload = responses_payload(
        OpenAIResponsesCompat(supports_additional_tools=True, supports_tool_search=True), context
    )
    additional_tools = next((item for item in payload["input"] if item.get("type") == "additional_tools"), None)

    assert openai_tool_names(payload) == ["base_tool"]
    assert additional_tools is not None
    assert additional_tools["role"] == "developer"
    assert [(tool["type"], tool["name"]) for tool in additional_tools["tools"]] == [("function", "late_tool")]
    assert all("defer_loading" not in tool for tool in additional_tools["tools"])
    assert not any(item.get("type") == "tool_search_call" for item in payload["input"])
    assert not any(item.get("type") == "tool_search_output" for item in payload["input"])


def test_preserves_an_additional_tools_marker_after_the_loaded_tool_is_used():
    from dataclasses import replace

    context = make_context([make_tool("base_tool"), make_tool("late_tool")])
    late_call = replace(
        make_assistant_tool_call(),
        content=[ToolCall(id="call_late|fc_late", name="late_tool", arguments={})],
        api="openai-responses",
        provider="openai",
        model="gpt-5.4",
    )
    late_result = replace(make_tool_result(["late_tool"]), tool_call_id="call_late|fc_late", tool_name="late_tool")
    context.messages[3:3] = [late_call, late_result]

    payload = responses_payload(
        OpenAIResponsesCompat(supports_additional_tools=True, supports_tool_search=True), context
    )
    additional_tool_indexes = [
        index for index, item in enumerate(payload["input"]) if item.get("type") == "additional_tools"
    ]
    late_call_index = next(
        index
        for index, item in enumerate(payload["input"])
        if item.get("type") == "function_call" and item.get("name") == "late_tool"
    )

    assert len(additional_tool_indexes) == 1
    assert additional_tool_indexes[0] < late_call_index
    assert openai_tool_names(payload) == ["base_tool"]


def test_falls_back_to_client_tool_search_when_additional_tools_is_unsupported():
    context = make_context([make_tool("base_tool"), make_tool("late_tool")])
    payload = responses_payload(
        OpenAIResponsesCompat(supports_additional_tools=False, supports_tool_search=True), context
    )
    search_call = next((item for item in payload["input"] if item.get("type") == "tool_search_call"), None)
    search_output = next((item for item in payload["input"] if item.get("type") == "tool_search_output"), None)

    assert openai_tool_names(payload) == ["base_tool"]
    assert search_call is not None
    assert search_call["execution"] == "client"
    assert search_call["status"] == "completed"
    assert search_output is not None
    assert search_output["call_id"] == search_call["call_id"]
    assert [(tool["type"], tool["name"], tool.get("defer_loading")) for tool in search_output["tools"]] == [
        ("function", "late_tool", True)
    ]
    assert not any(item.get("type") == "additional_tools" for item in payload["input"])


def test_selects_additional_tools_tool_search_or_top_level_tools_for_codex_models():
    context = make_context([make_tool("base_tool"), make_tool("late_tool")])
    additional_tools_payload = codex_payload(
        make_codex_model(
            "gpt-5.6-sol", OpenAIResponsesCompat(supports_additional_tools=True, supports_tool_search=True)
        ),
        context,
    )
    tool_search_payload = codex_payload(
        make_codex_model("gpt-5.4", OpenAIResponsesCompat(supports_tool_search=True)),
        context,
    )
    top_level_payload = codex_payload(make_codex_model("gpt-5.3-codex-spark", None), context)

    assert openai_tool_names(additional_tools_payload) == ["base_tool"]
    assert any(item.get("type") == "additional_tools" for item in additional_tools_payload["input"])
    assert not any(item.get("type") == "tool_search_call" for item in additional_tools_payload["input"])

    assert openai_tool_names(tool_search_payload) == ["base_tool"]
    assert any(item.get("type") == "tool_search_call" for item in tool_search_payload["input"])
    assert not any(item.get("type") == "additional_tools" for item in tool_search_payload["input"])

    assert openai_tool_names(top_level_payload) == ["base_tool", "late_tool"]
    assert not any(item.get("type") == "tool_search_call" for item in top_level_payload["input"])
    assert not any(item.get("type") == "additional_tools" for item in top_level_payload["input"])
