"""Mirror of pi's openai-responses-tool-call-replay.test.ts."""

from pidrei_ai.api.openai_responses_shared import convert_responses_messages
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pidrei_ai.utils.transcript import normalize_context


def tool_result(tool_call_id: str, tool_name: str) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        content=[TextContent(text="ok")],
        is_error=False,
        timestamp=1,
    )


# #12367: OpenAI returns tool calls with status "completed". Replaying them without it changes the input
# before later reasoning items, which GPT-6.2 then drops as prefix_mismatch once a Codex WebSocket
# continuation ends and the full input is sent again.
def test_replays_function_and_custom_tool_calls_with_the_status_the_api_returned():
    model = get_builtin_model("openai-codex", "gpt-6.2-sol")
    assistant = AssistantMessage(
        content=[
            ToolCall(id="call_1|fc_1", name="read", arguments={"path": "calc.py"}),
            ToolCall(id="call_2|ctc_2", name="codemode", arguments={"code": "text(1)"}),
        ],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="toolUse",
        timestamp=1,
    )
    context = normalize_context(
        Context(
            messages=[
                UserMessage(content="Use the tools.", timestamp=1),
                assistant,
                tool_result("call_1|fc_1", "read"),
                tool_result("call_2|ctc_2", "codemode"),
            ]
        )
    )

    input = convert_responses_messages(
        model, context, {"openai-codex"}, grammar_tool_input_properties={"codemode": "code"}
    )

    assert {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "read",
        # pi's text is compact ('{"path":"calc.py"}'); json.dumps writes it spaced. The difference is known.
        "arguments": '{"path": "calc.py"}',
        "status": "completed",
    } in input
    assert {
        "type": "custom_tool_call",
        "id": "ctc_2",
        "call_id": "call_2",
        "name": "codemode",
        "input": "text(1)",
        "status": "completed",
    } in input
