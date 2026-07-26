"""Mirror of pi's transform-messages-copilot-openai-to-anthropic.test.ts.

The behaviours here are also covered generically by `test_transform_messages.py`;
what this file pins is the Copilot session-migration case, where the *same*
provider serves both OpenAI and Anthropic models, so a stored session can carry
openai-responses content into an anthropic-messages request. pi hand-builds its
model literal; the real catalog entry is used instead, since it exists now.
"""

import re

from pidrei_ai.api.transform_messages import transform_messages
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import (
    AssistantMessage,
    Model,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def copilot_claude_model() -> Model:
    model = get_builtin_model("github-copilot", "claude-sonnet-4.6")
    assert model is not None and model.api == "anthropic-messages"
    return model


def anthropic_normalize_tool_call_id(tool_call_id: str, _model, _source) -> str:
    """The normalizer the anthropic adapter uses."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", tool_call_id)[:64]


def openai_assistant(content, *, api: str = "openai-responses", model: str = "gpt-5") -> AssistantMessage:
    return AssistantMessage(
        content=content,
        api=api,
        provider="github-copilot",
        model=model,
        usage=Usage(),
        stop_reason="toolUse",
        timestamp=1,
    )


def test_converts_thinking_blocks_to_plain_text_when_the_source_model_differs():
    messages = [
        UserMessage(content="hello", timestamp=1),
        openai_assistant(
            [
                ThinkingContent(thinking="Let me think about this...", thinking_signature="reasoning_content"),
                TextContent(text="Hi there!"),
            ],
            api="openai-completions",
            model="gpt-4o",
        ),
    ]

    result = transform_messages(
        messages, copilot_claude_model(), normalize_tool_call_id=anthropic_normalize_tool_call_id
    )
    assistant_message = next(message for message in result if message.role == "assistant")

    text_blocks = [block for block in assistant_message.content if block.type == "text"]
    thinking_blocks = [block for block in assistant_message.content if block.type == "thinking"]
    assert thinking_blocks == []
    assert len(text_blocks) >= 2


def test_removes_thought_signature_from_tool_calls_when_migrating_between_models():
    messages = [
        UserMessage(content="run a command", timestamp=1),
        openai_assistant(
            [
                ToolCall(
                    id="call_123",
                    name="bash",
                    arguments={"command": "ls"},
                    thought_signature='{"type":"reasoning.encrypted","id":"call_123","data":"encrypted"}',
                )
            ]
        ),
        ToolResultMessage(
            tool_call_id="call_123",
            tool_name="bash",
            content=[TextContent(text="output")],
            is_error=False,
            timestamp=2,
        ),
    ]

    result = transform_messages(
        messages, copilot_claude_model(), normalize_tool_call_id=anthropic_normalize_tool_call_id
    )
    assistant_message = next(message for message in result if message.role == "assistant")
    tool_call = next(block for block in assistant_message.content if block.type == "toolCall")

    assert tool_call.thought_signature is None


def test_adds_synthetic_tool_results_for_trailing_orphaned_tool_calls():
    messages = [
        UserMessage(content="read the file", timestamp=1),
        openai_assistant([ToolCall(id="call_123|fc_123", name="read", arguments={"path": "README.md"})]),
    ]

    result = transform_messages(
        messages, copilot_claude_model(), normalize_tool_call_id=anthropic_normalize_tool_call_id
    )
    last = result[-1]

    assert last.role == "toolResult"
    assert last.tool_call_id == "call_123_fc_123"
    assert last.tool_name == "read"
    assert last.is_error is True
    assert [block.text for block in last.content] == ["No result provided"]


def test_adds_synthetic_results_only_for_trailing_tool_calls_still_missing_results():
    messages = [
        UserMessage(content="run commands", timestamp=1),
        openai_assistant(
            [
                ToolCall(id="call_1|fc_1", name="read", arguments={"path": "README.md"}),
                ToolCall(id="call_2|fc_2", name="bash", arguments={"command": "pwd"}),
            ]
        ),
        ToolResultMessage(
            tool_call_id="call_1|fc_1",
            tool_name="read",
            content=[TextContent(text="done")],
            is_error=False,
            timestamp=2,
        ),
    ]

    result = transform_messages(
        messages, copilot_claude_model(), normalize_tool_call_id=anthropic_normalize_tool_call_id
    )
    synthetic = [message for message in result if message.role == "toolResult" and message.is_error]

    assert len(synthetic) == 1
    assert synthetic[0].tool_call_id == "call_2_fc_2"
    assert synthetic[0].tool_name == "bash"
    assert [block.text for block in synthetic[0].content] == ["No result provided"]
