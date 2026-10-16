"""Mirror of pi's transform-messages-cross-model-thinking.test.ts."""

from dataclasses import replace

from pidrei_ai.api.transform_messages import transform_messages
from pidrei_ai.types import (
    AssistantMessage,
    Model,
    ModelCost,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


KIMI = Model(
    id="moonshotai/Kimi-K3",
    name="Kimi K3",
    api="openai-completions",
    provider="sference",
    base_url="https://api.sference.com/v1",
    reasoning=True,
    input=["text", "image"],
    cost=ModelCost(),
    context_window=1048576,
    max_tokens=32768,
)


def assistant(model: str, content: list) -> AssistantMessage:
    return AssistantMessage(
        content=content,
        api="openai-completions",
        provider="sference",
        model=model,
        usage=Usage(),
        stop_reason="toolUse",
        timestamp=1,
    )


def thinking(text: str) -> ThinkingContent:
    return ThinkingContent(thinking=text, thinking_signature="reasoning_content")


def test_does_not_send_another_models_readable_thinking_as_text():
    # https://github.com/earendil-works/pi/issues/10995
    messages = [
        UserMessage(content="Why does curl work but node fails?", timestamp=1),
        assistant(
            "zai-org/GLM-5.3",
            [
                thinking("The user asks about ECONNRESET. Candidates: proxy, IPv6, TLS."),
                TextContent(text="Let me check the TLS versions."),
                ToolCall(id="call_1", name="bash", arguments={"command": "curl -v"}),
            ],
        ),
        ToolResultMessage(
            tool_call_id="call_1",
            tool_name="bash",
            content=[TextContent(text="TLSv1.2")],
            is_error=False,
            timestamp=1,
        ),
    ]

    result = transform_messages(messages, KIMI)
    assistant_message = next(message for message in result if message.role == "assistant")

    assert list(assistant_message.content) == [
        TextContent(text="Let me check the TLS versions."),
        ToolCall(id="call_1", name="bash", arguments={"command": "curl -v"}),
    ]


def test_keeps_thinking_only_from_the_target_model_when_models_alternate():
    messages = [
        UserMessage(content="1", timestamp=1),
        replace(assistant("moonshotai/Kimi-K3", [thinking("kimi 1"), TextContent(text="a")]), stop_reason="stop"),
        UserMessage(content="2", timestamp=1),
        replace(assistant("zai-org/GLM-5.3", [thinking("glm 1"), TextContent(text="b")]), stop_reason="stop"),
        UserMessage(content="3", timestamp=1),
        replace(assistant("moonshotai/Kimi-K3", [thinking("kimi 2"), TextContent(text="c")]), stop_reason="stop"),
        UserMessage(content="4", timestamp=1),
        replace(assistant("zai-org/GLM-5.3", [thinking("glm 2"), TextContent(text="d")]), stop_reason="stop"),
        UserMessage(content="hi", timestamp=1),
    ]

    result = transform_messages(messages, KIMI)
    contents = [list(message.content) for message in result if message.role == "assistant"]

    assert contents == [
        [thinking("kimi 1"), TextContent(text="a")],
        [TextContent(text="b")],
        [thinking("kimi 2"), TextContent(text="c")],
        [TextContent(text="d")],
    ]
