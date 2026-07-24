"""Tests for cross-provider message transforms (transform_messages)."""

from dataclasses import replace

from pppi_ai.api.transform_messages import (
    NON_VISION_TOOL_IMAGE_PLACEHOLDER,
    NON_VISION_USER_IMAGE_PLACEHOLDER,
    transform_messages,
)
from pppi_ai.types import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from tests.test_registry import make_model


MODEL = make_model(provider="anthropic", id="claude-x", api="anthropic-messages")


def assistant(content, *, same_model=True, stop_reason="stop") -> AssistantMessage:
    return AssistantMessage(
        content=content,
        api=MODEL.api if same_model else "openai-responses",
        provider=MODEL.provider if same_model else "openai",
        model=MODEL.id if same_model else "gpt-x",
        usage=Usage(),
        stop_reason=stop_reason,
        timestamp=1,
    )


def tool_result(tool_call_id: str, text: str = "ok") -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=tool_call_id,
        tool_name="tool",
        content=[TextContent(text=text)],
        is_error=False,
        timestamp=2,
    )


def test_non_vision_models_get_image_placeholders_deduplicated():
    model = replace(MODEL, input=["text"])
    image = ImageContent(data="...", mime_type="image/png")
    messages = [
        UserMessage(content=[TextContent(text="look"), image, image], timestamp=1),
        tool_result("t1"),
    ]
    messages[1] = replace(messages[1], content=[image])

    user, result = transform_messages(messages, model)
    assert [block.text for block in user.content] == ["look", NON_VISION_USER_IMAGE_PLACEHOLDER]
    assert [block.text for block in result.content] == [NON_VISION_TOOL_IMAGE_PLACEHOLDER]


def test_vision_models_keep_images():
    image = ImageContent(data="...", mime_type="image/png")
    messages = [UserMessage(content=[image], timestamp=1)]
    (user,) = transform_messages(messages, replace(MODEL, input=["text", "image"]))
    assert user.content == [image]


def test_same_model_keeps_thinking_and_signatures():
    thinking = ThinkingContent(thinking="", thinking_signature="sig")
    (out,) = transform_messages([assistant([thinking, TextContent(text="hi")])], MODEL)
    assert out.content[0] is thinking
    assert out.content[1].text == "hi"


def test_cross_model_thinking_becomes_text_and_redacted_is_dropped():
    messages = [
        assistant(
            [
                ThinkingContent(thinking="reasoning here"),
                ThinkingContent(thinking="secret", redacted=True),
                ThinkingContent(thinking="   "),
                TextContent(text="answer"),
            ],
            same_model=False,
        )
    ]
    (out,) = transform_messages(messages, MODEL)
    assert [(block.type, getattr(block, "text", None)) for block in out.content] == [
        ("text", "reasoning here"),
        ("text", "answer"),
    ]


def test_cross_model_tool_calls_lose_thought_signature_and_normalize_ids():
    call = ToolCall(id="original|id", name="tool", arguments={}, thought_signature="gsig")
    messages = [
        assistant([call], same_model=False, stop_reason="toolUse"),
        tool_result("original|id"),
    ]

    out_assistant, out_result = transform_messages(
        messages, MODEL, normalize_tool_call_id=lambda id, model, source: id.replace("|", "_")
    )
    assert out_assistant.content[0].id == "original_id"
    assert out_assistant.content[0].thought_signature is None
    assert out_result.tool_call_id == "original_id"


def test_errored_and_aborted_assistants_are_skipped():
    messages = [
        assistant([TextContent(text="keep")]),
        assistant([TextContent(text="drop")], stop_reason="error"),
        assistant([TextContent(text="drop too")], stop_reason="aborted"),
    ]
    out = transform_messages(messages, MODEL)
    assert [message.content[0].text for message in out] == ["keep"]


def test_orphaned_tool_calls_get_synthetic_results():
    call_a = ToolCall(id="a", name="tool", arguments={})
    call_b = ToolCall(id="b", name="tool", arguments={})
    messages = [
        assistant([call_a, call_b], stop_reason="toolUse"),
        tool_result("a"),
        UserMessage(content="interrupt", timestamp=3),
    ]

    out = transform_messages(messages, MODEL)
    roles = [message.role for message in out]
    assert roles == ["assistant", "toolResult", "toolResult", "user"]
    synthetic = out[2]
    assert synthetic.tool_call_id == "b"
    assert synthetic.is_error is True
    assert synthetic.content[0].text == "No result provided"


def test_trailing_orphaned_tool_calls_get_synthetic_results():
    call = ToolCall(id="tail", name="tool", arguments={})
    out = transform_messages([assistant([call], stop_reason="toolUse")], MODEL)
    assert out[-1].role == "toolResult"
    assert out[-1].tool_call_id == "tail"
