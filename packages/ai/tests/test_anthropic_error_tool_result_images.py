"""Mirror of pi's anthropic-error-tool-result-images.test.ts."""

from dataclasses import replace

import pytest

from pidrei_ai.types import (
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    ModelCost,
    SimpleStreamOptions,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from tests.anthropic_helpers import capture_payload


MODEL = Model(
    id="claude-test",
    name="Claude Test",
    api="anthropic-messages",
    provider="anthropic",
    base_url="http://127.0.0.1:9",
    reasoning=False,
    input=["text", "image"],
    cost=ModelCost(),
    context_window=200000,
    max_tokens=1024,
)

ASSISTANT = AssistantMessage(
    content=[
        ToolCall(id="call-1", name="codemode", arguments={}),
        ToolCall(id="call-2", name="read", arguments={}),
    ],
    api="anthropic-messages",
    provider="anthropic",
    model="claude-test",
    usage=Usage(),
    stop_reason="toolUse",
    timestamp=1,
)


OPTIONS = SimpleStreamOptions(cache_retention="none")


def image(data: str) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


@pytest.mark.tonio
async def test_moves_the_images_after_the_tool_results_of_the_turn():
    # Regression for #10794: Anthropic rejects error tool results that contain images.
    payload = await capture_payload(
        MODEL,
        OPTIONS,
        context=Context(
            messages=[
                UserMessage(content="Show the screenshots", timestamp=1),
                ASSISTANT,
                ToolResultMessage(
                    tool_call_id="call-1",
                    tool_name="codemode",
                    content=[
                        TextContent(text="Script failed"),
                        ImageContent(data="AAAA", mime_type="image/png"),
                        ImageContent(data="CCCC", mime_type="image/png"),
                        TextContent(text="Script error"),
                    ],
                    is_error=True,
                    timestamp=1,
                ),
                ToolResultMessage(
                    tool_call_id="call-2",
                    tool_name="read",
                    content=[ImageContent(data="BBBB", mime_type="image/png")],
                    is_error=False,
                    timestamp=1,
                ),
            ]
        ),
    )

    assert payload["messages"][2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call-1",
                "content": "Script failed\n(image attached after the tool results)\nScript error",
                "is_error": True,
            },
            {
                "type": "tool_result",
                "tool_use_id": "call-2",
                "content": [{"type": "text", "text": "(see attached image)"}, image("BBBB")],
                "is_error": False,
            },
            {"type": "text", "text": "Images from the error result of tool call call-1 (codemode):"},
            image("AAAA"),
            image("CCCC"),
        ],
    }


@pytest.mark.tonio
async def test_keeps_error_tool_results_without_images_unchanged():
    payload = await capture_payload(
        MODEL,
        OPTIONS,
        context=Context(
            messages=[
                UserMessage(content="Run it", timestamp=1),
                replace(ASSISTANT, content=[ASSISTANT.content[0]]),
                ToolResultMessage(
                    tool_call_id="call-1",
                    tool_name="codemode",
                    content=[TextContent(text="Script failed")],
                    is_error=True,
                    timestamp=1,
                ),
            ]
        ),
    )

    assert payload["messages"][2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "Script failed", "is_error": True}],
    }
