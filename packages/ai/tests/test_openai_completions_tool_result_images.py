"""Partial mirror of pi's openai-completions-tool-result-images.test.ts.

Holds the 0.87.1 empty-text-part case and the 1.4.0 placeholder case for
empty and whitespace-only tool results; the tool-result image batching cases
are a PARITY GAP. pi passes a hand-built full compat object; `detect_compat`
of the same model yields the equivalent here.
"""

import dataclasses
import time

import pytest

from pidrei_ai.api.openai_completions import convert_messages, detect_compat
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    ImageContent,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pidrei_ai.utils.transcript import normalize_context


def build_empty_tool_result(tool_call_id: str, timestamp: int, text: str = "") -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=tool_call_id,
        tool_name="bash",
        content=[TextContent(text=text)],
        is_error=False,
        timestamp=timestamp,
    )


# Regression test for https://github.com/earendil-works/pi/issues/9797
def test_omits_empty_text_parts_from_user_messages_with_images():
    model = dataclasses.replace(
        get_builtin_model("openai", "gpt-4o-mini"),
        api="openai-completions",
        compat=None,
        input=["text", "image"],
    )
    context = normalize_context(
        Context(
            messages=[
                UserMessage(
                    content=[TextContent(text=""), ImageContent(data="ZmFrZQ==", mime_type="image/png")],
                    timestamp=int(time.time() * 1000),
                )
            ]
        )
    )

    assert convert_messages(model, context, detect_compat(model)) == [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,ZmFrZQ=="}}]}
    ]


# Whitespace-only cases: https://github.com/earendil-works/pi/issues/8720
@pytest.mark.parametrize("text", ["", "\r\n", " \n\t"])
def test_uses_no_tool_output_placeholder_for_tool_result_text_without_images(text):
    model = dataclasses.replace(
        get_builtin_model("openai", "gpt-4o-mini"),
        api="openai-completions",
        compat=None,
        input=["text", "image"],
    )

    now = int(time.time() * 1000)
    assistant_message = AssistantMessage(
        content=[ToolCall(id="tool-1", name="bash", arguments={"command": "true"})],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="toolUse",
        timestamp=now,
    )

    context = normalize_context(
        Context(
            messages=[
                UserMessage(content="Run the command", timestamp=now - 1),
                assistant_message,
                build_empty_tool_result("tool-1", now + 1, text),
            ]
        )
    )

    messages = convert_messages(model, context, detect_compat(model))
    tool_message = next((m for m in messages if m["role"] == "tool"), None)
    assert tool_message is not None
    assert tool_message["content"] == "(no tool output)"
    assert "see attached image" not in tool_message["content"]
