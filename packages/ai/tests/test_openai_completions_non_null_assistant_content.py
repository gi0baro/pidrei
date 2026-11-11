"""Mirror of pi's openai-completions-non-null-assistant-content.test.ts.

pi passes a hand-built full compat object (`baseCompat`); `detect_compat` of the
same model, with the case's flag replaced, is the equivalent here.
"""

import dataclasses
import time

from pidrei_ai.api.openai_completions import convert_messages, detect_compat
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pidrei_ai.utils.transcript import normalize_context


def build_model() -> Model:
    return dataclasses.replace(
        get_builtin_model("openai", "gpt-4o-mini"), api="openai-completions", compat=None, input=["text", "image"]
    )


def base_compat(model: Model, **overrides):
    return dataclasses.replace(detect_compat(model), **overrides)


def build_context(model: Model):
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
    return normalize_context(
        Context(
            messages=[
                UserMessage(content="Run the command", timestamp=now - 1),
                assistant_message,
                ToolResultMessage(
                    tool_call_id="tool-1",
                    tool_name="bash",
                    content=[TextContent(text="ok")],
                    is_error=False,
                    timestamp=now + 1,
                ),
                UserMessage(content="Thanks", timestamp=now + 2),
            ]
        )
    )


def test_sends_null_content_for_tool_call_only_assistant_messages_by_default():
    model = build_model()
    messages = convert_messages(model, build_context(model), base_compat(model))
    assistant_messages = [message for message in messages if message["role"] == "assistant"]
    assert len(assistant_messages) == 1
    assert assistant_messages[0]["content"] is None


def test_sends_empty_string_content_without_inserting_an_assistant_message_after_tool_results():
    model = build_model()
    messages = convert_messages(
        model, build_context(model), base_compat(model, requires_non_null_assistant_content=True)
    )
    roles = [message["role"] for message in messages]
    assert roles == ["user", "assistant", "tool", "user"]
    assistant_messages = [message for message in messages if message["role"] == "assistant"]
    assert len(assistant_messages) == 1
    assert assistant_messages[0]["content"] == ""


def test_still_inserts_an_assistant_message_when_requires_assistant_after_tool_result_is_enabled():
    model = build_model()
    messages = convert_messages(
        model, build_context(model), base_compat(model, requires_assistant_after_tool_result=True)
    )
    roles = [message["role"] for message in messages]
    assert roles == ["user", "assistant", "tool", "assistant", "user"]
    assistant_messages = [message for message in messages if message["role"] == "assistant"]
    assert assistant_messages[0]["content"] == ""
    assert assistant_messages[1]["content"] == "I have processed the tool results."
