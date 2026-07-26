"""Copilot dynamic headers: the header builder, and that all three adapters use it.

pi tests the builder only through its anthropic suite
(`github-copilot-anthropic.test.ts`); the openai-completions and
openai-responses branches that call the same helper have no pi test. They were
`NotImplementedError` stubs until the copilot provider landed, so this file
pins all three call sites — a replacement nothing exercises is a replacement
nobody notices breaking.
"""

import time

import pytest

from pidrei_ai.api.github_copilot_headers import (
    build_copilot_dynamic_headers,
    has_copilot_vision_input,
    infer_copilot_initiator,
)
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    ImageContent,
    TextContent,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def now_ms() -> int:
    return int(time.time() * 1000)


def user(content) -> UserMessage:
    return UserMessage(content=content, timestamp=now_ms())


def assistant(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="openai-completions",
        provider="github-copilot",
        model="gpt-5",
        usage=Usage(),
        stop_reason="stop",
        timestamp=now_ms(),
    )


def image() -> ImageContent:
    return ImageContent(data="AAAA", mime_type="image/png")


def test_initiator_is_user_only_when_the_last_message_is_the_users():
    assert infer_copilot_initiator([]) == "user"
    assert infer_copilot_initiator([user("hi")]) == "user"
    assert infer_copilot_initiator([user("hi"), assistant("there")]) == "agent"
    assert (
        infer_copilot_initiator(
            [
                user("hi"),
                ToolResultMessage(
                    tool_call_id="1",
                    tool_name="read",
                    content=[TextContent(text="ok")],
                    is_error=False,
                    timestamp=now_ms(),
                ),
            ]
        )
        == "agent"
    )


def test_vision_input_is_detected_in_user_and_tool_result_content():
    assert has_copilot_vision_input([user("hi")]) is False
    assert has_copilot_vision_input([user([TextContent(text="hi")])]) is False
    assert has_copilot_vision_input([user([image()])]) is True
    assert (
        has_copilot_vision_input(
            [
                ToolResultMessage(
                    tool_call_id="1",
                    tool_name="read",
                    content=[image()],
                    is_error=False,
                    timestamp=now_ms(),
                )
            ]
        )
        is True
    )


def test_the_vision_header_is_only_sent_when_there_are_images():
    assert build_copilot_dynamic_headers([user("hi")], False) == {
        "X-Initiator": "user",
        "Openai-Intent": "conversation-edits",
    }
    assert build_copilot_dynamic_headers([user([image()])], True)["Copilot-Vision-Request"] == "true"


# --- the openai adapters reach the builder ------------------------------------


def copilot_context(messages=None) -> Context:
    return Context(system_prompt="sys", messages=messages or [user("hi")])


@pytest.mark.parametrize("api", ["openai-completions", "openai-responses"])
def test_the_openai_adapters_send_the_copilot_dynamic_headers(api):
    if api == "openai-responses":
        from pidrei_ai.api.openai_responses import _create_client

        from .test_openai_responses import make_model

        model = make_model(provider="github-copilot")
        headers = _create_client(model, copilot_context(), "test-key", None, None)._headers
    else:
        from pidrei_ai.api.openai_completions import _create_client, get_compat

        from .test_openai_completions import make_model

        model = make_model(provider="github-copilot")
        context = copilot_context()
        headers = _create_client(model, context, "test-key", None, None, get_compat(model))._headers

    assert headers["X-Initiator"] == "user"
    assert headers["Openai-Intent"] == "conversation-edits"
    assert "Copilot-Vision-Request" not in headers


@pytest.mark.parametrize("api", ["openai-completions", "openai-responses"])
def test_the_openai_adapters_flag_vision_requests(api):
    messages = [user([image()])]
    if api == "openai-responses":
        from pidrei_ai.api.openai_responses import _create_client

        from .test_openai_responses import make_model

        model = make_model(provider="github-copilot")
        headers = _create_client(model, copilot_context(messages), "test-key", None, None)._headers
    else:
        from pidrei_ai.api.openai_completions import _create_client, get_compat

        from .test_openai_completions import make_model

        model = make_model(provider="github-copilot")
        context = copilot_context(messages)
        headers = _create_client(model, context, "test-key", None, None, get_compat(model))._headers

    assert headers["X-Initiator"] == "user"
    assert headers["Copilot-Vision-Request"] == "true"
