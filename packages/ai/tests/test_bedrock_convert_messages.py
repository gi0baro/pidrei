"""Mirror of pi's bedrock-convert-messages.test.ts.

pi captures the command input through `onPayload` while the request is already
aborted; here the same is done with an already-cancelled `CancelToken`.

pi's "unknown content block" cases cast a bare object through `as any`. The
Python stand-in is a small dataclass with the same `type` discriminator, which is
all `convert_messages` reads.
"""

import contextlib
from dataclasses import dataclass
from typing import ClassVar

import pytest

from pidrei_ai.api import bedrock_converse_stream as bedrock
from pidrei_ai.api.bedrock_converse_stream import BedrockOptions, stream as stream_bedrock
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    JsonSchemaConstrainedSampling,
    TextContent,
    Tool,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pidrei_ai.utils.cancel import CancelToken


BASE_MODEL = get_builtin_model("amazon-bedrock", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")


@dataclass(slots=True)
class UnknownContent:
    """pi's `{ type: "unknown", data: "foo" } as any`."""

    data: str = "foo"
    type: str = "unknown"


class _UnusedClient:
    def __init__(self, _config):
        self.middleware_stack = _NullStack()

    async def send(self, _command, *, cancel=None):
        raise RuntimeError("mock send")


class _NullStack:
    def add(self, *_args, **_kwargs) -> None:
        pass


@contextlib.contextmanager
def _stubbed_client():
    original = bedrock.BedrockRuntimeClient
    bedrock.BedrockRuntimeClient = _UnusedClient
    try:
        yield
    finally:
        bedrock.BedrockRuntimeClient = original


async def capture_payload(context: Context, model=None) -> dict:
    captured: list[dict] = []
    cancel = CancelToken()
    cancel.cancel()

    async def on_payload(payload, _model):
        captured.append(payload)
        return payload

    options = BedrockOptions(cache_retention="none", cancel=cancel, on_payload=on_payload)
    with _stubbed_client():
        stream = stream_bedrock(model or BASE_MODEL, context, options)
        async for event in stream:
            if event.type == "error":
                break
    assert captured
    return captured[0]


def assistant(content) -> AssistantMessage:
    return AssistantMessage(
        content=content,
        api="bedrock-converse-stream",
        provider="amazon-bedrock",
        model=BASE_MODEL.id,
        usage=Usage(),
        stop_reason="stop",
        timestamp=1,
    )


# --- constrained sampling -----------------------------------------------------


@pytest.mark.tonio
async def test_gates_native_strict_tool_use_by_model_capability():
    tool = Tool(
        name="lookup",
        description="Look up a value",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}},
        constrained_sampling=JsonSchemaConstrainedSampling(strict="require"),
    )
    context = Context(messages=[UserMessage(content="Use the tool", timestamp=1)], tools=[tool])

    payload = await capture_payload(context)
    assert payload["toolConfig"]["tools"][0]["toolSpec"]["strict"] is True

    tool.constrained_sampling = JsonSchemaConstrainedSampling(strict="prefer")
    nova_payload = await capture_payload(context, get_builtin_model("amazon-bedrock", "amazon.nova-lite-v1:0"))
    assert "strict" not in nova_payload["toolConfig"]["tools"][0]["toolSpec"]


# --- tool arguments -------------------------------------------------------------


class _StreamingClient:
    """Client whose send() replays canned Converse stream events."""

    stream_events: ClassVar[list[dict]] = []

    def __init__(self, _config):
        self.middleware_stack = _NullStack()

    async def send(self, _command, *, cancel=None):
        from types import SimpleNamespace

        events = list(_StreamingClient.stream_events)

        async def stream():
            for event in events:
                yield event

        return SimpleNamespace(
            metadata=SimpleNamespace(request_id=None, http_status_code=200),
            stream=stream(),
        )


@contextlib.contextmanager
def _streaming_client(events: list[dict]):
    original = bedrock.BedrockRuntimeClient
    _StreamingClient.stream_events = events
    bedrock.BedrockRuntimeClient = _StreamingClient
    try:
        yield
    finally:
        bedrock.BedrockRuntimeClient = original
        _StreamingClient.stream_events = []


@pytest.mark.tonio
async def test_preserves_empty_property_names_in_streamed_tool_arguments():
    events = [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {"toolUse": {"toolUseId": "tool-1", "name": "edit"}},
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {
                    "toolUse": {
                        "input": (
                            '{"path":"/workspace/foobar/file.js","edits":[{"oldText":"first","newText":"updated first"},'
                            '{"oldText":"second","newText":"updated second","":""}]}'
                        )
                    }
                },
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]

    with _streaming_client(events):
        message = await stream_bedrock(
            BASE_MODEL,
            Context(messages=[UserMessage(content="Use the tool", timestamp=1)]),
            BedrockOptions(cache_retention="none"),
        ).result()

    block = message.content[0]
    assert block.type == "toolCall"
    assert block.id == "tool-1"
    assert block.name == "edit"
    assert block.arguments == {
        "path": "/workspace/foobar/file.js",
        "edits": [
            {"oldText": "first", "newText": "updated first"},
            {"oldText": "second", "newText": "updated second", "": ""},
        ],
    }


@pytest.mark.tonio
async def test_removes_empty_property_names_only_from_replayed_bedrock_input():
    from pidrei_ai.types import ToolCall

    tool_arguments = {
        "path": "/workspace/foobar/file.js",
        "edits": [
            {"oldText": "first", "newText": "updated first"},
            {"oldText": "second", "newText": "updated second", "": ""},
        ],
    }
    messages = [
        AssistantMessage(
            content=[ToolCall(id="tool-1", name="edit", arguments=tool_arguments)],
            api="bedrock-converse-stream",
            provider="amazon-bedrock",
            model=BASE_MODEL.id,
            usage=Usage(),
            stop_reason="toolUse",
            timestamp=1,
        ),
        ToolResultMessage(
            tool_call_id="tool-1",
            tool_name="edit",
            content=[TextContent(text="done")],
            is_error=False,
            timestamp=2,
        ),
        UserMessage(content="Continue", timestamp=3),
    ]

    payload = await capture_payload(Context(messages=messages))

    assert payload["messages"][0]["content"][0]["toolUse"]["input"] == {
        "path": "/workspace/foobar/file.js",
        "edits": [
            {"oldText": "first", "newText": "updated first"},
            {"oldText": "second", "newText": "updated second"},
        ],
    }
    assert tool_arguments["edits"][1] == {"oldText": "second", "newText": "updated second", "": ""}


# --- unknown content types ----------------------------------------------------


@pytest.mark.tonio
async def test_skips_unknown_user_content_blocks_instead_of_throwing():
    payload = await capture_payload(
        Context(messages=[UserMessage(content=[TextContent(text="hello"), UnknownContent()], timestamp=1)])
    )

    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["content"] == [{"text": "hello"}]


@pytest.mark.tonio
async def test_skips_unknown_assistant_content_blocks_instead_of_throwing():
    payload = await capture_payload(Context(messages=[assistant([TextContent(text="hello"), UnknownContent()])]))

    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["content"] == [{"text": "hello"}]


@pytest.mark.tonio
async def test_replaces_user_messages_with_only_unknown_content_blocks_with_a_placeholder():
    payload = await capture_payload(Context(messages=[UserMessage(content=[UnknownContent()], timestamp=1)]))

    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["content"] == [{"text": "<empty>"}]


@pytest.mark.tonio
async def test_replaces_blank_user_string_content_with_a_placeholder():
    payload = await capture_payload(Context(messages=[UserMessage(content="   ", timestamp=1)]))

    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["content"] == [{"text": "<empty>"}]


@pytest.mark.tonio
async def test_filters_blank_user_text_blocks_when_other_content_remains():
    payload = await capture_payload(
        Context(messages=[UserMessage(content=[TextContent(text=""), TextContent(text="hello")], timestamp=1)])
    )

    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["content"] == [{"text": "hello"}]


@pytest.mark.tonio
async def test_replaces_user_content_emptied_by_surrogate_sanitization_with_a_placeholder():
    payload = await capture_payload(Context(messages=[UserMessage(content="\ud83d", timestamp=1)]))

    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["content"] == [{"text": "<empty>"}]


@pytest.mark.tonio
async def test_skips_assistant_text_blocks_emptied_by_surrogate_sanitization():
    payload = await capture_payload(Context(messages=[assistant([TextContent(text="\ud83d")])]))

    assert len(payload["messages"]) == 0


@pytest.mark.tonio
async def test_replaces_blank_tool_result_content_with_a_placeholder():
    payload = await capture_payload(
        Context(
            messages=[
                ToolResultMessage(
                    tool_call_id="tool-1",
                    tool_name="tool",
                    content=[TextContent(text="")],
                    is_error=False,
                    timestamp=1,
                )
            ]
        )
    )

    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["content"][0]["toolResult"]["content"] == [{"text": "<empty>"}]


@pytest.mark.tonio
async def test_skips_assistant_messages_with_only_unknown_content_blocks():
    payload = await capture_payload(Context(messages=[assistant([UnknownContent()])]))

    assert len(payload["messages"]) == 0
