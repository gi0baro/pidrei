"""Mirror of pi's anthropic-sse-parsing.test.ts.

pi injects a fake SDK client whose `asResponse()` returns a canned SSE
Response; here the fake implements the adapter's `AnthropicClient` protocol.
"""

import json
import time

import pytest

from pidrei_ai.api.anthropic_messages import AnthropicOptions, stream as stream_anthropic
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, TextContent, Tool, UserMessage


def sse_body(events: list[tuple[str, str]]) -> bytes:
    # pi: events.map(({event, data}) => `event: ${event}\ndata: ${data}\n`).join("\n")
    return "\n".join(f"event: {event}\ndata: {data}\n" for event, data in events).encode()


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200):
        self.status = status
        self.headers = {"content-type": "text/event-stream"}
        self._body = body

    async def aiter_bytes(self):
        yield self._body


class FakeClient:
    def __init__(self, body: bytes):
        self._body = body
        self.requests: list[dict] = []

    async def create(self, params, *, timeout_ms, cancel):
        self.requests.append(params)
        return FakeResponse(self._body)


def minimal_anthropic_events() -> list[tuple[str, str]]:
    return [
        (
            "message_start",
            json.dumps(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_test",
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        },
                    },
                }
            ),
        ),
        (
            "content_block_start",
            json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ),
        (
            "content_block_delta",
            json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}),
        ),
        ("content_block_stop", json.dumps({"type": "content_block_stop", "index": 0})),
        (
            "message_delta",
            json.dumps(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                }
            ),
        ),
        ("message_stop", json.dumps({"type": "message_stop"})),
    ]


def haiku():
    model = get_builtin_model("anthropic", "claude-haiku-4-5")
    assert model is not None
    return model


def user_context(text: str, tools: list[Tool] | None = None) -> Context:
    return Context(
        messages=[UserMessage(content=text, timestamp=int(time.time() * 1000))],
        tools=tools,
    )


@pytest.mark.tonio
async def test_repairs_malformed_sse_json_and_malformed_streamed_tool_json():
    context = user_context(
        "Use the edit tool.",
        tools=[
            Tool(
                name="edit",
                description="Edit a file.",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["path", "text"],
                },
            )
        ],
    )

    malformed_tool_json_delta = (
        '{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta",'
        '"partial_json":"{\\"path\\":\\"A\\H\\",\\"text\\":\\"col1\tcol2\\"}"}}'
    )

    body = sse_body(
        [
            (
                "message_start",
                json.dumps(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_test",
                            "usage": {
                                "input_tokens": 12,
                                "output_tokens": 0,
                                "cache_read_input_tokens": 0,
                                "cache_creation_input_tokens": 0,
                            },
                        },
                    }
                ),
            ),
            (
                "content_block_start",
                json.dumps(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "tool_use", "id": "toolu_test", "name": "edit", "input": {}},
                    }
                ),
            ),
            ("content_block_delta", malformed_tool_json_delta),
            ("content_block_stop", json.dumps({"type": "content_block_stop", "index": 0})),
            (
                "message_delta",
                json.dumps(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "tool_use"},
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 5,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        },
                    }
                ),
            ),
            ("message_stop", json.dumps({"type": "message_stop"})),
        ]
    )

    result = await stream_anthropic(haiku(), context, AnthropicOptions(client=FakeClient(body))).result()

    assert result.stop_reason == "toolUse"
    assert result.error_message is None

    tool_call = next(block for block in result.content if block.type == "toolCall")
    assert tool_call.arguments == {"path": "A\\H", "text": "col1\tcol2"}


@pytest.mark.tonio
async def test_preserves_refusal_stop_details_from_message_delta():
    model = get_builtin_model("anthropic", "claude-fable-5")
    assert model is not None
    explanation = (
        "This request triggered restrictions on violative cyber content and was blocked under "
        "Anthropic's Usage Policy. To learn more, provide feedback, or request an exemption based "
        "on how you use Claude, visit our help center: "
        "https://support.claude.com/en/articles/14604842-real-time-cyber-safeguards-on-claude."
    )
    body = sse_body(
        [
            (
                "message_start",
                json.dumps(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_01XFUDYJgAACzvnptvVoYEL",
                            "usage": {
                                "input_tokens": 412,
                                "output_tokens": 0,
                                "cache_read_input_tokens": 0,
                                "cache_creation_input_tokens": 0,
                            },
                        },
                    }
                ),
            ),
            (
                "message_delta",
                json.dumps(
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": "refusal",
                            "stop_details": {"type": "refusal", "category": "cyber", "explanation": explanation},
                        },
                        "usage": {
                            "input_tokens": 412,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        },
                    }
                ),
            ),
            ("message_stop", json.dumps({"type": "message_stop"})),
        ]
    )

    result = await stream_anthropic(
        model, user_context("blocked request"), AnthropicOptions(client=FakeClient(body))
    ).result()

    assert result.stop_reason == "error"
    assert result.raw_stop_reason == "refusal"
    assert result.error_message == explanation


@pytest.mark.tonio
async def test_preserves_sensitive_stop_reasons_with_a_descriptive_error_message():
    model = get_builtin_model("anthropic", "claude-haiku-4-5")
    assert model is not None
    body = sse_body(
        [
            (
                "message_start",
                json.dumps(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_sensitive",
                            "usage": {
                                "input_tokens": 12,
                                "output_tokens": 0,
                                "cache_read_input_tokens": 0,
                                "cache_creation_input_tokens": 0,
                            },
                        },
                    }
                ),
            ),
            (
                "message_delta",
                json.dumps(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "sensitive"},
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        },
                    }
                ),
            ),
            ("message_stop", json.dumps({"type": "message_stop"})),
        ]
    )

    result = await stream_anthropic(
        model, user_context("blocked request"), AnthropicOptions(client=FakeClient(body))
    ).result()

    assert result.stop_reason == "error"
    assert result.raw_stop_reason == "sensitive"
    assert result.error_message == "Provider stopped with: sensitive"


@pytest.mark.tonio
async def test_message_delta_without_usage_is_noop_for_usage_accumulation():
    events = [
        ("message_delta", json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}}))
        if event == "message_delta"
        else (event, data)
        for event, data in minimal_anthropic_events()
    ]

    result = await stream_anthropic(
        haiku(), user_context("Say hello."), AnthropicOptions(client=FakeClient(sse_body(events)))
    ).result()

    assert result.stop_reason == "stop"
    assert result.error_message is None
    assert result.content == [TextContent(text="Hello")]
    assert result.usage.input == 12
    assert result.usage.total_tokens == 12


@pytest.mark.tonio
async def test_ignores_unknown_sse_events_after_message_stop():
    events = [*minimal_anthropic_events(), ("done", "[DONE]"), ("proxy.stats", "not json")]

    result = await stream_anthropic(
        haiku(), user_context("Say hello."), AnthropicOptions(client=FakeClient(sse_body(events)))
    ).result()

    assert result.stop_reason == "stop"
    assert result.error_message is None
    assert result.content == [TextContent(text="Hello")]
