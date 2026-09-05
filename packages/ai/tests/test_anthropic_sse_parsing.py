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
async def test_preserves_content_from_content_block_start_events():
    body = sse_body(
        [
            (
                "message_start",
                json.dumps(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_initial_content",
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
                        "content_block": {"type": "text", "text": "Initial text"},
                    }
                ),
            ),
            (
                "content_block_delta",
                json.dumps(
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " plus delta"}}
                ),
            ),
            ("content_block_stop", json.dumps({"type": "content_block_stop", "index": 0})),
            (
                "content_block_start",
                json.dumps(
                    {
                        "type": "content_block_start",
                        "index": 1,
                        "content_block": {
                            "type": "thinking",
                            "thinking": "Initial thinking",
                            "signature": "initial signature",
                        },
                    }
                ),
            ),
            (
                "content_block_delta",
                json.dumps(
                    {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {"type": "thinking_delta", "thinking": " plus delta"},
                    }
                ),
            ),
            (
                "content_block_delta",
                json.dumps(
                    {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {"type": "signature_delta", "signature": " plus delta"},
                    }
                ),
            ),
            ("content_block_stop", json.dumps({"type": "content_block_stop", "index": 1})),
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
    )

    result = await stream_anthropic(
        haiku(), user_context("Say hello."), AnthropicOptions(client=FakeClient(body))
    ).result()

    assert len(result.content) == 2
    assert result.content[0] == TextContent(text="Initial text plus delta")
    thinking = result.content[1]
    assert thinking.type == "thinking"
    assert thinking.thinking == "Initial thinking plus delta"
    assert thinking.thinking_signature == "initial signature plus delta"


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
async def test_fails_safely_when_anthropic_falls_back_after_output_begins():
    model = get_builtin_model("anthropic", "claude-opus-5")
    events = [
        (
            "message_start",
            json.dumps(
                {
                    "type": "message_start",
                    "message": {"id": "msg_fallback", "model": "claude-opus-5", "usage": {"input_tokens": 1}},
                }
            ),
        ),
        (
            "content_block_start",
            json.dumps(
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "partial"}}
            ),
        ),
        ("content_block_stop", json.dumps({"type": "content_block_stop", "index": 0})),
        (
            "content_block_start",
            json.dumps(
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "fallback",
                        "from": {"model": "claude-opus-5"},
                        "to": {"model": "claude-opus-4-8"},
                    },
                }
            ),
        ),
    ]

    result = await stream_anthropic(
        model, user_context("Hello"), AnthropicOptions(client=FakeClient(sse_body(events)))
    ).result()

    assert result.stop_reason == "error"
    assert "unsupported mid-output model fallback" in result.error_message


@pytest.mark.tonio
async def test_forces_streaming_after_an_on_payload_replacement():
    client = FakeClient(sse_body(minimal_anthropic_events()))

    async def on_payload(payload, _model):
        return {**payload, "stream": False}

    await stream_anthropic(
        get_builtin_model("anthropic", "claude-fable-5-1"),
        user_context("Hello"),
        AnthropicOptions(client=client, on_payload=on_payload),
    ).result()

    assert client.requests[0]["stream"] is True


@pytest.mark.tonio
async def test_omits_the_interleaved_thinking_beta_when_thinking_is_disabled():
    client = FakeClient(sse_body(minimal_anthropic_events()))

    await stream_anthropic(
        get_builtin_model("openrouter", "anthropic/claude-3-haiku"),
        user_context("Hello"),
        AnthropicOptions(client=client, thinking_enabled=False),
    ).result()

    assert "interleaved-thinking-2025-05-14" not in client.requests[0].get("betas", [])


@pytest.mark.tonio
async def test_passes_managed_beta_features_to_injected_clients():
    client = FakeClient(sse_body(minimal_anthropic_events()))

    result = await stream_anthropic(
        get_builtin_model("anthropic", "claude-fable-5-1"), user_context("Hello"), AnthropicOptions(client=client)
    ).result()

    assert result.stop_reason == "stop"
    assert "mid-conversation-output-config-2026-07-01" in client.requests[0]["betas"]
    assert "thinking-binding-controls-2026-08-01" in client.requests[0]["betas"]


@pytest.mark.tonio
async def test_uses_the_serving_model_input_transformations_from_the_final_stream_event():
    events = minimal_anthropic_events()
    events[0] = (
        "message_start",
        json.dumps(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_transformations",
                    "model": "claude-fable-5-1",
                    "usage": {"input_tokens": 12, "output_tokens": 0},
                    "input_transformations": [
                        {
                            "type": "thinking_dropped",
                            "path": "messages.1.content.0",
                            "reason": "prefix_binding_mismatch",
                        }
                    ],
                },
            }
        ),
    )
    delta = json.loads(events[4][1])
    delta["input_transformations"] = [
        {"type": "thinking_dropped", "path": "messages.3.content.0", "reason": "model_binding_mismatch"}
    ]
    events[4] = ("message_delta", json.dumps(delta))

    result = await stream_anthropic(
        get_builtin_model("anthropic", "claude-fable-5-1"),
        user_context("Hello"),
        AnthropicOptions(client=FakeClient(sse_body(events))),
    ).result()

    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.type == "anthropic_input_transformations"
    assert isinstance(diagnostic.timestamp, int)
    assert diagnostic.details == {
        "transformations": [
            {"type": "thinking_dropped", "path": "messages.3.content.0", "reason": "model_binding_mismatch"}
        ]
    }


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


@pytest.mark.tonio
async def test_cancel_during_time_to_first_byte_aborts_the_request():
    """Esc while the request head is still pending: the producer is parked in
    `client.create`, which nothing will ever answer; cancellation must end
    the stream with an aborted message, not wait for a read timeout."""
    import tonio.colored as tonio

    from pidrei_ai.utils.cancel import CancelToken

    head_pending = tonio.Event()

    class StalledClient:
        async def create(self, params, *, timeout_ms, cancel):
            head_pending.set()
            await tonio.Event().wait(None)  # never answered

    model = get_builtin_model("anthropic", "claude-sonnet-4-5")
    cancel = CancelToken()
    stream = stream_anthropic(
        model,
        Context(messages=[UserMessage(content="hi", timestamp=1)]),
        AnthropicOptions(api_key="k", client=StalledClient(), cancel=cancel),
    )
    await head_pending.wait(None)
    cancel.cancel()

    events = [event async for event in stream]
    result = await stream.result()
    assert [event.type for event in events] == ["error"]
    assert result.stop_reason == "aborted"
    assert result.error_message == "Request was aborted"
