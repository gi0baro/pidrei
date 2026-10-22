"""Mirror of pi ai/test/mistral-http-transport.test.ts.

pi injects a `fetch`; pidrei's transport seam is `MistralOptions.client`
(`FakeMistralClient` records the URL, wire payload, and headers the adapter
would have sent). The request-timeout cases use the `fetch` option, as pi
does. pi's active-stream case delivers its body in timed chunks; here the body
arrives at once. pi leaves body stalls to its HTTP client's idle timeout,
which pidrei's callers pass as `timeout_ms` and the request applies to each
read (never the whole stream); pidrei-only cases check that bound, with and
without a caller timeout, and the head timeout the request itself reports.
"""

import json
from dataclasses import replace

import pytest
import tonio.colored as tonio

from pidrei_ai.api.mistral_conversations import MistralOptions, stream as stream_mistral
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    ImageContent,
    ProviderResponse,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pidrei_ai.utils.identity import format_user_agent
from pidrei_ai.utils.transcript import normalize_context
from pidrei_http import http
from pidrei_utils.cancel import CancelToken
from tests.mistral_helpers import FakeMistralClient, FakeMistralResponse, sse_body


MODEL = get_builtin_model("mistral", "mistral-large-latest")


def terminal_event(finish_reason: str = "stop") -> dict:
    return {
        "id": "mistral-response-id",
        "model": "mistral-large-latest",
        "choices": [{"index": 0, "finish_reason": finish_reason, "delta": {}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def sse_client(events: list[dict], headers: dict[str, str] | None = None) -> FakeMistralClient:
    return FakeMistralClient(FakeMistralResponse(sse_body([json.dumps(event) for event in events]), headers=headers))


@pytest.mark.tonio
async def test_serializes_sdk_style_payloads_to_the_mistral_wire_format():
    context = Context(
        system_prompt="Be precise",
        messages=[
            UserMessage(
                content=[TextContent(text="describe"), ImageContent(data="aGVsbG8=", mime_type="image/png")],
                timestamp=1,
            )
        ],
        tools=[
            Tool(
                name="lookup",
                description="Look something up",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            )
        ],
    )
    client = sse_client([terminal_event()], headers={"x-request-id": "request-1"})
    captured_payload: list[dict] = []
    captured_response: list[ProviderResponse] = []

    async def on_payload(payload, _model):
        captured_payload.append(payload)
        return {
            **payload,
            "topP": 0.9,
            "randomSeed": 42,
            "responseFormat": {
                "type": "json_schema",
                "jsonSchema": {
                    "name": "result",
                    "schemaDefinition": {"type": "object", "properties": {"maxTokens": {"type": "number"}}},
                },
            },
            "presencePenalty": 0.1,
            "frequencyPenalty": 0.2,
            "parallelToolCalls": True,
            "safePrompt": True,
        }

    async def on_response(response, _model):
        captured_response.append(response)

    message = await stream_mistral(
        MODEL,
        normalize_context(context),
        MistralOptions(
            api_key="secret",
            client=client,
            headers={"x-custom": "value"},
            max_tokens=123,
            prompt_mode="reasoning",
            reasoning_effort="high",
            tool_choice={"type": "function", "function": {"name": "lookup"}},
            session_id="session-1",
            on_payload=on_payload,
            on_response=on_response,
        ),
    ).result()

    assert message.stop_reason == "stop"
    request = client.requests[0]
    assert request["url"] == "https://api.mistral.ai/v1/chat/completions"
    assert request["headers"]["authorization"] == "Bearer secret"
    assert request["headers"]["accept"] == "text/event-stream"
    assert request["headers"]["x-affinity"] == "session-1"
    assert request["headers"]["x-custom"] == "value"
    assert request["headers"]["user-agent"] == format_user_agent()
    assert captured_payload[0]["maxTokens"] == 123
    assert captured_payload[0]["promptMode"] == "reasoning"
    assert captured_payload[0]["promptCacheKey"] == "session-1"
    assert captured_response == [
        ProviderResponse(status=200, headers={"content-type": "text/event-stream", "x-request-id": "request-1"})
    ]

    wire_payload = request["payload"]
    assert wire_payload["max_tokens"] == 123
    assert wire_payload["prompt_mode"] == "reasoning"
    assert wire_payload["reasoning_effort"] == "high"
    assert wire_payload["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
    assert wire_payload["prompt_cache_key"] == "session-1"
    assert wire_payload["top_p"] == 0.9
    assert wire_payload["random_seed"] == 42
    assert wire_payload["presence_penalty"] == 0.1
    assert wire_payload["frequency_penalty"] == 0.2
    assert wire_payload["parallel_tool_calls"] is True
    assert wire_payload["safe_prompt"] is True
    assert wire_payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "result",
            "schema": {"type": "object", "properties": {"maxTokens": {"type": "number"}}},
        },
    }
    assert "maxTokens" not in wire_payload
    assert "promptMode" not in wire_payload
    assert "promptCacheKey" not in wire_payload
    assert wire_payload["messages"] == [
        {"role": "system", "content": "Be precise"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": "data:image/png;base64,aGVsbG8="},
            ],
        },
    ]


@pytest.mark.tonio
async def test_serializes_assistant_thinking_tool_calls_and_tool_results_for_replay():
    context = Context(
        messages=[
            AssistantMessage(
                api="mistral-conversations",
                provider="mistral",
                model=MODEL.id,
                content=[
                    ThinkingContent(thinking="reason"),
                    TextContent(text="answer"),
                    ToolCall(id="abc123456", name="lookup", arguments={"query": "pi"}),
                ],
                usage=Usage(),
                stop_reason="toolUse",
                timestamp=1,
            ),
            ToolResultMessage(
                tool_call_id="abc123456",
                tool_name="lookup",
                content=[TextContent(text="found"), ImageContent(data="aGVsbG8=", mime_type="image/png")],
                is_error=False,
                timestamp=2,
            ),
        ]
    )
    client = sse_client([terminal_event()])

    message = await stream_mistral(
        MODEL, normalize_context(context), MistralOptions(api_key="test", client=client)
    ).result()

    assert message.stop_reason == "stop"
    assert client.requests[0]["payload"]["messages"] == [
        {
            "role": "assistant",
            "prefix": False,
            "content": [
                {"type": "thinking", "thinking": [{"type": "text", "text": "reason"}]},
                {"type": "text", "text": "answer"},
            ],
            "tool_calls": [
                {
                    "id": "abc123456",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"query":"pi"}'},
                    "index": 0,
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "abc123456",
            "name": "lookup",
            "content": [
                {"type": "text", "text": "found"},
                {"type": "image_url", "image_url": "data:image/png;base64,aGVsbG8="},
            ],
        },
    ]


@pytest.mark.tonio
async def test_parses_native_thinking_text_fragmented_tool_calls_and_cached_token_usage():
    context = Context(messages=[UserMessage(content="hello", timestamp=1)])
    events = [
        {
            "id": "response-1",
            "model": MODEL.id,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {"content": [{"type": "thinking", "thinking": [{"type": "text", "text": "reason"}]}]},
                }
            ],
        },
        {
            "id": "response-1",
            "model": MODEL.id,
            "choices": [
                {"index": 0, "finish_reason": None, "delta": {"content": [{"type": "text", "text": "answer"}]}}
            ],
        },
        {
            "id": "response-1",
            "model": MODEL.id,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {
                        "tool_calls": [
                            {"id": "abc123456", "index": 0, "function": {"name": "lookup", "arguments": '{"query":'}}
                        ]
                    },
                }
            ],
        },
        {
            "id": "response-1",
            "model": MODEL.id,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "delta": {"tool_calls": [{"index": 0, "function": {"name": "", "arguments": '"pi"}'}}]},
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
                "prompt_tokens_details": {"cached_tokens": 3},
            },
        },
    ]

    message = await stream_mistral(
        MODEL, normalize_context(context), MistralOptions(api_key="test", client=sse_client(events))
    ).result()

    assert message.stop_reason == "toolUse"
    assert message.raw_stop_reason == "tool_calls"
    assert message.response_id == "response-1"
    assert message.content == [
        ThinkingContent(thinking="reason"),
        TextContent(text="answer"),
        ToolCall(id="abc123456", name="lookup", arguments={"query": "pi"}),
    ]
    assert message.usage.input == 7
    assert message.usage.output == 4
    assert message.usage.cache_read == 3
    assert message.usage.cache_write == 0
    assert message.usage.total_tokens == 14


# #9674: GLM models on Mistral send empty content deltas at the start, around tool calls,
# and sometimes mid-thinking. They must not open blocks or split thinking.
@pytest.mark.tonio
async def test_ignores_empty_content_deltas():
    model = get_builtin_model("mistral", "zai-glm-5-3")
    context = normalize_context(Context(messages=[UserMessage(content="hello", timestamp=1)]))

    def thinking(text: str) -> dict:
        return {"type": "thinking", "thinking": [{"type": "text", "text": text}]}

    def tool_call(args: str, first: bool) -> dict:
        return {
            "index": 0,
            **({"id": "abc123456"} if first else {}),
            "function": {"name": "read" if first else "", "arguments": args},
        }

    deltas = [
        {"content": ""},
        {"content": [thinking("first part,")]},
        {"content": ""},
        {"content": [{"type": "text", "text": ""}]},
        {"content": [thinking(" second part."), {"type": "text", "text": "Reading."}]},
        {"content": "", "tool_calls": [tool_call("", True)]},
        {"content": "", "tool_calls": [tool_call('{"path":', False)]},
        {"content": "", "tool_calls": [tool_call('"a.txt"}', False)]},
        {"content": ""},
    ]
    events = [
        {
            "id": "response-1",
            "model": model.id,
            "choices": [{"index": 0, "finish_reason": "tool_calls" if i == len(deltas) - 1 else None, "delta": delta}],
        }
        for i, delta in enumerate(deltas)
    ]

    message = await stream_mistral(model, context, MistralOptions(api_key="test", client=sse_client(events))).result()

    assert message.content == [
        ThinkingContent(thinking="first part, second part."),
        TextContent(text="Reading."),
        ToolCall(id="abc123456", name="read", arguments={"path": "a.txt"}),
    ]


@pytest.mark.tonio
async def test_forwards_each_parsed_sse_payload_before_normalizing_it():
    context = normalize_context(Context(messages=[UserMessage(content="hello", timestamp=1)]))
    events = [
        {
            "id": "response-1",
            "provider_metadata": {"request": "test"},
            "choices": [{"index": 0, "finish_reason": None, "delta": {"content": "hello"}}],
        },
        {
            "id": "response-1",
            "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    ]
    received: list = []
    event_models: list = []

    async def on_provider_stream_event(event, event_model) -> None:
        received.append(event)
        event_models.append(event_model)

    result = await stream_mistral(
        MODEL,
        context,
        MistralOptions(api_key="test", client=sse_client(events), on_provider_stream_event=on_provider_stream_event),
    ).result()

    assert received == events
    assert event_models == [MODEL, MODEL]
    assert result.stop_reason == "stop"
    assert result.content == [TextContent(text="hello")]


@pytest.mark.tonio
async def test_parses_sse_and_utf8_sequences_split_across_transport_chunks():
    context = Context(messages=[UserMessage(content="hello", timestamp=1)])
    event = {
        "id": "response-bytewise",
        "model": MODEL.id,
        "choices": [{"index": 0, "finish_reason": "stop", "delta": {"content": "héllo 🌍"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    body = sse_body([json.dumps(event, ensure_ascii=False)])
    client = FakeMistralClient(FakeMistralResponse(chunks=[bytes([byte]) for byte in body]))

    message = await stream_mistral(
        MODEL, normalize_context(context), MistralOptions(api_key="test", client=client)
    ).result()

    assert message.stop_reason == "stop"
    assert message.content == [TextContent(text="héllo 🌍")]


@pytest.mark.tonio
async def test_honors_case_insensitive_header_overrides_and_explicit_affinity_suppression():
    model = replace(MODEL, headers={"Authorization": "Bearer model-key", "X-Affinity": "model-affinity"})
    context = Context(messages=[UserMessage(content="hello", timestamp=1)])
    client = sse_client([terminal_event()])

    await stream_mistral(
        model,
        normalize_context(context),
        MistralOptions(
            api_key="request-key",
            client=client,
            session_id="automatic-affinity",
            headers={"authorization": None, "x-affinity": None, "User-Agent": "custom-agent"},
        ),
    ).result()

    request_headers = client.requests[0]["headers"]
    assert "authorization" not in request_headers
    assert "x-affinity" not in request_headers
    assert request_headers["user-agent"] == "custom-agent"


@pytest.mark.tonio
async def test_aborts_while_waiting_for_an_sse_chunk():
    context = Context(messages=[UserMessage(content="hello", timestamp=1)])
    client = FakeMistralClient(FakeMistralResponse(stall_forever=True))
    cancel = CancelToken()

    result = stream_mistral(
        MODEL, normalize_context(context), MistralOptions(api_key="test", client=client, cancel=cancel)
    ).result()
    cancel.cancel()
    message = await result

    assert message.stop_reason == "aborted"


@pytest.mark.tonio
async def test_applies_the_request_timeout_while_waiting_for_response_headers():
    context = Context(messages=[UserMessage(content="hello", timestamp=1)])

    async def fetch(_request, *, env=None):
        # Never answers: only the header deadline ends the wait.
        await tonio.Event().wait()

    message = await stream_mistral(
        MODEL, normalize_context(context), MistralOptions(api_key="test", fetch=fetch, timeout_ms=5)
    ).result()

    assert message.stop_reason == "error"
    assert message.error_message == "Mistral response headers timed out after 5ms"


# Regression test for #10609: an active stream must not be cut off after timeout_ms.
@pytest.mark.tonio
async def test_does_not_abort_an_active_stream_that_lasts_longer_than_the_request_timeout():
    context = Context(messages=[UserMessage(content="hello", timestamp=1)])
    thinking_event = {
        "choices": [
            {"index": 0, "delta": {"content": [{"type": "thinking", "thinking": [{"type": "text", "text": "x"}]}]}}
        ]
    }
    requests = []

    async def fetch(request, *, env=None):
        requests.append(request)
        body = sse_body([json.dumps(thinking_event)] * 5 + [json.dumps(terminal_event())])
        return http.Response(200, headers={"content-type": "text/event-stream"}, content=body, request=request)

    message = await stream_mistral(
        MODEL, normalize_context(context), MistralOptions(api_key="test", fetch=fetch, timeout_ms=20)
    ).result()

    assert message.stop_reason == "stop"
    assert message.content == [ThinkingContent(thinking="xxxxx")]


# pidrei-only: each read is bounded by the caller's timeout (the HTTP idle timeout pi applies to
# body stalls), else by the streaming default; pi's 60 s header default never bounds body reads.
@pytest.mark.tonio
@pytest.mark.parametrize(("timeout_ms", "read_s"), [(20, 0.02), (None, http.request_timeout(None).read)])
async def test_bounds_each_read_by_the_callers_timeout_or_the_streaming_default(timeout_ms, read_s):
    context = Context(messages=[UserMessage(content="hello", timestamp=1)])
    requests = []

    async def fetch(request, *, env=None):
        requests.append(request)
        body = sse_body([json.dumps(terminal_event())])
        return http.Response(200, headers={"content-type": "text/event-stream"}, content=body, request=request)

    message = await stream_mistral(
        MODEL, normalize_context(context), MistralOptions(api_key="test", fetch=fetch, timeout_ms=timeout_ms)
    ).result()

    assert message.stop_reason == "stop"
    assert requests[0].timeout.read == read_s
    assert requests[0].timeout.total is None


# pidrei-only: with a caller timeout, the request's read timeout bounds the head at the same deadline.
@pytest.mark.tonio
async def test_reports_a_head_read_timeout_as_a_response_headers_timeout():
    context = Context(messages=[UserMessage(content="hello", timestamp=1)])

    async def fetch(request, *, env=None):
        raise http.RequestTimeout("Timed out waiting for the response", request=request)

    message = await stream_mistral(
        MODEL, normalize_context(context), MistralOptions(api_key="test", fetch=fetch, timeout_ms=5_000)
    ).result()

    assert message.stop_reason == "error"
    assert message.error_message == "Mistral response headers timed out after 5000ms"


@pytest.mark.tonio
async def test_preserves_http_status_and_response_bodies_in_errors():
    context = Context(messages=[UserMessage(content="hello", timestamp=1)])
    client = FakeMistralClient(FakeMistralResponse(b'{"message":"blocked by gateway"}', status_code=403))

    message = await stream_mistral(
        MODEL, normalize_context(context), MistralOptions(api_key="test", client=client)
    ).result()

    assert message.stop_reason == "error"
    assert message.error_message == 'Mistral API error (403): {"message":"blocked by gateway"}'
