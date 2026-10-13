"""Mirror of pi ai test/anthropic-strict-tool-fallback.test.ts.

pi passes a custom `fetch` that answers with real `Response`s, so the SDK's
own HTTP error parsing runs; here the `fetch` option answers with
`http.Response`s and the adapter's punkreq client parses them. pi's
"preserves prefer through pi-messages" case is not mirrored: the pi-messages
adapter is not ported (radius).
"""

import copy
import json
from dataclasses import replace

import pytest

from pidrei_ai.api.anthropic_messages import AnthropicApiError, AnthropicOptions, stream as stream_anthropic
from pidrei_ai.types import (
    AnthropicMessagesCompat,
    Context,
    JsonSchemaConstrainedSampling,
    Model,
    ModelCost,
    SystemMessage,
    Tool,
    UserMessage,
)
from pidrei_ai.utils.transcript import normalize_context
from pidrei_http import http
from pidrei_utils.cancel import CancelToken


GRAMMAR_ERROR = (
    "The compiled grammar is too large, which would cause performance issues. Simplify your tool schemas or "
    "reduce the number of strict tools."
)

MODEL = Model(
    id="claude-haiku-4-5",
    name="Test Claude",
    api="anthropic-messages",
    provider="anthropic",
    base_url="https://anthropic.example",
    reasoning=False,
    input=["text"],
    cost=ModelCost(),
    context_window=200000,
    max_tokens=100,
    compat=AnthropicMessagesCompat(supports_strict_tools=True),
)

_PARAMETERS = {
    "type": "object",
    "properties": {"value": {"type": "string"}, "optional": {"type": "number"}},
    "required": ["value"],
}


def tool(name: str, strict: str | None = None, parameters: dict | None = None) -> Tool:
    return Tool(
        name=name,
        description="Look up a value",
        parameters=parameters if parameters is not None else _PARAMETERS,
        constrained_sampling=JsonSchemaConstrainedSampling(strict=strict) if strict else None,
    )


def context(tools: list[Tool]):
    return normalize_context(Context(tools=tools, messages=[UserMessage(content="Hello", timestamp=1)]))


def error_response(request, message=GRAMMAR_ERROR, status=400, type="invalid_request_error") -> http.Response:
    return http.Response(
        status,
        headers={"content-type": "application/json"},
        json={"type": "error", "error": {"type": type, "message": message}},
        request=request,
    )


def success_response(request) -> http.Response:
    events = [
        {
            "type": "message_start",
            "message": {"id": "msg_test", "model": MODEL.id, "usage": {"input_tokens": 10, "output_tokens": 0}},
        },
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ]
    body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
    return http.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode(), request=request)


class Harness:
    """pi's `harness`: records each request payload and answers with `respond(request, attempt)`."""

    def __init__(self, respond):
        self.payloads: list[dict] = []
        self._respond = respond

    async def __call__(self, request, *, env=None):
        self.payloads.append(json.loads(request.content))
        return self._respond(request, len(self.payloads))

    def options(self, **overrides) -> AnthropicOptions:
        return AnthropicOptions(**{"api_key": "test-key", "fetch": self, "max_retries": 0, **overrides})


def _assert_matches(actual, expected) -> None:
    """pi's `toMatchObject`: a subset check."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), actual
        for key, value in expected.items():
            assert key in actual, key
            _assert_matches(actual[key], value)
    else:
        assert actual == expected


# Regression tests for https://github.com/earendil-works/pi/issues/10409.


@pytest.mark.tonio
@pytest.mark.parametrize(
    "message",
    [
        GRAMMAR_ERROR,
        # Observed from the live API: 32 strict tools, then 20 tools with 24 optional fields each.
        (
            "Too many strict tools (32). The maximum number of strict tools supported is 20. Try reducing the "
            "number of tools marked as strict."
        ),
        (
            "Schemas contains too many parameters with union types (480 parameters with type arrays or anyOf). "
            "This causes exponential compilation cost. Reduce the number of nullable or union-typed parameters "
            "(limit: 16 parameters with unions)."
        ),
    ],
)
async def test_retries_with_original_non_strict_schemas_and_preserves_required_tools(message):
    preferred = tool("preferred", "prefer")
    required = tool("required", "require")
    ordinary = tool("ordinary")
    transcript = context([preferred, required, ordinary])
    original = copy.deepcopy(transcript)
    harness = Harness(
        lambda request, attempt: error_response(request, message) if attempt == 1 else success_response(request)
    )
    payload_calls = []
    response_calls = []

    async def on_payload(payload, _model):
        payload_calls.append(payload)
        return {**payload, "inference_geo": "us", "stream": False}

    async def on_response(response, _model):
        response_calls.append(response)

    stream = stream_anthropic(
        MODEL,
        transcript,
        harness.options(on_payload=on_payload, on_response=on_response, cache_retention="long"),
    )
    events = [event.type async for event in stream]
    result = await stream.result()

    assert result.stop_reason == "stop"
    assert events == ["start", "done"]
    assert len(harness.payloads) == 2
    assert len(payload_calls) == 2
    assert len(response_calls) == 1
    first = harness.payloads[0]["tools"]
    second = harness.payloads[1]["tools"]
    _assert_matches(
        first[0],
        {
            "strict": True,
            "input_schema": {
                "required": ["value", "optional"],
                "properties": {"optional": {"anyOf": [{"type": "number"}, {"type": "null"}]}},
            },
        },
    )
    assert "strict" not in second[0]
    assert second[0]["input_schema"] == {
        "type": "object",
        "properties": {"value": {"type": "string"}, "optional": {"type": "number"}},
        "required": ["value"],
    }
    assert second[1] == first[1]
    assert second[1]["strict"] is True
    assert second[2] == first[2]
    assert second[2]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert [payload["stream"] for payload in harness.payloads] == [True, True]
    assert harness.payloads[1]["inference_geo"] == "us"
    assert transcript == original
    assert result.diagnostics[0].type == "anthropic_strict_tools_fallback"
    assert result.diagnostics[0].details == {"reason": "strict_tool_limits_exceeded"}


# pidrei-only: pi's `config && ...` skips a tool whose constrained sampling is disabled (`false`).
@pytest.mark.tonio
async def test_retries_a_tool_set_that_also_disables_constrained_sampling_on_a_tool():
    disabled = Tool(name="disabled", description="Look up a value", parameters=_PARAMETERS, constrained_sampling=False)
    harness = Harness(lambda request, attempt: error_response(request) if attempt == 1 else success_response(request))

    result = await stream_anthropic(MODEL, context([tool("preferred", "prefer"), disabled]), harness.options()).result()

    assert result.stop_reason == "stop"
    assert len(harness.payloads) == 2
    assert "strict" not in harness.payloads[1]["tools"][0]
    assert harness.payloads[1]["tools"][1] == harness.payloads[0]["tools"][1]


@pytest.mark.tonio
async def test_downgrades_inline_additions_and_same_name_redefinitions_without_downgrading_require():
    transcript = context([tool("lookup", "require")])
    transcript = replace(
        transcript,
        messages=[
            *transcript.messages,
            SystemMessage(content="", tools_added=[tool("lookup", "prefer")], timestamp=2),
            SystemMessage(content="", tools_added=[tool("lookup", "require")], timestamp=3),
        ],
    )
    harness = Harness(lambda request, attempt: error_response(request) if attempt == 1 else success_response(request))
    model = replace(
        MODEL,
        compat=replace(MODEL.compat, supports_mid_convo_system_messages=True, supports_mid_convo_tool_changes=True),
    )

    result = await stream_anthropic(model, transcript, harness.options(api_key="sk-ant-oat-test")).result()

    assert result.stop_reason == "stop"
    assert len(harness.payloads) == 2
    assert harness.payloads[1]["tools"] == harness.payloads[0]["tools"]
    additions = [
        [
            block["tool"]["definition"]
            for message in payload["messages"]
            if isinstance(message["content"], list)
            for block in message["content"]
            if block["type"] == "tool_addition" and block["tool"]["type"] == "tool_definition"
        ]
        for payload in harness.payloads
    ]
    assert [definition.get("strict") for definition in additions[0]] == [True, True]
    assert [definition.get("strict") for definition in additions[1]] == [None, True]
    assert additions[1][0]["input_schema"]["required"] == ["value"]
    assert additions[1][1] == additions[0][1]


@pytest.mark.tonio
@pytest.mark.parametrize(
    "tools",
    [
        pytest.param([tool("lookup", "require")], id="required"),
        pytest.param([tool("lookup")], id="ordinary"),
        pytest.param(
            [
                tool(
                    "lookup",
                    "prefer",
                    {
                        "type": "object",
                        "properties": {"value": {"type": "number", "minimum": 1}},
                        "required": ["value"],
                    },
                )
            ],
            id="locally unsupported prefer",
        ),
    ],
)
async def test_does_not_retry_a_tool_set_without_preferred_strict_tools(tools):
    harness = Harness(lambda request, _attempt: error_response(request))
    result = await stream_anthropic(MODEL, context(tools), harness.options()).result()
    assert result.stop_reason == "error"
    assert GRAMMAR_ERROR in result.error_message
    assert len(harness.payloads) == 1


@pytest.mark.tonio
@pytest.mark.parametrize(
    ("message", "status", "type"),
    [
        pytest.param("Invalid tool schema", 400, "invalid_request_error", id="unrelated 400"),
        pytest.param(GRAMMAR_ERROR, 400, "authentication_error", id="wrong error type"),
        pytest.param(GRAMMAR_ERROR, 500, "invalid_request_error", id="wrong HTTP status"),
    ],
)
async def test_does_not_downgrade_on_other_errors(message, status, type):
    harness = Harness(lambda request, _attempt: error_response(request, message, status, type))
    result = await stream_anthropic(MODEL, context([tool("lookup", "prefer")]), harness.options()).result()
    assert result.stop_reason == "error"
    assert len(harness.payloads) == 1


@pytest.mark.tonio
async def test_surfaces_a_second_rejection_rather_than_retrying_again():
    harness = Harness(
        lambda request, attempt: error_response(
            request, GRAMMAR_ERROR if attempt == 1 else "Required grammar is still too large"
        )
    )
    result = await stream_anthropic(
        MODEL, context([tool("lookup", "prefer"), tool("required", "require")]), harness.options()
    ).result()
    assert result.stop_reason == "error"
    assert "Required grammar is still too large" in result.error_message
    assert len(harness.payloads) == 2
    assert harness.payloads[1]["tools"][1]["strict"] is True


@pytest.mark.tonio
async def test_does_not_downgrade_a_hook_exception_even_if_it_looks_like_the_provider_rejection():
    harness = Harness(lambda request, _attempt: success_response(request))

    async def on_payload(_payload, _model):
        raise AnthropicApiError(
            status=400,
            headers={},
            message=GRAMMAR_ERROR,
            error={"error": {"type": "invalid_request_error", "message": GRAMMAR_ERROR}},
        )

    result = await stream_anthropic(
        MODEL, context([tool("lookup", "prefer")]), harness.options(on_payload=on_payload)
    ).result()
    assert result.stop_reason == "error"
    assert harness.payloads == []
    assert result.diagnostics is None


@pytest.mark.tonio
async def test_honors_cancellation_before_the_fallback_attempt():
    cancel = CancelToken()

    def respond(request, _attempt):
        cancel.cancel()
        return error_response(request)

    harness = Harness(respond)
    result = await stream_anthropic(MODEL, context([tool("lookup", "prefer")]), harness.options(cancel=cancel)).result()
    assert result.stop_reason == "aborted"
    assert len(harness.payloads) == 1


@pytest.mark.tonio
async def test_does_not_retry_an_sse_error_after_response_headers():
    error_event = {"type": "error", "error": {"type": "invalid_request_error", "message": GRAMMAR_ERROR}}

    def respond(request, _attempt):
        return http.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"event: error\ndata: {json.dumps(error_event)}\n\n".encode(),
            request=request,
        )

    harness = Harness(respond)
    result = await stream_anthropic(MODEL, context([tool("lookup", "prefer")]), harness.options()).result()
    assert result.stop_reason == "error"
    assert len(harness.payloads) == 1


@pytest.mark.tonio
async def test_does_not_disable_strict_sampling_on_later_requests():
    harness = Harness(lambda request, attempt: error_response(request) if attempt == 1 else success_response(request))
    transcript = context([tool("lookup", "prefer")])
    await stream_anthropic(MODEL, transcript, harness.options()).result()
    await stream_anthropic(MODEL, transcript, harness.options()).result()
    assert [payload["tools"][0].get("strict") for payload in harness.payloads] == [True, None, True]
