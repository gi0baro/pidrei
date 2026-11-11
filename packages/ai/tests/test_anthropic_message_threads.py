"""Mirror of pi ai test/anthropic-message-threads.test.ts.

pi passes a custom `fetch` that plays a fake Anthropic endpoint with real
`Response`s; here the `fetch` option answers with `http.Response`s and the
adapter's punkreq client parses them, as in
`test_anthropic_strict_tool_fallback.py`. pi's `AbortController` is a
`CancelToken` fired from `on_provider_stream_event`. pi's `afterEach`
`cleanupSessionResources()` is the module's autouse fixture.
"""

import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from pidrei_ai.api.anthropic_messages import AnthropicOptions, stream
from pidrei_ai.types import (
    AnthropicMessagesCompat,
    AssistantMessage,
    Context,
    JsonSchemaConstrainedSampling,
    Message,
    Model,
    ModelCost,
    SystemMessage,
    TextContent,
    Tool,
    ToolResultMessage,
    UserMessage,
)
from pidrei_ai.utils.session_resources import cleanup_session_resources
from pidrei_ai.utils.transcript import normalize_context
from pidrei_http import http
from pidrei_utils.cancel import CancelToken


THREADS_BETA = "message-threads-2026-08-12"
GRAMMAR_ERROR = (
    "The compiled grammar is too large, which would cause performance issues. Simplify your tool schemas or "
    "reduce the number of strict tools."
)


@dataclass(slots=True)
class CapturedRequest:
    body: dict[str, Any]
    betas: list[str]


@dataclass(slots=True)
class ReplySpec:
    id: str | None = None
    model: str | None = None
    text: str | None = None
    thinking: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    stop_reason: str | None = None
    # Event to emit instead of finishing normally: "error_event" or "no_message_stop".
    end: str | None = None


def _sse_body(events: list[dict[str, Any]]) -> str:
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)


def reply(request: http.Request, captured: CapturedRequest, spec: ReplySpec | None = None) -> http.Response:
    spec = spec if spec is not None else ReplySpec()
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": spec.id if spec.id is not None else "msg_x",
                "model": spec.model if spec.model is not None else captured.body["model"],
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        }
    ]
    index = 0
    if spec.thinking is not None:
        events.extend(
            [
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "thinking_delta", "thinking": spec.thinking},
                },
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "signature_delta", "signature": "sig"},
                },
                {"type": "content_block_stop", "index": index},
            ]
        )
        index += 1
    events.extend(
        [
            {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": spec.text if spec.text is not None else "answer"},
            },
            {"type": "content_block_stop", "index": index},
        ]
    )
    index += 1
    for call in spec.tool_calls or []:
        events.extend(
            [
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "tool_use", "id": call["id"], "name": call["name"], "input": {}},
                },
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(call["input"])},
                },
                {"type": "content_block_stop", "index": index},
            ]
        )
        index += 1
    if spec.end == "error_event":
        error = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
        body = f"{_sse_body(events)}event: error\ndata: {json.dumps(error)}\n\n"
        return http.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode(), request=request)
    stop_reason = spec.stop_reason or ("tool_use" if spec.tool_calls else "end_turn")
    events.append({"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 1}})
    if spec.end != "no_message_stop":
        events.append({"type": "message_stop"})
    return http.Response(
        200, headers={"content-type": "text/event-stream"}, content=_sse_body(events).encode(), request=request
    )


def error_response(
    request: http.Request,
    status: int,
    message: str,
    error_code: str | None = None,
    headers: dict[str, str] | None = None,
) -> http.Response:
    type = "not_found_error" if status == 404 else "rate_limit_error" if status == 429 else "invalid_request_error"
    error: dict[str, Any] = {"type": type, "message": message}
    if error_code:
        error["details"] = {"error_code": error_code}
    return http.Response(
        status,
        headers={"content-type": "application/json", **(headers or {})},
        json={"type": "error", "error": error},
        request=request,
    )


type Responder = Callable[[http.Request, CapturedRequest], http.Response]

_session_counter = itertools.count(1)


def fingerprint(captured: CapturedRequest) -> str:
    """The fields Anthropic requires to stay the same within a thread (observed live)."""
    body = captured.body
    return json.dumps(
        [
            body.get("model"),
            body.get("thinking"),
            body.get("output_config"),
            body.get("tool_choice"),
            "fallbacks" in body,
            captured.betas,
        ]
    )


@dataclass(slots=True)
class Harness:
    """Fake Anthropic endpoint: records request bodies and beta lists, and answers from a queue or
    with a plain reply. Like the real server, it rejects a `continue` whose fixed fields differ
    from the thread's."""

    session_id: str
    requests: list[CapturedRequest] = field(default_factory=list)
    queue: list[Responder] = field(default_factory=list)
    thread_fingerprints: dict[str, str] = field(default_factory=dict)

    async def fetch(self, request: http.Request, *, env=None) -> http.Response:
        body = json.loads(request.content)
        beta = request.headers.get("anthropic-beta")
        captured = CapturedRequest(body=body, betas=beta.split(",") if beta else [])
        self.requests.append(captured)
        if self.queue:
            return self.queue.pop(0)(request, captured)
        id = f"msg_{len(self.requests)}"
        previous = (body.get("thread") or {}).get("previous_message_id")
        expected = fingerprint(captured) if previous is None else self.thread_fingerprints.get(previous)
        if expected != fingerprint(captured):
            return error_response(request, 400, "fixed field changed", "thread_fingerprint_mismatch")
        if body.get("thread"):
            self.thread_fingerprints[id] = expected
        return reply(request, captured, ReplySpec(id=id))

    def options(self) -> dict[str, Any]:
        return {"api_key": "test-key", "fetch": self.fetch, "session_id": self.session_id, "max_retries": 0}


def harness() -> Harness:
    return Harness(session_id=f"thread-test-{next(_session_counter)}")


BASE_MODEL = Model(
    id="claude-test",
    name="Test Claude",
    api="anthropic-messages",
    provider="anthropic",
    base_url="https://api.anthropic.com",
    reasoning=False,
    input=["text"],
    cost=ModelCost(),
    context_window=200000,
    max_tokens=100,
)


def model_with(**overrides) -> Model:
    return replace(BASE_MODEL, **overrides)


def tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"Tool {name}",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
    )


def user(text: str) -> UserMessage:
    return UserMessage(content=text, timestamp=1)


def tool_result(id: str, name: str) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=id, tool_name=name, content=[TextContent(text="result")], is_error=False, timestamp=1
    )


@dataclass(slots=True)
class TurnInput:
    model: Model | None = None
    system_prompt: str | None = None
    tools: list[Tool] | None = None
    options: dict[str, Any] | None = None


async def turn(h: Harness, messages: list[Message], input: TurnInput | None = None) -> AssistantMessage:
    input = input if input is not None else TurnInput()
    context = normalize_context(
        Context(
            system_prompt=input.system_prompt if input.system_prompt is not None else "You are a test assistant.",
            tools=input.tools if input.tools is not None else [tool("lookup")],
            messages=list(messages),
        )
    )
    options = AnthropicOptions(**{**h.options(), **(input.options or {})})
    return await stream(input.model if input.model is not None else BASE_MODEL, context, options).result()


async def say(h: Harness, messages: list[Message], input: TurnInput | None = None) -> AssistantMessage:
    """Runs one turn and appends the reply to the conversation."""
    result = await turn(h, messages, input)
    messages.append(result)
    return result


def thread_of(captured: CapturedRequest | None) -> dict | None:
    return captured.body.get("thread") if captured is not None else None


def _assert_matches(actual, expected) -> None:
    """pi's `toMatchObject`: a subset check (arrays match element by element)."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), actual
        for key, value in expected.items():
            assert key in actual, key
            _assert_matches(actual[key], value)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), actual
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_matches(actual_item, expected_item)
    else:
        assert actual == expected


@pytest.fixture(autouse=True)
def _cleanup_session_resources():
    yield
    cleanup_session_resources()


@pytest.mark.tonio
async def test_creates_a_thread_then_continues_it_with_only_the_new_messages():
    h = harness()
    messages: list[Message] = [user("first")]
    await say(h, messages)
    messages.append(user("second"))
    result = await say(h, messages)

    assert result.stop_reason == "stop"
    create, next_request = h.requests
    assert create.body["thread"] == {"type": "create"}
    assert "system" in create.body
    assert "tools" in create.body
    assert next_request.body["thread"] == {"type": "continue", "previous_message_id": "msg_1"}
    assert "system" not in next_request.body
    assert "tools" not in next_request.body
    # The first user message had string content turned into a cached block on request 1,
    # and plain string content on request 2; the cache marker moved to the new message.
    assert create.body["messages"][0]["content"] == [
        {"type": "text", "text": "first", "cache_control": {"type": "ephemeral"}}
    ]
    assert next_request.body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "second", "cache_control": {"type": "ephemeral"}}]}
    ]
    assert THREADS_BETA in create.betas
    assert next_request.betas == create.betas


@pytest.mark.tonio
async def test_continues_after_a_reply_that_ends_in_tool_calls_without_placeholder_results_in_the_prefix():
    h = harness()
    h.queue.append(
        lambda request, captured: reply(
            request,
            captured,
            ReplySpec(id="msg_tool", tool_calls=[{"id": "toolu_1", "name": "lookup", "input": {"value": "x"}}]),
        )
    )
    messages: list[Message] = [user("look it up")]
    first = await say(h, messages)
    assert first.stop_reason == "toolUse"
    messages.append(tool_result("toolu_1", "lookup"))
    await say(h, messages)

    next_request = h.requests[1]
    assert next_request.body["thread"] == {"type": "continue", "previous_message_id": "msg_tool"}
    assert len(next_request.body["messages"]) == 1
    _assert_matches(
        next_request.body["messages"][0],
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1"}]},
    )


def _effort_or_role(message: dict) -> str:
    return (message.get("output_config") or {}).get("effort") or message["role"]


@pytest.mark.tonio
async def test_continues_when_the_effort_level_changes_between_turns():
    model = model_with(
        reasoning=True,
        compat=AnthropicMessagesCompat(force_adaptive_thinking=True, supports_mid_convo_effort=True),
    )
    h = harness()
    h.queue.append(lambda request, captured: reply(request, captured, ReplySpec(id="msg_a", thinking="thought")))
    messages: list[Message] = [user("first")]
    await say(h, messages, TurnInput(model=model, options={"thinking_enabled": True, "effort": "high"}))
    messages.append(user("second"))
    await say(h, messages, TurnInput(model=model, options={"thinking_enabled": True, "effort": "low"}))

    assert [_effort_or_role(message) for message in h.requests[0].body["messages"]] == ["user", "high"]
    next_request = h.requests[1]
    assert next_request.body["thread"] == {"type": "continue", "previous_message_id": "msg_a"}
    assert [_effort_or_role(message) for message in next_request.body["messages"]] == ["user", "low"]


@pytest.mark.tonio
async def test_continues_when_a_system_message_with_a_native_tool_change_follows_the_reply():
    model = model_with(
        compat=AnthropicMessagesCompat(supports_mid_convo_system_messages=True, supports_mid_convo_tool_changes=True)
    )
    h = harness()
    messages: list[Message] = [user("first")]
    await say(h, messages, TurnInput(model=model))
    messages.extend([SystemMessage(content="Update", tools_added=[tool("other")], timestamp=2), user("second")])
    await say(h, messages, TurnInput(model=model))

    next_request = h.requests[1]
    assert next_request.body["thread"]["type"] == "continue"
    assert [message["role"] for message in next_request.body["messages"]] == ["user", "system"]
    assert "tool_addition" in json.dumps(next_request.body["messages"][1]["content"])


@pytest.mark.tonio
@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(TurnInput(), TurnInput(system_prompt="Different prompt."), id="system prompt"),
        pytest.param(TurnInput(), TurnInput(tools=[tool("lookup"), tool("other")]), id="tools"),
        pytest.param(TurnInput(), TurnInput(model=model_with(id="claude-other")), id="model"),
        pytest.param(TurnInput(), TurnInput(options={"tool_choice": "any"}), id="tool choice"),
        pytest.param(TurnInput(), TurnInput(options={"cache_retention": "long"}), id="cache retention"),
        pytest.param(
            TurnInput(
                model=model_with(reasoning=True), options={"thinking_enabled": True, "thinking_budget_tokens": 1024}
            ),
            TurnInput(
                model=model_with(reasoning=True), options={"thinking_enabled": True, "thinking_budget_tokens": 2048}
            ),
            id="thinking level",
        ),
    ],
)
async def test_creates_a_new_thread_when_a_setting_changes(before, after):
    h = harness()
    messages: list[Message] = [user("first")]
    await say(h, messages, before)
    messages.append(user("second"))
    await say(h, messages, after)
    assert thread_of(h.requests[1]) == {"type": "create"}
    assert len(h.requests[1].body["messages"]) == 3


@pytest.mark.tonio
async def test_creates_a_new_thread_when_the_request_has_no_messages_beyond_the_stored_prefix():
    h = harness()
    messages: list[Message] = [user("first")]
    await say(h, messages)
    await turn(h, messages)
    assert thread_of(h.requests[1]) == {"type": "create"}


@pytest.mark.tonio
@pytest.mark.parametrize(
    ("error_code", "status"),
    [
        ("thread_not_found", 404),
        ("thread_already_continued", 400),
        ("thread_continue_in_flight", 400),
        ("thread_fingerprint_mismatch", 400),
    ],
)
async def test_starts_a_new_thread_with_the_full_request_after_a_thread_rejection(error_code, status):
    h = harness()
    messages: list[Message] = [user("first")]
    await say(h, messages)
    messages.append(user("second"))
    h.queue.append(lambda request, captured: error_response(request, status, "thread rejected", error_code))
    payload_calls: list[Any] = []

    async def on_payload(payload, _model):
        payload_calls.append(payload)

    result = await say(h, messages, TurnInput(options={"on_payload": on_payload}))

    assert result.stop_reason == "stop"
    assert len(payload_calls) == 1
    assert len(h.requests) == 3
    _, rejected, resend = h.requests
    assert rejected.body["thread"]["type"] == "continue"
    assert resend.body["thread"] == {"type": "create"}
    assert len(resend.body["messages"]) == 3
    assert "system" in resend.body
    assert resend.betas == rejected.betas
    assert result.diagnostics is not None and len(result.diagnostics) == 1
    assert result.diagnostics[0].type == "anthropic_thread_fallback"
    assert result.diagnostics[0].details == {"reason": error_code}

    messages.append(user("third"))
    await say(h, messages)
    assert thread_of(h.requests[3]) == {"type": "continue", "previous_message_id": "msg_3"}


@pytest.mark.tonio
@pytest.mark.parametrize(
    ("error_code", "message"),
    [
        pytest.param("thread_unsupported_request", "rejected", id="thread_unsupported_request"),
        pytest.param(
            None,
            "thread: Extra inputs are not permitted",
            id="a 400 without error code about the thread field",
        ),
    ],
)
async def test_turns_threads_off_for_the_session_after_an_unsupported_continue(error_code, message):
    h = harness()
    messages: list[Message] = [user("first")]
    await say(h, messages)
    messages.append(user("second"))
    h.queue.append(lambda request, captured: error_response(request, 400, message, error_code))
    result = await say(h, messages)

    assert result.stop_reason == "stop"
    assert result.diagnostics
    assert result.diagnostics[0].type == "anthropic_thread_fallback"
    assert result.diagnostics[0].details == {"reason": error_code if error_code is not None else "400"}
    messages.append(user("third"))
    await say(h, messages)
    later = h.requests[3]
    assert "thread" not in later.body
    # The resend was accepted with the threads beta, so the beta list stays the same.
    assert later.betas == h.requests[0].betas


@pytest.mark.tonio
async def test_shows_the_error_when_the_resend_fails_too_and_keeps_threads_on():
    h = harness()
    messages: list[Message] = [user("first")]
    await say(h, messages)
    messages.append(user("second"))
    h.queue.extend(
        [
            lambda request, captured: error_response(request, 400, "thread: Extra inputs are not permitted"),
            lambda request, captured: error_response(request, 400, "Real problem"),
        ]
    )
    failed = await turn(h, messages)
    assert failed.stop_reason == "error"
    assert "Real problem" in failed.error_message
    assert len(h.requests) == 3

    await turn(h, messages)
    assert thread_of(h.requests[3]) == {"type": "create"}


@pytest.mark.tonio
@pytest.mark.parametrize(
    ("status", "message"),
    [(400, "prompt is too long: 250000 tokens > 200000 maximum"), (404, "model: claude-test")],
)
async def test_passes_an_error_that_is_not_about_the_thread_on_without_resending(status, message):
    h = harness()
    messages: list[Message] = [user("first")]
    await say(h, messages)
    messages.append(user("second"))
    h.queue.append(lambda request, captured: error_response(request, status, message))
    failed = await turn(h, messages)
    assert failed.stop_reason == "error"
    assert message in failed.error_message
    assert len(h.requests) == 2
    assert failed.diagnostics is None

    # Threads stay on: after a fix such as compaction, the next request starts a new thread.
    await turn(h, messages)
    assert thread_of(h.requests[2]) == {"type": "create"}


@pytest.mark.tonio
async def test_leaves_strict_tool_rejections_to_the_strict_tool_fallback():
    model = model_with(compat=AnthropicMessagesCompat(supports_strict_tools=True))
    strict_tool = Tool(
        name="lookup",
        description="Tool lookup",
        parameters=tool("lookup").parameters,
        constrained_sampling=JsonSchemaConstrainedSampling(strict="prefer"),
    )
    h = harness()
    h.queue.append(lambda request, captured: error_response(request, 400, GRAMMAR_ERROR))
    result = await turn(h, [user("first")], TurnInput(model=model, tools=[strict_tool]))
    assert result.stop_reason == "stop"
    assert len(h.requests) == 2
    assert [(thread_of(request) or {}).get("type") for request in h.requests] == ["create", "create"]
    assert [diagnostic.type for diagnostic in result.diagnostics or []] == ["anthropic_strict_tools_fallback"]


@pytest.mark.tonio
async def test_starts_a_new_thread_when_a_retried_continue_finds_the_thread_in_flight():
    h = harness()
    messages: list[Message] = [user("first")]
    await say(h, messages)
    messages.append(user("second"))
    h.queue.extend(
        [
            lambda request, captured: error_response(request, 503, "overloaded", None, {"retry-after-ms": "0"}),
            lambda request, captured: error_response(request, 400, "in flight", "thread_continue_in_flight"),
        ]
    )
    result = await say(h, messages, TurnInput(options={"max_retries": 1}))
    assert result.stop_reason == "stop"
    assert [(thread_of(request) or {}).get("type") for request in h.requests] == [
        "create",
        "continue",
        "continue",
        "create",
    ]


@pytest.mark.tonio
@pytest.mark.parametrize(
    "header",
    [
        pytest.param("fine-grained-tool-streaming-2025-05-14", id="without the threads beta"),
        pytest.param(None, id="set to null"),
    ],
)
async def test_fails_once_then_turns_threads_off_with_an_explicit_anthropic_beta_header(header):
    h = harness()

    def respond(request: http.Request, captured: CapturedRequest) -> http.Response:
        if captured.body.get("thread") and THREADS_BETA not in captured.betas:
            return error_response(request, 400, "thread: Extra inputs are not permitted")
        return reply(request, captured, ReplySpec(id=f"msg_{len(h.requests)}"))

    h.queue.extend([respond, respond, respond])
    options = {"headers": {"anthropic-beta": header}}
    messages: list[Message] = [user("first")]
    result = await say(h, messages, TurnInput(options=options))
    assert result.stop_reason == "stop"
    messages.append(user("second"))
    await say(h, messages, TurnInput(options=options))
    assert [(thread_of(request) or {}).get("type") for request in h.requests] == ["create", None, None]


@pytest.mark.tonio
@pytest.mark.parametrize(
    "spec",
    [
        pytest.param("abort", id="an aborted reply"),
        pytest.param(ReplySpec(end="error_event"), id="a stream error"),
        pytest.param(ReplySpec(end="no_message_stop"), id="a stream that ends early"),
        pytest.param(ReplySpec(stop_reason="refusal"), id="a refusal"),
        pytest.param(ReplySpec(model="claude-fallback"), id="a fallback model reply"),
    ],
)
async def test_does_not_record_an_unclean_reply(spec):
    h = harness()
    cancel = CancelToken()
    if spec != "abort":
        h.queue.append(lambda request, captured: reply(request, captured, replace(spec, id="msg_bad")))

    async def on_provider_stream_event(event, _model):
        if event.get("type") == "message_start":
            cancel.cancel()

    messages: list[Message] = [user("first")]
    first = await turn(
        h,
        messages,
        TurnInput(
            options={"cancel": cancel, "on_provider_stream_event": on_provider_stream_event} if spec == "abort" else {}
        ),
    )
    messages.extend([first, user("second")])
    await turn(h, messages)
    assert thread_of(h.requests[1]) == {"type": "create"}


@pytest.mark.tonio
async def test_does_not_use_threads_with_cache_retention_none():
    h = harness()
    messages: list[Message] = [user("first")]
    await say(h, messages, TurnInput(options={"cache_retention": "none"}))
    assert "thread" not in h.requests[0].body
    assert THREADS_BETA not in h.requests[0].betas


@pytest.mark.tonio
async def test_matches_claude_code_tool_names_across_turns_with_oauth():
    h = harness()
    h.queue.append(
        lambda request, captured: reply(
            request,
            captured,
            ReplySpec(id="msg_oauth", tool_calls=[{"id": "toolu_1", "name": "Read", "input": {"value": "x"}}]),
        )
    )
    options = {"api_key": "sk-ant-oat-test"}
    messages: list[Message] = [user("read it")]
    first = await say(h, messages, TurnInput(tools=[tool("read")], options=options))
    tool_call = next(block for block in first.content if block.type == "toolCall")
    assert tool_call.name == "read"
    messages.append(tool_result("toolu_1", "read"))
    await say(h, messages, TurnInput(tools=[tool("read")], options=options))
    assert thread_of(h.requests[1]) == {"type": "continue", "previous_message_id": "msg_oauth"}


@pytest.mark.tonio
async def test_sends_cache_warming_replays_without_a_thread_and_keeps_the_conversations_thread():
    h = harness()
    messages: list[Message] = [user("first")]
    await say(h, messages)
    await turn(h, messages[:1], TurnInput(options={"max_tokens": 1, "disable_continuation": True}))
    messages.append(user("second"))
    await say(h, messages)

    real, warm, next_request = h.requests
    assert "thread" not in warm.body
    assert warm.body["max_tokens"] == 1
    assert warm.body["messages"] == real.body["messages"]
    assert warm.betas == real.betas
    assert next_request.body["thread"] == {"type": "continue", "previous_message_id": "msg_1"}
