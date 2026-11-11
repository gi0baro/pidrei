"""Mirror of pi ai test/bedrock-strict-tool-fallback.test.ts.

pi replaces `@aws-sdk/client-bedrock-runtime` with `vi.mock`; here the stub
replaces `api/bedrock_runtime.BedrockRuntimeClient` by name, as in the other
bedrock mirrors. `send` raises `BedrockRuntimeServiceException` for a non-2xx
response, with the body's `message`, as pidrei's runtime does. pi's
`afterEach` `cleanupSessionResources()` is the module's autouse fixture.
"""

import contextlib
import copy
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from pidrei_ai.api import bedrock_converse_stream as bedrock
from pidrei_ai.api.bedrock_converse_stream import BedrockOptions, stream as stream_bedrock
from pidrei_ai.api.bedrock_runtime import BedrockRuntimeServiceException
from pidrei_ai.types import (
    BedrockCompat,
    Context,
    JsonSchemaConstrainedSampling,
    Model,
    ModelCost,
    Tool,
    UserMessage,
)
from pidrei_ai.utils.session_resources import cleanup_session_resources
from pidrei_ai.utils.transcript import normalize_context


MODEL = Model(
    id="us.anthropic.claude-opus-5-5",
    name="Claude Opus 5.5 (US)",
    api="bedrock-converse-stream",
    provider="amazon-bedrock",
    base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
    reasoning=False,
    input=["text"],
    cost=ModelCost(),
    context_window=200000,
    max_tokens=100,
    compat=BedrockCompat(supports_strict_mode=True),
)

# Observed from live Bedrock with 20 optional parameters across strict tools.
UNION_LIMIT_MESSAGE = (
    "The model returned the following errors: Schemas contains too many parameters with union types (20 parameters "
    "with type arrays or anyOf). This causes exponential compilation cost. Reduce the number of nullable or "
    "union-typed parameters (limit: 16 parameters with unions)."
)

_PARAMETERS = {
    "type": "object",
    "properties": {"value": {"type": "string"}, "optional": {"type": "number"}},
    "required": ["value"],
}


def tool(name: str, strict: str | None = None) -> Tool:
    return Tool(
        name=name,
        description="Look up a value",
        parameters=_PARAMETERS,
        constrained_sampling=JsonSchemaConstrainedSampling(strict=strict) if strict else None,
    )


def sdk_error(name: str, message: str) -> BedrockRuntimeServiceException:
    return BedrockRuntimeServiceException(name, message, status=400)


def success_response() -> Any:
    async def items():
        yield {"messageStart": {"role": "assistant"}}
        yield {"messageStop": {"stopReason": "end_turn"}}

    return SimpleNamespace(metadata=SimpleNamespace(http_status_code=200, request_id=None), stream=items())


@contextlib.contextmanager
def _client(respond: Callable[[int], Any]):
    inputs: list[dict[str, Any]] = []

    class _Fake:
        def __init__(self, _config):
            self.middleware_stack = SimpleNamespace(add=lambda *args, **kwargs: None)

        async def send(self, command, *, cancel=None):
            inputs.append(copy.deepcopy(command.input))
            return respond(len(inputs))

    original = bedrock.BedrockRuntimeClient
    bedrock.BedrockRuntimeClient = _Fake
    try:
        yield inputs
    finally:
        bedrock.BedrockRuntimeClient = original


def raising(error: Exception) -> Any:
    raise error


async def run(tools: list[Tool], on_payload=None, session_id: str | None = None):
    return await stream_bedrock(
        MODEL,
        normalize_context(Context(tools=tools, messages=[UserMessage(content="Hello", timestamp=1)])),
        BedrockOptions(cache_retention="none", on_payload=on_payload, session_id=session_id),
    ).result()


def strict_flags(inputs: list[dict[str, Any]]) -> list[list[bool | None]]:
    return [[tool["toolSpec"].get("strict") for tool in input["toolConfig"]["tools"]] for input in inputs]


@pytest.fixture(autouse=True)
def _cleanup_session_resources():
    yield
    cleanup_session_resources()


# Regression tests for https://github.com/earendil-works/pi/issues/11416.
@pytest.mark.tonio
async def test_retries_a_strict_tool_limit_rejection_with_preferred_tools_non_strict_and_keeps_required_tools_strict():
    payload_calls: list[Any] = []

    async def on_payload(payload, _model):
        payload_calls.append(payload)
        return {**payload, "requestMetadata": {"team": "a"}}

    def respond(attempt: int):
        return raising(sdk_error("ValidationException", UNION_LIMIT_MESSAGE)) if attempt == 1 else success_response()

    with _client(respond) as inputs:
        result = await run([tool("preferred", "prefer"), tool("required", "require"), tool("ordinary")], on_payload)

    assert result.stop_reason == "stop"
    assert len(payload_calls) == 2
    first, second = inputs
    first_spec = first["toolConfig"]["tools"][0]["toolSpec"]
    assert first_spec["strict"] is True
    assert first_spec["inputSchema"]["json"]["required"] == ["value", "optional"]
    assert first_spec["inputSchema"]["json"]["properties"]["optional"] == {
        "anyOf": [{"type": "number"}, {"type": "null"}]
    }
    assert "strict" not in second["toolConfig"]["tools"][0]["toolSpec"]
    assert second["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"] == {
        "type": "object",
        "properties": {"value": {"type": "string"}, "optional": {"type": "number"}},
        "required": ["value"],
    }
    assert second["toolConfig"]["tools"][1] == first["toolConfig"]["tools"][1]
    assert second["toolConfig"]["tools"][1]["toolSpec"]["strict"] is True
    assert second["toolConfig"]["tools"][2] == first["toolConfig"]["tools"][2]
    assert second["requestMetadata"] == {"team": "a"}
    assert result.diagnostics is not None
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.type == "bedrock_strict_tools_fallback"
    assert diagnostic.error is not None
    assert diagnostic.error.message == UNION_LIMIT_MESSAGE
    assert diagnostic.details == {"reason": "strict_tool_limits_exceeded"}


@pytest.mark.tonio
async def test_does_not_retry_without_preferred_strict_tools():
    with _client(lambda _attempt: raising(sdk_error("ValidationException", UNION_LIMIT_MESSAGE))) as inputs:
        result = await run([tool("required", "require"), tool("ordinary")])

    assert result.stop_reason == "error"
    assert result.error_message == f"Validation error: {UNION_LIMIT_MESSAGE}"
    assert len(inputs) == 1


@pytest.mark.tonio
async def test_does_not_retry_other_validation_errors():
    with _client(lambda _attempt: raising(sdk_error("ValidationException", "messages: field required"))) as inputs:
        result = await run([tool("preferred", "prefer")])

    assert result.stop_reason == "error"
    assert len(inputs) == 1


@pytest.mark.tonio
async def test_retries_only_once():
    with _client(lambda _attempt: raising(sdk_error("ValidationException", UNION_LIMIT_MESSAGE))) as inputs:
        result = await run([tool("preferred", "prefer")])

    assert result.stop_reason == "error"
    assert len(inputs) == 2


def _reject_first(attempt: int):
    return raising(sdk_error("ValidationException", UNION_LIMIT_MESSAGE)) if attempt == 1 else success_response()


@pytest.mark.tonio
async def test_does_not_remember_a_rejection_without_a_session():
    with _client(_reject_first) as inputs:
        await run([tool("preferred", "prefer")])
        await run([tool("preferred", "prefer")])

    assert strict_flags(inputs) == [[True], [None], [True]]


# Regression tests for https://github.com/earendil-works/pi/issues/11626.
@pytest.mark.tonio
async def test_sends_preferred_tools_non_strict_right_away_on_later_requests_of_a_rejected_session():
    tools = [tool("preferred", "prefer"), tool("required", "require")]
    with _client(_reject_first) as inputs:
        await run(tools, None, "rejected")
        second = await run(tools, None, "rejected")
        await run(tools, None, "other")

    assert second.stop_reason == "stop"
    assert second.diagnostics is None
    assert strict_flags(inputs) == [
        [True, True],
        [None, True],
        [None, True],
        [True, True],
    ]


@pytest.mark.tonio
async def test_tries_strict_sampling_again_after_the_sessions_resources_are_cleaned_up():
    with _client(_reject_first) as inputs:
        await run([tool("preferred", "prefer")], None, "cleaned")
        cleanup_session_resources("cleaned")
        await run([tool("preferred", "prefer")], None, "cleaned")

    assert strict_flags(inputs) == [[True], [None], [True]]
