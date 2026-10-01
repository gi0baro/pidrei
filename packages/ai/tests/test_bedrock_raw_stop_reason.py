"""Mirror of pi's bedrock-raw-stop-reason.test.ts.

pi replaces `@aws-sdk/client-bedrock-runtime` with `vi.mock`; here the stub
replaces `api/bedrock_runtime.BedrockRuntimeClient` by name, as in the other
bedrock mirrors.
"""

import contextlib
from types import SimpleNamespace

import pytest

from pidrei_ai.api import bedrock_converse_stream as bedrock
from pidrei_ai.api.bedrock_converse_stream import BedrockOptions, stream as stream_bedrock
from pidrei_ai.api.bedrock_runtime import BedrockRuntimeServiceException
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, TextContent, UserMessage


MODEL = get_builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8")
CONTEXT = Context(messages=[UserMessage(content="hello", timestamp=1)])


@contextlib.contextmanager
def _streaming(stop_reason: str = "end_turn", stream_events: list | None = None):
    class _Fake:
        def __init__(self, _config):
            self.middleware_stack = SimpleNamespace(add=lambda *args, **kwargs: None)

        async def send(self, _command, *, cancel=None):
            async def items():
                if stream_events is not None:
                    for item in stream_events:
                        yield item
                    return
                yield {"messageStart": {"role": "assistant"}}
                yield {"messageStop": {"stopReason": stop_reason}}
                yield {
                    "metadata": {
                        "usage": {"inputTokens": 1, "outputTokens": 0, "totalTokens": 1},
                    }
                }

            return SimpleNamespace(
                metadata=SimpleNamespace(http_status_code=None, request_id=None),
                stream=items(),
            )

    original = bedrock.BedrockRuntimeClient
    bedrock.BedrockRuntimeClient = _Fake
    try:
        yield
    finally:
        bedrock.BedrockRuntimeClient = original


async def _result(stop_reason: str):
    with _streaming(stop_reason):
        return await stream_bedrock(MODEL, CONTEXT, BedrockOptions(cache_retention="none")).result()


@pytest.mark.tonio
async def test_forwards_sdk_stream_items_in_order_before_normalizing_them():
    stream_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hello"}}},
        {"messageStop": {"stopReason": "end_turn", "additionalModelResponseFields": {"source": "test"}}},
        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}}},
    ]
    received: list = []
    event_models: list = []

    async def on_provider_stream_event(item, event_model) -> None:
        received.append(item)
        event_models.append(event_model)

    with _streaming(stream_events=stream_events):
        result = await stream_bedrock(
            MODEL,
            CONTEXT,
            BedrockOptions(cache_retention="none", on_provider_stream_event=on_provider_stream_event),
        ).result()

    assert received == stream_events
    for index, item in enumerate(received):
        assert item is stream_events[index]
    assert event_models == [MODEL, MODEL, MODEL, MODEL]
    assert result.stop_reason == "stop"
    assert result.content == [TextContent(text="hello")]


@pytest.mark.tonio
async def test_forwards_sdk_error_items_before_reporting_them():
    exception = BedrockRuntimeServiceException("InternalServerException", "bedrock stream failed")
    stream_events = [{"messageStart": {"role": "assistant"}}, {"internalServerException": exception}]
    received: list = []

    async def on_provider_stream_event(item, _model) -> None:
        received.append(item)

    with _streaming(stream_events=stream_events):
        result = await stream_bedrock(
            MODEL,
            CONTEXT,
            BedrockOptions(cache_retention="none", on_provider_stream_event=on_provider_stream_event),
        ).result()

    assert received == stream_events
    assert result.stop_reason == "error"
    # pi's mock throws a plain Error, so its message comes through bare; the
    # runtime's modelled exception gets the adapter's code prefix.
    assert result.error_message == "Internal server error: bedrock stream failed"


@pytest.mark.tonio
async def test_preserves_raw_bedrock_stop_reasons_for_successful_stops():
    message = await _result("end_turn")

    assert message.stop_reason == "stop"
    assert message.raw_stop_reason == "end_turn"
    assert message.error_message is None


@pytest.mark.tonio
async def test_preserves_raw_bedrock_stop_reasons_for_provider_error_stops():
    message = await _result("guardrail_intervened")

    assert message.stop_reason == "error"
    assert message.raw_stop_reason == "guardrail_intervened"
    assert message.error_message == "Provider stopped with: guardrail_intervened"
