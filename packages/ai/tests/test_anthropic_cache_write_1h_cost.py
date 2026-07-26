"""Mirror of pi's anthropic-cache-write-1h-cost.test.ts."""

import json

import pytest

from pidrei_ai.api.anthropic_messages import AnthropicOptions, stream as stream_anthropic
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, UserMessage
from tests.anthropic_helpers import now_ms
from tests.test_anthropic_sse_parsing import FakeClient, sse_body


def events_with_cache_creation(cache_creation: dict | None) -> list[tuple[str, str]]:
    start_usage: dict = {
        "input_tokens": 100,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 1_000_000,
    }
    if cache_creation:
        start_usage["cache_creation"] = cache_creation
    return [
        ("message_start", json.dumps({"type": "message_start", "message": {"id": "msg_test", "usage": start_usage}})),
        (
            "content_block_start",
            json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ),
        (
            "content_block_delta",
            json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}),
        ),
        ("content_block_stop", json.dumps({"type": "content_block_stop", "index": 0})),
        (
            "message_delta",
            json.dumps(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 1_000_000,
                    },
                }
            ),
        ),
        ("message_stop", json.dumps({"type": "message_stop"})),
    ]


def make_context() -> Context:
    return Context(messages=[UserMessage(content="hi", timestamp=now_ms())])


# claude-opus-4-8: input 5, cacheWrite (5m) 6.25 per Mtok. 1h write = 2x input = 10.


@pytest.mark.tonio
async def test_prices_the_1h_portion_at_2x_input_and_the_rest_at_the_5m_rate():
    model = get_builtin_model("anthropic", "claude-opus-4-8")
    body = sse_body(
        events_with_cache_creation({"ephemeral_5m_input_tokens": 600_000, "ephemeral_1h_input_tokens": 400_000})
    )
    result = await stream_anthropic(model, make_context(), AnthropicOptions(client=FakeClient(body))).result()

    assert result.usage.cache_write == 1_000_000
    assert result.usage.cache_write_1h == 400_000
    # 600k * 6.25/Mtok + 400k * 10/Mtok = 3.75 + 4.0 = 7.75
    assert result.usage.cost.cache_write == pytest.approx(7.75, abs=1e-10)


@pytest.mark.tonio
async def test_falls_back_to_the_5m_rate_when_no_breakdown_is_reported():
    model = get_builtin_model("anthropic", "claude-opus-4-8")
    body = sse_body(events_with_cache_creation(None))
    result = await stream_anthropic(model, make_context(), AnthropicOptions(client=FakeClient(body))).result()

    assert result.usage.cache_write == 1_000_000
    assert (result.usage.cache_write_1h or 0) == 0
    # 1M * 6.25/Mtok = 6.25
    assert result.usage.cost.cache_write == pytest.approx(6.25, abs=1e-10)
