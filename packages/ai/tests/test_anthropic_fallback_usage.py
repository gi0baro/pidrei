"""Anthropic server-side refusal fallback pricing.

pi ships this pricing path without a test of its own (`4809c2ab` added it,
`ed867e90` reworked it, and only the compaction mirror moved either time).
pidrei resolves the served model's price through `_find_fallback_cost` rather
than pi's inline lookup, so that helper's branches are covered here.
"""

import json

import pytest

from pidrei_ai.api.anthropic_messages import AnthropicOptions, stream as stream_anthropic
from pidrei_ai.types import (
    AnthropicAllowedFallbackModel,
    AnthropicMessagesCompat,
    Context,
    Model,
    ModelCost,
    UserMessage,
)
from tests.anthropic_helpers import now_ms
from tests.test_anthropic_sse_parsing import FakeClient, sse_body


FALLBACK_COST = ModelCost(input=5, output=25, cache_read=0.5, cache_write=62.5)
FALLBACK_TARGET = AnthropicAllowedFallbackModel(provider="anthropic", model="claude-opus-4-8", cost=FALLBACK_COST)


def requested_model(compat: AnthropicMessagesCompat | None = None) -> Model:
    # Priced an order of magnitude above the fallback so the two rates cannot be confused.
    return Model(
        id="claude-fable-5",
        name="Claude Fable 5",
        api="anthropic-messages",
        provider="anthropic",
        base_url="http://127.0.0.1:9",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=50, output=250, cache_read=5, cache_write=625),
        context_window=200000,
        max_tokens=32000,
        compat=compat,
    )


def events_served_by(model_id: str) -> list[tuple[str, str]]:
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    return [
        (
            "message_start",
            json.dumps({"type": "message_start", "message": {"id": "msg_test", "model": model_id, "usage": usage}}),
        ),
        ("message_delta", json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": usage})),
        ("message_stop", json.dumps({"type": "message_stop"})),
    ]


async def run(model: Model, served_model_id: str) -> tuple:
    client = FakeClient(sse_body(events_served_by(served_model_id)))
    context = Context(messages=[UserMessage(content="hi", timestamp=now_ms())])
    result = await stream_anthropic(model, context, AnthropicOptions(client=client)).result()
    return result, client.requests[0]


@pytest.mark.tonio
async def test_prices_a_served_fallback_with_the_model_metadata_cost():
    model = requested_model(AnthropicMessagesCompat(allowed_fallback_models=[FALLBACK_TARGET]))

    result, _request = await run(model, "claude-opus-4-8")

    assert result.model == "claude-opus-4-8"
    assert result.usage.cost.input == pytest.approx(5.0, abs=1e-10)


@pytest.mark.tonio
async def test_prices_the_requested_model_with_its_own_cost():
    model = requested_model(AnthropicMessagesCompat(allowed_fallback_models=[FALLBACK_TARGET]))

    result, _request = await run(model, "claude-fable-5")

    assert result.usage.cost.input == pytest.approx(50.0, abs=1e-10)


@pytest.mark.tonio
async def test_ignores_a_fallback_target_served_by_another_provider():
    target = AnthropicAllowedFallbackModel(provider="bedrock", model="claude-opus-4-8", cost=FALLBACK_COST)
    model = requested_model(AnthropicMessagesCompat(allowed_fallback_models=[target]))

    result, _request = await run(model, "claude-opus-4-8")

    # No local price for the served model: usage stays on the requested model's rate.
    assert result.usage.cost.input == pytest.approx(50.0, abs=1e-10)


@pytest.mark.tonio
async def test_sends_fallback_targets_without_their_local_metadata():
    model = requested_model(AnthropicMessagesCompat(allowed_fallback_models=[FALLBACK_TARGET]))

    _result, request = await run(model, "claude-fable-5")

    assert request["fallbacks"] == [{"model": "claude-opus-4-8"}]


@pytest.mark.tonio
async def test_omits_fallbacks_without_permitted_targets():
    _result, request = await run(requested_model(), "claude-fable-5")

    assert "fallbacks" not in request
