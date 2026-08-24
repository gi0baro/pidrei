"""Anthropic server-side refusal fallback pricing.

pi's `4809c2ab` ships this pricing path without a test of its own (only the
compaction mirror moved). pidrei resolves the served model's price through
`_find_fallback_cost` rather than pi's inline lookup, so the two branches that
helper adds are covered here.
"""

import json

import pytest

from pidrei_ai.api.anthropic_messages import AnthropicOptions, stream as stream_anthropic
from pidrei_ai.types import (
    AnthropicMessagesCompat,
    AnthropicRefusalFallbackTarget,
    Context,
    Model,
    ModelCost,
    UserMessage,
)
from tests.anthropic_helpers import now_ms
from tests.test_anthropic_sse_parsing import FakeClient, sse_body


FALLBACK_COST = ModelCost(input=5, output=25, cache_read=0.5, cache_write=6.25)


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
        cost=ModelCost(input=50, output=250, cache_read=5, cache_write=62.5),
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


async def run(model: Model, served_model_id: str, **option_kwargs):
    client = FakeClient(sse_body(events_served_by(served_model_id)))
    context = Context(messages=[UserMessage(content="hi", timestamp=now_ms())])
    return await stream_anthropic(model, context, AnthropicOptions(client=client, **option_kwargs)).result()


@pytest.mark.tonio
async def test_prices_a_served_fallback_with_the_request_fallback_cost():
    result = await run(
        requested_model(),
        "claude-opus-4-8",
        refusal_fallbacks=[AnthropicRefusalFallbackTarget(model="claude-opus-4-8", cost=FALLBACK_COST)],
    )

    assert result.model == "claude-opus-4-8"
    assert result.usage.cost.input == pytest.approx(5.0, abs=1e-10)


@pytest.mark.tonio
async def test_prices_a_served_fallback_with_the_model_metadata_cost():
    model = requested_model(
        AnthropicMessagesCompat(
            allowed_fallback_models=[AnthropicRefusalFallbackTarget(model="claude-opus-4-8", cost=FALLBACK_COST)]
        )
    )
    result = await run(model, "claude-opus-4-8", refusal_fallbacks="default")

    assert result.usage.cost.input == pytest.approx(5.0, abs=1e-10)


@pytest.mark.tonio
async def test_prices_the_requested_model_with_its_own_cost():
    result = await run(
        requested_model(),
        "claude-fable-5",
        refusal_fallbacks=[AnthropicRefusalFallbackTarget(model="claude-opus-4-8", cost=FALLBACK_COST)],
    )

    assert result.usage.cost.input == pytest.approx(50.0, abs=1e-10)


@pytest.mark.tonio
async def test_sends_fallback_targets_without_their_local_cost():
    client = FakeClient(sse_body(events_served_by("claude-fable-5")))
    context = Context(messages=[UserMessage(content="hi", timestamp=now_ms())])
    await stream_anthropic(
        requested_model(),
        context,
        AnthropicOptions(
            client=client,
            refusal_fallbacks=[AnthropicRefusalFallbackTarget(model="claude-opus-4-8", cost=FALLBACK_COST)],
        ),
    ).result()

    assert client.requests[0]["fallbacks"] == [{"model": "claude-opus-4-8"}]
