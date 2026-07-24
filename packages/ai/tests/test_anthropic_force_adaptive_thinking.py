"""Mirror of pi's anthropic-force-adaptive-thinking.test.ts.

The Kimi Coding cases join when that provider's catalog lands (PLAN.md).
"""

from dataclasses import replace

import pytest

from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import AnthropicMessagesCompat, Model, ModelCost, SimpleStreamOptions
from tests.anthropic_helpers import capture_payload


def make_custom_model(compat: AnthropicMessagesCompat | None = None) -> Model:
    return Model(
        # Id intentionally does not match any built-in adaptive substring
        # (mirrors corporate proxy schemes like `anthropic--claude-opus-latest`).
        id="vendor--claude-opus-latest",
        name="Vendor Proxy Opus Latest",
        api="anthropic-messages",
        provider="vendor-proxy",
        base_url="http://127.0.0.1:9",
        reasoning=True,
        input=["text"],
        cost=ModelCost(),
        context_window=200000,
        max_tokens=32000,
        compat=compat,
    )


@pytest.mark.tonio
async def test_sends_legacy_thinking_payload_for_custom_model_ids_by_default():
    payload = await capture_payload(make_custom_model(), SimpleStreamOptions(reasoning="medium"))

    assert payload["thinking"]["type"] == "enabled"
    assert "output_config" not in payload


@pytest.mark.tonio
async def test_sends_adaptive_thinking_payload_when_compat_force_adaptive_thinking_is_true():
    payload = await capture_payload(
        make_custom_model(AnthropicMessagesCompat(force_adaptive_thinking=True)),
        SimpleStreamOptions(reasoning="medium"),
    )

    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "medium"}


@pytest.mark.tonio
async def test_uses_adaptive_thinking_with_native_xhigh_effort_for_claude_fable_5():
    payload = await capture_payload(
        get_builtin_model("anthropic", "claude-fable-5"), SimpleStreamOptions(reasoning="xhigh")
    )

    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "xhigh"}


@pytest.mark.tonio
async def test_allows_builtin_adaptive_models_to_opt_out_with_compat_false():
    model = replace(
        get_builtin_model("anthropic", "claude-opus-4-8"),
        compat=AnthropicMessagesCompat(force_adaptive_thinking=False),
    )
    payload = await capture_payload(model, SimpleStreamOptions(reasoning="medium"))

    assert payload["thinking"]["type"] == "enabled"
    assert "output_config" not in payload


@pytest.mark.tonio
async def test_preserves_thinking_disabled_when_reasoning_is_off_regardless_of_override():
    payload = await capture_payload(make_custom_model(AnthropicMessagesCompat(force_adaptive_thinking=True)))

    assert payload["thinking"] == {"type": "disabled"}
    assert "output_config" not in payload
