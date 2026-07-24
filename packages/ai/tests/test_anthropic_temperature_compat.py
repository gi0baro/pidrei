"""Mirror of pi's anthropic-temperature-compat.test.ts."""

import pytest

from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import AnthropicMessagesCompat, Model, ModelCost, SimpleStreamOptions
from tests.anthropic_helpers import capture_payload


def make_custom_model(compat: AnthropicMessagesCompat | None = None) -> Model:
    return Model(
        id="vendor--claude-opus-4-7",
        name="Vendor Proxy Opus 4.7",
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
async def test_omits_temperature_for_claude_opus_4_7():
    payload = await capture_payload(
        get_builtin_model("anthropic", "claude-opus-4-7"), SimpleStreamOptions(temperature=0)
    )
    assert "temperature" not in payload


@pytest.mark.tonio
async def test_omits_temperature_for_claude_opus_4_8():
    payload = await capture_payload(
        get_builtin_model("anthropic", "claude-opus-4-8"), SimpleStreamOptions(temperature=0)
    )
    assert "temperature" not in payload


@pytest.mark.tonio
async def test_omits_default_temperature_for_claude_opus_4_7():
    payload = await capture_payload(
        get_builtin_model("anthropic", "claude-opus-4-7"), SimpleStreamOptions(temperature=1)
    )
    assert "temperature" not in payload


@pytest.mark.tonio
async def test_keeps_temperature_for_claude_opus_4_6():
    payload = await capture_payload(
        get_builtin_model("anthropic", "claude-opus-4-6"), SimpleStreamOptions(temperature=0)
    )
    assert payload["temperature"] == 0


@pytest.mark.tonio
async def test_keeps_temperature_for_claude_sonnet_4_6():
    payload = await capture_payload(
        get_builtin_model("anthropic", "claude-sonnet-4-6"), SimpleStreamOptions(temperature=0)
    )
    assert payload["temperature"] == 0


@pytest.mark.tonio
async def test_omits_temperature_for_custom_models_with_supports_temperature_disabled():
    payload = await capture_payload(
        make_custom_model(AnthropicMessagesCompat(supports_temperature=False)), SimpleStreamOptions(temperature=0)
    )
    assert "temperature" not in payload
