"""Mirror of pi's xiaomi-models.test.ts.

pi's `streamSimple` dispatches on the model's api; every Xiaomi model is
openai-completions, so the mirror calls that adapter's `stream_simple`.
"""

import pytest

from pidrei_ai.api.openai_completions import stream_simple as stream_simple_completions
from pidrei_ai.providers.all import get_builtin_models
from pidrei_ai.registry import get_supported_thinking_levels
from pidrei_ai.types import Context, SimpleStreamOptions, UserMessage
from pidrei_ai.utils.transcript import normalize_context


XIAOMI_PROVIDERS = ["xiaomi", "xiaomi-token-plan-cn", "xiaomi-token-plan-ams", "xiaomi-token-plan-sgp"]
DEPRECATED_MODEL_IDS = ["mimo-v2-flash", "mimo-v2-omni", "mimo-v2-pro"]
REPLACEMENT_MODEL_IDS = ["mimo-v2.6-flash", "mimo-v2.6-pro"]


@pytest.mark.parametrize("provider", XIAOMI_PROVIDERS)
def test_omits_deprecated_models(provider):
    model_ids = [model.id for model in get_builtin_models(provider)]

    for excluded in DEPRECATED_MODEL_IDS:
        assert excluded not in model_ids


@pytest.mark.parametrize("provider", XIAOMI_PROVIDERS)
def test_keeps_replacement_models(provider):
    model_ids = [model.id for model in get_builtin_models(provider)]

    for model_id in REPLACEMENT_MODEL_IDS:
        assert model_id in model_ids


async def capture_payload(model, reasoning: str | None) -> dict | None:
    captured: dict = {}

    async def on_payload(payload, _model):
        captured["payload"] = payload
        raise RuntimeError("payload captured")

    await stream_simple_completions(
        model,
        normalize_context(Context(messages=[UserMessage(content="test", timestamp=0)])),
        SimpleStreamOptions(api_key="test-xiaomi-key", reasoning=reasoning, on_payload=on_payload),
    ).result()
    return captured.get("payload")


# #10818: MiMo Chat Completions only supports a thinking on/off toggle.
@pytest.mark.tonio
@pytest.mark.parametrize("provider", XIAOMI_PROVIDERS)
async def test_models_reasoning_as_an_off_on_toggle(provider):
    models = [model for model in get_builtin_models(provider) if model.reasoning]
    assert len(models) > 0

    for model in models:
        assert get_supported_thinking_levels(model) == ["off", "high"]

        enabled = await capture_payload(model, "high")
        assert enabled["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in enabled

        disabled = await capture_payload(model, None)
        assert disabled["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in disabled
