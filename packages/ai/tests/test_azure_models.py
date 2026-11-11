"""Mirror of pi's azure-models.test.ts: the Azure catalog provider definition
run on a stub context."""

import copy
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from catalog.define import ProviderContext
from catalog.providers import azure


OPENAI_MODEL = {
    "id": "gpt-6-sol",
    "name": "GPT-6 Sol",
    "api": "openai-responses",
    "provider": "openai",
    "baseUrl": "https://api.openai.com/v1",
    "reasoning": True,
    "input": ["text", "image"],
    "cost": {
        "input": 2,
        "output": 10,
        "cacheRead": 0.2,
        "cacheWrite": 2.5,
        "tiers": [{"inputTokensAbove": 272000, "input": 4, "output": 15, "cacheRead": 0.4, "cacheWrite": 5}],
    },
    "contextWindow": 272000,
    "maxTokens": 128000,
}


def context() -> ProviderContext:
    return ProviderContext(
        provider_id="azure",
        sources=None,
        strict=False,
        provider_models=lambda provider_id: [copy.deepcopy(OPENAI_MODEL)] if provider_id == "openai" else [],
        models_dev_source=lambda _model: None,
        record_models_dev_source=lambda _model, _source: None,
    )


def test_keeps_openais_long_context_price_tier_on_azure_openai_models():
    model = next((candidate for candidate in azure.PROVIDER.models(context()) if candidate["id"] == "gpt-6-sol"), None)
    assert model is not None
    assert model["provider"] == "azure"
    assert model["api"] == "azure-openai-responses"
    assert model["contextWindow"] == 1050000
    assert model["cost"] == OPENAI_MODEL["cost"]
