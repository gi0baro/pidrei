"""Mirror of pi's xiaomi-models.test.ts."""

import pytest

from pidrei_ai.providers.all import get_builtin_models


XIAOMI_PROVIDERS = ["xiaomi", "xiaomi-token-plan-cn", "xiaomi-token-plan-ams", "xiaomi-token-plan-sgp"]
DEPRECATED_MODEL_IDS = ["mimo-v2-flash", "mimo-v2-omni", "mimo-v2-pro"]
REPLACEMENT_MODEL_IDS = ["mimo-v2.5", "mimo-v2.5-pro"]


@pytest.mark.skip(reason="catalog regen pending — unskip after `make models-data` (PORT_0.84.3 U10)")
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
