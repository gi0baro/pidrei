"""Mirror of pi's xiaomi-models.test.ts."""

import pytest

from pidrei_ai.providers.all import get_builtin_model, get_builtin_models


API_BILLING_ONLY_MODEL_IDS = ["mimo-v2-flash", "mimo-v2-omni"]


@pytest.mark.parametrize("model_id", API_BILLING_ONLY_MODEL_IDS)
def test_keeps_api_billing_models_on_the_api_billing_provider(model_id):
    assert get_builtin_model("xiaomi", model_id) is not None


@pytest.mark.parametrize("provider", ["xiaomi-token-plan-cn", "xiaomi-token-plan-ams", "xiaomi-token-plan-sgp"])
def test_omits_api_billing_only_models_from_token_plan_providers(provider):
    model_ids = [model.id for model in get_builtin_models(provider)]

    assert model_ids, f"{provider} must have models"
    for excluded in API_BILLING_ONLY_MODEL_IDS:
        assert excluded not in model_ids
