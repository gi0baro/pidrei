"""Mirror of pi's qwen-token-plan-models.test.ts."""

import pytest

from pidrei_ai.providers.all import get_builtin_models


TEXT_MODELS = [
    "MiniMax-M2.5",
    "deepseek-v3.2",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "glm-5",
    "glm-5.1",
    "glm-5.2",
    "kimi-k2.5",
    "kimi-k2.6",
    "kimi-k2.7-code",
    "qwen3.6-flash",
    "qwen3.6-plus",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.8-max-preview",
]

IMAGE_MODELS = ["qwen-image-2.0", "qwen-image-2.0-pro", "wan2.7-image", "wan2.7-image-pro"]

PROVIDERS = ["qwen-token-plan", "qwen-token-plan-cn"]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_exposes_all_text_models(provider):
    model_ids = [model.id for model in get_builtin_models(provider)]

    for expected in TEXT_MODELS:
        assert expected in model_ids, f"{provider} should include {expected}"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_omits_image_models(provider):
    model_ids = [model.id for model in get_builtin_models(provider)]

    for excluded in IMAGE_MODELS:
        assert excluded not in model_ids, f"{provider} should not include {excluded}"
