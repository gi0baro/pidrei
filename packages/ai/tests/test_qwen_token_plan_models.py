"""Mirror of pi's qwen-token-plan-models.test.ts.

pi asserts payloads via streamSimple + a mocked OpenAI SDK; here the payload
comes straight from `build_params`, which is the object pi's mock captures.

U11 note (pi 2f7f75a20): after the `make models-data` regen, every
`qwen3.8-max-preview` reference below must become `qwen3.8-max` (the generator
now excludes the retired preview id), and the skipped omits-retired-preview
test starts running against the new catalog.
"""

import pytest

from pidrei_ai.api.openai_completions import OpenAICompletionsOptions, build_params
from pidrei_ai.providers.all import get_builtin_model, get_builtin_models
from pidrei_ai.types import Context, UserMessage


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

INDIVIDUAL_TEXT_MODELS = [
    "deepseek-v4-flash-0731",
    "deepseek-v4-pro",
    "glm-5.2",
    "qwen3.6-flash",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.8-max",
]

IMAGE_MODELS = ["qwen-image-2.0", "qwen-image-2.0-pro", "wan2.7-image", "wan2.7-image-pro"]

PROVIDERS = ["qwen-token-plan", "qwen-token-plan-cn"]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_exposes_all_text_models(provider):
    model_ids = [model.id for model in get_builtin_models(provider)]

    for expected in TEXT_MODELS:
        assert expected in model_ids, f"{provider} should include {expected}"


@pytest.mark.skip(
    reason="catalog regen deferred to U11 (`make models-data`) — pi 2f7f75a20 excludes the retired id at generation time; unskip after regen"
)
@pytest.mark.parametrize("provider", [*PROVIDERS, "qwen-token-plan-individual"])
def test_omits_retired_qwen38_max_preview(provider):
    model_ids = [model.id for model in get_builtin_models(provider)]
    assert "qwen3.8-max-preview" not in model_ids


@pytest.mark.skip(
    reason="catalog regen deferred to U11 (`make models-data`) — pi c03d78bdc generates the Individual catalog; unskip after regen and extend the provider-parametrized cases with qwen-token-plan-individual per that commit"
)
def test_exposes_exactly_the_documented_individual_text_models():
    model_ids = sorted(model.id for model in get_builtin_models("qwen-token-plan-individual"))

    assert model_ids == sorted(INDIVIDUAL_TEXT_MODELS)


def test_reuses_the_international_token_plan_environment_variable():
    from pidrei_ai.env_api_keys import find_env_keys

    assert find_env_keys("qwen-token-plan-individual", {"QWEN_TOKEN_PLAN_API_KEY": "test"}) == [
        "QWEN_TOKEN_PLAN_API_KEY"
    ]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_omits_image_models(provider):
    model_ids = [model.id for model in get_builtin_models(provider)]

    for excluded in IMAGE_MODELS:
        assert excluded not in model_ids, f"{provider} should not include {excluded}"


THINKING_MODELS = [
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

REASONING_EFFORT_MODELS = ["deepseek-v4-flash", "deepseek-v4-pro", "glm-5", "glm-5.1", "glm-5.2"]


def _context() -> Context:
    return Context(messages=[UserMessage(content="Hi", timestamp=1)])


def _payload(model, reasoning_effort: str) -> dict:
    return build_params(model, _context(), OpenAICompletionsOptions(api_key="test", reasoning_effort=reasoning_effort))


# docs: https://modelstudio.console.alibabacloud.com/ap-southeast-1?tab=api&commonbuy=1#/api/?type=model&url=3016807
@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("model_id", THINKING_MODELS)
def test_sends_qwen_thinking_fields(provider, model_id):
    model = get_builtin_model(provider, model_id)
    assert model is not None, f"Missing model: {provider}/{model_id}"

    payload = _payload(model, "high")

    assert payload["enable_thinking"] is True
    assert "thinking" not in payload


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("model_id", REASONING_EFFORT_MODELS)
def test_exposes_qwen_reasoning_effort_levels(provider, model_id):
    model = get_builtin_model(provider, model_id)
    assert model is not None, f"Missing model: {provider}/{model_id}"

    assert model.thinking_level_map == {
        "minimal": None,
        "low": None,
        "medium": None,
        "high": "high",
        "xhigh": None,
        "max": "max",
    }


@pytest.mark.parametrize("provider", PROVIDERS)
def test_exposes_qwen38_reasoning_effort_levels(provider):
    model = get_builtin_model(provider, "qwen3.8-max-preview")
    assert model is not None, f"Missing model: {provider}/qwen3.8-max-preview"

    assert model.thinking_level_map == {
        "minimal": None,
        "low": "low",
        "medium": "medium",
        "high": None,
        "xhigh": "xhigh",
        "max": None,
    }


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("model_id", REASONING_EFFORT_MODELS)
def test_sends_qwen_reasoning_effort(provider, model_id):
    model = get_builtin_model(provider, model_id)
    assert model is not None, f"Missing model: {provider}/{model_id}"

    payload = _payload(model, "high")

    assert payload["reasoning_effort"] == "high"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_sends_qwen38_max_reasoning_effort(provider):
    model = get_builtin_model(provider, "qwen3.8-max-preview")
    assert model is not None, f"Missing model: {provider}/qwen3.8-max-preview"

    payload = _payload(model, "xhigh")

    assert payload["enable_thinking"] is True
    assert payload["reasoning_effort"] == "xhigh"
    assert "thinking" not in payload
