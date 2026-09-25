"""Mirror of pi's model-catalog-types.test.ts.

pi asserts this at the *type* level (`expectTypeOf`): its `flattenModelCatalog`
derives literal `api`/`id`/`provider` types from the grouped JSON. Python has no
analogue for the type assertion, but the value invariant it protects — the api
group a model is nested under is the api it loads with, and id/provider match
its position — is checkable at runtime, so that is what this mirror asserts.
"""

from pidrei_ai.providers.all import get_builtin_model, get_builtin_models


def test_derives_model_api_id_and_provider_from_grouped_model_data():
    grok_45 = get_builtin_model("xai", "grok-4.5")
    assert grok_45 is not None
    assert grok_45.api == "openai-responses"
    assert grok_45.id == "grok-4.5"
    assert grok_45.provider == "xai"

    grok_46 = get_builtin_model("xai", "grok-4.6")
    assert grok_46 is not None
    assert grok_46.api == "openai-responses"
    assert grok_46.id == "grok-4.6"

    grok_47 = get_builtin_model("xai", "grok-4.7")
    assert grok_47 is not None
    assert grok_47.api == "openai-responses"
    assert grok_47.id == "grok-4.7"

    grok_43 = get_builtin_model("xai", "grok-4.3")
    assert grok_43 is not None
    assert grok_43.api == "openai-responses"


def test_routes_github_copilot_grok_45_through_the_responses_api():
    model = get_builtin_model("github-copilot", "grok-4.5")
    assert model is not None
    assert model.api == "openai-responses"


# Regression test for https://github.com/earendil-works/pi/issues/9209
def test_routes_all_github_copilot_gpt_models_through_the_responses_api():
    gpt_models = [model for model in get_builtin_models("github-copilot") if model.id.startswith("gpt-")]
    assert len(gpt_models) > 0
    assert all(model.api == "openai-responses" for model in gpt_models)
    astra = get_builtin_model("github-copilot", "gpt-6-astra")
    assert astra is not None
    assert astra.api == "openai-responses"
    for model_id in ("gpt-6-sol", "gpt-6-luna"):
        model = get_builtin_model("github-copilot", model_id)
        assert model is not None, model_id
        assert model.api == "openai-responses"
        assert model.context_window == 1000000
        assert model.max_tokens == 128000
        assert model.thinking_level_map is not None
        assert model.thinking_level_map["off"] == "none"
        assert model.thinking_level_map["max"] == "max"


def test_every_catalog_model_matches_its_position():
    for provider_id in ("xai", "anthropic", "openai", "fireworks", "opencode"):
        models = get_builtin_models(provider_id)
        assert models, provider_id
        for model in models:
            assert model.provider == provider_id
            assert model.id
            assert model.api
