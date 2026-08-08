"""Mirror of pi's model-catalog-types.test.ts.

pi asserts this at the *type* level (`expectTypeOf`): its `flattenModelCatalog`
derives literal `api`/`id`/`provider` types from the grouped JSON. Python has no
analogue for the type assertion, but the value invariant it protects — the api
group a model is nested under is the api it loads with, and id/provider match
its position — is checkable at runtime, so that is what this mirror asserts.
"""

import pytest

from pidrei_ai.providers.all import get_builtin_model, get_builtin_models


def test_derives_model_api_id_and_provider_from_grouped_model_data():
    grok_45 = get_builtin_model("xai", "grok-4.5")
    assert grok_45 is not None
    assert grok_45.api == "openai-responses"
    assert grok_45.id == "grok-4.5"
    assert grok_45.provider == "xai"

    grok_43 = get_builtin_model("xai", "grok-4.3")
    assert grok_43 is not None
    assert grok_43.api == "openai-completions"


@pytest.mark.skip(
    reason="catalog regen deferred to U11 (`make models-data`) — pi 720f0e8ee routes it at generation time; unskip after regen"
)
def test_routes_github_copilot_grok_45_through_the_responses_api():
    model = get_builtin_model("github-copilot", "grok-4.5")
    assert model is not None
    assert model.api == "openai-responses"


def test_every_catalog_model_matches_its_position():
    for provider_id in ("xai", "anthropic", "openai", "fireworks", "opencode"):
        models = get_builtin_models(provider_id)
        assert models, provider_id
        for model in models:
            assert model.provider == provider_id
            assert model.id
            assert model.api
