"""Mirror of pi's openrouter-cache-control-models.test.ts."""

import pytest

from pidrei_ai.providers.all import get_builtin_model


OPENROUTER_ANTHROPIC_LATEST_MODEL_IDS = [
    "~anthropic/claude-fable-latest",
    "~anthropic/claude-haiku-latest",
    "~anthropic/claude-opus-latest",
    "~anthropic/claude-sonnet-latest",
]


@pytest.mark.parametrize("model_id", OPENROUTER_ANTHROPIC_LATEST_MODEL_IDS)
def test_keeps_completions_cache_control_for_openrouter_anthropic_latest_aliases(model_id):
    model = get_builtin_model("openrouter", model_id)

    assert model is not None
    assert model.api == "openai-completions"
    assert model.compat.cache_control_format == "anthropic"
