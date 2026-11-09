"""Mirror of pi's generate-models-strict.test.ts.

pi spawns the generator in an isolated package copy with mocked upstream
catalogs and asserts the strict allowlist check fails before any generated
file changes. PiDrei's generator builds the whole catalog in memory before it
writes anything (and writes through a staging directory, see
`catalog/generate.py`), so "fails before mutating" is structural; the mirror
asserts the allowlist failure itself, building in-process against the same
mocked sources (`catalog_generation_utils`).
"""

import pytest

from tests.catalog_generation_utils import generate_chat_models


def _source_model(model_id: str, *, tools: bool = True) -> dict:
    return {
        "id": model_id,
        "type": "chat",
        "name": model_id,
        "modalities": {"input": ["text"], "output": ["text"]},
        "capabilities": {"tools": {"supported": tools}, "reasoning": {"supported": False}},
        "api": {},
    }


def test_fails_when_an_individual_model_loses_tool_support():
    model_ids = [
        "deepseek-v4-flash-0731",
        "deepseek-v4-pro",
        "deepseek-v4-pro-0813",
        "glm-5.2",
        "qwen3.6-flash",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.8-flash",
        "qwen3.8-max",
        "qwen3.8-max-preview",
    ]
    source_models = {
        model_id: _source_model(model_id, tools=model_id != "deepseek-v4-flash-0731") for model_id in model_ids
    }

    with pytest.raises(RuntimeError) as excinfo:
        generate_chat_models(
            {"alibaba-token-plan": {"models": source_models}}, "qwen-token-plan-individual", strict=True
        )

    assert "qwen-token-plan-individual model IDs do not match (missing: deepseek-v4-flash-0731)" in str(excinfo.value)


def test_passes_when_the_individual_allowlist_matches():
    from catalog.providers.qwen_token_plan_individual import MODEL_IDS

    source_models = {model_id: _source_model(model_id) for model_id in [*MODEL_IDS, "qwen3.8-max-preview"]}

    models = generate_chat_models(
        {"alibaba-token-plan": {"models": source_models}}, "qwen-token-plan-individual", strict=True
    )

    assert sorted(models) == sorted(MODEL_IDS)
