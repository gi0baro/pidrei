"""Mirror of pi's generate-models-strict.test.ts.

pi spawns the generator in an isolated package copy with a mocked models.dev
fetch and asserts the strict allowlist check fails before any generated file
changes. pidrei's generator writes all artifacts after loading completes (and
through temp files — see generate_models.py), so "fails before mutating" is
structural; the mirror asserts the allowlist failure itself, in-process,
against the same fixture catalog.
"""

import importlib.util
from pathlib import Path

import pytest


def _load_generate_models():
    """Import the sibling generator script (not an installed module)."""
    path = Path(__file__).parents[1] / "scripts" / "generate_models.py"
    spec = importlib.util.spec_from_file_location("pidrei_ai_scripts_generate_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_models = _load_generate_models()


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
    catalog = {
        "alibaba-token-plan": {
            "models": {
                model_id: {
                    "id": model_id,
                    "name": model_id,
                    "tool_call": model_id != "deepseek-v4-flash-0731",
                }
                for model_id in model_ids
            }
        }
    }

    with pytest.raises(RuntimeError) as excinfo:
        generate_models._process_qwen_token_plan_models(catalog, lambda *_args: None)

    assert "qwen-token-plan-individual model IDs do not match (missing: deepseek-v4-flash-0731)" in str(excinfo.value)


def test_passes_when_the_individual_allowlist_matches():
    catalog = {
        "alibaba-token-plan": {
            "models": {
                model_id: {"id": model_id, "name": model_id, "tool_call": True}
                for model_id in [*generate_models.QWEN_TOKEN_PLAN_INDIVIDUAL_MODEL_IDS, "qwen3.8-max-preview"]
            }
        }
    }

    models = generate_models._process_qwen_token_plan_models(catalog, lambda *_args: None)

    individual = sorted(model["id"] for model in models if model["provider"] == "qwen-token-plan-individual")
    assert individual == sorted(generate_models.QWEN_TOKEN_PLAN_INDIVIDUAL_MODEL_IDS)
    assert not any(model["id"] == "qwen3.8-max-preview" for model in models)
