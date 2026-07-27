"""Mirror of pi's together-models.test.ts.

pi reads the catalog through `compat.ts`'s `getModel`; pidrei's equivalent read is
`providers/all.py`'s `get_builtin_model` (compat.ts is pi's temporary
back-compat entrypoint, deleted with its ModelManager migration).
"""

import pytest

from pidrei_ai.env_api_keys import find_env_keys, get_env_api_key
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import ModelCost, OpenAICompletionsCompat


def test_registers_default_kimi_k26_via_openai_completions():
    model = get_builtin_model("together", "moonshotai/Kimi-K2.6")

    assert model is not None
    assert model.api == "openai-completions"
    assert model.provider == "together"
    assert model.base_url == "https://api.together.ai/v1"
    assert model.reasoning is True
    assert model.thinking_level_map == {"minimal": None, "low": None, "medium": None}
    assert model.input == ["text", "image"]
    assert model.context_window == 262144
    assert model.max_tokens == 131000
    assert model.cost == ModelCost(input=1.2, output=4.5, cache_read=0.2, cache_write=0)
    assert model.compat == OpenAICompletionsCompat(
        supports_store=False,
        supports_developer_role=False,
        supports_reasoning_effort=False,
        max_tokens_field="max_tokens",
        thinking_format="together",
        supports_strict_mode=False,
        supports_long_cache_retention=False,
    )


def test_models_together_reasoning_controls_from_the_together_api_surface():
    gpt_oss = get_builtin_model("together", "openai/gpt-oss-120b")
    assert gpt_oss is not None
    assert gpt_oss.thinking_level_map == {
        "off": None,
        "minimal": None,
        "low": "low",
        "medium": "medium",
        "high": "high",
        "max": None,
        "xhigh": None,
    }
    assert gpt_oss.compat.supports_reasoning_effort is True
    assert gpt_oss.compat.thinking_format == "openai"

    deepseek_v4 = get_builtin_model("together", "deepseek-ai/DeepSeek-V4-Pro")
    assert deepseek_v4 is not None
    assert deepseek_v4.thinking_level_map == {
        "minimal": None,
        "low": None,
        "medium": None,
        "high": "high",
        "xhigh": None,
    }
    assert deepseek_v4.compat.supports_reasoning_effort is True
    assert deepseek_v4.compat.thinking_format == "together"

    minimax = get_builtin_model("together", "MiniMaxAI/MiniMax-M2.7")
    assert minimax is not None
    assert minimax.thinking_level_map == {"off": None, "minimal": None, "low": None, "medium": None}
    # Reasoning-only models keep Together's implicit format (pi: undefined).
    assert minimax.compat.thinking_format is None
    assert minimax.compat.supports_reasoning_effort is False


@pytest.mark.tonio
async def test_resolves_together_api_key_from_the_environment():
    env = {"TOGETHER_API_KEY": "test-together-key"}

    assert find_env_keys("together", env) == ["TOGETHER_API_KEY"]
    assert await get_env_api_key("together", env) == "test-together-key"
