"""Mirror of pi's baseten-models.test.ts.

pi resolves Baseten models from the generated catalog; the wire-behavior cases
here build the models the generator emits (`_process_baseten_models` in
scripts/generate_models.py) by hand so they assert the payload, not the catalog.
"""

import pytest

from pidrei_ai.api.openai_completions import stream_simple as stream_simple_completions
from pidrei_ai.env_api_keys import find_env_keys, get_env_api_key
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.registry import get_supported_thinking_levels
from pidrei_ai.types import (
    Context,
    Model,
    ModelCost,
    OpenAICompletionsCompat,
    SimpleStreamOptions,
    UserMessage,
)


GLM52_THINKING_LEVEL_MAP = {
    "off": "none",
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "xhigh": None,
    "max": "max",
}

TOGGLE_THINKING_LEVEL_MAP = {
    "off": "off",
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "xhigh": None,
    "max": None,
}


def baseten_model(model_id: str, *, supports_reasoning_effort: bool, thinking_level_map: dict) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="openai-completions",
        provider="baseten",
        base_url="https://inference.baseten.co/v1",
        reasoning=True,
        thinking_level_map=dict(thinking_level_map),
        input=["text"],
        cost=ModelCost(),
        context_window=1048576,
        max_tokens=262144,
        compat=OpenAICompletionsCompat(
            supports_store=False,
            supports_developer_role=False,
            supports_reasoning_effort=supports_reasoning_effort,
            supports_usage_in_streaming=True,
            max_tokens_field="max_tokens",
            supports_strict_mode=True,
            supports_long_cache_retention=False,
            thinking_format="baseten",
            chat_template_args={"enable_thinking": {"$var": "thinking.enabled"}},
        ),
    )


async def capture_payload(model: Model, reasoning: str | None = None) -> dict:
    captured: dict = {}

    async def on_payload(payload, _model):
        captured["payload"] = payload
        raise RuntimeError("payload captured")

    result = await stream_simple_completions(
        model,
        Context(messages=[UserMessage(content="test", timestamp=0)]),
        SimpleStreamOptions(api_key="test-baseten-key", reasoning=reasoning, on_payload=on_payload),
    ).result()
    assert result.stop_reason == "error"
    return captured["payload"]


def test_keeps_both_glm_52_endpoints_text_only():
    assert get_builtin_model("baseten", "zai-org/GLM-5.2").input == ["text"]
    assert get_builtin_model("baseten", "zai-org/GLM-5.2-Fast").input == ["text"]


@pytest.mark.tonio
async def test_models_kimi_k26_reasoning_as_an_explicit_off_on_toggle():
    model = baseten_model(
        "moonshotai/Kimi-K2.6",
        supports_reasoning_effort=False,
        thinking_level_map=TOGGLE_THINKING_LEVEL_MAP,
    )

    assert get_supported_thinking_levels(model) == ["off", "high"]

    payload = await capture_payload(model, reasoning="high")
    assert payload["chat_template_args"] == {"enable_thinking": True}
    assert payload.get("reasoning_effort") is None


@pytest.mark.tonio
async def test_sends_baseten_chat_template_args_with_reasoning_effort():
    model = baseten_model(
        "zai-org/GLM-5.2",
        supports_reasoning_effort=True,
        thinking_level_map=GLM52_THINKING_LEVEL_MAP,
    )

    payload = await capture_payload(model, reasoning="high")

    assert payload["chat_template_args"] == {"enable_thinking": True}
    assert payload["reasoning_effort"] == "high"


@pytest.mark.tonio
async def test_disables_baseten_opt_in_reasoning_when_thinking_is_off():
    model = baseten_model(
        "zai-org/GLM-5.2",
        supports_reasoning_effort=True,
        thinking_level_map=GLM52_THINKING_LEVEL_MAP,
    )

    payload = await capture_payload(model)

    assert payload["chat_template_args"] == {"enable_thinking": False}
    assert payload["reasoning_effort"] == "none"


@pytest.mark.tonio
async def test_resolves_baseten_api_key_from_the_environment():
    env = {"BASETEN_API_KEY": "test-baseten-key"}

    assert find_env_keys("baseten", env) == ["BASETEN_API_KEY"]
    assert await get_env_api_key("baseten", env) == "test-baseten-key"
