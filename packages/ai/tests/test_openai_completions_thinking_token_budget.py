"""Mirror of pi's openai-completions-thinking-token-budget.test.ts.

pi captures the request via a mocked OpenAI SDK; here `on_payload` captures the
same params object (and aborts the request before any transport is touched).
"""

import pytest

from pidrei_ai.api.openai_completions import stream_simple as stream_simple_completions
from pidrei_ai.types import (
    Context,
    Model,
    ModelCost,
    OpenAICompletionsCompat,
    SimpleStreamOptions,
    UserMessage,
)


def vllm_model(compat: OpenAICompletionsCompat | None = None) -> Model:
    # vLLM-served reasoning model: reasoning and the answer share max_tokens.
    return Model(
        id="zai-org/glm-5.2",
        name="GLM 5.2 (local vLLM)",
        api="openai-completions",
        provider="local-vllm",
        base_url="http://localhost:8000/v1",
        reasoning=True,
        input=["text"],
        cost=ModelCost(),
        context_window=262144,
        max_tokens=16384,
        compat=compat
        if compat is not None
        else OpenAICompletionsCompat(thinking_format="zai", supports_thinking_token_budget=True),
    )


async def capture(model: Model, **option_kwargs) -> dict:
    captured: dict = {}

    async def on_payload(payload, _model):
        captured["payload"] = payload
        raise RuntimeError("payload captured")

    await stream_simple_completions(
        model,
        Context(messages=[UserMessage(content="Hi", timestamp=1)]),
        SimpleStreamOptions(api_key="test", on_payload=on_payload, **option_kwargs),
    ).result()

    return captured["payload"]


@pytest.mark.tonio
async def test_sends_the_configured_budget_for_the_requested_level():
    params = await capture(vllm_model(), reasoning="medium", thinking_budgets={"medium": 4096})
    assert params["thinking_token_budget"] == 4096


@pytest.mark.tonio
async def test_omits_the_budget_when_neither_the_field_nor_the_alias_is_set():
    model = vllm_model(OpenAICompletionsCompat(thinking_format="zai"))
    params = await capture(model, reasoning="medium", thinking_budgets={"medium": 4096})
    assert "thinking_token_budget" not in params
    assert "thinking_budget" not in params
    assert "thinking_budget_tokens" not in params


@pytest.mark.tonio
async def test_omits_the_budget_when_thinking_is_off():
    params = await capture(vllm_model(), reasoning=None, thinking_budgets={"high": 8192})
    assert "thinking_token_budget" not in params


@pytest.mark.tonio
async def test_clamps_xhigh_and_max_to_the_high_budget():
    xhigh = await capture(vllm_model(), reasoning="xhigh", thinking_budgets={"high": 8192})
    max_params = await capture(vllm_model(), reasoning="max", thinking_budgets={"high": 8192})
    assert xhigh["thinking_token_budget"] == 8192
    assert max_params["thinking_token_budget"] == 8192


@pytest.mark.tonio
async def test_leaves_room_for_the_answer_when_the_budget_meets_the_response_ceiling():
    # Default high budget (16384) equals the model ceiling, which would leave no answer.
    params = await capture(vllm_model(), reasoning="high")
    assert params["thinking_token_budget"] == 16384 - 1024


@pytest.mark.tonio
async def test_uses_the_caller_max_tokens_as_the_ceiling_when_it_is_lower_than_the_model_cap():
    params = await capture(vllm_model(), reasoning="high", thinking_budgets={"high": 8192}, max_tokens=4096)
    assert params["thinking_token_budget"] == 4096 - 1024


@pytest.mark.tonio
@pytest.mark.parametrize("field", ["thinking_budget", "thinking_budget_tokens"])
async def test_sends_the_configured_field_when_thinking_token_budget_field_is_set(field):
    model = vllm_model(OpenAICompletionsCompat(thinking_format="qwen", thinking_token_budget_field=field))
    params = await capture(model, reasoning="medium", thinking_budgets={"medium": 4096})
    assert params[field] == 4096
    assert "thinking_token_budget" not in params


@pytest.mark.tonio
async def test_lets_thinking_token_budget_field_win_over_the_boolean_alias():
    model = vllm_model(
        OpenAICompletionsCompat(
            thinking_format="zai",
            supports_thinking_token_budget=True,
            thinking_token_budget_field="thinking_budget",
        )
    )
    params = await capture(model, reasoning="medium", thinking_budgets={"medium": 4096})
    assert params["thinking_budget"] == 4096
    assert "thinking_token_budget" not in params


def _chat_template_model() -> Model:
    return vllm_model(
        OpenAICompletionsCompat(
            thinking_format="chat-template",
            chat_template_kwargs={
                "enable_thinking": {"$var": "thinking.enabled"},
                "thinking_budget": {"$var": "thinking.budget"},
            },
        )
    )


@pytest.mark.tonio
async def test_puts_the_clamped_budget_in_chat_template_kwargs_when_var_is_thinking_budget():
    params = await capture(_chat_template_model(), reasoning="high")
    assert params["chat_template_kwargs"] == {"enable_thinking": True, "thinking_budget": 16384 - 1024}
    assert "thinking_token_budget" not in params


@pytest.mark.tonio
async def test_omits_thinking_budget_from_chat_template_kwargs_when_thinking_is_off():
    params = await capture(_chat_template_model(), reasoning=None)
    assert params["chat_template_kwargs"] == {"enable_thinking": False}
