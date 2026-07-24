"""Tests for simple-options (incl. the buildBaseOptions assertion deferred
from pi's context-estimate.test.ts)."""

from pppi_ai.api.simple_options import adjust_max_tokens_for_thinking, build_base_options, clamp_reasoning
from pppi_ai.types import Context, ModelCost, SimpleStreamOptions, ThinkingBudgets, UserMessage
from tests.test_estimate import create_assistant
from tests.test_registry import make_model


def estimate_model():
    return make_model(
        provider="openai",
        id="test-model",
        api="openai-responses",
        base_url="https://api.openai.com/v1",
        cost=ModelCost(),
        context_window=10_000,
        max_tokens=8_000,
    )


def test_build_base_options_clamps_max_tokens_to_context():
    # The remaining assertion from pi's context-estimate.test.ts: window 10_000
    # minus 1_005 estimated tokens minus 4_096 safety -> 4_899.
    context = Context(
        system_prompt="system",
        messages=[
            UserMessage(content="summary", timestamp=200),
            create_assistant(100, 9_500),
            UserMessage(content="x" * 4_000, timestamp=300),
        ],
    )
    assert build_base_options(estimate_model(), context).max_tokens == 4_899


def test_build_base_options_min_floor_and_model_cap():
    from dataclasses import replace

    model = estimate_model()
    empty = Context(messages=[])
    # Small window: clamped to window minus the 4096-token safety margin.
    assert build_base_options(model, empty).max_tokens == 5_904

    # Large window: capped at the model max.
    roomy = replace(model, context_window=100_000)
    assert build_base_options(roomy, empty).max_tokens == 8_000

    # Overflowing context still yields at least the minimum.
    huge = Context(messages=[UserMessage(content="x" * 100_000, timestamp=1)])
    assert build_base_options(model, huge).max_tokens == 1


def test_build_base_options_api_key_precedence_is_falsy():
    model = estimate_model()
    context = Context(messages=[])
    options = SimpleStreamOptions(api_key="from-options")
    # pi: `apiKey: apiKey || options?.apiKey` — empty string falls through.
    assert build_base_options(model, context, options, "").api_key == "from-options"
    assert build_base_options(model, context, options, "explicit").api_key == "explicit"


def test_clamp_reasoning():
    assert clamp_reasoning(None) is None
    assert clamp_reasoning("low") == "low"
    assert clamp_reasoning("xhigh") == "high"
    assert clamp_reasoning("max") == "high"


def test_adjust_max_tokens_for_thinking_defaults():
    max_tokens, budget = adjust_max_tokens_for_thinking(None, 32_000, "medium")
    assert (max_tokens, budget) == (32_000, 8_192)

    # Explicit caller cap: cap + budget, bounded by the model max.
    max_tokens, budget = adjust_max_tokens_for_thinking(4_000, 32_000, "low")
    assert (max_tokens, budget) == (6_048, 2_048)

    max_tokens, budget = adjust_max_tokens_for_thinking(30_000, 32_000, "high")
    assert (max_tokens, budget) == (32_000, 16_384)


def test_adjust_max_tokens_for_thinking_shrinks_budget_when_it_would_swallow_output():
    # maxTokens <= budget: reserve 1024 output tokens out of the window.
    max_tokens, budget = adjust_max_tokens_for_thinking(None, 8_192, "high")
    assert max_tokens == 8_192
    assert budget == 8_192 - 1_024


def test_adjust_max_tokens_for_thinking_custom_budgets_and_clamped_levels():
    budgets = ThinkingBudgets(high=4_096)
    max_tokens, budget = adjust_max_tokens_for_thinking(None, 32_000, "max", budgets)
    # xhigh/max clamp to "high" for budget lookup.
    assert (max_tokens, budget) == (32_000, 4_096)
