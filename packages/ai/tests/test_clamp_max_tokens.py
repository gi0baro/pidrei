"""Mirror of pi's clamp-max-tokens.test.ts.

pi copies `openai/gpt-4o-mini` from the catalog and overrides the window and
output limit; only those two fields matter, so the model is built directly.
"""

from pidrei_ai.api.simple_options import clamp_max_tokens_to_context
from pidrei_ai.types import AssistantMessage, Context, Model, TextContent, Usage, UserMessage
from pidrei_ai.utils.transcript import normalize_context
from tests.test_registry import make_model


WINDOW = 1_048_576


def model_with(max_tokens: int, context_window: int = WINDOW) -> Model:
    return make_model(
        provider="openai",
        id="gpt-4o-mini",
        api="openai-completions",
        context_window=context_window,
        max_tokens=max_tokens,
    )


# The last assistant usage anchors the estimate, so the estimated input is exactly `input_tokens`.
def context_with_input(input_tokens: int):
    assistant = AssistantMessage(
        content=[TextContent(text="ok")],
        api="openai-completions",
        provider="openai",
        model="gpt-4o-mini",
        usage=Usage(input=input_tokens, total_tokens=input_tokens),
        stop_reason="stop",
        timestamp=2,
    )
    return normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1), assistant]))


def test_leaves_15_percent_of_the_input_unused_so_an_openrouter_overcount_still_fits():
    # The failing request from #10497: OpenRouter counted 229,747 input tokens where pi estimated 220,815.
    max_tokens = clamp_max_tokens_to_context(model_with(943_718), context_with_input(220_815), 943_718)
    assert max_tokens == WINDOW - 220_815 - 33_123
    assert 229_747 + max_tokens <= WINDOW


def test_leaves_at_least_4096_tokens_unused_for_small_inputs():
    assert clamp_max_tokens_to_context(model_with(WINDOW), context_with_input(10_000), WINDOW) == WINDOW - 10_000 - 4096


def test_keeps_the_model_output_limit_when_it_already_fits():
    assert clamp_max_tokens_to_context(model_with(64_000), context_with_input(100_000), 64_000) == 64_000


def test_sends_up_to_32768_tokens_when_the_proportional_margin_leaves_no_room():
    assert clamp_max_tokens_to_context(model_with(943_718), context_with_input(912_000), 943_718) == 32_768


def test_uses_the_real_room_minus_4096_when_less_than_32768_tokens_fit():
    assert (
        clamp_max_tokens_to_context(model_with(943_718), context_with_input(1_030_000), 943_718)
        == WINDOW - 1_030_000 - 4096
    )


def test_keeps_only_the_fixed_4096_margin_for_small_windows():
    assert (
        clamp_max_tokens_to_context(model_with(32_768, 32_768), context_with_input(20_000), 32_768)
        == 32_768 - 20_000 - 4096
    )


def test_sends_1_token_when_the_input_already_exceeds_the_window():
    assert clamp_max_tokens_to_context(model_with(943_718), context_with_input(1_050_000), 943_718) == 1
