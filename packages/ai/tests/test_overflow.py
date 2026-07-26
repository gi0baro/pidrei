"""Mirror of pi's overflow.test.ts."""

import time

from pidrei_ai.types import AssistantMessage, Usage
from pidrei_ai.utils.overflow import is_context_overflow


def create_error_message(error_message: str) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api="openai-completions",
        provider="ollama",
        model="qwen3.5:35b",
        usage=Usage(),
        stop_reason="error",
        error_message=error_message,
        timestamp=int(time.time() * 1000),
    )


def create_length_stop_message(input_tokens: int, cache_read: int, output: int) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api="openai-completions",
        provider="xiaomi",
        model="mimo-v2.5-pro",
        usage=Usage(
            input=input_tokens,
            output=output,
            cache_read=cache_read,
            total_tokens=input_tokens + cache_read + output,
        ),
        stop_reason="length",
        timestamp=int(time.time() * 1000),
    )


def test_detects_explicit_ollama_prompt_too_long_errors():
    message = create_error_message("400 `prompt too long; exceeded max context length by 100918 tokens`")
    assert is_context_overflow(message, 32768) is True


def test_detects_together_ai_context_length_errors():
    message = create_error_message(
        "400 The input (516368 tokens) is longer than the model's context length (262144 tokens)."
    )
    assert is_context_overflow(message, 262144) is True


def test_detects_litellm_wrapped_openai_maximum_context_length_errors():
    message = create_error_message(
        "Error: 503 litellm.ServiceUnavailableError: litellm.MidStreamFallbackError: "
        "litellm.APIConnectionError: APIConnectionError: OpenAIException - Requested token "
        "count exceeds the model's maximum context length of 131072 tokens."
    )
    assert is_context_overflow(message, 131072) is True


def test_detects_openai_compatible_parenthesized_maximum_context_length_errors():
    message = create_error_message("Error: 400 Input length (265330) exceeds model's maximum context length (262144).")
    assert is_context_overflow(message, 262144) is True


def test_detects_openrouter_poolside_maximum_allowed_input_length_errors():
    message = create_error_message(
        "Provider returned error: Input length 131393 exceeds the maximum allowed input length of 131040 tokens."
    )
    assert is_context_overflow(message, 131072) is True


def test_detects_ds4_configured_context_size_errors():
    message = create_error_message("400 Prompt has 256468 tokens, but the configured context size is 256000 tokens")
    assert is_context_overflow(message, 256000) is True

    comma_message = create_error_message(
        "Prompt has 5,958,968 tokens, but the configured context size is 256,000 tokens"
    )
    assert is_context_overflow(comma_message, 256000) is True


def test_does_not_treat_generic_non_overflow_ollama_errors_as_overflow():
    message = create_error_message("500 `model runner crashed unexpectedly`")
    assert is_context_overflow(message, 32768) is False


def test_does_not_treat_bedrock_throttling_too_many_tokens_as_overflow():
    # Bedrock returns this for HTTP 429 rate limiting, NOT context overflow.
    message = create_error_message("Throttling error: Too many tokens, please wait before trying again.")
    assert is_context_overflow(message, 200000) is False


def test_does_not_treat_bedrock_service_unavailable_as_overflow():
    message = create_error_message("Service unavailable: The service is temporarily unavailable.")
    assert is_context_overflow(message, 200000) is False


def test_does_not_treat_generic_rate_limit_errors_as_overflow():
    message = create_error_message("Rate limit exceeded, please retry after 30 seconds.")
    assert is_context_overflow(message, 200000) is False


def test_does_not_treat_http_429_style_errors_as_overflow():
    message = create_error_message("Too many requests. Please slow down.")
    assert is_context_overflow(message, 200000) is False


def test_detects_xiaomi_style_overflow_length_stop_with_zero_output_and_filled_context():
    message = create_length_stop_message(58, 1048512, 0)
    assert is_context_overflow(message, 1048576) is True


def test_does_not_treat_normal_length_stops_with_output_as_overflow():
    message = create_length_stop_message(1000, 0, 4096)
    assert is_context_overflow(message, 200000) is False


def test_does_not_treat_length_stops_far_below_context_as_overflow():
    message = create_length_stop_message(100, 0, 0)
    assert is_context_overflow(message, 200000) is False
