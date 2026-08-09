"""Port of pi's context-overflow detector (packages/ai/src/utils/overflow.ts).

Detects context-window overflow from provider error messages (the pattern
table below), from silent overflow (z.ai style: success but usage.input beyond
the window), and from length-stop overflow (Xiaomi MiMo style: input truncated
to fill the window, zero output). See the pi source for the per-provider
reliability notes.
"""

import re

from pidrei_ai.types import AssistantMessage


_OVERFLOW_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"prompt is too long",  # Anthropic token overflow
        r"request_too_large",  # Anthropic request byte-size overflow (HTTP 413)
        r"input is too long for requested model",  # Amazon Bedrock
        r"exceeds the context window",  # OpenAI (Completions & Responses API)
        # OpenAI-compatible proxies (LiteLLM)
        r"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))",
        r"input token count.*exceeds the maximum",  # Google (Gemini)
        r"maximum prompt length is \d+",  # xAI (Grok)
        r"reduce the length of the messages",  # Groq
        r"maximum context length is \d+ tokens",  # OpenRouter (most backends)
        r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?",  # OpenRouter/Poolside
        r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)",  # Together AI
        r"exceeds the limit of \d+",  # GitHub Copilot
        r"exceeds the available context size",  # llama.cpp server
        r"greater than the context length",  # LM Studio
        r"context window exceeds limit",  # MiniMax
        r"exceeded model token limit",  # Kimi For Coding
        r"too large for model with \d+ maximum context length",  # Mistral
        r"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?",  # DS4 server
        r"model_context_window_exceeded",  # z.ai non-standard finish_reason surfaced as error text
        r"prompt too long; exceeded (?:max )?context length",  # Ollama explicit overflow error
        r"range of input length should be",  # DashScope / Qwen Token Plan
        r"context[_ ]length[_ ]exceeded",  # Generic fallback
        r"too many tokens",  # Generic fallback
        r"token limit exceeded",  # Generic fallback
        r"^4(?:00|13)\s*(?:status code)?\s*\(no body\)",  # Cerebras: 400/413 with no body
    )
]

# Errors matching any of these are excluded from overflow detection even when
# they also match an overflow pattern (e.g. Bedrock throttling "Too many tokens,
# please wait before trying again.").
_NON_OVERFLOW_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^(Throttling error|Service unavailable):",  # AWS Bedrock human-readable prefixes
        r"rate limit",  # Generic rate limiting
        r"too many requests",  # Generic HTTP 429 style
    )
]


def is_context_overflow(message: AssistantMessage, context_window: int | None = None) -> bool:
    """Check if an assistant message represents a context overflow error."""
    # Case 1: error message patterns
    if message.stop_reason == "error" and message.error_message:
        error_message = message.error_message
        is_non_overflow = any(pattern.search(error_message) for pattern in _NON_OVERFLOW_PATTERNS)
        if not is_non_overflow and any(pattern.search(error_message) for pattern in _OVERFLOW_PATTERNS):
            return True

    # Case 2: silent overflow (z.ai style) — successful but usage exceeds context
    if context_window and message.stop_reason == "stop":
        input_tokens = message.usage.input + message.usage.cache_read
        if input_tokens > context_window:
            return True

    # Case 3: length-stop overflow (Xiaomi MiMo style) — server truncates oversized
    # input to fill the window, leaving no room for output.
    if context_window and message.stop_reason == "length" and message.usage.output == 0:
        input_tokens = message.usage.input + message.usage.cache_read
        if input_tokens >= context_window * 0.99:
            return True

    return False


def is_recoverable_length(message: AssistantMessage, desired_max_output: int) -> bool:
    """Whether a length stop ended below the caller or model's intended output limit.

    Such responses may be caused by context pressure or provider-side truncation,
    so callers can make one bounded compact-and-retry attempt. `desired_max_output`
    must be the original limit before any context-based clamping.
    """
    return message.stop_reason == "length" and desired_max_output > 0 and message.usage.output < desired_max_output


def get_overflow_patterns() -> list[re.Pattern[str]]:
    """Get the overflow patterns for testing purposes."""
    return list(_OVERFLOW_PATTERNS)
