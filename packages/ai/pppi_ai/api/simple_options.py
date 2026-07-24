"""Port of pi's simple-options helpers (packages/ai/src/api/simple-options.ts)."""

from pppi_ai.types import Context, Model, SimpleStreamOptions, StreamOptions, ThinkingBudgets, ThinkingLevel
from pppi_ai.utils.estimate import estimate_context_tokens


CONTEXT_SAFETY_TOKENS = 4096
MIN_MAX_TOKENS = 1


def clamp_max_tokens_to_context(model: Model, context: Context, max_tokens: int) -> int:
    if model.context_window <= 0:
        return max(MIN_MAX_TOKENS, max_tokens)
    available = model.context_window - estimate_context_tokens(context).tokens - CONTEXT_SAFETY_TOKENS
    return min(max_tokens, max(MIN_MAX_TOKENS, available))


def build_base_options(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
    api_key: str | None = None,
) -> StreamOptions:
    requested_max = options.max_tokens if options is not None and options.max_tokens is not None else model.max_tokens
    return StreamOptions(
        temperature=options.temperature if options else None,
        max_tokens=clamp_max_tokens_to_context(model, context, requested_max),
        cancel=options.cancel if options else None,
        # pi: `apiKey: apiKey || options?.apiKey` — deliberately falsy `||`.
        api_key=api_key or (options.api_key if options else None),
        transport=options.transport if options else None,
        cache_retention=options.cache_retention if options else None,
        session_id=options.session_id if options else None,
        headers=options.headers if options else None,
        on_payload=options.on_payload if options else None,
        on_response=options.on_response if options else None,
        timeout_ms=options.timeout_ms if options else None,
        websocket_connect_timeout_ms=options.websocket_connect_timeout_ms if options else None,
        max_retries=options.max_retries if options else None,
        max_retry_delay_ms=options.max_retry_delay_ms if options else None,
        metadata=options.metadata if options else None,
        env=options.env if options else None,
    )


def clamp_reasoning(effort: ThinkingLevel | None) -> str | None:
    return "high" if effort in ("xhigh", "max") else effort


def adjust_max_tokens_for_thinking(
    # None means no explicit caller cap: use the model cap and fit thinking inside it.
    base_max_tokens: int | None,
    model_max_tokens: int,
    reasoning_level: ThinkingLevel,
    custom_budgets: ThinkingBudgets | None = None,
) -> tuple[int, int]:
    """Returns (max_tokens, thinking_budget)."""
    budgets = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 16384}
    if custom_budgets is not None:
        for level in budgets:
            custom = getattr(custom_budgets, level)
            if custom is not None:
                budgets[level] = custom

    min_output_tokens = 1024
    level = clamp_reasoning(reasoning_level)
    thinking_budget = budgets[level]  # type: ignore[index]
    max_tokens = (
        model_max_tokens if base_max_tokens is None else min(base_max_tokens + thinking_budget, model_max_tokens)
    )

    if max_tokens <= thinking_budget:
        thinking_budget = max(0, max_tokens - min_output_tokens)

    return max_tokens, thinking_budget
