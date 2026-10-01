"""Usage arithmetic shared across packages.

pi keeps `combineUsage` in coding-agent's `usage-totals.ts`. pidrei's agent
loop sums a tool's nested usage onto its result message (pi's session does it
at `message_start`), so the helper lives here, below both packages;
`pidrei.core.usage_totals` re-exports it at pi's location.
"""

from pidrei_ai.types import Usage, UsageCost


def combine_usage(first: Usage, second: Usage) -> Usage:
    """Sum of two usages, keeping the optional token splits when either side reports them."""
    return Usage(
        input=first.input + second.input,
        output=first.output + second.output,
        cache_read=first.cache_read + second.cache_read,
        cache_write=first.cache_write + second.cache_write,
        cache_write_1h=(
            (first.cache_write_1h or 0) + (second.cache_write_1h or 0)
            if first.cache_write_1h is not None or second.cache_write_1h is not None
            else None
        ),
        reasoning=(
            (first.reasoning or 0) + (second.reasoning or 0)
            if first.reasoning is not None or second.reasoning is not None
            else None
        ),
        total_tokens=first.total_tokens + second.total_tokens,
        cost=UsageCost(
            input=first.cost.input + second.cost.input,
            output=first.cost.output + second.cost.output,
            cache_read=first.cost.cache_read + second.cost.cache_read,
            cache_write=first.cost.cache_write + second.cost.cache_write,
            total=first.cost.total + second.cost.total,
        ),
    )
