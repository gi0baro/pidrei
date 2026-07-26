"""Mirror of pi coding-agent src/core/usage-totals.ts.

Session entries are plain camelCase dicts (see session_manager.py); message
values inside entries are pidrei_ai/pidrei_agent message dataclasses.
"""

from dataclasses import dataclass
from typing import Any

from pidrei_ai.types import Usage


@dataclass(slots=True)
class UsageTotals:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost: float = 0.0


def create_usage_totals() -> UsageTotals:
    return UsageTotals()


def add_usage_to_totals(totals: UsageTotals, usage: Usage) -> None:
    totals.input += usage.input
    totals.output += usage.output
    totals.cache_read += usage.cache_read
    totals.cache_write += usage.cache_write
    totals.cost += usage.cost.total


@dataclass(slots=True)
class UsageCostBreakdownEntry:
    key: str
    cost: float
    tokens: int


def get_usage_cost_breakdown(entries: list[dict[str, Any]]) -> list[UsageCostBreakdownEntry]:
    """Group attributable assistant usage by model and all other usage into a separate bucket."""
    totals_by_key: dict[str, UsageTotals] = {}

    for entry in entries:
        key: str | None = None
        usage: Usage | None = None
        message = entry.get("message")
        role = getattr(message, "role", None)
        if entry.get("type") == "message" and role == "assistant":
            response_model = message.response_model if message.response_model is not None else message.model
            key = f"{message.provider}/{response_model}"
            usage = message.usage
        elif entry.get("type") == "message" and role == "toolResult" and getattr(message, "usage", None):
            key = "Tools/summaries"
            usage = message.usage
        elif entry.get("type") in ("branch_summary", "compaction") and entry.get("usage"):
            key = "Tools/summaries"
            usage = entry["usage"]
        if not key or not usage:
            continue

        totals = totals_by_key.get(key)
        if totals is None:
            totals = create_usage_totals()
            totals_by_key[key] = totals
        add_usage_to_totals(totals, usage)

    breakdown = [
        UsageCostBreakdownEntry(
            key=key,
            cost=totals.cost,
            tokens=totals.input + totals.output + totals.cache_read + totals.cache_write,
        )
        for key, totals in totals_by_key.items()
    ]
    breakdown = [entry for entry in breakdown if entry.cost > 0 or entry.tokens > 0]
    breakdown.sort(key=lambda entry: entry.cost, reverse=True)
    return breakdown
