"""Port of pi's OpenRouter billed-cost helper (packages/ai/src/utils/openrouter-cost.ts).

OpenRouter adds its billing fields to streamed usage at runtime. pi types them
as `OpenRouterUsageAccounting`; here they are read off the raw usage dict:

- `cost`: amount charged to the user's OpenRouter account.
- `is_byok`: whether OpenRouter used the user's upstream provider key.
- `cost_details.upstream_inference_cost`: total inference cost billed through
  the user's upstream provider account, when available;
  `upstream_inference_prompt_cost` and `upstream_inference_completions_cost`
  are its prompt and completion parts.
"""

import math
from typing import Any

from pidrei_ai.builders import UsageBuilder
from pidrei_ai.types import Model


def _is_valid_cost(value: Any) -> bool:
    # JS `typeof value === "number"`: a bool is not a number there.
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def is_openrouter_model(model: Model) -> bool:
    return model.provider == "openrouter" or "openrouter.ai" in model.base_url


def apply_openrouter_reported_cost(usage: UsageBuilder, reported: dict[str, Any]) -> None:
    """Replace the catalog total with OpenRouter's billed cost when the response contains complete, valid accounting."""
    cost = reported.get("cost")
    if not _is_valid_cost(cost):
        return

    is_byok = reported.get("is_byok")
    if is_byok is False:
        usage.cost.total = cost
        usage.cost.total_from_provider = True
        return

    if is_byok is True:
        details = reported.get("cost_details")
        # pi's `!details`: anything but an object ends here or at the checks below.
        if not isinstance(details, dict):
            return
        # pi's `=== null`: only a present null falls back to the parts; a missing key does not.
        upstream_cost = details.get("upstream_inference_cost")
        if "upstream_inference_cost" in details and upstream_cost is None:
            prompt_cost = details.get("upstream_inference_prompt_cost")
            completions_cost = details.get("upstream_inference_completions_cost")
            if not _is_valid_cost(prompt_cost) or not _is_valid_cost(completions_cost):
                return
            upstream_cost = prompt_cost + completions_cost
        if not _is_valid_cost(upstream_cost):
            return
        usage.cost.total = cost + upstream_cost
        usage.cost.total_from_provider = True
