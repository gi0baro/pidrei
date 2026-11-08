"""Google Gemini and Gemma models (pi: catalog/families/google.ts)."""

import re
from typing import Any

from catalog.reasoning import get_effort_thinking_level_map


def _is_gemma_4_model(model_id: str) -> bool:
    return re.search(r"gemma-?4", model_id.lower()) is not None


def google_thinking_levels(
    model_id: str, reasoning_options: list[dict[str, Any]] | None = None
) -> dict[str, str | None] | None:
    """Gemini effort levels from models.dev, with a Gemma 4 fallback."""
    effort_map = get_effort_thinking_level_map(reasoning_options or [])
    if effort_map:
        return effort_map
    if _is_gemma_4_model(model_id):
        return {"off": None, "minimal": "MINIMAL", "low": None, "medium": None, "high": "HIGH"}
    return None
