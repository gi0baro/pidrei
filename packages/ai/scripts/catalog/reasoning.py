"""models.dev effort values to thinking levels (pi: catalog/reasoning.ts)."""

from typing import Any


THINKING_LEVELS = ("minimal", "low", "medium", "high", "xhigh", "max")


def get_effort_thinking_level_map(options: list[dict[str, Any]]) -> dict[str, str | None] | None:
    """Converts models.dev verified effort values into PiDrei's selectable thinking levels.
    Values without a PiDrei equivalent (`default` and JSON `null`) are intentionally
    omitted."""
    effort_values = [
        value for option in options if option.get("type") == "effort" for value in option.get("values") or []
    ]
    if not effort_values:
        return None

    supported = set(effort_values)
    if not any(level in supported for level in THINKING_LEVELS) and "none" not in supported:
        return None

    mapping: dict[str, str | None] = {"off": "none" if "none" in supported else None}
    for level in THINKING_LEVELS:
        mapping[level] = level if level in supported else None
    return mapping
