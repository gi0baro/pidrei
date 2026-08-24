"""Mirror of pi coding-agent src/core/defaults.ts."""

from pidrei_agent.types import ThinkingLevel


DEFAULT_THINKING_LEVEL: ThinkingLevel = "medium"
THINKING_LEVEL_OPTIONS: tuple[ThinkingLevel, ...] = (
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
