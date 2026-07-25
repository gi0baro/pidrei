"""CLI argument helpers (from pi coding-agent src/cli/args.ts).

Only the model-resolver-facing pieces live here for now; the full arg parser
lands with the modes slice.
"""

THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")


def is_valid_thinking_level(value: str) -> bool:
    return value in THINKING_LEVELS
