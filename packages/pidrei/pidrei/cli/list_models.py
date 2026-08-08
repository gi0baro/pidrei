"""Mirror of pi coding-agent src/cli/list-models.ts.

List available models with optional fuzzy search.
"""

import sys

from ..core.auth_guidance import format_no_models_available_message
from ..utils.colors import yellow
from ..utils.fuzzy import fuzzy_filter


def format_token_count(count: int) -> str:
    """Format a number as human-readable (e.g., 200000 -> "200K", 1000000 -> "1M")."""
    if count >= 1_000_000:
        millions = count / 1_000_000
        return f"{millions:g}M" if millions % 1 == 0 else f"{millions:.1f}M"
    if count >= 1_000:
        thousands = count / 1_000
        return f"{thousands:g}K" if thousands % 1 == 0 else f"{thousands:.1f}K"
    return str(count)


async def list_models(model_runtime, search_pattern: str | None = None, cancel=None) -> None:
    """List available models, optionally filtered by search pattern."""
    from pidrei_ai.auth.types import AuthOperationOptions

    load_error = model_runtime.get_error()
    if load_error:
        print(yellow(f"Warning: errors loading models.json:\n{load_error}"), file=sys.stderr)

    models = list(await model_runtime.get_available(None, AuthOperationOptions(cancel=cancel)))

    if not models:
        print(format_no_models_available_message())
        return

    # Apply fuzzy filter if search pattern provided
    filtered_models = models
    if search_pattern:
        filtered_models = fuzzy_filter(models, search_pattern, lambda m: f"{m.provider} {m.id}")

    if not filtered_models:
        print(f'No models matching "{search_pattern}"')
        return

    # Sort by provider, then by model id
    filtered_models.sort(key=lambda m: (m.provider, m.id))

    # Calculate column widths
    rows = [
        {
            "provider": m.provider,
            "model": m.id,
            "context": format_token_count(m.context_window),
            "max_out": format_token_count(m.max_tokens),
            "thinking": "yes" if m.reasoning else "no",
            "images": "yes" if "image" in m.input else "no",
        }
        for m in filtered_models
    ]

    headers = {
        "provider": "provider",
        "model": "model",
        "context": "context",
        "max_out": "max-out",
        "thinking": "thinking",
        "images": "images",
    }

    widths = {key: max(len(headers[key]), *(len(row[key]) for row in rows)) for key in headers}

    columns = ["provider", "model", "context", "max_out", "thinking", "images"]

    # Print header
    print("  ".join(headers[key].ljust(widths[key]) for key in columns))

    # Print rows
    for row in rows:
        print("  ".join(row[key].ljust(widths[key]) for key in columns))
