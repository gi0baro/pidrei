"""Port of pi's provider env resolution (packages/ai/src/utils/provider-env.ts).

Scoped overrides take precedence over the process environment. pi's extra
fallback (reading /proc/self/environ inside Bun sandboxes) is a Bun-specific
workaround and is intentionally not ported.
"""

import os

from pidrei_ai.types import ProviderEnv


def get_provider_env_value(name: str, env: ProviderEnv | None = None) -> str | None:
    """Resolve a provider env value from scoped overrides, then os.environ."""
    if env:
        value = env.get(name)
        if value:
            return value
    return os.environ.get(name) or None
