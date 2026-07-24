"""Port of pi's anthropic provider factory (packages/ai/src/providers/anthropic.ts)."""

from pppi_ai.api.lazy import lazy_api
from pppi_ai.auth.helpers import env_api_key_auth
from pppi_ai.auth.types import ProviderAuth
from pppi_ai.models_generated import MODELS
from pppi_ai.registry import Provider, create_provider


def anthropic_messages_api():
    """Lazy wrapper for the anthropic-messages adapter (pi: anthropic-messages.lazy.ts)."""

    async def load():
        from pppi_ai.api import anthropic_messages

        return anthropic_messages

    return lazy_api(load)


def anthropic_provider() -> Provider:
    return create_provider(
        id="anthropic",
        name="Anthropic",
        base_url="https://api.anthropic.com",
        auth=ProviderAuth(
            # ANTHROPIC_OAUTH_TOKEN takes precedence over ANTHROPIC_API_KEY
            api_key=env_api_key_auth("Anthropic API key", ["ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"]),
            # OAuth login flow (Anthropic Claude Pro/Max) lands in Phase 5.
            oauth=None,
        ),
        models=list(MODELS.get("anthropic", [])),
        api=anthropic_messages_api(),
    )
