"""Port of pi's anthropic provider factory (packages/ai/src/providers/anthropic.ts)."""

from pidrei_ai.api.anthropic_messages_lazy import anthropic_messages_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def anthropic_provider() -> Provider:
    return create_provider(
        id="anthropic",
        name="Anthropic",
        base_url="https://api.anthropic.com",
        auth=ProviderAuth(
            # ANTHROPIC_OAUTH_TOKEN takes precedence over ANTHROPIC_API_KEY
            api_key=env_api_key_auth("Anthropic API key", ["ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"]),
            # OAuth login flow (Anthropic Claude Pro/Max) lands in Phase 5b.
            oauth=None,
        ),
        models=list(MODELS.get("anthropic", [])),
        api=anthropic_messages_api(),
    )
