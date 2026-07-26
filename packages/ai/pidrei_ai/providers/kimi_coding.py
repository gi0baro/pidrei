"""Port of pi's kimi-coding provider factory (packages/ai/src/providers/kimi-coding.ts)."""

from pidrei_ai.api.anthropic_messages_lazy import anthropic_messages_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def kimi_coding_provider() -> Provider:
    return create_provider(
        id="kimi-coding",
        name="Kimi For Coding",
        base_url="https://api.kimi.com/coding",
        auth=ProviderAuth(
            api_key=env_api_key_auth("Kimi API key", ["KIMI_API_KEY"]),
            # OAuth login flow "Kimi Code (subscription)" lands in Phase 5b.
            oauth=None,
        ),
        models=list(MODELS.get("kimi-coding", [])),
        api=anthropic_messages_api(),
    )
