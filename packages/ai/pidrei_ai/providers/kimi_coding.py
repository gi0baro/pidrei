"""Port of pi's kimi-coding provider factory (packages/ai/src/providers/kimi-coding.ts)."""

from pidrei_ai.api.anthropic_messages_lazy import anthropic_messages_api
from pidrei_ai.auth.helpers import env_api_key_auth, lazy_oauth
from pidrei_ai.auth.oauth.load import load_kimi_coding_oauth
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
            oauth=lazy_oauth(
                name="Kimi Code (subscription)",
                is_subscription=True,
                load=load_kimi_coding_oauth,
                login_label="Sign in with Kimi Code",
            ),
        ),
        models=list(MODELS.get("kimi-coding", [])),
        api=anthropic_messages_api(),
    )
