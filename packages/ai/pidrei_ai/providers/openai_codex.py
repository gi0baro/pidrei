"""Port of pi's openai-codex provider factory (packages/ai/src/providers/openai-codex.ts)."""

from pidrei_ai.api.openai_codex_responses_lazy import openai_codex_responses_api
from pidrei_ai.auth.helpers import lazy_oauth
from pidrei_ai.auth.oauth.load import load_openai_codex_oauth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def openai_codex_provider() -> Provider:
    return create_provider(
        id="openai-codex",
        name="OpenAI Codex",
        base_url="https://chatgpt.com/backend-api",
        auth=ProviderAuth(
            oauth=lazy_oauth(
                name="OpenAI (ChatGPT Plus/Pro)",
                is_subscription=True,
                load=load_openai_codex_oauth,
            )
        ),
        models=list(MODELS.get("openai-codex", [])),
        api=openai_codex_responses_api(),
    )
