"""Port of pi's openai provider factory (packages/ai/src/providers/openai.ts)."""

from pidrei_ai.api.openai_responses_lazy import openai_responses_api
from pidrei_ai.auth.helpers import env_api_key_auth, lazy_oauth
from pidrei_ai.auth.oauth.load import load_openai_chatgpt_oauth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def openai_provider() -> Provider:
    return create_provider(
        id="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        auth=ProviderAuth(
            api_key=env_api_key_auth("OpenAI API key", ["OPENAI_API_KEY"]),
            oauth=lazy_oauth(
                name="OpenAI (ChatGPT subscription)",
                is_subscription=True,
                login_label="Sign in with ChatGPT",
                load=load_openai_chatgpt_oauth,
            ),
        ),
        models=list(MODELS.get("openai", [])),
        api=openai_responses_api(),
    )
