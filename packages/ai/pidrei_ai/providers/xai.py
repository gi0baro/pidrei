"""Port of pi's xai provider factory (packages/ai/src/providers/xai.ts).

`api` dispatches on `model.api`: openai-completions, openai-responses.
"""

from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.api.openai_responses_lazy import openai_responses_api
from pidrei_ai.auth.helpers import env_api_key_auth, lazy_oauth
from pidrei_ai.auth.oauth.load import load_xai_oauth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def xai_provider() -> Provider:
    return create_provider(
        id="xai",
        name="xAI",
        base_url="https://api.x.ai/v1",
        auth=ProviderAuth(
            api_key=env_api_key_auth("xAI API key", ["XAI_API_KEY"]),
            oauth=lazy_oauth(
                name="xAI (Grok/X subscription)",
                is_subscription=True,
                load=load_xai_oauth,
                login_label="Sign in with SuperGrok or X Premium",
            ),
        ),
        models=list(MODELS.get("xai", [])),
        api={
            "openai-completions": openai_completions_api(),
            "openai-responses": openai_responses_api(),
        },
    )
