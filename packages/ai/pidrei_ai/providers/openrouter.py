"""Port of pi's openrouter provider factory (packages/ai/src/providers/openrouter.ts)."""

from pidrei_ai.api.anthropic_messages_lazy import anthropic_messages_api
from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.api.openrouter_images_lazy import openrouter_images_api
from pidrei_ai.api.typesafe_system_one_lazy import typesafe_system_one_api
from pidrei_ai.auth.helpers import env_api_key_auth, lazy_oauth
from pidrei_ai.auth.oauth.load import load_openrouter_oauth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import CLASSIFIER_MODELS, IMAGE_MODELS, MODELS
from pidrei_ai.registry import Provider, create_provider


def openrouter_provider() -> Provider:
    return create_provider(
        id="openrouter",
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        auth=ProviderAuth(
            api_key=env_api_key_auth("OpenRouter API key", ["OPENROUTER_API_KEY"]),
            oauth=lazy_oauth(
                name="OpenRouter OAuth", load=load_openrouter_oauth, login_label="Sign in with OpenRouter"
            ),
        ),
        models=[
            *MODELS.get("openrouter", []),
            *IMAGE_MODELS.get("openrouter", []),
            *CLASSIFIER_MODELS.get("openrouter", []),
        ],
        api={
            "anthropic-messages": anthropic_messages_api(),
            "openai-completions": openai_completions_api(),
        },
        images={"openrouter-images": openrouter_images_api()},
        # OpenRouter serves TypeSafe's System One protocol at /api/v1/systemone.
        classifiers={"typesafe-system-one": typesafe_system_one_api()},
    )
