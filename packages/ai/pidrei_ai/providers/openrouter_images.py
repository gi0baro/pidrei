"""Port of pi's openrouter-images provider factory."""

from pidrei_ai.api.openrouter_images_lazy import openrouter_images_api
from pidrei_ai.auth.helpers import env_api_key_auth, lazy_oauth
from pidrei_ai.auth.oauth.load import load_openrouter_oauth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.image_models_generated import IMAGE_MODELS
from pidrei_ai.images_models import create_images_provider


def openrouter_images_provider():
    return create_images_provider(
        id="openrouter",
        name="OpenRouter",
        auth=ProviderAuth(
            api_key=env_api_key_auth("OpenRouter API key", ["OPENROUTER_API_KEY"]),
            oauth=lazy_oauth(
                name="OpenRouter OAuth",
                login_label="Sign in with OpenRouter",
                load=load_openrouter_oauth,
            ),
        ),
        models=list(IMAGE_MODELS.get("openrouter", {}).values()),
        api=openrouter_images_api(),
    )
