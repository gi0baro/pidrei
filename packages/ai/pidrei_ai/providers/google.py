"""Port of pi's google provider factory (packages/ai/src/providers/google.ts)."""

from pidrei_ai.api.google_generative_ai_lazy import google_generative_ai_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def google_provider() -> Provider:
    return create_provider(
        id="google",
        name="Google",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        auth=ProviderAuth(api_key=env_api_key_auth("Gemini API key", ["GEMINI_API_KEY"])),
        models=list(MODELS.get("google", [])),
        api=google_generative_ai_api(),
    )
