"""Port of pi's groq provider factory (packages/ai/src/providers/groq.ts)."""

from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def groq_provider() -> Provider:
    return create_provider(
        id="groq",
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Groq API key", ["GROQ_API_KEY"])),
        models=list(MODELS.get("groq", [])),
        api=openai_completions_api(),
    )
