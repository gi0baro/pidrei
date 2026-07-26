"""Port of pi's azure-openai-responses provider factory
(packages/ai/src/providers/azure-openai-responses.ts)."""

from pidrei_ai.api.azure_openai_responses_lazy import azure_openai_responses_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def azure_openai_responses_provider() -> Provider:
    return create_provider(
        id="azure-openai-responses",
        name="Azure OpenAI",
        auth=ProviderAuth(api_key=env_api_key_auth("Azure OpenAI API key", ["AZURE_OPENAI_API_KEY"])),
        models=list(MODELS.get("azure-openai-responses", [])),
        api=azure_openai_responses_api(),
    )
