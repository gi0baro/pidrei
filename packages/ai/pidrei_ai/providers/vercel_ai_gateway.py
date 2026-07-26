"""Port of pi's vercel-ai-gateway provider factory (packages/ai/src/providers/vercel-ai-gateway.ts)."""

from pidrei_ai.api.anthropic_messages_lazy import anthropic_messages_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def vercel_ai_gateway_provider() -> Provider:
    return create_provider(
        id="vercel-ai-gateway",
        name="Vercel AI Gateway",
        base_url="https://ai-gateway.vercel.sh",
        auth=ProviderAuth(api_key=env_api_key_auth("Vercel AI Gateway API key", ["AI_GATEWAY_API_KEY"])),
        models=list(MODELS.get("vercel-ai-gateway", [])),
        api=anthropic_messages_api(),
    )
