"""Port of pi's cloudflare-ai-gateway provider factory."""

from pidrei_ai.api.anthropic_messages_lazy import anthropic_messages_api
from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.api.openai_responses_lazy import openai_responses_api
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.providers.cloudflare_auth import cloudflare_ai_gateway_auth
from pidrei_ai.providers.cloudflare_stream import cloudflare_streams
from pidrei_ai.registry import Provider, create_provider


def cloudflare_ai_gateway_provider() -> Provider:
    return create_provider(
        id="cloudflare-ai-gateway",
        name="Cloudflare AI Gateway",
        auth=ProviderAuth(api_key=cloudflare_ai_gateway_auth()),
        models=list(MODELS.get("cloudflare-ai-gateway", [])),
        api={
            "anthropic-messages": cloudflare_streams(anthropic_messages_api()),
            "openai-completions": cloudflare_streams(openai_completions_api()),
            "openai-responses": cloudflare_streams(openai_responses_api()),
        },
    )
