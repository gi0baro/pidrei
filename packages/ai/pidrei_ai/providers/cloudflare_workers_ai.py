"""Port of pi's cloudflare-workers-ai provider factory."""

from pidrei_ai.api.cloudflare_workers_ai_system_one_lazy import cloudflare_workers_ai_system_one_api
from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import CLASSIFIER_MODELS, MODELS
from pidrei_ai.providers.cloudflare_auth import cloudflare_workers_ai_auth
from pidrei_ai.providers.cloudflare_stream import cloudflare_classifier, cloudflare_streams
from pidrei_ai.registry import Provider, create_provider


def cloudflare_workers_ai_provider() -> Provider:
    return create_provider(
        id="cloudflare-workers-ai",
        name="Cloudflare Workers AI",
        auth=ProviderAuth(api_key=cloudflare_workers_ai_auth()),
        models=[*MODELS.get("cloudflare-workers-ai", []), *CLASSIFIER_MODELS.get("cloudflare-workers-ai", [])],
        api=cloudflare_streams(openai_completions_api()),
        classifiers={"cloudflare-workers-ai-system-one": cloudflare_classifier(cloudflare_workers_ai_system_one_api())},
    )
