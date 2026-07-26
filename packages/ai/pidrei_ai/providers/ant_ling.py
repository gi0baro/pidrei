"""Port of pi's ant-ling provider factory (packages/ai/src/providers/ant-ling.ts)."""

from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def ant_ling_provider() -> Provider:
    return create_provider(
        id="ant-ling",
        name="Ant Ling",
        base_url="https://api.ant-ling.com/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Ant Ling API key", ["ANT_LING_API_KEY"])),
        models=list(MODELS.get("ant-ling", [])),
        api=openai_completions_api(),
    )
