"""Port of pi's xiaomi-token-plan-ams provider factory (packages/ai/src/providers/xiaomi-token-plan-ams.ts)."""

from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def xiaomi_token_plan_ams_provider() -> Provider:
    return create_provider(
        id="xiaomi-token-plan-ams",
        name="Xiaomi Token Plan AMS",
        base_url="https://token-plan-ams.xiaomimimo.com/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Xiaomi Token Plan AMS API key", ["XIAOMI_TOKEN_PLAN_AMS_API_KEY"])),
        models=list(MODELS.get("xiaomi-token-plan-ams", [])),
        api=openai_completions_api(),
    )
