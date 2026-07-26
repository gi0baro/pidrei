"""Port of pi's xiaomi-token-plan-sgp provider factory (packages/ai/src/providers/xiaomi-token-plan-sgp.ts)."""

from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def xiaomi_token_plan_sgp_provider() -> Provider:
    return create_provider(
        id="xiaomi-token-plan-sgp",
        name="Xiaomi Token Plan SGP",
        base_url="https://token-plan-sgp.xiaomimimo.com/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Xiaomi Token Plan SGP API key", ["XIAOMI_TOKEN_PLAN_SGP_API_KEY"])),
        models=list(MODELS.get("xiaomi-token-plan-sgp", [])),
        api=openai_completions_api(),
    )
