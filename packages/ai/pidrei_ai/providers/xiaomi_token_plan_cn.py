"""Port of pi's xiaomi-token-plan-cn provider factory (packages/ai/src/providers/xiaomi-token-plan-cn.ts)."""

from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def xiaomi_token_plan_cn_provider() -> Provider:
    return create_provider(
        id="xiaomi-token-plan-cn",
        name="Xiaomi Token Plan CN",
        base_url="https://token-plan-cn.xiaomimimo.com/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Xiaomi Token Plan CN API key", ["XIAOMI_TOKEN_PLAN_CN_API_KEY"])),
        models=list(MODELS.get("xiaomi-token-plan-cn", [])),
        api=openai_completions_api(),
    )
