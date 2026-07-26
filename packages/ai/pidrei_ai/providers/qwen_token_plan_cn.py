"""Port of pi's qwen-token-plan-cn provider factory (packages/ai/src/providers/qwen-token-plan-cn.ts)."""

from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def qwen_token_plan_cn_provider() -> Provider:
    return create_provider(
        id="qwen-token-plan-cn",
        name="Qwen Token Plan CN",
        base_url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Qwen Token Plan CN API key", ["QWEN_TOKEN_PLAN_CN_API_KEY"])),
        models=list(MODELS.get("qwen-token-plan-cn", [])),
        api=openai_completions_api(),
    )
