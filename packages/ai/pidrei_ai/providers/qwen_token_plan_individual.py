"""Port of pi's Qwen Token Plan Individual provider factory
(packages/ai/src/providers/qwen-token-plan-individual.ts)."""

from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def qwen_token_plan_individual_provider() -> Provider:
    return create_provider(
        id="qwen-token-plan-individual",
        name="Qwen Token Plan Individual",
        base_url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Qwen Token Plan Individual API key", ["QWEN_TOKEN_PLAN_API_KEY"])),
        models=list(MODELS.get("qwen-token-plan-individual", [])),
        api=openai_completions_api(),
    )
