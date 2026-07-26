"""Port of pi's moonshotai-cn provider factory (packages/ai/src/providers/moonshotai-cn.ts)."""

from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def moonshotai_cn_provider() -> Provider:
    return create_provider(
        id="moonshotai-cn",
        name="Moonshot AI CN",
        base_url="https://api.moonshot.cn/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Moonshot AI API key", ["MOONSHOT_API_KEY"])),
        models=list(MODELS.get("moonshotai-cn", [])),
        api=openai_completions_api(),
    )
