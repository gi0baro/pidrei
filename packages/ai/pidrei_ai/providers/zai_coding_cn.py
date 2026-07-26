"""Port of pi's zai-coding-cn provider factory (packages/ai/src/providers/zai-coding-cn.ts)."""

from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def zai_coding_cn_provider() -> Provider:
    return create_provider(
        id="zai-coding-cn",
        name="Z.AI Coding CN",
        base_url="https://open.bigmodel.cn/api/coding/paas/v4",
        auth=ProviderAuth(api_key=env_api_key_auth("Z.AI Coding CN API key", ["ZAI_CODING_CN_API_KEY"])),
        models=list(MODELS.get("zai-coding-cn", [])),
        api=openai_completions_api(),
    )
