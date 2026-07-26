"""Port of pi's builtin provider aggregate (packages/ai/src/providers/all.ts).

Provider factories join `builtin_providers()` as their adapters land
(PLAN.md); the catalog reads (`get_builtin_model`/`get_builtin_models`) cover
every provider present in the vendored data regardless.
"""

import json
from datetime import datetime
from pathlib import Path

from pidrei_ai.models_generated import MODELS
from pidrei_ai.providers.ant_ling import ant_ling_provider
from pidrei_ai.providers.anthropic import anthropic_provider
from pidrei_ai.providers.cerebras import cerebras_provider
from pidrei_ai.providers.deepseek import deepseek_provider
from pidrei_ai.providers.fireworks import fireworks_provider
from pidrei_ai.providers.groq import groq_provider
from pidrei_ai.providers.huggingface import huggingface_provider
from pidrei_ai.providers.kimi_coding import kimi_coding_provider
from pidrei_ai.providers.minimax import minimax_provider
from pidrei_ai.providers.minimax_cn import minimax_cn_provider
from pidrei_ai.providers.moonshotai import moonshotai_provider
from pidrei_ai.providers.moonshotai_cn import moonshotai_cn_provider
from pidrei_ai.providers.nvidia import nvidia_provider
from pidrei_ai.providers.openai import openai_provider
from pidrei_ai.providers.opencode import opencode_provider
from pidrei_ai.providers.opencode_go import opencode_go_provider
from pidrei_ai.providers.openrouter import openrouter_provider
from pidrei_ai.providers.qwen_token_plan import qwen_token_plan_provider
from pidrei_ai.providers.qwen_token_plan_cn import qwen_token_plan_cn_provider
from pidrei_ai.providers.together import together_provider
from pidrei_ai.providers.vercel_ai_gateway import vercel_ai_gateway_provider
from pidrei_ai.providers.xai import xai_provider
from pidrei_ai.providers.xiaomi import xiaomi_provider
from pidrei_ai.providers.xiaomi_token_plan_ams import xiaomi_token_plan_ams_provider
from pidrei_ai.providers.xiaomi_token_plan_cn import xiaomi_token_plan_cn_provider
from pidrei_ai.providers.xiaomi_token_plan_sgp import xiaomi_token_plan_sgp_provider
from pidrei_ai.providers.zai import zai_provider
from pidrei_ai.providers.zai_coding_cn import zai_coding_cn_provider
from pidrei_ai.registry import Models, Provider, create_models
from pidrei_ai.types import Model


def get_builtin_model(provider: str, model_id: str) -> Model | None:
    """Read of the generated built-in catalog."""
    for model in MODELS.get(provider, []):
        if model.id == model_id:
            return model
    return None


def get_builtin_models(provider: str) -> list[Model]:
    return list(MODELS.get(provider, []))


def get_builtin_providers() -> list[str]:
    return list(MODELS.keys())


def get_builtin_model_data_generated_at() -> int | None:
    """Generation timestamp (ms) shared by all built-in provider catalogs."""
    manifest_path = Path(__file__).parent / "data" / "_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        return int(datetime.fromisoformat(manifest["generatedAt"]).timestamp() * 1000)
    except Exception:
        return None


def builtin_providers() -> list[Provider]:
    """All built-in providers, freshly constructed, in pi's order.

    Still to join, with their adapters (PLAN.md): amazon-bedrock, azure-openai-
    responses, cloudflare-ai-gateway, cloudflare-workers-ai, github-copilot,
    google, google-vertex, mistral, openai-codex. Their models are in the
    vendored catalog already, so the catalog reads above cover them.
    """
    return [
        ant_ling_provider(),
        anthropic_provider(),
        cerebras_provider(),
        deepseek_provider(),
        fireworks_provider(),
        groq_provider(),
        huggingface_provider(),
        kimi_coding_provider(),
        minimax_provider(),
        minimax_cn_provider(),
        moonshotai_provider(),
        moonshotai_cn_provider(),
        nvidia_provider(),
        openai_provider(),
        opencode_provider(),
        opencode_go_provider(),
        openrouter_provider(),
        qwen_token_plan_provider(),
        qwen_token_plan_cn_provider(),
        together_provider(),
        vercel_ai_gateway_provider(),
        xai_provider(),
        xiaomi_provider(),
        xiaomi_token_plan_ams_provider(),
        xiaomi_token_plan_cn_provider(),
        xiaomi_token_plan_sgp_provider(),
        zai_provider(),
        zai_coding_cn_provider(),
    ]


def builtin_models(**options) -> Models:
    """A `Models` collection with every built-in provider registered."""
    models = create_models(**options)
    for provider in builtin_providers():
        models.set_provider(provider)
    return models
