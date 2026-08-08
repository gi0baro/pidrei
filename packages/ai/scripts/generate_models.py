"""Model catalog generator — Python port of pi's scripts/generate-models.ts.

Fetches https://models.dev/api.json (plus the NVIDIA NIM, OpenRouter and Vercel
AI Gateway live catalogs) and emits pi-shaped catalog JSON (camelCase keys,
grouped `{api: {modelId: Model}}`, sorted) into
`pidrei_ai/providers/data/<provider>.json`.

Key *order* inside each model object mirrors pi's object literals so the vendored
data stays diffable against pi's generated catalog; the metadata passes append
`compat`/`thinkingLevelMap` last, exactly as assigning a new property does in JS.

Deviation from pi: pi's fetch helpers log-and-continue unless `--strict` is
passed, which would silently vendor a catalog missing a whole provider. Here
every fetch failure is fatal — a failed run beats a quietly truncated artifact.

Run: make models-data   (network access required)
"""

import json
import re
import sys
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tonio.colored as tonio
from punkreq.tonio import Client


sys.path.insert(0, str(Path(__file__).parent))

from model_data import (  # sibling script, not an installed module
    MODEL_DATA_MANIFEST_FILE,
    ModelDataStructure,
    assert_exact_model_ids,
    create_model_data_manifest,
    validate_generated_model_data,
    validate_model_data_directory,
)


DATA_DIR = Path(__file__).parent.parent / "pidrei_ai" / "providers" / "data"

# --- provider constants (pi: generate-models.ts:142-395) ----------------------

COPILOT_STATIC_HEADERS = {
    "User-Agent": "GitHubCopilotChat/0.35.0",
    "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0",
    "Copilot-Integration-Id": "vscode-chat",
}

KIMI_STATIC_HEADERS = {"User-Agent": "KimiCLI/1.5"}

# pi: src/api/cloudflare.ts
CLOUDFLARE_WORKERS_AI_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1"
CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL = (
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/compat"
)
CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL = (
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/openai"
)
CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL = (
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/anthropic"
)

TOGETHER_BASE_URL = "https://api.together.ai/v1"
TOGETHER_BASE_COMPAT: dict[str, Any] = {
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "supportsReasoningEffort": False,
    "maxTokensField": "max_tokens",
    "supportsStrictMode": False,
    "supportsLongCacheRetention": False,
}
TOGETHER_TOGGLE_REASONING_COMPAT = {**TOGETHER_BASE_COMPAT, "thinkingFormat": "together"}
TOGETHER_REASONING_EFFORT_COMPAT = {
    **TOGETHER_BASE_COMPAT,
    "supportsReasoningEffort": True,
    "thinkingFormat": "openai",
}
TOGETHER_TOGGLE_REASONING_EFFORT_COMPAT = {**TOGETHER_TOGGLE_REASONING_COMPAT, "supportsReasoningEffort": True}
TOGETHER_REASONING_ONLY_MODELS = {"deepseek-ai/DeepSeek-R1", "MiniMaxAI/MiniMax-M2.7"}
TOGETHER_REASONING_EFFORT_MODELS = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
TOGETHER_TOGGLE_REASONING_EFFORT_MODELS = {"deepseek-ai/DeepSeek-V4-Pro"}
TOGETHER_FIXED_REASONING_LEVEL_MAP: dict[str, str | None] = {
    "off": None,
    "minimal": None,
    "low": None,
    "medium": None,
}
TOGETHER_REASONING_EFFORT_LEVEL_MAP: dict[str, str | None] = {"off": None, "minimal": None}
TOGETHER_DEEPSEEK_V4_THINKING_LEVEL_MAP: dict[str, str | None] = {
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "xhigh": None,
}
TOGETHER_TOGGLE_REASONING_LEVEL_MAP: dict[str, str | None] = {"minimal": None, "low": None, "medium": None}

AI_GATEWAY_MODELS_URL = "https://ai-gateway.vercel.sh/v1"
AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh"
VERTEX_BASE_URL = "https://{location}-aiplatform.googleapis.com"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_HEADERS = {"NVCF-POLL-SECONDS": "3600"}
NVIDIA_OPENAI_COMPAT: dict[str, Any] = {
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "supportsReasoningEffort": False,
    "maxTokensField": "max_tokens",
    "supportsStrictMode": False,
    "supportsLongCacheRetention": False,
}
NVIDIA_NIM_UNSUPPORTED_MODELS = {
    "abacusai/dracarys-llama-3.1-70b-instruct",
    "bytedance/seed-oss-36b-instruct",
    "deepseek-ai/deepseek-v4-flash",
    "deepseek-ai/deepseek-v4-pro",
    "google/gemma-2-2b-it",
    "google/gemma-3n-e2b-it",
    "google/gemma-3n-e4b-it",
    "google/gemma-4-31b-it",
    "meta/llama-3.2-1b-instruct",
    "meta/llama-4-maverick-17b-128e-instruct",
    "microsoft/phi-4-mini-instruct",
    "minimaxai/minimax-m2.7",
    "mistralai/mistral-nemotron",
    "nvidia/nemotron-mini-4b-instruct",
    "qwen/qwen3-next-80b-a3b-instruct",
    "qwen/qwen3.5-397b-a17b",
    "sarvamai/sarvam-m",
    "upstage/solar-10.7b-instruct",
}
ZAI_TOOL_STREAM_UNSUPPORTED_MODELS = {"glm-4.5", "glm-4.5-air", "glm-4.5-flash", "glm-4.5v"}
ZAI_GLM52_THINKING_LEVEL_MAP: dict[str, str | None] = {
    "minimal": None,
    "low": "high",
    "medium": "high",
    "high": "high",
    "max": "max",
}
OPENCODE_GO_GLM52_THINKING_LEVEL_MAP: dict[str, str | None] = {
    "off": None,
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "max": "max",
}
EAGER_TOOL_INPUT_STREAMING_UNSUPPORTED_ANTHROPIC_MODELS = {
    "github-copilot:claude-haiku-4.5",
    "github-copilot:claude-sonnet-4",
    "github-copilot:claude-sonnet-4.5",
}

DEEPSEEK_V4_THINKING_LEVEL_MAP: dict[str, str | None] = {
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "max": "max",
}

QWEN_TOKEN_PLAN_HIGH_MAX_THINKING_LEVEL_MAP: dict[str, str | None] = {
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "xhigh": None,
    "max": "max",
}
QWEN_TOKEN_PLAN_QWEN38_THINKING_LEVEL_MAP: dict[str, str | None] = {
    "minimal": None,
    "low": "low",
    "medium": "medium",
    "high": None,
    "xhigh": "xhigh",
    "max": None,
}
QWEN_TOKEN_PLAN_REASONING_EFFORT_UNSUPPORTED_MODEL_IDS = {
    "MiniMax-M2.5",
    "deepseek-v3.2",
    "kimi-k2.5",
    "kimi-k2.6",
    "kimi-k2.7-code",
    "qwen3.6-flash",
    "qwen3.6-plus",
    "qwen3.7-max",
    "qwen3.7-plus",
}
# Retired preview id — models.dev may still list it after GA ships.
QWEN_TOKEN_PLAN_EXCLUDED_MODEL_IDS = {"qwen3.8-max-preview"}
QWEN_TOKEN_PLAN_PROVIDER_IDS = {"qwen-token-plan", "qwen-token-plan-cn", "qwen-token-plan-individual"}
# QwenCloud Token Plan Individual text-model allowlist, verified 2026-08-05.
# Retired models remain excluded above even if the public catalog lags.
# https://docs.qwencloud.com/token-plan/personal/token-plan-personal-overview
QWEN_TOKEN_PLAN_INDIVIDUAL_MODEL_IDS = {
    "deepseek-v4-flash-0731",
    "deepseek-v4-pro",
    "glm-5.2",
    "qwen3.6-flash",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.8-max",
}

KIMI_K3_MAX_TOKENS = 131072
KIMI_K3_COST = {"input": 3, "output": 15, "cacheRead": 0.3, "cacheWrite": 0}
# Kimi Coding is subscription-backed, so models.dev reports zero cost. Use the
# equivalent Moonshot API rates to estimate the value of subscription usage.
KIMI_CODING_IMPLIED_COSTS: dict[str, dict[str, float]] = {
    "k3": KIMI_K3_COST,
    "kimi-for-coding": {"input": 0.95, "output": 4, "cacheRead": 0.19, "cacheWrite": 0},
    "kimi-for-coding-highspeed": {"input": 1.9, "output": 8, "cacheRead": 0.38, "cacheWrite": 0},
    "kimi-k2-thinking": {"input": 0.6, "output": 2.5, "cacheRead": 0.15, "cacheWrite": 0},
}
OPENROUTER_KIMI_K3_MODEL_IDS = {"moonshotai/kimi-k3", "~moonshotai/kimi-latest"}

ANT_LING_RING_THINKING_LEVEL_MAP: dict[str, str | None] = {
    "off": None,
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "xhigh": "xhigh",
}

BEDROCK_INFERENCE_PROFILE_ONLY_MODEL_IDS = {"anthropic.claude-opus-5"}
MODELS_DEV_OPENAI_UNSUPPORTED_MODEL_IDS = {"gpt-5.6"}
OPENAI_TOOL_SEARCH_MODEL_IDS = {
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-pro",
    "gpt-5.5",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
}
OPENAI_LONG_CONTEXT_INPUT_THRESHOLD = 272000
OPENAI_SHORT_CONTEXT_CAPPED_MODEL_IDS = {
    "gpt-5.4",
    "gpt-5.5",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
}
OPENAI_LONG_CONTEXT_PRICING_MODEL_IDS = {
    "gpt-5.4",
    "gpt-5.4-pro",
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
}
OPENAI_RESPONSES_NONE_REASONING_MODELS = {
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.5",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
}
XAI_RESPONSES_MODEL_ID = "grok-4.5"
XAI_BUILTIN_EXCLUDED_MODEL_IDS = {
    "grok-3",
    "grok-3-fast",
    "grok-4.20-0309-non-reasoning",
    "grok-4.20-0309-reasoning",
    "grok-code-fast-1",
}
XAI_RESPONSES_EFFORT_LEVEL_MAP: dict[str, str | None] = {"off": None, "minimal": None}
XAI_RESPONSES_COMPAT: dict[str, Any] = {"supportsLongCacheRetention": False}

OPENCODE_OPENAI_COMPLETIONS_LONG_CACHE_RETENTION_UNSUPPORTED_MODELS = {
    "opencode:deepseek-v4-flash",
    "opencode:deepseek-v4-pro",
    "opencode:kimi-k2.5",
    "opencode:kimi-k2.6",
    "opencode:minimax-m2.7",
    "opencode-go:kimi-k2.6",
}

# GitHub's "Models with extended capabilities" table lists these Copilot models as
# supporting the extended 1 million token context window.
GITHUB_COPILOT_EXTENDED_CONTEXT_MODELS = {
    "claude-fable-5",
    "claude-opus-4.6",
    "claude-opus-4.7",
    "claude-opus-4.8",
    "claude-opus-5",
    "claude-sonnet-4.6",
    "claude-sonnet-5",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-5.5",
}

# Checked manually against the authenticated GitHub Copilot /models endpoint on
# 2026-06-15. Narrow corrections over models.dev metadata, not a snapshot.
GITHUB_COPILOT_THINKING_LEVEL_OVERRIDES: dict[str, dict[str, str | None]] = {
    "claude-opus-4.7": {"minimal": "low"},
    "claude-opus-4.8": {"minimal": "low"},
    "claude-opus-5": {"minimal": "low"},
    "claude-sonnet-4.6": {"minimal": "low", "max": "max"},
}

THINKING_LEVELS = ["minimal", "low", "medium", "high", "xhigh", "max"]

_GEMINI_3_PRO_RE = re.compile(r"gemini-3(?:\.\d+)?-pro")
_GEMINI_3_FLASH_RE = re.compile(r"gemini-3(?:\.\d+)?-flash")
_GEMMA_4_RE = re.compile(r"gemma-?4")
_COPILOT_CLAUDE_RE = re.compile(r"^claude-(haiku|sonnet|opus)-[45]([.\-]|$)")


def with_openai_long_context_pricing(cost: dict[str, Any]) -> dict[str, Any]:
    return {
        **cost,
        "tiers": [
            {
                "inputTokensAbove": OPENAI_LONG_CONTEXT_INPUT_THRESHOLD,
                "input": round_cost(cost["input"] * 2),
                "output": round_cost(cost["output"] * 1.5),
                "cacheRead": round_cost(cost["cacheRead"] * 2),
                "cacheWrite": round_cost(cost["cacheWrite"] * 2),
            }
        ],
    }


# OpenAI reduced GPT-5.6 Terra and Luna prices on 2026-07-30. Keep these
# authoritative values until models.dev and passthrough catalogs catch up.
# https://developers.openai.com/api/docs/pricing
OPENAI_GPT_56_STANDARD_COSTS: dict[str, dict[str, Any]] = {
    "gpt-5.6-luna": {"input": 0.2, "output": 1.2, "cacheRead": 0.02, "cacheWrite": 0.25},
    "gpt-5.6-terra": {"input": 2, "output": 12, "cacheRead": 0.2, "cacheWrite": 2.5},
}


def merge_thinking_level_map(model: dict[str, Any], mapping: dict[str, str | None]) -> None:
    model["thinkingLevelMap"] = {**model.get("thinkingLevelMap", {}), **mapping}


def merge_compat(model: dict[str, Any], compat: dict[str, Any]) -> None:
    model["compat"] = {**model.get("compat", {}), **compat}


def get_effort_thinking_level_map(options: list[dict[str, Any]]) -> dict[str, str | None] | None:
    """Port of scripts/models-dev-reasoning-options.ts."""
    effort_values = [
        value for option in options if option.get("type") == "effort" for value in option.get("values", [])
    ]
    if not effort_values:
        return None

    supported = set(effort_values)
    if not any(level in supported for level in THINKING_LEVELS) and "none" not in supported:
        return None

    mapping: dict[str, str | None] = {"off": "none" if "none" in supported else None}
    for level in THINKING_LEVELS:
        mapping[level] = level if level in supported else None
    return mapping


def get_together_compat(model_id: str, reasoning: bool) -> dict[str, Any]:
    if not reasoning:
        return TOGETHER_BASE_COMPAT
    if model_id in TOGETHER_REASONING_EFFORT_MODELS:
        return TOGETHER_REASONING_EFFORT_COMPAT
    if model_id in TOGETHER_TOGGLE_REASONING_EFFORT_MODELS:
        return TOGETHER_TOGGLE_REASONING_EFFORT_COMPAT
    if model_id in TOGETHER_REASONING_ONLY_MODELS:
        return TOGETHER_BASE_COMPAT
    return TOGETHER_TOGGLE_REASONING_COMPAT


def get_together_thinking_level_map(model_id: str, reasoning: bool) -> dict[str, str | None] | None:
    if not reasoning:
        return None
    if model_id in TOGETHER_REASONING_EFFORT_MODELS:
        return dict(TOGETHER_REASONING_EFFORT_LEVEL_MAP)
    if model_id in TOGETHER_TOGGLE_REASONING_EFFORT_MODELS:
        return dict(TOGETHER_DEEPSEEK_V4_THINKING_LEVEL_MAP)
    if model_id in TOGETHER_REASONING_ONLY_MODELS:
        return dict(TOGETHER_FIXED_REASONING_LEVEL_MAP)
    return dict(TOGETHER_TOGGLE_REASONING_LEVEL_MAP)


def is_anthropic_adaptive_thinking_model(model_id: str) -> bool:
    return any(
        marker in model_id
        for marker in (
            "opus-4-6",
            "opus-4.6",
            "opus-4-7",
            "opus-4.7",
            "opus-4-8",
            "opus-4.8",
            "opus-5",
            "opus.5",
            "sonnet-4-6",
            "sonnet-4.6",
            "sonnet-5",
            "sonnet.5",
            "fable-5",
        )
    )


def is_anthropic_temperature_unsupported_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return any(marker in lowered for marker in ("opus-4-7", "opus-4.7", "opus-4-8", "opus-4.8", "opus-5", "opus.5"))


def supports_openai_xhigh(model_id: str) -> bool:
    return any(marker in model_id for marker in ("gpt-5.2", "gpt-5.3", "gpt-5.4", "gpt-5.5", "gpt-5.6"))


def supports_openai_max(model: dict[str, Any]) -> bool:
    return "gpt-5.6" in model["id"] and model["api"] in (
        "openai-responses",
        "azure-openai-responses",
        "openai-codex-responses",
        "openai-completions",
    )


def is_google_thinking_api(model: dict[str, Any]) -> bool:
    return model["api"] in ("google-generative-ai", "google-vertex")


# --- openai-completions compat auto-detection (pi: generate-models.ts:502-647) -

OPENAI_COMPLETIONS_DEFAULT_COMPAT: dict[str, Any] = {
    "supportsStore": True,
    "supportsDeveloperRole": True,
    "supportsReasoningEffort": True,
    "supportsUsageInStreaming": True,
    "supportsFinishReason": True,
    "maxTokensField": "max_completion_tokens",
    "requiresToolResultName": False,
    "requiresAssistantAfterToolResult": False,
    "requiresThinkingAsText": False,
    "requiresReasoningContentOnAssistantMessages": False,
    "thinkingFormat": "openai",
    "openRouterRouting": {},
    "vercelGatewayRouting": {},
    "chatTemplateKwargs": {},
    "chatTemplateArgs": {},
    "zaiToolStream": False,
    "supportsStrictMode": True,
    "supportsOpenAIGrammarTools": False,
    "sendSessionAffinityHeaders": False,
    "supportsLongCacheRetention": True,
}


def detect_openai_completions_compat(model: dict[str, Any]) -> dict[str, Any]:
    provider = model["provider"]
    base_url = model["baseUrl"]
    model_id = model["id"]

    is_zai = provider in ("zai", "zai-coding-cn") or "api.z.ai" in base_url or "open.bigmodel.cn" in base_url
    is_together = provider == "together" or "api.together.ai" in base_url or "api.together.xyz" in base_url
    is_moonshot = provider in ("moonshotai", "moonshotai-cn") or "api.moonshot." in base_url
    is_openrouter = provider == "openrouter" or "openrouter.ai" in base_url
    is_cloudflare_workers_ai = provider == "cloudflare-workers-ai" or "api.cloudflare.com" in base_url
    is_cloudflare_ai_gateway = provider == "cloudflare-ai-gateway" or "gateway.ai.cloudflare.com" in base_url
    is_nvidia = provider == "nvidia" or "integrate.api.nvidia.com" in base_url
    is_ant_ling = provider == "ant-ling" or "api.ant-ling.com" in base_url
    is_together_reasoning_only = is_together and model_id in TOGETHER_REASONING_ONLY_MODELS

    is_non_standard = (
        is_nvidia
        or provider == "cerebras"
        or "cerebras.ai" in base_url
        or provider == "xai"
        or "api.x.ai" in base_url
        or is_together
        or "chutes.ai" in base_url
        or "deepseek.com" in base_url
        or is_zai
        or is_moonshot
        or provider == "opencode"
        or "opencode.ai" in base_url
        or is_cloudflare_workers_ai
        or is_cloudflare_ai_gateway
        or is_ant_ling
    )

    use_max_tokens = (
        "chutes.ai" in base_url
        or is_moonshot
        or is_cloudflare_ai_gateway
        or is_together
        or is_nvidia
        or is_ant_ling
        or is_zai
    )

    is_grok = provider == "xai" or "api.x.ai" in base_url
    is_deepseek = provider == "deepseek" or "deepseek.com" in base_url
    is_openrouter_developer_role_model = is_openrouter and model_id.startswith(("anthropic/", "openai/"))
    # pi: /^~?anthropic\//
    cache_control_format = (
        "anthropic" if provider == "openrouter" and model_id.startswith(("anthropic/", "~anthropic/")) else None
    )

    if is_deepseek:
        thinking_format = "deepseek"
    elif is_zai:
        thinking_format = "zai"
    elif is_together and not is_together_reasoning_only:
        thinking_format = "together"
    elif is_ant_ling:
        thinking_format = "ant-ling"
    elif is_openrouter:
        thinking_format = "openrouter"
    else:
        thinking_format = "openai"

    detected: dict[str, Any] = {
        "supportsStore": not is_non_standard,
        "supportsDeveloperRole": is_openrouter_developer_role_model or (not is_non_standard and not is_openrouter),
        "supportsReasoningEffort": not (
            is_grok or is_zai or is_moonshot or is_together or is_cloudflare_ai_gateway or is_nvidia or is_ant_ling
        ),
        "supportsUsageInStreaming": True,
        "supportsFinishReason": True,
        "maxTokensField": "max_tokens" if use_max_tokens else "max_completion_tokens",
        "requiresToolResultName": False,
        "requiresAssistantAfterToolResult": False,
        "requiresThinkingAsText": False,
        "requiresReasoningContentOnAssistantMessages": is_deepseek,
        "thinkingFormat": thinking_format,
        "openRouterRouting": {},
        "vercelGatewayRouting": {},
        "chatTemplateKwargs": {},
        "chatTemplateArgs": {},
        "zaiToolStream": False,
        "supportsStrictMode": not (is_moonshot or is_together or is_cloudflare_ai_gateway or is_nvidia),
        "supportsOpenAIGrammarTools": False,
    }
    if cache_control_format is not None:
        detected["cacheControlFormat"] = cache_control_format
    detected["sendSessionAffinityHeaders"] = False
    detected["supportsLongCacheRetention"] = not (
        is_together or is_cloudflare_workers_ai or is_cloudflare_ai_gateway or is_nvidia or is_ant_ling
    )
    return detected


def _is_plain_empty_object(value: Any) -> bool:
    return isinstance(value, dict) and not value


def openai_completions_compat_delta(compat: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key, value in compat.items():
        default_value = OPENAI_COMPLETIONS_DEFAULT_COMPAT.get(key)
        if _is_plain_empty_object(value) and _is_plain_empty_object(default_value):
            continue
        if value != default_value:
            delta[key] = value
    return delta


def apply_openai_completions_compat_metadata(model: dict[str, Any]) -> None:
    if model["api"] != "openai-completions":
        return
    detected = openai_completions_compat_delta(detect_openai_completions_compat(model))
    model["compat"] = {**detected, **model.get("compat", {})}
    if not model["compat"]:
        del model["compat"]


def supports_direct_reasoning_effort(model: dict[str, Any]) -> bool:
    if model["api"] == "anthropic-messages":
        return model.get("compat", {}).get("forceAdaptiveThinking") is True
    if model["api"] in ("openai-responses", "azure-openai-responses", "openai-codex-responses"):
        return True
    if model["api"] != "openai-completions":
        return False
    compat = {**detect_openai_completions_compat(model), **model.get("compat", {})}
    return compat.get("thinkingFormat") == "openai" and bool(compat.get("supportsReasoningEffort"))


def apply_models_dev_reasoning_option_metadata(model: dict[str, Any], reasoning_options: dict[str, list]) -> None:
    options = reasoning_options.get(f"{model['provider']}:{model['id']}")
    if not options or not supports_direct_reasoning_effort(model):
        return
    mapping = get_effort_thinking_level_map(options)
    if mapping:
        merge_thinking_level_map(model, mapping)


def apply_thinking_level_metadata(model: dict[str, Any]) -> None:
    model_id = model["id"]
    provider = model["provider"]
    if model["api"] in ("openai-responses", "azure-openai-responses") and model_id.startswith("gpt-5"):
        merge_thinking_level_map(model, {"off": None})
    if provider == "github-copilot" and model_id.startswith("gpt-5"):
        merge_thinking_level_map(model, {"minimal": "low"})
    if (
        model["api"] == "openai-responses"
        and provider == "openai"
        and model_id in OPENAI_RESPONSES_NONE_REASONING_MODELS
    ):
        merge_thinking_level_map(model, {"off": "none"})
    if provider == "xai" and model["api"] == "openai-responses" and model_id == XAI_RESPONSES_MODEL_ID:
        merge_thinking_level_map(model, dict(XAI_RESPONSES_EFFORT_LEVEL_MAP))
    if supports_openai_xhigh(model_id):
        merge_thinking_level_map(model, {"xhigh": "xhigh"})
    if supports_openai_max(model):
        merge_thinking_level_map(model, {"max": "max"})
    if provider == "openai" and model_id == "gpt-5.5":
        merge_thinking_level_map(model, {"minimal": None})
    if model_id.endswith("gpt-5.5-pro"):
        merge_thinking_level_map(model, {"off": None, "minimal": None, "low": None})
    # Anthropic adaptive-thinking effort support:
    # - "max" is available on all adaptive-thinking Claude models.
    # - "xhigh" is only available on Opus 4.7/4.8/5, Sonnet 5, and Fable 5.
    if any(marker in model_id for marker in ("opus-4-6", "opus-4.6", "sonnet-4-6", "sonnet-4.6")):
        merge_thinking_level_map(model, {"max": "max"})
    if any(
        marker in model_id
        for marker in ("opus-4-7", "opus-4.7", "opus-4-8", "opus-4.8", "opus-5", "opus.5", "sonnet-5", "sonnet.5")
    ):
        merge_thinking_level_map(model, {"xhigh": "xhigh", "max": "max"})
    if "fable-5" in model_id:
        merge_thinking_level_map(model, {"off": None, "xhigh": "xhigh", "max": "max"})
    if model["api"] == "anthropic-messages" and is_anthropic_adaptive_thinking_model(model_id):
        merge_compat(model, {"forceAdaptiveThinking": True})
    if model["api"] == "anthropic-messages" and is_anthropic_temperature_unsupported_model(model_id):
        merge_compat(model, {"supportsTemperature": False})
    if model["api"] == "openai-completions" and "deepseek-v4" in model_id:
        merge_thinking_level_map(
            model,
            {**DEEPSEEK_V4_THINKING_LEVEL_MAP, "xhigh": "xhigh", "max": None}
            if provider == "openrouter"
            else dict(DEEPSEEK_V4_THINKING_LEVEL_MAP),
        )
    if is_google_thinking_api(model) and _GEMINI_3_PRO_RE.search(model_id.lower()):
        merge_thinking_level_map(model, {"off": None, "minimal": None, "low": "LOW", "medium": None, "high": "HIGH"})
    if is_google_thinking_api(model) and (
        _GEMINI_3_FLASH_RE.search(model_id.lower())
        or model_id.lower() in ("gemini-flash-latest", "gemini-flash-lite-latest")
    ):
        merge_thinking_level_map(model, {"off": None})
    if is_google_thinking_api(model) and _GEMMA_4_RE.search(model_id.lower()):
        merge_thinking_level_map(
            model, {"off": None, "minimal": "MINIMAL", "low": None, "medium": None, "high": "HIGH"}
        )
    if provider == "groq" and model_id == "qwen/qwen3.6-27b":
        merge_thinking_level_map(model, {"minimal": None, "low": None, "medium": None, "high": "default"})
    if provider == "openai-codex" and supports_openai_xhigh(model_id):
        merge_thinking_level_map(model, {"minimal": "low"})
    if provider in ("moonshotai", "moonshotai-cn") and model_id in ("kimi-k2.7-code", "kimi-k2.7-code-highspeed"):
        # Kimi K2.7 Code is always-thinking. Official docs say
        # `thinking: { type: "disabled" }` is rejected, and callers can omit the
        # thinking parameter to use the enabled default.
        merge_thinking_level_map(model, {"off": None})
    if provider == "openrouter" and model_id.startswith("inception/mercury-2"):
        # Mercury 2 in instant mode (reasoning_effort: "none") disables tool calling.
        # Mark "off" unsupported so the openai-completions provider omits the reasoning
        # param instead of defaulting to {reasoning:{effort:"none"}}.
        merge_thinking_level_map(model, {"off": None})
    if provider == "openrouter" and model_id == "z-ai/glm-5.2":
        merge_thinking_level_map(model, {"xhigh": "xhigh"})
    if provider == "fireworks" and "glm-5p2" in model_id:
        merge_thinking_level_map(model, {"off": "none", "minimal": None, "low": "high", "medium": "high", "max": "max"})
    if provider == "opencode-go" and model_id == "glm-5.2":
        merge_thinking_level_map(model, dict(OPENCODE_GO_GLM52_THINKING_LEVEL_MAP))
    if provider == "opencode-go" and model_id == "kimi-k2.6":
        # OpenCode Go exposes Kimi K2.6 thinking as on/off, not distinct effort tiers.
        merge_thinking_level_map(model, {"minimal": None, "low": None, "medium": None})
    if provider == "opencode" and model_id == "grok-build-0.1":
        # OpenCode Zen Grok Build reasons by default but rejects explicit reasoningEffort.
        merge_thinking_level_map(model, {"off": None, "minimal": None, "low": None, "medium": None})
    if provider == "ant-ling" and model["reasoning"]:
        # Ring reasons by default. Only high/xhigh have documented explicit effort controls.
        merge_thinking_level_map(model, dict(ANT_LING_RING_THINKING_LEVEL_MAP))
    if provider == "github-copilot":
        override = GITHUB_COPILOT_THINKING_LEVEL_OVERRIDES.get(model_id)
        if override:
            merge_thinking_level_map(model, dict(override))


def apply_strict_tool_compat_metadata(model: dict[str, Any]) -> None:
    if model["provider"] == "openai" and model["api"] == "openai-responses":
        merge_compat(model, {"supportsStrictMode": True})
    elif model["provider"] == "anthropic" and model["api"] == "anthropic-messages":
        merge_compat(model, {"supportsStrictTools": True})


# Responses endpoints verified (OpenAI, ChatGPT Codex backend, GitHub Copilot,
# opencode zen) or documented (Azure OpenAI, Cloudflare AI Gateway) to pass OpenAI
# custom grammar tools through. OpenAI rejects `type: "custom"` tools for pre-GPT-5
# models (gpt-4.x, gpt-4o, o-series).
OPENAI_GRAMMAR_TOOL_PROVIDERS = {
    "openai",
    "openai-codex",
    "azure-openai-responses",
    "github-copilot",
    "opencode",
    "cloudflare-ai-gateway",
}
OPENAI_GRAMMAR_TOOL_APIS = {"openai-responses", "azure-openai-responses", "openai-codex-responses"}


def apply_openai_grammar_tool_compat_metadata(model: dict[str, Any]) -> None:
    if model["api"] not in OPENAI_GRAMMAR_TOOL_APIS or model["provider"] not in OPENAI_GRAMMAR_TOOL_PROVIDERS:
        return
    model_id = model["id"]
    if not model_id.startswith("gpt-"):
        return
    major = model_id.removeprefix("gpt-").split(".")[0].split("-")[0]
    if not major.isdigit() or int(major) < 5:
        return
    merge_compat(model, {"supportsOpenAIGrammarTools": True})


def apply_openai_tool_search_metadata(model: dict[str, Any]) -> None:
    is_openai_responses = model["provider"] == "openai" and model["api"] == "openai-responses"
    is_openai_codex = model["provider"] == "openai-codex" and model["api"] == "openai-codex-responses"
    if not (is_openai_responses or is_openai_codex) or model["id"] not in OPENAI_TOOL_SEARCH_MODEL_IDS:
        return
    merge_compat(model, {"supportsToolSearch": True})


# OpenAI charges prompt-cache writes starting with the GPT-5.6 family, and exactly
# those models accept `prompt_cache_options`; older models reject the parameter.
def apply_openai_explicit_prompt_cache_metadata(model: dict[str, Any]) -> None:
    if model["provider"] != "openai" or model["api"] != "openai-responses":
        return
    if not model["cost"]["cacheWrite"] > 0:
        return
    merge_compat(model, {"supportsExplicitPromptCacheMode": True})


def get_anthropic_messages_compat(provider: str, model_id: str) -> dict[str, Any] | None:
    compat: dict[str, Any] = {}
    if f"{provider}:{model_id}" in EAGER_TOOL_INPUT_STREAMING_UNSUPPORTED_ANTHROPIC_MODELS:
        compat["supportsEagerToolInputStreaming"] = False
    if provider == "xiaomi" or provider.startswith("xiaomi-token-plan-"):
        compat["allowEmptySignature"] = True
    return compat or None


def get_bedrock_base_url(model_id: str) -> str:
    return (
        "https://bedrock-runtime.eu-central-1.amazonaws.com"
        if model_id.startswith("eu.")
        else "https://bedrock-runtime.us-east-1.amazonaws.com"
    )


def normalize_nvidia_model_id(model_id: str) -> str:
    return model_id.lower().replace("_", ".")


def round_cost(value: float) -> float:
    return float(f"{value:.6f}")


# --- models.dev field readers -------------------------------------------------


def _cost(source: dict[str, Any]) -> dict[str, Any]:
    cost = source.get("cost") or {}
    return {
        "input": cost.get("input") or 0,
        "output": cost.get("output") or 0,
        "cacheRead": cost.get("cache_read") or 0,
        "cacheWrite": cost.get("cache_write") or 0,
    }


def get_models_dev_cost(cost: dict[str, Any] | None) -> dict[str, Any]:
    """pi's tier-aware cost reader; used for GitHub Copilot."""
    cost = cost or {}
    tiers = []
    for tier in cost.get("tiers") or []:
        context = tier.get("tier") or {}
        if context.get("type") != "context" or context.get("size") is None:
            continue
        tiers.append(
            {
                "inputTokensAbove": context["size"],
                "input": tier.get("input") or 0,
                "output": tier.get("output") or 0,
                "cacheRead": tier.get("cache_read") or 0,
                "cacheWrite": tier.get("cache_write") or 0,
            }
        )

    result = {
        "input": cost.get("input") or 0,
        "output": cost.get("output") or 0,
        "cacheRead": cost.get("cache_read") or 0,
        "cacheWrite": cost.get("cache_write") or 0,
    }
    if tiers:
        result["tiers"] = tiers
    return result


def _input(source: dict[str, Any]) -> list[str]:
    modalities = source.get("modalities") or {}
    return ["text", "image"] if "image" in (modalities.get("input") or []) else ["text"]


def _context(source: dict[str, Any], default: int = 4096) -> int:
    return (source.get("limit") or {}).get("context") or default


def _max_tokens(source: dict[str, Any], default: int = 4096) -> int:
    return (source.get("limit") or {}).get("output") or default


def _tool_capable(source: dict[str, Any]) -> bool:
    return source.get("tool_call") is True


def _models_of(catalog: dict[str, Any], key: str) -> dict[str, Any]:
    return (catalog.get(key) or {}).get("models") or {}


# --- live catalog fetches -----------------------------------------------------


async def _fetch_json(client: Client, url: str, label: str) -> Any:
    response = await client.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"{label} returned {response.status_code}")
    return await response.json()


async def fetch_nvidia_nim_model_ids(client: Client) -> dict[str, str]:
    print("Fetching models from NVIDIA NIM API...")
    data = await _fetch_json(client, f"{NVIDIA_BASE_URL}/models", "NVIDIA NIM API")
    model_ids: dict[str, str] = {}
    entries = data.get("data") or []
    for model in entries:
        model_ids[model["id"]] = model["id"]
        model_ids[normalize_nvidia_model_id(model["id"])] = model["id"]
    print(f"Fetched {len(entries)} model IDs from NVIDIA NIM")
    return model_ids


async def fetch_openrouter_models(client: Client) -> list[dict[str, Any]]:
    print("Fetching models from OpenRouter API...")
    data = await _fetch_json(client, "https://openrouter.ai/api/v1/models", "OpenRouter API")
    models: list[dict[str, Any]] = []
    for model in data["data"]:
        supported = model.get("supported_parameters") or []
        # Only include models that support tools
        if "tools" not in supported:
            continue

        architecture = model.get("architecture") or {}
        model_input = ["text"]
        if "image" in (architecture.get("modality") or ""):
            model_input.append("image")

        pricing = model.get("pricing") or {}
        top_provider = model.get("top_provider") or {}
        models.append(
            {
                "id": model["id"],
                "name": model["name"],
                "api": "openai-completions",
                "baseUrl": "https://openrouter.ai/api/v1",
                "provider": "openrouter",
                "reasoning": "reasoning" in supported,
                "input": model_input,
                # models.dev prices per token; pi's catalog is per million tokens.
                "cost": {
                    "input": round_cost(float(pricing.get("prompt") or "0") * 1_000_000),
                    "output": round_cost(float(pricing.get("completion") or "0") * 1_000_000),
                    "cacheRead": round_cost(float(pricing.get("input_cache_read") or "0") * 1_000_000),
                    "cacheWrite": round_cost(float(pricing.get("input_cache_write") or "0") * 1_000_000),
                },
                "contextWindow": top_provider.get("context_length") or model.get("context_length") or 4096,
                "maxTokens": top_provider.get("max_completion_tokens") or 4096,
            }
        )
    print(f"Fetched {len(models)} tool-capable models from OpenRouter")
    return models


def _to_number(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value if value is not None else "0")
    except ValueError:
        return 0.0


async def fetch_ai_gateway_models(client: Client) -> list[dict[str, Any]]:
    print("Fetching models from Vercel AI Gateway API...")
    data = await _fetch_json(client, f"{AI_GATEWAY_MODELS_URL}/models", "Vercel AI Gateway API")
    models: list[dict[str, Any]] = []
    items = data.get("data") if isinstance(data.get("data"), list) else []
    for model in items:
        tags = model.get("tags") if isinstance(model.get("tags"), list) else []
        # Only include models that support tools
        if "tool-use" not in tags:
            continue

        model_input = ["text"]
        if "vision" in tags:
            model_input.append("image")

        pricing = model.get("pricing") or {}
        models.append(
            {
                "id": model["id"],
                "name": model.get("name") or model["id"],
                "api": "anthropic-messages",
                "baseUrl": AI_GATEWAY_BASE_URL,
                "provider": "vercel-ai-gateway",
                "reasoning": "reasoning" in tags,
                "input": model_input,
                "cost": {
                    "input": round_cost(_to_number(pricing.get("input")) * 1_000_000),
                    "output": round_cost(_to_number(pricing.get("output")) * 1_000_000),
                    "cacheRead": round_cost(_to_number(pricing.get("input_cache_read")) * 1_000_000),
                    "cacheWrite": round_cost(_to_number(pricing.get("input_cache_write")) * 1_000_000),
                },
                "contextWindow": model.get("context_window") or 4096,
                "maxTokens": model.get("max_tokens") or 4096,
            }
        )
    print(f"Fetched {len(models)} tool-capable models from Vercel AI Gateway")
    return models


# --- models.dev catalog -------------------------------------------------------

type _Recorder = Callable[[str, str, dict[str, Any]], None]


def _load_direct_providers(catalog: dict[str, Any], record: _Recorder) -> list[dict[str, Any]]:
    """First-party endpoints: Bedrock, Anthropic, Google (+Vertex), OpenAI, Groq, Cerebras."""
    models: list[dict[str, Any]] = []

    # Amazon Bedrock
    for model_id, source in _models_of(catalog, "amazon-bedrock").items():
        if not _tool_capable(source):
            continue
        # ai21.jamba does not support tool use in streaming mode;
        # mistral-7b-instruct-v0 does not support system messages.
        if model_id.startswith(("ai21.jamba", "mistral.mistral-7b-instruct-v0")):
            continue
        if model_id in BEDROCK_INFERENCE_PROFILE_ONLY_MODEL_IDS:
            continue
        model: dict[str, Any] = {
            "id": model_id,
            "name": source.get("name") or model_id,
            "api": "bedrock-converse-stream",
            "provider": "amazon-bedrock",
            "baseUrl": get_bedrock_base_url(model_id),
            "reasoning": source.get("reasoning") is True,
            "input": _input(source),
            "cost": _cost(source),
            "contextWindow": _context(source),
            "maxTokens": _max_tokens(source),
        }
        if source.get("structured_output") is True:
            model["compat"] = {"supportsStrictMode": True}
        models.append(model)
        record("amazon-bedrock", model_id, source)

    # Anthropic
    for model_id, source in _models_of(catalog, "anthropic").items():
        if not _tool_capable(source):
            continue
        models.append(
            {
                "id": model_id,
                "name": source.get("name") or model_id,
                "api": "anthropic-messages",
                "provider": "anthropic",
                "baseUrl": "https://api.anthropic.com",
                "reasoning": source.get("reasoning") is True,
                "input": _input(source),
                "cost": _cost(source),
                "contextWindow": _context(source),
                "maxTokens": _max_tokens(source),
            }
        )
        record("anthropic", model_id, source)

    # Google (Generative AI)
    google_models = _models_of(catalog, "google")
    for model_id, entry in google_models.items():
        if not _tool_capable(entry):
            continue
        source = entry
        if model_id == "gemini-flash-latest":
            source = google_models.get("gemini-3.5-flash") or entry
        if model_id == "gemini-flash-lite-latest":
            source = google_models.get("gemini-3.1-flash-lite") or entry
        models.append(
            {
                "id": model_id,
                "name": entry.get("name") or model_id,
                "api": "google-generative-ai",
                "provider": "google",
                "baseUrl": "https://generativelanguage.googleapis.com/v1beta",
                "reasoning": source.get("reasoning") is True,
                "input": _input(source),
                "cost": _cost(source),
                "contextWindow": _context(source),
                "maxTokens": _max_tokens(source),
            }
        )
        record("google", model_id, source)

    # Google Vertex — Gemini only. The models.dev google-vertex catalog also lists
    # Claude/OpenAI MaaS models that do not use the Gemini streaming path.
    vertex_models = _models_of(catalog, "google-vertex")
    for model_id, entry in vertex_models.items():
        if not _tool_capable(entry):
            continue
        if not model_id.startswith("gemini-"):
            continue
        if model_id == "gemini-3.1-flash-lite-preview":
            continue
        source = entry
        if model_id == "gemini-flash-latest":
            source = vertex_models.get("gemini-3.5-flash") or entry
        if model_id == "gemini-flash-lite-latest":
            source = vertex_models.get("gemini-3.1-flash-lite") or entry

        # models.dev reports Vertex cache_read/cache_write for Gemini 2.5 Flash that
        # do not match the official Gemini pricing table. pi only accounts
        # cachedContentTokenCount as cacheRead.
        source_cost = source.get("cost") or {}
        cache_read = 0.03 if model_id == "gemini-2.5-flash" else source_cost.get("cache_read") or 0
        models.append(
            {
                "id": model_id,
                "name": entry.get("name") or model_id,
                "api": "google-vertex",
                "provider": "google-vertex",
                "baseUrl": VERTEX_BASE_URL,
                "reasoning": source.get("reasoning") is True,
                "input": _input(source),
                "cost": {
                    "input": source_cost.get("input") or 0,
                    "output": source_cost.get("output") or 0,
                    "cacheRead": cache_read,
                    "cacheWrite": 0,
                },
                "contextWindow": _context(source),
                "maxTokens": _max_tokens(source),
            }
        )
        record("google-vertex", model_id, source)

    # OpenAI
    for model_id, source in _models_of(catalog, "openai").items():
        if not _tool_capable(source):
            continue
        # models.dev lists this alias, but it is not accepted by OpenAI APIs.
        if model_id in MODELS_DEV_OPENAI_UNSUPPORTED_MODEL_IDS:
            continue
        models.append(
            {
                "id": model_id,
                "name": source.get("name") or model_id,
                "api": "openai-responses",
                "provider": "openai",
                "baseUrl": "https://api.openai.com/v1",
                "reasoning": source.get("reasoning") is True,
                "input": _input(source),
                "cost": _cost(source),
                "contextWindow": _context(source),
                "maxTokens": _max_tokens(source),
            }
        )
        record("openai", model_id, source)

    # Groq
    for model_id, source in _models_of(catalog, "groq").items():
        if not _tool_capable(source):
            continue
        models.append(
            {
                "id": model_id,
                "name": source.get("name") or model_id,
                "api": "openai-completions",
                "provider": "groq",
                "baseUrl": "https://api.groq.com/openai/v1",
                "reasoning": source.get("reasoning") is True,
                "input": _input(source),
                "cost": _cost(source),
                "contextWindow": _context(source),
                "maxTokens": _max_tokens(source),
            }
        )
        record("groq", model_id, source)

    # Cerebras
    for model_id, source in _models_of(catalog, "cerebras").items():
        if not _tool_capable(source):
            continue
        models.append(
            {
                "id": model_id,
                "name": source.get("name") or model_id,
                "api": "openai-completions",
                "provider": "cerebras",
                "baseUrl": "https://api.cerebras.ai/v1",
                "reasoning": source.get("reasoning") is True,
                "input": _input(source),
                "cost": _cost(source),
                "contextWindow": _context(source),
                "maxTokens": _max_tokens(source),
            }
        )
        record("cerebras", model_id, source)

    return models


def _load_gateway_providers(
    catalog: dict[str, Any], record: _Recorder, nvidia_nim_model_ids: dict[str, str]
) -> list[dict[str, Any]]:
    """Cloudflare, xAI, Z.AI, Mistral, Hugging Face, Fireworks, NVIDIA NIM, Together."""
    models: list[dict[str, Any]] = []

    # Cloudflare Workers AI
    for model_id, source in _models_of(catalog, "cloudflare-workers-ai").items():
        if not _tool_capable(source):
            continue
        models.append(
            {
                "id": model_id,
                "name": source.get("name") or model_id,
                "api": "openai-completions",
                "provider": "cloudflare-workers-ai",
                "baseUrl": CLOUDFLARE_WORKERS_AI_BASE_URL,
                "reasoning": source.get("reasoning") is True,
                "input": _input(source),
                "cost": _cost(source),
                "contextWindow": _context(source),
                "maxTokens": _max_tokens(source),
                "compat": {"sendSessionAffinityHeaders": True},
            }
        )
        record("cloudflare-workers-ai", model_id, source)

    # Cloudflare AI Gateway
    for prefixed_id, source in _models_of(catalog, "cloudflare-ai-gateway").items():
        if not _tool_capable(source):
            continue
        slash_index = prefixed_id.find("/")
        if slash_index == -1:
            continue
        upstream = prefixed_id[:slash_index]
        native_id = prefixed_id[slash_index + 1 :]

        if upstream == "openai":
            api, base_url, model_id = "openai-responses", CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL, native_id
        elif upstream == "anthropic":
            api, base_url, model_id = "anthropic-messages", CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL, native_id
        elif upstream == "workers-ai":
            api, base_url, model_id = "openai-completions", CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL, prefixed_id
        else:
            continue

        model = {
            "id": model_id,
            "name": source.get("name") or model_id,
            "api": api,
            "provider": "cloudflare-ai-gateway",
            "baseUrl": base_url,
            "reasoning": source.get("reasoning") is True,
            "input": _input(source),
            "cost": _cost(source),
            "contextWindow": _context(source),
            "maxTokens": _max_tokens(source),
        }
        # Gateway passthroughs forward session affinity headers to upstreams that use
        # them for cache/routing affinity.
        if upstream in ("anthropic", "workers-ai"):
            model["compat"] = {"sendSessionAffinityHeaders": True}
        models.append(model)
        record("cloudflare-ai-gateway", model_id, source)

    # xAI
    for model_id, source in _models_of(catalog, "xai").items():
        if not _tool_capable(source):
            continue
        use_responses_api = model_id == XAI_RESPONSES_MODEL_ID
        model = {
            "id": model_id,
            "name": source.get("name") or model_id,
            "api": "openai-responses" if use_responses_api else "openai-completions",
            "provider": "xai",
            "baseUrl": "https://api.x.ai/v1",
        }
        if use_responses_api:
            model["compat"] = dict(XAI_RESPONSES_COMPAT)
        model |= {
            "reasoning": source.get("reasoning") is True,
            "input": _input(source),
            "cost": _cost(source),
            "contextWindow": _context(source),
            "maxTokens": _max_tokens(source),
        }
        models.append(model)
        record("xai", model_id, source)

    # Z.AI coding plan (two regional variants share one models.dev catalog)
    zai_models = _models_of(catalog, "zai-coding-plan")
    for provider, base_url in (
        ("zai", "https://api.z.ai/api/coding/paas/v4"),
        ("zai-coding-cn", "https://open.bigmodel.cn/api/coding/paas/v4"),
    ):
        for model_id, source in zai_models.items():
            if not _tool_capable(source):
                continue
            is_glm52 = model_id == "glm-5.2"
            model = {
                "id": model_id,
                "name": source.get("name") or model_id,
                "api": "openai-completions",
                "provider": provider,
                "baseUrl": base_url,
                "reasoning": source.get("reasoning") is True,
            }
            if is_glm52:
                model["thinkingLevelMap"] = dict(ZAI_GLM52_THINKING_LEVEL_MAP)
            compat: dict[str, Any] = {"supportsDeveloperRole": False, "thinkingFormat": "zai"}
            if is_glm52:
                compat["supportsReasoningEffort"] = True
            if model_id not in ZAI_TOOL_STREAM_UNSUPPORTED_MODELS:
                compat["zaiToolStream"] = True
            model |= {
                "input": _input(source),
                "cost": _cost(source),
                "compat": compat,
                "contextWindow": _context(source),
                "maxTokens": _max_tokens(source),
            }
            models.append(model)
            record(provider, model_id, source)

    # Mistral
    for model_id, source in _models_of(catalog, "mistral").items():
        if not _tool_capable(source):
            continue
        cost = source.get("cost") or {}
        # pi uses `??` here: an explicit 0 cache_read survives, only absent falls back.
        cache_read = cost.get("cache_read")
        if cache_read is None:
            cache_read = round_cost(cost["input"] * 0.1) if cost.get("input") else 0
        models.append(
            {
                "id": model_id,
                "name": source.get("name") or model_id,
                "api": "mistral-conversations",
                "provider": "mistral",
                "baseUrl": "https://api.mistral.ai",
                "reasoning": source.get("reasoning") is True,
                "input": _input(source),
                "cost": {
                    "input": cost.get("input") or 0,
                    "output": cost.get("output") or 0,
                    "cacheRead": cache_read,
                    "cacheWrite": cost.get("cache_write") or 0,
                },
                "contextWindow": _context(source),
                "maxTokens": _max_tokens(source),
            }
        )
        record("mistral", model_id, source)

    # Hugging Face
    for model_id, source in _models_of(catalog, "huggingface").items():
        if not _tool_capable(source):
            continue
        models.append(
            {
                "id": model_id,
                "name": source.get("name") or model_id,
                "api": "openai-completions",
                "provider": "huggingface",
                "baseUrl": "https://router.huggingface.co/v1",
                "reasoning": source.get("reasoning") is True,
                "input": _input(source),
                "cost": _cost(source),
                "compat": {"supportsDeveloperRole": False},
                "contextWindow": _context(source),
                "maxTokens": _max_tokens(source),
            }
        )
        record("huggingface", model_id, source)

    # Fireworks
    models.extend(_process_fireworks_models(_models_of(catalog, "fireworks-ai"), record))

    # NVIDIA NIM
    for model_id, source in _models_of(catalog, "nvidia").items():
        if not _tool_capable(source):
            continue
        modalities = source.get("modalities") or {}
        if "text" not in (modalities.get("input") or []):
            continue
        if "text" not in (modalities.get("output") or []):
            continue

        live_model_id = nvidia_nim_model_ids.get(model_id) or nvidia_nim_model_ids.get(
            normalize_nvidia_model_id(model_id)
        )
        if not live_model_id:
            continue
        if live_model_id in NVIDIA_NIM_UNSUPPORTED_MODELS:
            continue

        models.append(
            {
                "id": live_model_id,
                "name": source.get("name") or live_model_id,
                "api": "openai-completions",
                "provider": "nvidia",
                "baseUrl": NVIDIA_BASE_URL,
                "headers": dict(NVIDIA_HEADERS),
                "reasoning": source.get("reasoning") is True,
                "input": _input(source),
                "cost": _cost(source),
                "compat": dict(NVIDIA_OPENAI_COMPAT),
                "contextWindow": _context(source),
                "maxTokens": _max_tokens(source),
            }
        )
        record("nvidia", live_model_id, source)

    # Together AI
    together_models = (
        _models_of(catalog, "together") or _models_of(catalog, "togetherai") or _models_of(catalog, "together-ai")
    )
    for model_id, source in together_models.items():
        if not _tool_capable(source):
            continue
        if source.get("status") == "deprecated":
            continue

        reasoning = source.get("reasoning") is True
        thinking_level_map = get_together_thinking_level_map(model_id, reasoning)
        model = {
            "id": model_id,
            "name": source.get("name") or model_id,
            "api": "openai-completions",
            "provider": "together",
            "baseUrl": TOGETHER_BASE_URL,
            "reasoning": reasoning,
        }
        if thinking_level_map:
            model["thinkingLevelMap"] = thinking_level_map
        model |= {
            "input": _input(source),
            "cost": _cost(source),
            "compat": dict(get_together_compat(model_id, reasoning)),
            "contextWindow": _context(source),
            "maxTokens": _max_tokens(source),
        }
        models.append(model)
        record("together", model_id, source)

    # Baseten
    models.extend(_process_baseten_models(_models_of(catalog, "baseten"), record))

    return models


def _process_fireworks_models(fireworks_models: dict[str, Any], record: _Recorder) -> list[dict[str, Any]]:
    anthropic_compat = {
        "sendSessionAffinityHeaders": True,
        "supportsEagerToolInputStreaming": False,
        "supportsCacheControlOnTools": False,
        "supportsLongCacheRetention": False,
    }
    openai_compat = {
        "supportsStore": False,
        "supportsDeveloperRole": False,
        "sendSessionAffinityHeaders": True,
        "supportsLongCacheRetention": False,
    }
    kimi_k3_compat = {
        **openai_compat,
        "requiresReasoningContentOnAssistantMessages": True,
        "thinkingFormat": "openai",
        "deferredToolsMode": "kimi",
    }
    models: list[dict[str, Any]] = []

    for model_id, source in fireworks_models.items():
        if not _tool_capable(source):
            continue

        common = {
            "id": model_id,
            "name": source.get("name") or model_id,
            "provider": "fireworks",
            "reasoning": source.get("reasoning") is True,
            "input": _input(source),
            "cost": _cost(source),
            "contextWindow": _context(source),
            "maxTokens": _max_tokens(source),
        }

        if "glm-5p2" in model_id:
            models.append(
                {
                    **common,
                    "api": "openai-completions",
                    "baseUrl": "https://api.fireworks.ai/inference/v1",
                    "compat": dict(openai_compat),
                }
            )
        elif "kimi-k3" in model_id:
            models.append(
                {
                    **common,
                    "api": "openai-completions",
                    "baseUrl": "https://api.fireworks.ai/inference/v1",
                    "compat": dict(kimi_k3_compat),
                }
            )
        else:
            models.append(
                {
                    **common,
                    "api": "anthropic-messages",
                    # Fireworks Anthropic-compatible API - SDK appends /v1/messages.
                    "baseUrl": "https://api.fireworks.ai/inference",
                    # Fireworks prompt caching uses automatic prefix matching + session
                    # affinity: x-session-affinity routes requests to the same replica for
                    # cache hits, and cache_control on tools / eager_input_streaming are
                    # unsupported. https://docs.fireworks.ai/tools-sdks/anthropic-compatibility
                    "compat": dict(anthropic_compat),
                }
            )
        record("fireworks", model_id, source)

    return models


def _process_baseten_models(baseten_models: dict[str, Any], record: _Recorder) -> list[dict[str, Any]]:
    base_url = "https://inference.baseten.co/v1"
    base_compat: dict[str, Any] = {
        "supportsStore": False,
        "supportsDeveloperRole": False,
        "supportsReasoningEffort": False,
        "supportsUsageInStreaming": True,
        "maxTokensField": "max_tokens",
        "supportsStrictMode": True,
        "supportsLongCacheRetention": False,
    }
    reasoning_effort_compat = {**base_compat, "supportsReasoningEffort": True, "thinkingFormat": "openai"}
    toggle_reasoning_compat = {
        **base_compat,
        "thinkingFormat": "baseten",
        "chatTemplateArgs": {"enable_thinking": {"$var": "thinking.enabled"}},
    }
    toggle_reasoning_effort_compat = {
        **reasoning_effort_compat,
        "thinkingFormat": "baseten",
        "chatTemplateArgs": {"enable_thinking": {"$var": "thinking.enabled"}},
    }
    toggle_thinking_level_map = {
        "off": "off",
        "minimal": None,
        "low": None,
        "medium": None,
        "high": "high",
        "xhigh": None,
        "max": None,
    }
    glm52_thinking_level_map = {
        "off": "none",
        "minimal": None,
        "low": None,
        "medium": None,
        "high": "high",
        "xhigh": None,
        "max": "max",
    }
    models: list[dict[str, Any]] = []

    for model_id, source in baseten_models.items():
        if source.get("status") == "deprecated":
            continue

        reasoning = source.get("reasoning") is True
        reasoning_options = source.get("reasoning_options") or []
        is_glm52 = model_id in ("zai-org/GLM-5.2", "zai-org/GLM-5.2-Fast")
        supports_toggle = any(option.get("type") == "toggle" for option in reasoning_options) or is_glm52
        supports_effort = any(option.get("type") == "effort" for option in reasoning_options) or is_glm52
        if supports_toggle and supports_effort:
            compat = toggle_reasoning_effort_compat
        elif supports_toggle:
            compat = toggle_reasoning_compat
        elif supports_effort:
            compat = reasoning_effort_compat
        else:
            compat = base_compat
        if is_glm52:
            thinking_level_map = glm52_thinking_level_map
        elif supports_toggle:
            thinking_level_map = toggle_thinking_level_map
        else:
            thinking_level_map = get_effort_thinking_level_map(reasoning_options)

        model = {
            "id": model_id,
            "name": source.get("name") or model_id,
            "api": "openai-completions",
            "provider": "baseten",
            "baseUrl": base_url,
            "reasoning": reasoning,
        }
        if thinking_level_map:
            model["thinkingLevelMap"] = dict(thinking_level_map)
        model |= {
            "input": _input(source),
            "cost": _cost(source),
            "compat": dict(compat),
            "contextWindow": _context(source),
            "maxTokens": _max_tokens(source),
        }
        models.append(model)
        record("baseten", model_id, source)

    return models


def _load_aggregator_providers(catalog: dict[str, Any], record: _Recorder) -> list[dict[str, Any]]:
    """Catalog aggregators: OpenCode Zen (+Go) and GitHub Copilot."""
    models: list[dict[str, Any]] = []

    # OpenCode (Zen and Go). API mapping is based on the models.dev provider.npm field:
    #   @ai-sdk/openai → openai-responses, @ai-sdk/anthropic → anthropic-messages,
    #   @ai-sdk/google → google-generative-ai, otherwise openai-completions.
    for key, provider, base_path in (
        ("opencode", "opencode", "https://opencode.ai/zen"),
        ("opencode-go", "opencode-go", "https://opencode.ai/zen/go"),
    ):
        for model_id, source in _models_of(catalog, key).items():
            if not _tool_capable(source):
                continue
            if source.get("status") == "deprecated":
                continue

            npm = (source.get("provider") or {}).get("npm")
            compat: dict[str, Any] | None = None
            if npm == "@ai-sdk/openai":
                api = "openai-responses"
                base_url = f"{base_path}/v1"
                compat = {"sessionAffinityFormat": "openai-nosession"}
            elif npm == "@ai-sdk/anthropic":
                api = "anthropic-messages"
                # Anthropic SDK appends /v1/messages to baseURL
                base_url = base_path
            elif npm == "@ai-sdk/google":
                api = "google-generative-ai"
                base_url = f"{base_path}/v1"
            elif npm == "@ai-sdk/alibaba":
                api = "openai-completions"
                base_url = f"{base_path}/v1"
                compat = {"cacheControlFormat": "anthropic"}
            else:
                # null, undefined, or @ai-sdk/openai-compatible
                api = "openai-completions"
                base_url = f"{base_path}/v1"

            if provider == "opencode" and model_id == "grok-build-0.1":
                compat = {**(compat or {}), "supportsReasoningEffort": False}

            if model_id == "kimi-k2.6":
                # OpenCode Kimi K2.6 accepts Anthropic-style thinking objects and
                # rejects string thinking values or combined reasoning_effort.
                compat = {**(compat or {}), "thinkingFormat": "deepseek", "supportsReasoningEffort": False}

            # Fix known mismatches between models.dev npm data and actual OpenCode Go
            # endpoint behaviour: these are reported as @ai-sdk/anthropic but the Go
            # endpoints either reject Anthropic SDK auth (MiniMax M2.7) or are served
            # through the OpenAI-compatible /v1/chat/completions path (Qwen 3.5/3.6).
            if provider == "opencode-go":
                if model_id == "minimax-m2.7":
                    api = "openai-completions"
                    base_url = f"{base_path}/v1"
                if model_id in ("qwen3.5-plus", "qwen3.6-plus"):
                    api = "openai-completions"
                    base_url = f"{base_path}/v1"
                    # Qwen/DashScope uses enable_thinking at the top level.
                    compat = {**(compat or {}), "thinkingFormat": "qwen"}

            if api == "openai-completions":
                compat = {**(compat or {}), "maxTokensField": "max_tokens"}
                if f"{provider}:{model_id}" in OPENCODE_OPENAI_COMPLETIONS_LONG_CACHE_RETENTION_UNSUPPORTED_MODELS:
                    compat = {**compat, "supportsLongCacheRetention": False}

            model = {
                "id": model_id,
                "name": source.get("name") or model_id,
                "api": api,
                "provider": provider,
                "baseUrl": base_url,
                "reasoning": source.get("reasoning") is True,
                "input": _input(source),
                "cost": _cost(source),
            }
            if compat:
                model["compat"] = compat
            model |= {"contextWindow": _context(source), "maxTokens": _max_tokens(source)}
            models.append(model)
            record(provider, model_id, source)

    # GitHub Copilot
    for model_id, source in _models_of(catalog, "github-copilot").items():
        if not _tool_capable(source):
            continue
        if source.get("status") == "deprecated":
            continue

        # Claude 4.x and 5.x models route to the Anthropic Messages API.
        is_copilot_claude = _COPILOT_CLAUDE_RE.match(model_id) is not None
        # Grok 4.5, gpt-5, oswe and MAI-Code models are only served through /responses.
        needs_responses_api = model_id == "grok-4.5" or model_id.startswith(("gpt-5", "oswe", "mai-"))
        api = (
            "anthropic-messages"
            if is_copilot_claude
            else ("openai-responses" if needs_responses_api else "openai-completions")
        )

        model = {
            "id": model_id,
            "name": source.get("name") or model_id,
            "api": api,
            "provider": "github-copilot",
            "baseUrl": "https://api.individual.githubcopilot.com",
            "reasoning": source.get("reasoning") is True,
            "input": _input(source),
            "cost": get_models_dev_cost(source.get("cost")),
            "contextWindow": _context(source, 128000),
            "maxTokens": _max_tokens(source, 8192),
            "headers": dict(COPILOT_STATIC_HEADERS),
        }
        anthropic_compat = get_anthropic_messages_compat("github-copilot", model_id) if is_copilot_claude else None
        if anthropic_compat:
            model["compat"] = anthropic_compat
        # compat only applies to openai-completions
        if api == "openai-completions":
            model["compat"] = {
                "supportsStore": False,
                "supportsDeveloperRole": False,
                "supportsReasoningEffort": False,
            }
        models.append(model)
        record("github-copilot", model_id, source)

    return models


def _load_regional_providers(catalog: dict[str, Any], record: _Recorder) -> list[dict[str, Any]]:
    """Providers shipped as regional variants: MiniMax, Kimi, Moonshot, Xiaomi, Qwen."""
    models: list[dict[str, Any]] = []

    # MiniMax
    for key, provider, base_url in (
        ("minimax", "minimax", "https://api.minimax.io/anthropic"),
        ("minimax-cn", "minimax-cn", "https://api.minimaxi.com/anthropic"),
    ):
        for model_id, source in _models_of(catalog, key).items():
            if not _tool_capable(source):
                continue
            models.append(
                {
                    "id": model_id,
                    "name": source.get("name") or model_id,
                    "api": "anthropic-messages",
                    "provider": provider,
                    # MiniMax's Anthropic-compatible API - SDK appends /v1/messages
                    "baseUrl": base_url,
                    "reasoning": source.get("reasoning") is True,
                    "input": _input(source),
                    "cost": _cost(source),
                    "contextWindow": _context(source),
                    "maxTokens": _max_tokens(source),
                }
            )
            record(provider, model_id, source)

    # Kimi For Coding
    kimi_models = _models_of(catalog, "kimi-for-coding")
    has_canonical_kimi_model = "kimi-for-coding" in kimi_models
    kimi_aliases = {"k2p5", "k2p6", "k2p7"}
    for model_id, source in kimi_models.items():
        if not _tool_capable(source):
            continue
        # models.dev may expose versioned aliases (k2p5/k2p6/k2p7). Normalize them to
        # the canonical id and drop duplicates when the canonical entry exists.
        if model_id in kimi_aliases and has_canonical_kimi_model:
            continue

        is_alias = model_id in kimi_aliases
        normalized_id = "kimi-for-coding" if is_alias else model_id
        normalized_name = "Kimi For Coding" if is_alias else (source.get("name") or normalized_id)
        is_kimi_k3 = normalized_id == "k3"
        allow_empty_signature = is_kimi_k3 or normalized_id == "kimi-for-coding"
        implied_cost = KIMI_CODING_IMPLIED_COSTS.get(normalized_id) or {}
        cost = source.get("cost") or {}

        compat = {}
        if allow_empty_signature:
            compat["allowEmptySignature"] = True
        compat["forceAdaptiveThinking"] = True

        models.append(
            {
                "id": normalized_id,
                "name": normalized_name,
                "api": "anthropic-messages",
                "provider": "kimi-coding",
                # Kimi For Coding's Anthropic-compatible API - SDK appends /v1/messages
                "baseUrl": "https://api.kimi.com/coding",
                "headers": dict(KIMI_STATIC_HEADERS),
                "compat": compat,
                "reasoning": is_kimi_k3 or source.get("reasoning") is True,
                "input": _input(source),
                "cost": {
                    "input": cost.get("input") or implied_cost.get("input") or 0,
                    "output": cost.get("output") or implied_cost.get("output") or 0,
                    "cacheRead": cost.get("cache_read") or implied_cost.get("cacheRead") or 0,
                    "cacheWrite": cost.get("cache_write") or implied_cost.get("cacheWrite") or 0,
                },
                "contextWindow": _context(source),
                "maxTokens": _max_tokens(source),
            }
        )
        record("kimi-coding", normalized_id, source)

    # Moonshot AI
    moonshot_compat: dict[str, Any] = {
        "supportsStore": False,
        "supportsDeveloperRole": False,
        "supportsReasoningEffort": False,
        "maxTokensField": "max_tokens",
        "supportsStrictMode": False,
        "thinkingFormat": "deepseek",
    }
    for key, provider, base_url in (
        ("moonshotai", "moonshotai", "https://api.moonshot.ai/v1"),
        ("moonshotai-cn", "moonshotai-cn", "https://api.moonshot.cn/v1"),
    ):
        for model_id, source in _models_of(catalog, key).items():
            if not _tool_capable(source):
                continue
            is_kimi_k3 = model_id == "kimi-k3"
            compat = dict(moonshot_compat)
            if is_kimi_k3:
                compat["requiresReasoningContentOnAssistantMessages"] = True
                compat["deferredToolsMode"] = "kimi"
                compat["thinkingFormat"] = "openai"
                compat["supportsReasoningEffort"] = True
            cost = source.get("cost") or {}
            models.append(
                {
                    "id": model_id,
                    "name": source.get("name") or model_id,
                    "api": "openai-completions",
                    "provider": provider,
                    "baseUrl": base_url,
                    "reasoning": is_kimi_k3 or source.get("reasoning") is True,
                    "input": _input(source),
                    "cost": {
                        "input": cost.get("input") or (KIMI_K3_COST["input"] if is_kimi_k3 else 0),
                        "output": cost.get("output") or (KIMI_K3_COST["output"] if is_kimi_k3 else 0),
                        "cacheRead": cost.get("cache_read") or (KIMI_K3_COST["cacheRead"] if is_kimi_k3 else 0),
                        "cacheWrite": cost.get("cache_write") or (KIMI_K3_COST["cacheWrite"] if is_kimi_k3 else 0),
                    },
                    "contextWindow": _context(source),
                    "maxTokens": _max_tokens(source),
                    "compat": compat,
                }
            )
            record(provider, model_id, source)

    # Xiaomi MiMo. Built-in `xiaomi` targets the API billing endpoint (single stable
    # URL, keys from platform.xiaomimimo.com). The three `xiaomi-token-plan-*`
    # providers cover prepaid Token Plan endpoints in cn / ams / sgp.
    xiaomi_compat: dict[str, Any] = {
        "requiresReasoningContentOnAssistantMessages": True,
        "thinkingFormat": "deepseek",
    }
    for source_key, provider, base_url in (
        ("xiaomi", "xiaomi", "https://api.xiaomimimo.com/v1"),
        ("xiaomi-token-plan-cn", "xiaomi-token-plan-cn", "https://token-plan-cn.xiaomimimo.com/v1"),
        ("xiaomi-token-plan-ams", "xiaomi-token-plan-ams", "https://token-plan-ams.xiaomimimo.com/v1"),
        ("xiaomi-token-plan-sgp", "xiaomi-token-plan-sgp", "https://token-plan-sgp.xiaomimimo.com/v1"),
    ):
        for model_id, source in _models_of(catalog, source_key).items():
            if not _tool_capable(source):
                continue
            models.append(
                {
                    "id": model_id,
                    "name": source.get("name") or model_id,
                    "api": "openai-completions",
                    "provider": provider,
                    "baseUrl": base_url,
                    "compat": dict(xiaomi_compat),
                    "reasoning": source.get("reasoning") is True,
                    "input": _input(source),
                    "cost": _cost(source),
                    "contextWindow": _context(source),
                    "maxTokens": _max_tokens(source),
                }
            )
            record(provider, model_id, source)

    models.extend(_process_qwen_token_plan_models(catalog, record))

    return models


def _process_qwen_token_plan_models(catalog: dict[str, Any], record: _Recorder) -> list[dict[str, Any]]:
    """Alibaba Cloud Model Studio Token Plan. International and China use separate
    endpoints and API keys (sk-sp- prefix). The Individual provider reuses the
    international source and endpoint with a narrower catalog. models.dev keys are
    "alibaba-token-plan[-cn]"; pi exposes them as "qwen-token-plan[-cn]" plus the
    Individual catalog view.
    """
    qwen_token_plan_compat: dict[str, Any] = {
        "thinkingFormat": "qwen",
        "supportsDeveloperRole": False,
        "supportsStore": False,
        "supportsReasoningEffort": True,
    }
    models: list[dict[str, Any]] = []
    for source_key, provider, base_url, allowed_model_ids in (
        (
            "alibaba-token-plan",
            "qwen-token-plan",
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
            None,
        ),
        (
            "alibaba-token-plan",
            "qwen-token-plan-individual",
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
            QWEN_TOKEN_PLAN_INDIVIDUAL_MODEL_IDS,
        ),
        (
            "alibaba-token-plan-cn",
            "qwen-token-plan-cn",
            "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            None,
        ),
    ):
        emitted_model_ids: set[str] | None = set() if allowed_model_ids is not None else None
        for model_id, source in _models_of(catalog, source_key).items():
            if not _tool_capable(source):
                continue
            if model_id in QWEN_TOKEN_PLAN_EXCLUDED_MODEL_IDS:
                continue
            if allowed_model_ids is not None and model_id not in allowed_model_ids:
                continue
            supports_reasoning_effort = model_id not in QWEN_TOKEN_PLAN_REASONING_EFFORT_UNSUPPORTED_MODEL_IDS
            entry: dict[str, Any] = {
                "id": model_id,
                "name": source.get("name") or model_id,
                "api": "openai-completions",
                "provider": provider,
                "baseUrl": base_url,
                "compat": dict(qwen_token_plan_compat)
                if supports_reasoning_effort
                else {**qwen_token_plan_compat, "supportsReasoningEffort": False},
                "reasoning": source.get("reasoning") is True,
                "input": _input(source),
                "cost": _cost(source),
                "contextWindow": _context(source),
                "maxTokens": _max_tokens(source),
            }
            if supports_reasoning_effort:
                entry["thinkingLevelMap"] = dict(
                    QWEN_TOKEN_PLAN_QWEN38_THINKING_LEVEL_MAP
                    if model_id == "qwen3.8-max"
                    else QWEN_TOKEN_PLAN_HIGH_MAX_THINKING_LEVEL_MAP
                )
            models.append(entry)
            if emitted_model_ids is not None:
                emitted_model_ids.add(model_id)
            record(provider, model_id, source)

        # pi gates this on `--strict`; this generator is always-strict (see the
        # module docstring's deviation note).
        if allowed_model_ids is not None and emitted_model_ids is not None:
            assert_exact_model_ids(provider, allowed_model_ids, emitted_model_ids)

    return models


def load_models_dev_data(
    catalog: dict[str, Any],
    reasoning_options: dict[str, list],
    nvidia_nim_model_ids: dict[str, str],
) -> list[dict[str, Any]]:
    """pi's loadModelsDevData, split into provider groups (one flat function trips mccabe)."""

    def record(provider: str, model_id: str, source: dict[str, Any]) -> None:
        if source.get("reasoning_options") is not None:
            reasoning_options[f"{provider}:{model_id}"] = source["reasoning_options"]

    models = [
        *_load_direct_providers(catalog, record),
        *_load_gateway_providers(catalog, record, nvidia_nim_model_ids),
        *_load_aggregator_providers(catalog, record),
        *_load_regional_providers(catalog, record),
    ]
    print(f"Loaded {len(models)} tool-capable models from models.dev")
    return models


# --- explicit catalogs (not fetched) ------------------------------------------

MISSING_OPENAI_MODELS: list[dict[str, Any]] = [
    {
        "id": "gpt-5.6-sol",
        "name": "GPT-5.6 Sol",
        "api": "openai-responses",
        "baseUrl": "https://api.openai.com/v1",
        "provider": "openai",
        "reasoning": True,
        "input": ["text", "image"],
        "cost": with_openai_long_context_pricing({"input": 5, "output": 30, "cacheRead": 0.5, "cacheWrite": 6.25}),
        "contextWindow": OPENAI_LONG_CONTEXT_INPUT_THRESHOLD,
        "maxTokens": 128000,
    },
    {
        "id": "gpt-5.6-terra",
        "name": "GPT-5.6 Terra",
        "api": "openai-responses",
        "baseUrl": "https://api.openai.com/v1",
        "provider": "openai",
        "reasoning": True,
        "input": ["text", "image"],
        "cost": with_openai_long_context_pricing(OPENAI_GPT_56_STANDARD_COSTS["gpt-5.6-terra"]),
        "contextWindow": OPENAI_LONG_CONTEXT_INPUT_THRESHOLD,
        "maxTokens": 128000,
    },
    {
        "id": "gpt-5.6-luna",
        "name": "GPT-5.6 Luna",
        "api": "openai-responses",
        "baseUrl": "https://api.openai.com/v1",
        "provider": "openai",
        "reasoning": True,
        "input": ["text", "image"],
        "cost": with_openai_long_context_pricing(OPENAI_GPT_56_STANDARD_COSTS["gpt-5.6-luna"]),
        "contextWindow": OPENAI_LONG_CONTEXT_INPUT_THRESHOLD,
        "maxTokens": 128000,
    },
    {
        "id": "gpt-5-chat-latest",
        "name": "GPT-5 Chat Latest",
        "api": "openai-responses",
        "baseUrl": "https://api.openai.com/v1",
        "provider": "openai",
        "reasoning": False,
        "input": ["text", "image"],
        "cost": {"input": 1.25, "output": 10, "cacheRead": 0.125, "cacheWrite": 0},
        "contextWindow": 128000,
        "maxTokens": 16384,
    },
]

DEEPSEEK_COMPAT: dict[str, Any] = {
    "requiresReasoningContentOnAssistantMessages": True,
    "thinkingFormat": "deepseek",
}

DEEPSEEK_V4_MODELS: list[dict[str, Any]] = [
    {
        "id": "deepseek-v4-flash",
        "name": "DeepSeek V4 Flash",
        "api": "openai-completions",
        "baseUrl": "https://api.deepseek.com",
        "provider": "deepseek",
        "reasoning": True,
        "input": ["text"],
        "cost": {"input": 0.14, "output": 0.28, "cacheRead": 0.0028, "cacheWrite": 0},
        "contextWindow": 1000000,
        "maxTokens": 384000,
        "compat": DEEPSEEK_COMPAT,
    },
    {
        "id": "deepseek-v4-pro",
        "name": "DeepSeek V4 Pro",
        "api": "openai-completions",
        "baseUrl": "https://api.deepseek.com",
        "provider": "deepseek",
        "reasoning": True,
        "input": ["text"],
        "cost": {"input": 0.435, "output": 0.87, "cacheRead": 0.003625, "cacheWrite": 0},
        "contextWindow": 1000000,
        "maxTokens": 384000,
        "compat": DEEPSEEK_COMPAT,
    },
]

ANT_LING_COMPAT: dict[str, Any] = {
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "supportsReasoningEffort": False,
    "maxTokensField": "max_tokens",
    "supportsLongCacheRetention": False,
}

ANT_LING_MODELS: list[dict[str, Any]] = [
    {
        "id": "Ling-2.6-flash",
        "name": "Ling 2.6 Flash",
        "api": "openai-completions",
        "baseUrl": "https://api.ant-ling.com/v1",
        "provider": "ant-ling",
        "reasoning": False,
        "input": ["text"],
        "cost": {"input": 0.01, "output": 0.02, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 262144,
        "maxTokens": 65536,
        "compat": ANT_LING_COMPAT,
    },
    {
        "id": "Ling-2.6-1T",
        "name": "Ling 2.6 1T",
        "api": "openai-completions",
        "baseUrl": "https://api.ant-ling.com/v1",
        "provider": "ant-ling",
        "reasoning": False,
        "input": ["text"],
        "cost": {"input": 0.06, "output": 0.25, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 262144,
        "maxTokens": 65536,
        "compat": ANT_LING_COMPAT,
    },
    {
        "id": "Ring-2.6-1T",
        "name": "Ring 2.6 1T",
        "api": "openai-completions",
        "baseUrl": "https://api.ant-ling.com/v1",
        "provider": "ant-ling",
        "reasoning": True,
        "input": ["text"],
        "cost": {"input": 0.06, "output": 0.25, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 262144,
        "maxTokens": 65536,
        "compat": {**ANT_LING_COMPAT, "thinkingFormat": "ant-ling"},
    },
]

MINIMAX_DIRECT_SUPPORTED_IDS = {"MiniMax-M2.7", "MiniMax-M2.7-highspeed", "MiniMax-M3"}

# OpenAI Codex (ChatGPT OAuth) models. Not fetched from models.dev; a small explicit
# list avoids aliases. Older limits are based on observed server behavior; GPT-5.6
# follows Codex's 272k catalog limit (formerly 372k).
CODEX_BASE_URL = "https://chatgpt.com/backend-api"
CODEX_CONTEXT = 272000
CODEX_GPT_56_CONTEXT = 272000
CODEX_SPARK_CONTEXT = 128000
CODEX_MAX_TOKENS = 128000


def _codex_model(
    model_id: str,
    name: str,
    cost: dict[str, Any],
    *,
    context_window: int,
    model_input: list[str],
) -> dict[str, Any]:
    return {
        "id": model_id,
        "name": name,
        "api": "openai-codex-responses",
        "provider": "openai-codex",
        "baseUrl": CODEX_BASE_URL,
        "reasoning": True,
        "input": model_input,
        "cost": cost,
        "contextWindow": context_window,
        "maxTokens": CODEX_MAX_TOKENS,
    }


CODEX_MODELS: list[dict[str, Any]] = [
    _codex_model(
        "gpt-5.3-codex-spark",
        "GPT-5.3 Codex Spark",
        {"input": 1.75, "output": 14, "cacheRead": 0.175, "cacheWrite": 0},
        context_window=CODEX_SPARK_CONTEXT,
        model_input=["text"],
    ),
    _codex_model(
        "gpt-5.4",
        "GPT-5.4",
        with_openai_long_context_pricing({"input": 2.5, "output": 15, "cacheRead": 0.25, "cacheWrite": 0}),
        context_window=CODEX_CONTEXT,
        model_input=["text", "image"],
    ),
    _codex_model(
        "gpt-5.4-mini",
        "GPT-5.4 mini",
        {"input": 0.75, "output": 4.5, "cacheRead": 0.075, "cacheWrite": 0},
        context_window=CODEX_CONTEXT,
        model_input=["text", "image"],
    ),
    _codex_model(
        "gpt-5.5",
        "GPT-5.5",
        with_openai_long_context_pricing({"input": 5, "output": 30, "cacheRead": 0.5, "cacheWrite": 0}),
        context_window=CODEX_CONTEXT,
        model_input=["text", "image"],
    ),
    _codex_model(
        "gpt-5.6-luna",
        "GPT-5.6 Luna",
        with_openai_long_context_pricing(OPENAI_GPT_56_STANDARD_COSTS["gpt-5.6-luna"]),
        context_window=CODEX_GPT_56_CONTEXT,
        model_input=["text", "image"],
    ),
    _codex_model(
        "gpt-5.6-sol",
        "GPT-5.6 Sol",
        with_openai_long_context_pricing({"input": 5, "output": 30, "cacheRead": 0.5, "cacheWrite": 6.25}),
        context_window=CODEX_GPT_56_CONTEXT,
        model_input=["text", "image"],
    ),
    _codex_model(
        "gpt-5.6-terra",
        "GPT-5.6 Terra",
        with_openai_long_context_pricing(OPENAI_GPT_56_STANDARD_COSTS["gpt-5.6-terra"]),
        context_window=CODEX_GPT_56_CONTEXT,
        model_input=["text", "image"],
    ),
]

# Azure Foundry deploys these with larger context windows than OpenAI's own
# short-tier defaults. See models-sold-directly-by-azure docs.
AZURE_CONTEXT_WINDOW_OVERRIDES = {
    "gpt-5.4": 1050000,
    "gpt-5.5": 1050000,
    "gpt-5.6-luna": 1050000,
    "gpt-5.6-sol": 1050000,
    "gpt-5.6-terra": 1050000,
}


def apply_overrides(models: list[dict[str, Any]]) -> None:
    """Temporary overrides until upstream model metadata is corrected."""
    for candidate in models:
        provider = candidate["provider"]
        model_id = candidate["id"]

        if provider == "github-copilot" and model_id in GITHUB_COPILOT_EXTENDED_CONTEXT_MODELS:
            candidate["contextWindow"] = 1000000

        if provider in ("anthropic", "opencode", "opencode-go") and model_id in (
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-opus-4.6",
            "claude-sonnet-4.6",
        ):
            candidate["contextWindow"] = 1000000

        # OpenCode variants list Claude Sonnet 4/4.5 with 1M context; actual limit is 200K.
        if provider in ("opencode", "opencode-go") and model_id in ("claude-sonnet-4-5", "claude-sonnet-4"):
            candidate["contextWindow"] = 200000
        if provider in ("opencode", "opencode-go") and model_id == "gpt-5.4":
            candidate["contextWindow"] = 272000
            candidate["maxTokens"] = 128000
        # Keep direct OpenAI requests in the short-context pricing tier by default.
        # Users can opt into the larger context through model overrides, so retain
        # long-context cost metadata on the capped models.
        if provider == "openai" and model_id in OPENAI_SHORT_CONTEXT_CAPPED_MODEL_IDS:
            candidate["contextWindow"] = OPENAI_LONG_CONTEXT_INPUT_THRESHOLD
            candidate["maxTokens"] = 128000
        if provider == "openai" and model_id in OPENAI_LONG_CONTEXT_PRICING_MODEL_IDS:
            standard_cost = OPENAI_GPT_56_STANDARD_COSTS.get(model_id)
            candidate["cost"] = with_openai_long_context_pricing(
                standard_cost if standard_cost is not None else candidate["cost"]
            )
        # Cloudflare AI Gateway passes OpenAI usage through at OpenAI list prices.
        if provider == "cloudflare-ai-gateway":
            standard_cost = OPENAI_GPT_56_STANDARD_COSTS.get(model_id)
            if standard_cost:
                candidate["cost"] = with_openai_long_context_pricing(standard_cost)
        # models.dev reports gpt-5-pro output as 272000 (a duplicate of the input
        # sub-limit); the actual max output is 128000. Propagates to the Azure clone.
        if provider == "openai" and model_id == "gpt-5-pro":
            candidate["maxTokens"] = 128000
        # Keep Kimi K3's canonical output limit when gateway metadata is missing/wrong.
        if (provider == "openrouter" and model_id in OPENROUTER_KIMI_K3_MODEL_IDS) or (
            provider == "vercel-ai-gateway" and model_id == "moonshotai/kimi-k3"
        ):
            candidate["maxTokens"] = KIMI_K3_MAX_TOKENS
        # Keep selected OpenRouter model metadata stable until upstream settles.
        if provider == "openrouter" and model_id == "moonshotai/kimi-k2.5":
            candidate["cost"]["input"] = 0.41
            candidate["cost"]["output"] = 2.06
            candidate["cost"]["cacheRead"] = 0.07
            candidate["maxTokens"] = 4096
        if provider == "openrouter" and model_id.startswith("moonshotai/kimi-k2.6"):
            candidate["compat"] = {
                **candidate.get("compat", {}),
                "supportsDeveloperRole": False,
                "requiresReasoningContentOnAssistantMessages": True,
            }
        if provider == "openrouter" and model_id == "z-ai/glm-5":
            candidate["cost"]["input"] = 0.6
            candidate["cost"]["output"] = 1.9
            candidate["cost"]["cacheRead"] = 0.119


def apply_deepseek_v4_compat(models: list[dict[str, Any]]) -> None:
    for candidate in models:
        if (
            candidate["api"] != "openai-completions"
            or "deepseek-v4" not in candidate["id"]
            or candidate["provider"] in QWEN_TOKEN_PLAN_PROVIDER_IDS
        ):
            continue
        preserves_native_reasoning_effort = candidate["provider"] in ("openrouter", "opencode")
        candidate["compat"] = {
            **candidate.get("compat", {}),
            **(
                {
                    "requiresReasoningContentOnAssistantMessages": DEEPSEEK_COMPAT[
                        "requiresReasoningContentOnAssistantMessages"
                    ]
                }
                if preserves_native_reasoning_effort
                else DEEPSEEK_COMPAT
            ),
        }


def build_azure_clones(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clones = []
    for model in models:
        if model["provider"] != "openai" or model["api"] != "openai-responses":
            continue
        clone = dict(model)
        clone["api"] = "azure-openai-responses"
        clone["provider"] = "azure-openai-responses"
        clone["baseUrl"] = ""
        clone["cost"] = {
            "input": model["cost"]["input"],
            "output": model["cost"]["output"],
            "cacheRead": model["cost"]["cacheRead"],
            "cacheWrite": model["cost"]["cacheWrite"],
        }
        clone["contextWindow"] = AZURE_CONTEXT_WINDOW_OVERRIDES.get(model["id"], model["contextWindow"])
        clones.append(clone)
    return clones


def _clone(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deep copy of literal catalogs, so the metadata passes cannot mutate them."""
    return json.loads(json.dumps(models))


# --- entry point --------------------------------------------------------------


@tonio.main
async def main() -> None:
    reasoning_options: dict[str, list] = {}

    async with Client() as client:
        print("Fetching models from models.dev API...")
        catalog = await _fetch_json(client, "https://models.dev/api.json", "models.dev API")
        nvidia_nim_model_ids = await fetch_nvidia_nim_model_ids(client) if _models_of(catalog, "nvidia") else {}
        models_dev_models = load_models_dev_data(catalog, reasoning_options, nvidia_nim_model_ids)
        openrouter_models = await fetch_openrouter_models(client)
        ai_gateway_models = await fetch_ai_gateway_models(client)

    # models.dev has priority over the live gateway catalogs (the dedupe below keeps
    # the first entry for a given provider/id pair).
    all_models = [
        model
        for model in (*models_dev_models, *openrouter_models, *ai_gateway_models)
        if not (model["provider"] == "xai" and model["id"] in XAI_BUILTIN_EXCLUDED_MODEL_IDS)
        and not (model["provider"] in ("opencode", "opencode-go") and model["id"] == "gpt-5.3-codex-spark")
    ]

    apply_overrides(all_models)

    for model in _clone(MISSING_OPENAI_MODELS):
        if not any(m["provider"] == model["provider"] and m["id"] == model["id"] for m in all_models):
            all_models.append(model)

    all_models.extend(_clone(DEEPSEEK_V4_MODELS))
    all_models.extend(_clone(ANT_LING_MODELS))

    apply_deepseek_v4_compat(all_models)

    all_models = [
        model
        for model in all_models
        if not (model["provider"] in ("minimax", "minimax-cn") and model["id"] not in MINIMAX_DIRECT_SUPPORTED_IDS)
    ]

    all_models.extend(_clone(CODEX_MODELS))

    # Mistral Medium 3.5, until models.dev includes it.
    if not any(m["provider"] == "mistral" and m["id"] == "mistral-medium-3.5" for m in all_models):
        all_models.append(
            {
                "id": "mistral-medium-3.5",
                "name": "Mistral Medium 3.5",
                "api": "mistral-conversations",
                "provider": "mistral",
                "baseUrl": "https://api.mistral.ai",
                "reasoning": True,
                "input": ["text", "image"],
                "cost": {"input": 1.5, "output": 7.5, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 262144,  # 256k tokens
                "maxTokens": 262144,
            }
        )

    # "auto" alias for openrouter/auto.
    if not any(m["provider"] == "openrouter" and m["id"] == "auto" for m in all_models):
        all_models.append(
            {
                "id": "auto",
                "name": "Auto",
                "api": "openai-completions",
                "provider": "openrouter",
                "baseUrl": "https://openrouter.ai/api/v1",
                "reasoning": True,
                "input": ["text", "image"],
                # Costs are unknown: OpenRouter auto routes to different models and
                # charges for the underlying model.
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 2000000,
                "maxTokens": 30000,
            }
        )

    # "fusion" alias for openrouter/fusion. OpenRouter exposes Fusion as a router
    # alias/plugin entry point; its model metadata does not advertise tools, but the
    # alias resolves to a concrete model that can invoke caller tools and has the
    # openrouter:fusion server tool auto-injected.
    if not any(m["provider"] == "openrouter" and m["id"] == "openrouter/fusion" for m in all_models):
        all_models.append(
            {
                "id": "openrouter/fusion",
                "name": "OpenRouter: Fusion",
                "api": "openai-completions",
                "provider": "openrouter",
                "baseUrl": "https://openrouter.ai/api/v1",
                "reasoning": True,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 1000000,
                "maxTokens": 30000,
            }
        )

    all_models.extend(build_azure_clones(all_models))

    # Metadata passes, in pi's exact order (generate-models.ts:2503-2511) — the order
    # is load-bearing: reasoning-options runs before forceAdaptiveThinking is set, so
    # anthropic models never take models.dev effort maps.
    for model in all_models:
        apply_openai_completions_compat_metadata(model)
        apply_models_dev_reasoning_option_metadata(model, reasoning_options)
        apply_thinking_level_metadata(model)
        apply_strict_tool_compat_metadata(model)
        apply_openai_grammar_tool_compat_metadata(model)
        apply_openai_tool_search_metadata(model)
        apply_openai_explicit_prompt_cache_metadata(model)

    # Group by provider, dedupe by model id (first wins), sort, group by api.
    providers: dict[str, dict[str, dict[str, Any]]] = {}
    for model in all_models:
        providers.setdefault(model["provider"], {}).setdefault(model["id"], model)

    # Serialize into memory first, so a failed integrity check leaves the committed
    # catalog untouched (pi stages into a temp dir and swaps; same guarantee).
    file_contents: dict[str, str] = {}
    structure: ModelDataStructure = {}
    for provider_id in sorted(providers):
        by_api: dict[str, dict[str, Any]] = {}
        provider_models = providers[provider_id]
        structure[provider_id] = {}
        for api in sorted({model["api"] for model in provider_models.values()}):
            by_api[api] = {
                model_id: provider_models[model_id]
                for model_id in sorted(provider_models)
                if provider_models[model_id]["api"] == api
            }
            for model_id in by_api[api]:
                structure[provider_id][model_id] = api
        file_contents[f"{provider_id}.json"] = json.dumps(by_api, indent=2) + "\n"

    manifest = create_model_data_manifest(
        structure,
        file_contents,
        datetime.now(UTC).isoformat(timespec="seconds"),
        source="https://models.dev/api.json",
    )

    with tempfile.TemporaryDirectory(prefix="pidrei-model-data-") as staging:
        staging_dir = Path(staging)
        for filename, content in file_contents.items():
            (staging_dir / filename).write_text(content)
        (staging_dir / MODEL_DATA_MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
        validate_model_data_directory(structure, staging_dir)

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for stale in DATA_DIR.glob("*.json"):
            if stale.name != MODEL_DATA_MANIFEST_FILE and stale.name not in file_contents:
                stale.unlink()
        for path in staging_dir.glob("*.json"):
            (DATA_DIR / path.name).write_text(path.read_text())
        for filename in sorted(file_contents):
            print(f"Wrote {len(providers[filename.removesuffix('.json')])} models to {DATA_DIR / filename}")
    validate_generated_model_data(DATA_DIR)

    reasoning_count = sum(1 for model in all_models if model["reasoning"])
    print("\nModel Statistics:")
    print(f"  Total tool-capable models: {len(all_models)}")
    print(f"  Reasoning-capable models: {reasoning_count}")
    for provider_id in sorted(providers):
        print(f"  {provider_id}: {len(providers[provider_id])} models")


if __name__ == "__main__":
    main()
