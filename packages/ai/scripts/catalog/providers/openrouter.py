"""OpenRouter (pi: catalog/providers/openrouter.ts)."""

import copy
import re
from typing import Any

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, add_missing, merge_compat, merge_thinking
from catalog.families.anthropic import enable_claude_mid_convo_effort, supports_claude_mid_convo_effort
from catalog.families.kimi import KIMI_K3_MAX_TOKENS, is_kimi_k3_family_id
from catalog.families.openai import OPENAI_MID_CONVO_MODEL_IDS


BASE_URL = "https://openrouter.ai/api/v1"
# OpenRouter rejects `configuration_update` system messages on Opus 5 ("Mid-conversation
# reasoning effort (configuration_update) is not supported on anthropic/claude-opus-5-20260723")
# while accepting them on Fable 5.1.
MID_CONVO_EFFORT_UNSUPPORTED_MODEL_IDS = frozenset({"anthropic/claude-opus-5"})


def _correct_upstream(model: ChatModel) -> ChatModel:
    """Keep selected model metadata stable until upstream settles."""
    # Keep Kimi K3's canonical output limit when gateway metadata is missing or incorrect.
    if model["id"] == "~moonshotai/kimi-latest" or (
        model["id"].startswith("moonshotai/") and is_kimi_k3_family_id(model["id"][len("moonshotai/") :])
    ):
        model["maxTokens"] = KIMI_K3_MAX_TOKENS
    if model["id"] == "moonshotai/kimi-k2.5":
        model["cost"] = {**model["cost"], "input": 0.41, "output": 2.06, "cacheRead": 0.07}
        model["maxTokens"] = 4096
    if model["id"].startswith("moonshotai/kimi-k2.6"):
        merge_compat(
            model,
            {
                "supportsDeveloperRole": False,
                "requiresReasoningContentOnAssistantMessages": True,
            },
        )
    if model["id"] == "z-ai/glm-5":
        model["cost"] = {**model["cost"], "input": 0.6, "output": 1.9, "cacheRead": 0.119}
    return model


# Router aliases. Their price depends on the model they route to, so it is unknown here.
ROUTER_MODELS: list[ChatModel] = [
    {
        "id": "auto",
        "name": "Auto",
        "api": "openai-completions",
        "provider": "openrouter",
        "baseUrl": BASE_URL,
        "reasoning": True,
        "input": ["text", "image"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 2000000,
        "maxTokens": 30000,
    },
    # OpenRouter exposes Fusion as a router alias/plugin entry point; its model
    # metadata does not advertise tools, but the alias resolves to a concrete model
    # that can invoke caller tools and has the openrouter:fusion server tool auto-injected.
    {
        "id": "openrouter/fusion",
        "name": "OpenRouter: Fusion",
        "api": "openai-completions",
        "provider": "openrouter",
        "baseUrl": BASE_URL,
        "reasoning": True,
        "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 1000000,
        "maxTokens": 30000,
    },
]

_ANTHROPIC_OR_OPENAI_RE = re.compile(r"^(anthropic|openai)/")
_ANTHROPIC_RE = re.compile(r"^~?anthropic/")


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return add_missing(
        [_correct_upstream(model) for model in copy.deepcopy(ctx.sources.open_router["chat"])],
        copy.deepcopy(ROUTER_MODELS),
    )


def _images(ctx: ProviderContext) -> list[dict[str, Any]]:
    return copy.deepcopy(ctx.sources.open_router["images"])


def _classifiers(ctx: ProviderContext) -> list[dict[str, Any]]:
    return copy.deepcopy(ctx.sources.open_router["classifiers"])


def _completions_compat(model: ChatModel) -> dict[str, Any]:
    return {
        # OpenRouter passes developer messages through to Anthropic and OpenAI models only.
        **({} if _ANTHROPIC_OR_OPENAI_RE.search(model["id"]) is not None else {"supportsDeveloperRole": False}),
        "thinkingFormat": "openrouter",
        "supportsStrictMode": True,
        **({"cacheControlFormat": "anthropic"} if _ANTHROPIC_RE.search(model["id"]) is not None else {}),
        "sendSessionAffinityHeaders": True,
    }


def _finalize(model: ChatModel, _ctx: ProviderContext) -> None:
    if (
        model["api"] == "anthropic-messages"
        and supports_claude_mid_convo_effort(model["id"])
        and model["id"] not in MID_CONVO_EFFORT_UNSUPPORTED_MODEL_IDS
    ):
        enable_claude_mid_convo_effort(model)
    if model["id"].startswith("inception/mercury-2"):
        # Mercury 2 in instant mode (reasoning_effort: "none") disables tool calling.
        # Mark "off" unsupported so the openai-completions provider omits the reasoning param
        # instead of defaulting to {reasoning:{effort:"none"}}.
        # PiDrei's low/medium/high pass through verbatim; OpenRouter normalizes to Mercury's vocabulary.
        merge_thinking(model, {"off": None})
    if model["id"] == "z-ai/glm-5.2":
        merge_thinking(model, {"xhigh": "xhigh"})
    # OpenAI models behind OpenRouter accept plain system text in place.
    if (
        model["api"] == "openai-completions"
        and model["id"].startswith("openai/")
        and model["id"][len("openai/") :] in OPENAI_MID_CONVO_MODEL_IDS
    ):
        merge_compat(model, {"supportsMidConvoSystemMessages": True})


PROVIDER = ProviderDefinition(
    id="openrouter",
    models=_models,
    images=_images,
    classifiers=_classifiers,
    compat={"openai-completions": _completions_compat},
    finalize=_finalize,
)
