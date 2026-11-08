"""OpenAI (pi: catalog/providers/openai.ts)."""

import copy
from typing import Any

from catalog.define import (
    ChatModel,
    ProviderContext,
    ProviderDefinition,
    add_missing,
    merge_compat,
    merge_thinking,
    models_dev_models,
    with_input_limits,
)
from catalog.families.openai import (
    OPENAI_LONG_CONTEXT_INPUT_THRESHOLD,
    OPENAI_STANDARD_COSTS,
    OPENAI_TOOL_SEARCH_MODEL_IDS,
    enable_openai_grammar_tools,
    enable_openai_mid_convo_system_messages,
    with_openai_long_context_pricing,
)
from catalog.sources.models_dev import ModelsDevModel


BASE_URL = "https://api.openai.com/v1"

# models.dev lists this alias, but it is not accepted by OpenAI APIs.
UNSUPPORTED_MODEL_IDS = frozenset({"gpt-5.6"})

# Keep direct OpenAI requests in the short-context pricing tier by default. Users can opt into the
# larger context through model overrides, so retain long-context cost metadata on the capped models.
SHORT_CONTEXT_CAPPED_MODEL_IDS = frozenset(
    {
        "gpt-5.4",
        "gpt-5.5",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-6-astra",
        "gpt-6-sol",
        "gpt-6-luna",
        "gpt-6.1-sol",
        "gpt-6.1-terra",
    }
)
LONG_CONTEXT_PRICING_MODEL_IDS = frozenset({*SHORT_CONTEXT_CAPPED_MODEL_IDS, "gpt-5.4-pro", "gpt-5.5-pro"})

NONE_REASONING_MODEL_IDS = frozenset(
    {
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
        "gpt-6-sol",
        "gpt-6-luna",
    }
)
# OpenAI documents configuration_update for the GPT-6 family.
# https://developers.openai.com/api/docs/guides/reasoning
MID_CONVO_EFFORT_MODEL_IDS = frozenset({"gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol", "gpt-6.1-terra"})

# Not listed by models.dev.
GPT_5_CHAT_LATEST: ChatModel = {
    "id": "gpt-5-chat-latest",
    "name": "GPT-5 Chat Latest",
    "api": "openai-responses",
    "baseUrl": BASE_URL,
    "provider": "openai",
    "reasoning": False,
    "input": ["text", "image"],
    "cost": {"input": 1.25, "output": 10, "cacheRead": 0.125, "cacheWrite": 0},
    "contextWindow": 128000,
    "maxTokens": 16384,
}

# OpenAI Decisions API (public beta): gpt-6-luna is its only model. It bills input tokens only,
# with the same long-context multiplier as chat requests. Only API keys work: Sign in with ChatGPT
# tokens are rejected on /v1/decisions, and the Codex backend has no Decisions route.
# The endpoint rejects inputs above 922K tokens (the model's documented maximum input), but
# requests running longer than about five seconds, currently above roughly 600K input tokens,
# fail with a gateway timeout.
# https://developers.openai.com/api/docs/guides/decisions
CLASSIFIER_MODELS: list[dict[str, Any]] = [
    {
        "type": "classifier",
        "id": "gpt-6-luna",
        "name": "GPT-6 Luna",
        "api": "openai-decisions",
        "provider": "openai",
        "baseUrl": BASE_URL,
        "input": ["text", "image"],
        "cost": with_openai_long_context_pricing({"input": 0.1, "output": 0, "cacheRead": 0, "cacheWrite": 0}),
        "contextWindow": 922000,
    },
]


def _correct_upstream(model: ChatModel, _source: ModelsDevModel) -> ChatModel:
    if model["id"] in SHORT_CONTEXT_CAPPED_MODEL_IDS:
        model["contextWindow"] = OPENAI_LONG_CONTEXT_INPUT_THRESHOLD
        model["maxTokens"] = 128000
    if model["id"] in LONG_CONTEXT_PRICING_MODEL_IDS:
        standard_cost = OPENAI_STANDARD_COSTS.get(model["id"])
        model["cost"] = with_openai_long_context_pricing(model["cost"] if standard_cost is None else standard_cost)
    # models.dev reports gpt-5-pro output as 272000 (a duplicate of the input sub-limit);
    # the actual max output is 128000. Also propagates to the derived Azure clone.
    if model["id"] == "gpt-5-pro":
        model["maxTokens"] = 128000
    return model


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return add_missing(
        models_dev_models(
            ctx,
            "openai",
            api="openai-responses",
            base_url=BASE_URL,
            # Audio models are only served on Chat Completions.
            protocol="responses",
            include=lambda source: source.id not in UNSUPPORTED_MODEL_IDS,
            map_model=_correct_upstream,
        ),
        [copy.deepcopy(GPT_5_CHAT_LATEST)],
    )


def _classifiers(_ctx: ProviderContext) -> list[dict[str, Any]]:
    return copy.deepcopy(CLASSIFIER_MODELS)


def _finalize(model: ChatModel, _ctx: ProviderContext) -> None:
    if model["api"] != "openai-responses":
        return
    if model["id"] in NONE_REASONING_MODEL_IDS:
        merge_thinking(model, {"off": "none"})
    if model["id"] == "gpt-5.5":
        merge_thinking(model, {"minimal": None})
    merge_compat(model, {"supportsStrictMode": True})
    enable_openai_grammar_tools(model)
    if model["id"] in OPENAI_TOOL_SEARCH_MODEL_IDS:
        merge_compat(model, {"supportsAdditionalTools": True, "supportsToolSearch": True})
    enable_openai_mid_convo_system_messages(model, proxy=False)
    # OpenAI charges prompt-cache writes starting with the GPT-5.6 family, and exactly
    # those models accept `prompt_cache_options`; older models reject the parameter.
    # https://developers.openai.com/api/docs/guides/prompt-caching
    if model["cost"]["cacheWrite"] > 0:
        merge_compat(model, {"supportsExplicitPromptCacheMode": True})
    if model["id"] in MID_CONVO_EFFORT_MODEL_IDS:
        merge_compat(model, {"supportsMidConvoEffort": True})
    with_input_limits(model, {"maxRequestBytes": 512 * 1024 * 1024, "images": {"maxPerRequest": 1500}})


PROVIDER = ProviderDefinition(
    id="openai",
    models=_models,
    classifiers=_classifiers,
    finalize=_finalize,
)
