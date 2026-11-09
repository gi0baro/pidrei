"""OpenCode Zen (pi: catalog/providers/opencode.ts)."""

import re
import sys
from typing import Any

from catalog.define import (
    ChatModel,
    ProviderContext,
    ProviderDefinition,
    merge_compat,
    merge_thinking,
    models_dev_models,
)
from catalog.families.anthropic import supports_claude_mid_convo_system_messages
from catalog.families.google import google_thinking_levels
from catalog.families.kimi import is_kimi_k3_family_id
from catalog.families.openai import enable_openai_grammar_tools, enable_openai_mid_convo_system_messages
from catalog.reasoning import get_effort_thinking_level_map
from catalog.sources.models_dev import ModelsDevModel


# OpenAI-compatible endpoints reject the long prompt-cache retention parameter for these models.
LONG_CACHE_RETENTION_UNSUPPORTED_MODELS = frozenset(
    {
        "opencode:deepseek-v4-flash",
        "opencode:deepseek-v4.1-flash",
        "opencode:deepseek-v4-pro",
        "opencode:kimi-k2.5",
        "opencode:kimi-k2.6",
        "opencode:minimax-m2.7",
        "opencode-go:kimi-k2.6",
    }
)

# OpenCode serves each model on one protocol-specific endpoint. Native protocols
# come before generic Chat Completions if models.dev ever lists several.
OPENCODE_PROTOCOLS = ("responses", "messages", "generate-content", "chat-completions")

_V1_SUFFIX_RE = re.compile(r"/v1/?$")
_ANTHROPIC_SUFFIX_RE = re.compile(r"/anthropic$")


def _opencode_api(ctx: ProviderContext, source: ModelsDevModel) -> dict[str, Any] | None:
    """The API and base URL models.dev lists for a model. Models served only on a
    protocol PiDrei does not support fail strict generation and are skipped otherwise.

    Returns {"api", "baseUrl", "compat"?} (pi's `OpencodeApi`)."""
    entry = next(
        (
            candidate
            for candidate in (
                next((api for api in source.apis if api.protocol == protocol), None) for protocol in OPENCODE_PROTOCOLS
            )
            if candidate is not None
        ),
        None,
    )
    if entry is None:
        protocols = ", ".join(candidate.protocol for candidate in source.apis) or "none"
        message = f"models.dev lists no supported API for {ctx.provider_id}/{source.id} ({protocols})"
        if ctx.strict:
            raise RuntimeError(message)
        print(f"Skipping model: {message}", file=sys.stderr)
        return None
    match entry.protocol:
        case "responses":
            return {
                "api": "openai-responses",
                "baseUrl": entry.base_url,
                "compat": {"sessionAffinityFormat": "openai-nosession"},
            }
        case "messages":
            # The Anthropic SDK appends /v1/messages itself.
            return {"api": "anthropic-messages", "baseUrl": _V1_SUFFIX_RE.sub("", entry.base_url, count=1)}
        case "generate-content":
            return {"api": "google-generative-ai", "baseUrl": entry.base_url}
        case "chat-completions":
            return {
                "api": "openai-completions",
                "baseUrl": entry.base_url,
                **({"compat": {"cacheControlFormat": "anthropic"}} if entry.aisdk == "@ai-sdk/alibaba" else {}),
            }
    return None


def opencode_models(ctx: ProviderContext) -> list[ChatModel]:
    """OpenCode Zen and Go models, on the endpoints models.dev lists for them."""
    provider = ctx.provider_id

    def map_model(model: ChatModel, source: ModelsDevModel) -> ChatModel | None:
        served = _opencode_api(ctx, source)
        if served is None:
            return None
        api = served["api"]
        base_url = served["baseUrl"]
        compat = served.get("compat")
        if provider == "opencode" and model["id"] == "grok-build-0.1":
            compat = {**(compat or {}), "supportsReasoningEffort": False}
        if model["id"] == "kimi-k2.6":
            # OpenCode Kimi K2.6 accepts Anthropic-style thinking objects
            # and rejects string thinking values or combined reasoning_effort.
            compat = {**(compat or {}), "thinkingFormat": "deepseek", "supportsReasoningEffort": False}
        # models.dev reports these as @ai-sdk/anthropic, but OpenCode Go serves
        # them through the OpenAI-compatible /openai/v1/chat/completions path.
        if (
            provider == "opencode-go"
            and api == "anthropic-messages"
            and (model["id"] == "qwen3.5-plus" or model["id"] == "qwen3.6-plus")
        ):
            api = "openai-completions"
            base_url = _ANTHROPIC_SUFFIX_RE.sub("/openai/v1", base_url, count=1)
            # Qwen/DashScope uses enable_thinking at the top level.
            compat = {**(compat or {}), "thinkingFormat": "qwen"}
        if api == "openai-completions":
            compat = {
                **(compat or {}),
                "maxTokensField": "max_tokens",
                **(
                    {"supportsLongCacheRetention": False}
                    if f"{provider}:{model['id']}" in LONG_CACHE_RETENTION_UNSUPPORTED_MODELS
                    else {}
                ),
            }
        if api == "google-generative-ai":
            thinking_level_map = google_thinking_levels(model["id"], source.reasoning_options)
        elif provider == "opencode-go" and model["id"] == "deepseek-v4.1-flash":
            thinking_level_map = get_effort_thinking_level_map(
                [] if source.reasoning_options is None else source.reasoning_options
            )
        else:
            thinking_level_map = None
        result: ChatModel = {
            **model,
            "api": api,
            "baseUrl": base_url,
            **({"thinkingLevelMap": thinking_level_map} if thinking_level_map is not None else {}),
        }
        if compat is not None:
            result["compat"] = compat
        # OpenCode variants list Claude Sonnet 4/4.5 with 1M context, actual limit is 200K.
        if model["id"] == "claude-sonnet-4-5" or model["id"] == "claude-sonnet-4":
            result["contextWindow"] = 200000
        if model["id"] in ("claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4.6", "claude-sonnet-4.6"):
            result["contextWindow"] = 1000000
        if model["id"] == "gpt-5.4":
            result["contextWindow"] = 272000
            result["maxTokens"] = 128000
        return result

    return models_dev_models(
        ctx,
        provider,
        api="openai-completions",
        # Replaced by the base URL of the API models.dev lists for each model.
        base_url="",
        skip_deprecated=True,
        # Codex Spark is not usable through OpenCode.
        include=lambda source: source.id != "gpt-5.3-codex-spark",
        map_model=map_model,
    )


OPENCODE_COMPLETIONS_COMPAT: dict[str, Any] = {
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "supportsStrictMode": True,
}


def finalize_opencode_model(model: ChatModel, _ctx: ProviderContext) -> None:
    """Behavior shared by OpenCode Zen and Go."""
    if model["provider"] == "opencode-go" and model["id"] == "glm-5.2":
        merge_thinking(model, {"off": None, "minimal": None, "low": None, "medium": None, "high": "high", "max": "max"})
    if model["provider"] == "opencode-go" and model["id"] == "kimi-k2.6":
        # OpenCode Go exposes Kimi K2.6 thinking as on/off, not distinct effort tiers.
        merge_thinking(model, {"minimal": None, "low": None, "medium": None})
    if model["provider"] == "opencode" and model["id"] == "grok-build-0.1":
        # OpenCode Zen Grok Build reasons by default but rejects explicit reasoningEffort.
        merge_thinking(model, {"off": None, "minimal": None, "low": None, "medium": None})
    if model["api"] == "anthropic-messages":
        # OpenCode Zen forwards mid-conversation system messages but rejects
        # `tool_addition`/`tool_removal` blocks, so tool changes stay top-level.
        if model["provider"] == "opencode" and supports_claude_mid_convo_system_messages(model["id"]):
            merge_compat(model, {"supportsMidConvoSystemMessages": True})
        # OpenCode Qwen 3.8 Flash emits and accepts thinking blocks with empty signatures.
        if model["id"] == "qwen3.8-flash":
            merge_compat(model, {"allowEmptySignature": True})
    # The Kimi K3 family accepts tool-bearing system messages; OpenCode passes them through.
    if model["api"] == "openai-completions" and is_kimi_k3_family_id(model["id"]):
        merge_compat(
            model,
            {
                "supportsMidConvoSystemMessages": True,
                "supportsMidConvoToolAdditions": True,
            },
        )
    # OpenCode passes mid-conversation developer messages and `additional_tools`
    # through to OpenAI unchanged; tool search is not verified through it.
    enable_openai_mid_convo_system_messages(model, proxy=True)


# OpenCode Zen serves Jev through its TypeSafe-compatible System One endpoint.
# Neither its /inference/openai/v1/models listing nor models.dev carries metadata for it.
# https://opencode.ai/docs/zen
def _jev(model_id: str, name: str, input_cost: float) -> dict[str, Any]:
    return {
        "type": "classifier",
        "id": model_id,
        "name": name,
        "api": "typesafe-system-one",
        "provider": "opencode",
        "baseUrl": "https://opencode.ai/inference/systemone/v1",
        "input": ["text"],
        "cost": {"input": input_cost, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 32000,
    }


def _classifiers(_ctx: ProviderContext) -> list[dict[str, Any]]:
    return [_jev("jev-1.13", "Jev 1.13", 0.042), _jev("jev-1.13-free", "Jev 1.13 Free", 0)]


def _finalize(model: ChatModel, ctx: ProviderContext) -> None:
    finalize_opencode_model(model, ctx)
    enable_openai_grammar_tools(model)


PROVIDER = ProviderDefinition(
    id="opencode",
    models=opencode_models,
    classifiers=_classifiers,
    compat={"openai-completions": OPENCODE_COMPLETIONS_COMPAT},
    finalize=_finalize,
)
