"""OpenAI Codex (pi: catalog/providers/openai-codex.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, merge_compat, merge_thinking
from catalog.families.openai import (
    OPENAI_TOOL_SEARCH_MODEL_IDS,
    enable_openai_grammar_tools,
    enable_openai_mid_convo_system_messages,
    openai_standard_cost,
    supports_openai_xhigh,
)


# ChatGPT OAuth models. models.dev does not list them; keep a small, explicit
# list without aliases. Only list models the Codex backend accepts for ChatGPT
# accounts: GPT-5.5 and GPT-5.3 Codex Spark are rejected with "not supported
# when using Codex with a ChatGPT account".
MODELS: tuple[tuple[str, str, int], ...] = (
    # GPT-6.1 uses the 500k default returned by /backend-api/codex/models.
    ("gpt-6.1-sol", "GPT-6.1 Sol", 500000),
    ("gpt-6.1-terra", "GPT-6.1 Terra", 500000),
    # GPT-5.6 and GPT-6 use Codex's 272k default catalog limit.
    ("gpt-6-astra", "GPT-6 Astra", 272000),
    ("gpt-6-sol", "GPT-6 Sol", 272000),
    ("gpt-6-luna", "GPT-6 Luna", 272000),
    ("gpt-5.6-luna", "GPT-5.6 Luna", 272000),
    ("gpt-5.6-sol", "GPT-5.6 Sol", 272000),
    ("gpt-5.6-terra", "GPT-5.6 Terra", 272000),
)

# Codex CLI keeps additional_tools behind its own model list.
ADDITIONAL_TOOLS_MODEL_IDS = frozenset(
    {
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


def _models(_ctx: ProviderContext) -> list[ChatModel]:
    return [
        {
            "id": model_id,
            "name": name,
            "api": "openai-codex-responses",
            "provider": "openai-codex",
            "baseUrl": "https://chatgpt.com/backend-api",
            "reasoning": True,
            "input": ["text", "image"],
            "cost": openai_standard_cost(model_id),
            "contextWindow": context_window,
            "maxTokens": 128000,
        }
        for model_id, name, context_window in MODELS
    ]


def _finalize(model: ChatModel, _ctx: ProviderContext) -> None:
    if supports_openai_xhigh(model["id"]):
        merge_thinking(model, {"minimal": "low"})
    enable_openai_grammar_tools(model)
    if model["id"] in OPENAI_TOOL_SEARCH_MODEL_IDS:
        merge_compat(
            model,
            {
                **({"supportsAdditionalTools": True} if model["id"] in ADDITIONAL_TOOLS_MODEL_IDS else {}),
                "supportsToolSearch": True,
            },
        )
    enable_openai_mid_convo_system_messages(model, proxy=False)


PROVIDER = ProviderDefinition(
    id="openai-codex",
    models=_models,
    finalize=_finalize,
)
