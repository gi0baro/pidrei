"""OpenAI GPT models: facts that hold on every host that serves them
(pi: catalog/families/openai.ts)."""

import re
from typing import Any

from catalog.define import ChatModel, merge_compat, merge_thinking
from catalog.sources.models_dev import round_cost


OPENAI_LONG_CONTEXT_INPUT_THRESHOLD = 272000


def with_openai_long_context_pricing(cost: dict[str, Any]) -> dict[str, Any]:
    """OpenAI bills prompts above 272K input tokens at 2x input and 1.5x output rates."""
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


# Keep current OpenAI prices authoritative until models.dev and passthrough
# catalogs catch up.
# https://developers.openai.com/api/docs/pricing
OPENAI_STANDARD_COSTS: dict[str, dict[str, Any]] = {
    "gpt-5.6-luna": {"input": 0.2, "output": 1.2, "cacheRead": 0.02, "cacheWrite": 0.25},
    "gpt-5.6-sol": {"input": 4, "output": 20, "cacheRead": 0.4, "cacheWrite": 5},
    "gpt-5.6-terra": {"input": 2, "output": 12, "cacheRead": 0.2, "cacheWrite": 2.5},
    "gpt-6-astra": {"input": 10, "output": 50, "cacheRead": 1, "cacheWrite": 12.5},
    "gpt-6-luna": {"input": 0.1, "output": 0.5, "cacheRead": 0.01, "cacheWrite": 0.125},
    "gpt-6-sol": {"input": 2, "output": 10, "cacheRead": 0.2, "cacheWrite": 2.5},
    "gpt-6.1-sol": {"input": 2, "output": 10, "cacheRead": 0.1, "cacheWrite": 2.5},
    "gpt-6.1-terra": {"input": 1, "output": 5, "cacheRead": 0.05, "cacheWrite": 1.25},
}


def openai_standard_cost(model_id: str) -> dict[str, Any]:
    """The standard price of an OpenAI model with long-context pricing."""
    return with_openai_long_context_pricing(OPENAI_STANDARD_COSTS[model_id])


# GPT-5.5 and older put tools loaded through tool search or additional_tools in a namespace named
# after the tool and reject replayed calls without it, so history from other models fails there
# (#11415). Like Codex, dynamic tool loading therefore starts with GPT-5.6, which does not do this.
# Models that accept tool search and `additional_tools`.
OPENAI_TOOL_SEARCH_MODEL_IDS = frozenset(
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
# Models that accept mid-conversation developer messages.
OPENAI_MID_CONVO_MODEL_IDS = frozenset(
    {
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-pro",
        "gpt-5.5",
        *OPENAI_TOOL_SEARCH_MODEL_IDS,
    }
)

_RESPONSES_APIS = frozenset({"openai-responses", "azure-openai-responses", "openai-codex-responses"})


def _is_responses_api(model: ChatModel) -> bool:
    return model["api"] in _RESPONSES_APIS


def supports_openai_xhigh(model_id: str) -> bool:
    return any(marker in model_id for marker in ("gpt-5.2", "gpt-5.3", "gpt-5.4", "gpt-5.5", "gpt-5.6", "gpt-6"))


_GPT_6_MODEL_IDS = frozenset({"gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol", "gpt-6.1-terra"})


def apply_openai_family_rules(model: ChatModel) -> None:
    """Thinking levels of GPT models on any host."""
    model_id = model["id"]
    if model["api"] in ("openai-responses", "azure-openai-responses") and model_id.startswith("gpt-5"):
        merge_thinking(model, {"off": None})
    if model_id in _GPT_6_MODEL_IDS and _is_responses_api(model):
        merge_thinking(
            model,
            {
                # GPT-6 Astra and GPT-6.1 Sol reject reasoning.effort "none"; GPT-6.1 Terra accepts it. All reject
                # "minimal" (verified on Codex, October 2026).
                "off": None if model_id in ("gpt-6-astra", "gpt-6.1-sol") else "none",
                "minimal": None,
                "low": "low",
                "medium": "medium",
                "high": "high",
                "xhigh": "xhigh",
                "max": "max",
            },
        )
    if supports_openai_xhigh(model_id):
        merge_thinking(model, {"xhigh": "xhigh"})
    if ("gpt-5.6" in model_id or "gpt-6" in model_id) and (
        _is_responses_api(model) or model["api"] == "openai-completions"
    ):
        merge_thinking(model, {"max": "max"})
    if model_id.endswith("gpt-5.5-pro"):
        merge_thinking(model, {"off": None, "minimal": None, "low": None})


_GPT_MAJOR_RE = re.compile(r"^gpt-(\d+)")


def enable_openai_grammar_tools(model: ChatModel) -> None:
    """OpenAI custom grammar tools on Responses endpoints. OpenAI rejects
    `type: "custom"` tools for pre-GPT-5 models (gpt-4.x, gpt-4o, o-series)."""
    if not _is_responses_api(model):
        return
    match = _GPT_MAJOR_RE.search(model["id"])
    if not match or int(match.group(1)) < 5:
        return
    merge_compat(model, {"supportsOpenAIGrammarTools": True})


def enable_openai_mid_convo_system_messages(model: ChatModel, *, proxy: bool) -> None:
    """Newer OpenAI Responses models accept developer messages after the
    conversation has started. Proxies that pass them through also pass
    `additional_tools` items through unchanged, for the models that accept them."""
    if model["api"] not in ("openai-responses", "openai-codex-responses"):
        return
    if model["id"] not in OPENAI_MID_CONVO_MODEL_IDS:
        return
    merge_compat(
        model,
        {
            "supportsMidConvoSystemMessages": True,
            **({"supportsAdditionalTools": True} if proxy and model["id"] in OPENAI_TOOL_SEARCH_MODEL_IDS else {}),
        },
    )
