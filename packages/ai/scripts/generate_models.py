"""Model catalog generator — Python port of pi's scripts/generate-models.ts.

Fetches https://models.dev/api.json and emits pi-shaped catalog JSON
(camelCase keys, grouped `{api: {modelId: Model}}`, sorted) into
`pppi_ai/providers/data/<provider>.json`. Keeping pi's exact JSON shape keeps
our vendored data diffable against pi's generated catalog.

Scope grows with the adapters (PLAN.md): currently anthropic + openai.
pi's OpenRouter/AI-Gateway/NVIDIA fetches, the other ~35 providers, and the
data manifest land with their provider wiring.

Run: make models-data   (network access to models.dev required)
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tonio.colored as tonio
from punkreq.tonio import Client


DATA_DIR = Path(__file__).parent.parent / "pppi_ai" / "providers" / "data"

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

THINKING_LEVELS = ["minimal", "low", "medium", "high", "xhigh", "max"]


def with_openai_long_context_pricing(cost: dict[str, Any]) -> dict[str, Any]:
    return {
        **cost,
        "tiers": [
            {
                "inputTokensAbove": OPENAI_LONG_CONTEXT_INPUT_THRESHOLD,
                "input": cost["input"] * 2,
                "output": cost["output"] * 1.5,
                "cacheRead": cost["cacheRead"] * 2,
                "cacheWrite": cost["cacheWrite"] * 2,
            }
        ],
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
            "sonnet-4-6",
            "sonnet-4.6",
            "sonnet-5",
            "sonnet.5",
            "fable-5",
        )
    )


def is_anthropic_temperature_unsupported_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return any(marker in lowered for marker in ("opus-4-7", "opus-4.7", "opus-4-8", "opus-4.8"))


def supports_openai_xhigh(model_id: str) -> bool:
    return any(marker in model_id for marker in ("gpt-5.2", "gpt-5.3", "gpt-5.4", "gpt-5.5", "gpt-5.6"))


def supports_openai_max(model: dict[str, Any]) -> bool:
    return "gpt-5.6" in model["id"] and model["api"] in (
        "openai-responses",
        "azure-openai-responses",
        "openai-codex-responses",
        "openai-completions",
    )


def supports_direct_reasoning_effort(model: dict[str, Any]) -> bool:
    # Subset of pi's check for the providers we emit; openai-completions compat
    # detection joins in with the completions providers.
    if model["api"] == "anthropic-messages":
        return model.get("compat", {}).get("forceAdaptiveThinking") is True
    return model["api"] in ("openai-responses", "azure-openai-responses", "openai-codex-responses")


def apply_models_dev_reasoning_option_metadata(model: dict[str, Any], reasoning_options: dict[str, list]) -> None:
    options = reasoning_options.get(f"{model['provider']}:{model['id']}")
    if not options or not supports_direct_reasoning_effort(model):
        return
    mapping = get_effort_thinking_level_map(options)
    if mapping:
        merge_thinking_level_map(model, mapping)


def apply_thinking_level_metadata(model: dict[str, Any]) -> None:
    # Branches for providers pppi does not emit yet are added with those providers.
    model_id = model["id"]
    if model["api"] in ("openai-responses", "azure-openai-responses") and model_id.startswith("gpt-5"):
        merge_thinking_level_map(model, {"off": None})
    if (
        model["api"] == "openai-responses"
        and model["provider"] == "openai"
        and model_id in OPENAI_RESPONSES_NONE_REASONING_MODELS
    ):
        merge_thinking_level_map(model, {"off": "none"})
    if supports_openai_xhigh(model_id):
        merge_thinking_level_map(model, {"xhigh": "xhigh"})
    if supports_openai_max(model):
        merge_thinking_level_map(model, {"max": "max"})
    if model["provider"] == "openai" and model_id == "gpt-5.5":
        merge_thinking_level_map(model, {"minimal": None})
    if model_id.endswith("gpt-5.5-pro"):
        merge_thinking_level_map(model, {"off": None, "minimal": None, "low": None})
    # Anthropic adaptive-thinking effort support:
    # - "max" is available on all adaptive-thinking Claude models.
    # - "xhigh" is only available on Opus 4.7/4.8, Sonnet 5, and Fable 5.
    if any(marker in model_id for marker in ("opus-4-6", "opus-4.6", "sonnet-4-6", "sonnet-4.6")):
        merge_thinking_level_map(model, {"max": "max"})
    if any(marker in model_id for marker in ("opus-4-7", "opus-4.7", "opus-4-8", "opus-4.8", "sonnet-5", "sonnet.5")):
        merge_thinking_level_map(model, {"xhigh": "xhigh", "max": "max"})
    if "fable-5" in model_id:
        merge_thinking_level_map(model, {"off": None, "xhigh": "xhigh", "max": "max"})
    if model["api"] == "anthropic-messages" and is_anthropic_adaptive_thinking_model(model_id):
        merge_compat(model, {"forceAdaptiveThinking": True})
    if model["api"] == "anthropic-messages" and is_anthropic_temperature_unsupported_model(model_id):
        merge_compat(model, {"supportsTemperature": False})


def apply_strict_tool_compat_metadata(model: dict[str, Any]) -> None:
    if model["provider"] == "openai" and model["api"] == "openai-responses":
        merge_compat(model, {"supportsStrictMode": True})
    elif model["provider"] == "anthropic" and model["api"] == "anthropic-messages":
        merge_compat(model, {"supportsStrictTools": True})


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


def apply_openai_explicit_prompt_cache_metadata(model: dict[str, Any]) -> None:
    if model["provider"] != "openai" or model["api"] != "openai-responses":
        return
    if not model["cost"]["cacheWrite"] > 0:
        return
    merge_compat(model, {"supportsExplicitPromptCacheMode": True})


def _base_model(
    model_id: str,
    source: dict[str, Any],
    *,
    api: str,
    provider: str,
    base_url: str,
) -> dict[str, Any]:
    modalities = source.get("modalities") or {}
    cost = source.get("cost") or {}
    limit = source.get("limit") or {}
    return {
        "id": model_id,
        "name": source.get("name") or model_id,
        "api": api,
        "provider": provider,
        "baseUrl": base_url,
        "reasoning": source.get("reasoning") is True,
        "input": ["text", "image"] if "image" in (modalities.get("input") or []) else ["text"],
        "cost": {
            "input": cost.get("input") or 0,
            "output": cost.get("output") or 0,
            "cacheRead": cost.get("cache_read") or 0,
            "cacheWrite": cost.get("cache_write") or 0,
        },
        "contextWindow": limit.get("context") or 4096,
        "maxTokens": limit.get("output") or 4096,
    }


def load_models_dev_data(catalog: dict[str, Any], reasoning_options: dict[str, list]) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []

    def record_reasoning_options(provider: str, model_id: str, source: dict[str, Any]) -> None:
        if source.get("reasoning_options") is not None:
            reasoning_options[f"{provider}:{model_id}"] = source["reasoning_options"]

    for model_id, source in ((catalog.get("anthropic") or {}).get("models") or {}).items():
        if source.get("tool_call") is not True:
            continue
        models.append(
            _base_model(
                model_id,
                source,
                api="anthropic-messages",
                provider="anthropic",
                base_url="https://api.anthropic.com",
            )
        )
        record_reasoning_options("anthropic", model_id, source)

    for model_id, source in ((catalog.get("openai") or {}).get("models") or {}).items():
        if source.get("tool_call") is not True:
            continue
        # models.dev lists this alias, but it is not accepted by OpenAI APIs.
        if model_id in MODELS_DEV_OPENAI_UNSUPPORTED_MODEL_IDS:
            continue
        models.append(
            _base_model(
                model_id,
                source,
                api="openai-responses",
                provider="openai",
                base_url="https://api.openai.com/v1",
            )
        )
        record_reasoning_options("openai", model_id, source)

    return models


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
        "cost": with_openai_long_context_pricing({"input": 2.5, "output": 15, "cacheRead": 0.25, "cacheWrite": 3.125}),
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
        "cost": with_openai_long_context_pricing({"input": 1, "output": 6, "cacheRead": 0.1, "cacheWrite": 1.25}),
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


def apply_overrides(models: list[dict[str, Any]]) -> None:
    """Temporary overrides until upstream model metadata is corrected (pi 2029-2101)."""
    for candidate in models:
        if candidate["provider"] == "anthropic" and candidate["id"] in (
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-opus-4.6",
            "claude-sonnet-4.6",
        ):
            candidate["contextWindow"] = 1000000
        # Keep direct OpenAI requests in the short-context pricing tier by default.
        if candidate["provider"] == "openai" and candidate["id"] in OPENAI_SHORT_CONTEXT_CAPPED_MODEL_IDS:
            candidate["contextWindow"] = OPENAI_LONG_CONTEXT_INPUT_THRESHOLD
            candidate["maxTokens"] = 128000
        if candidate["provider"] == "openai" and candidate["id"] in OPENAI_LONG_CONTEXT_PRICING_MODEL_IDS:
            candidate["cost"] = with_openai_long_context_pricing(candidate["cost"])
        # models.dev reports gpt-5-pro output as 272000 (a duplicate of the
        # input sub-limit); the actual max output is 128000.
        if candidate["provider"] == "openai" and candidate["id"] == "gpt-5-pro":
            candidate["maxTokens"] = 128000


async def fetch_catalog() -> dict[str, Any]:
    print("Fetching models from models.dev API...")
    async with Client() as client:
        response = await client.get("https://models.dev/api.json")
        if response.status_code != 200:
            raise RuntimeError(f"models.dev API returned {response.status_code}")
        return await response.json()


@tonio.main
async def main() -> None:
    catalog = await fetch_catalog()
    reasoning_options: dict[str, list] = {}
    all_models = load_models_dev_data(catalog, reasoning_options)

    apply_overrides(all_models)

    for model in MISSING_OPENAI_MODELS:
        if not any(m["provider"] == model["provider"] and m["id"] == model["id"] for m in all_models):
            all_models.append(dict(model))

    # Metadata passes, in pi's exact order (generate-models.ts:2503-2511) — the
    # order is load-bearing: reasoning-options runs before forceAdaptiveThinking
    # is set, so anthropic models never take models.dev effort maps.
    for model in all_models:
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

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for provider_id in sorted(providers):
        by_api: dict[str, dict[str, Any]] = {}
        provider_models = providers[provider_id]
        for api in sorted({model["api"] for model in provider_models.values()}):
            by_api[api] = {
                model_id: provider_models[model_id]
                for model_id in sorted(provider_models)
                if provider_models[model_id]["api"] == api
            }
        path = DATA_DIR / f"{provider_id}.json"
        path.write_text(json.dumps(by_api, indent=2) + "\n")
        count = len(provider_models)
        print(f"Wrote {count} models to {path}")

    manifest = {
        "generatedAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "https://models.dev/api.json",
        "providers": sorted(providers),
    }
    (DATA_DIR / "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
