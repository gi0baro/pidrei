"""Cloudflare AI Gateway (pi: catalog/providers/cloudflare-ai-gateway.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, merge_compat, models_dev_models
from catalog.families.openai import (
    OPENAI_STANDARD_COSTS,
    enable_openai_grammar_tools,
    with_openai_long_context_pricing,
)
from catalog.sources.models_dev import ModelsDevModel
from pidrei_ai.api.cloudflare import (
    CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL,
    CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL,
    CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL,
)


def _route_gateway_model(model: ChatModel, _source: ModelsDevModel) -> ChatModel | None:
    """models.dev lists gateway models as `<upstream>/<id>`; route each upstream to its passthrough."""
    slash = model["id"].find("/")
    if slash == -1:
        return None
    upstream = model["id"][:slash]
    native_id = model["id"][slash + 1 :]
    if upstream == "openai":
        # Cloudflare AI Gateway passes OpenAI usage through at OpenAI list prices.
        standard_cost = OPENAI_STANDARD_COSTS.get(native_id)
        return {
            **model,
            "id": native_id,
            "api": "openai-responses",
            "baseUrl": CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL,
            **({"cost": with_openai_long_context_pricing(standard_cost)} if standard_cost is not None else {}),
        }
    if upstream == "anthropic":
        # The /anthropic passthrough forwards the model ID to Anthropic unchanged.
        # models.dev lists dotted versions (claude-opus-5.5), but Anthropic only
        # accepts dashed IDs (claude-opus-5-5).
        return {
            **model,
            "id": native_id.replace(".", "-"),
            "api": "anthropic-messages",
            "baseUrl": CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL,
            # Gateway passthroughs forward session affinity headers to upstreams that
            # use them for cache/routing affinity.
            "compat": {"sendSessionAffinityHeaders": True},
        }
    if upstream == "workers-ai":
        return {**model, "compat": {"sendSessionAffinityHeaders": True}}
    return None


def _models(ctx: ProviderContext) -> list[ChatModel]:
    models = models_dev_models(
        ctx,
        "cloudflare-ai-gateway",
        api="openai-completions",
        base_url=CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL,
        map_model=_route_gateway_model,
    )
    # The gateway proxies Workers AI through its OpenAI-compatible /compat endpoint,
    # but models.dev may omit or intermittently drop those `workers-ai/*` entries
    # from the AI Gateway catalog. Mirror the Workers AI catalog under the documented
    # prefix so the gateway keeps its OpenAI-compatible models stable.
    ids = {model["id"] for model in models}
    mirrored = models_dev_models(
        ctx,
        "cloudflare-workers-ai",
        api="openai-completions",
        base_url=CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL,
        compat={"sendSessionAffinityHeaders": True},
        include=lambda source: f"workers-ai/{source.id}" not in ids,
        map_model=lambda model, _source: {**model, "id": f"workers-ai/{model['id']}"},
    )
    return [*models, *mirrored]


def _finalize(model: ChatModel, _ctx: ProviderContext) -> None:
    if model["api"] == "openai-responses":
        merge_compat(model, {"supportsStrictMode": True})
    enable_openai_grammar_tools(model)


PROVIDER = ProviderDefinition(
    id="cloudflare-ai-gateway",
    models=_models,
    compat={
        "openai-completions": {
            "supportsStore": False,
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
            "maxTokensField": "max_tokens",
            "supportsLongCacheRetention": False,
        },
    },
    finalize=_finalize,
)
