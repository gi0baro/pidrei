"""Amazon Bedrock (pi: catalog/providers/amazon-bedrock.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, models_dev_models, with_input_limits
from catalog.sources.models_dev import ModelsDevModel


# Opus 5 is only callable through cross-region inference profiles.
_INFERENCE_PROFILE_ONLY_MODEL_IDS = frozenset({"anthropic.claude-opus-5"})


def _bedrock_base_url(model_id: str) -> str:
    return (
        "https://bedrock-runtime.eu-central-1.amazonaws.com"
        if model_id.startswith("eu.")
        else "https://bedrock-runtime.us-east-1.amazonaws.com"
    )


def _include(source: ModelsDevModel) -> bool:
    return (
        source.id not in _INFERENCE_PROFILE_ONLY_MODEL_IDS
        # Jamba does not support tool use in streaming mode.
        and not source.id.startswith("ai21.jamba")
        # Mistral 7B Instruct does not support system messages.
        and not source.id.startswith("mistral.mistral-7b-instruct-v0")
    )


# Includes models.dev pricing tiers, e.g. the long-context tier for OpenAI models (#10326).
def _map(model: ChatModel, source: ModelsDevModel) -> ChatModel:
    return {
        **model,
        "baseUrl": _bedrock_base_url(model["id"]),
        **({"compat": {"supportsStrictMode": True}} if source.structured_output else {}),
    }


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return models_dev_models(
        ctx,
        "amazon-bedrock",
        api="bedrock-converse-stream",
        base_url="",
        include=_include,
        map_model=_map,
    )


def _finalize(model: ChatModel, _ctx: ProviderContext) -> None:
    with_input_limits(model, {"images": {"maxPerMessage": 20}})


PROVIDER = ProviderDefinition(id="amazon-bedrock", models=_models, finalize=_finalize)
