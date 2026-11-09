"""NVIDIA NIM (pi: catalog/providers/nvidia.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, models_dev_models
from catalog.sources import NVIDIA_BASE_URL, normalize_nvidia_model_id
from catalog.sources.models_dev import ModelsDevModel


# Listed by NIM but unusable through PiDrei (no tool calls, broken streaming, or retired).
_UNSUPPORTED_MODEL_IDS = frozenset(
    {
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
)


def _models(ctx: ProviderContext) -> list[ChatModel]:
    live_ids = ctx.sources.nvidia_model_ids

    def live_id(model_id: str) -> str | None:
        found = live_ids.get(model_id)
        return live_ids.get(normalize_nvidia_model_id(model_id)) if found is None else found

    # Only models the live NIM API serves, under the id it uses.
    def include(source: ModelsDevModel) -> bool:
        if not source.text_input or not source.text_output:
            return False
        model_id = live_id(source.id)
        return model_id is not None and model_id not in _UNSUPPORTED_MODEL_IDS

    def map_model(model: ChatModel, _source: ModelsDevModel) -> ChatModel:
        found = live_id(model["id"])
        model_id = model["id"] if found is None else found
        return {**model, "id": model_id, "name": model_id if model["name"] == model["id"] else model["name"]}

    return models_dev_models(
        ctx,
        "nvidia",
        api="openai-completions",
        base_url=NVIDIA_BASE_URL,
        headers={"NVCF-POLL-SECONDS": "3600"},
        compat={
            "supportsStore": False,
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
            "maxTokensField": "max_tokens",
            "supportsStrictMode": False,
            "supportsLongCacheRetention": False,
        },
        include=include,
        map_model=map_model,
    )


PROVIDER = ProviderDefinition(
    id="nvidia",
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
)
