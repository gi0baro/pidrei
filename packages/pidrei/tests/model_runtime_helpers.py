"""Mirror of pi coding-agent test/model-runtime-test-utils.ts."""

from pidrei.core.model_registry import ModelRegistry
from pidrei.core.model_runtime import ModelRuntime
from pidrei.core.models_store import InMemoryCodingAgentModelsStore
from pidrei_ai.types import Model, ModelCost


async def create_model_registry(credentials, models_path: str) -> ModelRegistry:
    """Load optional models.json configuration without introducing file-backed
    catalog locks into unit tests."""
    runtime = await ModelRuntime.create(
        credentials=credentials,
        models_path=models_path,
        models_store=InMemoryCodingAgentModelsStore(),
        allow_model_network=False,
    )
    return ModelRegistry(runtime)


async def create_in_memory_model_registry(credentials) -> ModelRegistry:
    runtime = await ModelRuntime.create(credentials=credentials, models_path=None, allow_model_network=False)
    return ModelRegistry(runtime)


def get_model_runtime(model_registry: ModelRegistry) -> ModelRuntime:
    return model_registry._runtime


def make_model(
    provider: str,
    id: str,
    *,
    api: str = "openai-completions",
    base_url: str = "https://example.test/v1",
    name: str | None = None,
    reasoning: bool = False,
    input: list | None = None,
    cost: ModelCost | None = None,
    context_window: int = 1000,
    max_tokens: int = 100,
    **overrides,
) -> Model:
    return Model(
        id=id,
        name=name if name is not None else id,
        api=api,
        provider=provider,
        base_url=base_url,
        reasoning=reasoning,
        input=input if input is not None else ["text"],
        cost=cost if cost is not None else ModelCost(0, 0, 0, 0),
        context_window=context_window,
        max_tokens=max_tokens,
        **overrides,
    )
