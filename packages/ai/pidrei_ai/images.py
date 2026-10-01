"""Port of pi's images.ts: the top-level `generate_images` entry point."""

from pidrei_ai.images_api_registry import get_images_api_provider
from pidrei_ai.providers.images import register_builtins as _register_builtins  # noqa: F401 - registers on import
from pidrei_ai.types import AssistantImages, ImageApi, ImageModel, ImagesContext, ImagesOptions


def _resolve_images_api_provider(api: ImageApi):
    provider = get_images_api_provider(api)
    if provider is None:
        raise ValueError(f"No API provider registered for api: {api}")
    return provider


async def generate_images(
    model: ImageModel, context: ImagesContext, options: ImagesOptions | None = None
) -> AssistantImages:
    """Global image generation dispatched on `model.api` through the images api
    registry. Auth must be passed explicitly via `options.api_key`; prefer
    `Models.generate_images()`, which resolves provider auth."""
    provider = _resolve_images_api_provider(model.api)
    return await provider.generate_images(model, context, options)
