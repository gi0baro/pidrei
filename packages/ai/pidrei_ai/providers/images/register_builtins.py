"""Port of pi's providers/images/register-builtins.ts.

Registers the built-in images APIs at import time. The adapter module itself
loads lazily on first use, and a load failure becomes an error `AssistantImages`
rather than an exception, mirroring `api/lazy.py` for the streaming side.
"""

import time

from pidrei_ai.images_api_registry import ImagesApiProvider, register_images_api_provider
from pidrei_ai.types import AssistantImages, ImagesContext, ImagesModel, ImagesOptions


def _create_lazy_load_error_images(model: ImagesModel, error: Exception) -> AssistantImages:
    return AssistantImages(
        api=model.api,
        provider=model.provider,
        model=model.id,
        output=[],
        stop_reason="error",
        error_message=str(error),
        timestamp=int(time.time() * 1000),
    )


async def generate_images_openrouter(
    model: ImagesModel, context: ImagesContext, options: ImagesOptions | None = None
) -> AssistantImages:
    try:
        # lazy: api adapters load on demand (see api/*_lazy.py)
        from pidrei_ai.api import openrouter_images

        return await openrouter_images.generate_images(model, context, options)
    except Exception as error:
        return _create_lazy_load_error_images(model, error)


def register_built_in_images_api_providers() -> None:
    register_images_api_provider(ImagesApiProvider(api="openrouter-images", generate_images=generate_images_openrouter))


register_built_in_images_api_providers()
