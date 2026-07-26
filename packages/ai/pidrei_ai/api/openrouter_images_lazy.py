"""Port of pi's openrouter-images.lazy.ts."""

from pidrei_ai.providers.images.register_builtins import generate_images_openrouter
from pidrei_ai.types import ProviderImages


class LazyOpenRouterImagesApi(ProviderImages):
    async def generate_images(self, model, context, options=None):
        return await generate_images_openrouter(model, context, options)


def openrouter_images_api() -> LazyOpenRouterImagesApi:
    return LazyOpenRouterImagesApi()
