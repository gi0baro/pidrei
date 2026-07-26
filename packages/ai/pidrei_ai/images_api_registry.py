"""Port of pi's images-api-registry.ts."""

import threading
from dataclasses import dataclass
from typing import Any

from pidrei_ai.types import ImagesApi, ImagesContext, ImagesModel, ImagesOptions


@dataclass(slots=True)
class ImagesApiProvider:
    api: ImagesApi
    generate_images: Any


@dataclass(slots=True)
class _Registered:
    provider: ImagesApiProvider
    source_id: str | None = None


_registry: dict[str, _Registered] = {}
# pi relies on JavaScript's single thread here; registration can race with a
# lookup from another turn, so the map is guarded.
_registry_guard = threading.Lock()


def _wrap_generate_images(api: ImagesApi, generate_images):
    async def wrapped(model: ImagesModel, context: ImagesContext, options: ImagesOptions | None = None):
        if model.api != api:
            raise ValueError(f"Mismatched api: {model.api} expected {api}")
        return await generate_images(model, context, options)

    return wrapped


def register_images_api_provider(provider: ImagesApiProvider, source_id: str | None = None) -> None:
    entry = _Registered(
        provider=ImagesApiProvider(
            api=provider.api, generate_images=_wrap_generate_images(provider.api, provider.generate_images)
        ),
        source_id=source_id,
    )
    with _registry_guard:
        _registry[provider.api] = entry


def get_images_api_provider(api: ImagesApi) -> ImagesApiProvider | None:
    with _registry_guard:
        entry = _registry.get(api)
    return entry.provider if entry is not None else None
