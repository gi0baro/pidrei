"""Port of pi's image-models.ts: the static image-model registry."""

from pidrei_ai.image_models_generated import IMAGE_MODELS
from pidrei_ai.types import ImagesModel


def get_image_model(provider: str, model_id: str) -> ImagesModel | None:
    return IMAGE_MODELS.get(provider, {}).get(model_id)


def get_image_providers() -> list[str]:
    return list(IMAGE_MODELS.keys())


def get_image_models(provider: str) -> list[ImagesModel]:
    return list(IMAGE_MODELS.get(provider, {}).values())
