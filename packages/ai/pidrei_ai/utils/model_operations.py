"""Port of pi's model operations (packages/ai/src/utils/model-operations.ts)."""

from pidrei_ai.types import AnyModel, AssistantImages, ClassifierModel, ClassifierResult, ImageModel, ModelType
from pidrei_ai.utils.models_error import ModelsError
from pidrei_utils import clock


def get_model_type(model: AnyModel) -> ModelType:
    """The type of a model. Models without `type` are chat models."""
    return model.type if model.type is not None else "chat"


def is_model_type(model: AnyModel, type: ModelType) -> bool:
    """Runtime-checked model type test, including legacy chat models without `type`."""
    return get_model_type(model) == type


def assert_chat_model(model: AnyModel) -> None:
    if not is_model_type(model, "chat"):
        raise ModelsError("provider", f"Model {model.provider}/{model.id} is not a chat model")


def assert_image_model(model: AnyModel) -> None:
    if not is_model_type(model, "image"):
        raise ModelsError("provider", f"Model {model.provider}/{model.id} is not an image model")


def assert_classifier_model(model: AnyModel) -> None:
    if not is_model_type(model, "classifier"):
        raise ModelsError("provider", f"Model {model.provider}/{model.id} is not a classifier model")


def image_error_result(model: ImageModel, error: BaseException | object, aborted: bool = False) -> AssistantImages:
    return AssistantImages(
        api=model.api,
        provider=model.provider,
        model=model.id,
        output=[],
        stop_reason="aborted" if aborted else "error",
        error_message=str(error),
        timestamp=clock.now_ms(),
    )


def classifier_error_result(
    model: ClassifierModel, error: BaseException | object, aborted: bool = False
) -> ClassifierResult:
    return ClassifierResult(
        api=model.api,
        provider=model.provider,
        model=model.id,
        answers={},
        stop_reason="aborted" if aborted else "error",
        error_message=str(error),
        timestamp=clock.now_ms(),
    )
