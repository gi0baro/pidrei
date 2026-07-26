"""Mirror of pi's openrouter-images.test.ts.

pi mocks the `openai` package's `chat.completions.create`. Here the client is
`api/openrouter_images._OpenRouterImagesClient`, which the adapter constructs in
`create_client`, so the stub replaces that instead — the same interception point.
"""

import contextlib

import pytest

from pidrei_ai.api import openrouter_images
from pidrei_ai.images import generate_images
from pidrei_ai.types import (
    ImagesContext,
    ImagesModel,
    ImagesOptions,
    ModelCost,
    ProviderResponse,
    TextContent,
)
from pidrei_ai.utils.cancel import CancelToken


last_params: list[dict] = []

RESPONSE = {
    "id": "img-1",
    "usage": {"prompt_tokens": 12, "completion_tokens": 34, "prompt_tokens_details": {"cached_tokens": 0}},
    "choices": [
        {
            "message": {
                "content": "Here is your image.",
                "images": [{"image_url": "data:image/png;base64,ZmFrZS1wbmc="}],
            }
        }
    ],
}


class _FakeClient:
    def __init__(self, *_args, **_kwargs):
        pass

    async def create(self, params, *, timeout_ms=None, cancel=None):
        if cancel is not None and cancel.cancelled:
            raise RuntimeError("Request aborted")
        last_params.append(params)
        return RESPONSE, ProviderResponse(status=200, headers={})


@contextlib.contextmanager
def _stubbed_client():
    original = openrouter_images._OpenRouterImagesClient
    openrouter_images._OpenRouterImagesClient = _FakeClient
    try:
        yield
    finally:
        openrouter_images._OpenRouterImagesClient = original


@pytest.fixture(autouse=True)
def _reset():
    last_params.clear()


def make_model(id: str, output: list[str], headers: dict | None = None) -> ImagesModel:
    return ImagesModel(
        id=id,
        name=id,
        api="openrouter-images",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        input=["text", "image"],
        output=output,
        cost=ModelCost(input=0.015, output=0.03),
        headers=headers,
    )


def make_context() -> ImagesContext:
    return ImagesContext(input=[TextContent(text="Generate a dog")])


@pytest.mark.tonio
async def test_returns_text_plus_images_in_final_output():
    model = make_model(
        "google/gemini-3.1-flash-image-preview",
        ["text", "image"],
        headers={"HTTP-Referer": "https://example.com"},
    )

    with _stubbed_client():
        output = await generate_images(model, make_context(), ImagesOptions(api_key="test"))

    assert output.stop_reason == "stop"
    assert output.response_id == "img-1"
    assert output.output[0].type == "text"
    assert output.output[0].text == "Here is your image."
    assert output.output[1].type == "image"
    assert output.output[1].mime_type == "image/png"
    assert output.output[1].data == "ZmFrZS1wbmc="

    params = last_params[0]
    assert params["stream"] is False
    assert params["modalities"] == ["image", "text"]
    assert params["messages"][0]["content"][0] == {"type": "text", "text": "Generate a dog"}


@pytest.mark.tonio
async def test_passes_through_the_cancel_token_and_returns_an_aborted_result():
    model = make_model("black-forest-labs/flux.2-pro", ["image"])
    cancel = CancelToken()
    cancel.cancel()

    with _stubbed_client():
        output = await generate_images(model, make_context(), ImagesOptions(api_key="test", cancel=cancel))

    # pi asserts the SDK received the aborted AbortSignal; the pidrei client
    # takes the token and refuses before sending, so the stub sees it too.
    assert output.stop_reason == "aborted"
    assert output.error_message == "Request aborted"


@pytest.mark.tonio
async def test_generate_images_resolves_the_final_assistant_images_result():
    model = make_model("black-forest-labs/flux.2-pro", ["image"])

    with _stubbed_client():
        output = await generate_images(model, make_context(), ImagesOptions(api_key="test"))

    assert any(item.type == "image" for item in output.output)


@pytest.mark.tonio
async def test_image_only_models_ask_for_the_image_modality_alone():
    model = make_model("black-forest-labs/flux.2-pro", ["image"])

    with _stubbed_client():
        await generate_images(model, make_context(), ImagesOptions(api_key="test"))

    assert last_params[0]["modalities"] == ["image"]


@pytest.mark.tonio
async def test_a_missing_api_key_is_an_error_result_not_a_raise():
    model = make_model("black-forest-labs/flux.2-pro", ["image"])

    with _stubbed_client():
        output = await generate_images(model, make_context(), ImagesOptions())

    assert output.stop_reason == "error"
    assert "No API key for provider: openrouter" in output.error_message


@pytest.mark.tonio
async def test_a_mismatched_api_is_rejected_by_the_registry():
    model = make_model("black-forest-labs/flux.2-pro", ["image"])
    model.api = "some-other-images-api"

    with pytest.raises(ValueError, match="No API provider registered"):
        await generate_images(model, make_context(), ImagesOptions(api_key="test"))
