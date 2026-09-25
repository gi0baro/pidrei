"""Partial mirror of pi coding-agent test/image-resize-callers.test.ts: the
0.87.1 model-resize-profile cases. The "Image omitted" resize-fallback cases
are not mirrored here.

pi mocks `resizeImage`; here `image_process.resize_image` is swapped on the
module, which is where `process_image` looks it up.
"""

import base64
from types import SimpleNamespace

import pytest

from pidrei.cli.file_processor import process_file_arguments
from pidrei.core.tools.read import create_read_tool_definition
from pidrei.utils import image_process
from pidrei_ai.types import (
    Model,
    ModelCost,
    ModelImageInputLimits,
    ModelImageResizeOptions,
    ModelInputLimits,
)


TINY_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="


@pytest.fixture
def resize_calls(monkeypatch) -> list[tuple]:
    calls: list[tuple] = []

    def resize_image(input_bytes, mime_type, options=None):
        calls.append((input_bytes, mime_type, options))

    monkeypatch.setattr(image_process, "resize_image", resize_image)
    return calls


@pytest.mark.tonio
async def test_passes_the_current_model_resize_profile_to_the_read_tool(tmp_path, resize_calls):
    image_path = tmp_path / "test.png"
    image_path.write_bytes(base64.b64decode(TINY_PNG_BASE64))
    resize = ModelImageResizeOptions(max_width=1234, max_height=1000, max_bytes=500000, jpeg_quality=70)
    model = Model(
        id="vision-model",
        name="Vision model",
        api="test",
        provider="test",
        base_url="https://example.com",
        reasoning=False,
        input=["text", "image"],
        input_limits=ModelInputLimits(images=ModelImageInputLimits(resize=resize)),
        cost=ModelCost(),
        context_window=1000,
        max_tokens=100,
    )
    ctx = SimpleNamespace(cwd=str(tmp_path), model=model)

    await create_read_tool_definition(str(tmp_path)).execute(
        "test-read-model-profile", {"path": str(image_path)}, None, None, ctx
    )

    assert len(resize_calls) == 1
    input_bytes, mime_type, options = resize_calls[0]
    assert isinstance(input_bytes, bytes)
    assert mime_type == "image/png"
    assert options is resize


def test_can_defer_resizing_file_attachments_until_prompt_dispatch(tmp_path, resize_calls):
    image_path = tmp_path / "test.png"
    image_path.write_bytes(base64.b64decode(TINY_PNG_BASE64))

    result = process_file_arguments([str(image_path)], auto_resize_images=False)

    assert len(result.images) == 1
    assert resize_calls == []
