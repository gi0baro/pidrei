"""Partial mirror of pi's suite/agent-session-prompt.test.ts: the 0.87.0
image-normalization case. The rest of the prompt characterization suite is
not mirrored here.

pi mocks `processImage`; here `agent_session.process_image` is swapped, which
is the name the prompt path calls.
"""

import base64

import pytest

from pidrei.core import agent_session as agent_session_module
from pidrei.core.agent_session import PromptOptions
from pidrei.utils.image_process import ProcessImageResult
from pidrei_ai.providers.faux import faux_assistant_message
from pidrei_ai.types import (
    ImageContent,
    ModelImageInputLimits,
    ModelImageResizeOptions,
    ModelInputLimits,
)

from .harness import create_harness


TINY_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
NORMALIZED_BASE64 = base64.b64encode(b"normalized").decode("ascii")


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


@pytest.fixture
def process_image_calls(monkeypatch) -> list[tuple]:
    calls: list[tuple] = []

    def process_image(data, mime_type, **options):
        calls.append((data, mime_type, options))
        return ProcessImageResult(ok=True, data=NORMALIZED_BASE64, mime_type=mime_type, hints=[])

    monkeypatch.setattr(agent_session_module, "process_image", process_image)
    return calls


# Regression test for https://github.com/earendil-works/pi/issues/9631
@pytest.mark.tonio
async def test_uses_the_model_selected_by_before_agent_start_for_image_normalization(harnesses, process_image_calls):
    strict_model = None

    def factory(pi) -> None:
        async def on_before_agent_start(_event, _ctx):
            if strict_model is None:
                raise Exception("Expected strict model")
            await pi.set_model(strict_model)

        pi.on("before_agent_start", on_before_agent_start)

    harness = await create_harness(models=[{"id": "wide"}, {"id": "strict"}], extension_factories=[factory])
    harnesses.append(harness)
    strict_model = harness.get_model("strict")
    resize_options = ModelImageResizeOptions(max_width=1000, max_height=1000, max_bytes=500000, jpeg_quality=70)
    strict_model.input_limits = ModelInputLimits(images=ModelImageInputLimits(resize=resize_options))
    harness.set_responses([faux_assistant_message("done")])

    await harness.session.prompt(
        "inspect", PromptOptions(images=[ImageContent(data=TINY_PNG_BASE64, mime_type="image/png")])
    )

    assert harness.session.model.id == "strict"
    assert len(process_image_calls) == 1
    data, mime_type, options = process_image_calls[0]
    assert isinstance(data, bytes)
    assert mime_type == "image/png"
    assert options == {"auto_resize_images": True, "resize_options": resize_options}
    user_message = next(message for message in harness.session.messages if message.role == "user")
    assert ImageContent(data=NORMALIZED_BASE64, mime_type="image/png") in user_message.content
