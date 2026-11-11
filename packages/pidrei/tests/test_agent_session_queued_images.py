"""Mirror of pi's agent-session-queued-images.test.ts.

pi delays the first response by 300 ms and polls `isStreaming`; here the first
response waits on an Event the test sets once the messages are queued. The PNG is
built with Pillow (as in `test_agent_session_tool_result_images.py`), and the
responses record the contexts the model was sent, for pi's `faux.contexts`.
"""

import base64
import io

import pytest
import tonio.colored as tonio
from PIL import Image

from pidrei.core.agent_session import PromptOptions
from pidrei_ai.providers.faux import faux_assistant_message
from pidrei_ai.types import ImageContent

from .harness import create_harness


def _create_png(width: int, height: int) -> str:
    """A solid gray RGB PNG of the given size, base64 encoded."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (0x80, 0x80, 0x80)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


TALL_PNG_BASE64 = _create_png(1200, 9000)


def _persisted_user_content(harness, text: str) -> list:
    for entry in harness.session_manager.get_entries():
        if entry["type"] != "message" or entry["message"].role != "user":
            continue
        content = entry["message"].content
        if isinstance(content, str):
            continue
        if content and content[0].type == "text" and content[0].text.split("\n")[0] == text:
            return content
    raise AssertionError(f'No user message "{text}"')


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


# Regression test for #11738: queued messages skipped the image resize idle prompts get.
@pytest.mark.tonio
async def test_resizes_images_in_steering_follow_up_and_streaming_prompt_messages(harnesses):
    harness = await create_harness()
    harnesses.append(harness)
    contexts: list[list] = []
    first_started = tonio.Event()
    release_first = tonio.Event()

    async def first(context, *_rest):
        contexts.append(list(context.messages))
        first_started.set()
        await release_first.wait(5)
        return faux_assistant_message("first")

    def respond(text: str):
        async def step(context, *_rest):
            contexts.append(list(context.messages))
            return faux_assistant_message(text)

        return step

    harness.set_responses([first, respond("second"), respond("third"), respond("fourth")])
    image = ImageContent(data=TALL_PNG_BASE64, mime_type="image/png")

    run = tonio.spawn(harness.session.prompt("idle"))
    await first_started.wait(5)
    assert first_started.is_set(), "the first response never started"
    try:
        await harness.session.steer("steer", [image])
        await harness.session.follow_up("follow up", [image])
        await harness.session.prompt("streaming prompt", PromptOptions(images=[image], streaming_behavior="followUp"))
    finally:
        release_first.set()
    await run

    assert len(contexts) == 4
    for text in ["steer", "follow up", "streaming prompt"]:
        content = _persisted_user_content(harness, text)
        images = [block for block in content if block.type == "image"]
        assert len(images) == 1
        assert images[0].data != image.data
        assert "original 1200x9000, displayed at 267x2000" in content[0].text
    sent_images = [
        block
        for messages in contexts
        for message in messages
        if not isinstance(message.content, str)
        for block in message.content
        if block.type == "image"
    ]
    assert len(sent_images) > 0
    assert not any(block.data == image.data for block in sent_images)
    assert harness.session.pending_message_count == 0
