"""Mirror of pi's suite/regressions/copilot-request-size-overflow.test.ts.

pi measures the request with `JSON.stringify(context)`; here the normalized
context is serialized with `dataclasses.asdict` and `json.dumps`, and the
retried request is checked for image blocks directly.
"""

import dataclasses
import json
import time

import pytest

from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call
from pidrei_ai.types import ImageContent, ToolResultMessage, UserMessage

from .harness import Harness, create_harness


COPILOT_413 = 'github-copilot API error (413): {"message":"failed to parse request","code":""}'
REQUEST_LIMIT = 32 * 1024 * 1024
RECENT_PROMPT = "Continue the task using the recent notes."


def _as_model(harness: Harness, message):
    model = harness.get_model()
    return dataclasses.replace(message, provider=model.provider, api=model.api, model=model.id)


async def seed_image_history(harness: Harness, image_data: str) -> None:
    timestamp = int(time.time() * 1000) - 10_000
    session_manager = harness.session_manager
    await session_manager.append_message(UserMessage(content="Review the old images", timestamp=timestamp))
    await session_manager.append_message(
        _as_model(
            harness,
            faux_assistant_message(
                faux_tool_call("read", {"path": "old-images"}, id="read-images"),
                stop_reason="toolUse",
                timestamp=timestamp,
            ),
        )
    )
    await session_manager.append_message(
        ToolResultMessage(
            tool_call_id="read-images",
            tool_name="read",
            content=[ImageContent(mime_type="image/png", data=image_data) for _ in range(89)],
            is_error=False,
            timestamp=timestamp,
        )
    )
    await session_manager.append_message(
        _as_model(harness, faux_assistant_message("Finished reviewing the old images", timestamp=timestamp))
    )
    # Keep a completed recent turn so recovery does not summarize the entire current
    # request when it omits the failed assistant attempt. The old images are outside
    # the default 20k-token retained tail.
    await session_manager.append_message(UserMessage(content=f"Recent task notes: {'x' * 80_000}", timestamp=timestamp))
    await session_manager.append_message(
        _as_model(harness, faux_assistant_message("Ready to continue", timestamp=timestamp))
    )
    harness.session.agent.state.messages = session_manager.build_session_context().messages


def _has_images(context) -> bool:
    return any(
        isinstance(block, ImageContent)
        for message in context.messages
        if not isinstance(message.content, str)
        for block in message.content
    )


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


@pytest.mark.tonio
async def test_compacts_an_image_heavy_history_and_retries_after_the_real_413(harnesses):
    harness = await create_harness(
        provider="github-copilot",
        models=[{"id": "faux-copilot", "context_window": 1_000_000}],
    )
    harnesses.append(harness)
    # Synthetic base64 only; no image decoding or network calls are involved.
    await seed_image_history(harness, "A" * (384 * 1024))
    request_sizes: list[int] = []
    retried_with_images: list[bool] = []

    async def oversized(context, *_rest):
        request_sizes.append(len(json.dumps(dataclasses.asdict(context), default=str).encode()))
        # Raising produces the zero-usage error seen in the affected session.
        raise RuntimeError(COPILOT_413)

    async def summary(*_rest):
        # The overflow recovery summarization runs on the compacted, image-free context.
        return faux_assistant_message("The old images have been reviewed. Continue with the recent task.")

    async def retried(context, *_rest):
        # The retried request no longer carries the images, so it stays under the byte limit.
        retried_with_images.append(_has_images(context))
        return faux_assistant_message("Continued after compaction")

    harness.set_responses([oversized, summary, retried])

    await harness.session.prompt(RECENT_PROMPT)

    assert request_sizes and request_sizes[0] > REQUEST_LIMIT
    assert retried_with_images == [False]
    assert harness.faux.state.call_count == 3
    starts = harness.events_of_type("compaction_start")
    assert [(event.type, event.reason) for event in starts] == [("compaction_start", "overflow")]
    last_end = harness.events_of_type("compaction_end")[-1]
    assert last_end.reason == "overflow"
    assert last_end.aborted is False
    assert last_end.will_retry is True
    # The session continued to a final answer instead of dying on the 413.
    assert harness.session.get_last_assistant_text() == "Continued after compaction"
