"""Mirror of pi's opencode-provider-headers.test.ts."""

import pytest

from pidrei_ai.providers.faux import faux_assistant_message
from pidrei_ai.providers.opencode_headers import with_opencode_session_header
from pidrei_ai.types import (
    Context,
    DoneEvent,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    UserMessage,
)
from pidrei_ai.utils.event_stream import AssistantMessageEventStream


MODEL = Model(
    id="test-model",
    name="Test model",
    api="test-api",
    provider="opencode",
    base_url="https://opencode.ai/zen/v1",
    reasoning=False,
    input=["text"],
    cost=ModelCost(),
    context_window=1000,
    max_tokens=100,
)
CONTEXT = Context(messages=[UserMessage(content="hi", timestamp=0)])


def _completed_stream() -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    message = faux_assistant_message("ok")
    stream.push(StartEvent(partial=message))
    stream.push(DoneEvent(reason="stop", message=message))
    stream.end(message)
    return stream


class _RecordingStreams:
    def __init__(self) -> None:
        self.captured: list = []

    def stream(self, _model, _context, options=None):
        self.captured.append(options)
        return _completed_stream()

    def stream_simple(self, _model, _context, options=None):
        self.captured.append(options)
        return _completed_stream()


# Regression test for https://github.com/earendil-works/pi/issues/9326
@pytest.mark.tonio
@pytest.mark.parametrize(
    ("method", "options_type"), [("stream", StreamOptions), ("stream_simple", SimpleStreamOptions)]
)
async def test_maps_session_id_even_without_cache_retention(method, options_type):
    recording = _RecordingStreams()
    streams = with_opencode_session_header(recording)

    getattr(streams, method)(MODEL, CONTEXT, options_type(session_id="conversation-1", cache_retention="none"))

    assert recording.captured[-1].headers == {"x-opencode-session": "conversation-1"}


@pytest.mark.tonio
@pytest.mark.parametrize("value", ["caller-value", None])
async def test_preserves_a_case_insensitive_caller_override(value):
    recording = _RecordingStreams()
    streams = with_opencode_session_header(recording)

    streams.stream_simple(
        MODEL, CONTEXT, SimpleStreamOptions(session_id="generated-value", headers={"X-OpenCode-Session": value})
    )

    assert recording.captured[-1].headers == {"X-OpenCode-Session": value}


@pytest.mark.tonio
async def test_does_not_fabricate_a_session_header_when_session_id_is_absent():
    recording = _RecordingStreams()
    streams = with_opencode_session_header(recording)

    streams.stream_simple(MODEL, CONTEXT, SimpleStreamOptions(headers={"x-custom": "value"}))

    assert recording.captured[-1].headers == {"x-custom": "value"}
