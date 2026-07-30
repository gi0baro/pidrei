"""pidrei-only: the Gemini/Vertex streaming state machines.

pi exercises these only through credential-gated E2E cases, so nothing in its
suite covers the block bookkeeping — which is how a missing `ThinkingStartEvent`
import survived the first port of this slice: every mirrored test streamed text
only. Both adapters carry their own copy of the loop (pi duplicates it too), so
every case runs against both.
"""

import contextlib

import pytest

from pidrei_ai.api import google_generative_ai, google_vertex
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, SimpleStreamOptions, StreamOptions, UserMessage


ADAPTERS = [google_generative_ai, google_vertex]
ADAPTER_IDS = ["gemini", "vertex"]

CONTEXT = Context(messages=[UserMessage(content="hi", timestamp=1)])


def _model(adapter):
    provider = "google" if adapter is google_generative_ai else "google-vertex"
    return get_builtin_model(provider, "gemini-3-pro-preview" if provider == "google" else "gemini-3.1-pro-preview")


@contextlib.contextmanager
def _streaming(adapter, chunks, *, raises: Exception | None = None):
    class _Fake:
        def __init__(self, _config):
            pass

        async def generate_content_stream(self, _params, *, env=None, cancel=None):
            for chunk in chunks:
                yield chunk
            if raises is not None:
                raise raises

    original = adapter.GoogleGenAI
    adapter.GoogleGenAI = _Fake
    try:
        yield
    finally:
        adapter.GoogleGenAI = original


async def _run(adapter, chunks, *, raises=None, options=None):
    """Returns (events, result message)."""
    with _streaming(adapter, chunks, raises=raises):
        stream = adapter.stream(_model(adapter), CONTEXT, options or StreamOptions(api_key="test-key"))
        events = [event async for event in stream]
        return events, await stream.result()


def _part(text, **extra):
    return {"text": text, **extra}


def _chunk(parts=None, *, finish_reason=None, usage=None, response_id=None):
    candidate: dict = {}
    if parts is not None:
        candidate["content"] = {"parts": parts}
    if finish_reason is not None:
        candidate["finishReason"] = finish_reason
    chunk: dict = {"candidates": [candidate]}
    if usage is not None:
        chunk["usageMetadata"] = usage
    if response_id is not None:
        chunk["responseId"] = response_id
    return chunk


def _types(events):
    return [event.type for event in events]


@pytest.mark.tonio
@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
async def test_text_deltas_become_one_text_block(adapter):
    events, result = await _run(adapter, [_chunk([_part("Hel")]), _chunk([_part("lo")], finish_reason="STOP")])

    assert _types(events) == ["start", "text_start", "text_delta", "text_delta", "text_end", "done"]
    assert [b.text for b in result.content] == ["Hello"]
    assert result.stop_reason == "stop"


@pytest.mark.tonio
@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
async def test_a_thought_part_becomes_a_thinking_block(adapter):
    events, result = await _run(
        adapter,
        [
            _chunk([_part("weighing", thought=True, thoughtSignature="c2ln")]),
            _chunk([_part(" options", thought=True)], finish_reason="STOP"),
        ],
    )

    assert _types(events) == [
        "start",
        "thinking_start",
        "thinking_delta",
        "thinking_delta",
        "thinking_end",
        "done",
    ]
    assert result.content[0].type == "thinking"
    assert result.content[0].thinking == "weighing options"
    # The signature arrived on the first delta only; it must not be dropped.
    assert result.content[0].thinking_signature == "c2ln"


@pytest.mark.tonio
@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
async def test_switching_from_thinking_to_text_closes_the_open_block(adapter):
    events, result = await _run(
        adapter,
        [_chunk([_part("think", thought=True), _part("answer")], finish_reason="STOP")],
    )

    assert _types(events) == [
        "start",
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "text_start",
        "text_delta",
        "text_end",
        "done",
    ]
    assert [b.type for b in result.content] == ["thinking", "text"]


@pytest.mark.tonio
@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
async def test_a_signature_on_a_text_part_lands_on_the_text_block(adapter):
    _events, result = await _run(adapter, [_chunk([_part("hi", thoughtSignature="c2ln")], finish_reason="STOP")])

    # `thoughtSignature` can ride on any part; it does not make the part thinking.
    assert result.content[0].type == "text"
    assert result.content[0].text_signature == "c2ln"


@pytest.mark.tonio
@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
async def test_a_function_call_becomes_a_tool_call_and_forces_tool_use(adapter):
    events, result = await _run(
        adapter,
        [
            _chunk(
                [{"functionCall": {"id": "call_1", "name": "bash", "args": {"command": "ls"}}}],
                finish_reason="STOP",
            )
        ],
    )

    assert _types(events) == ["start", "toolcall_start", "toolcall_delta", "toolcall_end", "done"]
    tool_call = result.content[0]
    assert (tool_call.id, tool_call.name, tool_call.arguments) == ("call_1", "bash", {"command": "ls"})
    # A tool call overrides the reported finish reason.
    assert result.stop_reason == "toolUse"
    assert events[2].delta == '{"command":"ls"}'


@pytest.mark.tonio
@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
async def test_an_open_text_block_is_closed_before_a_tool_call(adapter):
    events, _result = await _run(
        adapter,
        [_chunk([_part("about to call"), {"functionCall": {"name": "bash", "args": {}}}], finish_reason="STOP")],
    )

    assert _types(events) == [
        "start",
        "text_start",
        "text_delta",
        "text_end",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]


@pytest.mark.tonio
@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
async def test_missing_and_duplicate_tool_call_ids_are_replaced(adapter):
    _events, result = await _run(
        adapter,
        [
            _chunk(
                [
                    {"functionCall": {"name": "bash", "args": {}}},
                    {"functionCall": {"id": "dup", "name": "read", "args": {}}},
                    {"functionCall": {"id": "dup", "name": "read", "args": {}}},
                ],
                finish_reason="STOP",
            )
        ],
    )

    ids = [block.id for block in result.content]
    assert len(set(ids)) == 3
    assert ids[0].startswith("bash_")
    assert ids[1] == "dup"
    assert ids[2].startswith("read_")


@pytest.mark.tonio
@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
async def test_a_thought_signature_is_carried_onto_the_tool_call(adapter):
    _events, result = await _run(
        adapter,
        [
            _chunk(
                [{"functionCall": {"name": "bash", "args": {}}, "thoughtSignature": "c2ln"}],
                finish_reason="STOP",
            )
        ],
    )

    assert result.content[0].thought_signature == "c2ln"


@pytest.mark.tonio
@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
async def test_usage_metadata_splits_cached_and_thinking_tokens(adapter):
    _events, result = await _run(
        adapter,
        [
            _chunk(
                [_part("hi")],
                finish_reason="STOP",
                usage={
                    "promptTokenCount": 100,
                    "cachedContentTokenCount": 40,
                    "candidatesTokenCount": 10,
                    "thoughtsTokenCount": 5,
                    "totalTokenCount": 115,
                },
            )
        ],
    )

    assert result.usage.input == 60  # prompt minus cached
    assert result.usage.output == 15  # candidates plus thoughts
    assert result.usage.cache_read == 40
    assert result.usage.reasoning == 5
    assert result.usage.total_tokens == 115


@pytest.mark.tonio
@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
async def test_max_tokens_maps_to_length(adapter):
    _events, result = await _run(adapter, [_chunk([_part("cut off")], finish_reason="MAX_TOKENS")])

    assert result.stop_reason == "length"


@pytest.mark.tonio
@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
async def test_a_safety_finish_reason_ends_the_stream_in_an_error(adapter):
    events, result = await _run(adapter, [_chunk([], finish_reason="SAFETY")])

    assert _types(events)[-1] == "error"
    assert result.stop_reason == "error"
    assert result.error_message == "Provider stopped with: SAFETY"


@pytest.mark.tonio
@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
async def test_an_unrecognized_finish_reason_is_an_error_not_a_silent_stop(adapter):
    events, result = await _run(adapter, [_chunk([], finish_reason="SOMETHING_NEW")])

    assert _types(events)[-1] == "error"
    assert "Unhandled stop reason: SOMETHING_NEW" in result.error_message


@pytest.mark.tonio
@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
async def test_the_first_non_empty_response_id_is_kept(adapter):
    _events, result = await _run(
        adapter,
        [
            _chunk([_part("a")]),
            _chunk([_part("b")], response_id="first"),
            _chunk([_part("c")], finish_reason="STOP", response_id="second"),
        ],
    )

    assert result.response_id == "first"


@pytest.mark.tonio
@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
async def test_a_transport_failure_mid_stream_becomes_an_error_event(adapter):
    events, result = await _run(adapter, [_chunk([_part("partial")])], raises=RuntimeError("connection reset"))

    assert _types(events)[-1] == "error"
    assert result.stop_reason == "error"
    assert "connection reset" in result.error_message
    # The text seen before the failure stays on the partial message.
    assert result.content[0].text == "partial"


@pytest.mark.tonio
@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
async def test_a_trailing_open_block_is_closed_when_the_stream_ends(adapter):
    events, _result = await _run(adapter, [_chunk([_part("no finish reason chunk")], finish_reason="STOP")])

    assert _types(events)[-2:] == ["text_end", "done"]


@pytest.mark.tonio
async def test_the_gemini_adapter_requires_an_api_key():
    events, result = await _run(
        google_generative_ai, [_chunk([_part("hi")], finish_reason="STOP")], options=StreamOptions()
    )

    assert _types(events) == ["error"]
    assert result.error_message == "No API key for provider: google"


@pytest.mark.tonio
async def test_the_gemini_stream_simple_requires_an_api_key():
    with pytest.raises(RuntimeError, match="No API key for provider: google"):
        google_generative_ai.stream_simple(_model(google_generative_ai), CONTEXT, SimpleStreamOptions())


@pytest.mark.tonio
async def test_vertex_without_a_key_or_project_reports_what_is_missing():
    events, result = await _run(
        google_vertex, [_chunk([_part("hi")], finish_reason="STOP")], options=StreamOptions(env={})
    )

    assert _types(events) == ["error"]
    assert "Vertex AI requires a project ID" in result.error_message


@pytest.mark.tonio
async def test_vertex_with_a_project_but_no_location_reports_the_location():
    events, result = await _run(
        google_vertex,
        [_chunk([_part("hi")], finish_reason="STOP")],
        options=google_vertex.GoogleVertexOptions(project="p", env={}),
    )

    assert _types(events) == ["error"]
    assert "Vertex AI requires a location" in result.error_message
