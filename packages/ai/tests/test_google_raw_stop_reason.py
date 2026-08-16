"""Mirror of pi's google-raw-stop-reason.test.ts (stream driver shared with
test_google_stream.py)."""

import pytest

from pidrei_ai.api import google_generative_ai, google_vertex

from .test_google_stream import ADAPTER_IDS, ADAPTERS, _chunk, _part, _run


_FUNCTION_CALL_PART = {"functionCall": {"id": "call-1", "name": "echo", "args": {"value": "truncated"}}}


@pytest.mark.tonio
async def test_preserves_raw_gemini_finish_reasons_for_google_generative_ai_errors():
    _events, message = await _run(
        google_generative_ai,
        [_chunk([_part("hi")], finish_reason="MALFORMED_FUNCTION_CALL")],
    )

    assert message.stop_reason == "error"
    assert message.raw_stop_reason == "MALFORMED_FUNCTION_CALL"
    assert message.error_message == "Provider stopped with: MALFORMED_FUNCTION_CALL"


@pytest.mark.tonio
async def test_preserves_raw_gemini_finish_reasons_for_google_vertex_errors():
    _events, message = await _run(
        google_vertex,
        [_chunk([_part("hi")], finish_reason="SAFETY")],
    )

    assert message.stop_reason == "error"
    assert message.raw_stop_reason == "SAFETY"
    assert message.error_message == "Provider stopped with: SAFETY"


@pytest.mark.tonio
@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
async def test_preserves_max_tokens_with_a_tool_call_as_length(adapter):
    _events, message = await _run(
        adapter,
        [_chunk([_FUNCTION_CALL_PART], finish_reason="MAX_TOKENS")],
    )

    assert message.stop_reason == "length"
    assert message.raw_stop_reason == "MAX_TOKENS"
    assert any(block.type == "toolCall" for block in message.content)


@pytest.mark.tonio
@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
async def test_maps_stop_with_a_tool_call_to_tool_use(adapter):
    _events, message = await _run(
        adapter,
        [_chunk([_FUNCTION_CALL_PART], finish_reason="STOP")],
    )

    assert message.stop_reason == "toolUse"
    assert message.raw_stop_reason == "STOP"
    assert any(block.type == "toolCall" for block in message.content)
