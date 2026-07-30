"""Mirror of pi's google-raw-stop-reason.test.ts (stream driver shared with
test_google_stream.py)."""

import pytest

from pidrei_ai.api import google_generative_ai, google_vertex

from .test_google_stream import _chunk, _part, _run


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
