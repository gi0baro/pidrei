"""Mirror of pi's openai-completions-raw-stop-reason.test.ts (SSE driver shared
with test_openai_completions.py)."""

import pytest

from .test_openai_completions import FakeClient, chunk_body, consume


@pytest.mark.tonio
async def test_preserves_raw_finish_reasons_for_successful_stops():
    client = FakeClient(
        chunk_body([{"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}])
    )

    message = await consume(client)

    assert message.stop_reason == "stop"
    assert message.raw_stop_reason == "stop"
    assert message.error_message is None


@pytest.mark.tonio
async def test_preserves_raw_finish_reasons_for_provider_error_stops():
    client = FakeClient(
        chunk_body([{"id": "chatcmpl-2", "choices": [{"index": 0, "delta": {}, "finish_reason": "content_filter"}]}])
    )

    message = await consume(client)

    assert message.stop_reason == "error"
    assert message.raw_stop_reason == "content_filter"
    assert message.error_message == "Provider finish_reason: content_filter"
