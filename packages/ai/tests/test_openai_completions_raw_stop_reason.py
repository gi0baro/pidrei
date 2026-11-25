"""Mirror of pi's openai-completions-raw-stop-reason.test.ts (SSE driver shared
with test_openai_completions.py)."""

import pytest

from pidrei_ai.utils.retry import is_retryable_assistant_error

from .test_openai_completions import FakeClient, chunk_body, consume, make_model


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
    assert is_retryable_assistant_error(message) is False


# #12206
@pytest.mark.tonio
async def test_retries_a_stream_that_ends_with_finish_reason_error():
    client = FakeClient(
        chunk_body([{"id": "chatcmpl-3", "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}]}])
    )

    message = await consume(client)

    assert message.stop_reason == "error"
    assert message.raw_stop_reason == "error"
    assert message.error_message == "Provider finish_reason: error"
    assert is_retryable_assistant_error(message) is True


# --- OpenRouter native_finish_reason (#12407, #11931) -------------------------

OPEN_ROUTER_MODEL = make_model(
    provider="openrouter", base_url="https://openrouter.ai/api/v1", id="google/gemini-4-neon"
)


def _stop(chunk_id: str, finish_reason: str, native_finish_reason: str | None = None) -> dict:
    choice: dict = {"index": 0, "delta": {}, "finish_reason": finish_reason}
    if native_finish_reason is not None:
        choice["native_finish_reason"] = native_finish_reason
    return {"id": chunk_id, "choices": [choice]}


@pytest.mark.tonio
async def test_records_a_native_reason_hidden_behind_a_successful_finish_reason_without_changing_a_turn_with_text():
    client = FakeClient(
        chunk_body(
            [
                {"id": "gen-1", "choices": [{"index": 0, "delta": {"content": "Done."}, "finish_reason": None}]},
                _stop("gen-1", "stop", "MALFORMED_RESPONSE"),
            ]
        )
    )

    message = await consume(client, OPEN_ROUTER_MODEL)

    assert message.stop_reason == "stop"
    assert message.raw_stop_reason == "stop.MALFORMED_RESPONSE"
    assert message.error_message is None


@pytest.mark.tonio
async def test_retries_a_stop_after_reasoning_only_when_the_native_reason_is_not_a_clean_stop():
    client = FakeClient(
        chunk_body(
            [
                {
                    "id": "gen-8",
                    "choices": [{"index": 0, "delta": {"reasoning": "Let me think."}, "finish_reason": None}],
                },
                _stop("gen-8", "stop", "MALFORMED_RESPONSE"),
            ]
        )
    )

    message = await consume(client, OPEN_ROUTER_MODEL)

    assert message.stop_reason == "error"
    assert message.raw_stop_reason == "stop.MALFORMED_RESPONSE"
    assert message.error_message == "OpenRouter returned an empty response (native_finish_reason: MALFORMED_RESPONSE)"
    assert is_retryable_assistant_error(message) is True


@pytest.mark.tonio
async def test_retries_an_empty_stop_whose_native_reason_openrouter_counts_as_clean_but_its_provider_documents_as_a_failure():
    client = FakeClient(
        chunk_body(
            [
                {"id": "gen-9", "choices": [{"index": 0, "delta": {"content": "  "}, "finish_reason": None}]},
                _stop("gen-9", "stop", "OTHER"),
            ]
        )
    )

    message = await consume(client, OPEN_ROUTER_MODEL)

    assert message.stop_reason == "error"
    assert is_retryable_assistant_error(message) is True


@pytest.mark.tonio
async def test_keeps_an_empty_stop_whose_native_reason_is_a_clean_stop():
    client = FakeClient(chunk_body([_stop("gen-10", "stop", "end_turn")]))

    message = await consume(client, OPEN_ROUTER_MODEL)

    assert message.stop_reason == "stop"
    assert message.raw_stop_reason == "stop.end_turn"
    assert message.error_message is None


@pytest.mark.tonio
async def test_matches_clean_stop_reasons_case_insensitively():
    client = FakeClient(chunk_body([_stop("gen-11", "stop", "End_Turn")]))

    message = await consume(client, OPEN_ROUTER_MODEL)

    assert message.stop_reason == "stop"
    assert message.error_message is None


@pytest.mark.tonio
async def test_records_the_native_reason_of_a_tool_call():
    client = FakeClient(
        chunk_body(
            [
                {
                    "id": "gen-2",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "id": "call_1", "function": {"name": "read", "arguments": "{}"}}
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                # Gemini reports STOP for tool calls; OpenRouter's finish_reason carries the tool call.
                _stop("gen-2", "tool_calls", "STOP"),
            ]
        )
    )

    message = await consume(client, OPEN_ROUTER_MODEL)

    assert message.stop_reason == "toolUse"
    assert message.raw_stop_reason == "tool_calls.STOP"


@pytest.mark.tonio
async def test_adds_the_native_reason_to_an_error_message_and_keeps_it_retryable():
    client = FakeClient(chunk_body([_stop("gen-3", "error", "MALFORMED_FUNCTION_CALL")]))

    message = await consume(client, OPEN_ROUTER_MODEL)

    assert message.stop_reason == "error"
    assert message.raw_stop_reason == "error.MALFORMED_FUNCTION_CALL"
    assert message.error_message == "Provider finish_reason: error (native_finish_reason: MALFORMED_FUNCTION_CALL)"
    assert is_retryable_assistant_error(message) is True


@pytest.mark.tonio
async def test_keeps_the_native_reason_when_a_trailing_chunk_repeats_finish_reason_without_it():
    client = FakeClient(chunk_body([_stop("gen-4", "error", "MALFORMED_FUNCTION_CALL"), _stop("gen-4", "error")]))

    message = await consume(client, OPEN_ROUTER_MODEL)

    assert message.raw_stop_reason == "error.MALFORMED_FUNCTION_CALL"
    assert message.error_message == "Provider finish_reason: error (native_finish_reason: MALFORMED_FUNCTION_CALL)"


@pytest.mark.tonio
async def test_does_not_repeat_a_native_reason_equal_to_finish_reason():
    client = FakeClient(chunk_body([_stop("gen-5", "stop", "stop")]))

    message = await consume(client, OPEN_ROUTER_MODEL)

    assert message.raw_stop_reason == "stop"


@pytest.mark.tonio
async def test_ignores_native_finish_reason_from_providers_other_than_openrouter():
    client = FakeClient(chunk_body([_stop("chatcmpl-6", "stop", "MALFORMED_RESPONSE")]))

    message = await consume(client)

    assert message.raw_stop_reason == "stop"
