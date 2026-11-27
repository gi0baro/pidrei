"""Mirror of pi's retry.test.ts."""

from dataclasses import replace

import pytest
import tonio.colored as tonio

from pidrei_ai.providers.faux import faux_assistant_message
from pidrei_ai.types import TextContent
from pidrei_ai.utils.diagnostics import NETWORK_ERROR_DIAGNOSTIC, create_assistant_message_diagnostic
from pidrei_ai.utils.retry import (
    RetryCallbacks,
    RetryPolicy,
    is_retryable_assistant_error,
    retry_assistant_call,
    retry_delay_ms,
)
from pidrei_http import http
from pidrei_utils.cancel import CancelToken


OPENAI_EXPLICIT_RETRY_MESSAGE = (
    "An error occurred while processing your request. You can retry your request, or contact us "
    "through our help center at help.openai.com if the error persists. Please include the request "
    "ID req_******** in your message."
)
BEDROCK_EXPLICIT_RETRY_MESSAGE = (
    '{"message":"The system encountered an unexpected error during processing. Try your request again."}'
)
NVIDIA_NIM_RESOURCE_EXHAUSTED_MESSAGE = "ResourceExhausted: Worker local total request limit reached (288/48)"
BUN_FETCH_SOCKET_CLOSED_MESSAGE = (
    "The socket connection was closed unexpectedly. For more information, pass `verbose: true` "
    "in the second argument to fetch()"
)
OPENAI_RESPONSES_EARLY_EOF_MESSAGE = "OpenAI Responses stream ended before a terminal response event"
WRAPPED_DNS_LOOKUP_ERROR = (
    "The pending stream has been canceled (caused by: getaddrinfo ENOTFOUND bedrock-runtime.us-east-1.amazonaws.com)"
)
AZURE_PEAK_LOAD_ERROR = (
    "The system is currently experiencing high demand and cannot process your request. Your request exceeds the "
    "maximum usage size allowed during peak load. For improved capacity reliability, consider switching to "
    "Provisioned Throughput."
)
CODEX_CONNECTION_LIMIT_ERROR = (
    "Codex error: Responses websocket connection limit reached (60 minutes). Create a new websocket connection "
    "to continue."
)


def error_message(text: str):
    return faux_assistant_message("", stop_reason="error", error_message=text)


class TestProviderRetryClassification:
    # Regression for #12124: the Codex WebSocket lifetime limit is recoverable on a fresh connection.
    @pytest.mark.parametrize("text", [CODEX_CONNECTION_LIMIT_ERROR, "Codex error: websocket_connection_limit_reached"])
    def test_matches_the_codex_websocket_connection_lifetime_limit(self, text):
        assert is_retryable_assistant_error(error_message(text)) is True

    def test_does_not_match_a_generic_connection_limit(self):
        assert is_retryable_assistant_error(error_message("connection limit reached")) is False

    def test_matches_explicit_provider_retry_guidance(self):
        assert is_retryable_assistant_error(error_message(OPENAI_EXPLICIT_RETRY_MESSAGE)) is True
        assert is_retryable_assistant_error(error_message(BEDROCK_EXPLICIT_RETRY_MESSAGE)) is True
        assert is_retryable_assistant_error(error_message(NVIDIA_NIM_RESOURCE_EXHAUSTED_MESSAGE)) is True

    def test_matches_bun_fetch_socket_drop_wording(self):
        assert is_retryable_assistant_error(error_message(BUN_FETCH_SOCKET_CLOSED_MESSAGE)) is True

    def test_matches_upstream_request_buffer_exhaustion_wording(self):
        assert (
            is_retryable_assistant_error(error_message("Error: exceeded request buffer limit while retrying upstream"))
            is True
        )

    @pytest.mark.parametrize(
        "text",
        [
            WRAPPED_DNS_LOOKUP_ERROR,
            "connect ENOTFOUND api.example.com",
            "EAI_AGAIN api.example.com",
            "getaddrinfo failed for api.example.com",
        ],
    )
    def test_matches_dns_transport_failure_wording(self, text):
        assert is_retryable_assistant_error(error_message(text)) is True

    @pytest.mark.parametrize(
        "text",
        [
            # Regression for #12128: bare Node socket error codes.
            "read ECONNRESET",
            "connect ETIMEDOUT 140.82.113.3:443",
            "write EPIPE",
            "connect ECONNREFUSED 127.0.0.1:443",
            "ECONNABORTED",
            "Connection reset by peer",
        ],
    )
    def test_matches_node_socket_error_codes(self, text):
        assert is_retryable_assistant_error(error_message(text)) is True

    @pytest.mark.parametrize(
        "text", ["Invalid value EXECONNRESET", "Unknown field ETIMEDOUTX", "Invalid key ERR_EPIPE_ID"]
    )
    def test_does_not_match_socket_error_codes_inside_longer_identifiers(self, text):
        assert is_retryable_assistant_error(error_message(text)) is False

    @pytest.mark.parametrize(
        "text",
        ["The pending stream has been canceled", "The pending stream has been canceled (caused by: socket closed)"],
    )
    def test_matches_http2_pending_stream_cancellation(self, text):
        # Regression for #10379.
        assert is_retryable_assistant_error(error_message(text)) is True

    @pytest.mark.parametrize(
        "text",
        [
            # Regression for #10965 (provider unknown, no status code in the message).
            "The service is temporarily unavailable. Please retry later.",
            # Bedrock ServiceUnavailableException as formatted by format_bedrock_error.
            "Service unavailable: The service is temporarily unavailable.",
        ],
    )
    def test_matches_service_unavailable_wording(self, text):
        assert is_retryable_assistant_error(error_message(text)) is True

    @pytest.mark.parametrize(
        "text",
        [
            # Regression for #10996: node-fetch errors from the google-vertex OAuth token refresh.
            "request to https://oauth2.googleapis.com/token failed, reason:",
            "request to https://oauth2.googleapis.com/token failed, reason: connect ECONNREFUSED 127.0.0.1:9",
        ],
    )
    def test_matches_node_fetch_request_failure_wording(self, text):
        assert is_retryable_assistant_error(error_message(text)) is True

    @pytest.mark.parametrize(
        "text",
        [
            # Regression for #10920: gateway upstream failures sent as `response.failed` error codes.
            "unknown: Unable to forward upstream stream",
            "upstream_error: Upstream request failed",
            "upstream_stream_read_error: Upstream response stream was interrupted",
            "Upstream HTTP/2 stream failed",
        ],
    )
    def test_matches_gateway_upstream_failure_wording(self, text):
        assert is_retryable_assistant_error(error_message(text)) is True

    def test_matches_gateway_wording_for_a_stream_cut_short_mid_reply(self):
        # Regression for #11879.
        text = "Claude Code's stream from Anthropic was cut short after the reply had begun"
        assert is_retryable_assistant_error(error_message(text)) is True

    @pytest.mark.parametrize(
        "text",
        [
            # Regression for #12227: gateway mid-stream `event: error` frames without a status code.
            "The provider is temporarily unavailable. Retry shortly.",
            "The upstream provider is temporarily unavailable. Try again shortly.",
            "The provider ended the response before it was complete.",
            "The connection to the provider was reset. Retry shortly.",
            "The provider reported a failure while generating the response.",
        ],
    )
    def test_matches_gateway_provider_failure_wording(self, text):
        assert is_retryable_assistant_error(error_message(text)) is True

    @pytest.mark.parametrize(
        "text", ["Provider unavailable for this model", "The provider is unavailable in your region"]
    )
    def test_does_not_match_lasting_provider_unavailable_wording(self, text):
        assert is_retryable_assistant_error(error_message(text)) is False

    def test_matches_gateway_wording_for_a_closed_connection(self):
        # Regression for #12064.
        text = '{"type":"error","error":{"type":"api_error","message":"Connection closed."}}'
        assert is_retryable_assistant_error(error_message(text)) is True

    @pytest.mark.parametrize(
        "text",
        [
            "Your request rate increased too quickly",
            (
                "Request rate increased too quickly. To ensure system stability, please adjust your client logic "
                "to scale requests more smoothly over time."
            ),
        ],
    )
    def test_matches_request_rate_throttle_wording(self, text):
        # Regression for #12240 and #2952.
        assert is_retryable_assistant_error(error_message(text)) is True

    def test_matches_openai_responses_streams_that_end_before_terminal_events(self):
        assert is_retryable_assistant_error(error_message(OPENAI_RESPONSES_EARLY_EOF_MESSAGE)) is True

    def test_matches_azure_peak_load_capacity_errors(self):
        # Regression for #9669.
        assert is_retryable_assistant_error(error_message(AZURE_PEAK_LOAD_ERROR)) is True

    def test_keeps_provider_limit_errors_non_retryable(self):
        assert is_retryable_assistant_error(error_message("429 quota exceeded")) is False

    def test_keeps_the_chatgpt_subscription_usage_limit_non_retryable(self):
        message = 'OpenAI API error (429): {"code":"subscription_sharing_usage_limit_exceeded","message":"Usage limit reached."}'
        assert is_retryable_assistant_error(error_message(message)) is False

    def test_keeps_the_chatgpt_business_workspace_usage_limit_non_retryable(self):
        message = (
            'OpenAI API error (429): {"code":"subscription_sharing_workplace_individual_usage_limit_exceeded",'
            '"message":"Usage limit reached."}'
        )
        assert is_retryable_assistant_error(error_message(message)) is False

    @pytest.mark.parametrize(
        "text",
        [
            "subscription_sharing_usage_unavailable: Usage cannot be checked.",
            "subscription_sharing_user_unavailable: User cannot be loaded.",
        ],
    )
    def test_retries_temporary_chatgpt_subscription_errors(self, text):
        assert is_retryable_assistant_error(error_message(text)) is True

    def test_classifies_assistant_error_messages(self):
        assert is_retryable_assistant_error(error_message("overloaded_error")) is True
        # Regression for #9627.
        assert is_retryable_assistant_error(error_message("520 status code (no body)")) is True
        assert is_retryable_assistant_error(error_message("524 status code (no body)")) is True
        assert (
            is_retryable_assistant_error(error_message("525 <!DOCTYPE html><title>SSL handshake failed</title>"))
            is True
        )
        assert is_retryable_assistant_error(faux_assistant_message("not an error")) is False

    @pytest.mark.parametrize(
        "text",
        [
            # Regression for #11057: status codes inside ids or other numbers are not status codes.
            (
                '401 {"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"},'
                '"request_id":"req_011CU503xyz"}'
            ),
            "400 image exceeds 5 MB maximum: 5316784 bytes > 5242880 bytes",
        ],
    )
    def test_does_not_match_status_codes_inside_ids_or_longer_numbers(self, text):
        assert is_retryable_assistant_error(error_message(text)) is False

    @pytest.mark.parametrize(
        "text",
        [
            "Unable to process request. Repeat the request later.",
            "waited 73.4s in queue, reaching this model's admission ceiling of 60s; retry after the hinted delay",
            (
                '{"type":"error","error":{"details":null,"type":"api_error","message":"Unable to complete this '
                'request right now. Please try again."},"request_id":"req_011CgfE1bTgPTDsor4TsApFv"}'
            ),
            "Provider finish_reason: error",
            "Provider returned an empty response",
            "Upstream error from Relace: The model stopped before completing the response.",
        ],
    )
    def test_retries_temporary_provider_failures(self, text):
        assert is_retryable_assistant_error(error_message(text)) is True

    @pytest.mark.parametrize(
        "text",
        [
            # A cancellation, usually the user's.
            "The operation was aborted",
            # Only Anthropic's internal error is retried, not other error types.
            (
                '{"type":"error","error":{"type":"invalid_request_error","message":"messages: at least one '
                'message is required"}}'
            ),
            # A refusal ends the stream with a finish reason other than `error`.
            "Provider finish_reason: content_filter",
            # The quota exclusions are checked before the retryable patterns.
            '{"type":"error","error":{"type":"api_error","message":"Monthly quota exceeded"}}',
        ],
    )
    def test_does_not_retry(self, text):
        assert is_retryable_assistant_error(error_message(text)) is False


# pidrei-only: a network failure is marked by the adapter that caught it; pi matches the SDK's text.
def test_retries_a_message_marked_as_a_network_error_whatever_its_text():
    message = error_message("Could not connect to api.example.com")
    assert is_retryable_assistant_error(message) is False

    error = http.TransportError("[Errno 101] Network is unreachable")
    marked = replace(message, diagnostics=[create_assistant_message_diagnostic(NETWORK_ERROR_DIAGNOSTIC, error)])
    assert is_retryable_assistant_error(marked) is True


def test_caps_agent_retry_delay():
    # Regression for #8826.
    assert retry_delay_ms(RetryPolicy(enabled=True, max_retries=3, base_delay_ms=2000), 6) == 60000
    assert (
        retry_delay_ms(RetryPolicy(enabled=True, max_retries=3, base_delay_ms=2000, max_agent_delay_ms=5000), 5) == 5000
    )
    assert retry_delay_ms(RetryPolicy(enabled=True, max_retries=3, base_delay_ms=2000, max_agent_delay_ms=0), 5) == 0


DISABLED = RetryPolicy(enabled=False, max_retries=3, base_delay_ms=0)
ENABLED = RetryPolicy(enabled=True, max_retries=3, base_delay_ms=0)


class Recorder:
    def __init__(self):
        self.scheduled: list[tuple] = []
        self.attempt_starts = 0
        self.finished: list[tuple] = []

    def callbacks(self) -> RetryCallbacks:
        async def on_scheduled(*args):
            self.scheduled.append(args)

        async def on_attempt_start():
            self.attempt_starts += 1

        async def on_finished(*args):
            self.finished.append(args)

        return RetryCallbacks(
            on_retry_scheduled=on_scheduled,
            on_retry_attempt_start=on_attempt_start,
            on_retry_finished=on_finished,
        )


@pytest.mark.tonio
async def test_returns_successful_response_immediately_without_retrying():
    calls = 0

    async def produce():
        nonlocal calls
        calls += 1
        return faux_assistant_message("ok")

    result = await retry_assistant_call(produce, ENABLED, None)
    assert result.content == [TextContent(text="ok")]
    assert calls == 1


@pytest.mark.tonio
async def test_does_not_retry_an_aborted_message():
    calls = 0
    recorder = Recorder()

    async def produce():
        nonlocal calls
        calls += 1
        return faux_assistant_message("", stop_reason="aborted")

    result = await retry_assistant_call(produce, ENABLED, None, recorder.callbacks())
    assert result.stop_reason == "aborted"
    assert calls == 1
    assert recorder.scheduled == []


@pytest.mark.tonio
async def test_does_not_retry_a_non_retryable_error():
    calls = 0
    recorder = Recorder()

    async def produce():
        nonlocal calls
        calls += 1
        return error_message("insufficient_quota")

    result = await retry_assistant_call(produce, ENABLED, None, recorder.callbacks())
    assert result.stop_reason == "error"
    assert calls == 1
    assert recorder.scheduled == []
    assert recorder.finished == []


@pytest.mark.tonio
async def test_retries_transient_error_up_to_max_retries_then_returns_final_error():
    calls = 0
    recorder = Recorder()

    async def produce():
        nonlocal calls
        calls += 1
        return error_message("terminated")

    result = await retry_assistant_call(produce, ENABLED, None, recorder.callbacks())
    assert result.stop_reason == "error"
    assert calls == 4  # 1 initial + 3 retries
    assert len(recorder.scheduled) == 3
    assert recorder.finished == [(False, 3, "terminated")]


@pytest.mark.tonio
async def test_reports_capped_retry_delays():
    # Regression for #8826.
    calls = 0
    recorder = Recorder()
    policy = RetryPolicy(enabled=True, max_retries=4, base_delay_ms=10, max_agent_delay_ms=15)

    async def produce():
        nonlocal calls
        calls += 1
        return error_message("terminated") if calls < 5 else faux_assistant_message("recovered")

    await retry_assistant_call(produce, policy, None, recorder.callbacks())

    assert [call[2] for call in recorder.scheduled] == [10, 15, 15, 15]


# Regression for #10506.
@pytest.mark.tonio
async def test_stops_retrying_once_a_call_succeeds_and_removes_backoff_abort_listeners():
    # pi also asserts no abort listener is left on the signal; the backoff here
    # waits on the token and registers nothing on it, so there is nothing to count.
    cancel = CancelToken()
    calls = 0
    recorder = Recorder()

    async def produce():
        nonlocal calls
        calls += 1
        if calls < 4:
            return error_message("terminated")
        return faux_assistant_message("recovered")

    result = await retry_assistant_call(produce, ENABLED, cancel, recorder.callbacks())
    assert result.content == [TextContent(text="recovered")]
    assert calls == 4
    assert recorder.finished == [(True, 3)]


@pytest.mark.tonio
async def test_reports_an_aborted_retried_call_as_unsuccessful():
    calls = 0
    recorder = Recorder()

    async def produce():
        nonlocal calls
        calls += 1
        if calls == 1:
            return error_message("terminated")
        return faux_assistant_message("", stop_reason="aborted")

    result = await retry_assistant_call(produce, ENABLED, None, recorder.callbacks())
    assert result.stop_reason == "aborted"
    assert calls == 2
    assert recorder.finished == [(False, 1)]


@pytest.mark.tonio
async def test_does_not_retry_when_policy_is_disabled():
    calls = 0
    recorder = Recorder()

    async def produce():
        nonlocal calls
        calls += 1
        return error_message("terminated")

    result = await retry_assistant_call(produce, DISABLED, None, recorder.callbacks())
    assert result.stop_reason == "error"
    assert calls == 1
    assert recorder.scheduled == []
    assert recorder.finished == []


@pytest.mark.tonio
async def test_emits_on_retry_attempt_start_after_backoff_before_each_retried_call():
    events: list[str] = []
    calls = 0

    async def produce():
        nonlocal calls
        events.append(f"produce:{calls}")
        calls += 1
        if calls < 3:
            return error_message("terminated")
        return faux_assistant_message("recovered")

    async def on_scheduled(attempt, *rest):
        events.append(f"retry:{attempt}")

    async def on_attempt_start():
        events.append("attempt-start")

    callbacks = RetryCallbacks(
        on_retry_scheduled=on_scheduled,
        on_retry_attempt_start=on_attempt_start,
    )
    result = await retry_assistant_call(produce, ENABLED, None, callbacks)
    assert result.content == [TextContent(text="recovered")]
    assert events == [
        "produce:0",
        "retry:1",
        "attempt-start",
        "produce:1",
        "retry:2",
        "attempt-start",
        "produce:2",
    ]


@pytest.mark.tonio
async def test_aborts_backoff_sleep_via_cancel_and_returns_aborted_message():
    cancel = CancelToken()
    calls = 0
    recorder = Recorder()
    produced = tonio.Event()

    async def produce():
        nonlocal calls
        calls += 1
        produced.set()
        return error_message("terminated")

    policy = RetryPolicy(enabled=True, max_retries=5, base_delay_ms=10_000)

    async def run():
        return await retry_assistant_call(produce, policy, cancel, recorder.callbacks())

    async def abort_after_first_produce():
        await produced.wait()
        await tonio.yield_now()
        cancel.cancel()

    result, _ = await tonio.spawn(run(), abort_after_first_produce())
    assert result.stop_reason == "aborted"
    assert result.error_message is None
    assert calls == 1
    assert recorder.finished == [(False, 1, "terminated")]
