"""Mirror of pi's retry.test.ts."""

import pytest
import tonio.colored as tonio

from pppi_ai.providers.faux import faux_assistant_message
from pppi_ai.types import TextContent
from pppi_ai.utils.cancel import CancelToken
from pppi_ai.utils.retry import RetryCallbacks, RetryPolicy, is_retryable_assistant_error, retry_assistant_call


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


def error_message(text: str):
    return faux_assistant_message("", stop_reason="error", error_message=text)


class TestProviderRetryClassification:
    def test_matches_explicit_provider_retry_guidance(self):
        assert is_retryable_assistant_error(error_message(OPENAI_EXPLICIT_RETRY_MESSAGE)) is True
        assert is_retryable_assistant_error(error_message(BEDROCK_EXPLICIT_RETRY_MESSAGE)) is True
        assert is_retryable_assistant_error(error_message(NVIDIA_NIM_RESOURCE_EXHAUSTED_MESSAGE)) is True

    def test_matches_bun_fetch_socket_drop_wording(self):
        assert is_retryable_assistant_error(error_message(BUN_FETCH_SOCKET_CLOSED_MESSAGE)) is True

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

    def test_matches_openai_responses_streams_that_end_before_terminal_events(self):
        assert is_retryable_assistant_error(error_message(OPENAI_RESPONSES_EARLY_EOF_MESSAGE)) is True

    def test_keeps_provider_limit_errors_non_retryable(self):
        assert is_retryable_assistant_error(error_message("429 quota exceeded")) is False

    def test_classifies_assistant_error_messages(self):
        assert is_retryable_assistant_error(error_message("overloaded_error")) is True
        assert is_retryable_assistant_error(error_message("524 status code (no body)")) is True
        assert is_retryable_assistant_error(faux_assistant_message("not an error")) is False


DISABLED = RetryPolicy(enabled=False, max_retries=3, base_delay_ms=0)
ENABLED = RetryPolicy(enabled=True, max_retries=3, base_delay_ms=0)


class Recorder:
    def __init__(self):
        self.scheduled: list[tuple] = []
        self.attempt_starts = 0
        self.finished: list[tuple] = []

    def callbacks(self) -> RetryCallbacks:
        return RetryCallbacks(
            on_retry_scheduled=lambda *args: self.scheduled.append(args),
            on_retry_attempt_start=lambda: setattr(self, "attempt_starts", self.attempt_starts + 1),
            on_retry_finished=lambda *args: self.finished.append(args),
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
async def test_stops_retrying_once_a_call_succeeds():
    calls = 0
    recorder = Recorder()

    async def produce():
        nonlocal calls
        calls += 1
        if calls < 3:
            return error_message("terminated")
        return faux_assistant_message("recovered")

    result = await retry_assistant_call(produce, ENABLED, None, recorder.callbacks())
    assert result.content == [TextContent(text="recovered")]
    assert calls == 3
    assert recorder.finished == [(True, 2)]


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

    callbacks = RetryCallbacks(
        on_retry_scheduled=lambda attempt, *rest: events.append(f"retry:{attempt}"),
        on_retry_attempt_start=lambda: events.append("attempt-start"),
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
