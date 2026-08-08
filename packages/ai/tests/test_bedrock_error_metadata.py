"""Mirror of pi's bedrock-error-metadata.test.ts.

pi replaces `@aws-sdk/client-bedrock-runtime` with `vi.mock`; here the stub
replaces `api/bedrock_runtime.BedrockRuntimeClient` by name, as in the other
bedrock mirrors.

One documented divergence: pi's SDK delivers *modeled* mid-stream exceptions as
bare object literals, so upstream expects only the request id for them.
pidrei's hand-rolled runtime raises them as `BedrockRuntimeServiceException`
(see `_decode_event`), so the modeled code is available and the diagnostic is
strictly richer — those cases assert the pidrei behavior.
"""

import contextlib
from types import SimpleNamespace

import pytest

from pidrei_ai.api import bedrock_converse_stream as bedrock
from pidrei_ai.api.bedrock_converse_stream import BedrockOptions, stream as stream_bedrock
from pidrei_ai.api.bedrock_runtime import BedrockRuntimeServiceException
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, UserMessage
from pidrei_ai.utils.cancel import CancelToken


DIAGNOSTIC_TYPE = "bedrock_response_failure"
VALIDATION_MESSAGE = "The provided model identifier is invalid."
REQUEST_ID = "11111111-2222-3333-4444-555555555555"

MODEL = get_builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8")
CONTEXT = Context(messages=[UserMessage(content="hello", timestamp=1)])


def make_service_exception(name: str, **extra) -> BedrockRuntimeServiceException:
    """What `send` raises for a non-2xx response; `name` is the modeled AWS error code."""
    return BedrockRuntimeServiceException(name, VALIDATION_MESSAGE, **extra)


@contextlib.contextmanager
def _client(*, send_error=None, stream_error=None):
    class _Fake:
        def __init__(self, _config):
            self.middleware_stack = SimpleNamespace(add=lambda *args, **kwargs: None)

        async def send(self, _command, *, cancel=None):
            if send_error is not None:
                raise send_error

            # Fails after `messageStart`, raising from inside the iterator like
            # the SDK's message unmarshaller does.
            async def items():
                yield {"messageStart": {"role": "assistant"}}
                raise stream_error

            return SimpleNamespace(
                metadata=SimpleNamespace(http_status_code=200, request_id=REQUEST_ID),
                stream=items(),
            )

    original = bedrock.BedrockRuntimeClient
    bedrock.BedrockRuntimeClient = _Fake
    try:
        yield
    finally:
        bedrock.BedrockRuntimeClient = original


async def run_bedrock(cancel: CancelToken | None = None):
    return await stream_bedrock(MODEL, CONTEXT, BedrockOptions(cache_retention="none", cancel=cancel)).result()


def find_diagnostic(message):
    return next((d for d in message.diagnostics or [] if d.type == DIAGNOSTIC_TYPE), None)


@pytest.mark.tonio
async def test_records_status_error_code_and_request_id_for_a_non_2xx_from_send():
    error = make_service_exception("ValidationException", status=400, request_id=REQUEST_ID)
    with _client(send_error=error):
        message = await run_bedrock()
    diagnostic = find_diagnostic(message)

    assert message.stop_reason == "error"
    assert diagnostic is not None
    assert diagnostic.details == {"status": 400, "errorCode": "ValidationException", "requestId": REQUEST_ID}
    assert diagnostic.error is None


@pytest.mark.tonio
async def test_leaves_error_message_untouched_so_retry_classification_is_unaffected():
    error = make_service_exception("ValidationException", status=400, request_id=REQUEST_ID)
    with _client(send_error=error):
        message = await run_bedrock()

    assert message.error_message == f"Validation error: {VALIDATION_MESSAGE}"


@pytest.mark.tonio
async def test_reports_code_and_request_id_for_a_modeled_mid_stream_exception():
    # pi sees a bare object literal here (details == {requestId} only); pidrei's
    # runtime raises the modeled exception, so the code is also available.
    error = BedrockRuntimeServiceException("ThrottlingException", "Too many requests, please wait.")
    with _client(stream_error=error):
        message = await run_bedrock()

    assert message.stop_reason == "error"
    assert find_diagnostic(message).details == {"errorCode": "ThrottlingException", "requestId": REQUEST_ID}


@pytest.mark.tonio
async def test_captures_the_error_code_for_an_unmodeled_mid_stream_error():
    # The `:error-code` frame branch raises a real exception named after the code.
    error = BedrockRuntimeServiceException("ModelStreamErrorException", "Model stream terminated unexpectedly.")
    with _client(stream_error=error):
        message = await run_bedrock()

    assert find_diagnostic(message).details == {
        "errorCode": "ModelStreamErrorException",
        "requestId": REQUEST_ID,
    }


@pytest.mark.tonio
async def test_does_not_report_a_transport_failure_name_as_a_provider_error_code():
    # Informative exception, but not a modeled AWS code; modeled ones end in "Exception".
    with _client(stream_error=TimeoutError("Connection timed out after 1000 ms")):
        message = await run_bedrock()

    assert find_diagnostic(message).details == {"requestId": REQUEST_ID}


@pytest.mark.tonio
async def test_emits_no_diagnostic_when_the_failure_carries_no_provider_metadata():
    with _client(send_error=RuntimeError("socket hang up")):
        message = await run_bedrock()

    assert message.stop_reason == "error"
    assert message.error_message == "socket hang up"
    assert find_diagnostic(message) is None


@pytest.mark.tonio
async def test_emits_no_diagnostic_for_an_aborted_turn():
    cancel = CancelToken()
    cancel.cancel()
    error = make_service_exception("ValidationException", status=400, request_id=REQUEST_ID)
    with _client(send_error=error):
        message = await run_bedrock(cancel)

    assert message.stop_reason == "aborted"
    assert find_diagnostic(message) is None


@pytest.mark.tonio
async def test_drops_header_derived_values_that_exceed_the_length_bound():
    error = make_service_exception("E" * 5000 + "Exception", status=400, request_id="R" * 5000)
    with _client(send_error=error):
        message = await run_bedrock()

    assert find_diagnostic(message).details == {"status": 400}


@pytest.mark.tonio
async def test_omits_the_unknown_placeholder_instead_of_reporting_it_as_a_code():
    # The runtime's fallback when the response carried no `x-amzn-errortype`.
    error = make_service_exception("UnknownError", status=403, request_id=REQUEST_ID)
    with _client(send_error=error):
        message = await run_bedrock()

    assert find_diagnostic(message).details == {"status": 403, "requestId": REQUEST_ID}
