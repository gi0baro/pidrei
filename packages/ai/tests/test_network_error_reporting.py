"""pidrei-only: a response that failed in the exchange itself (connect, read,
write, protocol, timeout) is reported from the class of the failure and marked
with a `network_error` diagnostic, which is what makes it retryable.

pi's SDKs report every such failure as `Connection error.` and undici a cut
body as `terminated`, and pi's retry patterns match those texts. PiDrei's
transport words them differently, so the adapters decide from the class.
"""

import json

import pytest

from pidrei_ai.api.anthropic_messages import AnthropicOptions, stream as stream_anthropic
from pidrei_ai.api.google_generative_ai import stream_simple as stream_google_generative_ai
from pidrei_ai.api.mistral_conversations import format_mistral_error
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, Model, ModelCost, SimpleStreamOptions, UserMessage
from pidrei_ai.utils.retry import is_retryable_assistant_error
from pidrei_ai.utils.transcript import normalize_context
from pidrei_http import http


# What punkreq says for a body cut by a clean close: no retry pattern matches it.
_CUT_BODY_TEXT = "error reading a body from connection: end of file before message length reached"
_UNREACHABLE_TEXT = "Failed to connect to https://upstream.test:443: [Errno 101] Network is unreachable"


def _context() -> Context:
    return Context(messages=[UserMessage(content="hello", timestamp=1)])


@pytest.mark.tonio
async def test_a_failed_connect_is_reported_by_class_and_is_retryable():
    model = Model(
        id="test-model",
        name="Test Model",
        api="google-generative-ai",
        provider="test-provider",
        base_url="https://upstream.test/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=10_000,
        max_tokens=1_000,
    )

    async def fetch(request, *, env=None):
        raise http.ConnectError(_UNREACHABLE_TEXT, request=request)

    message = await stream_google_generative_ai(
        model, normalize_context(_context()), SimpleStreamOptions(api_key="test-key", fetch=fetch)
    ).result()

    assert message.stop_reason == "error"
    assert message.error_message == "Could not connect to upstream.test"
    [diagnostic] = message.diagnostics
    assert diagnostic.type == "network_error"
    # The transport's own error stays in the session for debugging.
    assert (diagnostic.error.name, diagnostic.error.message) == ("ConnectError", _UNREACHABLE_TEXT)
    assert is_retryable_assistant_error(message) is True


class _CutResponse:
    """An SSE response whose body is cut after `message_start`."""

    status = 200
    headers = {"content-type": "text/event-stream"}  # noqa: RUF012

    async def aiter_bytes(self):
        start = {"type": "message_start", "message": {"id": "msg_test", "usage": {"input_tokens": 1}}}
        yield f"event: message_start\ndata: {json.dumps(start)}\n\n".encode()
        raise http.ReadError(_CUT_BODY_TEXT, request=http.Request("POST", "https://api.anthropic.com/v1/messages"))


class _CutClient:
    async def create(self, params, *, timeout_ms, cancel):
        return _CutResponse()


@pytest.mark.tonio
async def test_a_response_cut_mid_stream_is_reported_by_class_and_is_retryable():
    model = get_builtin_model("anthropic", "claude-haiku-4-5")
    assert model is not None

    message = await stream_anthropic(model, _context(), AnthropicOptions(client=_CutClient())).result()

    assert message.stop_reason == "error"
    assert message.error_message == "Connection to api.anthropic.com was lost"
    assert [diagnostic.type for diagnostic in message.diagnostics] == ["network_error"]
    assert is_retryable_assistant_error(message) is True


def test_mistral_reports_a_network_failure_by_class():
    error = http.ReadError(_CUT_BODY_TEXT, request=http.Request("POST", "https://api.mistral.ai/v1/chat/completions"))

    assert format_mistral_error(error) == "Connection to api.mistral.ai was lost"
