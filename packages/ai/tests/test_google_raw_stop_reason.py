"""Mirror of pi's google-raw-stop-reason.test.ts (stream driver shared with
test_google_stream.py)."""

import contextlib

import pytest

from pidrei_ai.api import google_generative_ai, google_vertex
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, StreamOptions, UserMessage
from pidrei_ai.utils.user_agent import get_user_agent

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


@contextlib.contextmanager
def _recording_client(adapter, constructor_calls: list[dict]):
    """pi's `vi.mock` records the GoogleGenAI constructor config; the shared stream
    stub in test_google_stream.py discards it, so this one keeps it."""

    class _Fake:
        def __init__(self, config):
            constructor_calls.append(dict(config))

        async def generate_content_stream(self, _params, *, env=None, cancel=None):
            async def _chunks():
                yield _chunk([_part("hi")], finish_reason="STOP")

            return _chunks()

    original = adapter.GoogleGenAI
    adapter.GoogleGenAI = _Fake
    try:
        yield
    finally:
        adapter.GoogleGenAI = original


async def _capture_google_headers(headers: dict | None = None) -> dict:
    constructor_calls: list[dict] = []
    model = get_builtin_model("google", "gemini-3.1-pro-preview")
    with _recording_client(google_generative_ai, constructor_calls):
        await google_generative_ai.stream(
            model,
            Context(messages=[UserMessage(content="hello", timestamp=1)]),
            StreamOptions(api_key="test-api-key", headers=headers),
        ).result()

    assert len(constructor_calls) == 1
    return (constructor_calls[0].get("httpOptions") or {}).get("headers") or {}


@pytest.mark.tonio
async def test_uses_the_runtime_user_agent_by_default():
    assert (await _capture_google_headers())["User-Agent"] == get_user_agent()


@pytest.mark.tonio
async def test_lets_explicit_headers_override_the_default_user_agent():
    assert (await _capture_google_headers({"User-Agent": "custom-agent"}))["User-Agent"] == "custom-agent"
