"""Mirror of pi's openai-completions-provider-stream-event.test.ts.

pi replaces the `openai` package with `vi.mock` and drives `completeSimple`;
here `stream_simple` runs the same simple-options path and the stub replaces
the adapter's `_create_client` by name, so the chunks come from a canned SSE
body.
"""

import contextlib
import json

import pytest

from pidrei_ai.api import openai_completions
from pidrei_ai.types import Context, Model, ModelCost, SimpleStreamOptions, TextContent, UsageCost, UserMessage
from pidrei_ai.utils.transcript import normalize_context


class _FakeResponse:
    def __init__(self, chunks: list[dict]):
        self.status = 200
        self.headers = {"x-request-id": "req-1"}
        self._body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks).encode()

    async def aiter_bytes(self):
        yield self._body


@contextlib.contextmanager
def _streaming(chunks: list[dict]):
    class _FakeClient:
        async def create(self, _params, *, timeout_ms, cancel):
            return _FakeResponse(chunks)

    original = openai_completions._create_client
    openai_completions._create_client = lambda *_args, **_kwargs: _FakeClient()
    try:
        yield
    finally:
        openai_completions._create_client = original


def open_router_model(provider: str = "openrouter") -> Model:
    return Model(
        id="openrouter/auto",
        name="OpenRouter Auto",
        api="openai-completions",
        provider=provider,
        base_url="https://openrouter.ai/api/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=200_000,
        max_tokens=8192,
    )


# Regression test for #9784.
@pytest.mark.tonio
async def test_exposes_provider_chunks_including_openrouter_metadata():
    first_chunk = {
        "id": "chatcmpl-1",
        "model": "anthropic/claude-sonnet-4.6",
        "choices": [{"index": 0, "delta": {"content": "hello"}}],
    }
    final_chunk = {
        "id": "chatcmpl-1",
        "model": "anthropic/claude-sonnet-4.6",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "cost": 0.0012,
            "is_byok": False,
        },
        "openrouter_metadata": {"strategy": "direct", "region": "iad"},
    }
    events: list = []

    async def on_provider_stream_event(data, _model) -> None:
        events.append(data)

    with _streaming([first_chunk, final_chunk]):
        message = await openai_completions.stream_simple(
            open_router_model(),
            normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
            SimpleStreamOptions(api_key="test", on_provider_stream_event=on_provider_stream_event),
        ).result()

    assert message.content == [TextContent(text="hello")]
    assert events == [first_chunk, final_chunk]


# Regression test for #9980.
@pytest.mark.tonio
async def test_uses_openrouters_reported_total_without_double_counting_non_byok_upstream_cost():
    chunks = [
        {
            "id": "chatcmpl-cost",
            "model": "openrouter/auto",
            "choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}],
            # These billing fields were captured from a production OpenRouter stream
            # even though the OpenAI SDK omits them from CompletionUsage.
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cost": 0.004,
                "is_byok": False,
                "cost_details": {"upstream_inference_cost": 0.003},
            },
        }
    ]

    with _streaming(chunks):
        message = await openai_completions.stream_simple(
            open_router_model("openrouter-work"),
            normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
            SimpleStreamOptions(api_key="test"),
        ).result()

    assert message.usage.cost == UsageCost(
        input=0, output=0, cache_read=0, cache_write=0, total=0.004, total_from_provider=True
    )


@pytest.mark.tonio
async def test_includes_upstream_inference_cost_for_openrouter_byok_requests():
    chunks = [
        {
            "id": "chatcmpl-byok",
            "model": "openrouter/auto",
            "choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cost": 0.0002,
                "is_byok": True,
                "cost_details": {"upstream_inference_cost": 0.003},
            },
        }
    ]

    with _streaming(chunks):
        message = await openai_completions.stream_simple(
            open_router_model(),
            normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
            SimpleStreamOptions(api_key="test"),
        ).result()

    assert message.usage.cost.total == pytest.approx(0.0032, abs=1e-10)
    assert message.usage.cost.total_from_provider is True
