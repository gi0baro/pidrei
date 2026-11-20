"""Mirror of pi's google-stop-reason-continuation.test.ts.

pi's `FinishReason.CONTINUATION` is the wire string here, and pi's `vi.fn`
fetch is a `FetchFunction` returning a synthetic SSE `http.Response`.
"""

import json

import pytest

from pidrei_ai.api.google_generative_ai import stream_simple as stream_google_generative_ai
from pidrei_ai.api.google_shared import map_stop_reason
from pidrei_ai.api.google_vertex import stream_simple as stream_google_vertex
from pidrei_ai.types import Context, Model, ModelCost, SimpleStreamOptions, UserMessage
from pidrei_ai.utils.transcript import normalize_context
from pidrei_http import http


CONTEXT = normalize_context(Context(messages=[UserMessage(content="hello", timestamp=1)]))


def create_model(api: str) -> Model:
    return Model(
        id="test-model",
        name="Test Model",
        api=api,
        provider="test-provider",
        base_url="https://upstream.test/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=10_000,
        max_tokens=1_000,
    )


# #12097: @google/genai 2.32.0, needed for httpOptions.fetch, adds FinishReason.CONTINUATION.
def test_maps_continuation_to_length_like_max_tokens():
    assert map_stop_reason("CONTINUATION") == "length"


# #12097: the SDK continues automatically by default; the adapters turn that off.
@pytest.mark.tonio
async def test_ends_the_turn_as_length_instead_of_letting_the_sdk_send_a_continuation_request():
    chunk = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "partial"}]},
                "finishReason": "CONTINUATION",
                "continuationToken": "token-1",
            }
        ]
    }
    requests: list[http.Request] = []

    async def fetch(request, *, env=None):
        requests.append(request)
        return http.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"data: {json.dumps(chunk)}\n\n".encode(),
            request=request,
        )

    google = await stream_google_generative_ai(
        create_model("google-generative-ai"), CONTEXT, SimpleStreamOptions(api_key="test-key", fetch=fetch)
    ).result()
    vertex = await stream_google_vertex(
        create_model("google-vertex"), CONTEXT, SimpleStreamOptions(api_key="test-key", fetch=fetch)
    ).result()

    assert len(requests) == 2
    assert google.stop_reason == "length"
    assert vertex.stop_reason == "length"
