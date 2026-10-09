"""Mirror of pi's fetch-option.test.ts.

pi stubs `globalThis.fetch` with one that must never be called and passes a
custom fetch through the stream options; here the pooled client behind
`http.client_for` is the stub (it never sends) and the custom fetch is a
`FetchFunction` returning synthetic `http.Response`s. pi's `pi-messages` case
is not mirrored: that API is not ported.
"""

import base64
import json

import pytest

from pidrei_ai.api.anthropic_messages import stream_simple as stream_anthropic
from pidrei_ai.api.azure_openai_responses import stream_simple as stream_azure_openai_responses
from pidrei_ai.api.google_generative_ai import stream_simple as stream_google_generative_ai
from pidrei_ai.api.google_vertex import stream_simple as stream_google_vertex
from pidrei_ai.api.mistral_conversations import stream_simple as stream_mistral
from pidrei_ai.api.openai_codex_responses import stream_simple as stream_openai_codex_responses
from pidrei_ai.api.openai_completions import stream_simple as stream_openai_completions
from pidrei_ai.api.openai_responses import stream_simple as stream_openai_responses
from pidrei_ai.api.typesafe_system_one import classify as classify_typesafe
from pidrei_ai.images import generate_images
from pidrei_ai.types import (
    ClassifierChoiceQuestion,
    ClassifierContext,
    ClassifierModel,
    ClassifierOptions,
    Context,
    ImageModel,
    ImagesContext,
    ImagesOptions,
    Model,
    ModelCost,
    SimpleStreamOptions,
    TextContent,
    UserMessage,
)
from pidrei_ai.utils.transcript import normalize_context
from pidrei_http import http


try:
    from compression import zstd
except ImportError:  # pragma: no cover - stdlib built without libzstd
    zstd = None


def make_model(api: str, **overrides) -> Model:
    defaults: dict = {
        "id": "test-model",
        "name": "Test Model",
        "api": api,
        "provider": "test-provider",
        "base_url": "https://upstream.test/v1",
        "reasoning": False,
        "input": ["text"],
        "cost": ModelCost(),
        "context_window": 10_000,
        "max_tokens": 1_000,
    }
    defaults.update(overrides)
    return Model(**defaults)


def make_context():
    return normalize_context(Context(messages=[UserMessage(content="hello", timestamp=1)]))


class AmbientClient:
    """pi's stubbed `globalThis.fetch`: the pooled client that must never send."""

    def __init__(self):
        self.sends = 0

    async def send(self, request, **kwargs):
        self.sends += 1
        raise AssertionError("ambient fetch must not be called")

    def post(self, *args, **kwargs):
        return self.send(None)

    def request(self, *args, **kwargs):
        return self.send(None)


class CustomFetch:
    """Records every prepared request and answers each with `response_for`."""

    def __init__(self, response_for):
        self.requests: list[http.Request] = []
        self.envs: list = []
        self._response_for = response_for

    async def __call__(self, request, *, env=None):
        self.requests.append(request)
        self.envs.append(env)
        return self._response_for(request)


def rejected(request: http.Request) -> http.Response:
    return http.Response(
        401,
        headers={"content-type": "application/json"},
        json={"error": {"message": "upstream rejected request"}},
        request=request,
    )


def anthropic_sse(request: http.Request) -> http.Response:
    events = [
        {"type": "message_start", "message": {"id": "msg_test", "usage": {"input_tokens": 1, "output_tokens": 0}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ]
    body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()
    return http.Response(200, headers={"content-type": "text/event-stream"}, content=body, request=request)


@pytest.fixture
def ambient(monkeypatch) -> AmbientClient:
    client = AmbientClient()
    monkeypatch.setattr(http, "client_for", lambda url, env=None: client)
    return client


@pytest.mark.tonio
async def test_passes_fetch_through_stream_simple_to_anthropic(ambient):
    custom = CustomFetch(rejected)
    result = await stream_anthropic(
        make_model("anthropic-messages"),
        make_context(),
        SimpleStreamOptions(api_key="test-key", fetch=custom, max_retries=0),
    ).result()

    assert len(custom.requests) == 1
    assert ambient.sends == 0
    assert result.stop_reason == "error"
    assert "upstream rejected request" in result.error_message


@pytest.mark.tonio
async def test_anthropic_fetch_receives_the_prepared_request(ambient):
    custom = CustomFetch(anthropic_sse)
    env = {"ANTHROPIC_API_KEY": "unused"}
    result = await stream_anthropic(
        make_model("anthropic-messages"),
        make_context(),
        SimpleStreamOptions(
            api_key="test-key", fetch=custom, env=env, headers={"anthropic-beta": "custom-beta"}, max_retries=0
        ),
    ).result()

    assert result.stop_reason == "stop"
    assert [block.text for block in result.content] == ["Hello"]
    assert ambient.sends == 0
    assert custom.envs == [env]
    (request,) = custom.requests
    assert request.method == "POST"
    assert str(request.url) == "https://upstream.test/v1/v1/messages"
    assert request.headers["x-api-key"] == "test-key"
    # The SDK lifts `betas` into the header; the wire body never carries it.
    assert request.headers["anthropic-beta"] == "custom-beta"
    body = json.loads(request.content)
    assert "betas" not in body
    assert body["stream"] is True
    assert body["model"] == "test-model"
    assert request.headers["content-length"] == str(len(request.content))
    assert request.timeout.read == http.STREAMING_TIMEOUT.read


CODEX_API_KEY = "header.{}.signature".format(
    base64.urlsafe_b64encode(
        json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": "account"}}).encode()
    ).decode()
)


@pytest.mark.tonio
async def test_passes_fetch_through_stream_simple_to_openai_adapters(ambient):
    custom = CustomFetch(rejected)
    runs = [
        (stream_openai_completions, "openai-completions"),
        (stream_openai_responses, "openai-responses"),
        (stream_azure_openai_responses, "azure-openai-responses"),
    ]
    for run, api in runs:
        result = await run(
            make_model(api), make_context(), SimpleStreamOptions(api_key="test-key", fetch=custom, max_retries=0)
        ).result()
        assert result.stop_reason == "error", api
        assert "upstream rejected request" in result.error_message, api

    assert len(custom.requests) == len(runs)
    assert ambient.sends == 0
    assert [str(request.url) for request in custom.requests] == [
        "https://upstream.test/v1/chat/completions",
        "https://upstream.test/v1/responses",
        "https://upstream.test/v1/responses?api-version=v1",
    ]


@pytest.mark.tonio
async def test_uses_fetch_for_mistral_and_codex_sse_requests(ambient):
    custom = CustomFetch(rejected)
    mistral = await stream_mistral(
        make_model("mistral-conversations"), make_context(), SimpleStreamOptions(api_key="test-key", fetch=custom)
    ).result()
    codex = await stream_openai_codex_responses(
        make_model("openai-codex-responses"),
        make_context(),
        SimpleStreamOptions(api_key=CODEX_API_KEY, fetch=custom, transport="sse", max_retries=0),
    ).result()

    assert mistral.stop_reason == "error"
    assert codex.stop_reason == "error"
    assert "upstream rejected request" in codex.error_message
    assert len(custom.requests) == 2
    assert ambient.sends == 0

    # Mistral: the callback sees the wire payload, after field conversion.
    mistral_request, codex_request = custom.requests
    assert str(mistral_request.url) == "https://upstream.test/v1/v1/chat/completions"
    mistral_body = json.loads(mistral_request.content)
    assert "maxTokens" not in mistral_body
    assert mistral_body["max_tokens"] == 1_000
    assert mistral_body["messages"][0] == {"role": "user", "content": "hello"}

    # Codex SSE: the callback sees the final bytes, compressed when zstd is built in.
    assert str(codex_request.url) == "https://upstream.test/v1/codex/responses"
    assert codex_request.headers["chatgpt-account-id"] == "account"
    if codex_request.headers.get("content-encoding") == "zstd":
        assert zstd is not None
        codex_body = json.loads(zstd.decompress(codex_request.content))
    else:
        codex_body = json.loads(codex_request.content)
    assert codex_body["model"] == "test-model"
    assert codex_body["stream"] is True


@pytest.mark.tonio
async def test_uses_fetch_for_image_generation(ambient):
    custom = CustomFetch(rejected)
    model = ImageModel(
        id="test-model",
        name="Test Model",
        api="openrouter-images",
        provider="openrouter",
        base_url="https://upstream.test/v1",
        input=["text"],
        output=["image"],
        cost=ModelCost(),
    )
    result = await generate_images(
        model,
        ImagesContext(input=[TextContent(text="draw")]),
        ImagesOptions(api_key="test-key", fetch=custom, max_retries=0),
    )

    assert result.stop_reason == "error"
    assert "upstream rejected request" in result.error_message
    assert len(custom.requests) == 1
    assert ambient.sends == 0
    assert str(custom.requests[0].url) == "https://upstream.test/v1/chat/completions"


@pytest.mark.tonio
async def test_uses_fetch_for_classifier_requests(ambient):
    custom = CustomFetch(rejected)
    model = ClassifierModel(
        id="test-model",
        name="Test Model",
        api="typesafe-system-one",
        provider="test-provider",
        base_url="https://upstream.test/v1",
        input=["text"],
        cost=ModelCost(),
        context_window=10_000,
    )
    context = ClassifierContext(
        state={"text": "hello"},
        questions={"category": ClassifierChoiceQuestion(instructions="Classify", criteria={"a": "A", "b": "B"})},
    )

    result = await classify_typesafe(model, context, ClassifierOptions(api_key="test-key", fetch=custom, max_retries=0))

    assert result.stop_reason == "error"
    assert "401" in result.error_message
    assert len(custom.requests) == 1
    assert ambient.sends == 0
    assert custom.requests[0].headers["authorization"] == "Bearer test-key"


class GoogleAmbientClient:
    """The pooled client the Google client posts through (it takes no transport)."""

    def __init__(self):
        self.posts = 0

    async def post(self, url, *, json, headers, timeout):
        self.posts += 1
        return http.Response(
            401,
            headers={"content-type": "application/json"},
            json={"error": {"message": "upstream rejected request"}},
        )


@pytest.mark.tonio
async def test_rejects_custom_fetch_for_google_adapters_instead_of_silently_bypassing_it(ambient):
    custom = CustomFetch(rejected)
    google = await stream_google_generative_ai(
        make_model("google-generative-ai"), make_context(), SimpleStreamOptions(api_key="test-key", fetch=custom)
    ).result()
    vertex = await stream_google_vertex(
        make_model("google-vertex"), make_context(), SimpleStreamOptions(api_key="test-key", fetch=custom)
    ).result()

    assert "Custom fetch is not supported by the Google Generative AI adapter" in google.error_message
    assert "Custom fetch is not supported by the Google Vertex adapter" in vertex.error_message
    assert custom.requests == []
    assert ambient.sends == 0


@pytest.mark.tonio
async def test_allows_google_adapters_to_receive_the_default_fetch_explicitly(monkeypatch):
    client = GoogleAmbientClient()
    monkeypatch.setattr(http, "client_for", lambda url, env=None: client)
    result = await stream_google_generative_ai(
        make_model("google-generative-ai"),
        make_context(),
        SimpleStreamOptions(api_key="test-key", fetch=http.default_fetch),
    ).result()

    assert client.posts == 1
    assert result.stop_reason == "error"
    assert "Custom fetch is not supported" not in result.error_message
