"""Mirror of pi's google-vertex-api-key-resolution.test.ts.

pi replaces the whole `@google/genai` module with `vi.mock` and asserts on the
config object handed to the `GoogleGenAI` constructor. Here the constructor is
`api/google_client.GoogleGenAI`, imported by name into the adapter, so the stub
replaces exactly the same thing — and because that module keeps the SDK's own
camelCase config keys, the assertions are pi's verbatim.

`GOOGLE_CLOUD_API_KEY` is cleared per test as pi does; the adapter reads it
through `options.env`/`os.environ`, so a value left in the ambient environment
would otherwise pick the API-key client.
"""

import contextlib
import os
import time
from dataclasses import replace

import pytest

from pidrei_ai.api import google_vertex
from pidrei_ai.api.google_vertex import stream as stream_google_vertex
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, UserMessage


MODEL = get_builtin_model("google-vertex", "gemini-3-flash-preview")
CONTEXT = Context(messages=[UserMessage(content="hello", timestamp=int(time.time() * 1000))])

VERTEX_CHUNK = {
    "responseId": "vertex-response-id",
    "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
}


class _RecordingGoogleGenAI:
    """Stands in for the client; records its config and yields one chunk."""

    def __init__(self, config):
        constructor_calls.append(dict(config))

    async def generate_content_stream(self, _params, *, env=None, cancel=None):
        yield VERTEX_CHUNK


constructor_calls: list[dict] = []


@contextlib.contextmanager
def _stubbed_client():
    original = google_vertex.GoogleGenAI
    google_vertex.GoogleGenAI = _RecordingGoogleGenAI
    try:
        yield
    finally:
        google_vertex.GoogleGenAI = original


@pytest.fixture(autouse=True)
def _isolate(request):
    constructor_calls.clear()
    original = os.environ.pop("GOOGLE_CLOUD_API_KEY", None)

    def restore():
        if original is None:
            os.environ.pop("GOOGLE_CLOUD_API_KEY", None)
        else:
            os.environ["GOOGLE_CLOUD_API_KEY"] = original

    request.addfinalizer(restore)


async def _run(model=None, **option_kwargs) -> None:
    options = google_vertex.GoogleVertexOptions(**option_kwargs)
    with _stubbed_client():
        await stream_google_vertex(model or MODEL, CONTEXT, options).result()


def _only_call() -> dict:
    assert len(constructor_calls) == 1
    return constructor_calls[0]


def _assert_matches(config: dict, expected: dict) -> None:
    """pi's `toMatchObject`: every expected entry present and equal."""
    for key, value in expected.items():
        assert config.get(key) == value, key


@pytest.mark.tonio
async def test_falls_back_to_adc_when_options_api_key_is_a_placeholder_marker():
    await _run(api_key="<authenticated>", project="test-project", location="us-central1")

    config = _only_call()
    _assert_matches(
        config, {"vertexai": True, "project": "test-project", "location": "us-central1", "apiVersion": "v1"}
    )
    assert "apiKey" not in config


@pytest.mark.tonio
async def test_falls_back_to_adc_when_options_api_key_is_the_gcp_vertex_credentials_marker():
    await _run(api_key="gcp-vertex-credentials", project="test-project", location="us-central1")

    config = _only_call()
    _assert_matches(
        config, {"vertexai": True, "project": "test-project", "location": "us-central1", "apiVersion": "v1"}
    )
    assert "apiKey" not in config


@pytest.mark.tonio
async def test_falls_back_to_adc_when_google_cloud_api_key_is_a_placeholder_marker():
    os.environ["GOOGLE_CLOUD_API_KEY"] = "<authenticated>"

    await _run(project="test-project", location="us-central1")

    config = _only_call()
    _assert_matches(
        config, {"vertexai": True, "project": "test-project", "location": "us-central1", "apiVersion": "v1"}
    )
    assert "apiKey" not in config


@pytest.mark.tonio
async def test_still_uses_the_api_key_client_for_real_api_keys():
    await _run(api_key="AIzaSyExampleRealisticLookingApiKey123456")

    config = _only_call()
    _assert_matches(
        config,
        {"vertexai": True, "apiKey": "AIzaSyExampleRealisticLookingApiKey123456", "apiVersion": "v1"},
    )
    assert "project" not in config
    assert "location" not in config


@pytest.mark.tonio
async def test_does_not_forward_generated_vertex_base_url_placeholders():
    await _run(project="test-project", location="us-central1")

    assert _only_call().get("httpOptions") is None


@pytest.mark.tonio
async def test_forwards_custom_base_url_to_the_adc_client():
    custom_model = replace(MODEL, base_url="https://proxy.example.com")
    await _run(custom_model, project="test-project", location="us-central1")

    config = _only_call()
    _assert_matches(
        config,
        {
            "vertexai": True,
            "project": "test-project",
            "location": "us-central1",
            "apiVersion": "v1",
            "httpOptions": {
                "baseUrl": "https://proxy.example.com",
                "baseUrlResourceScope": "COLLECTION",
            },
        },
    )


@pytest.mark.tonio
async def test_forwards_custom_base_url_to_the_api_key_client():
    custom_model = replace(MODEL, base_url="https://proxy.example.com")
    await _run(custom_model, api_key="AIzaSyExampleRealisticLookingApiKey123456")

    config = _only_call()
    _assert_matches(
        config,
        {
            "vertexai": True,
            "apiKey": "AIzaSyExampleRealisticLookingApiKey123456",
            "apiVersion": "v1",
            "httpOptions": {
                "baseUrl": "https://proxy.example.com",
                "baseUrlResourceScope": "COLLECTION",
            },
        },
    )


@pytest.mark.tonio
async def test_does_not_append_api_version_when_custom_base_url_already_includes_one():
    custom_model = replace(MODEL, base_url="https://proxy.example.com/v1/projects/test-project/locations/global")
    await _run(custom_model, project="test-project", location="us-central1")

    _assert_matches(
        _only_call(),
        {
            "httpOptions": {
                "baseUrl": "https://proxy.example.com/v1/projects/test-project/locations/global",
                "baseUrlResourceScope": "COLLECTION",
                "apiVersion": "",
            }
        },
    )
