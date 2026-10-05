"""Mirror of pi's azure-openai-base-url.test.ts.

pi replaces the `openai` package with `vi.mock` and asserts on the `AzureOpenAI`
constructor config and on the params handed to `responses.create`. Here the
client is `api/azure_openai_responses.AzureOpenAI`, so the stub replaces the same
thing, and because that class keeps the SDK's config keys the assertions stay
pi's verbatim.
"""

import contextlib
import json
import os
from dataclasses import replace

import pytest

from pidrei_ai.api import azure_openai_responses as azure
from pidrei_ai.api.azure_openai_responses import AzureOpenAIResponsesOptions, stream as stream_azure
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import (
    Context,
    JsonSchemaConstrainedSampling,
    OpenAIResponsesCompat,
    Tool,
    UserMessage,
)
from pidrei_ai.utils.transcript import normalize_context
from pidrei_ai.utils.user_agent import get_user_agent


CONTEXT = normalize_context(Context(messages=[UserMessage(content="hello", timestamp=1)]))

constructor_calls: list[dict] = []
last_params: list[dict] = []
# pi: azureMock.streamEvents — when set, `create` streams them instead of throwing.
stream_events: list[list[dict]] = []


class _RecordingAzureOpenAI:
    def __init__(self, config):
        constructor_calls.append(dict(config))
        self.responses = _RecordingResponses()


class _StreamResponse:
    def __init__(self, events: list[dict]):
        self.status = 200
        self.headers: dict[str, str] = {}
        self._body = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()

    async def aiter_bytes(self):
        yield self._body


class _RecordingResponses:
    async def create(self, params, *, timeout_ms=None, cancel=None):
        last_params.append(params)
        if not stream_events:
            raise RuntimeError("mock create")
        return _StreamResponse(stream_events[0])


@contextlib.contextmanager
def _stubbed_client():
    original = azure.AzureOpenAI
    azure.AzureOpenAI = _RecordingAzureOpenAI
    try:
        yield
    finally:
        azure.AzureOpenAI = original


_MANAGED_ENV = (
    "AZURE_OPENAI_BASE_URL",
    "AZURE_OPENAI_RESOURCE_NAME",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_API_KEY",
)


@pytest.fixture(autouse=True)
def _isolate(request):
    constructor_calls.clear()
    last_params.clear()
    stream_events.clear()
    saved = {name: os.environ.get(name) for name in _MANAGED_ENV}
    for name in _MANAGED_ENV:
        os.environ.pop(name, None)

    def restore():
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    request.addfinalizer(restore)


def model_fixture():
    return get_builtin_model("azure", "gpt-4o-mini")


async def capture_client_base_url(base_url: str) -> str:
    os.environ["AZURE_OPENAI_BASE_URL"] = base_url
    with _stubbed_client():
        await stream_azure(model_fixture(), CONTEXT, AzureOpenAIResponsesOptions(api_key="test-api-key")).result()
    assert len(constructor_calls) == 1
    return constructor_calls[0]["baseURL"]


@pytest.mark.tonio
async def test_normalizes_cognitive_services_root_endpoints_to_openai_v1():
    base_url = await capture_client_base_url("https://marc-quicktests-resource.cognitiveservices.azure.com")
    assert base_url == "https://marc-quicktests-resource.cognitiveservices.azure.com/openai/v1"


@pytest.mark.tonio
async def test_normalizes_microsoft_foundry_root_endpoints_to_openai_v1():
    base_url = await capture_client_base_url("https://marc-quicktests-resource.ai.azure.com")
    assert base_url == "https://marc-quicktests-resource.ai.azure.com/openai/v1"


@pytest.mark.tonio
async def test_normalizes_azure_openai_root_endpoints_to_openai_v1():
    base_url = await capture_client_base_url("https://my-resource.openai.azure.com")
    assert base_url == "https://my-resource.openai.azure.com/openai/v1"


@pytest.mark.tonio
async def test_normalizes_openai_to_openai_v1():
    base_url = await capture_client_base_url("https://my-resource.cognitiveservices.azure.com/openai")
    assert base_url == "https://my-resource.cognitiveservices.azure.com/openai/v1"


@pytest.mark.tonio
async def test_preserves_openai_v1_endpoints():
    base_url = await capture_client_base_url("https://my-resource.cognitiveservices.azure.com/openai/v1")
    assert base_url == "https://my-resource.cognitiveservices.azure.com/openai/v1"


@pytest.mark.tonio
async def test_normalizes_openai_v1_responses_to_openai_v1():
    base_url = await capture_client_base_url("https://my-resource.services.ai.azure.com/openai/v1/responses")
    assert base_url == "https://my-resource.services.ai.azure.com/openai/v1"


@pytest.mark.tonio
async def test_preserves_explicit_non_azure_proxy_paths():
    base_url = await capture_client_base_url("https://my-proxy.example.com/v1")
    assert base_url == "https://my-proxy.example.com/v1"


@pytest.mark.tonio
async def test_strips_query_params_when_normalizing_azure_host_urls():
    base_url = await capture_client_base_url("https://my-resource.openai.azure.com/openai?api-version=2024-12-01")
    assert base_url == "https://my-resource.openai.azure.com/openai/v1"


@pytest.mark.tonio
async def test_preserves_query_params_on_non_azure_proxy_urls():
    base_url = await capture_client_base_url("https://my-proxy.example.com/v1?custom=true")
    assert base_url == "https://my-proxy.example.com/v1?custom=true"


@pytest.mark.tonio
async def test_throws_on_invalid_urls():
    os.environ["AZURE_OPENAI_BASE_URL"] = "not-a-url"
    with _stubbed_client():
        result = await stream_azure(
            model_fixture(), CONTEXT, AzureOpenAIResponsesOptions(api_key="test-api-key")
        ).result()

    assert result.stop_reason == "error"
    assert "Invalid Azure OpenAI base URL" in result.error_message


@pytest.mark.tonio
async def test_clamps_prompt_cache_key_to_openais_64_character_limit():
    with _stubbed_client():
        await stream_azure(
            model_fixture(),
            CONTEXT,
            AzureOpenAIResponsesOptions(
                api_key="test-api-key",
                azure_base_url="https://my-resource.openai.azure.com",
                session_id="x" * 67,
            ),
        ).result()

    assert last_params[0]["prompt_cache_key"] == "x" * 64


@pytest.mark.tonio
async def test_disables_server_side_response_storage():
    with _stubbed_client():
        await stream_azure(
            model_fixture(),
            CONTEXT,
            AzureOpenAIResponsesOptions(api_key="test-api-key", azure_base_url="https://my-resource.openai.azure.com"),
        ).result()

    assert last_params[0]["store"] is False


@pytest.mark.tonio
async def test_honors_supports_strict_mode_false():
    base_model = model_fixture()
    compat = base_model.compat or OpenAIResponsesCompat()
    model = replace(base_model, compat=replace(compat, supports_strict_mode=False))

    context = normalize_context(
        Context(
            messages=CONTEXT.messages,
            tools=[
                Tool(
                    name="preferred",
                    description="Preferred constrained tool",
                    parameters={"type": "object", "properties": {"value": {"type": "string"}}},
                    constrained_sampling=JsonSchemaConstrainedSampling(strict="prefer"),
                )
            ],
        )
    )

    with _stubbed_client():
        await stream_azure(
            model,
            context,
            AzureOpenAIResponsesOptions(api_key="test-api-key", azure_base_url="https://my-resource.openai.azure.com"),
        ).result()

    assert "strict" not in last_params[0]["tools"][0]


@pytest.mark.tonio
async def test_builds_correct_default_url_from_azure_openai_resource_name():
    os.environ["AZURE_OPENAI_RESOURCE_NAME"] = "my-resource"
    with _stubbed_client():
        await stream_azure(model_fixture(), CONTEXT, AzureOpenAIResponsesOptions(api_key="test-api-key")).result()

    assert len(constructor_calls) == 1
    assert constructor_calls[0]["baseURL"] == "https://my-resource.openai.azure.com/openai/v1"


# --- provider stream events --------------------------------------------------


@pytest.mark.tonio
async def test_forwards_parsed_events_in_order_before_normalizing_the_response():
    # pi also asserts identity with the mocked SDK's event objects; here the
    # events are parsed from the SSE body, so equality is the observable.
    events = [
        {"type": "response.created", "sequence_number": 0, "response": {"id": "resp_azure"}},
        {
            "type": "response.completed",
            "sequence_number": 1,
            "response": {"id": "resp_azure", "status": "completed"},
        },
    ]
    stream_events.append(events)
    model = model_fixture()
    received: list = []
    event_models: list = []

    async def on_provider_stream_event(event, event_model) -> None:
        received.append(event)
        event_models.append(event_model)

    with _stubbed_client():
        result = await stream_azure(
            model,
            CONTEXT,
            AzureOpenAIResponsesOptions(
                api_key="test-api-key",
                azure_base_url="https://my-resource.openai.azure.com",
                on_provider_stream_event=on_provider_stream_event,
            ),
        ).result()

    assert received == events
    assert event_models == [model, model]
    assert result.stop_reason == "stop"
    assert result.response_id == "resp_azure"


async def capture_client_headers(headers: dict | None = None) -> dict:
    with _stubbed_client():
        await stream_azure(
            model_fixture(),
            CONTEXT,
            AzureOpenAIResponsesOptions(
                api_key="test-api-key", azure_base_url="https://my-resource.openai.azure.com", headers=headers
            ),
        ).result()
    assert len(constructor_calls) == 1
    return constructor_calls[0].get("defaultHeaders") or {}


@pytest.mark.tonio
async def test_uses_the_runtime_user_agent_by_default():
    assert (await capture_client_headers())["User-Agent"] == get_user_agent()


@pytest.mark.tonio
async def test_lets_explicit_headers_override_the_default_user_agent():
    assert (await capture_client_headers({"User-Agent": "custom-agent"}))["User-Agent"] == "custom-agent"
