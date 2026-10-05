"""Mirror of pi's azure-openai-completions.test.ts.

pi replaces the `openai` package with `vi.mock`: a fake `OpenAI` records the
client options and the Chat Completions params, a fake `AzureOpenAI` marks a
Responses dispatch. Here the same two seams are the Chat Completions adapter's
default client (`openai_completions._PunkreqOpenAIClient`) and the Responses
adapter's `AzureOpenAI`, swapped with `monkeypatch`.
"""

import json
from dataclasses import dataclass

import pytest

from pidrei_ai.api import azure_openai_responses, openai_completions
from pidrei_ai.api.openai_completions import OpenAICompletionsOptions
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.providers.azure import azure_provider
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    SimpleStreamOptions,
    StreamOptions,
    SystemMessage,
    TextContent,
    ThinkingContent,
    Usage,
    UserMessage,
)
from pidrei_ai.utils.transcript import normalize_context


@dataclass(slots=True)
class _MockState:
    last_params: dict | None = None
    last_client_base_url: str | None = None
    dispatched_to: str | None = None


class _FakeResponse:
    def __init__(self) -> None:
        self.status = 200
        self.headers: dict[str, str] = {}

    async def aiter_bytes(self):
        chunk = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 0}},
        }
        yield f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()


@pytest.fixture
def mock_state(monkeypatch) -> _MockState:
    state = _MockState()

    class FakeOpenAI:
        def __init__(self, base_url, headers, env=None):
            state.last_client_base_url = base_url

        async def create(self, params, *, timeout_ms, cancel):
            state.last_params = params
            state.dispatched_to = "chat.completions"
            return _FakeResponse()

    class FakeAzureResponses:
        async def create(self, params, *, timeout_ms=None, cancel=None):
            state.dispatched_to = "responses"
            raise RuntimeError("responses reached")

    class FakeAzureOpenAI:
        def __init__(self, config):
            self.responses = FakeAzureResponses()

    monkeypatch.setattr(openai_completions, "_PunkreqOpenAIClient", FakeOpenAI)
    monkeypatch.setattr(azure_openai_responses, "AzureOpenAI", FakeAzureOpenAI)
    monkeypatch.delenv("PIDREI_CACHE_RETENTION", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://my-resource.services.ai.azure.com")
    return state


azure = azure_provider()

CONTEXT = normalize_context(
    Context(system_prompt="sys", messages=[UserMessage(content="hi", timestamp=0)]),
)


def deep_seek_model():
    return get_builtin_model("azure", "deepseek-v4-pro")


def _assert_matches(value: dict | None, expected: dict) -> None:
    """pi's `toMatchObject`: every expected entry present and equal."""
    assert value is not None
    for key, item in expected.items():
        assert value.get(key) == item, key


# Regression for #9645: Azure Foundry rejects DeepSeek's thinking field and every prompt cache parameter here.


@pytest.mark.tonio
async def test_turns_thinking_on_with_reasoning_effort_instead_of_deepseeks_thinking_field(mock_state):
    await azure.stream_simple(
        deep_seek_model(), CONTEXT, SimpleStreamOptions(api_key="test-key", reasoning="high")
    ).result()

    assert mock_state.last_params["reasoning_effort"] == "high"
    assert "thinking" not in mock_state.last_params


@pytest.mark.tonio
async def test_clamps_thinking_levels_the_deployment_does_not_accept(mock_state):
    await azure.stream_simple(
        deep_seek_model(), CONTEXT, SimpleStreamOptions(api_key="test-key", reasoning="max")
    ).result()

    assert mock_state.last_params["reasoning_effort"] == "high"


@pytest.mark.tonio
async def test_sends_no_reasoning_effort_when_no_thinking_level_is_requested(mock_state):
    await azure.stream_simple(deep_seek_model(), CONTEXT, SimpleStreamOptions(api_key="test-key")).result()

    assert "reasoning_effort" not in mock_state.last_params
    assert "thinking" not in mock_state.last_params


@pytest.mark.tonio
async def test_omits_prompt_cache_parameters_when_long_retention_comes_from_pidrei_cache_retention(
    mock_state, monkeypatch
):
    monkeypatch.setenv("PIDREI_CACHE_RETENTION", "long")

    await azure.stream(deep_seek_model(), CONTEXT, StreamOptions(api_key="test-key", session_id="session-env")).result()

    assert "prompt_cache_key" not in mock_state.last_params
    assert "prompt_cache_retention" not in mock_state.last_params


# The deployment discards a `developer` system message once reasoning_effort is set, without
# billing it, so the system prompt has to go out under the system role.
@pytest.mark.tonio
async def test_sends_the_system_prompt_under_the_system_role(mock_state):
    await azure.stream(
        deep_seek_model(), CONTEXT, OpenAICompletionsOptions(api_key="test-key", reasoning_effort="low")
    ).result()

    _assert_matches(mock_state.last_params["messages"][0], {"role": "system", "content": "sys"})


# The deployment honours system messages sent mid-conversation, so pidrei must not collapse them.
@pytest.mark.tonio
async def test_keeps_mid_conversation_system_messages_in_place(mock_state):
    resumed = normalize_context(
        Context(
            system_prompt="first",
            messages=[
                UserMessage(content="hi", timestamp=0),
                SystemMessage(content="second", timestamp=0),
                UserMessage(content="again", timestamp=0),
            ],
        )
    )

    await azure.stream(deep_seek_model(), resumed, StreamOptions(api_key="test-key")).result()

    assert [message["role"] for message in mock_state.last_params["messages"]] == ["system", "user", "system", "user"]


@pytest.mark.tonio
async def test_omits_prompt_cache_parameters_even_when_long_retention_is_requested(mock_state):
    await azure.stream(
        deep_seek_model(),
        CONTEXT,
        StreamOptions(api_key="test-key", cache_retention="long", session_id="session-1"),
    ).result()

    assert "prompt_cache_key" not in mock_state.last_params
    assert "prompt_cache_retention" not in mock_state.last_params


@pytest.mark.tonio
async def test_replays_reasoning_content_on_assistant_turns_so_the_cached_prefix_is_unchanged(mock_state):
    assistant = AssistantMessage(
        content=[
            ThinkingContent(thinking="internal reasoning", thinking_signature="reasoning_content"),
            TextContent(text="answer"),
        ],
        provider="azure",
        api="openai-completions",
        model="deepseek-v4-pro",
        timestamp=0,
        usage=Usage(),
        stop_reason="stop",
    )
    resumed = Context(
        system_prompt="sys",
        messages=[
            UserMessage(content="first", timestamp=0),
            assistant,
            UserMessage(content="second", timestamp=0),
        ],
    )

    await azure.stream(deep_seek_model(), normalize_context(resumed), StreamOptions(api_key="test-key")).result()

    assistant_message = next(
        message for message in mock_state.last_params["messages"] if message["role"] == "assistant"
    )
    assert assistant_message.get("reasoning_content") == "internal reasoning"


# -- endpoint resolution --------------------------------------------------------


@pytest.mark.tonio
async def test_normalizes_the_azure_endpoint_the_completions_client_is_built_with(mock_state):
    await azure.stream(deep_seek_model(), CONTEXT, StreamOptions(api_key="test-key")).result()

    assert mock_state.last_client_base_url == "https://my-resource.services.ai.azure.com/openai/v1"


@pytest.mark.tonio
async def test_surfaces_an_unconfigured_endpoint_as_an_error_event_rather_than_raising_out_of_stream(
    mock_state, monkeypatch
):
    monkeypatch.delenv("AZURE_OPENAI_BASE_URL")

    result = await azure.stream(deep_seek_model(), CONTEXT, StreamOptions(api_key="test-key")).result()

    assert result.stop_reason == "error"
    assert "Azure OpenAI base URL is required" in result.error_message


# The id is persisted on the assistant message and read back by name, so it has to stay a catalog id.
@pytest.mark.tonio
async def test_sends_the_model_id_as_the_request_model(mock_state):
    result = await azure.stream(deep_seek_model(), CONTEXT, StreamOptions(api_key="test-key")).result()

    assert mock_state.last_params["model"] == "deepseek-v4-pro"
    assert result.model == "deepseek-v4-pro"


@pytest.mark.tonio
async def test_sends_the_mapped_deployment_name_while_keeping_the_catalog_id_on_the_message(mock_state, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", "deepseek-v4-pro=my-deepseek")

    result = await azure.stream_simple(deep_seek_model(), CONTEXT, SimpleStreamOptions(api_key="test-key")).result()

    assert mock_state.last_params["model"] == "my-deepseek"
    assert result.model == "deepseek-v4-pro"


@pytest.mark.tonio
async def test_passes_the_deployment_name_through_a_callers_on_payload(mock_state, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", "deepseek-v4-pro=my-deepseek")
    seen: dict = {}

    async def on_payload(payload, _model):
        seen["model"] = payload["model"]
        return {**payload, "temperature": 0.1}

    await azure.stream(deep_seek_model(), CONTEXT, StreamOptions(api_key="test-key", on_payload=on_payload)).result()

    assert seen["model"] == "my-deepseek"
    _assert_matches(mock_state.last_params, {"model": "my-deepseek", "temperature": 0.1})


# -- api map ---------------------------------------------------------------------


@pytest.mark.tonio
async def test_still_routes_responses_models_to_the_responses_api(mock_state):
    await azure.stream(get_builtin_model("azure", "gpt-4o-mini"), CONTEXT, StreamOptions(api_key="test-key")).result()

    assert mock_state.dispatched_to == "responses"


@pytest.mark.tonio
async def test_routes_openai_completions_models_to_chat_completions(mock_state):
    await azure.stream(deep_seek_model(), CONTEXT, StreamOptions(api_key="test-key")).result()

    assert mock_state.dispatched_to == "chat.completions"
