"""Tests for the openai-completions adapter.

Includes the mirror of pi's openai-completions-retry.test.ts (SDK mock →
client injection; pi's "disables SDK retries" case is structural in pidrei —
the transport has no built-in retries) plus adapter-level streaming, compat
detection, and params-builder coverage that pi exercises via e2e suites.
"""

import json
import time

import pytest

from pidrei_ai.api.openai_completions import (
    OpenAICompletionsOptions,
    build_params,
    detect_compat,
    get_compat,
    stream as stream_completions,
)
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    OpenAICompletionsCompat,
    TextContent,
    ThinkingContent,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def make_model(provider="openai", base_url="https://api.openai.com/v1", **overrides) -> Model:
    defaults: dict = {
        "id": "test-model",
        "name": "Test Model",
        "api": "openai-completions",
        "provider": provider,
        "base_url": base_url,
        "reasoning": False,
        "input": ["text"],
        "cost": ModelCost(),
        "context_window": 100_000,
        "max_tokens": 8_000,
    }
    defaults.update(overrides)
    return Model(**defaults)


def user_context(text: str = "hi") -> Context:
    return Context(messages=[UserMessage(content=text, timestamp=int(time.time() * 1000))])


def chunk_body(chunks: list[dict], include_done: bool = True) -> bytes:
    parts = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    if include_done:
        parts.append("data: [DONE]\n\n")
    return "".join(parts).encode()


def text_chunks() -> list[dict]:
    return [
        {"id": "chatcmpl-test", "choices": [{"index": 0, "delta": {"content": "ok"}}]},
        {"id": "chatcmpl-test", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200):
        self.status = status
        self.headers = {"content-type": "text/event-stream"}
        self._body = body

    async def aiter_bytes(self):
        yield self._body


class FakeClient:
    def __init__(self, body: bytes | None = None, errors: list[Exception] | None = None):
        self._body = body if body is not None else chunk_body(text_chunks())
        self._errors = list(errors or [])
        self.requests: list[dict] = []

    async def create(self, params, *, timeout_ms, cancel):
        self.requests.append(params)
        if self._errors:
            raise self._errors.pop(0)
        return FakeResponse(self._body)


class ProviderError(Exception):
    def __init__(self, message: str, status: int, headers: dict[str, str]):
        super().__init__(message)
        self.status = status
        self.headers = headers


async def consume(client: FakeClient, model: Model | None = None, **option_overrides) -> AssistantMessage:
    model = model if model is not None else make_model()
    options = OpenAICompletionsOptions(api_key="test", client=client, **option_overrides)
    stream = stream_completions(model, user_context(), options)
    async for _event in stream:
        pass
    return await stream.result()


# --- retry mirror (openai-completions-retry.test.ts) --------------------------


@pytest.mark.tonio
async def test_sends_a_single_request_by_default():
    client = FakeClient()
    result = await consume(client)
    assert result.stop_reason == "stop"
    assert len(client.requests) == 1


@pytest.mark.tonio
async def test_honors_provider_retries():
    client = FakeClient(
        errors=[
            ProviderError("rate limited", 429, {"retry-after-ms": "20"}),
            ProviderError("server error", 500, {"retry-after-ms": "20"}),
        ]
    )
    result = await consume(client, max_retries=2, max_retry_delay_ms=100)

    assert result.stop_reason == "stop"
    assert result.content == [TextContent(text="ok")]
    assert len(client.requests) == 3


@pytest.mark.tonio
async def test_fails_immediately_when_provider_requested_delay_exceeds_the_limit():
    client = FakeClient(errors=[ProviderError("rate limited", 429, {"retry-after": "277403"})])
    result = await consume(client, max_retries=2, max_retry_delay_ms=1000)

    assert result.stop_reason == "error"
    assert "Server requested 277403s retry delay (max: 1s)" in result.error_message
    assert "rate limited" in result.error_message
    assert len(client.requests) == 1


# --- streaming ----------------------------------------------------------------


@pytest.mark.tonio
async def test_assembles_text_and_captures_response_metadata():
    chunks = [
        {"id": "chatcmpl-1", "model": "actual-model", "choices": [{"index": 0, "delta": {"content": "Hel"}}]},
        {"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {"content": "lo"}}]},
        {"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    result = await consume(FakeClient(chunk_body(chunks)))

    assert result.content == [TextContent(text="Hello")]
    assert result.response_id == "chatcmpl-1"
    assert result.response_model == "actual-model"
    assert result.stop_reason == "stop"


@pytest.mark.tonio
async def test_streams_tool_calls_by_index_and_parses_arguments():
    chunks = [
        {
            "id": "c",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "echo", "arguments": '{"te'}}]
                    },
                }
            ],
        },
        {
            "id": "c",
            "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'xt": "hi"}'}}]}}],
        },
        {"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]

    events: list[str] = []
    model = make_model()
    stream = stream_completions(
        model, user_context(), OpenAICompletionsOptions(api_key="k", client=FakeClient(chunk_body(chunks)))
    )
    async for event in stream:
        events.append(event.type)
    result = await stream.result()

    assert result.stop_reason == "toolUse"
    tool_call = result.content[0]
    assert tool_call.type == "toolCall"
    assert tool_call.id == "call_1"
    assert tool_call.name == "echo"
    assert tool_call.arguments == {"text": "hi"}
    assert "toolcall_start" in events
    assert "toolcall_delta" in events
    assert "toolcall_end" in events


@pytest.mark.tonio
async def test_parses_usage_with_cache_and_reasoning_details():
    chunks = [
        {"id": "c", "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]},
        {
            "id": "c",
            "choices": [],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 100},
                "completion_tokens_details": {"reasoning_tokens": 20},
            },
        },
    ]
    result = await consume(FakeClient(chunk_body(chunks)))

    assert result.usage.input == 300  # 1000 - 600 read - 100 write
    assert result.usage.cache_read == 600
    assert result.usage.cache_write == 100
    assert result.usage.output == 50
    assert result.usage.reasoning == 20
    assert result.usage.total_tokens == 1050


@pytest.mark.tonio
async def test_usage_in_choice_fallback():
    chunks = [
        {
            "id": "c",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "ok"},
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                }
            ],
        },
    ]
    result = await consume(FakeClient(chunk_body(chunks)))
    assert result.usage.input == 7
    assert result.usage.output == 3


@pytest.mark.tonio
async def test_reasoning_content_becomes_thinking_block_with_signature():
    chunks = [
        {"id": "c", "choices": [{"index": 0, "delta": {"reasoning_content": "let me think"}}]},
        {"id": "c", "choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": "stop"}]},
    ]
    result = await consume(FakeClient(chunk_body(chunks)))

    assert result.content[0] == ThinkingContent(thinking="let me think", thinking_signature="reasoning_content")
    assert result.content[1] == TextContent(text="answer")


@pytest.mark.tonio
async def test_ignores_empty_custom_objects_on_function_tool_call_deltas():
    chunks = [
        {
            "id": "chatcmpl-empty-custom",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read", "arguments": '{"path":"README.md"}'},
                                "custom": {},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
    ]
    result = await consume(FakeClient(chunk_body(chunks)))

    assert len(result.content) == 1
    block = result.content[0]
    assert block.type == "toolCall"
    assert block.id == "call_1"
    assert block.name == "read"
    assert block.arguments == {"path": "README.md"}
    assert result.stop_reason == "toolUse"


@pytest.mark.tonio
async def test_missing_finish_reason_is_an_error():
    chunks = [{"id": "c", "choices": [{"index": 0, "delta": {"content": "ok"}}]}]
    result = await consume(FakeClient(chunk_body(chunks)))

    assert result.stop_reason == "error"
    assert result.error_message == "Stream ended without finish_reason"


@pytest.mark.tonio
async def test_accepts_streams_without_finish_reason_when_compat_disables_it():
    chunks = [
        {
            "id": "chatcmpl-no-finish-reason",
            "choices": [{"index": 0, "delta": {"content": "complete answer"}, "finish_reason": None}],
        }
    ]
    model = make_model(compat=OpenAICompletionsCompat(supports_finish_reason=False))
    result = await consume(FakeClient(chunk_body(chunks)), model)

    assert result.stop_reason == "stop"
    assert result.error_message is None
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert result.content[0].text == "complete answer"


@pytest.mark.tonio
async def test_error_chunks_become_stream_errors():
    body = b'data: {"error": {"message": "upstream exploded"}}\n\n'
    result = await consume(FakeClient(body))

    assert result.stop_reason == "error"
    assert "upstream exploded" in result.error_message


@pytest.mark.tonio
async def test_content_filter_finish_reason_maps_to_error():
    chunks = [{"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "content_filter"}]}]
    result = await consume(FakeClient(chunk_body(chunks)))

    assert result.stop_reason == "error"
    assert result.error_message == "Provider finish_reason: content_filter"


# --- compat detection ---------------------------------------------------------


def test_detect_compat_openai_defaults():
    compat = detect_compat(make_model())
    assert compat.supports_store is True
    assert compat.supports_developer_role is True
    assert compat.supports_reasoning_effort is True
    assert compat.max_tokens_field == "max_completion_tokens"
    assert compat.thinking_format == "openai"
    assert compat.supports_strict_mode is True
    assert compat.session_affinity_format == "openai"


def test_detect_compat_deepseek():
    compat = detect_compat(make_model(provider="deepseek", base_url="https://api.deepseek.com"))
    assert compat.supports_store is False
    assert compat.thinking_format == "deepseek"
    assert compat.requires_reasoning_content_on_assistant_messages is True


def test_detect_compat_openrouter_anthropic_models():
    model = make_model(provider="openrouter", base_url="https://openrouter.ai/api/v1", id="anthropic/claude-x")
    compat = detect_compat(model)
    assert compat.thinking_format == "openrouter"
    assert compat.cache_control_format == "anthropic"
    assert compat.supports_developer_role is True  # anthropic/ prefixed models
    assert compat.session_affinity_format == "openrouter"

    other = detect_compat(make_model(provider="openrouter", base_url="https://openrouter.ai/api/v1", id="meta/llama"))
    assert other.supports_developer_role is False
    assert other.cache_control_format is None


def test_detect_compat_moonshot_and_together():
    moonshot = detect_compat(make_model(provider="moonshotai", base_url="https://api.moonshot.ai/v1"))
    assert moonshot.max_tokens_field == "max_tokens"
    assert moonshot.supports_strict_mode is False
    assert moonshot.supports_reasoning_effort is False

    together = detect_compat(make_model(provider="together", base_url="https://api.together.ai/v1"))
    assert together.thinking_format == "together"
    assert together.supports_long_cache_retention is False


def test_model_compat_overrides_detection():
    model = make_model(compat=OpenAICompletionsCompat(supports_store=False, thinking_format="qwen"))
    compat = get_compat(model)
    assert compat.supports_store is False
    assert compat.thinking_format == "qwen"
    assert compat.supports_developer_role is True  # still detected


# --- params builder -----------------------------------------------------------


def opts(**kwargs) -> OpenAICompletionsOptions:
    return OpenAICompletionsOptions(**kwargs)


def test_build_params_openai_defaults():
    model = make_model()
    params = build_params(model, user_context(), opts(max_tokens=100, session_id="s" * 80))

    assert params["model"] == "test-model"
    assert params["stream"] is True
    assert params["store"] is False
    assert params["stream_options"] == {"include_usage": True}
    assert params["max_completion_tokens"] == 100
    # openai base URL + short retention: prompt_cache_key clamped to 64 chars.
    assert params["prompt_cache_key"] == "s" * 64
    assert "prompt_cache_retention" not in params


def test_build_params_long_retention():
    model = make_model()
    params = build_params(model, user_context(), opts(session_id="sess", cache_retention="long"))
    assert params["prompt_cache_key"] == "sess"
    assert params["prompt_cache_retention"] == "24h"


def test_build_params_max_tokens_field_for_moonshot():
    model = make_model(provider="moonshotai", base_url="https://api.moonshot.ai/v1")
    params = build_params(model, user_context(), opts(max_tokens=42))
    assert params["max_tokens"] == 42
    assert "max_completion_tokens" not in params


def test_sends_max_tokens_for_zai_completions_models():
    from pidrei_ai.providers.all import get_builtin_model

    # Regenerated Z.AI catalog entries carry the compat directly.
    for model_id in ("glm-5-turbo", "glm-5.2"):
        catalog_model = get_builtin_model("zai", model_id)
        assert catalog_model is not None
        assert catalog_model.compat is not None and catalog_model.compat.max_tokens_field == "max_tokens"

    # detect_compat stays as the fallback for custom/self-hosted Z.AI base URLs.
    model = make_model(provider="zai", base_url="https://api.z.ai/api/paas/v4")
    assert detect_compat(model).max_tokens_field == "max_tokens"
    params = build_params(model, user_context(), opts(max_tokens=123))
    assert params["max_tokens"] == 123
    assert "max_completion_tokens" not in params


def test_build_params_reasoning_effort_mapping():
    model = make_model(reasoning=True, thinking_level_map={"high": "very-high"})
    params = build_params(model, user_context(), opts(reasoning_effort="high"))
    assert params["reasoning_effort"] == "very-high"

    unmapped = build_params(model, user_context(), opts(reasoning_effort="low"))
    assert unmapped["reasoning_effort"] == "low"

    off_string = make_model(reasoning=True, thinking_level_map={"off": "none"})
    params = build_params(off_string, user_context(), opts())
    assert params["reasoning_effort"] == "none"


def test_build_params_deepseek_thinking():
    model = make_model(provider="deepseek", base_url="https://api.deepseek.com", reasoning=True)
    enabled = build_params(model, user_context(), opts(reasoning_effort="high"))
    assert enabled["thinking"] == {"type": "enabled"}

    disabled = build_params(model, user_context(), opts())
    assert disabled["thinking"] == {"type": "disabled"}

    off_null = make_model(
        provider="deepseek", base_url="https://api.deepseek.com", reasoning=True, thinking_level_map={"off": None}
    )
    no_thinking = build_params(off_null, user_context(), opts())
    assert "thinking" not in no_thinking


def test_build_params_openrouter_reasoning_and_cache_control():
    model = make_model(
        provider="openrouter", base_url="https://openrouter.ai/api/v1", id="anthropic/claude-x", reasoning=True
    )
    context = Context(
        system_prompt="sys",
        messages=[UserMessage(content="hello", timestamp=1)],
        tools=[],
    )
    params = build_params(model, context, opts(reasoning_effort="medium"))
    assert params["reasoning"] == {"effort": "medium"}

    # Anthropic-style cache_control: system prompt + last conversation message.
    system_message = params["messages"][0]
    assert system_message["content"][0]["cache_control"] == {"type": "ephemeral"}
    last_message = params["messages"][-1]
    assert last_message["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_build_params_empty_tools_for_tool_history():
    model = make_model()
    context = Context(
        messages=[
            UserMessage(content="go", timestamp=1),
            ToolResultMessage(
                tool_call_id="t1", tool_name="tool", content=[TextContent(text="out")], is_error=False, timestamp=2
            ),
        ]
    )
    params = build_params(model, context, opts())
    assert params["tools"] == []


# --- message conversion (through build_params) --------------------------------


def assistant_message(content, model: Model) -> AssistantMessage:
    return AssistantMessage(
        content=content,
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="stop",
        timestamp=1,
    )


def test_developer_role_for_reasoning_models():
    model = make_model(reasoning=True)
    params = build_params(model, Context(system_prompt="sys", messages=[UserMessage(content="q", timestamp=1)]), opts())
    assert params["messages"][0] == {"role": "developer", "content": "sys"}

    non_reasoning = build_params(
        make_model(), Context(system_prompt="sys", messages=[UserMessage(content="q", timestamp=1)]), opts()
    )
    assert non_reasoning["messages"][0] == {"role": "system", "content": "sys"}


def test_requires_assistant_after_tool_result_bridges_user_messages():
    model = make_model(compat=OpenAICompletionsCompat(requires_assistant_after_tool_result=True))
    context = Context(
        messages=[
            UserMessage(content="go", timestamp=1),
            ToolResultMessage(
                tool_call_id="t1", tool_name="tool", content=[TextContent(text="out")], is_error=False, timestamp=2
            ),
            UserMessage(content="next", timestamp=3),
        ]
    )
    params = build_params(model, context, opts())
    roles = [message["role"] for message in params["messages"]]
    assert roles == ["user", "tool", "assistant", "user"]
    assert params["messages"][2]["content"] == "I have processed the tool results."


def test_tool_result_name_field_when_required():
    model = make_model(compat=OpenAICompletionsCompat(requires_tool_result_name=True))
    context = Context(
        messages=[
            UserMessage(content="go", timestamp=1),
            ToolResultMessage(
                tool_call_id="t1", tool_name="mytool", content=[TextContent(text="out")], is_error=False, timestamp=2
            ),
        ]
    )
    params = build_params(model, context, opts())
    tool_message = next(message for message in params["messages"] if message["role"] == "tool")
    assert tool_message["name"] == "mytool"


def test_pipe_separated_tool_ids_are_normalized():
    model = make_model()
    call_id = "call_abc|fc_" + "x" * 10
    other_model_message = AssistantMessage(
        content=[
            # Cross-model so normalize_tool_call_id applies.
            *[],
        ],
        api="openai-responses",
        provider="other",
        model="other-model",
        usage=Usage(),
        stop_reason="toolUse",
        timestamp=1,
    )
    from pidrei_ai.types import ToolCall

    other_model_message.content = [ToolCall(id=call_id, name="tool", arguments={})]
    context = Context(
        messages=[
            UserMessage(content="go", timestamp=1),
            other_model_message,
            ToolResultMessage(
                tool_call_id=call_id, tool_name="tool", content=[TextContent(text="out")], is_error=False, timestamp=2
            ),
        ]
    )
    params = build_params(model, context, opts())
    assistant = next(message for message in params["messages"] if message["role"] == "assistant")
    tool_result = next(message for message in params["messages"] if message["role"] == "tool")

    normalized = assistant["tool_calls"][0]["id"]
    assert "|" not in normalized
    assert normalized == "call_abc_fc_" + "x" * 10
    assert tool_result["tool_call_id"] == normalized


def test_thinking_replay_as_field_and_as_text():
    model = make_model()
    thinking = ThinkingContent(thinking="deep", thinking_signature="reasoning_content")
    context = Context(
        messages=[
            UserMessage(content="q", timestamp=1),
            assistant_message([thinking, TextContent(text="ans")], model),
            UserMessage(content="next", timestamp=2),
        ]
    )
    params = build_params(model, context, opts())
    assistant = next(message for message in params["messages"] if message["role"] == "assistant")
    assert assistant["content"] == "ans"
    assert assistant["reasoning_content"] == "deep"

    as_text_model = make_model(compat=OpenAICompletionsCompat(requires_thinking_as_text=True))
    context_same = Context(
        messages=[
            UserMessage(content="q", timestamp=1),
            assistant_message([thinking, TextContent(text="ans")], as_text_model),
            UserMessage(content="next", timestamp=2),
        ]
    )
    params = build_params(as_text_model, context_same, opts())
    assistant = next(message for message in params["messages"] if message["role"] == "assistant")
    assert assistant["content"][0] == {"type": "text", "text": "deep"}
    assert assistant["content"][1] == {"type": "text", "text": "ans"}


def test_empty_assistant_messages_are_skipped():
    model = make_model()
    context = Context(
        messages=[
            UserMessage(content="q", timestamp=1),
            assistant_message([], model),
            UserMessage(content="next", timestamp=2),
        ]
    )
    params = build_params(model, context, opts())
    assert [message["role"] for message in params["messages"]] == ["user", "user"]
