"""Mirror of pi's openai-responses-compat.test.ts.

pi captures the headers the SDK puts on the wire by mocking global fetch;
pidrei inspects the transport built by `_create_client` directly — the same
observable request headers without a mocked network layer.
"""

import time

from pidrei_ai.api.openai_responses import OpenAIResponsesOptions, _create_client, build_params
from pidrei_ai.types import Context, OpenAIResponsesCompat, UserMessage
from tests.test_openai_responses import make_model


def make_context() -> Context:
    return Context(system_prompt="sys", messages=[UserMessage(content="hi", timestamp=int(time.time() * 1000))])


def transport_headers(model, session_id: str | None = "session-123", options_headers=None) -> dict:
    client = _create_client(model, make_context(), "test-key", options_headers, session_id)
    return client._headers


def test_omits_reasoning_when_no_reasoning_is_requested():
    # pi uses a github-copilot catalog model: the provider guard skips the
    # reasoning-off effort payload entirely.
    model = make_model(provider="github-copilot")
    params = build_params(model, make_context(), OpenAIResponsesOptions())
    assert "reasoning" not in params


def test_forwards_required_tool_choice():
    params = build_params(make_model(), make_context(), OpenAIResponsesOptions(tool_choice="required"))
    assert params["tool_choice"] == "required"


def test_clamps_prompt_cache_key_to_64_characters():
    params = build_params(make_model(), make_context(), OpenAIResponsesOptions(session_id="s" * 80))
    assert params["prompt_cache_key"] == "s" * 64


def test_sets_cache_affinity_headers_for_official_openai_requests():
    headers = transport_headers(make_model())
    assert headers["session_id"] == "session-123"
    assert headers["x-client-request-id"] == "session-123"
    assert "x-session-id" not in headers


def test_sets_cache_affinity_headers_for_proxy_requests():
    headers = transport_headers(make_model(base_url="https://my-proxy.example.com/v1"))
    assert headers["session_id"] == "session-123"
    assert headers["x-client-request-id"] == "session-123"


def test_uses_openrouter_session_affinity_header_when_configured():
    model = make_model(compat=OpenAIResponsesCompat(session_affinity_format="openrouter"))
    headers = transport_headers(model)
    assert headers["x-session-id"] == "session-123"
    assert "session_id" not in headers
    assert "x-client-request-id" not in headers


def test_auto_detects_openrouter_session_affinity_for_openrouter_endpoints():
    model = make_model(provider="openrouter", base_url="https://openrouter.ai/api/v1")
    headers = transport_headers(model)
    assert headers["x-session-id"] == "session-123"
    assert "session_id" not in headers


def test_uses_openai_nosession_format_when_configured():
    model = make_model(compat=OpenAIResponsesCompat(session_affinity_format="openai-nosession"))
    headers = transport_headers(model)
    assert "session_id" not in headers
    assert headers["x-client-request-id"] == "session-123"
    assert "x-session-id" not in headers


def test_explicit_headers_override_default_cache_affinity_headers():
    headers = transport_headers(make_model(), options_headers={"session_id": "explicit-session"})
    assert headers["session_id"] == "explicit-session"
    assert headers["x-client-request-id"] == "session-123"


def test_omits_cache_affinity_headers_when_session_is_absent():
    # The adapter passes session_id=None when cacheRetention is "none".
    headers = transport_headers(make_model(), session_id=None)
    assert "session_id" not in headers
    assert "x-client-request-id" not in headers
    assert "x-session-id" not in headers


def test_sets_strict_mode_explicitly_for_cloudflare_openai_responses_tools():
    from pidrei_ai.providers.all import get_builtin_model
    from pidrei_ai.types import JsonSchemaConstrainedSampling, Tool

    model = get_builtin_model("cloudflare-ai-gateway", "gpt-5.6-sol")
    assert model.compat is not None and model.compat.supports_strict_mode is True

    context = Context(
        messages=[UserMessage(content="Use a tool.", timestamp=1)],
        tools=[
            Tool(
                name="ordinary",
                description="An ordinary tool",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "offset": {"type": "number"}},
                    "required": ["path"],
                },
            ),
            Tool(
                name="constrained",
                description="A constrained tool",
                parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
                constrained_sampling=JsonSchemaConstrainedSampling(strict="prefer"),
            ),
        ],
    )
    params = build_params(model, context, OpenAIResponsesOptions())
    assert [(tool.get("name"), tool.get("strict")) for tool in params["tools"]] == [
        ("ordinary", False),
        ("constrained", True),
    ]
