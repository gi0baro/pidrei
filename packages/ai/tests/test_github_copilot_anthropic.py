"""Mirror of pi's github-copilot-anthropic.test.ts.

pi asserts on the mocked SDK client's constructor options; here the adapter's own
transport records the headers it built (`anthropic_helpers.capture_request`), so
the assertions read the same values off the real header-assembly path.
"""

import pytest

from pidrei_ai.api.anthropic_messages import AnthropicOptions
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.registry import get_supported_thinking_levels
from pidrei_ai.types import Context, SimpleStreamOptions, UserMessage

from .anthropic_helpers import capture_request, now_ms


def copilot_context() -> Context:
    return Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content="Hello", timestamp=now_ms())],
    )


def test_applies_copilot_specific_adaptive_thinking_effort_overrides():
    opus47 = get_builtin_model("github-copilot", "claude-opus-4.7")
    assert opus47 is not None
    assert opus47.thinking_level_map is not None
    assert opus47.thinking_level_map["minimal"] == "low"
    assert opus47.thinking_level_map["xhigh"] == "xhigh"
    assert opus47.thinking_level_map["max"] == "max"
    assert "xhigh" in get_supported_thinking_levels(opus47)
    assert "max" in get_supported_thinking_levels(opus47)

    sonnet46 = get_builtin_model("github-copilot", "claude-sonnet-4.6")
    assert sonnet46 is not None
    assert sonnet46.thinking_level_map is not None
    assert sonnet46.thinking_level_map["minimal"] == "low"
    assert sonnet46.thinking_level_map["max"] == "max"
    assert "max" in get_supported_thinking_levels(sonnet46)
    assert "xhigh" not in get_supported_thinking_levels(sonnet46)


@pytest.mark.tonio
async def test_uses_bearer_auth_copilot_headers_and_a_valid_anthropic_messages_payload():
    model = get_builtin_model("github-copilot", "claude-sonnet-4.6")
    assert model is not None and model.api == "anthropic-messages"

    headers, payload = await capture_request(
        model,
        SimpleStreamOptions(api_key="tid_copilot_session_test_token"),
        copilot_context(),
    )

    # Auth: Bearer, never x-api-key.
    assert headers["authorization"] == "Bearer tid_copilot_session_test_token"
    assert "x-api-key" not in headers

    # Copilot static headers from model.headers.
    assert "GitHubCopilotChat" in headers["User-Agent"]
    assert headers["Copilot-Integration-Id"] == "vscode-chat"

    # Dynamic headers.
    assert headers["X-Initiator"] == "user"
    assert headers["Openai-Intent"] == "conversation-edits"

    # No fine-grained-tool-streaming (Copilot does not support it).
    assert "fine-grained-tool-streaming" not in headers.get("anthropic-beta", "")

    assert payload["model"] == "claude-sonnet-4.6"
    assert payload["stream"] is True
    assert payload["max_tokens"] == model.max_tokens
    assert isinstance(payload["messages"], list)


@pytest.mark.tonio
async def test_omits_interleaved_thinking_beta_for_adaptive_thinking_models():
    model = get_builtin_model("github-copilot", "claude-sonnet-4.6")
    assert model is not None

    headers, _payload = await capture_request(
        model,
        AnthropicOptions(api_key="tid_copilot_session_test_token", interleaved_thinking=True),
        copilot_context(),
    )

    assert "interleaved-thinking-2025-05-14" not in headers.get("anthropic-beta", "")
