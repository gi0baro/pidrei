"""Mirror of pi's bedrock-thinking-payload.test.ts.

pi captures the command input from `onPayload` and aborts the request by throwing
out of the callback; the same trick works here. The credential-gated max-tokens
E2E block is not mirrored, as with pi's other live suites.
"""

import contextlib
from dataclasses import replace
from typing import Any

import pytest

from pidrei_ai.api import bedrock_converse_stream as bedrock
from pidrei_ai.api.bedrock_converse_stream import BedrockOptions, stream as stream_bedrock
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, UserMessage


class PayloadCaptured(Exception):
    """Aborts the request once the payload has been seen."""


def make_context() -> Context:
    return Context(
        system_prompt="You are helpful.",
        messages=[UserMessage(content="Hello", timestamp=1)],
    )


class _UnusedClient:
    """Never reached: `on_payload` raises before the client is sent to."""

    def __init__(self, _config):
        self.middleware_stack = _NullStack()

    async def send(self, _command, *, cancel=None):
        raise AssertionError("the payload capture should abort before send")


class _NullStack:
    def add(self, *_args, **_kwargs) -> None:
        pass


@contextlib.contextmanager
def _stubbed_client():
    original = bedrock.BedrockRuntimeClient
    bedrock.BedrockRuntimeClient = _UnusedClient
    try:
        yield
    finally:
        bedrock.BedrockRuntimeClient = original


async def capture_payload(model, options: BedrockOptions | None = None) -> dict[str, Any]:
    captured: list[dict] = []
    opts = options or BedrockOptions()
    if opts.reasoning is None:
        opts.reasoning = "high"

    async def on_payload(payload, _model):
        captured.append(payload)
        raise PayloadCaptured

    opts.on_payload = on_payload

    with _stubbed_client():
        stream = stream_bedrock(model, make_context(), opts)
        async for event in stream:
            if event.type == "error":
                break

    assert captured, "Expected Bedrock payload to be captured before request abort"
    return captured[0]


def _opus_48():
    base = get_builtin_model("amazon-bedrock", "global.anthropic.claude-opus-4-6-v1")
    return replace(base, id="global.anthropic.claude-opus-4-8-v1", name="Claude Opus 4.8 (Global)")


@pytest.mark.tonio
async def test_uses_adaptive_thinking_for_claude_opus_4_8_when_reasoning_is_enabled():
    payload = await capture_payload(_opus_48())

    fields = payload["additionalModelRequestFields"]
    assert fields["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert fields["output_config"] == {"effort": "high"}
    assert fields.get("anthropic_beta") is None


@pytest.mark.tonio
async def test_maps_xhigh_reasoning_to_effort_xhigh_for_claude_opus_4_8():
    payload = await capture_payload(_opus_48(), BedrockOptions(reasoning="xhigh"))

    fields = payload["additionalModelRequestFields"]
    assert fields["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert fields["output_config"] == {"effort": "xhigh"}
    assert fields.get("anthropic_beta") is None


@pytest.mark.tonio
async def test_uses_adaptive_thinking_for_claude_fable_5_when_reasoning_is_enabled():
    payload = await capture_payload(get_builtin_model("amazon-bedrock", "global.anthropic.claude-fable-5"))

    fields = payload["additionalModelRequestFields"]
    assert fields["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert fields["output_config"] == {"effort": "high"}
    assert fields.get("anthropic_beta") is None


@pytest.mark.tonio
async def test_uses_adaptive_thinking_for_claude_sonnet_5_when_reasoning_is_enabled():
    payload = await capture_payload(get_builtin_model("amazon-bedrock", "global.anthropic.claude-sonnet-5"))

    fields = payload["additionalModelRequestFields"]
    assert fields["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert fields["output_config"] == {"effort": "high"}
    assert fields.get("anthropic_beta") is None


@pytest.mark.tonio
async def test_maps_xhigh_reasoning_to_effort_xhigh_for_claude_fable_5():
    payload = await capture_payload(
        get_builtin_model("amazon-bedrock", "global.anthropic.claude-fable-5"),
        BedrockOptions(reasoning="xhigh"),
    )

    fields = payload["additionalModelRequestFields"]
    assert fields["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert fields["output_config"] == {"effort": "xhigh"}


@pytest.mark.tonio
async def test_omits_display_for_govcloud_model_ids_on_non_adaptive_claude_thinking():
    base = get_builtin_model("amazon-bedrock", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    model = replace(
        base,
        id="us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0",
        name="Claude Sonnet 4.5 (GovCloud)",
    )

    payload = await capture_payload(model)

    fields = payload["additionalModelRequestFields"]
    assert fields["thinking"] == {"type": "enabled", "budget_tokens": 16384}
    assert fields["anthropic_beta"] == ["interleaved-thinking-2025-05-14"]


@pytest.mark.tonio
async def test_omits_display_for_govcloud_regions_on_adaptive_claude_thinking():
    payload = await capture_payload(_opus_48(), BedrockOptions(region="us-gov-west-1"))

    fields = payload["additionalModelRequestFields"]
    assert fields["thinking"] == {"type": "adaptive"}
    assert fields["output_config"] == {"effort": "high"}
    assert fields.get("anthropic_beta") is None


# --- application inference profiles -------------------------------------------


@pytest.mark.tonio
async def test_uses_adaptive_thinking_when_model_name_carries_it_but_the_arn_does_not():
    base = get_builtin_model("amazon-bedrock", "global.anthropic.claude-opus-4-6-v1")
    model = replace(
        base,
        id="arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-profile",
        name="Claude Opus 4.6",
    )

    payload = await capture_payload(model)

    fields = payload["additionalModelRequestFields"]
    assert fields["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert fields["output_config"] == {"effort": "high"}


@pytest.mark.tonio
async def test_injects_cache_points_when_model_name_identifies_a_supported_claude_model():
    base = get_builtin_model("amazon-bedrock", "global.anthropic.claude-opus-4-6-v1")
    model = replace(
        base,
        id="arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-profile",
        name="Claude Sonnet 4.6",
    )

    payload = await capture_payload(model, BedrockOptions())

    # System prompt should have a cache point
    assert len(payload["system"]) == 2
    assert "cachePoint" in payload["system"][1]

    # Last user message should have a cache point
    last_message = payload["messages"][-1]
    assert "cachePoint" in last_message["content"][-1]


@pytest.mark.tonio
async def test_falls_back_to_fixed_budget_thinking_for_non_adaptive_claude_via_model_name():
    base = get_builtin_model("amazon-bedrock", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    model = replace(
        base,
        id="arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-profile",
        name="Claude Sonnet 4.5",
    )

    payload = await capture_payload(model)

    thinking = payload["additionalModelRequestFields"]["thinking"]
    assert thinking["type"] == "enabled"
    assert isinstance(thinking["budget_tokens"], int)
    assert payload["additionalModelRequestFields"]["anthropic_beta"] == ["interleaved-thinking-2025-05-14"]
