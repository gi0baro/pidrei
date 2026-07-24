"""Mirror of pi's anthropic-thinking-disable.test.ts (payload suite; the
ANTHROPIC_API_KEY-gated E2E block is not mirrored)."""

import pytest

from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import SimpleStreamOptions
from tests.anthropic_helpers import capture_payload


@pytest.mark.tonio
async def test_sends_thinking_disabled_for_budget_based_reasoning_models_when_thinking_is_off():
    payload = await capture_payload(get_builtin_model("anthropic", "claude-sonnet-4-5"))

    assert payload["thinking"] == {"type": "disabled"}
    assert "output_config" not in payload


@pytest.mark.tonio
async def test_sends_thinking_disabled_for_adaptive_reasoning_models_when_thinking_is_off():
    payload = await capture_payload(get_builtin_model("anthropic", "claude-opus-4-6"))

    assert payload["thinking"] == {"type": "disabled"}
    assert "output_config" not in payload


@pytest.mark.tonio
async def test_sends_thinking_disabled_for_claude_opus_4_8_when_thinking_is_off():
    payload = await capture_payload(get_builtin_model("anthropic", "claude-opus-4-8"))

    assert payload["thinking"] == {"type": "disabled"}
    assert "output_config" not in payload


@pytest.mark.tonio
async def test_omits_thinking_disabled_for_claude_fable_5_when_thinking_is_off():
    payload = await capture_payload(get_builtin_model("anthropic", "claude-fable-5"))

    assert "thinking" not in payload
    assert "output_config" not in payload


@pytest.mark.tonio
async def test_uses_adaptive_thinking_for_claude_opus_4_8_when_reasoning_is_enabled():
    payload = await capture_payload(
        get_builtin_model("anthropic", "claude-opus-4-8"), SimpleStreamOptions(reasoning="high")
    )

    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "high"}


@pytest.mark.tonio
async def test_uses_adaptive_thinking_for_claude_sonnet_5_when_reasoning_is_enabled():
    payload = await capture_payload(
        get_builtin_model("anthropic", "claude-sonnet-5"), SimpleStreamOptions(reasoning="high")
    )

    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "high"}


@pytest.mark.tonio
async def test_maps_xhigh_reasoning_to_effort_xhigh_for_claude_opus_4_8():
    payload = await capture_payload(
        get_builtin_model("anthropic", "claude-opus-4-8"), SimpleStreamOptions(reasoning="xhigh")
    )

    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "xhigh"}
