"""Mirror of pi's anthropic-adaptive-thinking-models.test.ts.

pi's expected list spans providers pidrei has not wired yet (cloudflare,
kimi-coding, opencode, vercel); those entries join with their providers.
"""

import re

from pidrei_ai.models_generated import MODELS
from pidrei_ai.types import AnthropicMessagesCompat


EXPECTED_CURRENT_ADAPTIVE_THINKING_MODELS = [
    "anthropic/claude-fable-5",
    "anthropic/claude-opus-4-8",
    "anthropic/claude-opus-5",
    "anthropic/claude-sonnet-5",
]

_ADAPTIVE_ID_PATTERN = re.compile(r"(opus[-.](4[-.][678]|5)|sonnet[-.]4[-.]6|sonnet[-.]5|fable[-.]5|kimi-coding/)")


def get_all_models():
    return [model for models in MODELS.values() for model in models]


def test_marks_builtin_anthropic_messages_models_that_use_adaptive_thinking():
    flagged_models = sorted(
        f"{model.provider}/{model.id}"
        for model in get_all_models()
        if model.api == "anthropic-messages"
        and isinstance(model.compat, AnthropicMessagesCompat)
        and model.compat.force_adaptive_thinking is True
    )

    assert set(EXPECTED_CURRENT_ADAPTIVE_THINKING_MODELS) <= set(flagged_models)
    assert flagged_models == [model_id for model_id in flagged_models if _ADAPTIVE_ID_PATTERN.search(model_id)]
