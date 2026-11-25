"""Mirror of pi's claude-model-generation.test.ts.

pi runs the generator script end to end against a mocked models.dev catalog
and reads the provider JSON it writes. Here the whole catalog build runs
in-process against the same mocked sources (`catalog_generation_utils`), as
`test_fireworks_model_generation.py` does.
"""

from typing import Any

from pidrei_ai.models_generated import parse_model_dict
from pidrei_ai.registry import get_supported_thinking_levels
from tests.catalog_generation_utils import generate_chat_models


def _claude_model(id: str, reasoning: dict[str, Any], canonical_id: str | None = None) -> dict[str, Any]:
    return {
        "id": id,
        **({"canonical_id": canonical_id} if canonical_id else {}),
        "type": "chat",
        "name": id,
        "modalities": {"input": ["text"], "output": ["text"]},
        "capabilities": {"tools": {"supported": True}, "reasoning": reasoning},
        "api": {},
    }


def _levels(models: dict[str, dict], model_id: str) -> list[str]:
    return get_supported_thinking_levels(parse_model_dict(models[model_id]))


EFFORT = ["low", "medium", "high", "xhigh", "max"]


# Copilot lists no thinking toggle on any Claude model; Sonnet 5.5 rejects `disabled` there (#12429).
def test_follows_the_thinking_toggle_of_anthropics_own_models_dev_entry_on_other_hosts():
    models = generate_chat_models(
        {
            "anthropic": {
                "models": {
                    "claude-sonnet-5-5": _claude_model("claude-sonnet-5-5", {"supported": True, "effort": EFFORT}),
                    "claude-haiku-5-5": _claude_model(
                        "claude-haiku-5-5", {"supported": True, "toggle": True, "effort": EFFORT}
                    ),
                },
            },
            "github-copilot": {
                "models": {
                    "claude-sonnet-5.5": _claude_model(
                        "claude-sonnet-5.5", {"supported": True, "effort": EFFORT}, "anthropic/claude-sonnet-5-5"
                    ),
                    "claude-haiku-5.5": _claude_model(
                        "claude-haiku-5.5", {"supported": True, "effort": EFFORT}, "anthropic/claude-haiku-5-5"
                    ),
                },
            },
        },
        "github-copilot",
    )
    assert "off" not in _levels(models, "claude-sonnet-5.5")
    assert "off" in _levels(models, "claude-haiku-5.5")


def test_finds_anthropics_entry_through_canonical_id_when_the_host_id_does_not_resemble_anthropics():
    models = generate_chat_models(
        {
            "anthropic": {
                "models": {
                    "claude-sonnet-5-5": _claude_model("claude-sonnet-5-5", {"supported": True, "effort": EFFORT})
                },
            },
            "amazon-bedrock": {
                "models": {
                    "us.anthropic.claude-sonnet-5-5": _claude_model(
                        "us.anthropic.claude-sonnet-5-5",
                        {"supported": True, "effort": EFFORT},
                        "anthropic/claude-sonnet-5-5",
                    ),
                },
            },
        },
        "amazon-bedrock",
    )
    assert "off" not in _levels(models, "us.anthropic.claude-sonnet-5-5")


def test_keeps_thinking_off_unsupported_on_the_authored_fable_5_2_when_models_dev_does_not_list_it():
    models = generate_chat_models({}, "anthropic")
    assert "off" not in _levels(models, "claude-fable-5-2")
