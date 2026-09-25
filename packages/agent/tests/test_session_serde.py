"""pidrei-only: session serde round-trips.

pi persists messages by serializing the whole object, so every field survives
a reload for free. pidrei's serde names each field, and a field missing there
is silently dropped from session JSONL and `--mode json` output.
"""

import json

from pidrei_agent.harness.session.serde import parse_message, serialize_message
from pidrei_ai.types import AssistantMessage, TextContent, Usage


def test_assistant_provider_thinking_level_survives_a_reload():
    # Anthropic replays each assistant turn's effort from this field after a resume.
    message = AssistantMessage(
        content=[TextContent(text="answer")],
        api="anthropic-messages",
        provider="anthropic",
        model="claude-fable-5-1",
        usage=Usage(),
        stop_reason="stop",
        timestamp=1,
        provider_thinking_level="high",
    )

    data = json.loads(json.dumps(serialize_message(message)))

    assert data["providerThinkingLevel"] == "high"
    assert parse_message(data).provider_thinking_level == "high"
