"""Mirror of pi's mistral-tool-schema.test.ts.

pi's case exists because TypeBox attaches **symbol-keyed** metadata to schemas,
which the Mistral SDK's validator chokes on; `stripSymbolKeys` removes it. Python
has no such thing — `Tool.parameters` is a plain dict with string keys — so that
helper has no counterpart here and was deliberately not ported.

What survives the translation is the outcome pi's case actually guards: the tool
reaches the payload with `strict: true` and a schema that serializes cleanly,
nesting included.
"""

import contextlib
import json

import pytest

from pidrei_ai.api import mistral_conversations as mistral
from pidrei_ai.api.mistral_conversations import stream as stream_mistral
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, JsonSchemaConstrainedSampling, Tool, UserMessage


captured: list[dict] = []


class _CapturingClient:
    def __init__(self, _api_key, _server_url=None, env=None):
        pass

    async def chat_stream(self, payload, *, headers=None, cancel=None, timeout_ms=None):
        captured.append(payload)
        raise RuntimeError("payload captured")


@contextlib.contextmanager
def _stubbed_client():
    original = mistral.MistralClient
    mistral.MistralClient = _CapturingClient
    try:
        yield
    finally:
        mistral.MistralClient = original


@pytest.mark.tonio
async def test_tool_schemas_reach_the_payload_intact_and_strict():
    captured.clear()
    model = get_builtin_model("mistral", "devstral-medium-latest")
    parameters = {
        "type": "object",
        "properties": {"nested": {"type": "object", "properties": {"value": {"type": "string"}}}},
        "required": ["nested"],
    }
    context = Context(
        messages=[UserMessage(content="Hi", timestamp=1)],
        tools=[
            Tool(
                name="inspect_schema",
                description="Inspect the schema",
                parameters=parameters,
                constrained_sampling=JsonSchemaConstrainedSampling(strict="require"),
            )
        ],
    )

    with _stubbed_client():
        response = await stream_mistral(model, context, mistral.MistralOptions(api_key="fake-key")).result()

    assert len(captured[0]["tools"]) == 1
    function = captured[0]["tools"][0]["function"]
    assert function["strict"] is True
    assert function["parameters"] == parameters
    # The payload must be JSON-serializable end to end, nesting included.
    assert (
        json.loads(json.dumps(mistral.to_wire_payload(captured[0])))["tools"][0]["function"]["parameters"] == parameters
    )
    assert response.stop_reason == "error"
