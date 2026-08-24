"""Mirror of pi's mistral-tool-schema.test.ts.

pi's case exists because TypeBox attaches **symbol-keyed** metadata to schemas,
which the Mistral SDK's validator chokes on; `stripSymbolKeys` removes it. Python
has no such thing — `Tool.parameters` is a plain dict with string keys — so that
helper has no counterpart here and was deliberately not ported.

What survives the translation is the outcome pi's case actually guards: the tool
reaches the payload with `strict: true` and a schema that serializes cleanly,
nesting included.
"""

import json

import pytest

from pidrei_ai.api import mistral_conversations as mistral
from pidrei_ai.api.constrained_sampling import make_strict_json_schema
from pidrei_ai.api.mistral_conversations import stream as stream_mistral
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, JsonSchemaConstrainedSampling, Tool, UserMessage


captured: list[dict] = []


async def _capturing_on_payload(payload, _model):
    captured.append(payload)
    raise RuntimeError("payload captured")


@pytest.mark.tonio
async def test_tool_schemas_reach_the_payload_intact_and_strict():
    captured.clear()
    # pi picks `devstral-medium-latest`; models.dev retired every devstral model
    # before this catalog regen, so any live mistral-conversations model stands in.
    model = get_builtin_model("mistral", "mistral-medium-latest")
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

    response = await stream_mistral(
        model, context, mistral.MistralOptions(api_key="fake-key", on_payload=_capturing_on_payload)
    ).result()

    assert len(captured[0]["tools"]) == 1
    function = captured[0]["tools"][0]["function"]
    assert function["strict"] is True
    # Since 0.84.2 (7915cdac) strict tools carry the converted closed-object
    # schema, not the original; pi's test asserts a defined, symbol-free
    # payload — here the checks are the strict conversion and that the tool's
    # own definition stays untouched.
    assert function["parameters"] == make_strict_json_schema(parameters)
    assert context.tools[0].parameters == parameters
    # The payload must be JSON-serializable end to end, nesting included.
    wire = json.loads(json.dumps(mistral.to_mistral_wire_payload(captured[0])))
    assert wire["tools"][0]["function"]["parameters"] == function["parameters"]
    assert response.stop_reason == "error"
