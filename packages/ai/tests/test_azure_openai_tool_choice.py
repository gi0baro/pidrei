"""Mirror of pi's azure-openai-tool-choice.test.ts.

The payload is captured through `on_payload`, which raises to stop the request
before any transport work — the same interception pi uses, so the fake base URL
is never dialled.
"""

import pytest

from pidrei_ai.api.azure_openai_responses import AzureOpenAIResponsesOptions, stream, stream_simple
from pidrei_ai.types import Context, Model, ModelCost, SimpleStreamOptions, Tool, UserMessage


MODEL = Model(
    id="test-deployment",
    name="Test Deployment",
    api="azure-openai-responses",
    provider="azure-openai-responses",
    base_url="http://127.0.0.1:9/openai/v1",
    reasoning=False,
    input=["text"],
    cost=ModelCost(),
    context_window=10_000,
    max_tokens=1_000,
)


def _context() -> Context:
    return Context(
        messages=[UserMessage(content="Summarize this", timestamp=1)],
        tools=[
            Tool(
                name="read",
                description="Read a file",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            )
        ],
    )


def _payload_capture() -> tuple:
    captured: list = []

    async def on_payload(payload, _model):
        captured.append(payload)
        raise RuntimeError("payload captured")

    return on_payload, captured


@pytest.mark.tonio
async def test_forwards_provider_specific_tool_choice_while_preserving_tool_definitions():
    on_payload, captured = _payload_capture()

    await stream(
        MODEL,
        _context(),
        AzureOpenAIResponsesOptions(api_key="test-key", tool_choice="required", on_payload=on_payload),
    ).result()

    assert len(captured) == 1
    assert captured[0]["tool_choice"] == "required"
    assert len(captured[0]["tools"]) == 1


@pytest.mark.tonio
async def test_forwards_provider_neutral_tool_choice_from_simple_options():
    on_payload, captured = _payload_capture()

    await stream_simple(
        MODEL,
        _context(),
        SimpleStreamOptions(api_key="test-key", tool_choice="none", on_payload=on_payload),
    ).result()

    assert len(captured) == 1
    assert captured[0]["tool_choice"] == "none"
    assert len(captured[0]["tools"]) == 1
