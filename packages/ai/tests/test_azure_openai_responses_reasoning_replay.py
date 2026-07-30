"""Mirror of pi's azure-openai-responses-reasoning-replay.test.ts.

Exercises the shared Responses stream processor directly (no client involved):
`response.completed` must not clobber an `encrypted_content` that
`response.output_item.done` already carried, but must fill one in when it did not.
"""

import pytest

from pidrei_ai.api.openai_responses_shared import convert_responses_messages, process_responses_stream
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    Usage,
    UserMessage,
)
from pidrei_ai.utils.event_stream import AssistantMessageEventStream


def create_model() -> Model:
    return Model(
        id="gpt-5-mini",
        name="GPT-5 Mini",
        api="azure-openai-responses",
        provider="azure-openai-responses",
        base_url="https://example.invalid",
        reasoning=True,
        input=["text"],
        cost=ModelCost(),
        context_window=400000,
        max_tokens=128000,
    )


def create_output(model: Model) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="pending",
        timestamp=1,
    )


async def create_events(done_item: dict, completed_item: dict):
    yield {
        "type": "response.output_item.added",
        "output_index": 0,
        "sequence_number": 0,
        "item": {"type": "reasoning", "id": done_item["id"], "summary": []},
    }
    yield {
        "type": "response.output_item.done",
        "output_index": 0,
        "sequence_number": 1,
        "item": done_item,
    }
    yield {
        "type": "response.completed",
        "sequence_number": 2,
        "response": {"id": "resp_test", "status": "completed", "output": [completed_item]},
    }


def get_replayed_reasoning(model: Model, assistant: AssistantMessage) -> dict | None:
    context = Context(
        messages=[
            UserMessage(content="first", timestamp=1),
            assistant,
            UserMessage(content="follow-up", timestamp=2),
        ]
    )
    input_items = convert_responses_messages(model, context, {"azure-openai-responses"})
    return next((item for item in input_items if item.get("type") == "reasoning"), None)


@pytest.mark.tonio
async def test_preserves_existing_encrypted_content_from_output_item_done():
    model = create_model()
    output = create_output(model)
    done_item = {
        "type": "reasoning",
        "id": "rs_done",
        "summary": [],
        "encrypted_content": "from-output-item-done",
    }
    completed_item = {**done_item, "encrypted_content": "from-response-completed"}

    await process_responses_stream(
        create_events(done_item, completed_item), output, AssistantMessageEventStream(), model
    )

    reasoning = get_replayed_reasoning(model, output)
    assert reasoning is not None
    assert reasoning["id"] == "rs_done"
    assert reasoning["encrypted_content"] == "from-output-item-done"


@pytest.mark.tonio
async def test_fills_encrypted_content_when_output_item_done_omitted_it():
    model = create_model()
    output = create_output(model)
    done_item = {"type": "reasoning", "id": "rs_missing", "summary": []}
    completed_item = {**done_item, "encrypted_content": "from-response-completed"}

    await process_responses_stream(
        create_events(done_item, completed_item), output, AssistantMessageEventStream(), model
    )

    reasoning = get_replayed_reasoning(model, output)
    assert reasoning is not None
    assert reasoning["id"] == "rs_missing"
    assert reasoning["encrypted_content"] == "from-response-completed"
