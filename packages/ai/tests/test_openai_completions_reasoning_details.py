"""Mirror of pi's openai-completions-reasoning-details.test.ts.

pi replaces the OpenAI SDK with `vi.mock` and queues one chunk set per request;
here the injected client does the same and records every request payload.
"""

import json
from dataclasses import replace

import pytest

from pidrei_ai.api.openai_completions import OpenAICompletionsOptions, stream as stream_completions
from pidrei_ai.types import AssistantMessage, Context, Model, ModelCost, Tool
from tests.test_openai_completions import FakeResponse, chunk_body


REASONING_DETAIL = {"type": "reasoning.encrypted", "id": "call_1", "data": "encrypted-signature"}
SIGNED_REASONING_TEXT_DETAIL = {
    "type": "reasoning.text",
    "text": "I should call the read tool.",
    "signature": "sha256:signed-text",
    "id": "reasoning-text-1",
    "format": "anthropic-claude-v1",
    "index": 0,
}
REASONING_SUMMARY_DETAIL = {
    "type": "reasoning.summary",
    "summary": "Decided to inspect the requested file.",
    "id": "reasoning-summary-1",
    "format": "anthropic-claude-v1",
    "index": 1,
}
READ_TOOL = Tool(
    name="read",
    description="Read a file",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
)


def model() -> Model:
    return Model(
        id="google/gemini-test",
        name="Gemini Test",
        api="openai-completions",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        reasoning=True,
        input=["text"],
        cost=ModelCost(),
        context_window=100_000,
        max_tokens=4096,
    )


def chunk(delta: dict, finish_reason: str | None = None) -> dict:
    return {
        "id": "chatcmpl-test",
        "model": "google/gemini-test",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def tool_call_chunk() -> dict:
    return chunk(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":"README.md"}'},
                }
            ]
        }
    )


class QueuedClient:
    """One chunk set per request, in order; every payload is recorded."""

    def __init__(self, chunk_sets: list[list[dict]]):
        self._chunk_sets = list(chunk_sets)
        self.payloads: list[dict] = []

    async def create(self, params, *, timeout_ms, cancel):
        self.payloads.append(params)
        chunks = self._chunk_sets.pop(0) if self._chunk_sets else []
        return FakeResponse(chunk_body(chunks))


async def run_stream(client: QueuedClient, messages: list | None = None) -> AssistantMessage:
    context = Context(messages=list(messages or []), tools=[READ_TOOL])
    return await stream_completions(model(), context, OpenAICompletionsOptions(api_key="test", client=client)).result()


def assistant_payload(payload: dict) -> dict | None:
    return next((message for message in payload.get("messages") or [] if message.get("role") == "assistant"), None)


def thinking_block(message: AssistantMessage):
    return next(block for block in message.content if block.type == "thinking")


def tool_call_chunk_sets() -> list[list[dict]]:
    return [
        [chunk({"reasoning_details": [REASONING_DETAIL]}), tool_call_chunk(), chunk({}, "tool_calls")],
        [chunk({"content": "ok"}), chunk({}, "stop")],
    ]


@pytest.mark.tonio
async def test_preserves_reasoning_details_in_the_thinking_signature():
    client = QueuedClient(tool_call_chunk_sets())

    assistant_message = await run_stream(client)

    thinking = thinking_block(assistant_message)
    assert (thinking.thinking, thinking.thinking_signature) == ("", json.dumps([REASONING_DETAIL]))
    tool_call = next(block for block in assistant_message.content if block.type == "toolCall")
    assert (tool_call.id, tool_call.name, tool_call.arguments) == ("call_1", "read", {"path": "README.md"})
    assert tool_call.thought_signature is None

    await run_stream(client, [assistant_message])

    assert assistant_payload(client.payloads[1])["reasoning_details"] == [REASONING_DETAIL]


@pytest.mark.tonio
async def test_falls_back_to_encrypted_tool_call_signatures_for_older_stored_assistant_messages():
    client = QueuedClient(tool_call_chunk_sets())

    assistant_message = await run_stream(client)
    # Step 2 relaxation (PROPER_MT_DESIGN.md): messages are frozen values now,
    # so the older stored shape is built by construction instead of mutation.
    blocks = [
        replace(block, thought_signature=json.dumps(REASONING_DETAIL)) if block.type == "toolCall" else block
        for block in assistant_message.content
        if block.type != "thinking"
    ]
    assistant_message = replace(assistant_message, content=blocks)

    await run_stream(client, [assistant_message])

    assert assistant_payload(client.payloads[1])["reasoning_details"] == [REASONING_DETAIL]


@pytest.mark.tonio
async def test_preserves_signed_text_and_summary_reasoning_details_in_their_original_sequence():
    client = QueuedClient(
        [
            [
                chunk(
                    {
                        "reasoning": SIGNED_REASONING_TEXT_DETAIL["text"],
                        "reasoning_details": [SIGNED_REASONING_TEXT_DETAIL],
                    }
                ),
                chunk({"reasoning_details": [REASONING_DETAIL, REASONING_SUMMARY_DETAIL]}),
                tool_call_chunk(),
                chunk({}, "tool_calls"),
            ],
            [chunk({"content": "ok"}), chunk({}, "stop")],
        ]
    )

    assistant_message = await run_stream(client)
    expected = [SIGNED_REASONING_TEXT_DETAIL, REASONING_DETAIL, REASONING_SUMMARY_DETAIL]
    thinking = thinking_block(assistant_message)
    assert thinking.thinking == SIGNED_REASONING_TEXT_DETAIL["text"]
    assert thinking.thinking_signature == json.dumps(expected)

    await run_stream(client, [assistant_message])

    payload = assistant_payload(client.payloads[1])
    assert payload["reasoning_details"] == expected
    assert "reasoning" not in payload


@pytest.mark.tonio
async def test_merges_consecutive_text_and_summary_reasoning_details_deltas_before_replay():
    text_delta = {"type": "reasoning.text", "text": "The", "index": 0}
    text_delta_with_signature = {
        "type": "reasoning.text",
        "text": " user wants the time.",
        "signature": "sha256:text-signature",
        "format": "openai-responses-v1",
        "index": 0,
    }
    summary_delta = {"type": "reasoning.summary", "summary": "Looked", "index": 0}
    summary_delta_with_format = {
        "type": "reasoning.summary",
        "summary": " up time.",
        "format": "openai-responses-v1",
        "index": 0,
    }
    later_summary_delta = {
        "type": "reasoning.summary",
        "summary": "After encrypted block.",
        "format": "openai-responses-v1",
        "index": 0,
    }
    expected = [
        {
            "type": "reasoning.text",
            "text": "The user wants the time.",
            "index": 0,
            "signature": "sha256:text-signature",
            "format": "openai-responses-v1",
        },
        {
            "type": "reasoning.summary",
            "summary": "Looked up time.",
            "index": 0,
            "format": "openai-responses-v1",
        },
        REASONING_DETAIL,
        later_summary_delta,
    ]

    client = QueuedClient(
        [
            [
                chunk({"reasoning_details": [text_delta]}),
                chunk({"reasoning_details": [text_delta_with_signature]}),
                chunk({"reasoning_details": [summary_delta]}),
                chunk({"reasoning_details": [summary_delta_with_format]}),
                chunk({"reasoning_details": [REASONING_DETAIL]}),
                chunk({"reasoning_details": [later_summary_delta]}),
                tool_call_chunk(),
                chunk({}, "tool_calls"),
            ],
            [chunk({"content": "ok"}), chunk({}, "stop")],
        ]
    )

    assistant_message = await run_stream(client)
    thinking = thinking_block(assistant_message)
    assert (thinking.thinking, thinking.thinking_signature) == ("", json.dumps(expected))

    await run_stream(client, [assistant_message])

    assert assistant_payload(client.payloads[1])["reasoning_details"] == expected
