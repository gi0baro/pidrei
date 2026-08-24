"""Mirror of pi's bedrock-redacted-reasoning.test.ts.

OpenAI models served through Bedrock Converse (e.g. `global.openai.gpt-5.6-terra`)
return encrypted reasoning as the opaque `redactedContent` member of
`reasoningContent`, not as `reasoningText`. The AWS SDK decodes the wire blob to
bytes.
https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ReasoningContentBlockDelta.html

pi replaces `@aws-sdk/client-bedrock-runtime` with `vi.mock`; here the stub
replaces `api/bedrock_runtime.BedrockRuntimeClient` by name, as in the other
bedrock mirrors. pi's `"redactedChunks" in thinking` / `"index" in thinking`
assertions guard scratch fields it stashes on the block itself; `ThinkingContent`
is a slotted dataclass and pidrei keeps that scratch in a side table, so those
assertions could never fail and are not mirrored.
"""

import base64
import contextlib
from types import SimpleNamespace

import pytest

from pidrei_ai.api import bedrock_converse_stream as bedrock
from pidrei_ai.api.bedrock_converse_stream import BedrockOptions, stream as stream_bedrock
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pidrei_ai.utils.cancel import CancelToken


REDACTED_BASE64 = "cnNuXzVaVnJpZjRKMGJYSXFtV2RsZWRqN1FJRmVOaWtSUWJF"
REDACTED_BYTES = base64.b64decode(REDACTED_BASE64)

GPT_MODEL = Model(
    id="global.openai.gpt-5.6-terra",
    name="GPT-5.6 Terra (Global)",
    api="bedrock-converse-stream",
    provider="amazon-bedrock",
    base_url="https://bedrock-runtime.ap-northeast-1.amazonaws.com",
    reasoning=True,
    input=["text"],
    cost=ModelCost(input=1.25, output=10, cache_read=0.125, cache_write=0),
    context_window=400000,
    max_tokens=128000,
)


def redacted_reasoning_events() -> list[dict]:
    """Mirrors the ConverseStream frames GPT-5.6 emits: encrypted reasoning, then text."""
    return [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"redactedContent": REDACTED_BYTES}},
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "done"}}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]


class _NullStack:
    def add(self, *_args, **_kwargs) -> None:
        pass


@contextlib.contextmanager
def _stubbed_client(stream_events: list[dict] | None):
    class _Fake:
        def __init__(self, _config):
            self.middleware_stack = _NullStack()

        async def send(self, _command, *, cancel=None):
            if stream_events is None:
                raise RuntimeError("mock send")

            async def items():
                for event in stream_events:
                    yield event

            return SimpleNamespace(metadata=SimpleNamespace(http_status_code=200, request_id=None), stream=items())

    original = bedrock.BedrockRuntimeClient
    bedrock.BedrockRuntimeClient = _Fake
    try:
        yield
    finally:
        bedrock.BedrockRuntimeClient = original


async def run_stream(stream_events: list[dict]) -> AssistantMessage:
    context = Context(messages=[UserMessage(content="hello", timestamp=1)])
    with _stubbed_client(stream_events):
        return await stream_bedrock(GPT_MODEL, context, BedrockOptions(cache_retention="none")).result()


async def capture_payload(context: Context) -> dict:
    captured: list[dict] = []
    cancel = CancelToken()
    cancel.cancel()

    async def on_payload(payload, _model):
        captured.append(payload)
        return payload

    options = BedrockOptions(cache_retention="none", cancel=cancel, on_payload=on_payload)
    with _stubbed_client(None):
        stream = stream_bedrock(GPT_MODEL, context, options)
        async for event in stream:
            if event.type == "error":
                break
    assert captured, "Expected Bedrock payload to be captured before request abort"
    return captured[0]


def assistant(content: list) -> AssistantMessage:
    return AssistantMessage(
        content=content,
        api="bedrock-converse-stream",
        provider="amazon-bedrock",
        model=GPT_MODEL.id,
        usage=Usage(),
        stop_reason="stop",
        timestamp=1,
    )


def thinking_block(message: AssistantMessage) -> ThinkingContent:
    block = next((c for c in message.content if c.type == "thinking"), None)
    assert block is not None
    return block


@pytest.mark.tonio
async def test_does_not_fail_the_stream_when_reasoning_arrives_as_redacted_content():
    response = await run_stream(redacted_reasoning_events())

    assert response.stop_reason != "error", response.error_message
    # Reasoning precedes the answer, matching the order Bedrock streamed it.
    assert [c.type for c in response.content] == ["thinking", "text"]
    assert response.content[1] == TextContent(text="done")


@pytest.mark.tonio
async def test_preserves_the_encrypted_reasoning_payload_on_the_assistant_message():
    response = await run_stream(redacted_reasoning_events())

    # Same representation Anthropic redacted thinking already uses: the opaque
    # payload rides in `thinking_signature` with `redacted=True`.
    thinking = thinking_block(response)
    assert thinking.redacted is True
    assert thinking.thinking_signature == REDACTED_BASE64


@pytest.mark.tonio
async def test_encodes_the_payload_when_the_stream_never_sends_content_block_stop():
    response = await run_stream(
        [
            {"messageStart": {"role": "assistant"}},
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"reasoningContent": {"redactedContent": REDACTED_BYTES}},
                }
            },
            {"messageStop": {"stopReason": "end_turn"}},
        ]
    )

    assert thinking_block(response).thinking_signature == REDACTED_BASE64


@pytest.mark.tonio
async def test_joins_encrypted_reasoning_split_across_deltas():
    head, tail = REDACTED_BYTES[:7], REDACTED_BYTES[7:]
    response = await run_stream(
        [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"redactedContent": head}}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"redactedContent": tail}}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
    )

    thinking = thinking_block(response)
    assert thinking.thinking_signature == REDACTED_BASE64
    # The placeholder marks the block once, not once per delta.
    assert thinking.thinking == "[Reasoning redacted]"


@pytest.mark.tonio
async def test_replays_redacted_reasoning_as_reasoning_content_redacted_content():
    context = Context(
        messages=[
            UserMessage(content="hello", timestamp=1),
            assistant(
                [
                    ThinkingContent(thinking="", thinking_signature=REDACTED_BASE64, redacted=True),
                    TextContent(text="done"),
                ]
            ),
            UserMessage(content="continue", timestamp=1),
        ]
    )

    payload = await capture_payload(context)

    assistant_payload = next(m for m in payload["messages"] if m["role"] == "assistant")
    assert assistant_payload["content"] == [
        {"reasoningContent": {"redactedContent": REDACTED_BYTES}},
        {"text": "done"},
    ]


@pytest.mark.tonio
async def test_replays_redacted_reasoning_before_the_tool_use_block_it_belongs_to():
    # Bedrock rejects a tool continuation whose reasoning block is missing or
    # reordered, so the opaque payload must land ahead of the matching toolUse.
    context = Context(
        messages=[
            UserMessage(content="read the file", timestamp=1),
            assistant(
                [
                    ThinkingContent(thinking="", thinking_signature=REDACTED_BASE64, redacted=True),
                    ToolCall(id="tool-1", name="read", arguments={"path": "/tmp/a.txt"}),
                ]
            ),
            ToolResultMessage(
                tool_call_id="tool-1",
                tool_name="read",
                content=[TextContent(text="file body")],
                is_error=False,
                timestamp=1,
            ),
        ]
    )

    payload = await capture_payload(context)

    assistant_payload = next(m for m in payload["messages"] if m["role"] == "assistant")
    assert assistant_payload["content"] == [
        {"reasoningContent": {"redactedContent": REDACTED_BYTES}},
        {"toolUse": {"toolUseId": "tool-1", "name": "read", "input": {"path": "/tmp/a.txt"}}},
    ]
