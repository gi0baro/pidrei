"""Mirror of pi's bedrock-stream-state.test.ts (#9265: Bedrock kept `index`,
`partialJson` and the encrypted reasoning bytes on the content blocks while
streaming, and re-parsed the accumulated tool-call arguments on every delta).

pi replaces `@aws-sdk/client-bedrock-runtime` with `vi.mock`; here the stub
replaces `api/bedrock_runtime.BedrockRuntimeClient` by name, as in the other
Bedrock mirrors. pi's check that no stream state reaches a partial block is not
mirrored: blocks are slotted dataclasses and the adapter keeps that state in
its own maps, so the check could never fail. "allocates linearly" measures V8
heap allocation; here it measures the stream's wall time, best of two runs,
for arguments ten times longer than the other's.
"""

import contextlib
import json
import math
import time
from types import SimpleNamespace

import pytest

from pidrei_ai.api import bedrock_converse_stream as bedrock
from pidrei_ai.api.bedrock_converse_stream import BedrockOptions, stream as stream_bedrock
from pidrei_ai.types import Context, Model, ModelCost, ThinkingContent, ToolCall, UserMessage


MODEL = Model(
    id="global.anthropic.claude-test",
    name="Claude Test",
    api="bedrock-converse-stream",
    provider="amazon-bedrock",
    base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
    reasoning=False,
    input=["text"],
    cost=ModelCost(),
    context_window=200_000,
    max_tokens=64_000,
)

CONTEXT = Context(messages=[UserMessage(content="write", timestamp=0)])

CONTENT = "".join(f'const line{line} = "value {line}"; // \\ done\n' for line in range(20_000))


def write_call(size: int) -> tuple[dict, str]:
    args = {"path": "/src/file.ts", "content": CONTENT[:size]}
    return args, json.dumps(args)


def events(text: str) -> list[dict]:
    """Encrypted reasoning, then a `write` tool call whose arguments arrive in 32-character deltas."""
    return [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"redactedContent": b"\x01\x02"}},
            }
        },
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"redactedContent": b"\x03"}}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"contentBlockStart": {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "tool_1", "name": "write"}}}},
        *(
            {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": text[index : index + 32]}}}}
            for index in range(0, len(text), 32)
        ),
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]


class _NullStack:
    def add(self, *_args, **_kwargs) -> None:
        pass


@contextlib.contextmanager
def _stubbed_client(stream_events: list[dict]):
    class _Fake:
        def __init__(self, _config):
            self.middleware_stack = _NullStack()

        async def send(self, _command, *, cancel=None):
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


@pytest.mark.tonio
async def test_finishes_the_tool_call_with_its_exact_arguments_and_shows_them_growing():
    args, text = write_call(5_000)
    shown = -1
    with _stubbed_client(events(text)):
        stream = stream_bedrock(MODEL, CONTEXT, BedrockOptions(cache_retention="none"))
        async for event in stream:
            if event.type != "toolcall_delta":
                continue
            call = event.partial.content[event.content_index]
            partial = call.arguments.get("content") if call.type == "toolCall" else None
            if not isinstance(partial, str):
                continue
            assert args["content"].startswith(partial)
            assert len(partial) >= shown
            shown = len(partial)
        message = await stream.result()
    assert message.stop_reason == "toolUse", message.error_message
    assert message.content == [
        ThinkingContent(thinking="[Reasoning redacted]", thinking_signature="AQID", redacted=True),
        ToolCall(id="tool_1", name="write", arguments=args),
    ]


async def _wall_seconds(size: int) -> float:
    best = math.inf
    for _ in range(2):
        args, text = write_call(size)
        with _stubbed_client(events(text)):
            started = time.perf_counter()
            message = await stream_bedrock(MODEL, CONTEXT, BedrockOptions(cache_retention="none")).result()
            best = min(best, time.perf_counter() - started)
        assert message.content[1].arguments == args
    return best


@pytest.mark.tonio
async def test_takes_time_linear_in_the_argument_text():
    small = await _wall_seconds(25_000)
    large = await _wall_seconds(250_000)
    assert large / small < 25
