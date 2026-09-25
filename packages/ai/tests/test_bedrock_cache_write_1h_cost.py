"""Mirror of pi's bedrock-cache-write-1h-cost.test.ts.

pi replaces `@aws-sdk/client-bedrock-runtime` with `vi.mock`; here the stub
replaces `api/bedrock_runtime.BedrockRuntimeClient` by name, as in the other
bedrock mirrors.
"""

from types import SimpleNamespace

import pytest

from pidrei_ai.api import bedrock_converse_stream as bedrock
from pidrei_ai.api.bedrock_converse_stream import BedrockOptions, stream as stream_bedrock
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, UserMessage


MODEL = get_builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8")
CONTEXT = Context(messages=[UserMessage(content="hi", timestamp=1)])


class _FakeClient:
    def __init__(self, _config):
        self.middleware_stack = SimpleNamespace(add=lambda *args, **kwargs: None)

    async def send(self, _command, *, cancel=None):
        async def items():
            yield {"messageStart": {"role": "assistant"}}
            yield {
                "metadata": {
                    "usage": {
                        "inputTokens": 100,
                        "outputTokens": 5,
                        "totalTokens": 1_000_105,
                        "cacheWriteInputTokens": 1_000_000,
                        "cacheDetails": [
                            {"ttl": "1h", "inputTokens": 150_000},
                            {"ttl": "5m", "inputTokens": 600_000},
                            {"ttl": "1h", "inputTokens": 250_000},
                        ],
                    }
                }
            }
            yield {"messageStop": {"stopReason": "end_turn"}}

        return SimpleNamespace(metadata=SimpleNamespace(http_status_code=200, request_id=None), stream=items())


@pytest.mark.tonio
async def test_prices_the_1h_cache_details_at_2x_while_preserving_the_total_cache_write():
    # Regression test for https://github.com/earendil-works/pi/issues/9457
    original = bedrock.BedrockRuntimeClient
    bedrock.BedrockRuntimeClient = _FakeClient
    try:
        result = await stream_bedrock(MODEL, CONTEXT, BedrockOptions(cache_retention="none")).result()
    finally:
        bedrock.BedrockRuntimeClient = original

    assert result.usage.cache_write == 1_000_000
    assert result.usage.cache_write_1h == 400_000
    expected_cache_write_cost = (600_000 * MODEL.cost.cache_write + 400_000 * MODEL.cost.input * 2) / 1_000_000
    assert result.usage.cost.cache_write == pytest.approx(expected_cache_write_cost, abs=1e-10)
