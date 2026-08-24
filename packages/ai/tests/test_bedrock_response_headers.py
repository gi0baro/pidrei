"""Mirror of pi's bedrock-response-headers.test.ts.

pi starts a local HTTP server so the SDK's Smithy stack sees real response
headers; pidrei's Bedrock client goes through the `utils/http` seam, so the
transport is stubbed there instead (the pattern `test_bedrock_runtime.py`
uses). The point is the same: the raw response headers reach `on_response`,
not just the two `metadata` fields Bedrock models.
"""

import pytest

from pidrei_ai.api.bedrock_converse_stream import BedrockOptions, stream as stream_bedrock
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, ProviderResponse, UserMessage
from pidrei_ai.utils import http


MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

CONTEXT = Context(messages=[UserMessage(content="hello", timestamp=1)])


class _EmptyEventStream:
    """A 200 with headers and no frames: the callback must fire before consumption."""

    status_code = 200

    def __init__(self, headers: dict[str, str]):
        self.headers = headers

    def iter_bytes(self):
        async def gen():
            return
            yield b""

        return gen()

    async def close(self) -> None:
        pass


@pytest.fixture
def stub_transport(request):
    original = http.client_for
    request.addfinalizer(lambda: setattr(http, "client_for", original))

    def install(headers: dict[str, str]) -> None:
        async def post(_url, *, content=None, headers=None, timeout=None):
            return _EmptyEventStream(response_headers)

        response_headers = headers
        http.client_for = lambda *_args, **_kwargs: type("C", (), {"post": staticmethod(post)})()

    return install


@pytest.mark.tonio
async def test_forwards_raw_response_headers_to_on_response(stub_transport):
    stub_transport(
        {
            "content-type": "application/vnd.amazon.eventstream",
            "x-bifrost-provider": "bedrock",
            "x-bifrost-resolved-model": MODEL_ID,
            "x-amzn-requestid": "req-123",
        }
    )
    model = get_builtin_model("amazon-bedrock", MODEL_ID)
    responses: list[ProviderResponse] = []

    async def on_response(response, _model):
        responses.append(response)

    result = await stream_bedrock(
        model,
        CONTEXT,
        BedrockOptions(env={"AWS_BEDROCK_SKIP_AUTH": "1"}, on_response=on_response),
    ).result()

    # The stubbed transport intentionally returns an empty event stream; this
    # assertion documents that the header callback still fires before consumption.
    assert result.stop_reason == "error"
    assert len(responses) == 1
    assert responses[0].status == 200
    assert responses[0].headers["x-amzn-requestid"] == "req-123"
    assert responses[0].headers["x-bifrost-provider"] == "bedrock"
    assert responses[0].headers["x-bifrost-resolved-model"] == MODEL_ID
