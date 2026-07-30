"""Mirror of pi's bedrock-credentials.test.ts.

pi replaces `@aws-sdk/client-bedrock-runtime` with `vi.mock` and asserts on the
config handed to the `BedrockRuntimeClient` constructor; here the stub replaces
`api/bedrock_runtime.BedrockRuntimeClient` by name, as in
test_bedrock_endpoint_resolution.py.
"""

import contextlib
import os

import pytest

from pidrei_ai.api import bedrock_converse_stream as bedrock
from pidrei_ai.api.bedrock_converse_stream import BedrockOptions, stream as stream_bedrock
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, UserMessage


CONTEXT = Context(messages=[UserMessage(content="hello", timestamp=1)])

constructor_calls: list[dict] = []


class _RecordingClient:
    def __init__(self, config):
        constructor_calls.append(dict(config))
        self.middleware_stack = _NullStack()

    async def send(self, _command, *, cancel=None):
        raise RuntimeError("mock send")


class _NullStack:
    def add(self, *_args, **_kwargs) -> None:
        pass


@contextlib.contextmanager
def _stubbed_client():
    original = bedrock.BedrockRuntimeClient
    bedrock.BedrockRuntimeClient = _RecordingClient
    try:
        yield
    finally:
        bedrock.BedrockRuntimeClient = original


_MANAGED_ENV = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION")


@pytest.fixture(autouse=True)
def _isolate(request):
    constructor_calls.clear()
    saved = {name: os.environ.get(name) for name in _MANAGED_ENV}
    for name in _MANAGED_ENV:
        os.environ.pop(name, None)

    def restore():
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    request.addfinalizer(restore)


async def capture_client_config(model, options: BedrockOptions | None = None) -> dict:
    constructor_calls.clear()
    opts = options or BedrockOptions()
    opts.cache_retention = "none"
    with _stubbed_client():
        await stream_bedrock(model, CONTEXT, opts).result()
    assert len(constructor_calls) == 1
    return constructor_calls[0]


@pytest.mark.tonio
async def test_prefers_explicit_and_scoped_profiles_over_ambient_aws_access_keys():
    os.environ["AWS_ACCESS_KEY_ID"] = "AKIAEXAMPLE"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "secretexample"
    model = get_builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8")

    config = await capture_client_config(model, BedrockOptions(profile="explicit-profile"))

    assert config["profile"] == "explicit-profile"
    assert config.get("credentials") is None

    config = await capture_client_config(model, BedrockOptions(env={"AWS_PROFILE": "scoped-profile"}))

    assert config["profile"] == "scoped-profile"
    assert config.get("credentials") is None


@pytest.mark.tonio
async def test_uses_ambient_aws_access_keys_when_no_profile_is_configured():
    os.environ["AWS_ACCESS_KEY_ID"] = "AKIAEXAMPLE"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "secretexample"
    model = get_builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8")

    config = await capture_client_config(model)

    assert config.get("profile") is None
    assert config["credentials"] == {
        "accessKeyId": "AKIAEXAMPLE",
        "secretAccessKey": "secretexample",
    }


@pytest.mark.tonio
async def test_uses_ambient_aws_access_keys_when_only_an_ambient_profile_is_set():
    os.environ["AWS_ACCESS_KEY_ID"] = "AKIAEXAMPLE"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "secretexample"
    os.environ["AWS_PROFILE"] = "ambient-profile"
    model = get_builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8")

    config = await capture_client_config(model)

    assert config["profile"] == "ambient-profile"
    assert config["credentials"] == {
        "accessKeyId": "AKIAEXAMPLE",
        "secretAccessKey": "secretexample",
    }
