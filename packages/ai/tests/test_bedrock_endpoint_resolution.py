"""Mirror of pi's bedrock-endpoint-resolution.test.ts.

pi replaces `@aws-sdk/client-bedrock-runtime` with `vi.mock` and asserts on the
config handed to the `BedrockRuntimeClient` constructor. Here that constructor is
`api/bedrock_runtime.BedrockRuntimeClient`, imported by name into the adapter, so
the stub replaces the same thing — and since that module keeps the SDK's own
config keys, the assertions are pi's verbatim.
"""

import contextlib
import os
from dataclasses import replace

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


_MANAGED_ENV = ("AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE")


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


def test_assigns_eu_central_1_runtime_urls_to_built_in_eu_inference_profiles():
    model = get_builtin_model("amazon-bedrock", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")

    assert model.base_url == "https://bedrock-runtime.eu-central-1.amazonaws.com"


@pytest.mark.tonio
async def test_does_not_pin_standard_aws_endpoints_when_aws_region_is_configured():
    os.environ["AWS_REGION"] = "us-east-2"
    model = get_builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8")

    config = await capture_client_config(model)

    assert config["region"] == "us-east-2"
    assert config.get("endpoint") is None


@pytest.mark.tonio
async def test_derives_region_from_a_built_in_eu_endpoint_when_no_region_or_profile_is_configured():
    model = get_builtin_model("amazon-bedrock", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")

    config = await capture_client_config(model)

    assert config["endpoint"] == "https://bedrock-runtime.eu-central-1.amazonaws.com"
    assert config["region"] == "eu-central-1"


@pytest.mark.tonio
async def test_handles_missing_regions_for_explicit_scoped_and_ambient_profiles():
    model = get_builtin_model("amazon-bedrock", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")

    config = await capture_client_config(model, BedrockOptions(profile="bedrock-profile"))

    assert config["profile"] == "bedrock-profile"
    assert config["endpoint"] == "https://bedrock-runtime.eu-central-1.amazonaws.com"
    assert config["region"] == "eu-central-1"

    config = await capture_client_config(model, BedrockOptions(env={"AWS_PROFILE": "scoped-bedrock-profile"}))

    assert config["profile"] == "scoped-bedrock-profile"
    assert config["endpoint"] == "https://bedrock-runtime.eu-central-1.amazonaws.com"
    assert config["region"] == "eu-central-1"

    os.environ["AWS_PROFILE"] = "ambient-bedrock-profile"
    config = await capture_client_config(model)

    assert config["profile"] == "ambient-bedrock-profile"
    assert config.get("endpoint") is None
    assert config.get("region") is None


@pytest.mark.tonio
async def test_still_passes_custom_bedrock_endpoints_through_to_the_sdk_client():
    os.environ["AWS_REGION"] = "us-west-2"
    base_model = get_builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8")
    model = replace(base_model, base_url="https://bedrock-vpc.example.com")

    config = await capture_client_config(model)

    assert config["endpoint"] == "https://bedrock-vpc.example.com"
    assert config["region"] == "us-west-2"


@pytest.mark.tonio
async def test_extracts_region_from_inference_profile_arn_regardless_of_aws_region():
    os.environ["AWS_REGION"] = "us-east-1"
    base_model = get_builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8")
    model = replace(base_model, id="arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/abc123")

    config = await capture_client_config(model)

    assert config["region"] == "us-west-2"


@pytest.mark.tonio
async def test_extracts_region_from_govcloud_inference_profile_arn():
    os.environ["AWS_REGION"] = "us-east-1"
    base_model = get_builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8")
    model = replace(
        base_model,
        id="arn:aws-us-gov:bedrock:us-gov-west-1:123456789012:application-inference-profile/abc123",
    )

    config = await capture_client_config(model)

    assert config["region"] == "us-gov-west-1"


@pytest.mark.tonio
async def test_preserves_ambient_aws_auth_for_custom_model_ids_through_compat_dispatch():
    os.environ["AWS_PROFILE"] = "bedrock-profile"
    base_model = get_builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8")
    model = replace(base_model, id="arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/example")

    config = await capture_client_config(model)

    assert config["profile"] == "bedrock-profile"
    assert config.get("token") is None
    assert config.get("authSchemePreference") is None


@pytest.mark.tonio
async def test_uses_the_generic_api_key_option_as_a_bedrock_bearer_token():
    model = get_builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8")

    config = await capture_client_config(model, BedrockOptions(api_key="bedrock-api-key"))

    assert config["token"] == {"token": "bedrock-api-key"}
    assert config["authSchemePreference"] == ["httpBearerAuth"]
