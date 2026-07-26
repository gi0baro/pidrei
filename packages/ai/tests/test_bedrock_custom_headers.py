"""Mirror of pi's bedrock-custom-headers.test.ts.

The middleware is captured from the stubbed client's `middleware_stack` and then
driven by hand against a fake request, exactly as pi drives the registered
handler with a `fakeArgs` object.
"""

import contextlib

import pytest

from pidrei_ai.api import bedrock_converse_stream as bedrock
from pidrei_ai.api.bedrock_converse_stream import (
    BedrockOptions,
    stream as stream_bedrock,
    stream_simple as stream_simple_bedrock,
)
from pidrei_ai.api.bedrock_runtime import HttpRequest, MiddlewareArgs, MiddlewareRegistration
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, SimpleStreamOptions, UserMessage


CONTEXT = Context(messages=[UserMessage(content="hello", timestamp=1)])
MIDDLEWARE_NAME = "pidrei-ai-custom-headers"

middleware_registrations: list[MiddlewareRegistration] = []


class _RecordingStack:
    def add(self, handler, *, step=None, name=None, priority=None) -> None:
        middleware_registrations.append(MiddlewareRegistration(handler, step=step, name=name, priority=priority))


class _RecordingClient:
    def __init__(self, _config):
        self.middleware_stack = _RecordingStack()

    async def send(self, _command, *, cancel=None):
        raise RuntimeError("mock send")


@contextlib.contextmanager
def _stubbed_client():
    original = bedrock.BedrockRuntimeClient
    bedrock.BedrockRuntimeClient = _RecordingClient
    try:
        yield
    finally:
        bedrock.BedrockRuntimeClient = original


@pytest.fixture(autouse=True)
def _reset():
    middleware_registrations.clear()


def model_fixture():
    return get_builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8")


async def drive_bedrock(options: BedrockOptions) -> None:
    """Drive a stream to completion so the middleware, registered before `send`,
    is captured even though the stubbed `send` raises."""
    with _stubbed_client():
        await stream_bedrock(model_fixture(), CONTEXT, options).result()


def find_custom_headers_registrations() -> list[MiddlewareRegistration]:
    return [r for r in middleware_registrations if r.name == MIDDLEWARE_NAME]


def fake_args(headers: dict[str, str]) -> MiddlewareArgs:
    return MiddlewareArgs(request=HttpRequest(method="POST", url="https://example.invalid/", headers=headers, body=b""))


async def _run_middleware(registration: MiddlewareRegistration, args: MiddlewareArgs) -> list[MiddlewareArgs]:
    seen: list[MiddlewareArgs] = []

    async def next_handler(inner_args):
        seen.append(inner_args)
        return inner_args

    await registration.handler(next_handler)(args)
    return seen


@pytest.mark.tonio
async def test_vc1_registers_a_build_step_middleware_that_injects_the_caller_header():
    await drive_bedrock(BedrockOptions(cache_retention="none", headers={"x-caller": "value"}))

    registrations = find_custom_headers_registrations()
    assert len(registrations) == 1
    registration = registrations[0]
    assert registration.step == "build"
    assert registration.priority == "low"

    args = fake_args({})
    seen = await _run_middleware(registration, args)

    assert args.request.headers["x-caller"] == "value"
    assert len(seen) == 1


@pytest.mark.tonio
async def test_vc2_skips_reserved_headers_case_insensitively_while_applying_allowed_ones():
    await drive_bedrock(
        BedrockOptions(
            cache_retention="none",
            headers={
                "authorization": "evil",
                "x-amz-date": "evil",
                "x-allowed": "ok",
                "Authorization": "evil2",
                "X-Amz-Date": "evil2",
                "HOST": "evil3",
            },
        )
    )

    registrations = find_custom_headers_registrations()
    assert registrations

    args = fake_args({"authorization": "real-auth", "x-amz-date": "real-date", "host": "real-host"})
    seen = await _run_middleware(registrations[0], args)

    headers = args.request.headers
    assert headers["authorization"] == "real-auth"
    assert headers["x-amz-date"] == "real-date"
    assert headers["host"] == "real-host"
    assert headers["x-allowed"] == "ok"
    # Mixed-case reserved keys must be skipped too: a case-sensitive guard would
    # add them back as distinct capitalised keys.
    assert "Authorization" not in headers
    assert "X-Amz-Date" not in headers
    assert "HOST" not in headers
    assert sorted(headers) == sorted(["authorization", "host", "x-allowed", "x-amz-date"])
    assert len(seen) == 1


@pytest.mark.tonio
async def test_vc3_registers_no_middleware_when_headers_is_none():
    await drive_bedrock(BedrockOptions(cache_retention="none"))

    assert find_custom_headers_registrations() == []


@pytest.mark.tonio
async def test_vc3_registers_no_middleware_when_headers_is_empty():
    await drive_bedrock(BedrockOptions(cache_retention="none", headers={}))

    assert find_custom_headers_registrations() == []


@pytest.mark.tonio
async def test_vc3_structural_guard_passes_through_unchanged_when_the_request_has_no_headers():
    await drive_bedrock(BedrockOptions(cache_retention="none", headers={"x-caller": "value"}))

    registrations = find_custom_headers_registrations()
    assert registrations

    class _HeaderlessArgs:
        request = None

    args = _HeaderlessArgs()
    seen = await _run_middleware(registrations[0], args)

    assert seen == [args]


@pytest.mark.tonio
async def test_vc4_stream_simple_forwards_headers_end_to_end():
    with _stubbed_client():
        await stream_simple_bedrock(
            model_fixture(), CONTEXT, SimpleStreamOptions(headers={"x-caller": "value"})
        ).result()

    registrations = find_custom_headers_registrations()
    assert len(registrations) == 1

    args = fake_args({})
    await _run_middleware(registrations[0], args)
    assert args.request.headers["x-caller"] == "value"
