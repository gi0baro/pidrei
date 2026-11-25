"""PiDrei's replacement for pi's bedrock-proxy-credentials.test.ts.

pi's file runs a local CONNECT proxy and a temporary SSO profile and checks that
the AWS SDK's default credential chain sends its SSO request through
`HTTPS_PROXY` (#12353). Here credentials come from botocore, which reads a proxy
in the process environment itself, so that file would test botocore. What
PiDrei owns is handing a provider-scoped proxy (`options.env`) to the Bedrock
request and to botocore's credential session; these cases check that, with
`http.client_for` and botocore's `Session` replaced by recorders.
"""

import botocore.session
import pytest
from botocore.credentials import Credentials

from pidrei_ai.api.bedrock_converse_stream import BedrockOptions, stream as stream_bedrock
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, UserMessage
from pidrei_http import http


CONTEXT = Context(messages=[UserMessage(content="hello", timestamp=1)])
PROXY = "http://proxy.test:8080"

_CLEARED_ENV = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_BEDROCK_SKIP_AUTH",
    "AWS_PROFILE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "http_proxy",
    "HTTP_PROXY",
    "https_proxy",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
    "no_proxy",
    "NO_PROXY",
)


class _StopSend(Exception):
    pass


class _RecordingSession:
    def __init__(self, profile=None) -> None:
        self.profile = profile
        self.default_client_config = None

    def set_default_client_config(self, config) -> None:
        self.default_client_config = config

    def get_credentials(self):
        return Credentials("AKIDEXAMPLE", "secretexample")


class _Recorder:
    def __init__(self) -> None:
        self.sessions: list[_RecordingSession] = []
        self.client_for_calls: list[tuple[str, object]] = []


@pytest.fixture
def recorder(monkeypatch) -> _Recorder:
    for name in _CLEARED_ENV:
        monkeypatch.delenv(name, raising=False)
    recorded = _Recorder()

    class _Session(_RecordingSession):
        def __init__(self, profile=None):
            super().__init__(profile)
            recorded.sessions.append(self)

    async def fake_post(url, *, content, headers, timeout):
        raise _StopSend("mock send")

    def fake_client_for(url, env=None):
        recorded.client_for_calls.append((url, env))
        return type("C", (), {"post": staticmethod(fake_post)})()

    monkeypatch.setattr(botocore.session, "Session", _Session)
    monkeypatch.setattr(http, "client_for", fake_client_for)
    return recorded


async def _run(env: dict[str, str]) -> None:
    model = get_builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8")
    message = await stream_bedrock(model, CONTEXT, BedrockOptions(env=env, cache_retention="none")).result()
    assert message.stop_reason == "error"
    assert message.error_message == "mock send"


@pytest.mark.tonio
async def test_a_scoped_proxy_reaches_the_bedrock_request_and_the_credential_session(recorder):
    env = {"HTTPS_PROXY": PROXY}

    await _run(env)

    assert len(recorder.client_for_calls) == 1
    url, client_env = recorder.client_for_calls[0]
    assert url.startswith("https://bedrock-runtime.")
    assert client_env == env
    assert len(recorder.sessions) == 1
    assert recorder.sessions[0].default_client_config.proxies == {"http": f"{PROXY}/", "https": f"{PROXY}/"}


@pytest.mark.tonio
async def test_a_proxy_from_the_process_environment_is_left_to_botocore(recorder, monkeypatch):
    # botocore reads it per host (NO_PROXY included), so the session gets no proxy of its own.
    monkeypatch.setenv("HTTPS_PROXY", PROXY)
    env = {"AWS_REGION": "us-east-1"}

    await _run(env)

    assert [client_env for _url, client_env in recorder.client_for_calls] == [env]
    assert len(recorder.sessions) == 1
    assert recorder.sessions[0].default_client_config is None
