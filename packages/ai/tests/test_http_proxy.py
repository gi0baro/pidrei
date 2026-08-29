"""Mirror of pi's node-http-proxy.test.ts, plus the punkreq plumbing.

pi mutates `process.env` and restores it in `afterEach`; these tests pass the
process-env half through a context manager instead, so nothing leaks between
tests (predates tonio 0.9.14; `monkeypatch` works in tonio tests now).
"""

import contextlib
import os
import time

import pytest

from pidrei_ai.api import anthropic_messages, openai_completions, openai_responses
from pidrei_ai.types import Context, Model, ModelCost, SimpleStreamOptions, UserMessage
from pidrei_ai.utils import http
from pidrei_ai.utils.http_proxy import UNSUPPORTED_PROXY_PROTOCOL_MESSAGE, resolve_http_proxy_url_for_target


PROXY_ENV_KEYS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
]

TARGET = "https://bedrock-runtime.us-east-1.amazonaws.com"


@contextlib.contextmanager
def process_proxy_env(**values: str):
    """Replace every proxy var in os.environ with `values` for the duration."""
    saved = {key: os.environ.pop(key, None) for key in PROXY_ENV_KEYS}
    os.environ.update(values)
    try:
        yield
    finally:
        for key in values:
            os.environ.pop(key, None)
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def test_respects_no_proxy_exclusions():
    with process_proxy_env(
        HTTPS_PROXY="http://proxy.example:8080",
        NO_PROXY="bedrock-runtime.us-east-1.amazonaws.com",
    ):
        assert resolve_http_proxy_url_for_target(TARGET) is None


def test_resolves_http_and_https_proxy_urls():
    with process_proxy_env(HTTPS_PROXY="http://proxy.example:8080"):
        assert resolve_http_proxy_url_for_target(TARGET) == "http://proxy.example:8080/"


def test_prefers_scoped_proxy_env_aliases_before_process_env_aliases():
    with process_proxy_env(https_proxy="http://process-proxy.example:8080"):
        resolved = resolve_http_proxy_url_for_target(TARGET, {"HTTPS_PROXY": "http://scoped-proxy.example:8080"})

    assert resolved == "http://scoped-proxy.example:8080/"


def test_rejects_socks_and_pac_proxy_urls_explicitly():
    with (
        process_proxy_env(HTTPS_PROXY="socks5://proxy.example:1080"),
        pytest.raises(ValueError, match="SOCKS and PAC proxy URLs are not supported"),
    ):
        resolve_http_proxy_url_for_target(TARGET)


def test_unsupported_protocol_message_is_pis_wording():
    assert UNSUPPORTED_PROXY_PROTOCOL_MESSAGE == (
        "Unsupported proxy protocol. SOCKS and PAC proxy URLs are not supported; use an HTTP or HTTPS proxy URL."
    )


# --- punkreq plumbing ---------------------------------------------------------


def test_client_for_reuses_the_shared_client_without_scoped_env():
    with process_proxy_env():
        assert http.client_for(TARGET) is http.shared_client()
        assert http.client_for(TARGET, {}) is http.shared_client()


def test_client_for_pools_one_client_per_scoped_proxy():
    with process_proxy_env():
        first = http.client_for(TARGET, {"HTTPS_PROXY": "http://scoped.example:8080"})
        again = http.client_for(TARGET, {"HTTPS_PROXY": "http://scoped.example:8080"})
        other = http.client_for(TARGET, {"HTTPS_PROXY": "http://elsewhere.example:8080"})

    assert first is again
    assert first is not other
    assert first is not http.shared_client()


def test_client_for_honours_a_scoped_no_proxy_over_an_ambient_proxy():
    with process_proxy_env(HTTPS_PROXY="http://ambient.example:8080"):
        # punkreq's trust_env would proxy through the ambient value; a scoped
        # NO_PROXY must win, so this cannot be the shared client.
        scoped = http.client_for(TARGET, {"NO_PROXY": "bedrock-runtime.us-east-1.amazonaws.com"})

        assert scoped is not http.shared_client()
        assert scoped is http.client_for(TARGET, {"NO_PROXY": "*"})


# --- the adapters reach client_for with their scoped env ----------------------


class _ClientResolved(Exception):
    """Aborts the request once the transport has resolved its client."""


@contextlib.contextmanager
def recording_client_for():
    calls: list[tuple[str, object]] = []
    original = http.client_for

    def fake_client_for(target_url, env=None):
        calls.append((target_url, env))
        raise _ClientResolved()

    http.client_for = fake_client_for
    try:
        yield calls
    finally:
        http.client_for = original


ADAPTERS = [
    ("anthropic-messages", "https://api.anthropic.com", "/v1/messages"),
    ("openai-completions", "https://api.example.test/v1", "/chat/completions"),
    ("openai-responses", "https://api.example.test/v1", "/responses"),
]


def _make_model(api: str, base_url: str) -> Model:
    return Model(
        id="proxy-probe",
        name="Proxy Probe",
        api=api,
        provider="proxy-probe-provider",
        base_url=base_url,
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=10000,
        max_tokens=1000,
    )


@pytest.mark.tonio
@pytest.mark.parametrize(("api", "base_url", "path"), ADAPTERS)
async def test_adapters_resolve_their_client_from_the_scoped_env(api, base_url, path):
    adapter = {
        "anthropic-messages": anthropic_messages,
        "openai-completions": openai_completions,
        "openai-responses": openai_responses,
    }[api]
    context = Context(messages=[UserMessage(content="hi", timestamp=int(time.time() * 1000))])
    scoped_env = {"HTTPS_PROXY": "http://scoped.example:8080"}

    with recording_client_for() as calls, process_proxy_env():
        await adapter.stream_simple(
            _make_model(api, base_url),
            context,
            SimpleStreamOptions(api_key="fake-key", env=scoped_env),
        ).result()

    assert calls == [(f"{base_url}{path}", scoped_env)]
