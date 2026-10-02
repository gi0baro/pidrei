"""pidrei-only: `client_for` pools punkreq clients by the proxy a request's
provider-scoped env resolves to (pi configures a Node agent per request)."""

import contextlib
import os

from pidrei_http import http


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
