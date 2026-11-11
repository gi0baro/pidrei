"""Shared fixtures for the ai package tests."""

import warnings

import pytest

from pidrei_ai.api import anthropic_messages, strict_tool_limits
from pidrei_ai.auth import anthropic_federation, google_adc
from pidrei_http import http


_HTTP_SEAM_FUNCTIONS = {"client_for": http.client_for, "shared_client": http.shared_client}


@pytest.fixture(autouse=True)
def _http_seam_guard():
    """Fail-loud restore of the HTTP seam's client lookups.

    Adapter tests replace `http.client_for` (or `shared_client`) on the module
    to hand the transport a fake client; one left replaced would serve that
    fake to every later request in the shared runtime. The warning names the
    polluting test; the restore keeps the fake from spreading.
    """
    yield
    for name, original in _HTTP_SEAM_FUNCTIONS.items():
        if getattr(http, name) is not original:
            setattr(http, name, original)
            warnings.warn(f"test left pidrei_http.http.{name} replaced; restored", stacklevel=1)


@pytest.fixture(autouse=True)
def _anthropic_federation_cache_guard():
    """Fail-loud reset of the process-wide Anthropic federation token cache.

    Building an Anthropic transport with federation config publishes a cache
    holding the exchanged access token; one left behind would serve that token
    to every later federated request in the shared runtime. The warning names
    the polluting test; the reset keeps the token from leaking further.
    """
    yield
    if anthropic_federation.reset_federation_token_cache():
        warnings.warn(
            "test left the Anthropic federation token cache set (reset_federation_token_cache was not called); reset",
            stacklevel=1,
        )


@pytest.fixture(autouse=True)
def _anthropic_thread_state_guard():
    """Fail-loud reset of the Anthropic adapter's process-wide message thread state.

    A request with a session id on a threads endpoint records where that
    session's next request can continue; an entry left behind turns a later
    test's request with the same session id into a `continue` without `system`
    and `tools`. The warning names the polluting test (it did not call
    `cleanup_session_resources`); the reset keeps the entry from leaking further.
    """
    yield
    if anthropic_messages.reset_thread_state():
        warnings.warn(
            "test left Anthropic message thread state set (cleanup_session_resources was not called); reset",
            stacklevel=1,
        )


@pytest.fixture(autouse=True)
def _strict_tool_limit_sessions_guard():
    """Fail-loud reset of the process-wide set of sessions over Claude's strict
    tool limits (shared by the Anthropic and Bedrock adapters). A session left
    marked sends a later test's `strict: "prefer"` tools non-strict from the
    first request. The warning names the polluting test (it did not call
    `cleanup_session_resources`); the reset keeps the mark from leaking further.
    """
    yield
    if strict_tool_limits.reset_sessions_over_strict_tool_limits():
        warnings.warn(
            "test left sessions marked over the strict tool limits (cleanup_session_resources was not called); reset",
            stacklevel=1,
        )


@pytest.fixture(autouse=True)
def _google_adc_token_cache_guard():
    """Fail-loud reset of the process-wide Google ADC token cache: a token left
    cached would be served to every later Vertex request for that credential."""
    yield
    if google_adc.reset_google_adc_token_cache():
        warnings.warn(
            "test left Google ADC access tokens cached (reset_google_adc_token_cache was not called); reset",
            stacklevel=1,
        )
