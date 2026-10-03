"""Shared fixtures for the ai package tests."""

import warnings

import pytest

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
def _google_adc_token_cache_guard():
    """Fail-loud reset of the process-wide Google ADC token cache: a token left
    cached would be served to every later Vertex request for that credential."""
    yield
    if google_adc.reset_google_adc_token_cache():
        warnings.warn(
            "test left Google ADC access tokens cached (reset_google_adc_token_cache was not called); reset",
            stacklevel=1,
        )
