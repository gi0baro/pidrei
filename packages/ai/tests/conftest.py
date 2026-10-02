"""Shared fixtures for the ai package tests."""

import warnings

import pytest

from pidrei_ai.auth import anthropic_federation, google_adc


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
