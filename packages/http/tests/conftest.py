"""Shared fixtures for the http package tests. The clock and timer seams are
guarded by the root conftest."""

import warnings

import pytest

from pidrei_http import http


_SEAM_FUNCTIONS = {"client_for": http.client_for, "shared_client": http.shared_client}


@pytest.fixture(autouse=True)
def _tonio_runtime(tonio_runtime):
    """Every test pulls in the runtime, so async fixtures run for plain sync
    tests too."""


@pytest.fixture(autouse=True)
def _http_seam_guard():
    """Fail-loud restore of the HTTP seam's client lookups.

    Tests replace `http.client_for` (or `shared_client`) on the module to hand
    adapters a fake client; one left replaced would serve that fake to every
    later request in the shared runtime. The warning names the polluting test;
    the restore keeps the fake from spreading.
    """
    yield
    for name, original in _SEAM_FUNCTIONS.items():
        if getattr(http, name) is not original:
            setattr(http, name, original)
            warnings.warn(f"test left pidrei_http.http.{name} replaced; restored", stacklevel=1)
