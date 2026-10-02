"""Shared fixtures for the agent package tests. The clock and timer seams are
guarded by the root conftest."""

import pytest


@pytest.fixture(autouse=True)
def _tonio_runtime(tonio_runtime):
    """Every test pulls in the runtime: `clock.monotonic` reads its clock, and
    async fixtures then run for plain sync tests too."""
