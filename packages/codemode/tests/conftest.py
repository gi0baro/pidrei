"""Shared fixtures for the codemode package tests."""

import pytest

from pidrei_codemode import CodemodePool


@pytest.fixture(autouse=True)
def _tonio_runtime(tonio_runtime):
    """Every test pulls in the runtime: `clock.monotonic` reads its clock, and
    async fixtures then run for plain sync tests too."""


@pytest.fixture
async def pool():
    """A pool per test, always closed: Monty pools are only ever created
    through this fixture, so none outlives its test."""
    pool = await CodemodePool()
    try:
        yield pool
    finally:
        await pool.close()
