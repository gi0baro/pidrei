"""Mirror of pi ai test/strict-tool-limits.test.ts.

pi's `afterEach` `cleanupSessionResources()` is the module's autouse fixture.
"""

import pytest

from pidrei_ai.api.strict_tool_limits import has_exceeded_strict_tool_limits, mark_strict_tool_limits_exceeded
from pidrei_ai.utils.session_resources import cleanup_session_resources


@pytest.fixture(autouse=True)
def _cleanup_session_resources():
    yield
    cleanup_session_resources()


def test_remembers_marked_sessions_until_their_cleanup():
    mark_strict_tool_limits_exceeded("a")
    mark_strict_tool_limits_exceeded("b")
    mark_strict_tool_limits_exceeded(None)
    assert has_exceeded_strict_tool_limits("a") is True
    assert has_exceeded_strict_tool_limits("c") is False
    assert has_exceeded_strict_tool_limits(None) is False

    cleanup_session_resources("a")
    assert has_exceeded_strict_tool_limits("a") is False
    assert has_exceeded_strict_tool_limits("b") is True

    cleanup_session_resources()
    assert has_exceeded_strict_tool_limits("b") is False


def test_drops_the_oldest_session_beyond_10_000():
    for index in range(10_001):
        mark_strict_tool_limits_exceeded(f"session-{index}")
    assert has_exceeded_strict_tool_limits("session-0") is False
    assert has_exceeded_strict_tool_limits("session-1") is True
    assert has_exceeded_strict_tool_limits("session-10000") is True
