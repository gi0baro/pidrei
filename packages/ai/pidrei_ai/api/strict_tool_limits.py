"""Mirror of pi packages/ai/src/api/strict-tool-limits.ts.

The set of sessions whose strict tools Claude rejected is process-wide and
shared by the Anthropic and Bedrock adapters, whose `stream()` runs on any
worker; it sits behind a thread lock that is never held across an await (pi
relies on its single thread). An insertion-ordered dict stands for pi's `Set`,
so the cap evicts the oldest session.
"""

import threading

from pidrei_ai.utils.session_resources import register_session_resource_cleanup


# Claude rejects (on the Anthropic API and on Bedrock) a request whose strict tools exceed its limits with one of these
# messages. Observed limits: 20 strict tools, and 16 optional or union-typed parameters across all strict tools.
_STRICT_TOOL_LIMIT_ERROR_MESSAGES = (
    "The compiled grammar is too large",
    "Too many strict tools (",
    "Schemas contains too many parameters with union types (",
)


def is_strict_tool_limit_error_message(message: str) -> bool:
    """True when a provider error message says the request's strict tools exceed the provider's limits."""
    return any(limit_message in message for limit_message in _STRICT_TOOL_LIMIT_ERROR_MESSAGES)


# Sessions whose strict tools Claude rejected for exceeding its limits. Their later requests send `strict: "prefer"`
# tools non-strict right away: the tool set rarely changes between turns, so trying strict again would only repeat
# the rejection, and switching back to strict later would rewrite the tool definitions and miss the prompt cache.
# Shared by the Anthropic and Bedrock providers, which run the same Claude limits.
_sessions_over_strict_tool_limits: dict[str, None] = {}
_sessions_over_strict_tool_limits_guard = threading.Lock()
MAX_SESSIONS_OVER_STRICT_TOOL_LIMITS = 10_000


def mark_strict_tool_limits_exceeded(session_id: str | None) -> None:
    """Records that a session's strict tools exceeded Claude's limits; does nothing without a session. Keeps at most
    `MAX_SESSIONS_OVER_STRICT_TOOL_LIMITS` sessions.

    Entries are normally freed by `cleanup_session_resources(session_id)` when a session ends. The cap bounds memory
    for library users that never call it; a dropped session just pays one more rejection.
    """
    if session_id is None:
        return
    with _sessions_over_strict_tool_limits_guard:
        # `setdefault` keeps a marked session where it is, as pi's `Set.add` does.
        _sessions_over_strict_tool_limits.setdefault(session_id, None)
        if len(_sessions_over_strict_tool_limits) > MAX_SESSIONS_OVER_STRICT_TOOL_LIMITS:
            del _sessions_over_strict_tool_limits[next(iter(_sessions_over_strict_tool_limits))]


def has_exceeded_strict_tool_limits(session_id: str | None) -> bool:
    """True when an earlier request of this session had its strict tools rejected for exceeding Claude's limits."""
    if session_id is None:
        return False
    with _sessions_over_strict_tool_limits_guard:
        return session_id in _sessions_over_strict_tool_limits


def _cleanup_sessions_over_strict_tool_limits(session_id: str | None) -> None:
    with _sessions_over_strict_tool_limits_guard:
        # pi's `if (sessionId)`: an empty id clears every session, as no id does.
        if session_id:
            _sessions_over_strict_tool_limits.pop(session_id, None)
        else:
            _sessions_over_strict_tool_limits.clear()


register_session_resource_cleanup(_cleanup_sessions_over_strict_tool_limits)


def reset_sessions_over_strict_tool_limits() -> bool:
    """Forget every session (tests); returns whether any was marked."""
    with _sessions_over_strict_tool_limits_guard:
        had_sessions = bool(_sessions_over_strict_tool_limits)
        _sessions_over_strict_tool_limits.clear()
    return had_sessions
