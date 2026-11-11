"""Mirror of pi packages/ai/src/session-resources.ts.

Registry of per-session resource cleanups. Registrants: the Codex WebSocket
adapter (its cached sockets), the Anthropic adapter (message thread state) and
`api/strict_tool_limits.py` (sessions over Claude's strict tool limits).
AgentSession.dispose() calls cleanup_session_resources(session_id); no id
clears every session.
"""

import threading
from collections.abc import Callable


SessionResourceCleanup = Callable[[str | None], None]

_cleanups: list[SessionResourceCleanup] = []
_guard = threading.Lock()


def register_session_resource_cleanup(cleanup: SessionResourceCleanup) -> Callable[[], None]:
    with _guard:
        if cleanup not in _cleanups:
            _cleanups.append(cleanup)

    def unregister() -> None:
        with _guard:
            if cleanup in _cleanups:
                _cleanups.remove(cleanup)

    return unregister


def cleanup_session_resources(session_id: str | None = None) -> None:
    with _guard:
        cleanups = list(_cleanups)
    errors: list[Exception] = []
    for cleanup in cleanups:
        try:
            cleanup(session_id)
        except Exception as error:
            errors.append(error)
    if errors:
        raise ExceptionGroup("Failed to cleanup session resources", errors)
