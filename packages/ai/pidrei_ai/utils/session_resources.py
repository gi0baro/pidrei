"""Mirror of pi packages/ai/src/session-resources.ts.

Registry of per-session resource cleanups (pi's only registrant is the Codex
WebSocket adapter — Phase 5 here); AgentSession.dispose() calls
cleanup_session_resources unconditionally, so the seam ships now.
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
