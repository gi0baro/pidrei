"""Mirror of pi coding-agent src/client/index.ts.

pi's static `RemoteSession.open`/`RemoteSession.create` factories are exported
here as `open_remote_session`/`create_remote_session` (Python cannot overload
the instance methods of the same name).
"""

from .remote_session import (
    CreateRemoteSessionOptions,
    RemoteSession,
    RemoteSessionLifecycle,
    RemoteSessionOperation,
    RemoteSessionOptions,
    RemoteSessionState,
    create_remote_session,
    open_remote_session,
)
from .transcript import (
    TranscriptState,
    apply_transcript_progress,
    apply_transcript_snapshot,
    create_transcript_state,
    select_transcript,
)


__all__ = [
    "CreateRemoteSessionOptions",
    "RemoteSession",
    "RemoteSessionLifecycle",
    "RemoteSessionOperation",
    "RemoteSessionOptions",
    "RemoteSessionState",
    "TranscriptState",
    "apply_transcript_progress",
    "apply_transcript_snapshot",
    "create_remote_session",
    "create_transcript_state",
    "open_remote_session",
    "select_transcript",
]
