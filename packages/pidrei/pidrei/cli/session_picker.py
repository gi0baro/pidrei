"""Basic session picker for --resume (Phase 3).

pi's session picker (src/cli/session-picker.ts) is a TUI selector with
search, rename, and delete; that lands with the Phase 4 TUI slice. This is
the headless stand-in the PLAN calls "session picker (basic)": a numbered
list on stderr and a line-based selection prompt on stdin. It keeps pi's
selectSession() contract: returns the selected session path, or None when
nothing was selected.
"""

import sys
from collections.abc import Awaitable, Callable
from typing import Any

import tonio.colored as tonio

from ..utils.colors import dim


# How many of the most recent sessions to offer.
_MAX_LISTED_SESSIONS = 20


def _format_session_line(index: int, session: Any) -> str:
    label = session.name or session.first_message.replace("\n", " ")
    if len(label) > 60:
        label = label[:59] + "…"
    modified = session.modified.strftime("%Y-%m-%d %H:%M")
    return f"  {index:>2}. {modified}  {label}  {dim(session.id[:8])}"


async def select_session(
    list_sessions: Callable[[Callable[[int, int], None] | None], Awaitable[list[Any]]],
    list_all_sessions: Callable[[Callable[[int, int], None] | None], Awaitable[list[Any]]],
    settings_manager: Any,
) -> str | None:
    """Pick a session to resume. Returns its path, or None if aborted."""
    if not sys.stdin.isatty():
        return None

    sessions = await list_sessions(None)
    if not sessions:
        sessions = await list_all_sessions(None)
    if not sessions:
        return None

    sessions = sorted(sessions, key=lambda session: session.modified, reverse=True)[:_MAX_LISTED_SESSIONS]

    print("Select a session to resume:", file=sys.stderr)
    for index, session in enumerate(sessions, start=1):
        print(_format_session_line(index, session), file=sys.stderr)
    print(f"Session [1-{len(sessions)}, empty to cancel]: ", end="", file=sys.stderr, flush=True)

    line = await tonio.spawn_blocking(sys.stdin.readline)
    choice = line.strip()
    if not choice:
        return None
    try:
        index = int(choice)
    except ValueError:
        return None
    if not 1 <= index <= len(sessions):
        return None
    return sessions[index - 1].path
