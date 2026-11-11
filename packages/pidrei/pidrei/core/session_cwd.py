"""Mirror of pi coding-agent src/core/session-cwd.ts.

The existence checks are async (pi's are `existsSync`): everything doing I/O is.
"""

from dataclasses import dataclass
from typing import Any, Literal

from tonio.colored import fs


@dataclass(slots=True)
class SessionCwdIssue:
    session_cwd: str
    fallback_cwd: str
    session_file: str | None = None


async def get_missing_session_cwd_issue(session_manager: Any, fallback_cwd: str) -> SessionCwdIssue | None:
    session_file = session_manager.get_session_file()
    if not session_file:
        return None

    session_cwd = session_manager.get_cwd()
    if not session_cwd or await fs.Path(session_cwd).exists():
        return None

    return SessionCwdIssue(session_file=session_file, session_cwd=session_cwd, fallback_cwd=fallback_cwd)


def format_missing_session_cwd_error(issue: SessionCwdIssue) -> str:
    session_file = f"\nSession file: {issue.session_file}" if issue.session_file else ""
    return (
        f"Stored session working directory does not exist: {issue.session_cwd}{session_file}"
        f"\nCurrent working directory: {issue.fallback_cwd}"
    )


def format_missing_session_cwd_prompt(issue: SessionCwdIssue) -> str:
    return f"Session cwd does not exist: {issue.session_cwd}"


# How to open a session whose stored cwd is missing: run it in the current cwd once, or fork it there.
type MissingSessionCwdChoice = Literal["continue", "fork"]


def get_missing_session_cwd_options(issue: SessionCwdIssue) -> list[dict[str, Any]]:
    """Prompt options for a missing session cwd, naming the current cwd; a `None` value cancels."""
    return [
        {"label": f"Continue in {issue.fallback_cwd} (asks again next time)", "value": "continue"},
        {"label": f"Fork into {issue.fallback_cwd} (new session ID, original kept)", "value": "fork"},
        {"label": "Cancel", "value": None},
    ]


class MissingSessionCwdError(Exception):
    def __init__(self, issue: SessionCwdIssue):
        super().__init__(format_missing_session_cwd_error(issue))
        self.name = "MissingSessionCwdError"
        self.issue = issue


async def assert_session_cwd_exists(session_manager: Any, fallback_cwd: str) -> None:
    issue = await get_missing_session_cwd_issue(session_manager, fallback_cwd)
    if issue:
        raise MissingSessionCwdError(issue)
