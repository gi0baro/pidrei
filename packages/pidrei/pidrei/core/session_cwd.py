"""Mirror of pi coding-agent src/core/session-cwd.ts.

The existence checks are async (pi's are `existsSync`): everything doing I/O is.
"""

from dataclasses import dataclass
from typing import Any

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
    return f"cwd from session file does not exist\n{issue.session_cwd}\n\ncontinue in current cwd\n{issue.fallback_cwd}"


class MissingSessionCwdError(Exception):
    def __init__(self, issue: SessionCwdIssue):
        super().__init__(format_missing_session_cwd_error(issue))
        self.name = "MissingSessionCwdError"
        self.issue = issue


async def assert_session_cwd_exists(session_manager: Any, fallback_cwd: str) -> None:
    issue = await get_missing_session_cwd_issue(session_manager, fallback_cwd)
    if issue:
        raise MissingSessionCwdError(issue)
