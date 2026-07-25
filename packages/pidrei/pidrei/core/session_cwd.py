"""Mirror of pi coding-agent src/core/session-cwd.ts."""

import os
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SessionCwdIssue:
    session_cwd: str
    fallback_cwd: str
    session_file: str | None = None


def get_missing_session_cwd_issue(session_manager: Any, fallback_cwd: str) -> SessionCwdIssue | None:
    session_file = session_manager.get_session_file()
    if not session_file:
        return None

    session_cwd = session_manager.get_cwd()
    if not session_cwd or os.path.exists(session_cwd):
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


def assert_session_cwd_exists(session_manager: Any, fallback_cwd: str) -> None:
    issue = get_missing_session_cwd_issue(session_manager, fallback_cwd)
    if issue:
        raise MissingSessionCwdError(issue)
