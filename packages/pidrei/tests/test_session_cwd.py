"""Mirrors pi coding-agent test/session-cwd.test.ts."""

import json
import os

import pytest

from pidrei.core.agent_session_runtime import create_agent_session_runtime
from pidrei.core.session_cwd import MissingSessionCwdError, SessionCwdIssue, get_missing_session_cwd_issue
from pidrei.core.session_manager import SessionManager


def _write_session_file(path: str, cwd: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "session",
                    "version": 3,
                    "id": "session-id",
                    "timestamp": "2025-01-01T00:00:00.000Z",
                    "cwd": cwd,
                }
            )
            + "\n"
        )


def test_detects_missing_session_cwd_from_persisted_sessions(tmp_path):
    fallback_cwd = str(tmp_path / "fallback")
    os.makedirs(fallback_cwd)
    missing_cwd = os.path.join(fallback_cwd, "does-not-exist")
    session_dir = str(tmp_path / "session-dir")
    os.makedirs(session_dir)
    session_file = os.path.join(session_dir, "session.jsonl")
    _write_session_file(session_file, missing_cwd)

    session_manager = SessionManager.open(session_file)
    issue = get_missing_session_cwd_issue(session_manager, fallback_cwd)
    assert issue == SessionCwdIssue(
        session_file=session_manager.get_session_file(),
        session_cwd=missing_cwd,
        fallback_cwd=fallback_cwd,
    )


def test_supports_overriding_effective_cwd_when_opening_session(tmp_path):
    fallback_cwd = str(tmp_path / "override")
    os.makedirs(fallback_cwd)
    missing_cwd = os.path.join(fallback_cwd, "does-not-exist")
    session_dir = str(tmp_path / "override-session-dir")
    os.makedirs(session_dir)
    session_file = os.path.join(session_dir, "session.jsonl")
    _write_session_file(session_file, missing_cwd)

    session_manager = SessionManager.open(session_file, None, fallback_cwd)
    assert session_manager.get_cwd() == fallback_cwd
    assert get_missing_session_cwd_issue(session_manager, fallback_cwd) is None


@pytest.mark.tonio
async def test_throws_controlled_error_before_runtime_creation_when_stored_cwd_missing(tmp_dir):
    fallback_cwd = os.path.join(str(tmp_dir), "runtime")
    os.makedirs(fallback_cwd)
    missing_cwd = os.path.join(fallback_cwd, "does-not-exist")
    session_dir = os.path.join(str(tmp_dir), "runtime-session-dir")
    os.makedirs(session_dir)
    session_file = os.path.join(session_dir, "session.jsonl")
    _write_session_file(session_file, missing_cwd)

    session_manager = SessionManager.open(session_file)
    create_runtime_called = False

    async def create_runtime(**_kwargs):
        nonlocal create_runtime_called
        create_runtime_called = True
        raise Exception("should not be called")

    with pytest.raises(MissingSessionCwdError):
        await create_agent_session_runtime(
            create_runtime,
            cwd=fallback_cwd,
            agent_dir=fallback_cwd,
            session_manager=session_manager,
        )
    assert create_runtime_called is False
