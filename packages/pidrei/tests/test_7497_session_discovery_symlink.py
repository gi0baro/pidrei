"""Mirror of pi's suite/regressions/7497-session-discovery-symlink.test.ts.

pidrei's `list_all` already used `os.path.isdir` (which follows links) where pi
used a `withFileTypes` dirent check, so the fix itself was a no-op here; the
mirror lands as a guard. Windows link-type branches are dropped (POSIX rule).
"""

import json
import os

import pytest

from pidrei.config import ENV_AGENT_DIR
from pidrei.core.session_manager import SessionManager


@pytest.fixture
def dirs(tmp_dir, request):
    agent_dir = tmp_dir / "agent"
    sessions_dir = agent_dir / "sessions"
    sessions_dir.mkdir(parents=True)

    previous = os.environ.get(ENV_AGENT_DIR)
    os.environ[ENV_AGENT_DIR] = str(agent_dir)

    def restore() -> None:
        if previous is None:
            os.environ.pop(ENV_AGENT_DIR, None)
        else:
            os.environ[ENV_AGENT_DIR] = previous

    request.addfinalizer(restore)
    return tmp_dir, sessions_dir


def _write_session(root, directory, session_id: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    header = {
        "type": "session",
        "version": 3,
        "id": session_id,
        "timestamp": "2026-08-03T00:00:00.000Z",
        "cwd": str(root / "project"),
    }
    (directory / f"{session_id}.jsonl").write_text(json.dumps(header) + "\n", encoding="utf-8")


@pytest.mark.tonio
async def test_discovers_a_session_through_a_directory_link_and_preserves_the_alias_path(dirs):
    root, sessions_dir = dirs
    target_dir = root / "linked-sessions"
    _write_session(root, target_dir, "linked")
    alias_dir = sessions_dir / "--linked--"
    os.symlink(target_dir, alias_dir, target_is_directory=True)

    sessions = await SessionManager.list_all()

    assert [session.id for session in sessions] == ["linked"]
    assert sessions[0].path == str(alias_dir / "linked.jsonl")


@pytest.mark.tonio
async def test_ignores_a_broken_directory_link_without_hiding_valid_sessions(dirs):
    root, sessions_dir = dirs
    _write_session(root, sessions_dir / "--regular--", "regular")
    target_dir = root / "removed-sessions"
    target_dir.mkdir()
    os.symlink(target_dir, sessions_dir / "--broken--", target_is_directory=True)
    target_dir.rmdir()

    sessions = await SessionManager.list_all()

    assert [session.id for session in sessions] == ["regular"]


@pytest.mark.tonio
async def test_ignores_links_to_files(dirs):
    root, sessions_dir = dirs
    _write_session(root, sessions_dir / "--regular--", "regular")
    target_file = root / "not-a-directory"
    target_file.write_text("", encoding="utf-8")
    os.symlink(target_file, sessions_dir / "--file--")

    sessions = await SessionManager.list_all()

    assert [session.id for session in sessions] == ["regular"]
