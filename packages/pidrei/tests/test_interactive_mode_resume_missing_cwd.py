"""Mirror of pi coding-agent test/interactive-mode-resume-missing-cwd.test.ts.

pi calls the prototype methods on a plain context object; here they are bound to a
`SimpleNamespace` context with `functools.partial`, as in
`test_interactive_mode_import_command.py`. `vi.fn` spies are recording lists.
"""

import json
import os
from datetime import UTC, datetime
from functools import partial
from types import SimpleNamespace

import pytest

from pidrei.core.session_cwd import MissingSessionCwdError, SessionCwdIssue
from pidrei.core.session_manager import SessionManager
from pidrei.modes.interactive.interactive_mode import InteractiveMode


def _assert_matches(actual: dict, expected: dict) -> None:
    for key, value in expected.items():
        assert actual.get(key) == value, f"{key}: {actual.get(key)!r} != {value!r}"


async def _setup(tmp_path, *, choice: str | None, switch_cancelled: bool):
    current_cwd = str(tmp_path / "current")
    missing_cwd = os.path.join(current_cwd, "renamed-away")
    old_session_dir = str(tmp_path / "old-sessions")
    current_session_dir = str(tmp_path / "current-sessions")
    for directory in (current_cwd, old_session_dir, current_session_dir):
        os.makedirs(directory)

    session_path = os.path.join(old_session_dir, "orphan.jsonl")
    timestamp = datetime.now(UTC).isoformat()
    original = (
        json.dumps({"type": "session", "version": 3, "id": "orphan", "timestamp": timestamp, "cwd": missing_cwd})
        + "\n"
        + json.dumps(
            {
                "type": "message",
                "id": "m1",
                "parentId": None,
                "timestamp": timestamp,
                "message": {"role": "user", "content": "hello", "timestamp": 1},
            }
        )
        + "\n"
    )
    with open(session_path, "w", encoding="utf-8") as handle:
        handle.write(original)

    switch_calls: list = []

    async def switch_session(path, *, cwd_override=None, with_session=None, project_trust_context_factory=None):
        switch_calls.append((path, cwd_override))
        if path == session_path and cwd_override is None:
            raise MissingSessionCwdError(
                SessionCwdIssue(session_file=session_path, session_cwd=missing_cwd, fallback_cwd=current_cwd)
            )
        return {"cancelled": switch_cancelled}

    async def show_extension_selector(_title, labels, _opts=None, _blocked=None):
        return labels[["continue", "fork", None].index(choice)]

    async def handle_fatal_runtime_error(_prefix, _error):
        raise AssertionError("unexpected fatal error")

    status: list[str] = []
    context = SimpleNamespace(
        runtime_host=SimpleNamespace(switch_session=switch_session),
        session_manager=await SessionManager(current_cwd, current_session_dir),
        show_status=status.append,
        _clear_status_indicator=lambda kind=None: None,
        _show_extension_selector=show_extension_selector,
        _create_project_trust_context=lambda _cwd: None,
        _handle_fatal_runtime_error=handle_fatal_runtime_error,
    )
    context._prompt_for_missing_session_cwd = partial(InteractiveMode._prompt_for_missing_session_cwd, context)
    context._switch_to_forked_session = partial(InteractiveMode._switch_to_forked_session, context)
    return SimpleNamespace(
        context=context,
        current_cwd=current_cwd,
        current_session_dir=current_session_dir,
        session_path=session_path,
        original=original,
        switch_calls=switch_calls,
        status=status,
    )


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


# #11060
@pytest.mark.tonio
async def test_forks_the_session_into_the_current_cwd_and_leaves_the_original_unchanged(tmp_path):
    setup = await _setup(tmp_path, choice="fork", switch_cancelled=False)

    result = await InteractiveMode._handle_resume_session(setup.context, setup.session_path)

    assert result == {"cancelled": False}
    fork_path = setup.switch_calls[1][0]
    assert fork_path.startswith(setup.current_session_dir)
    lines = _read(fork_path).strip().split("\n")
    header = json.loads(lines[0])
    _assert_matches(header, {"type": "session", "cwd": setup.current_cwd, "parentSession": setup.session_path})
    assert header["id"] != "orphan"
    _assert_matches(json.loads(lines[1]), {"id": "m1"})
    assert _read(setup.session_path) == setup.original
    assert f"Forked into new session {header['id']} in {setup.current_cwd}" in setup.status


@pytest.mark.tonio
async def test_continues_the_original_session_in_the_current_cwd_without_writing_a_file(tmp_path):
    setup = await _setup(tmp_path, choice="continue", switch_cancelled=False)

    result = await InteractiveMode._handle_resume_session(setup.context, setup.session_path)

    assert result == {"cancelled": False}
    assert setup.switch_calls[-1] == (setup.session_path, setup.current_cwd)
    assert _read(setup.session_path) == setup.original
    assert os.listdir(setup.current_session_dir) == []
    assert f"Resumed session in {setup.current_cwd}" in setup.status


@pytest.mark.tonio
async def test_removes_the_fork_when_an_extension_cancels_the_switch(tmp_path):
    setup = await _setup(tmp_path, choice="fork", switch_cancelled=True)

    result = await InteractiveMode._handle_resume_session(setup.context, setup.session_path)

    assert result == {"cancelled": True}
    assert not os.path.exists(setup.switch_calls[1][0])
    assert os.listdir(setup.current_session_dir) == []


@pytest.mark.tonio
async def test_creates_no_fork_when_the_user_cancels(tmp_path):
    setup = await _setup(tmp_path, choice=None, switch_cancelled=False)

    result = await InteractiveMode._handle_resume_session(setup.context, setup.session_path)

    assert result == {"cancelled": True}
    assert len(setup.switch_calls) == 1
    assert os.listdir(setup.current_session_dir) == []
    assert "Resume cancelled" in setup.status
