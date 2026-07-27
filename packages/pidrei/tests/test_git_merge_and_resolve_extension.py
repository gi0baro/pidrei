"""Mirror of pi's git-merge-and-resolve-extension.test.ts.

pi keys its exec stub on `[cmd, ...args].join(" ")`; same here, and an
unlisted command falls through to a failure exactly as pi's `?? fail` does.
"""

import os
import shutil
import tempfile
from types import SimpleNamespace

import pytest

from pidrei.core.exec import ExecResult

from .example_extensions import load_example


OK = ExecResult(stdout="", stderr="", code=0, killed=False)
FAIL = ExecResult(stdout="", stderr="error", code=1, killed=False)


def with_upstream(results: dict[str, ExecResult]) -> dict[str, ExecResult]:
    """A clean repo tracking origin/main, not in a merge."""
    results.setdefault("git rev-parse --git-dir", OK)
    results.setdefault("git rev-parse MERGE_HEAD", FAIL)
    results.setdefault("git status --porcelain", OK)
    results.setdefault(
        "git rev-parse --abbrev-ref --symbolic-full-name @{u}",
        ExecResult(stdout="origin/main\n", stderr="", code=0, killed=False),
    )
    results.setdefault("git fetch origin", OK)
    return results


@pytest.fixture
def temp_dir(request):
    path = tempfile.mkdtemp(prefix="pidrei-merge-test-")
    request.addfinalizer(lambda: shutil.rmtree(path, ignore_errors=True))
    return path


def write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


async def setup(cwd: str, exec_results: dict[str, ExecResult]):
    handlers: dict = {}
    exec_calls: list[tuple[str, list[str]]] = []
    messages: list[tuple] = []

    async def fake_exec(command, args, **_kwargs):
        exec_calls.append((command, args))
        return exec_results.get(" ".join([command, *args]), FAIL)

    api = SimpleNamespace(
        on=lambda event, handler: handlers.__setitem__(event, handler),
        exec=fake_exec,
        send_user_message=lambda content, options=None: messages.append((content, options)),
    )
    load_example("git_merge_and_resolve").extension(api)

    ctx = SimpleNamespace(cwd=cwd, ui=SimpleNamespace(notify=lambda *_args: None))

    async def trigger() -> None:
        await handlers["agent_end"]({"type": "agent_end"}, ctx)

    return trigger, exec_calls, messages


@pytest.mark.tonio
async def test_skips_when_not_a_git_repository(temp_dir):
    trigger, exec_calls, messages = await setup(temp_dir, {"git rev-parse --git-dir": FAIL})

    await trigger()

    assert len(exec_calls) == 1
    assert messages == []


@pytest.mark.tonio
async def test_skips_when_no_upstream_is_configured(temp_dir):
    trigger, _exec_calls, messages = await setup(
        temp_dir,
        {
            "git rev-parse --git-dir": OK,
            "git rev-parse --abbrev-ref --symbolic-full-name @{u}": FAIL,
        },
    )

    await trigger()

    assert messages == []


@pytest.mark.tonio
async def test_resends_conflicts_when_in_an_unfinished_merge(temp_dir):
    write(
        os.path.join(temp_dir, "file.py"),
        "<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> origin/main",
    )
    trigger, exec_calls, messages = await setup(
        temp_dir,
        {
            "git rev-parse --git-dir": OK,
            "git rev-parse MERGE_HEAD": OK,
            "git diff --name-only --diff-filter=U": ExecResult(stdout="file.py\n", stderr="", code=0, killed=False),
        },
    )

    await trigger()

    # No new fetch or merge is attempted.
    assert ("git", ["fetch", "origin"]) not in exec_calls
    assert len(messages) == 1
    assert "file.py:1-5" in messages[0][0]


@pytest.mark.tonio
async def test_skips_when_the_working_tree_is_dirty_and_not_in_a_merge(temp_dir):
    trigger, exec_calls, messages = await setup(
        temp_dir,
        {
            "git rev-parse --git-dir": OK,
            "git rev-parse MERGE_HEAD": FAIL,
            "git status --porcelain": ExecResult(stdout=" M src/index.py\n", stderr="", code=0, killed=False),
        },
    )

    await trigger()

    assert ("git", ["fetch", "origin"]) not in exec_calls
    assert messages == []


@pytest.mark.tonio
async def test_skips_when_fetch_fails(temp_dir):
    trigger, _exec_calls, messages = await setup(temp_dir, with_upstream({"git fetch origin": FAIL}))

    await trigger()

    assert messages == []


@pytest.mark.tonio
async def test_skips_when_the_merge_is_clean(temp_dir):
    trigger, _exec_calls, messages = await setup(temp_dir, with_upstream({"git merge --no-ff origin/main": OK}))

    await trigger()

    assert messages == []


@pytest.mark.tonio
async def test_sends_the_conflict_report_as_a_follow_up(temp_dir):
    write(
        os.path.join(temp_dir, "src", "index.py"),
        (
            "line 1\n"
            "<<<<<<< HEAD\n"
            "our change\n"
            "=======\n"
            "their change\n"
            ">>>>>>> origin/main\n"
            "line 7\n"
            "<<<<<<< HEAD\n"
            "second conflict\n"
            "=======\n"
            "their second\n"
            ">>>>>>> origin/main"
        ),
    )
    trigger, _exec_calls, messages = await setup(
        temp_dir,
        with_upstream(
            {
                "git merge --no-ff origin/main": FAIL,
                "git diff --name-only --diff-filter=U": ExecResult(
                    stdout="src/index.py\n", stderr="", code=0, killed=False
                ),
            }
        ),
    )

    await trigger()

    assert len(messages) == 1
    content, options = messages[0]
    assert "src/index.py:2-6 (ours 3, theirs 5)" in content
    assert "src/index.py:8-12 (ours 9, theirs 11)" in content
    assert options == {"deliverAs": "followUp"}


@pytest.mark.tonio
async def test_handles_empty_ours_or_theirs_sections(temp_dir):
    write(
        os.path.join(temp_dir, "empty-ours.py"),
        "<<<<<<< HEAD\n=======\nonly theirs\n>>>>>>> origin/main",
    )
    trigger, _exec_calls, messages = await setup(
        temp_dir,
        with_upstream(
            {
                "git merge --no-ff origin/main": FAIL,
                "git diff --name-only --diff-filter=U": ExecResult(
                    stdout="empty-ours.py\n", stderr="", code=0, killed=False
                ),
            }
        ),
    )

    await trigger()

    assert len(messages) == 1
    assert "empty-ours.py:1-4 (ours empty, theirs 3)" in messages[0][0]


@pytest.mark.tonio
async def test_sends_nothing_when_the_merge_fails_but_no_markers_are_found(temp_dir):
    trigger, _exec_calls, messages = await setup(
        temp_dir,
        with_upstream(
            {
                "git merge --no-ff origin/main": FAIL,
                "git diff --name-only --diff-filter=U": OK,
            }
        ),
    )

    await trigger()

    assert messages == []
