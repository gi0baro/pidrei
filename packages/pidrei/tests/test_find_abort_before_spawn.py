"""Mirror of pi coding-agent test/find-abort-before-spawn.test.ts.

pi mocks `child_process.spawn`, `ensureTool` and `pathExists`. The seams here
are find's own module bindings: `ensure_tool`, the `.git` lookup
(`_is_inside_git_repo_blocking`, which fires the token) and
`_run_streaming_lines`, which records a call instead of starting fd. pi waits
10 ms for the rest of `execute()` to run after the rejection; here the
rejection is `execute()` returning, so nothing runs after it.
"""

import pytest

import pidrei.core.tools.find as find_module
from pidrei.core.tools.find import create_find_tool_definition
from pidrei_utils.cancel import CancelToken


@pytest.mark.tonio
async def test_does_not_start_fd_after_an_abort_during_the_git_lookup(monkeypatch, tmp_path):
    cancel = CancelToken()
    spawned: list[list[str]] = []

    async def fake_ensure_tool(_tool, _on_status=None):
        return "fd"

    # Abort while find looks for a .git directory above the search path.
    def aborting_git_lookup_blocking(_path: str) -> bool:
        cancel.cancel()
        return False

    async def recording_run_streaming_lines(argv, _cancel, _on_line, *, cwd=None):
        spawned.append(argv)
        return 0, ""

    monkeypatch.setattr(find_module, "ensure_tool", fake_ensure_tool)
    monkeypatch.setattr(find_module, "_is_inside_git_repo_blocking", aborting_git_lookup_blocking)
    monkeypatch.setattr(find_module, "_run_streaming_lines", recording_run_streaming_lines)

    definition = create_find_tool_definition(str(tmp_path))
    # pi passes `{}` as the ctx stub; the Python "no cwd override" spelling is None.
    with pytest.raises(Exception, match="Operation aborted"):
        await definition.execute("call-1", {"pattern": "*.ts"}, cancel, None, None)

    assert spawned == []
