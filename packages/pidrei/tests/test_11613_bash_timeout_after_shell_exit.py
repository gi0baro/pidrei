"""Mirror of pi's suite/regressions/11613-bash-timeout-after-shell-exit.test.ts.

The bash tool waits for stdout/stderr to close, not only for the shell to
exit (#5303). A background process that inherited those pipes and keeps
writing keeps the tool waiting after the shell exited. When the timeout then
hit, the tool reported only "Command timed out", so the shell's real exit code
and the fact that it had already finished were lost. The tool now reports
both.

Adaptations:
- The post-exit grace window (`EXIT_STDIO_GRACE_S`) is widened for the cases
  where the shell exits first, so that a background writer stalled for 100 ms
  on a loaded runner cannot end the wait before the deadline or the abort;
  pi's cases rely on the writer never pausing that long.
- The background writers stop on their own after a few seconds, and every
  call is bounded, so a regression fails fast without leaving a writer
  behind.
- pidrei-only: an abort after the shell exited still kills the background
  writer (the deadline and the abort used to be disarmed at shell exit). The
  writer prints `shell-gone` once the shell has been reaped (`kill -0` stops
  succeeding), and the abort waits for that line.
"""

import re

import pytest
import tonio.colored as tonio
from tonio.colored import time as tonio_time

import pidrei.core.tools.bash as bash_module
from pidrei.core.tools import create_bash_tool
from pidrei_utils.cancel import CancelToken


# Bounds every call: far past each case's deadline, well short of the writers' own end.
CALL_BOUND_S = 5
# Prints `tick` every 20 ms for about ten seconds, then stops by itself.
BOUNDED_WRITER = "i=0; while [ $i -lt 500 ]; do echo tick; sleep 0.02; i=$((i + 1)); done"


async def _run_expecting_error(bash_tool, params, cancel=None, on_update=None) -> str:
    with pytest.raises(Exception) as excinfo:
        await tonio_time.timeout(bash_tool.execute("test-11613", params, cancel, on_update), CALL_BOUND_S)
    return str(excinfo.value)


@pytest.mark.tonio
async def test_reports_the_shells_exit_code_when_background_processes_kept_writing_until_the_timeout(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(bash_module, "EXIT_STDIO_GRACE_S", 30.0)
    bash_tool = create_bash_tool(str(tmp_path))

    message = await _run_expecting_error(
        bash_tool, {"command": f"({BOUNDED_WRITER}) & sleep 0.2; echo boom >&2; exit 3", "timeout": 1}
    )

    assert "boom" in message
    assert re.search(
        r"Command timed out after 1 seconds\. The shell exited with code 3 after 0\.\d seconds, "
        r"but background processes it started kept writing output until the timeout\.$",
        message,
    )


@pytest.mark.tonio
async def test_keeps_the_plain_timeout_message_when_the_shell_was_still_running_at_the_deadline(tmp_path):
    bash_tool = create_bash_tool(str(tmp_path))

    message = await _run_expecting_error(bash_tool, {"command": "echo started; sleep 30", "timeout": 0.3})

    assert re.search(r"started\s+Command timed out after 0\.3 seconds$", message)


@pytest.mark.tonio
async def test_abort_after_the_shell_exited_kills_its_background_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(bash_module, "EXIT_STDIO_GRACE_S", 30.0)
    bash_tool = create_bash_tool(str(tmp_path))
    cancel = CancelToken()
    shell_gone = tonio.Event()

    def on_update(result) -> None:
        if any("shell-gone" in getattr(block, "text", "") for block in result.content):
            shell_gone.set()

    async def abort_once_the_shell_is_gone() -> None:
        await shell_gone.wait(CALL_BOUND_S)
        cancel.cancel()

    aborter = tonio.spawn(abort_once_the_shell_is_gone())
    command = f"(while kill -0 $$ 2>/dev/null; do sleep 0.01; done; echo shell-gone; {BOUNDED_WRITER}) & exit 3"

    message = await _run_expecting_error(bash_tool, {"command": command}, cancel, on_update)
    await aborter

    assert shell_gone.is_set()
    assert re.search(r"shell-gone\s+(tick\s+)*Command aborted$", message)
