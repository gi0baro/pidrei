"""Mirror of pi coding-agent test/bash-executor-truncation.test.ts.

pi waits for the log file with `vi.waitFor` because its executor closes the
file stream without awaiting it; `execute_bash_with_operations` drains the
temp file before it returns, so the file is read directly.
"""

import os

import pytest

from pidrei.core.bash_executor import execute_bash_with_operations
from pidrei.core.messages import BashExecutionMessage, bash_execution_to_text
from pidrei.core.tools.bash import BashExecResult
from pidrei_utils.cancel import CancelToken


# #10164: the retained tail can fit even though earlier output was discarded.
@pytest.mark.tonio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_reports_discarded_chunks(cancelled):
    first = "error: missing build dependency\n".ljust(64 * 1024, "x")
    last = "remaining build output\n".ljust(44 * 1024, "y")
    cancel = CancelToken()

    class Operations:
        async def exec(self, _command, _cwd, *, on_data, cancel=None, timeout=None, env=None):
            on_data(first.encode())
            on_data(last.encode())
            if cancelled:
                cancel.cancel()
                raise Exception("aborted")
            return BashExecResult(exit_code=0)

    result = await execute_bash_with_operations("cat build.log", os.getcwd(), Operations(), cancel=cancel)
    try:
        with open(result.full_output_path, encoding="utf-8") as full_output:
            assert full_output.read() == first + last

        assert result.cancelled is cancelled
        assert result.output == last
        assert result.truncated is True
        message = BashExecutionMessage(
            command="cat build.log",
            output=result.output,
            exit_code=result.exit_code,
            cancelled=result.cancelled,
            truncated=result.truncated,
            timestamp=0,
            full_output_path=result.full_output_path,
        )
        assert f"[Output truncated. Full output: {result.full_output_path}]" in bash_execution_to_text(message)
    finally:
        if result.full_output_path:
            os.remove(result.full_output_path)


@pytest.mark.tonio
async def test_does_not_mark_short_output_as_truncated():
    class Operations:
        async def exec(self, _command, _cwd, *, on_data, cancel=None, timeout=None, env=None):
            on_data(b"done\n")
            return BashExecResult(exit_code=0)

    result = await execute_bash_with_operations("echo done", os.getcwd(), Operations())

    assert result.output == "done\n"
    assert result.truncated is False
    assert result.full_output_path is None
