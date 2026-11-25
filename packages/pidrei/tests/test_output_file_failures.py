"""Mirror of pi coding-agent test/output-file-failures.test.ts.

#10585: a full, over-quota or unwritable temp dir must not crash pidrei. Bash
output that cannot be saved to a temp file keeps its in-memory tail and says
why the full output is missing.

Adaptations:
- Every output file goes to the test's own directory: `output_files.TEMP_DIR`
  (resolved once at import) is swapped where pi stubs TMPDIR.
- An open failure is a real one: the temp dir points at a missing directory.
  A file already at the path is real too: the writer is given a path that
  exists. pi injects the other two (every write fails with ENOSPC, closing the
  descriptor fails with EDQUOT) by mocking `createWriteStream`; here
  `temp_file_writer.create_private_file_blocking` is swapped for one that
  opens the file for real and returns a handle whose `write` or `close` raises.
- The reason is Python's error text (`[Errno 28] No space left on device`),
  where pi's is Node's (`ENOSPC: no space left on device, write`), so the cases
  match the errno.
- pi's `vi.waitFor` on a failed write waits for the writer coroutine to end
  (its finished Event), bounded.
- pi aborts and sends late output with `setImmediate` while the `!` executor
  closes its file; here the file's `close()` is wrapped to do both, so they
  land after the command ended and before the file is closed, without timing.
"""

import dataclasses
import errno
import os
import re
from types import SimpleNamespace

import pytest
from tonio.colored import fs

from pidrei.core import bash_executor
from pidrei.core.bash_executor import execute_bash_with_operations
from pidrei.core.messages import BashExecutionMessage, bash_execution_to_text
from pidrei.core.tools.bash import BashExecResult, create_bash_tool, create_bash_tool_definition
from pidrei.modes.interactive.components import BashExecutionComponent
from pidrei.modes.interactive.theme import init_theme, theme
from pidrei.utils import output_files, temp_file_writer
from pidrei.utils.ansi import strip_ansi
from pidrei.utils.output_files import FailedOutputFile, SavedOutputFile, TempFileWriter, create_output_file_stream
from pidrei_utils.cancel import CancelToken


_OPEN_CODE = f"[Errno {errno.ENOENT}]"
_WRITE_CODE = f"[Errno {errno.ENOSPC}]"
_CLOSE_CODE = f"[Errno {errno.EDQUOT}]"
_EXISTS_CODE = f"[Errno {errno.EEXIST}]"


def printing(text: str, exit_code: int = 0):
    """Operations that print `text` in 64 KiB chunks and exit with `exit_code`."""
    data = text.encode()

    class Operations:
        async def exec(self, _command, _cwd, *, on_data, cancel=None, timeout=None, env=None):
            for i in range(0, len(data), 65536):
                on_data(data[i : i + 65536])
            return BashExecResult(exit_code=exit_code)

    return Operations()


def numbers(count: int) -> str:
    """`seq 1 count` output."""
    return "".join(f"{i + 1}\n" for i in range(count))


# 3000 numbered lines: over the 2000-line limit, so the full output goes to a temp file; about 14 KB.
SHORT = numbers(3000)
# 300000 numbered lines, about 2 MB: more than the accumulator keeps in memory.
LONG = numbers(300_000)


class _FailingHandle:
    """A file opened for real whose every write fails (ENOSPC), or whose close fails
    after closing the descriptor (a delayed EDQUOT on quota and network file systems)."""

    def __init__(self, handle, failure: str):
        self._handle = handle
        self._failure = failure

    def write(self, data: bytes) -> int:
        if self._failure == "write":
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        return self._handle.write(data)

    def close(self) -> None:
        self._handle.close()
        if self._failure == "close":
            raise OSError(errno.EDQUOT, os.strerror(errno.EDQUOT))


@pytest.fixture
def temp_root(tmp_path, monkeypatch):
    # Every file the tests create goes here and is removed with it.
    monkeypatch.setattr(output_files, "TEMP_DIR", fs.Path(str(tmp_path)))
    return tmp_path


@pytest.fixture
def created_paths():
    return []


@pytest.fixture
def fail_output_files(temp_root, monkeypatch, created_paths):
    """Make output files fail on open (missing temp dir), write (full disk) or
    close (quota). Returns the errno text the reason carries."""

    def fail(failure: str) -> str:
        if failure == "open":
            monkeypatch.setattr(output_files, "TEMP_DIR", fs.Path(str(temp_root / "missing")))
            return _OPEN_CODE
        create = temp_file_writer.create_private_file_blocking

        def create_failing(path: str):
            handle = create(path)
            created_paths.append(path)
            return _FailingHandle(handle, failure)

        monkeypatch.setattr(temp_file_writer, "create_private_file_blocking", create_failing)
        return _WRITE_CODE if failure == "write" else _CLOSE_CODE

    return fail


@pytest.fixture(autouse=True)
async def _theme():
    await init_theme(None, False)


def _open_output_file() -> tuple[TempFileWriter, str]:
    """Open an output file and return it with its path, which a failed result no longer carries."""
    file = create_output_file_stream("pidrei-test", ".log")
    assert file.result.kind == "saved", "a new output file starts as saved"
    return file, file.result.path


async def _exists(path: str) -> bool:
    return await fs.Path(path).exists()


def _text(result) -> str:
    return "".join(block.text for block in result.content if getattr(block, "type", None) == "text")


# -- create_output_file_stream ------------------------------------------------


@pytest.mark.tonio
async def test_reports_an_open_failure_from_close_instead_of_crashing(fail_output_files):
    code = fail_output_files("open")
    file, path = _open_output_file()
    file.write(b"data")
    result = await file.close()
    assert isinstance(result, FailedOutputFile)
    assert code in result.error
    assert not await _exists(path)


@pytest.mark.tonio
async def test_drops_a_partially_written_file_and_does_not_hang_on_close(fail_output_files):
    code = fail_output_files("write")
    file, path = _open_output_file()
    file.write(b"data")
    # The writer coroutine ends on the failure, before any close().
    await file._finished.wait(5)
    assert file._finished.is_set()
    assert isinstance(file.result, FailedOutputFile)
    assert code in file.result.error
    file.write(b"more")
    result = await file.close()
    assert isinstance(result, FailedOutputFile)
    assert code in result.error
    assert not await _exists(path)


@pytest.mark.tonio
async def test_reports_a_failure_while_closing_the_file_not_saved(fail_output_files):
    code = fail_output_files("close")
    file, path = _open_output_file()
    file.write(b"data")
    result = await file.close()
    assert isinstance(result, FailedOutputFile)
    assert code in result.error
    assert not await _exists(path)


@pytest.mark.tonio
async def test_leaves_a_file_it_did_not_create_in_place(temp_root):
    path = str(temp_root / "pidrei-test-existing.log")
    await fs.Path(path).write_text("someone else's")
    file = TempFileWriter(path)
    file.write(b"data")
    result = await file.close()
    assert isinstance(result, FailedOutputFile)
    assert _EXISTS_CODE in result.error
    assert await fs.Path(path).read_text() == "someone else's"


@pytest.mark.tonio
async def test_saves_the_full_output_when_writing_succeeds(temp_root):
    file, path = _open_output_file()
    file.write(b"data")
    assert await file.close() == SavedOutputFile(path)
    assert file.result == SavedOutputFile(path)
    assert await fs.Path(path).read_text() == "data"


@pytest.mark.tonio
async def test_ignores_writes_once_closing_started(temp_root):
    file, path = _open_output_file()
    file.write(b"data")
    closing = file.close()
    file.write(b"late")
    assert await closing == SavedOutputFile(path)
    assert await fs.Path(path).read_text() == "data"


# -- bash tool ----------------------------------------------------------------


@pytest.mark.tonio
@pytest.mark.parametrize("failure", ["open", "write"])
async def test_bash_tool_keeps_the_output_tail_when_the_temp_file_fails(
    temp_root, fail_output_files, created_paths, failure
):
    code = fail_output_files(failure)
    result = await create_bash_tool(str(temp_root), operations=printing(LONG)).execute(
        "call", {"command": "seq 300000"}
    )

    text = _text(result)
    assert "\n299999\n300000\n" in text
    assert re.search(
        rf"\[Showing lines \d+-300000 of 300000\. Could not save the full output: .*{re.escape(code)}", text
    )
    assert result.details.full_output_path is None
    assert code in result.details.full_output_error
    structured = result.structured_content
    assert structured["truncated"] is True
    assert "full_output_path" not in structured
    assert code in structured["full_output_error"]
    assert re.match(r"^\[\.\.\. [1-9]\d* bytes omitted \.\.\.\]\n\n", structured["output"])
    assert structured["output"].endswith("299999\n300000\n")
    for path in created_paths:
        assert not await _exists(path)


@pytest.mark.tonio
async def test_gives_scripts_the_whole_output_when_it_all_fits_in_memory(temp_root, fail_output_files):
    code = fail_output_files("open")
    result = await create_bash_tool(str(temp_root), operations=printing(SHORT)).execute("call", {"command": "seq 3000"})

    # The model sees the usual 2000 lines, without the file.
    assert result.details.truncation.truncated is True
    assert code in result.details.full_output_error
    # Scripts take up to 1 MiB and lose nothing.
    assert result.structured_content["output"] == SHORT
    assert result.structured_content["truncated"] is False
    assert "full_output_error" not in result.structured_content


@pytest.mark.tonio
async def test_keeps_the_end_of_one_over_long_line_for_scripts(temp_root, fail_output_files):
    fail_output_files("open")
    result = await create_bash_tool(str(temp_root), operations=printing(f"{'x' * 300_000}\n")).execute(
        "call", {"command": "long-line"}
    )

    structured = result.structured_content
    assert structured["truncated"] is True
    assert re.match(r"^\[\.\.\. [1-9]\d* bytes omitted \.\.\.\]\n\nx{1000,}\n$", structured["output"])


@pytest.mark.tonio
async def test_shows_why_the_full_output_is_missing_once_below_the_output(temp_root, fail_output_files):
    code = fail_output_files("open")
    operations = printing(LONG)
    result = await create_bash_tool(str(temp_root), operations=operations).execute("call", {"command": "seq 300000"})
    component = create_bash_tool_definition(str(temp_root), operations=operations).render_result(
        result,
        {"expanded": False, "isPartial": False},
        theme,
        {
            "state": {},
            "showImages": False,
            "executionStarted": True,
            "isError": False,
            "durationMs": None,
            "invalidate": lambda: None,
        },
    )

    # The notice holds the temp file path, so it wraps when the temp dir path is long (macOS).
    rendered = re.sub(r"\s+", " ", strip_ansi("\n".join(component.render(200))))
    assert "[Showing lines" not in rendered
    assert re.search(rf"\[Could not save the full output: .*{re.escape(code)}.*\. Truncated: ", rendered)


# -- user bash executor -------------------------------------------------------


@pytest.mark.tonio
@pytest.mark.parametrize("failure", ["open", "write"])
async def test_user_bash_keeps_the_output_tail_when_the_temp_file_fails(
    temp_root, fail_output_files, created_paths, failure
):
    code = fail_output_files(failure)
    result = await execute_bash_with_operations("seq 3000", str(temp_root), printing(SHORT))

    assert result.truncated is True
    assert result.full_output_path is None
    assert code in result.full_output_error
    assert re.search(r"\n2999\n3000\n?$", result.output)
    for path in created_paths:
        assert not await _exists(path)

    # What the model reads and what the user sees both say the output is cut, and why.
    message = BashExecutionMessage(command="seq 3000", timestamp=0, **dataclasses.asdict(result))
    assert re.search(
        rf"\[Output truncated\. Could not save the full output: .*{re.escape(code)}.*\]$",
        bash_execution_to_text(message),
    )
    component = BashExecutionComponent("seq 3000", SimpleNamespace(request_render=lambda: None))
    component.append_output(result.output)
    # As interactive mode passes it.
    component.set_complete(
        result.exit_code,
        result.cancelled,
        SimpleNamespace(truncated=True, content=result.output),
        result.full_output_path,
        result.full_output_error,
    )
    # The line wraps at the terminal width.
    rendered = re.sub(r"\s+", " ", strip_ansi(" ".join(component.render(120))))
    assert re.search(rf"Output truncated\. Could not save the full output: .*{re.escape(code)}", rendered)


@pytest.mark.tonio
async def test_reports_no_failure_for_output_that_is_not_truncated(temp_root, fail_output_files):
    fail_output_files("open")
    # Over 50 KB of color codes around one short line: the temp file opens on the raw bytes, but
    # stripping the codes leaves nothing to truncate.
    colored = "\x1b[31m\x1b[0m" * 7000 + "done\n"
    result = await execute_bash_with_operations("build", str(temp_root), printing(colored))

    assert result.truncated is False
    assert result.full_output_error is None


@pytest.mark.tonio
async def test_decides_the_result_when_the_command_ends_not_after_the_file_is_closed(temp_root, monkeypatch):
    cancel = CancelToken()
    late = b"late output\n"
    late_data = {}

    class Operations:
        async def exec(self, _command, _cwd, *, on_data, cancel=None, timeout=None, env=None):
            on_data(LONG.encode())
            late_data["on_data"] = on_data
            return BashExecResult(exit_code=7)

    create = bash_executor.create_output_file_stream

    def create_with_late_events(prefix: str, extension: str):
        file = create(prefix, extension)
        close = file.close

        def close_with_late_events():
            # While the file is being closed: an abort, and output from a process the command left running.
            cancel.cancel()
            late_data["on_data"](late)
            return close()

        file.close = close_with_late_events
        return file

    monkeypatch.setattr(bash_executor, "create_output_file_stream", create_with_late_events)
    result = await execute_bash_with_operations("seq 300000", str(temp_root), Operations(), cancel=cancel)

    assert result.exit_code == 7
    assert result.cancelled is False
    assert result.truncated is True
    assert "late output" not in result.output
    assert await fs.Path(result.full_output_path).read_text() == LONG


# Output truncated without a saved file and without a reason (for example a result an extension returned).
def test_still_says_the_output_is_truncated():
    text = bash_execution_to_text(
        BashExecutionMessage(command="x", output="tail", exit_code=0, cancelled=False, truncated=True, timestamp=0)
    )
    assert re.search(r"\[Output truncated\]$", text)
