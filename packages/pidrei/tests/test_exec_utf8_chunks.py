"""Mirror of pi coding-agent test/exec-utf8-chunks.test.ts.

pi stubs `spawn` with PassThrough streams to fix the chunk boundaries. Here
the child is real (TonIO's process is never stubbed) and writes the same three
pieces with pauses between them; the expected result holds whichever way the
pipes deliver them, since the cap is exceeded by the last piece in every case.
"""

import os
import sys

import pytest

from pidrei.core.exec import exec_command


# #11360
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
@pytest.mark.tonio
async def test_discards_pending_utf8_bytes_when_either_stream_exceeds_the_cap(stream):
    output, other = (1, 2) if stream == "stdout" else (2, 1)
    script = (
        "import os, time\n"
        # The decoder first returns "a" and buffers the first byte of €.
        f"os.write({output}, bytes([0x61, 0xE2]))\n"
        "time.sleep(0.05)\n"
        # The other decoder also buffers the first byte of € before the shared cap is exceeded.
        f"os.write({other}, bytes([0xE2]))\n"
        "time.sleep(0.05)\n"
        f"os.write({output}, bytes([0x82, 0xAC]))\n"
        "while True:\n"
        "    time.sleep(1)\n"
    )
    result = await exec_command(sys.executable, ["-c", script], os.getcwd(), max_buffer=3)

    assert result.stdout == ("a" if stream == "stdout" else "")
    assert result.stderr == ("a" if stream == "stderr" else "")
    assert (result.truncated, result.killed) == (True, True)
    assert result.code != 0
    assert len(result.stdout.encode()) + len(result.stderr.encode()) <= 3
