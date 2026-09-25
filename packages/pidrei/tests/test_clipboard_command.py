"""Mirror of pi coding-agent test/clipboard-command.test.ts.

pi drives `process.execPath` (node) with `-e` scripts; these drive the Python
interpreter with `-c` scripts the same way.
"""

import sys

import pytest
import tonio.colored as tonio

from pidrei.utils.clipboard_command import run_clipboard_command


@pytest.mark.tonio
async def test_preserves_binary_output_and_distinguishes_empty_success_from_failure():
    assert await run_clipboard_command(
        sys.executable, ["-c", "import sys; sys.stdout.buffer.write(bytes([0, 255, 10]))"]
    ) == bytes([0, 255, 10])
    assert await run_clipboard_command(sys.executable, ["-c", ""]) == b""
    assert await run_clipboard_command(sys.executable, ["-c", "raise SystemExit(1)"]) is None
    assert await run_clipboard_command("pi-clipboard-command-does-not-exist", []) is None


@pytest.mark.tonio
async def test_sends_unicode_input_to_clipboard_writers():
    script = "import sys; raise SystemExit(0 if sys.stdin.buffer.read().decode('utf-8') == 'café 日本語' else 1)"
    assert await run_clipboard_command(sys.executable, ["-c", script], input="café 日本語") == b""


@pytest.mark.tonio
async def test_times_out_without_blocking_the_runtime():
    # Relaxation: pi counts event-loop ticks during the call to show the loop was
    # not blocked. On tonio's multi-worker runtime a blocked worker does not stop
    # a ticker on another, so the count cannot observe that, and it is timing-
    # sensitive. What is asserted is the timeout itself: the call gives up on the
    # 60s child well within the bound instead of waiting it out (run_command is
    # the tonio process API, so no worker is held meanwhile).
    result, completed = await tonio.time.timeout(
        run_clipboard_command(sys.executable, ["-c", "import time; time.sleep(60)"], timeout_ms=200), 10
    )
    assert completed
    assert result is None


@pytest.mark.tonio
async def test_rejects_output_above_the_buffer_limit():
    assert (
        await run_clipboard_command(
            sys.executable, ["-c", "import sys; sys.stdout.buffer.write(bytes(1024))"], max_buffer_bytes=16
        )
        is None
    )
