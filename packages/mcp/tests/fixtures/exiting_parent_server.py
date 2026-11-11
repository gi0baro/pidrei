"""Mirror of pi mcp test/fixtures/exiting-parent-server.mjs: starts a helper
that inherits stdout and stderr, then exits while the helper keeps running."""

import os
import subprocess
import sys


helper = subprocess.Popen(
    [sys.executable, "-c", "import time\nwhile True:\n    time.sleep(1000)"],
    stdin=subprocess.DEVNULL,
)
sys.stderr.write(f"helper {helper.pid}\n")
sys.stderr.flush()
# `process.exit(0)`, without the interpreter teardown that warns about the still-running helper.
os._exit(0)
