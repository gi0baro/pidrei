"""Mirror of pi mcp test/fixtures/stubborn-server.mjs: answers initialize,
spawns a grandchild that outlives stdin, and ignores stdin EOF and SIGTERM.

pi's grandchild ignores its stdio; this one keeps the server's stderr, so
the transport's close, which waits for stderr to reach EOF, can only return
once the grandchild is gone too (a dead process answers `kill(pid, 0)`
until it is reaped, so probing it says nothing).
"""

import json
import signal
import subprocess
import sys
import time


signal.signal(signal.SIGTERM, signal.SIG_IGN)
grandchild = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nwhile True:\n    time.sleep(1000)",
    ],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
)
sys.stderr.write(f"grandchild {grandchild.pid}\n")
sys.stderr.flush()

for line in sys.stdin:
    message = json.loads(line)
    if message.get("method") != "initialize":
        continue
    result = {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "serverInfo": {"name": "stubborn-fixture", "version": "1.0.0"},
    }
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}) + "\n")
    sys.stdout.flush()

while True:
    time.sleep(1000)
