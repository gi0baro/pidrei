"""Shared subprocess helpers for the CLI-spawn suites.

pi spawns `node src/cli.ts` with PI_CODING_AGENT_DIR/PI_OFFLINE; the
mirrors spawn `python -m pidrei` with the pidrei equivalents.
"""

import json
import os
import select
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Self

from pidrei.config import ENV_AGENT_DIR


@dataclass(slots=True)
class CliResult:
    code: int | None
    stdout: str
    stderr: str


def _cli_env(agent_dir: str, env: dict[str, str] | None) -> dict[str, str]:
    return {
        **os.environ,
        ENV_AGENT_DIR: agent_dir,
        "PIDREI_OFFLINE": "1",
        **(env or {}),
    }


def run_cli(
    args: list[str],
    *,
    cwd: str,
    agent_dir: str,
    env: dict[str, str] | None = None,
    stdin: str | None = None,
    timeout: float = 60.0,
) -> CliResult:
    process_env = _cli_env(agent_dir, env)
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pidrei", *args],
        cwd=cwd,
        env=process_env,
        input=stdin if stdin is not None else "",
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return CliResult(code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


class RpcCli:
    """`pidrei --mode rpc` as a child that stays open between commands.

    `run_cli` writes stdin whole and closes it, and end of input starts the shutdown in parallel with
    the commands. Here `request()` writes one command and returns once its response is on stdout;
    `close()` then ends stdin and collects the rest. Every wait is bounded by `timeout`.
    """

    def __init__(
        self,
        args: list[str],
        *,
        cwd: str,
        agent_dir: str,
        env: dict[str, str] | None = None,
        timeout: float = 20.0,
    ):
        self._timeout = timeout
        self._stdout = bytearray()
        self._stderr = bytearray()
        self._process = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", "pidrei", "--mode", "rpc", *args],
            cwd=cwd,
            env=_cli_env(agent_dir, env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._open_fds: set[int] = {self._process.stdout.fileno(), self._process.stderr.fileno()}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc) -> None:
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait(self._timeout)
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            if not stream.closed:
                stream.close()

    @property
    def stdout(self) -> str:
        return self._stdout.decode("utf-8", "replace")

    @property
    def stderr(self) -> str:
        return self._stderr.decode("utf-8", "replace")

    def _read_until(self, done, deadline: float) -> bool:
        """Read stdout and stderr until `done()` holds or both reach end of file."""
        stdout_fd = self._process.stdout.fileno()
        while not done():
            if not self._open_fds:
                return False
            remaining = deadline - time.monotonic()
            assert remaining > 0, f"timed out waiting for the RPC child. stderr:\n{self.stderr}"
            readable, _, _ = select.select(list(self._open_fds), [], [], remaining)
            for fd in readable:
                chunk = os.read(fd, 65536)
                if not chunk:
                    self._open_fds.discard(fd)
                elif fd == stdout_fd:
                    self._stdout += chunk
                else:
                    self._stderr += chunk
        return True

    def _response(self, request_id: str) -> dict[str, Any] | None:
        # Complete lines only: the part after the last newline may still be arriving.
        for line in self.stdout.split("\n")[:-1]:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "response" and record.get("id") == request_id:
                return record
        return None

    def request(self, command: dict[str, Any]) -> dict[str, Any]:
        """Write one command and wait for its response."""
        self._process.stdin.write((json.dumps(command) + "\n").encode())
        self._process.stdin.flush()
        deadline = time.monotonic() + self._timeout
        responded = self._read_until(lambda: self._response(command["id"]) is not None, deadline)
        assert responded, f"RPC process exited early. stderr:\n{self.stderr}"
        return self._response(command["id"])

    def close(self) -> CliResult:
        """End stdin, read the rest of the output and wait for the exit."""
        self._process.stdin.close()
        self._read_until(lambda: False, time.monotonic() + self._timeout)
        code = self._process.wait(self._timeout)
        return CliResult(code=code, stdout=self.stdout, stderr=self.stderr)
