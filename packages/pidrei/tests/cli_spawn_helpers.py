"""Shared subprocess helpers for the CLI-spawn suites.

pi spawns `node src/cli.ts` with PI_CODING_AGENT_DIR/PI_OFFLINE; the
mirrors spawn `python -m pidrei` with the pidrei equivalents.
"""

import os
import subprocess
import sys
from dataclasses import dataclass

from pidrei.config import ENV_AGENT_DIR


@dataclass(slots=True)
class CliResult:
    code: int | None
    stdout: str
    stderr: str


def run_cli(
    args: list[str],
    *,
    cwd: str,
    agent_dir: str,
    env: dict[str, str] | None = None,
    stdin: str | None = None,
    timeout: float = 60.0,
) -> CliResult:
    process_env = {
        **os.environ,
        ENV_AGENT_DIR: agent_dir,
        "PIDREI_OFFLINE": "1",
        **(env or {}),
    }
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
