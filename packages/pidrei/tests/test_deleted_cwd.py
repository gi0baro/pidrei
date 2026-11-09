"""Mirror of pi coding-agent test/deleted-cwd.test.ts.

pi stubs `process.cwd()` to throw; patching `os.getcwd` would change it for
the whole process, so this starts the CLI in a directory that a shell
deletes right before `exec`. Not mirrored: "rethrows unrelated cwd errors"
(only a deleted directory can be produced for real; any other error keeps
reaching `main()`'s generic report).
"""

import os
import subprocess
import sys

from pidrei.config import ENV_AGENT_DIR


# Regression test for https://github.com/earendil-works/pi/issues/11509
def test_prints_an_actionable_error_and_exits_1_instead_of_throwing(tmp_path):
    gone = tmp_path / "gone"
    gone.mkdir()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    completed = subprocess.run(  # noqa: S603
        [
            "/bin/sh",
            "-c",
            'cd "$1" && rmdir "$1" && exec "$2" -m pidrei --no-extensions --version',
            "sh",
            gone,
            sys.executable,
        ],
        cwd=tmp_path,
        env={**os.environ, ENV_AGENT_DIR: str(agent_dir), "PIDREI_OFFLINE": "1"},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert completed.stderr.strip() == (
        "Error: The current working directory no longer exists. Change to an existing directory and try again."
    )
