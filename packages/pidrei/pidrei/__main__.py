"""Mirror of pi coding-agent src/cli.ts.

CLI entry point for the coding agent. Runs main() under the tonio runtime.
"""

import os
import sys

import tonio.colored as tonio

from .main import main
from .utils.fd_io import snapshot_std_blocking
from .utils.runtime_options import runtime_options


def run() -> None:
    os.environ["PIDREI_CODING_AGENT"] = "true"
    # Cross-tool convention (pi #7493): the NAME stays as upstream publishes it
    # so third-party tooling detects an agent session; only the value renames.
    os.environ["AI_AGENT"] = "pidrei"
    # Before any fd registration flips O_NONBLOCK on the shell's descriptors:
    # the atexit restore this registers is half of the stdio teardown policy
    # (`hard_exit` is the other half).
    snapshot_std_blocking()
    sys.exit(tonio.run(main(sys.argv[1:]), **runtime_options()))


if __name__ == "__main__":
    run()
