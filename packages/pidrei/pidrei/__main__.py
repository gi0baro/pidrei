"""Mirror of pi coding-agent src/cli.ts.

CLI entry point for the coding agent. Runs main() under the tonio runtime.
"""

import os
import sys

import tonio.colored as tonio

from .main import main


def run() -> None:
    os.environ["PIDREI_CODING_AGENT"] = "true"
    sys.exit(tonio.run(main(sys.argv[1:])))


if __name__ == "__main__":
    run()
