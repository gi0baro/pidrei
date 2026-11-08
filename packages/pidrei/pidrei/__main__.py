"""Mirror of pi coding-agent src/cli.ts.

CLI entry point for the coding agent. Runs main() under the tonio runtime.
"""

import os
import signal
import sys

import tonio.colored as tonio

from .core.output_guard import install_stdio_streams, start_output_writer, stop_output_writer
from .main import main
from .utils.fd_io import snapshot_stdio_flags
from .utils.runtime_options import runtime_options
from .utils.shell import SignalDispatcher, exit_on_signal, stop_on_signal


async def _run_main(args: list[str]) -> int:
    # Every stdio write is queued for the run, and delivered before it ends:
    # what is written afterwards (a traceback on the way out) goes straight
    # to the fd.
    start_output_writer()
    # The runtime is created with the handled signals (`runtime_options`), so
    # their receiver runs for the whole process. Until a mode installs its
    # own, SIGTERM/SIGHUP end the process, SIGTSTP stops it, and the others
    # do nothing.
    signal_dispatcher = SignalDispatcher()
    signal_dispatcher.set_handler(signal.SIGTERM, exit_on_signal)
    signal_dispatcher.set_handler(signal.SIGHUP, exit_on_signal)
    signal_dispatcher.set_handler(signal.SIGTSTP, stop_on_signal)
    tonio.spawn.without_tracking(signal_dispatcher.watch())
    try:
        return await main(args, signal_dispatcher=signal_dispatcher)
    finally:
        await stop_output_writer()


def run() -> None:
    os.environ["PIDREI_CODING_AGENT"] = "true"
    # Cross-tool convention (pi #7493): the NAME stays as upstream publishes it
    # so third-party tooling detects an agent session; only the value renames.
    os.environ["AI_AGENT"] = "pidrei"
    # Before any fd registration flips O_NONBLOCK on the shell's descriptors:
    # the atexit restore this registers is half of the stdio teardown policy
    # (`hard_exit` is the other half).
    snapshot_stdio_flags()
    install_stdio_streams()
    sys.exit(tonio.run(_run_main(sys.argv[1:]), **runtime_options()))


if __name__ == "__main__":
    run()
