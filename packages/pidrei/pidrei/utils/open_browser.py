"""Mirror of pi coding-agent src/utils/open-browser.ts (POSIX targets only).

This intentionally never invokes a shell.
"""

import subprocess
import sys

import tonio.colored as tonio
from tonio.exceptions import RuntimeNotInitializedError


def open_browser(target: str) -> None:
    """Open a URL or file in the platform browser/default handler.

    Browser launch is best-effort: callers still present the target to the
    user, so launcher failures (for example, missing xdg-open) never raise.
    """
    cmd = ["open", target] if sys.platform == "darwin" else ["xdg-open", target]

    def launch() -> None:
        try:
            subprocess.Popen(  # noqa: S603
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            pass

    # Reached from a sync OAuth callback that runs on a runtime worker, and the
    # fork/exec blocks it. Nothing awaits the launch and failures are already
    # swallowed, so hand it to the pool and return.
    try:
        tonio.spawn.without_tracking(tonio.spawn_blocking(launch))
    except RuntimeNotInitializedError:
        launch()  # no runtime, so no worker to protect
