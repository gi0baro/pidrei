"""Mirror of pi coding-agent src/utils/open-browser.ts (POSIX targets only).

This intentionally never invokes a shell.
"""

import subprocess
import sys


def open_browser(target: str) -> None:
    """Open a URL or file in the platform browser/default handler.

    Browser launch is best-effort: callers still present the target to the
    user, so launcher failures (for example, missing xdg-open) never raise.
    """
    cmd = ["open", target] if sys.platform == "darwin" else ["xdg-open", target]

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
