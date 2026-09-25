"""Mirror of pi coding-agent src/utils/wsl.ts."""

import os
import re


def is_wsl(env=None) -> bool:
    """Windows Subsystem for Linux, where Windows executables are reachable through interop.

    Blocking (reads `/proc/version`): async callers offload it.
    """
    env = env if env is not None else os.environ
    if env.get("WSL_DISTRO_NAME") or env.get("WSLENV"):
        return True

    try:
        with open("/proc/version", encoding="utf-8") as f:
            release = f.read()
        return re.search(r"microsoft|wsl", release, re.IGNORECASE) is not None
    except OSError:
        return False
