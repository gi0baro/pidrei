"""Mirror of pi coding-agent src/utils/pi-user-agent.ts."""

import platform
import sys


def get_pidrei_user_agent(version: str) -> str:
    runtime = f"python/{platform.python_version()}"
    return f"pidrei/{version} ({sys.platform}; {runtime}; {platform.machine()})"
