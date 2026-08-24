"""Runtime user-agent string (port of pi `utils/pi-user-agent.ts`).

pi: `pi (${os.platform()} ${os.release()}; ${os.arch()})`, "pi (browser)" when
node:os is unavailable (a branch with no Python counterpart). Node reports
"x64"/"arm64" where `platform.machine()` reports "x86_64"/"aarch64"; the raw
Python value is sent.

The identity is pidrei's, not pi's: the Codex `originator`, the Codex and
Kimi Coding `User-Agent` all name this program. Presenting as pi to a provider
would be misattribution, whatever a backend may or may not gate on.
"""

import platform
from typing import Any


CLIENT_NAME = "pidrei"

# The `originator` the Codex adapter sends on every request and in the OAuth
# authorize URL.
ORIGINATOR = CLIENT_NAME


def get_user_agent() -> str:
    return f"{CLIENT_NAME} ({platform.system().lower()} {platform.release()}; {platform.machine()})"


def force_user_agent(headers: dict[str, Any]) -> None:
    """Replace any caller-supplied User-Agent with this program's own.

    pi's `forcePiUserAgent`: providers that gate on the client identity (xAI)
    must see it, so a header set by the model or the caller is dropped first —
    header names are case-insensitive, the dict is not.
    """
    for name in [name for name in headers if name.lower() == "user-agent"]:
        del headers[name]
    headers["User-Agent"] = get_user_agent()
