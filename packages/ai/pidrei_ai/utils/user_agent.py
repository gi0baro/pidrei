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


def set_default_user_agent(headers: dict[str, Any]) -> None:
    """Add this program's User-Agent unless the model or caller already set one.

    pi spreads `{ "User-Agent": getPiUserAgent() }` in front of the other header
    sources and lets the provider SDK fold the names case-insensitively. A Python
    dict does not fold, so an override under any casing would otherwise reach the
    wire alongside the default; the check happens here instead. A `None` value is
    the caller's delete sentinel and counts as an override, as it does upstream.
    """
    if any(name.lower() == "user-agent" for name in headers):
        return
    headers["User-Agent"] = get_user_agent()
