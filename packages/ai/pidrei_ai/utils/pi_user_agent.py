"""Runtime user-agent string (port of pi `utils/pi-user-agent.ts`).

pi: `pi (${os.platform()} ${os.release()}; ${os.arch()})`, "pi (browser)" when
node:os is unavailable (a branch with no Python counterpart). Node reports
"x64"/"arm64" where `platform.machine()` reports "x86_64"/"aarch64"; the raw
Python value is sent.

Deliberately still "pi", unlike the Phase 7 step 1 attribution swap: the Codex
adapter pairs this with its `originator: pi` header, and provider backends may
gate on known agent values. Changing it needs live accounts to verify — see
PLAN, step 1 follow-up.
"""

import platform


def get_pi_user_agent() -> str:
    return f"pi ({platform.system().lower()} {platform.release()}; {platform.machine()})"
