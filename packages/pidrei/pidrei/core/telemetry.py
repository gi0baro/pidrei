"""Mirror of pi coding-agent src/core/telemetry.ts (PI_TELEMETRY -> PIDREI_TELEMETRY)."""

import os
from typing import Any


_UNSET = object()


def _is_truthy_env_flag(value: str | None) -> bool:
    if not value:
        return False
    return value == "1" or value.lower() in ("true", "yes")


def is_install_telemetry_enabled(settings_manager: Any, telemetry_env: Any = _UNSET) -> bool:
    if telemetry_env is _UNSET:
        telemetry_env = os.environ.get("PIDREI_TELEMETRY")
    if telemetry_env is not None:
        return _is_truthy_env_flag(telemetry_env)
    return settings_manager.get_enable_install_telemetry()
