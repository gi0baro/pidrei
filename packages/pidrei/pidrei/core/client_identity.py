"""How pidrei's requests identify it to providers (port of pi `core/client-identity.ts`).

A detailed `User-Agent` with pidrei's version, and attribution to pidrei as a
CLI agent while provider attribution is enabled. pi gates attribution on its
install-telemetry setting; pidrei has no install ping, so the gate is its own
`enableProviderAttribution` setting, overridden for the session by
`PIDREI_PROVIDER_ATTRIBUTION`.
"""

import os

from pidrei_ai.types import ClientIdentity
from pidrei_ai.utils.identity import PIDREI_ATTRIBUTION

from ..config import VERSION
from .settings_manager import SettingsManager


#: Session-scoped override for the `enableProviderAttribution` setting.
ATTRIBUTION_ENV = "PIDREI_PROVIDER_ATTRIBUTION"


def _is_truthy_env_flag(value: str) -> bool:
    return value == "1" or value.lower() in ("true", "yes")


def is_attribution_enabled(settings_manager: SettingsManager) -> bool:
    attribution_env = os.environ.get(ATTRIBUTION_ENV)
    if attribution_env is not None:
        return _is_truthy_env_flag(attribution_env)
    return settings_manager.get_enable_provider_attribution()


def get_client_identity(settings_manager: SettingsManager) -> ClientIdentity:
    return {
        "user_agent": "detailed",
        "version": VERSION,
        "attribution": PIDREI_ATTRIBUTION if is_attribution_enabled(settings_manager) else None,
    }
