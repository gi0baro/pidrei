"""Provider-facing attribution headers (diverges from pi's provider-attribution.ts).

pi identifies itself to a handful of providers on requests the user is already
making: OpenRouter's public app leaderboard (`HTTP-Referer` + title), NVIDIA's
billing-origin tag, and opencode's client tag. These are not phone-home — they
ride along with an inference call — but the values are pi's, so a pidrei build
sending them credits someone else's project for our users' traffic.

Phase 7 step 1 (2026-07-26) swaps every value to pidrei's own identity rather
than removing the mechanism, since attribution itself is legitimate and useful.

The gate below was `core/telemetry.py`, which existed to carry pi's
install-ping opt-out. The ping is gone; the toggle survives as the switch for
these headers and now lives with its only consumer. Step 2 (2026-07-26)
renamed it to match: `PIDREI_TELEMETRY` → `PIDREI_PROVIDER_ATTRIBUTION`,
`enableInstallTelemetry` → `enableProviderAttribution`, the old settings key
being carried across by `SettingsManager._migrate_settings`.
"""

import os
from typing import Any
from urllib.parse import urlparse


#: Where pidrei points providers that show an app link (OpenRouter's leaderboard).
ATTRIBUTION_URL = "https://github.com/gi0baro/pidrei"
#: Short name sent as the app/client/billing-origin identifier.
ATTRIBUTION_NAME = "pidrei"
#: Session-scoped override for the `enableProviderAttribution` setting.
ATTRIBUTION_ENV = "PIDREI_PROVIDER_ATTRIBUTION"

_OPENROUTER_HOST = "openrouter.ai"
_NVIDIA_NIM_HOST = "integrate.api.nvidia.com"
_CLOUDFLARE_API_HOST = "api.cloudflare.com"
_CLOUDFLARE_AI_GATEWAY_HOST = "gateway.ai.cloudflare.com"
_OPENCODE_HOST = "opencode.ai"

_UNSET = object()


def _is_truthy_env_flag(value: str | None) -> bool:
    if not value:
        return False
    return value == "1" or value.lower() in ("true", "yes")


def is_attribution_enabled(settings_manager: Any, attribution_env: Any = _UNSET) -> bool:
    if attribution_env is _UNSET:
        attribution_env = os.environ.get(ATTRIBUTION_ENV)
    if attribution_env is not None:
        return _is_truthy_env_flag(attribution_env)
    return settings_manager.get_enable_provider_attribution()


def _matches_host(base_url: str, expected_host: str) -> bool:
    try:
        return urlparse(base_url).hostname == expected_host
    except Exception:
        return False


def _is_openrouter_model(model: Any) -> bool:
    return model.provider == "openrouter" or _OPENROUTER_HOST in model.base_url


def _is_nvidia_nim_model(model: Any) -> bool:
    return model.provider == "nvidia" or _matches_host(model.base_url, _NVIDIA_NIM_HOST)


def _is_cloudflare_model(model: Any) -> bool:
    return (
        model.provider == "cloudflare-workers-ai"
        or model.provider == "cloudflare-ai-gateway"
        or _matches_host(model.base_url, _CLOUDFLARE_API_HOST)
        or _matches_host(model.base_url, _CLOUDFLARE_AI_GATEWAY_HOST)
    )


def _get_default_attribution_headers(model: Any, settings_manager: Any) -> dict[str, str] | None:
    if not is_attribution_enabled(settings_manager):
        return None

    if _is_openrouter_model(model):
        return {
            "HTTP-Referer": ATTRIBUTION_URL,
            "X-OpenRouter-Title": ATTRIBUTION_NAME,
            "X-OpenRouter-Categories": "cli-agent",
        }

    if _is_nvidia_nim_model(model):
        return {"X-BILLING-INVOKE-ORIGIN": ATTRIBUTION_NAME}

    if _is_cloudflare_model(model):
        return {"User-Agent": "pidrei-coding-agent"}

    return None


def _get_session_headers(model: Any, session_id: str | None) -> dict[str, str] | None:
    if not session_id:
        return None
    if (
        model.provider != "opencode"
        and model.provider != "opencode-go"
        and not _matches_host(model.base_url, _OPENCODE_HOST)
    ):
        return None
    return {"x-opencode-session": session_id, "x-opencode-client": ATTRIBUTION_NAME}


def merge_provider_attribution_headers(
    model: Any,
    settings_manager: Any,
    session_id: str | None,
    *header_sources: dict[str, Any] | None,
) -> dict[str, Any] | None:
    merged: dict[str, Any] = {
        **(_get_session_headers(model, session_id) or {}),
        **(_get_default_attribution_headers(model, settings_manager) or {}),
    }

    for headers in header_sources:
        if headers:
            merged.update(headers)

    return merged if merged else None
