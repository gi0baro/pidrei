"""Mirror of pi coding-agent src/core/provider-attribution.ts.

Attribution header values keep pi's provider-facing identifiers ("pi",
"https://pi.dev", ...) where they attribute the upstream harness family;
they are model-visible nowhere and provider-visible everywhere, so the
pidrei fork keeps its own name only in the User-Agent-style value.
"""

from typing import Any
from urllib.parse import urlparse

from .telemetry import is_install_telemetry_enabled


_OPENROUTER_HOST = "openrouter.ai"
_NVIDIA_NIM_HOST = "integrate.api.nvidia.com"
_CLOUDFLARE_API_HOST = "api.cloudflare.com"
_CLOUDFLARE_AI_GATEWAY_HOST = "gateway.ai.cloudflare.com"
_OPENCODE_HOST = "opencode.ai"


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
    if not is_install_telemetry_enabled(settings_manager):
        return None

    if _is_openrouter_model(model):
        return {
            "HTTP-Referer": "https://pi.dev",
            "X-OpenRouter-Title": "pi",
            "X-OpenRouter-Categories": "cli-agent",
        }

    if _is_nvidia_nim_model(model):
        return {"X-BILLING-INVOKE-ORIGIN": "Pi"}

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
    return {"x-opencode-session": session_id, "x-opencode-client": "pi"}


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
