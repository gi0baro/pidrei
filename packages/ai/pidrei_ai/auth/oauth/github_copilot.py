"""Port of pi's GitHub Copilot flow (packages/ai/src/auth/oauth/github-copilot.ts).

Device authorization against github.com (or a GitHub Enterprise domain), then an
exchange of the resulting GitHub token for a short-lived Copilot token. The
Copilot token carries the account's proxy endpoint, which is where `to_auth`
derives the per-credential base URL from.
"""

import base64
import re
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import urlsplit

import tonio.colored as tonio

from pidrei_ai.auth.oauth import http as oauth_http
from pidrei_ai.auth.oauth.device_code import OAuthDeviceCodePollResult, poll_oauth_device_code_flow
from pidrei_ai.auth.oauth.urls import http_or_https_url
from pidrei_ai.auth.types import (
    AuthEvent,
    AuthPrompt,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
    ProviderAuthInteraction,
)
from pidrei_ai.models_generated import MODELS
from pidrei_ai.utils.cancel import CancelToken


def _decode(value: str) -> str:
    return base64.b64decode(value).decode("utf-8")


CLIENT_ID = _decode("SXYxLmI1MDdhMDhjODdlY2ZlOTg=")

COPILOT_HEADERS = {
    "User-Agent": "GitHubCopilotChat/0.35.0",
    "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0",
    "Copilot-Integration-Id": "vscode-chat",
}
COPILOT_API_VERSION = "2026-06-01"
MODELS_FETCH_TIMEOUT_MS = 5000


@dataclass(slots=True)
class _DeviceCodeResponse:
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: float
    interval: float | None = None


@dataclass(slots=True)
class _CopilotUrls:
    device_code_url: str
    access_token_url: str
    copilot_token_url: str


def _normalize_domain(value: str) -> str | None:
    trimmed = value.strip()
    if not trimmed:
        return None
    split = urlsplit(trimmed if "://" in trimmed else f"https://{trimmed}")
    try:
        hostname = split.hostname
    except ValueError:
        return None
    return hostname or None


def _get_urls(domain: str) -> _CopilotUrls:
    return _CopilotUrls(
        device_code_url=f"https://{domain}/login/device/code",
        access_token_url=f"https://{domain}/login/oauth/access_token",
        copilot_token_url=f"https://api.{domain}/copilot_internal/v2/token",
    )


def _get_base_url_from_token(token: str) -> str | None:
    """Parse the proxy-ep out of a Copilot token and convert it to an API base URL.

    Token format: tid=...;exp=...;proxy-ep=proxy.individual.githubcopilot.com;...
    """
    match = re.search(r"proxy-ep=([^;]+)", token)
    if not match:
        return None
    # Convert proxy.xxx to api.xxx
    api_host = re.sub(r"^proxy\.", "api.", match.group(1))
    return f"https://{api_host}"


def _get_base_url(token: str | None = None, enterprise_domain: str | None = None) -> str:
    # If we have a token, extract the base URL from proxy-ep
    if token:
        url_from_token = _get_base_url_from_token(token)
        if url_from_token:
            return url_from_token
    # Fallback for enterprise or if token parsing fails
    if enterprise_domain:
        return f"https://copilot-api.{enterprise_domain}"
    return "https://api.individual.githubcopilot.com"


def _as_record(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _parse_available_copilot_model_ids(raw: Any, allow_policy_fallback: bool) -> list[str]:
    data = (_as_record(raw) or {}).get("data")
    if not isinstance(data, list):
        raise RuntimeError("Invalid Copilot models response")  # noqa: TRY004 - pi throws a plain Error

    picker_ids: list[str] = []
    policy_enabled_ids: list[str] = []
    for raw_item in data:
        item = _as_record(raw_item)
        model_id = item.get("id") if item else None
        if item is None or not isinstance(model_id, str):
            continue

        capabilities = _as_record(item.get("capabilities"))
        supports = _as_record(capabilities.get("supports")) if capabilities else None
        if (supports or {}).get("tool_calls") is False:
            continue
        policy = _as_record(item.get("policy"))
        if item.get("model_picker_enabled") is True and (policy or {}).get("state") != "disabled":
            picker_ids.append(model_id)
        if (policy or {}).get("state") == "enabled":
            policy_enabled_ids.append(model_id)
    return picker_ids if picker_ids or not allow_policy_fallback else policy_enabled_ids


async def _fetch_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str],
    form: dict[str, str] | None = None,
    timeout_ms: float | None = None,
    cancel: CancelToken | None = None,
) -> Any:
    response = await oauth_http.request(
        url, method=method, headers=headers, form=form, timeout_ms=timeout_ms, cancel=cancel
    )
    if not response.ok:
        try:
            status_text = HTTPStatus(response.status).phrase
        except ValueError:
            status_text = ""
        raise RuntimeError(f"{response.status} {status_text}: {response.text}")
    return response.json()


async def _fetch_available_model_ids(
    copilot_token: str, enterprise_domain: str | None, cancel: CancelToken
) -> list[str]:
    base_url = _get_base_url(copilot_token, enterprise_domain)
    # Some Individual accounts return false for every picker flag despite explicit enabled policies.
    # Limit the fallback to that endpoint so other account types keep strict picker semantics.
    allow_policy_fallback = base_url == "https://api.individual.githubcopilot.com"
    raw = await _fetch_json(
        f"{base_url}/models",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {copilot_token}",
            **COPILOT_HEADERS,
            "X-GitHub-Api-Version": COPILOT_API_VERSION,
        },
        timeout_ms=MODELS_FETCH_TIMEOUT_MS,
        cancel=cancel,
    )
    return _parse_available_copilot_model_ids(raw, allow_policy_fallback)


async def _start_device_flow(domain: str, cancel: CancelToken) -> _DeviceCodeResponse:
    urls = _get_urls(domain)
    data = await _fetch_json(
        urls.device_code_url,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "GitHubCopilotChat/0.35.0",
        },
        form={"client_id": CLIENT_ID, "scope": "read:user"},
        cancel=cancel,
    )

    if not isinstance(data, dict):
        raise RuntimeError("Invalid device code response")  # noqa: TRY004 - pi throws a plain Error

    device_code = data.get("device_code")
    user_code = data.get("user_code")
    verification_uri = data.get("verification_uri")
    interval = data.get("interval")
    expires_in = data.get("expires_in")

    if (
        not isinstance(device_code, str)
        or not isinstance(user_code, str)
        or not isinstance(verification_uri, str)
        or (interval is not None and (isinstance(interval, bool) or not isinstance(interval, int | float)))
        or isinstance(expires_in, bool)
        or not isinstance(expires_in, int | float)
    ):
        raise RuntimeError("Invalid device code response fields")

    # The verification URI is opened in the user's browser and to prevent `open` from
    # opening an executable or similar, we force it to be a URL.
    parsed_uri = http_or_https_url(verification_uri)
    if parsed_uri is None:
        raise RuntimeError("Untrusted verification_uri in device code response")

    return _DeviceCodeResponse(
        device_code=device_code,
        user_code=user_code,
        verification_uri=parsed_uri,
        interval=interval,
        expires_in=expires_in,
    )


async def _poll_for_github_access_token(domain: str, device: _DeviceCodeResponse, cancel: CancelToken) -> str:
    urls = _get_urls(domain)

    async def poll() -> OAuthDeviceCodePollResult:
        raw = await _fetch_json(
            urls.access_token_url,
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "GitHubCopilotChat/0.35.0",
            },
            form={
                "client_id": CLIENT_ID,
                "device_code": device.device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            cancel=cancel,
        )

        if isinstance(raw, dict) and isinstance(raw.get("access_token"), str):
            return OAuthDeviceCodePollResult(status="complete", value=raw["access_token"])

        if isinstance(raw, dict) and isinstance(raw.get("error"), str):
            error = raw["error"]
            description = raw.get("error_description")
            interval = raw.get("interval")
            if error == "authorization_pending":
                return OAuthDeviceCodePollResult(status="pending")

            if error == "slow_down":
                return OAuthDeviceCodePollResult(
                    status="slow_down",
                    interval_seconds=(
                        interval if isinstance(interval, int | float) and not isinstance(interval, bool) else None
                    ),
                )

            description_suffix = f": {description}" if description else ""
            return OAuthDeviceCodePollResult(
                status="failed", message=f"Device flow failed: {error}{description_suffix}"
            )

        return OAuthDeviceCodePollResult(status="failed", message="Invalid device token response")

    return await poll_oauth_device_code_flow(
        poll=poll,
        interval_seconds=device.interval,
        expires_in_seconds=device.expires_in,
        wait_before_first_poll=True,
        cancel=cancel,
    )


async def _refresh_access_token(
    refresh_token: str, enterprise_domain: str | None, cancel: CancelToken
) -> OAuthCredential:
    domain = enterprise_domain or "github.com"
    urls = _get_urls(domain)

    raw = await _fetch_json(
        urls.copilot_token_url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {refresh_token}",
            **COPILOT_HEADERS,
        },
        cancel=cancel,
    )

    if not isinstance(raw, dict):
        raise RuntimeError("Invalid Copilot token response")  # noqa: TRY004 - pi throws a plain Error

    token = raw.get("token")
    expires_at = raw.get("expires_at")

    if not isinstance(token, str) or isinstance(expires_at, bool) or not isinstance(expires_at, int | float):
        raise RuntimeError("Invalid Copilot token response fields")  # noqa: TRY004 - pi throws a plain Error

    return OAuthCredential(
        refresh=refresh_token,
        access=token,
        expires=int(expires_at * 1000 - 5 * 60 * 1000),
        extra={"enterpriseUrl": enterprise_domain} if enterprise_domain else {},
    )


async def _refresh_copilot_token(
    refresh_token: str, enterprise_domain: str | None, cancel: CancelToken
) -> OAuthCredential:
    credential = await _refresh_access_token(refresh_token, enterprise_domain, cancel)
    credential.extra["availableModelIds"] = await _fetch_available_model_ids(
        credential.access, enterprise_domain, cancel
    )
    return credential


async def _enable_model(token: str, model_id: str, enterprise_domain: str | None, cancel: CancelToken) -> bool:
    """Enable a model for the user's GitHub Copilot account.

    This is required for some models (like Claude, Grok) before they can be used.
    """
    base_url = _get_base_url(token, enterprise_domain)
    url = f"{base_url}/models/{model_id}/policy"

    try:
        response = await oauth_http.request(
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
                **COPILOT_HEADERS,
                "openai-intent": "chat-policy",
                "x-interaction-type": "chat-policy",
            },
            json_body={"state": "enabled"},
            cancel=cancel,
        )
        return response.ok
    except Exception:
        if cancel.cancelled:
            raise
        return False


async def _enable_all_models(token: str, enterprise_domain: str | None, cancel: CancelToken) -> None:
    """Enable every known Copilot model that may require policy acceptance, so the
    catalog is usable right after login."""
    models = list(MODELS.get("github-copilot", []))
    if not models:  # pragma: no cover - the vendored catalog always has models
        return
    await tonio.spawn(*[_enable_model(token, model.id, enterprise_domain, cancel) for model in models])


async def _login_github_copilot(interaction: ProviderAuthInteraction) -> OAuthCredential:
    value = await interaction.prompt(
        AuthPrompt(
            type="text",
            message="GitHub Enterprise URL/domain (blank for github.com)",
            placeholder="company.ghe.com",
        )
    )
    if interaction.cancel.cancelled:
        raise RuntimeError("Login cancelled")

    trimmed = value.strip()
    enterprise_domain = _normalize_domain(value)
    if trimmed and not enterprise_domain:
        raise RuntimeError("Invalid GitHub Enterprise URL/domain")
    domain = enterprise_domain or "github.com"

    device = await _start_device_flow(domain, interaction.cancel)
    interaction.notify(
        AuthEvent(
            type="device_code",
            user_code=device.user_code,
            verification_uri=device.verification_uri,
            interval_seconds=device.interval,
            expires_in_seconds=device.expires_in,
        )
    )

    github_access_token = await _poll_for_github_access_token(domain, device, interaction.cancel)
    credential = await _refresh_access_token(github_access_token, enterprise_domain, interaction.cancel)
    interaction.notify(AuthEvent(type="progress", message="Enabling models..."))
    await _enable_all_models(credential.access, enterprise_domain, interaction.cancel)
    credential.extra["availableModelIds"] = await _fetch_available_model_ids(
        credential.access, enterprise_domain, interaction.cancel
    )
    return credential


def _enterprise_domain(credential: OAuthCredential) -> str | None:
    enterprise_url = credential.extra.get("enterpriseUrl")
    if not isinstance(enterprise_url, str) or not enterprise_url:
        return None
    return _normalize_domain(enterprise_url)


async def _refresh(credential: OAuthCredential, cancel: CancelToken) -> OAuthCredential:
    return await _refresh_copilot_token(credential.refresh, _enterprise_domain(credential), cancel)


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    """Derive the credential-specific proxy endpoint for each request."""
    return ModelAuth(
        api_key=credential.access,
        base_url=_get_base_url(credential.access, _enterprise_domain(credential)),
    )


github_copilot_oauth = OAuthAuth(
    name="GitHub Copilot",
    is_subscription=True,
    login=_login_github_copilot,
    refresh=_refresh,
    to_auth=_to_auth,
)
