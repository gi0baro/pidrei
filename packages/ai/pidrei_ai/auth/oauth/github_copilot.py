"""Port of pi's GitHub Copilot flow (packages/ai/src/auth/oauth/github-copilot.ts).

Device authorization against github.com (or a GitHub Enterprise domain), then an
exchange of the resulting GitHub token for a short-lived Copilot token. The
Copilot token carries the account's proxy endpoint, which is where `to_auth`
derives the per-credential base URL from.
"""

import base64
import math
import re
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from http import HTTPStatus
from typing import Any
from urllib.parse import urlsplit

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
from pidrei_ai.utils import clock
from pidrei_ai.utils.cancel import CancelToken
from pidrei_ai.utils.sleep import sleep


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

# Known Copilot model ids; only these get a policy update at login.
_KNOWN_MODEL_IDS = frozenset(model.id for model in MODELS.get("github-copilot", []))


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


@dataclass(slots=True)
class _AccountModel:
    id: str
    picker_enabled: bool
    policy_state: Any


@dataclass(slots=True)
class _CopilotModelCatalog:
    #: Models the account may already use.
    available_model_ids: list[str]
    #: Known, tool-capable models still awaiting a policy decision.
    policy_model_ids: list[str]


def _parse_github_copilot_model_catalog(raw: Any, allow_policy_fallback: bool) -> _CopilotModelCatalog:
    data = (_as_record(raw) or {}).get("data")
    if not isinstance(data, list):
        raise RuntimeError("Invalid Copilot models response")  # noqa: TRY004 - pi throws a plain Error

    account_models: list[_AccountModel] = []
    for raw_item in data:
        item = _as_record(raw_item)
        model_id = item.get("id") if item else None
        if item is None or not isinstance(model_id, str):
            continue

        capabilities = _as_record(item.get("capabilities"))
        supports = _as_record(capabilities.get("supports")) if capabilities else None
        if (supports or {}).get("tool_calls") is False:
            continue
        account_models.append(
            _AccountModel(
                id=model_id,
                picker_enabled=item.get("model_picker_enabled") is True,
                policy_state=(_as_record(item.get("policy")) or {}).get("state"),
            )
        )

    picker_model_ids = [
        model.id for model in account_models if model.picker_enabled and model.policy_state != "disabled"
    ]
    use_policy_fallback = allow_policy_fallback and not picker_model_ids
    available_model_ids = (
        picker_model_ids
        if picker_model_ids or not allow_policy_fallback
        else [model.id for model in account_models if model.policy_state == "enabled"]
    )
    policy_model_ids = [
        model.id
        for model in account_models
        if model.policy_state == "unconfigured"
        and model.id in _KNOWN_MODEL_IDS
        and (model.picker_enabled or use_policy_fallback)
    ]
    return _CopilotModelCatalog(available_model_ids=available_model_ids, policy_model_ids=policy_model_ids)


def _status_text(status: int) -> str:
    """`Response.statusText` for a known status; unknown ones render empty."""
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return ""


def _retry_after_delay_ms(header_value: str) -> float | None:
    """The `Retry-After` delay in milliseconds, or None when it is unusable.

    Mirrors pi: `Number.parseFloat` first, then the HTTP-date form through
    `Date.parse`; a non-finite result makes the caller give up on retrying.
    """
    try:
        return float(header_value) * 1000
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(header_value)
    except TypeError, ValueError:
        return None
    return parsed.timestamp() * 1000 - clock.now_ms()


async def _fetch_with_rate_limit_retry(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str],
    json_body: Any = None,
    cancel: CancelToken,
    max_retries: int,
    max_elapsed_ms: float,
) -> oauth_http.OAuthHttpResponse:
    """One request, retried while the server answers 429 and the budget allows.

    pi combines the caller's signal with an `AbortSignal.timeout(maxElapsedMs)`
    so a request in flight is aborted once the budget is spent. Here the budget
    is only a deadline check between attempts; every attempt still carries the
    same per-request timeout and the caller's cancel token.
    """
    retry_deadline = clock.now_ms() + max_elapsed_ms if max_retries > 0 and max_elapsed_ms > 0 else None
    retry = 0
    while True:
        response = await oauth_http.request(
            url,
            method=method,
            headers=headers,
            json_body=json_body,
            timeout_ms=MODELS_FETCH_TIMEOUT_MS,
            cancel=cancel,
        )
        if response.status != HTTPStatus.TOO_MANY_REQUESTS or retry == max_retries:
            return response

        retry_after = response.headers.get("retry-after")
        delay_ms: float | None = 500 * 2**retry
        if retry_after:
            delay_ms = _retry_after_delay_ms(retry_after)
            if delay_ms is None or not math.isfinite(delay_ms):
                return response
        delay_ms = max(0.0, delay_ms)
        if retry_deadline is not None and delay_ms >= retry_deadline - clock.now_ms():
            return response
        await sleep(delay_ms, cancel)
        retry += 1


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
        raise RuntimeError(f"{response.status} {_status_text(response.status)}: {response.text}")
    return response.json()


async def _fetch_github_copilot_models(
    copilot_token: str,
    enterprise_domain: str | None,
    cancel: CancelToken,
    max_retries: int,
    max_elapsed_ms: float,
) -> _CopilotModelCatalog:
    base_url = _get_base_url(copilot_token, enterprise_domain)
    # Some Individual accounts return false for every picker flag despite explicit enabled policies.
    # Limit the fallback to that endpoint so other account types keep strict picker semantics.
    allow_policy_fallback = base_url == "https://api.individual.githubcopilot.com"

    response = await _fetch_with_rate_limit_retry(
        f"{base_url}/models",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {copilot_token}",
            **COPILOT_HEADERS,
            "X-GitHub-Api-Version": COPILOT_API_VERSION,
        },
        cancel=cancel,
        max_retries=max_retries,
        max_elapsed_ms=max_elapsed_ms,
    )
    if not response.ok:
        raise RuntimeError(f"{response.status} {_status_text(response.status)}: {response.text}")
    return _parse_github_copilot_model_catalog(response.json(), allow_policy_fallback)


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
    models = await _fetch_github_copilot_models(
        credential.access, enterprise_domain, cancel, max_retries=0, max_elapsed_ms=0
    )
    credential.extra["availableModelIds"] = models.available_model_ids
    return credential


async def _enable_model(token: str, model_id: str, enterprise_domain: str | None, cancel: CancelToken) -> bool:
    """Enable a model for the user's GitHub Copilot account.

    This is required for some models (like Claude, Grok) before they can be used.
    """
    base_url = _get_base_url(token, enterprise_domain)
    url = f"{base_url}/models/{model_id}/policy"

    try:
        response = await _fetch_with_rate_limit_retry(
            url,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
                **COPILOT_HEADERS,
                "openai-intent": "chat-policy",
                "x-interaction-type": "chat-policy",
            },
            json_body={"state": "enabled"},
            cancel=cancel,
            max_retries=2,
            max_elapsed_ms=5000,
        )
    except Exception:
        if cancel.cancelled:
            raise
        return False
    if response.status == HTTPStatus.TOO_MANY_REQUESTS:
        raise RuntimeError(f"{response.status} {_status_text(response.status)}: {response.text}")
    return response.ok


async def _enable_models(
    token: str, model_ids: list[str], enterprise_domain: str | None, cancel: CancelToken
) -> list[str]:
    """Enable the requested Copilot models and return the successful ids.
    Policy updates are best effort; exhausted rate limiting stops the batch."""
    enabled_model_ids: list[str] = []
    for model_id in model_ids:
        try:
            if await _enable_model(token, model_id, enterprise_domain, cancel):
                enabled_model_ids.append(model_id)
        except Exception:
            if cancel.cancelled:
                raise
            break
    return enabled_model_ids


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
    models = await _fetch_github_copilot_models(
        credential.access, enterprise_domain, interaction.cancel, max_retries=2, max_elapsed_ms=5000
    )
    enabled_model_ids: list[str] = []
    if models.policy_model_ids:
        interaction.notify(AuthEvent(type="progress", message="Enabling models..."))
        enabled_model_ids = await _enable_models(
            credential.access, models.policy_model_ids, enterprise_domain, interaction.cancel
        )
    credential.extra["availableModelIds"] = list(dict.fromkeys([*models.available_model_ids, *enabled_model_ids]))
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
