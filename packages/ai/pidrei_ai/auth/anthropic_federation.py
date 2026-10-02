"""Anthropic workload identity federation — the slice of the Anthropic SDK pi uses.

pi hands `@anthropic-ai/sdk` a federation `config` and lets the SDK exchange the
identity token for a short-lived access token (the RFC 7523 jwt-bearer grant
against `/v1/oauth/token`) and cache it. There is no SDK here, so this module
ports that behaviour from the version pi pins (0.124.0: `lib/credentials/`
`oidc-federation`, `token-cache`, `identity-token`, `types`), over the
`auth/oauth/http.py` seam every token exchange uses.

The cache keeps the SDK's policy: more than 120 s left serves the cached token;
30–120 s serves it and refreshes in the background (backing off 5 s after a
failed background refresh); under 30 s or expired waits for a refresh.
Concurrent refreshes coalesce, and `invalidate()` (a 401 on a request that used
the token) makes the next caller start a fresh exchange instead of joining one
in flight.

Concurrency: every exchange runs as a detached coroutine and callers wait on its
Event, which is what pi's shared refresh promise is — a cancelled caller stops
waiting while the exchange completes for everyone else. Cache state lives
behind a thread lock that is never held across an await.

Deviations from the SDK: the exchange has a 30 s timeout (the TypeScript SDK
sets none; a shared exchange that hung would park every request waiting on it),
the 1 MiB response cap is not ported (the seam reads whole bodies), and the
cache is keyed on `(base_url, config)` only (pi also keys on the per-request
`fetch`, which has no counterpart here).
"""

import json
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import tonio.colored as tonio
from tonio.colored import fs

from pidrei_ai.auth.oauth import http as oauth_http
from pidrei_ai.utils.user_agent import get_user_agent
from pidrei_utils import clock


GRANT_TYPE_JWT_BEARER = "urn:ietf:params:oauth:grant-type:jwt-bearer"
TOKEN_ENDPOINT = "/v1/oauth/token"  # noqa: S105 - an endpoint, not a secret
# `anthropic-beta` value required on API requests authenticated with the access token.
OAUTH_API_BETA_HEADER = "oauth-2025-04-20"
# `anthropic-beta` value that routes the jwt-bearer exchange to the federation service.
FEDERATION_BETA_HEADER = "oidc-federation-2026-04-01"
ADVISORY_REFRESH_THRESHOLD_S = 120
MANDATORY_REFRESH_THRESHOLD_S = 30
ADVISORY_REFRESH_BACKOFF_S = 5
TOKEN_EXCHANGE_TIMEOUT_MS = 30_000
_MAX_ASSERTION_CHARS = 16 * 1024
_MAX_ERROR_BODY_CHARS = 2000
# RFC 6749 §5.2 error fields; anything else in a token endpoint error body may be echoed input.
_SAFE_ERROR_KEYS = ("error", "error_description", "error_uri")
_MISSING = object()


@dataclass(frozen=True, slots=True)
class AnthropicFederationConfig:
    """The federation settings pi reads from the `ANTHROPIC_*` variables (the SDK's `config`)."""

    federation_rule_id: str
    organization_id: str
    identity_token_file: str
    service_account_id: str | None = None
    workspace_id: str | None = None


class AnthropicWorkloadIdentityError(Exception):
    """The identity token could not be read or exchanged (the SDK's `WorkloadIdentityError`).

    It carries no `status`/`headers`, so provider retry never retries it, as in pi.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        body: Any = None,
        request_id: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.request_id = request_id

    def copy(self) -> AnthropicWorkloadIdentityError:
        return AnthropicWorkloadIdentityError(str(self), self.status_code, self.body, self.request_id)


@dataclass(frozen=True, slots=True)
class _AccessToken:
    token: str
    expires_at: float  # Unix seconds


def _now_s() -> int:
    """The SDK's `nowAsSeconds()`."""
    return clock.now_ms() // 1000


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _redact_sensitive(body: Any) -> Any:
    """A token endpoint body made safe to put in an exception: strings are truncated,
    objects keep only the RFC 6749 error fields."""
    if body is None:
        return None
    if isinstance(body, str):
        try:
            parsed = json.loads(body)
        except ValueError:
            if len(body) <= _MAX_ERROR_BODY_CHARS:
                return body
            return body[:_MAX_ERROR_BODY_CHARS] + f"... <{len(body) - _MAX_ERROR_BODY_CHARS} more chars>"
        return _json(_redact_sensitive(parsed))
    if isinstance(body, dict):
        return {key: value for key, value in body.items() if key in _SAFE_ERROR_KEYS}
    return None


def _js_number(value: Any) -> float:
    """The SDK's `Number(data.expires_in)` for the shapes a JSON body can carry."""
    if value is _MISSING:
        return math.nan  # Number(undefined)
    if value is None:
        return 0.0  # Number(null)
    if isinstance(value, bool | int | float):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0.0
        try:
            return float(text)
        except ValueError:
            return math.nan
    return math.nan


def _require_secure_token_endpoint(base_url: str) -> None:
    """Refuse to send the assertion over cleartext HTTP; loopback hosts are allowed."""
    if not base_url:
        return
    try:
        parts = urlsplit(base_url)
        host = (parts.hostname or "").lower()
    except ValueError as error:
        raise AnthropicWorkloadIdentityError(f'Invalid token endpoint base URL "{base_url}": {error}') from error
    if parts.scheme == "https":
        return
    if parts.scheme == "http" and host in ("localhost", "127.0.0.1", "::1"):
        return
    raise AnthropicWorkloadIdentityError(f'Refusing to send credential over non-https token endpoint "{base_url}"')


async def _read_identity_token(path: str) -> str:
    """Read the identity token on every exchange, so a rotated file (e.g. a Kubernetes
    projected service-account token) is picked up."""
    try:
        content = await fs.Path(path).read_text("utf-8")
    except (OSError, ValueError) as error:
        raise AnthropicWorkloadIdentityError(f"Failed to read identity token file at {path}: {error}") from error
    token = content.strip()
    if not token:
        raise AnthropicWorkloadIdentityError(f"Identity token file at {path} is empty")
    return token


def _parse_token_response(response: oauth_http.OAuthHttpResponse, request_id: str | None) -> dict[str, Any]:
    text = response.text
    try:
        data = json.loads(text)
    except ValueError as error:
        raise AnthropicWorkloadIdentityError(
            f"Token endpoint returned non-JSON response (status {response.status})",
            response.status,
            _redact_sensitive(text),
            request_id,
        ) from error
    if not isinstance(data, dict) or not data.get("access_token"):
        raise AnthropicWorkloadIdentityError(
            f"Token endpoint response missing access_token: {_json(_redact_sensitive(data))}",
            response.status,
            _redact_sensitive(data),
            request_id,
        )
    token_type = data.get("token_type")
    if token_type and (not isinstance(token_type, str) or token_type.lower() != "bearer"):
        raise AnthropicWorkloadIdentityError(
            f'Token endpoint response: unsupported token_type "{token_type}" (want Bearer)',
            response.status,
            _redact_sensitive(data),
            request_id,
        )
    return data


async def _exchange(base_url: str, config: AnthropicFederationConfig, env: Mapping[str, str] | None) -> _AccessToken:
    """One jwt-bearer exchange (the SDK's `oidcFederationProvider`)."""
    _require_secure_token_endpoint(base_url)
    jwt = await _read_identity_token(config.identity_token_file)
    # The token endpoint enforces a 16 KiB assertion limit; fail before the round-trip.
    if len(jwt) > _MAX_ASSERTION_CHARS:
        raise AnthropicWorkloadIdentityError(
            f"Identity token is {math.ceil(len(jwt) / 1024)} KiB, exceeds the 16 KiB assertion limit"
        )
    body = {
        "grant_type": GRANT_TYPE_JWT_BEARER,
        "assertion": jwt,
        "federation_rule_id": config.federation_rule_id,
        "organization_id": config.organization_id,
    }
    if config.service_account_id:
        body["service_account_id"] = config.service_account_id
    if config.workspace_id:
        body["workspace_id"] = config.workspace_id

    url = f"{base_url}{TOKEN_ENDPOINT}"
    try:
        response = await oauth_http.request(
            url,
            headers={
                "Content-Type": "application/json",
                "anthropic-beta": f"{OAUTH_API_BETA_HEADER},{FEDERATION_BETA_HEADER}",
                "User-Agent": get_user_agent(),
            },
            json_body=body,
            timeout_ms=TOKEN_EXCHANGE_TIMEOUT_MS,
            env=env,
        )
    except Exception as error:
        raise AnthropicWorkloadIdentityError(f"Failed to reach token endpoint {url}: {error}") from error

    request_id = response.headers.get("request-id")
    if not response.ok:
        redacted = _redact_sensitive(response.text)
        # A 401 is hard to debug from the status alone: point at the federation rule,
        # the workspace id (the usual fix when none is set) and Console's event log.
        hint = ""
        if response.status == 401:
            hint_middle = (
                ""
                if config.workspace_id
                else (
                    "If your federation rule is scoped to multiple workspaces, set the ANTHROPIC_WORKSPACE_ID "
                    "environment variable, the 'workspace_id' config key, or the `workspaceId` option. "
                )
            )
            hint = (
                f" Ensure your federation rule matches your identity token. {hint_middle}"
                "View your authentication events in the Workload identity page of Claude Console for more details."
            )
        request_id_text = f" (request-id {request_id})" if request_id else ""
        raise AnthropicWorkloadIdentityError(
            f"Token exchange failed with status {response.status}{request_id_text}: {redacted}{hint}",
            response.status,
            redacted,
            request_id,
        )

    data = _parse_token_response(response, request_id)
    expires_in = _js_number(data.get("expires_in", _MISSING))
    if not math.isfinite(expires_in):
        raise AnthropicWorkloadIdentityError(
            f"Token endpoint response missing required fields: {_json(_redact_sensitive(data))}",
            response.status,
            _redact_sensitive(data),
            request_id,
        )
    return _AccessToken(token=data["access_token"], expires_at=_now_s() + expires_in)


class _Refresh:
    """One exchange in flight; every caller that joined it waits on `done`."""

    __slots__ = ("advisory", "done", "outcome", "sequence")

    def __init__(self, sequence: int, advisory: bool):
        self.sequence = sequence
        self.advisory = advisory
        self.done = tonio.Event()
        self.outcome = tonio.Result()

    def settle(self, token: _AccessToken | None, error: AnthropicWorkloadIdentityError | None) -> None:
        self.outcome.store((token, error))
        self.done.set()

    async def wait(self) -> _AccessToken:
        await self.done.wait()
        token, error = self.outcome.fetch()
        if error is not None:
            # Each waiter raises its own instance: the waiters run in parallel, and
            # raising one shared exception would interleave their tracebacks on it.
            raise error.copy() from error
        return token


class _FederationTokenCache:
    """The SDK's `TokenCache` around the federation exchange, for one `(base_url, config)`."""

    def __init__(self, base_url: str, config: AnthropicFederationConfig):
        self.base_url = base_url
        self.config = config
        self._guard = threading.Lock()
        self._cached: _AccessToken | None = None
        self._cached_sequence = 0
        self._pending: _Refresh | None = None
        self._next_force = False
        self._last_advisory_error = 0
        self._sequence = 0

    async def get_token(self, env: Mapping[str, str] | None = None) -> str:
        """A bearer token; `env` scopes the exchange's proxy if this call starts one."""
        with self._guard:
            token, wait_for, started = self._next_step()
        if started is not None:
            tonio.spawn.without_tracking(self._run(started, env))
        if wait_for is None:
            return token
        return (await wait_for.wait()).token

    def invalidate(self) -> None:
        """Drop the cached token and make the next caller start a fresh exchange
        rather than join one in flight (called after a 401)."""
        with self._guard:
            self._cached = None
            self._next_force = True

    def _next_step(self) -> tuple[str | None, _Refresh | None, _Refresh | None]:
        """The SDK's `getToken` decision, under the guard: the token to serve, the
        refresh to wait for, and the refresh this caller starts."""
        force = self._next_force
        self._next_force = False
        cached = self._cached
        if force or cached is None:
            return self._mandatory_refresh(force)
        remaining = cached.expires_at - _now_s()
        if remaining > ADVISORY_REFRESH_THRESHOLD_S:
            return cached.token, None, None
        if remaining > MANDATORY_REFRESH_THRESHOLD_S:
            return cached.token, None, self._advisory_refresh()
        return self._mandatory_refresh(False)

    def _mandatory_refresh(self, force: bool) -> tuple[None, _Refresh, _Refresh | None]:
        # A forced refresh never joins one in flight.
        if self._pending is not None and not force:
            return None, self._pending, None
        refresh = self._start(advisory=False)
        return None, refresh, refresh

    def _advisory_refresh(self) -> _Refresh | None:
        if self._pending is not None:
            return None
        if _now_s() - self._last_advisory_error < ADVISORY_REFRESH_BACKOFF_S:
            return None
        return self._start(advisory=True)

    def _start(self, *, advisory: bool) -> _Refresh:
        self._sequence += 1
        refresh = _Refresh(self._sequence, advisory)
        self._pending = refresh
        return refresh

    async def _run(self, refresh: _Refresh, env: Mapping[str, str] | None) -> None:
        try:
            token = await _exchange(self.base_url, self.config, env)
        except AnthropicWorkloadIdentityError as error:
            self._finish(refresh, None, error)
        except Exception as error:
            failure = AnthropicWorkloadIdentityError(str(error))
            failure.__cause__ = error
            self._finish(refresh, None, failure)
        except BaseException as error:
            # Only a runtime shutdown interrupts a detached exchange; its waiters must not park.
            self._finish(refresh, None, AnthropicWorkloadIdentityError(f"Token exchange interrupted: {error!r}"))
            raise
        else:
            self._finish(refresh, token, None)

    def _finish(
        self, refresh: _Refresh, token: _AccessToken | None, error: AnthropicWorkloadIdentityError | None
    ) -> None:
        with self._guard:
            # A slower exchange started earlier must not replace a newer token.
            if token is not None and refresh.sequence > self._cached_sequence:
                self._cached = token
                self._cached_sequence = refresh.sequence
            if error is not None and refresh.advisory:
                # A failed background refresh keeps the stale token served, and backs off.
                self._last_advisory_error = _now_s()
            if self._pending is refresh:
                self._pending = None
        refresh.settle(token, error)


# Pi keeps one federation client (and so one token cache) for the current
# `(baseUrl, config)`; a different pair replaces it.
_cache: _FederationTokenCache | None = None
_cache_guard = threading.Lock()


def federation_token_cache(base_url: str, config: AnthropicFederationConfig) -> _FederationTokenCache:
    """The token cache for this base URL and config, replacing the current one if they differ."""
    global _cache
    # The SDK strips trailing slashes from the exchange's base URL.
    base_url = base_url.rstrip("/")
    with _cache_guard:
        cache = _cache
        if cache is None or cache.base_url != base_url or cache.config != config:
            cache = _cache = _FederationTokenCache(base_url, config)
        return cache


def reset_federation_token_cache() -> bool:
    """Drop the current token cache (tests); returns whether one was set."""
    global _cache
    with _cache_guard:
        had_cache = _cache is not None
        _cache = None
    return had_cache
