"""Port of pi's auth resolution (packages/ai/src/auth/resolve.ts).

A stored credential owns the provider: ambient/env is consulted only when
nothing is stored. No silent env fallback after a failed refresh or for a
credential type without a matching handler.
"""

import time
from dataclasses import dataclass

from pidrei_ai.auth.types import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthContext,
    AuthResult,
    Credential,
    CredentialStore,
    OAuthAuth,
    OAuthCredential,
    ProviderAuth,
)
from pidrei_ai.types import ProviderEnv
from pidrei_ai.utils.diagnostics import format_thrown_value


type ModelsErrorCode = str  # "model_source" | "model_validation" | "provider" | "stream" | "auth" | "oauth"


class ModelsError(Exception):
    def __init__(self, code: ModelsErrorCode, message: str, *, cause: BaseException | object | None = None):
        super().__init__(_with_cause_detail(message, cause))
        self.code = code
        if isinstance(cause, BaseException):
            self.__cause__ = cause


def _with_cause_detail(message: str, cause: BaseException | object | None) -> str:
    """Callers surface `str(error)` only, so keep the underlying reason in it."""
    if cause is None:
        return message
    detail = format_thrown_value(cause).strip()
    if not detail or detail in message:
        return message
    return f"{message}: {detail}"


@dataclass(slots=True)
class AuthResolutionOverrides:
    api_key: str | None = None
    env: ProviderEnv | None = None
    #: Require this much remaining OAuth-token validity; defaults to five minutes.
    min_oauth_validity_ms: int | None = None


def _now_ms() -> int:
    return int(time.time() * 1000)


class _OverlayEnvAuthContext:
    __slots__ = ("_base", "_env")

    def __init__(self, base: AuthContext, env: ProviderEnv):
        self._base = base
        self._env = env

    async def env(self, name: str) -> str | None:
        return self._env.get(name) or await self._base.env(name)

    async def file_exists(self, path: str) -> bool:
        return await self._base.file_exists(path)


async def resolve_provider_auth(
    provider,  # anything with `.id` and `.auth: ProviderAuth`
    credentials: CredentialStore,
    auth_context: AuthContext,
    overrides: AuthResolutionOverrides | None = None,
) -> AuthResult | None:
    """Auth resolution shared by the Models collection."""
    provider_auth: ProviderAuth = provider.auth
    request_auth_context: AuthContext = (
        _OverlayEnvAuthContext(auth_context, overrides.env) if overrides is not None and overrides.env else auth_context
    )

    if overrides is not None and overrides.api_key is not None and provider_auth.api_key is not None:
        return await _resolve_api_key(
            request_auth_context,
            provider_auth.api_key,
            provider.id,
            ApiKeyCredential(key=overrides.api_key, env=overrides.env),
        )

    stored = await _read_credential(credentials, provider.id)
    if stored is not None:
        if stored.type == "oauth" and provider_auth.oauth is not None:
            return await _resolve_stored_oauth(
                credentials,
                provider.id,
                provider_auth.oauth,
                stored,
                overrides.min_oauth_validity_ms if overrides is not None else None,
            )
        if stored.type == "api_key" and provider_auth.api_key is not None:
            credential = stored
            if overrides is not None and overrides.env:
                credential = ApiKeyCredential(key=stored.key, env={**(stored.env or {}), **overrides.env})
            return await _resolve_api_key(request_auth_context, provider_auth.api_key, provider.id, credential)
        return None

    # Ambient (env vars, AWS profiles, ADC files).
    if provider_auth.api_key is not None:
        return await _resolve_api_key(request_auth_context, provider_auth.api_key, provider.id, None)
    return None


DEFAULT_OAUTH_MINIMUM_VALIDITY_MS = 5 * 60 * 1000


async def _resolve_stored_oauth(
    credentials: CredentialStore,
    provider_id: str,
    oauth: OAuthAuth,
    stored: OAuthCredential,
    min_oauth_validity_ms: int | None = None,
) -> AuthResult | None:
    """OAuth resolution with double-checked locking: tokens with less than five
    minutes remaining lock, re-check expiry under the lock, refresh once
    globally, and persist the rotated credential before release.
    """
    minimum_validity_ms = max(DEFAULT_OAUTH_MINIMUM_VALIDITY_MS, min_oauth_validity_ms or 0)

    def expires_soon(current: OAuthCredential) -> bool:
        return _now_ms() + minimum_validity_ms >= current.expires

    credential = stored

    if expires_soon(credential):
        # Optimistic check said expired; the authoritative check runs under the lock.
        async def _refresh_under_lock(current: Credential | None) -> Credential | None:
            if current is None or current.type != "oauth":
                return None  # logged out meanwhile
            if not expires_soon(current):
                return None  # another process/request refreshed
            try:
                return await oauth.refresh(current, None)
            except Exception as error:
                raise ModelsError("oauth", f"OAuth refresh failed for {provider_id}", cause=error)

        try:
            post = await credentials.modify(provider_id, _refresh_under_lock)
        except ModelsError:
            raise
        except Exception as error:
            raise ModelsError("auth", f"Credential store modify failed for {provider_id}", cause=error)
        if post is None or post.type != "oauth":
            return None  # logged out meanwhile
        credential = post
        # The normal five-minute window triggers a refresh but does not impose a
        # provider contract. Explicit callers (such as bearer-token export) do
        # require the requested minimum after the refresh.
        if min_oauth_validity_ms is not None and expires_soon(credential):
            raise ModelsError("oauth", f"OAuth refresh returned a token that expires too soon for {provider_id}")

    try:
        return AuthResult(auth=await oauth.to_auth(credential), source="OAuth")
    except Exception as error:
        raise ModelsError("oauth", f"OAuth auth derivation failed for {provider_id}", cause=error)


async def _resolve_api_key(
    auth_context: AuthContext,
    api_key: ApiKeyAuth,
    provider_id: str,
    credential: ApiKeyCredential | None,
) -> AuthResult | None:
    try:
        return await api_key.resolve(auth_context, credential)
    except Exception as error:
        raise ModelsError("auth", f"API key auth failed for provider {provider_id}", cause=error)


async def _read_credential(credentials: CredentialStore, provider_id: str) -> Credential | None:
    try:
        return await credentials.read(provider_id)
    except Exception as error:
        raise ModelsError("auth", f"Credential store read failed for {provider_id}", cause=error)
