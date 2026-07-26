"""Google Cloud Application Default Credentials — the slice Vertex AI needs.

pi gets this for free: `@google/genai` constructs a `GoogleAuth` from
`google-auth-library` and calls `getRequestHeaders()`. There is no equivalent
dependency here, so this module resolves ADC the way that library documents and
mints the `Authorization: Bearer` header itself, over the same
`auth/oauth/http.py` request seam every other token exchange uses.

Resolution order mirrors ADC: an explicit credentials file
(`GOOGLE_APPLICATION_CREDENTIALS`, which is what pi passes through as
`googleAuthOptions.keyFilename`), then gcloud's well-known user credentials, then
the GCE/Cloud Run metadata server. Two of the three JSON credential types are
supported: `authorized_user` (a plain refresh-token grant) and `service_account`
(a self-signed JWT assertion, hence the `cryptography` dependency — the only
place in pidrei that needs asymmetric signing). Workload-identity federation
(`external_account`) and `impersonated_service_account` raise rather than
half-work; google-auth reaches out to further token exchanges for those.

Access tokens are cached until shortly before expiry, keyed by the credential
they came from: `getRequestHeaders()` caches too, and a token exchange per
streaming request would be both slow and rate-limited.
"""

import base64
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from pidrei_ai.auth.oauth import http as oauth_http
from pidrei_ai.types import ProviderEnv
from pidrei_ai.utils.provider_env import get_provider_env_value


CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
DEFAULT_TOKEN_URI = "https://oauth2.googleapis.com/token"  # noqa: S105 - an endpoint, not a secret
JWT_BEARER_GRANT = "urn:ietf:params:oauth:grant-type:jwt-bearer"
WELL_KNOWN_ADC_PATH = ".config/gcloud/application_default_credentials.json"
METADATA_TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"  # noqa: S105 - an endpoint, not a secret
# Service-account assertions are valid for an hour; google-auth uses the same.
_ASSERTION_LIFETIME_S = 3600
# Refresh this far before expiry so an in-flight request cannot outlive its token.
_EXPIRY_MARGIN_S = 60


class GoogleAdcError(Exception):
    """ADC could not be resolved, or a token exchange failed."""


_token_cache: dict[str, tuple[str, float]] = {}
_token_cache_guard = threading.Lock()


def reset_google_adc_token_cache() -> None:
    """Drop cached access tokens (tests; also useful after a credential change)."""
    with _token_cache_guard:
        _token_cache.clear()


def _credentials_path(env: ProviderEnv | None) -> Path | None:
    explicit = get_provider_env_value("GOOGLE_APPLICATION_CREDENTIALS", env)
    if explicit:
        return Path(explicit).expanduser()
    well_known = Path(os.path.expanduser("~")) / WELL_KNOWN_ADC_PATH
    return well_known if well_known.exists() else None


def _load_credentials(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text("utf-8"))
    except OSError as error:
        raise GoogleAdcError(f"Could not read Google credentials file {path}: {error}") from error
    except ValueError as error:
        raise GoogleAdcError(f"Google credentials file {path} is not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise GoogleAdcError(f"Google credentials file {path} does not contain a JSON object")
    return parsed


def _base64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _sign_assertion(credentials: dict[str, Any], token_uri: str, scope: str) -> str:
    private_key_pem = credentials.get("private_key")
    client_email = credentials.get("client_email")
    if not isinstance(private_key_pem, str) or not isinstance(client_email, str):
        raise GoogleAdcError("Service account credentials are missing client_email or private_key")

    # Imported here, not at module scope: only service-account credentials need
    # asymmetric signing, and `cryptography` is a compiled extension nothing else
    # in pidrei pulls in.
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    try:
        private_key = serialization.load_pem_private_key(private_key_pem.encode("utf-8"), password=None)
    except Exception as error:
        raise GoogleAdcError(f"Service account private key could not be loaded: {error}") from error
    if not isinstance(private_key, rsa.RSAPrivateKey):
        raise GoogleAdcError("Service account private key is not an RSA key")

    issued_at = int(time.time())
    header = {"alg": "RS256", "typ": "JWT"}
    claims = {
        "iss": client_email,
        "scope": scope,
        "aud": token_uri,
        "exp": issued_at + _ASSERTION_LIFETIME_S,
        "iat": issued_at,
    }
    signing_input = ".".join(
        _base64url(json.dumps(part, separators=(",", ":")).encode("utf-8")) for part in (header, claims)
    )
    signature = private_key.sign(signing_input.encode("ascii"), padding.PKCS1v15(), hashes.SHA256())
    return f"{signing_input}.{_base64url(signature)}"


async def _exchange(url: str, form: dict[str, str]) -> tuple[str, float]:
    response = await oauth_http.request(
        url,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        form=form,
    )
    if not response.ok:
        raise GoogleAdcError(f"Google token exchange failed ({response.status}): {response.text}")
    return _read_token(response.json_object(), url)


def _read_token(payload: dict[str, Any] | None, source: str) -> tuple[str, float]:
    if payload is None:
        raise GoogleAdcError(f"Google token response from {source} was not a JSON object")
    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise GoogleAdcError(f"Google token response from {source} carried no access_token")
    expires_in = payload.get("expires_in")
    lifetime = float(expires_in) if isinstance(expires_in, (int, float)) else float(_ASSERTION_LIFETIME_S)
    return access_token, time.time() + lifetime


async def _fetch_service_account_token(credentials: dict[str, Any], scope: str) -> tuple[str, float]:
    token_uri = credentials.get("token_uri")
    token_uri = token_uri if isinstance(token_uri, str) and token_uri else DEFAULT_TOKEN_URI
    assertion = _sign_assertion(credentials, token_uri, scope)
    return await _exchange(token_uri, {"grant_type": JWT_BEARER_GRANT, "assertion": assertion})


async def _fetch_authorized_user_token(credentials: dict[str, Any]) -> tuple[str, float]:
    client_id = credentials.get("client_id")
    client_secret = credentials.get("client_secret")
    refresh_token = credentials.get("refresh_token")
    if not (isinstance(client_id, str) and isinstance(client_secret, str) and isinstance(refresh_token, str)):
        raise GoogleAdcError("gcloud user credentials are incomplete; re-run `gcloud auth application-default login`")
    token_uri = credentials.get("token_uri")
    token_uri = token_uri if isinstance(token_uri, str) and token_uri else DEFAULT_TOKEN_URI
    return await _exchange(
        token_uri,
        {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
        },
    )


async def _fetch_metadata_token() -> tuple[str, float]:
    response = await oauth_http.request(
        METADATA_TOKEN_URL,
        method="GET",
        headers={"Metadata-Flavor": "Google"},
        timeout_ms=3000,
    )
    if not response.ok:
        raise GoogleAdcError(f"Metadata server token request failed ({response.status})")
    return _read_token(response.json_object(), "the metadata server")


async def _fetch_token(env: ProviderEnv | None, scope: str) -> tuple[str, str, float]:
    """Return `(cache key, access token, expiry)` for the resolved credentials."""
    path = _credentials_path(env)
    if path is not None:
        credentials = _load_credentials(path)
        credential_type = credentials.get("type")
        if credential_type == "service_account":
            token, expiry = await _fetch_service_account_token(credentials, scope)
            return f"service_account:{credentials.get('client_email')}:{scope}", token, expiry
        if credential_type == "authorized_user":
            token, expiry = await _fetch_authorized_user_token(credentials)
            return f"authorized_user:{credentials.get('client_id')}", token, expiry
        if credential_type in ("external_account", "external_account_authorized_user"):
            raise GoogleAdcError(
                "Workload identity federation credentials are not supported; use a service account "
                "key, `gcloud auth application-default login`, or a Vertex AI API key"
            )
        if credential_type == "impersonated_service_account":
            raise GoogleAdcError(
                "Impersonated service account credentials are not supported; use a service account "
                "key, `gcloud auth application-default login`, or a Vertex AI API key"
            )
        raise GoogleAdcError(f"Unsupported Google credential type in {path}: {credential_type}")

    token, expiry = await _fetch_metadata_token()
    return f"metadata:{scope}", token, expiry


async def get_access_token(env: ProviderEnv | None = None, scope: str = CLOUD_PLATFORM_SCOPE) -> str:
    """An OAuth access token for `scope` from Application Default Credentials."""
    cached = _cached_token(env, scope)
    if cached is not None:
        return cached

    cache_key, token, expiry = await _fetch_token(env, scope)
    with _token_cache_guard:
        _token_cache[cache_key] = (token, expiry)
    return token


def _cached_token(env: ProviderEnv | None, scope: str) -> str | None:
    """A live cached token for these credentials, without performing a fetch.

    The cache key carries the credential's identity, which is only known after
    reading the credentials file — cheap and local, and it keeps a rotated key
    from being served the previous key's token.
    """
    path = _credentials_path(env)
    if path is None:
        cache_key = f"metadata:{scope}"
    else:
        try:
            credentials = _load_credentials(path)
        except GoogleAdcError:
            return None
        credential_type = credentials.get("type")
        if credential_type == "service_account":
            cache_key = f"service_account:{credentials.get('client_email')}:{scope}"
        elif credential_type == "authorized_user":
            cache_key = f"authorized_user:{credentials.get('client_id')}"
        else:
            return None

    now = time.time()
    with _token_cache_guard:
        entry = _token_cache.get(cache_key)
    return entry[0] if entry is not None and entry[1] - _EXPIRY_MARGIN_S > now else None
