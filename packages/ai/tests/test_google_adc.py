"""pidrei-only: Application Default Credentials for Vertex AI.

pi has no spec for this — `google-auth-library` does it inside `@google/genai`.
Here it is `auth/google_adc.py`, so the resolution order, the JWT assertion a
service-account key has to produce, and the token cache are pinned here. The
assertion in particular is verified against the key's own public half: a
malformed signature would only ever surface as a 401 from Google.

`oauth_http.request` is the interception point (as in the OAuth flow mirrors);
`HOME` is redirected per test so a developer's real
`~/.config/gcloud/application_default_credentials.json` can never leak in.
"""

import base64
import contextlib
import json
import os
import tempfile
import time
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from pidrei_ai.auth import google_adc
from pidrei_ai.auth.google_adc import GoogleAdcError, get_access_token, reset_google_adc_token_cache
from pidrei_ai.auth.oauth import http as oauth_http


@pytest.fixture(autouse=True)
def _isolate(request):
    """Empty HOME, no ambient credentials env, empty token cache."""
    reset_google_adc_token_cache()
    home = tempfile.mkdtemp(prefix="pidrei-adc-home-")
    saved = {name: os.environ.get(name) for name in ("HOME", "GOOGLE_APPLICATION_CREDENTIALS")}
    os.environ["HOME"] = home
    os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

    def restore():
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        reset_google_adc_token_cache()

    request.addfinalizer(restore)


@contextlib.contextmanager
def _stub_request(handler):
    """Replace the one request function every token exchange goes through."""
    original = oauth_http.request
    calls: list[dict] = []

    async def stub(url, *, method="POST", headers=None, json_body=None, form=None, timeout_ms=None, cancel=None):
        calls.append({"url": url, "method": method, "headers": headers or {}, "form": form})
        return handler(url)

    oauth_http.request = stub
    try:
        yield calls
    finally:
        oauth_http.request = original


def _response(payload: dict, status: int = 200) -> oauth_http.OAuthHttpResponse:
    return oauth_http.OAuthHttpResponse(status=status, body=json.dumps(payload).encode())


def _write_credentials(payload: dict) -> str:
    path = Path(tempfile.mkdtemp(prefix="pidrei-adc-")) / "credentials.json"
    path.write_text(json.dumps(payload))
    return str(path)


_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_KEY_PEM = _KEY.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
).decode()


def _service_account_file(**overrides) -> str:
    payload = {
        "type": "service_account",
        "client_email": "robot@example.iam.gserviceaccount.com",
        "private_key": _KEY_PEM,
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    payload.update(overrides)
    return _write_credentials(payload)


def _decode_segment(segment: str) -> dict:
    padded = segment + "=" * (-len(segment) % 4)
    return json.loads(base64.urlsafe_b64decode(padded))


@pytest.mark.tonio
async def test_a_service_account_key_is_exchanged_for_a_token_with_a_signed_jwt_assertion():
    env = {"GOOGLE_APPLICATION_CREDENTIALS": _service_account_file()}

    with _stub_request(lambda _url: _response({"access_token": "sa-token", "expires_in": 3600})) as calls:
        token = await get_access_token(env)

    assert token == "sa-token"
    assert len(calls) == 1
    assert calls[0]["url"] == "https://oauth2.googleapis.com/token"
    assert calls[0]["form"]["grant_type"] == "urn:ietf:params:oauth:grant-type:jwt-bearer"

    header_segment, claims_segment, signature_segment = calls[0]["form"]["assertion"].split(".")
    assert _decode_segment(header_segment) == {"alg": "RS256", "typ": "JWT"}
    claims = _decode_segment(claims_segment)
    assert claims["iss"] == "robot@example.iam.gserviceaccount.com"
    assert claims["aud"] == "https://oauth2.googleapis.com/token"
    assert claims["scope"] == "https://www.googleapis.com/auth/cloud-platform"
    assert claims["exp"] - claims["iat"] == 3600
    assert claims["iat"] <= int(time.time())

    # The signature must verify against the key's public half, not merely exist.
    padded = signature_segment + "=" * (-len(signature_segment) % 4)
    _KEY.public_key().verify(
        base64.urlsafe_b64decode(padded),
        f"{header_segment}.{claims_segment}".encode(),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )


@pytest.mark.tonio
async def test_a_custom_token_uri_is_honoured():
    env = {"GOOGLE_APPLICATION_CREDENTIALS": _service_account_file(token_uri="https://sts.example/token")}

    with _stub_request(lambda _url: _response({"access_token": "t", "expires_in": 60})) as calls:
        await get_access_token(env)

    assert calls[0]["url"] == "https://sts.example/token"
    assert _decode_segment(calls[0]["form"]["assertion"].split(".")[1])["aud"] == "https://sts.example/token"


@pytest.mark.tonio
async def test_gcloud_user_credentials_use_a_refresh_token_grant():
    env = {
        "GOOGLE_APPLICATION_CREDENTIALS": _write_credentials(
            {
                "type": "authorized_user",
                "client_id": "cid",
                "client_secret": "secret",
                "refresh_token": "refresh",
            }
        )
    }

    with _stub_request(lambda _url: _response({"access_token": "user-token", "expires_in": 3599})) as calls:
        token = await get_access_token(env)

    assert token == "user-token"
    assert calls[0]["form"] == {
        "grant_type": "refresh_token",
        "client_id": "cid",
        "client_secret": "secret",
        "refresh_token": "refresh",
    }


@pytest.mark.tonio
async def test_the_well_known_gcloud_path_is_used_when_no_env_var_is_set():
    well_known = Path(os.environ["HOME"]) / ".config/gcloud/application_default_credentials.json"
    well_known.parent.mkdir(parents=True)
    well_known.write_text(
        json.dumps({"type": "authorized_user", "client_id": "c", "client_secret": "s", "refresh_token": "r"})
    )

    with _stub_request(lambda _url: _response({"access_token": "well-known", "expires_in": 60})) as calls:
        assert await get_access_token() == "well-known"

    assert calls[0]["form"]["grant_type"] == "refresh_token"


@pytest.mark.tonio
async def test_with_no_credentials_file_the_metadata_server_is_asked():
    with _stub_request(lambda _url: _response({"access_token": "metadata-token", "expires_in": 3600})) as calls:
        assert await get_access_token() == "metadata-token"

    assert calls[0]["url"].startswith("http://metadata.google.internal/")
    assert calls[0]["method"] == "GET"
    assert calls[0]["headers"]["Metadata-Flavor"] == "Google"


@pytest.mark.tonio
async def test_a_token_is_reused_until_it_nears_expiry():
    env = {"GOOGLE_APPLICATION_CREDENTIALS": _service_account_file()}

    with _stub_request(lambda _url: _response({"access_token": "cached", "expires_in": 3600})) as calls:
        assert await get_access_token(env) == "cached"
        assert await get_access_token(env) == "cached"

    assert len(calls) == 1


@pytest.mark.tonio
async def test_a_token_about_to_expire_is_refetched():
    env = {"GOOGLE_APPLICATION_CREDENTIALS": _service_account_file()}
    issued = iter(["first", "second"])

    with _stub_request(lambda _url: _response({"access_token": next(issued), "expires_in": 30})) as calls:
        assert await get_access_token(env) == "first"
        assert await get_access_token(env) == "second"

    assert len(calls) == 2


@pytest.mark.tonio
async def test_rotating_the_key_file_does_not_serve_the_previous_tokens():
    first = {"GOOGLE_APPLICATION_CREDENTIALS": _service_account_file(client_email="one@example.com")}
    second = {"GOOGLE_APPLICATION_CREDENTIALS": _service_account_file(client_email="two@example.com")}
    issued = iter(["token-one", "token-two"])

    with _stub_request(lambda _url: _response({"access_token": next(issued), "expires_in": 3600})):
        assert await get_access_token(first) == "token-one"
        assert await get_access_token(second) == "token-two"


@pytest.mark.tonio
async def test_workload_identity_federation_reports_the_supported_alternatives():
    env = {"GOOGLE_APPLICATION_CREDENTIALS": _write_credentials({"type": "external_account"})}

    with _stub_request(lambda _url: _response({})), pytest.raises(GoogleAdcError, match="Workload identity federation"):
        await get_access_token(env)


@pytest.mark.tonio
async def test_an_impersonated_service_account_reports_the_supported_alternatives():
    env = {"GOOGLE_APPLICATION_CREDENTIALS": _write_credentials({"type": "impersonated_service_account"})}

    with _stub_request(lambda _url: _response({})), pytest.raises(GoogleAdcError, match="Impersonated service account"):
        await get_access_token(env)


@pytest.mark.tonio
async def test_an_unknown_credential_type_is_named_in_the_error():
    env = {"GOOGLE_APPLICATION_CREDENTIALS": _write_credentials({"type": "gremlin"})}

    with (
        _stub_request(lambda _url: _response({})),
        pytest.raises(GoogleAdcError, match="Unsupported Google credential type"),
    ):
        await get_access_token(env)


@pytest.mark.tonio
async def test_a_rejected_token_exchange_surfaces_the_status_and_body():
    env = {"GOOGLE_APPLICATION_CREDENTIALS": _service_account_file()}

    with (
        _stub_request(lambda _url: _response({"error": "invalid_grant"}, status=400)),
        pytest.raises(GoogleAdcError, match="failed \\(400\\).*invalid_grant"),
    ):
        await get_access_token(env)


@pytest.mark.tonio
async def test_a_token_response_without_an_access_token_is_an_error():
    env = {"GOOGLE_APPLICATION_CREDENTIALS": _service_account_file()}

    with (
        _stub_request(lambda _url: _response({"expires_in": 60})),
        pytest.raises(GoogleAdcError, match="carried no access_token"),
    ):
        await get_access_token(env)


@pytest.mark.tonio
async def test_an_unreadable_credentials_file_is_an_error():
    env = {"GOOGLE_APPLICATION_CREDENTIALS": "/nonexistent/credentials.json"}

    with pytest.raises(GoogleAdcError, match="Could not read Google credentials file"):
        await get_access_token(env)


@pytest.mark.tonio
async def test_a_credentials_file_that_is_not_json_is_an_error():
    path = Path(tempfile.mkdtemp(prefix="pidrei-adc-")) / "credentials.json"
    path.write_text("not json at all")

    with pytest.raises(GoogleAdcError, match="is not valid JSON"):
        await get_access_token({"GOOGLE_APPLICATION_CREDENTIALS": str(path)})


def test_the_module_exposes_the_scope_vertex_requires():
    assert google_adc.CLOUD_PLATFORM_SCOPE == "https://www.googleapis.com/auth/cloud-platform"
