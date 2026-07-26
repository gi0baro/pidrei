"""pidrei-only: the request `@google/genai` builds for pi.

No pi spec covers any of this — pi's Google tests stub the SDK away, so its URL
joining, params→body mapping and auth headers are only ever exercised against
the live API. Here that code is ours (`api/google_client.py`), and a wrong URL or
a dropped body field is a silently broken provider, so each rule read out of
`@google/genai` 1.52.0 gets pinned here.
"""

import contextlib
import json

import pytest

from pidrei_ai.api import google_client
from pidrei_ai.api.google_client import GoogleApiError, GoogleGenAI, build_request_body
from pidrei_ai.utils.cancel import CancelToken


# --- URL construction ---------------------------------------------------------


def _url(config: dict, model_id: str = "gemini-3-pro-preview") -> str:
    return GoogleGenAI(config)._request_url(model_id)


def test_gemini_api_url_uses_the_model_base_url_verbatim():
    # What `create_client` passes: base_url already carries /v1beta, so the
    # adapter blanks apiVersion to stop it being appended twice.
    url = _url(
        {
            "apiKey": "k",
            "httpOptions": {"baseUrl": "https://generativelanguage.googleapis.com/v1beta", "apiVersion": ""},
        }
    )
    assert url == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:streamGenerateContent?alt=sse"
    )


def test_gemini_api_url_falls_back_to_the_sdk_default_endpoint_and_version():
    assert _url({"apiKey": "k"}) == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:streamGenerateContent?alt=sse"
    )


def test_vertex_adc_url_prepends_the_project_and_location_resource_path():
    url = _url({"vertexai": True, "project": "p", "location": "us-central1", "apiVersion": "v1"})
    assert url == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1"
        "/publishers/google/models/gemini-3-pro-preview:streamGenerateContent?alt=sse"
    )


def test_vertex_global_location_uses_the_location_free_hostname():
    url = _url({"vertexai": True, "project": "p", "location": "global", "apiVersion": "v1"})
    assert url.startswith("https://aiplatform.googleapis.com/v1/projects/p/locations/global/")


@pytest.mark.parametrize("location", ["us", "eu"])
def test_vertex_multi_regional_locations_use_the_rep_hostname(location):
    url = _url({"vertexai": True, "project": "p", "location": location, "apiVersion": "v1"})
    assert url.startswith(f"https://aiplatform.{location}.rep.googleapis.com/v1/")


def test_vertex_api_key_skips_the_project_path_entirely():
    # Vertex express mode: an API key means there is no project/location to scope to.
    url = _url({"vertexai": True, "apiKey": "k", "apiVersion": "v1"})
    assert url == (
        "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-3-pro-preview"
        ":streamGenerateContent?alt=sse"
    )


def test_collection_resource_scope_suppresses_the_project_path():
    url = _url(
        {
            "vertexai": True,
            "project": "p",
            "location": "us-central1",
            "apiVersion": "v1",
            "httpOptions": {"baseUrl": "https://proxy.example.com", "baseUrlResourceScope": "COLLECTION"},
        }
    )
    assert url == (
        "https://proxy.example.com/v1/publishers/google/models/gemini-3-pro-preview:streamGenerateContent?alt=sse"
    )


def test_a_blank_api_version_is_not_appended():
    url = _url(
        {
            "vertexai": True,
            "project": "p",
            "location": "us-central1",
            "apiVersion": "v1",
            "httpOptions": {
                "baseUrl": "https://proxy.example.com/v1/projects/p/locations/global",
                "baseUrlResourceScope": "COLLECTION",
                "apiVersion": "",
            },
        }
    )
    assert url == (
        "https://proxy.example.com/v1/projects/p/locations/global"
        "/publishers/google/models/gemini-3-pro-preview:streamGenerateContent?alt=sse"
    )


def test_a_fully_qualified_model_name_is_left_alone():
    assert "/publishers/acme/models/custom:" in _url(
        {"vertexai": True, "apiKey": "k", "apiVersion": "v1"}, "acme/custom"
    )
    assert "/projects/other/models/x:" in _url(
        {"vertexai": True, "apiKey": "k", "apiVersion": "v1"}, "projects/other/models/x"
    )


@pytest.mark.parametrize("model_id", ["../escape", "with?query", "with&amp"])
def test_a_model_id_that_could_escape_the_path_is_rejected(model_id):
    with pytest.raises(ValueError, match="invalid model parameter"):
        _url({"apiKey": "k"}, model_id)


# --- request body -------------------------------------------------------------


def test_config_is_split_between_the_body_and_generation_config():
    body = build_request_body(
        {
            "model": "gemini-3-pro-preview",
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "config": {
                "temperature": 0.5,
                "maxOutputTokens": 100,
                "thinkingConfig": {"includeThoughts": True},
                "systemInstruction": "be brief",
                "tools": [{"functionDeclarations": []}],
                "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
            },
        },
        vertexai=False,
    )

    assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert body["generationConfig"] == {
        "temperature": 0.5,
        "maxOutputTokens": 100,
        "thinkingConfig": {"includeThoughts": True},
    }
    # A bare string system instruction becomes a user-role Content (`tContent`).
    assert body["systemInstruction"] == {"role": "user", "parts": [{"text": "be brief"}]}
    assert body["tools"] == [{"functionDeclarations": []}]
    assert body["toolConfig"] == {"functionCallingConfig": {"mode": "AUTO"}}
    assert "model" not in body


def test_generation_config_is_present_even_when_empty():
    body = build_request_body({"model": "m", "contents": [], "config": {}}, vertexai=False)
    assert body["generationConfig"] == {}


def test_unknown_config_keys_are_dropped_rather_than_sent():
    # The SDK's converters read a whitelist; `cancel` in particular must never
    # reach the wire (it is not JSON-serializable).
    body = build_request_body(
        {"model": "m", "contents": [], "config": {"abortSignal": CancelToken(), "somethingElse": 1}},
        vertexai=False,
    )
    assert body == {"contents": [], "generationConfig": {}}
    assert json.dumps(body)


# --- headers ------------------------------------------------------------------


async def _headers(config: dict, env=None) -> dict:
    return await GoogleGenAI(config)._headers(env)


@pytest.mark.tonio
async def test_an_api_key_travels_in_the_goog_api_key_header():
    headers = await _headers({"apiKey": "secret-key"})
    assert headers["x-goog-api-key"] == "secret-key"
    assert "Authorization" not in headers
    assert headers["User-Agent"].startswith("google-genai-sdk/")
    assert headers["User-Agent"] == headers["x-goog-api-client"]
    assert "gl-python/" in headers["User-Agent"]


@contextlib.contextmanager
def _stub_token(fake):
    """Replace the ADC token fetch.

    Deliberately not `monkeypatch`: it is a yield fixture, and those abort the
    tonio runtime rather than failing (see PLAN.md Phase 4 notes).
    """
    original = google_client.get_access_token
    google_client.get_access_token = fake
    try:
        yield
    finally:
        google_client.get_access_token = original


@pytest.mark.tonio
async def test_adc_travels_in_the_authorization_header():
    async def fake_token(env=None, scope=None):
        return "adc-token"

    with _stub_token(fake_token):
        headers = await _headers({"vertexai": True, "project": "p", "location": "global"})

    assert headers["Authorization"] == "Bearer adc-token"
    assert "x-goog-api-key" not in headers


@pytest.mark.tonio
async def test_the_key_filename_from_google_auth_options_reaches_the_token_fetch():
    seen: list = []

    async def fake_token(env=None, scope=None):
        seen.append(env)
        return "adc-token"

    with _stub_token(fake_token):
        await _headers(
            {
                "vertexai": True,
                "project": "p",
                "location": "global",
                "googleAuthOptions": {"keyFilename": "/k.json"},
            }
        )

    assert seen == [{"GOOGLE_APPLICATION_CREDENTIALS": "/k.json"}]


@pytest.mark.tonio
async def test_a_caller_supplied_auth_header_is_not_overwritten():
    async def fail(env=None, scope=None):
        raise AssertionError("must not fetch a token when Authorization is already set")

    with _stub_token(fail):
        headers = await _headers(
            {
                "vertexai": True,
                "project": "p",
                "location": "global",
                "httpOptions": {"headers": {"Authorization": "Bearer mine"}},
            }
        )

    assert headers["Authorization"] == "Bearer mine"


@pytest.mark.tonio
async def test_a_none_valued_header_is_suppressed():
    headers = await _headers({"apiKey": "k", "httpOptions": {"headers": {"x-drop": None, "x-keep": "v"}}})
    assert "x-drop" not in headers
    assert headers["x-keep"] == "v"


# --- errors -------------------------------------------------------------------


def test_a_json_error_body_becomes_the_error_message():
    message = google_client._error_message(429, '{"error": {"code": 429, "message": "quota"}}')
    assert json.loads(message) == {"error": {"code": 429, "message": "quota"}}


def test_a_non_json_error_body_is_wrapped_in_the_sdk_envelope():
    parsed = json.loads(google_client._error_message(502, "upstream boom"))
    assert parsed["error"]["message"] == "upstream boom"
    assert parsed["error"]["code"] == 502


@pytest.mark.tonio
async def test_an_error_payload_mid_stream_raises_rather_than_being_yielded():
    async def body():
        yield b'data: {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}\n\n'
        yield b'data: {"error": {"code": 500, "status": "INTERNAL", "message": "boom"}}\n\n'

    class _Response:
        def iter_bytes(self):
            return body()

        async def close(self):
            pass

    chunks = []
    with pytest.raises(GoogleApiError) as excinfo:
        async for chunk in google_client._iterate_chunks(_Response(), None):
            chunks.append(chunk)

    assert len(chunks) == 1
    assert excinfo.value.status == 500
    assert "boom" in str(excinfo.value)
