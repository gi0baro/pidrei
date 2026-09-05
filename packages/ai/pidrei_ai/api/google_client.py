"""The slice of `@google/genai` that pi's two Google adapters depend on.

pi calls `new GoogleGenAI(config).models.generateContentStream(params)` and the
SDK owns everything below it: URL and apiVersion joining, the
`GenerateContentParameters` → wire-body mapping, auth (an API key header or ADC
through `google-auth-library`), and SSE decoding. There is no equivalent
dependency here — the Python SDK is asyncio/httpx-based, which would sit outside
tonio and outside the punkreq seam — so this module is that slice, hand-rolled
like every other adapter's transport.

It deliberately keeps the SDK's own camelCase config keys (`vertexai`,
`apiVersion`, `httpOptions`, `baseUrlResourceScope`): the config record *is* the
surface pi's `google-vertex-api-key-resolution.test.ts` asserts on, so mirroring
that spec means mirroring the keys.

Every wire detail below was read out of `@google/genai` 1.52.0 — the version pi
pins — rather than inferred from documentation, because a wrong guess here is a
silently broken provider:

- Body: `{contents, systemInstruction?, tools?, toolConfig?, generationConfig}`,
  where `generationConfig` collects temperature / maxOutputTokens /
  thinkingConfig and is emitted even when empty. `abortSignal` rides in
  `params.config` and is dropped from the body, because the SDK's converter
  reads a whitelist.
- Model resource: `models/{id}` for the Gemini API, `publishers/google/models/{id}`
  for Vertex.
- `baseUrlResourceScope: "COLLECTION"` means "do not prepend
  `projects/{project}/locations/{location}`" — as does an API key, which is what
  makes Vertex express mode work.
- Headers: `x-goog-api-key` for a key, `Authorization: Bearer` for ADC, and
  neither overwrites a header the caller already set.

One divergence: `User-Agent`/`x-goog-api-client` keep the SDK's label but report
the real runtime, `gl-python/3.14.x` instead of `gl-node/v22.x`. Claiming to be
a Node process would be a lie told to a third party for no behavioural gain.
"""

import json
import platform
from collections.abc import AsyncGenerator, Mapping
from typing import Any
from urllib.parse import urlencode

from pidrei_ai.auth.google_adc import get_access_token
from pidrei_ai.types import ProviderEnv
from pidrei_ai.utils import http
from pidrei_ai.utils.cancel import CancelToken
from pidrei_ai.utils.sse import iterate_sse_messages


# The `@google/genai` release pi pins; the label is what identifies the wire
# protocol this module implements.
GENAI_SDK_VERSION = "1.52.0"
LIBRARY_LABEL = f"google-genai-sdk/{GENAI_SDK_VERSION}"

GEMINI_API_DEFAULT_VERSION = "v1beta"
VERTEX_API_DEFAULT_VERSION = "v1beta1"
GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/"
VERTEX_GLOBAL_BASE_URL = "https://aiplatform.googleapis.com/"
# Vertex serves these two through a dedicated multi-regional hostname.
MULTI_REGIONAL_LOCATIONS = frozenset({"us", "eu"})
RESOURCE_SCOPE_COLLECTION = "COLLECTION"
GOOGLE_API_KEY_HEADER = "x-goog-api-key"


class GoogleApiError(Exception):
    """The SDK's `ApiError`: an HTTP status plus the response body as its message."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _language_label() -> str:
    return f"gl-python/{platform.python_version()}"


class GoogleGenAI:
    """`new GoogleGenAI(config)`, for the one call pi makes on it."""

    def __init__(self, config: Mapping[str, Any]):
        self._config = dict(config)
        http_options = self._config.get("httpOptions") or {}
        self._http_options: dict[str, Any] = dict(http_options)

    # -- config resolution -----------------------------------------------------

    @property
    def _vertexai(self) -> bool:
        return bool(self._config.get("vertexai"))

    def _api_version(self) -> str:
        if "apiVersion" in self._http_options:
            return self._http_options["apiVersion"] or ""
        api_version = self._config.get("apiVersion")
        if api_version is not None:
            return api_version
        return VERTEX_API_DEFAULT_VERSION if self._vertexai else GEMINI_API_DEFAULT_VERSION

    def _base_url(self) -> str:
        custom = self._http_options.get("baseUrl")
        if custom:
            return custom
        if not self._vertexai:
            return GEMINI_DEFAULT_BASE_URL
        location = self._config.get("location")
        if self._config.get("apiKey") or location == "global":
            return VERTEX_GLOBAL_BASE_URL
        if location in MULTI_REGIONAL_LOCATIONS:
            return f"https://aiplatform.{location}.rep.googleapis.com/"
        return f"https://{location}-aiplatform.googleapis.com/"

    def _model_resource(self, model_id: str) -> str:
        if ".." in model_id or "?" in model_id or "&" in model_id:
            raise ValueError("invalid model parameter")
        if self._vertexai:
            if model_id.startswith(("publishers/", "projects/", "models/")):
                return model_id
            if "/" in model_id:
                publisher, name = model_id.split("/", 1)
                return f"publishers/{publisher}/models/{name}"
            return f"publishers/google/models/{model_id}"
        if model_id.startswith(("models/", "tunedModels/")):
            return model_id
        return f"models/{model_id}"

    def _prepends_project_location(self, path: str) -> bool:
        if self._http_options.get("baseUrl") and (
            self._http_options.get("baseUrlResourceScope") == RESOURCE_SCOPE_COLLECTION
        ):
            return False
        if self._config.get("apiKey"):
            return False
        if not self._vertexai:
            return False
        return not path.startswith("projects/")

    def _request_url(self, model_id: str) -> str:
        resource = self._model_resource(model_id)
        elements = [self._base_url().rstrip("/")]
        api_version = self._api_version()
        if api_version:
            elements.append(api_version)
        if self._prepends_project_location(resource):
            elements.append(f"projects/{self._config.get('project')}/locations/{self._config.get('location')}")
        elements.append(f"{resource}:streamGenerateContent")
        return f"{'/'.join(elements)}?{urlencode({'alt': 'sse'})}"

    async def _headers(self, env: ProviderEnv | None) -> dict[str, str]:
        label = f"{LIBRARY_LABEL} {_language_label()}"
        headers: dict[str, str] = {"User-Agent": label, "x-goog-api-client": label}
        for key, value in (self._http_options.get("headers") or {}).items():
            if value is not None:
                headers[key] = value

        # The SDK never overwrites a header the caller already supplied.
        present = {key.lower() for key in headers}
        api_key = self._config.get("apiKey")
        if api_key:
            if GOOGLE_API_KEY_HEADER not in present:
                headers[GOOGLE_API_KEY_HEADER] = api_key
        elif "authorization" not in present:
            key_filename = (self._config.get("googleAuthOptions") or {}).get("keyFilename")
            token_env = dict(env or {})
            if key_filename:
                token_env["GOOGLE_APPLICATION_CREDENTIALS"] = key_filename
            headers["Authorization"] = f"Bearer {await get_access_token(token_env or None)}"
        return headers

    # -- request ---------------------------------------------------------------

    async def generate_content_stream(
        self,
        params: Mapping[str, Any],
        *,
        env: ProviderEnv | None = None,
        cancel: CancelToken | None = None,
    ) -> AsyncGenerator[dict[str, Any]]:
        """`await client.models.generateContentStream(params)`.

        pi never touches the rest of the `models` namespace, so the method sits
        directly on the client instead of behind an empty holder object. Like
        the SDK's promise, this resolves once the response headers arrive (so a
        retry wrapper around the call covers the request, and `GoogleApiError`
        raises here); only the body is consumed through the returned generator.
        """
        url = self._request_url(params["model"])
        body = build_request_body(params, vertexai=self._vertexai)
        headers = await self._headers(env)

        client = http.client_for(url, env)
        response = await client.post(url, json=body, headers=headers, timeout=http.STREAMING_TIMEOUT)
        if not 200 <= response.status_code < 300:
            raw = (await response.read()).decode("utf-8", "replace")
            raise GoogleApiError(response.status_code, _error_message(response.status_code, raw))

        return _iterate_chunks(response, cancel)


def _error_message(status: int, raw: str) -> str:
    """The SDK's `ApiError` message: the error body, JSON-stringified.

    A non-JSON body is wrapped in the same `{"error": {...}}` envelope the SDK
    synthesizes, so `format_provider_error` sees one shape either way.
    """
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = {"error": {"message": raw, "code": status, "status": ""}}
    return json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)


async def _iterate_chunks(response: Any, cancel: CancelToken | None) -> AsyncGenerator[dict[str, Any]]:
    body = response.iter_bytes()
    ended = False
    try:
        async for message in iterate_sse_messages(body):
            chunk = json.loads(message.data)
            if not isinstance(chunk, dict):
                continue
            # A 200 response can still carry an error payload mid-stream. The SDK
            # sniffs every raw chunk for one; checking the decoded event is the
            # same test on better-framed input.
            error = chunk.get("error")
            if isinstance(error, dict):
                code = error.get("code")
                if isinstance(code, int) and 400 <= code < 600:
                    raise GoogleApiError(code, json.dumps(chunk, separators=(",", ":"), ensure_ascii=False))
            yield chunk
        ended = True
    finally:
        await http.finish_body(body, response, drain=ended, cancel=cancel)


# Config keys the SDK lifts out of `config` and onto the request body itself;
# everything else it recognizes stays under `generationConfig`.
_PARENT_LEVEL_CONFIG_KEYS = (
    "systemInstruction",
    "safetySettings",
    "tools",
    "toolConfig",
    "cachedContent",
    "serviceTier",
)
# The `generationConfig` fields the two adapters actually set. The SDK's
# converter recognizes many more; an unknown key is dropped rather than sent, so
# this list is the whitelist, not a convenience.
_GENERATION_CONFIG_KEYS = (
    "temperature",
    "topP",
    "topK",
    "candidateCount",
    "maxOutputTokens",
    "stopSequences",
    "seed",
    "responseMimeType",
    "responseJsonSchema",
    "responseModalities",
    "mediaResolution",
    "thinkingConfig",
    "imageConfig",
)


def _to_content(value: Any) -> Any:
    """`tContent`: a bare string becomes a user-role Content."""
    if isinstance(value, str):
        return {"role": "user", "parts": [{"text": value}]}
    return value


def build_request_body(params: Mapping[str, Any], *, vertexai: bool) -> dict[str, Any]:
    """`generateContentParametersTo{Mldev,Vertex}` for the fields pi sets.

    Both directions agree on every key the adapters produce, so one function
    covers them; `vertexai` is taken for symmetry with the SDK and to keep the
    divergence visible if that stops being true.
    """
    del vertexai
    body: dict[str, Any] = {"contents": params["contents"]}
    config = params.get("config")
    if config is None:
        return body

    for key in _PARENT_LEVEL_CONFIG_KEYS:
        if config.get(key) is not None:
            body[key] = _to_content(config[key]) if key == "systemInstruction" else config[key]

    generation_config: dict[str, Any] = {}
    for key in _GENERATION_CONFIG_KEYS:
        if config.get(key) is not None:
            generation_config[key] = config[key]
    # Set unconditionally: the SDK emits `generationConfig` whenever a config
    # object was passed, empty or not.
    body["generationConfig"] = generation_config
    return body
