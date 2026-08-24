"""The slice of `@aws-sdk/client-bedrock-runtime` that pi's Bedrock adapter uses.

pi builds a `BedrockRuntimeClient`, registers a Smithy `build`-step middleware
for caller headers, and `send()`s a `ConverseStreamCommand`. The SDK then owns
credential resolution, SigV4 signing, the HTTP call, and decoding the
`vnd.amazon.eventstream` binary framing the reply arrives in.

**The split here (decided 2026-07-26): vendor the swamps, own the HTTP.**
botocore supplies the three pieces that are genuinely hard or genuinely
open-ended, all of them sans-io — `Credentials` resolution (env, profiles, SSO,
IMDS, assume-role, `credential_process`), `SigV4Auth` for signing, and
`EventStreamBuffer` for frame decoding — while the request itself goes over the
punkreq seam like every other adapter. That keeps one cancellation story, one
proxy configuration and one TLS stack, and it keeps Bedrock exercising the tonio
stack, which shipping the AWS SDK's own transport would not.

botocore is pure Python (no compiled extension, so free-threading is safe by
construction) and imports in ~0.3s, behind the lazy provider.

Credential resolution reads files and can reach the network (SSO, IMDS), so it
runs on `spawn_blocking` and its result is cached per client.
"""

import json
import threading
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlparse

import tonio.colored as tonio

from pidrei_ai.utils import http
from pidrei_ai.utils.cancel import CancelToken


# `StopReason` in the SDK; these are its wire values.
STOP_REASON_END_TURN = "end_turn"
STOP_REASON_STOP_SEQUENCE = "stop_sequence"
STOP_REASON_MAX_TOKENS = "max_tokens"
STOP_REASON_MODEL_CONTEXT_WINDOW_EXCEEDED = "model_context_window_exceeded"
STOP_REASON_TOOL_USE = "tool_use"

CACHE_POINT_TYPE_DEFAULT = "default"
CACHE_TTL_ONE_HOUR = "ONE_HOUR"

CONVERSATION_ROLE_USER = "user"
CONVERSATION_ROLE_ASSISTANT = "assistant"

IMAGE_FORMAT_JPEG = "jpeg"
IMAGE_FORMAT_PNG = "png"
IMAGE_FORMAT_GIF = "gif"
IMAGE_FORMAT_WEBP = "webp"

TOOL_RESULT_STATUS_ERROR = "error"
TOOL_RESULT_STATUS_SUCCESS = "success"

BEDROCK_SERVICE_NAME = "bedrock"
DEFAULT_REGION = "us-east-1"


class BedrockRuntimeServiceException(Exception):
    """The SDK's base service exception.

    `name` is the modelled exception name (`ValidationException`, ...); the
    adapter maps it to pi's human-readable prefixes, and downstream retry logic
    matches on those.
    """

    def __init__(
        self,
        name: str,
        message: str,
        status: int | None = None,
        body: str | None = None,
        request_id: str | None = None,
    ):
        super().__init__(message)
        self.name = name
        self.status = status
        self.body = body
        # The SDK's `$metadata.requestId`; present only when the error came from
        # an HTTP response that carried `x-amzn-requestid`.
        self.request_id = request_id


@dataclass(slots=True)
class ConverseStreamCommand:
    input: dict[str, Any]


@dataclass(slots=True)
class MiddlewareRegistration:
    handler: Callable
    step: str | None = None
    name: str | None = None
    priority: str | None = None


@dataclass(slots=True)
class HttpRequest:
    """The Smithy request object a `build`-step middleware mutates."""

    method: str
    url: str
    headers: dict[str, str]
    body: bytes


@dataclass(slots=True)
class MiddlewareArgs:
    request: HttpRequest


class MiddlewareStack:
    """`client.middlewareStack`, for the two steps pi registers on."""

    def __init__(self) -> None:
        self.registrations: list[MiddlewareRegistration] = []

    def add(
        self,
        handler: Callable,
        *,
        step: str | None = None,
        name: str | None = None,
        priority: str | None = None,
    ) -> None:
        self.registrations.append(MiddlewareRegistration(handler, step=step, name=name, priority=priority))

    async def apply_build(self, args: MiddlewareArgs) -> MiddlewareArgs:
        """Run the `build`-step middleware, innermost last, as Smithy does."""

        async def terminal(final_args: MiddlewareArgs) -> MiddlewareArgs:
            return final_args

        handler = terminal
        for registration in reversed(self.registrations):
            if registration.step == "build":
                handler = registration.handler(handler)
        return await handler(args)

    async def apply_deserialize(self, terminal: Callable, args: MiddlewareArgs) -> DeserializeOutput:
        """Run the `deserialize`-step middleware around the transport call.

        Smithy hands this step the raw HTTP response after the SDK receives it
        and before the modelled output is consumed, so `terminal` is the send
        itself rather than a no-op.
        """
        handler = terminal
        for registration in reversed(self.registrations):
            if registration.step == "deserialize":
                handler = registration.handler(handler)
        return await handler(args)


@dataclass(slots=True)
class ResponseMetadata:
    http_status_code: int | None = None
    request_id: str | None = None


@dataclass(slots=True)
class ConverseStreamResponse:
    metadata: ResponseMetadata
    stream: AsyncGenerator[dict[str, Any]]


@dataclass(slots=True)
class DeserializeOutput:
    """What a `deserialize`-step middleware sees: the modelled output plus the raw response."""

    output: ConverseStreamResponse
    response: Any


# Modelled stream-level exceptions, keyed by the event name Bedrock sends.
_STREAM_EXCEPTION_EVENTS = {
    "internalServerException": "InternalServerException",
    "modelStreamErrorException": "ModelStreamErrorException",
    "validationException": "ValidationException",
    "throttlingException": "ThrottlingException",
    "serviceUnavailableException": "ServiceUnavailableException",
}


class BedrockRuntimeClient:
    """`new BedrockRuntimeClient(config)` for the one command pi sends.

    The config keys are the SDK's own (`region`, `endpoint`, `credentials`,
    `profile`, `token`, `authSchemePreference`, `requestHandler`), because that
    record is the surface pi's endpoint-resolution spec asserts on.
    """

    def __init__(self, config: Mapping[str, Any] | None = None):
        self.config: dict[str, Any] = dict(config or {})
        self.middleware_stack = MiddlewareStack()
        self._credentials: Any = None
        self._credentials_resolved = False
        self._credentials_guard = threading.Lock()

    # -- request assembly ------------------------------------------------------

    def _region(self) -> str:
        return self.config.get("region") or DEFAULT_REGION

    def _endpoint(self) -> str:
        endpoint = self.config.get("endpoint")
        if endpoint:
            return str(endpoint).rstrip("/")
        return f"https://bedrock-runtime.{self._region()}.amazonaws.com"

    def _url(self, model_id: str) -> str:

        return f"{self._endpoint()}/model/{quote(model_id, safe='')}/converse-stream"

    def _bearer_token(self) -> str | None:
        token = self.config.get("token")
        if isinstance(token, Mapping):
            return token.get("token")
        return None

    async def _resolve_credentials(self) -> Any:
        """botocore's credential chain, off the event loop.

        It reads `~/.aws/{credentials,config}` and may reach SSO or IMDS, so it
        never runs inline; the result is cached for the client's lifetime, as the
        SDK's own provider chain does. The explicit-credentials branch is on the
        pool too: even a bare botocore *import* reads files, which is what the
        blocking-fs detector caught on this path.
        """
        with self._credentials_guard:
            if self._credentials_resolved:
                return self._credentials

        resolved = await tonio.spawn_blocking(self._build_credentials)

        with self._credentials_guard:
            self._credentials = resolved
            self._credentials_resolved = True
        return resolved

    def _build_credentials(self) -> Any:
        # Preload the modules `_sign` imports inline, so those imports are
        # `sys.modules` hits instead of file reads on a runtime worker.
        import botocore.auth
        import botocore.awsrequest  # noqa: F401

        explicit = self.config.get("credentials")
        if isinstance(explicit, Mapping):
            from botocore.credentials import Credentials

            return Credentials(
                access_key=explicit.get("accessKeyId"),
                secret_key=explicit.get("secretAccessKey"),
                token=explicit.get("sessionToken"),
            )
        return self._botocore_credentials()

    def _botocore_credentials(self) -> Any:
        import botocore.session

        session = botocore.session.Session(profile=self.config.get("profile") or None)
        credentials = session.get_credentials()
        if credentials is None:
            raise BedrockRuntimeServiceException(
                "CredentialsError",
                "No AWS credentials found. Configure a profile, IAM keys, or set "
                "AWS_BEARER_TOKEN_BEDROCK for Bedrock API key auth.",
            )
        return credentials

    async def _sign(self, request: HttpRequest) -> None:
        """SigV4, via botocore. Mutates `request.headers` in place."""
        credentials = await self._resolve_credentials()

        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        aws_request = AWSRequest(
            method=request.method, url=request.url, data=request.body, headers=dict(request.headers)
        )
        SigV4Auth(credentials.get_frozen_credentials(), BEDROCK_SERVICE_NAME, self._region()).add_auth(aws_request)
        for key, value in aws_request.headers.items():
            request.headers[key] = value

    # -- send ------------------------------------------------------------------

    async def send(self, command: ConverseStreamCommand, *, cancel: CancelToken | None = None):
        body = json.dumps(_without_model_id(command.input), separators=(",", ":")).encode("utf-8")
        model_id = command.input.get("modelId") or ""
        url = self._url(model_id)
        request = HttpRequest(
            method="POST",
            url=url,
            headers={
                "content-type": "application/json",
                "accept": "application/vnd.amazon.eventstream",
                "host": urlparse(url).netloc,
            },
            body=body,
        )

        # The build step runs after serialisation and before signing, so injected
        # headers are covered by the signature — the SDK's ordering.
        args = await self.middleware_stack.apply_build(MiddlewareArgs(request=request))
        request = args.request

        bearer_token = self._bearer_token()
        if bearer_token:
            request.headers["authorization"] = f"Bearer {bearer_token}"
        else:
            await self._sign(request)

        async def send_request(final_args: MiddlewareArgs) -> DeserializeOutput:
            final_request = final_args.request
            client = http.client_for(final_request.url)
            response = await client.post(
                final_request.url,
                content=final_request.body,
                headers=final_request.headers,
                timeout=http.STREAMING_TIMEOUT,
            )
            if not 200 <= response.status_code < 300:
                raw = (await response.read()).decode("utf-8", "replace")
                raise _service_exception_from_response(response.status_code, dict(response.headers), raw)

            metadata = ResponseMetadata(
                http_status_code=response.status_code,
                request_id=response.headers.get("x-amzn-requestid"),
            )
            return DeserializeOutput(
                output=ConverseStreamResponse(metadata=metadata, stream=_iterate_event_stream(response, cancel)),
                response=response,
            )

        result = await self.middleware_stack.apply_deserialize(send_request, MiddlewareArgs(request=request))
        return result.output


def _without_model_id(command_input: Mapping[str, Any]) -> dict[str, Any]:
    """`modelId` is a URI label in the Converse API, not a body field."""
    return {key: value for key, value in command_input.items() if key != "modelId" and value is not None}


def _service_exception_from_response(
    status: int, headers: Mapping[str, str], raw: str
) -> BedrockRuntimeServiceException:
    """Rebuild the modelled exception the SDK would have raised.

    Bedrock names the error in `x-amzn-errortype` (sometimes suffixed with a
    URL) or in the body's `__type`; the message is the body's `message` field.
    """
    name = headers.get("x-amzn-errortype", "").split(":")[0].split("/")[-1]
    message = raw
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        name = name or str(parsed.get("__type", "")).split("#")[-1]
        message = parsed.get("message") or parsed.get("Message") or raw
    return BedrockRuntimeServiceException(
        name or "UnknownError", message, status=status, body=raw, request_id=headers.get("x-amzn-requestid")
    )


async def _iterate_event_stream(response: Any, cancel: CancelToken | None) -> AsyncGenerator[dict[str, Any]]:
    """Decode `vnd.amazon.eventstream` frames into the SDK's event shapes.

    botocore's `EventStreamBuffer` owns the binary framing (prelude, headers,
    payload, both CRC32s); this only maps the decoded frames onto the
    `{messageStart: ...}` / `{contentBlockDelta: ...}` union the adapter reads,
    and re-raises modelled exceptions as the SDK does.
    """
    from botocore.eventstream import EventStreamBuffer

    body = response.aiter_bytes() if hasattr(response, "aiter_bytes") else response.iter_bytes()
    buffer = EventStreamBuffer()
    ended = False
    try:
        async for chunk in body:
            buffer.add_data(chunk)
            for event in buffer:
                decoded = _decode_event(event)
                if decoded is not None:
                    yield decoded
        ended = True
    finally:
        await http.finish_body(body, response, drain=ended)


def _decode_event(event: Any) -> dict[str, Any] | None:
    headers = event.headers
    message_type = headers.get(":message-type")
    event_type = headers.get(":event-type")
    payload = event.payload

    if message_type == "exception":
        exception_type = headers.get(":exception-type") or "InternalServerException"
        raise BedrockRuntimeServiceException(exception_type, _payload_message(payload) or exception_type)
    if message_type == "error":
        raise BedrockRuntimeServiceException(
            headers.get(":error-code") or "UnknownError",
            headers.get(":error-message") or "",
        )
    if event_type is None:
        return None

    parsed = json.loads(payload.decode("utf-8")) if payload else {}
    if event_type in _STREAM_EXCEPTION_EVENTS:
        raise BedrockRuntimeServiceException(_STREAM_EXCEPTION_EVENTS[event_type], parsed.get("message") or event_type)
    return {event_type: parsed}


def _payload_message(payload: bytes) -> str | None:
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except ValueError:
        return None
    return parsed.get("message") if isinstance(parsed, dict) else None
