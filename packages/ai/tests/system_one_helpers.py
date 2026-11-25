"""Stub for classifier requests (pi's tests pass a `fetch` option instead).

`stub_system_one(handler)` swaps `classifier_shared._ClassifierClient` — the one
POST every System One transport and the OpenAI Decisions API make — for a
client that records each request
and answers with `await handler(request)`, a `(status, headers, body)` triple.
The recorded `payload` is the JSON-serializable request body pi's tests read
back with `JSON.parse(init.body)`.
"""

import contextlib
import json
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from typing import Any

from pidrei_ai.api import classifier_shared
from pidrei_ai.types import ProviderResponse
from pidrei_utils.cancel import CancelToken


@dataclass(slots=True)
class SystemOneRequest:
    url: str
    payload: Any
    headers: dict[str, str]
    cancel: CancelToken | None


type Handler = Callable[[SystemOneRequest], Awaitable[tuple[int, dict[str, str], str]]]


def json_response(
    body: Any, status: int = 200, headers: dict[str, str] | None = None
) -> tuple[int, dict[str, str], str]:
    return status, headers or {}, json.dumps(body)


def respond_json(body: Any) -> Handler:
    async def handler(_request: SystemOneRequest) -> tuple[int, dict[str, str], str]:
        return json_response(body)

    return handler


@contextlib.contextmanager
def stub_system_one(handler: Handler) -> Iterator[list[SystemOneRequest]]:
    requests: list[SystemOneRequest] = []

    class _StubClient:
        def __init__(self, env=None, fetch=None):
            pass

        async def post(self, url, payload, headers, cancel, on_response):
            request = SystemOneRequest(url=url, payload=json.loads(json.dumps(payload)), headers=headers, cancel=cancel)
            requests.append(request)
            status, response_headers, body = await handler(request)
            # The real client reports the head before it reads the body.
            await on_response(ProviderResponse(status=status, headers=response_headers))
            return status, response_headers, body

    original = classifier_shared._ClassifierClient
    classifier_shared._ClassifierClient = _StubClient
    try:
        yield requests
    finally:
        classifier_shared._ClassifierClient = original
