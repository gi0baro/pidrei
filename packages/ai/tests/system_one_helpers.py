"""Stub for System One requests (pi's tests pass a `fetch` option instead).

`stub_system_one(handler)` swaps `system_one_shared._SystemOneClient` — the one
POST every System One transport makes — for a client that records each request
and answers with `await handler(request)`, a `(status, headers, body)` triple.
The recorded `payload` is the JSON-serializable request body pi's tests read
back with `JSON.parse(init.body)`.
"""

import contextlib
import json
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from typing import Any

from pidrei_ai.api import system_one_shared
from pidrei_ai.utils.cancel import CancelToken


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
        def __init__(self, env=None):
            pass

        async def post(self, url, payload, headers, cancel):
            request = SystemOneRequest(url=url, payload=json.loads(json.dumps(payload)), headers=headers, cancel=cancel)
            requests.append(request)
            return await handler(request)

    original = system_one_shared._SystemOneClient
    system_one_shared._SystemOneClient = _StubClient
    try:
        yield requests
    finally:
        system_one_shared._SystemOneClient = original
