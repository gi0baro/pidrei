"""Shared helpers for the OAuth flow mirrors.

pi drives these suites with two vitest facilities that have no direct
equivalent, so each gets a pidrei-owned seam:

- `vi.useFakeTimers()` + `advanceTimersByTimeAsync(n)` → `virtual_clock()`,
  which replaces `utils/clock.py`'s two functions with a clock that only moves
  when the code under test sleeps. It advances on its own rather than on the
  test's command, so pi's "nothing happened yet after 4999 ms" assertions become
  assertions on the recorded poll *times* — which carry the same information: a
  poll recorded at `start + 5000` waited exactly 5 s.
- `vi.stubGlobal("fetch", ...)` → `stub_oauth_http()`, replacing the single
  request function every flow goes through.

Neither uses a yield fixture, which aborts under `@pytest.mark.tonio`.
"""

import contextlib
import inspect
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from tonio.colored import time as tonio_time

from pidrei_ai.auth.oauth import http as oauth_http
from pidrei_ai.auth.types import AuthEvent, AuthPrompt
from pidrei_ai.utils import clock
from pidrei_ai.utils.cancel import AbortError, CancelToken


# An arbitrary fixed instant; pi picks one per suite with `vi.setSystemTime`.
DEFAULT_START_MS = 1773014400000  # 2026-03-09T00:00:00Z


@contextlib.contextmanager
def virtual_clock(start_ms: int = DEFAULT_START_MS):
    """Run with a clock that advances only through `clock.sleep_ms`.

    Yields a one-key dict so a test can read the current virtual time.
    """
    state = {"now": start_ms}
    original_now = clock.now_ms
    original_sleep = clock.sleep_ms

    def now_ms() -> int:
        return state["now"]

    async def sleep_ms(ms: float, cancel: CancelToken | None = None) -> None:
        if cancel is not None and cancel.cancelled:
            raise AbortError("Operation was aborted")
        state["now"] += int(ms)
        await tonio_time.sleep(0)  # a checkpoint, so concurrent tasks still interleave

    clock.now_ms = now_ms
    clock.sleep_ms = sleep_ms
    try:
        yield state
    finally:
        clock.now_ms = original_now
        clock.sleep_ms = original_sleep


@contextlib.contextmanager
def process_env(**values: str):
    """`vi.stubEnv`: set (or clear, with None) process env vars for the duration."""
    saved = {name: os.environ.pop(name, None) for name in values}
    os.environ.update({name: value for name, value in values.items() if value is not None})
    try:
        yield
    finally:
        for name in values:
            os.environ.pop(name, None)
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value


@dataclass(slots=True)
class OAuthRequest:
    """One recorded request through the OAuth HTTP seam."""

    url: str
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    json_body: Any = None
    form: dict[str, str] = field(default_factory=dict)
    timeout_ms: float | None = None
    cancel: CancelToken | None = None


def json_response(body: Any, status: int = 200) -> oauth_http.OAuthHttpResponse:
    return oauth_http.OAuthHttpResponse(
        status=status,
        body=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"},
    )


def text_response(body: str, status: int = 200, headers: dict[str, str] | None = None) -> oauth_http.OAuthHttpResponse:
    return oauth_http.OAuthHttpResponse(
        status=status,
        body=body.encode("utf-8"),
        headers={"content-type": "text/plain", **(headers or {})},
    )


type _Handler = Callable[[OAuthRequest], oauth_http.OAuthHttpResponse | Awaitable[oauth_http.OAuthHttpResponse]]


@contextlib.contextmanager
def stub_oauth_http(handler: _Handler):
    """Answer every OAuth request from `handler`; yields the recorded requests."""
    calls: list[OAuthRequest] = []
    original = oauth_http.request

    async def request(
        url: str,
        *,
        method: str = "POST",
        headers: dict[str, str] | None = None,
        json_body: Any = None,
        form: dict[str, str] | None = None,
        timeout_ms: float | None = None,
        cancel: CancelToken | None = None,
    ) -> oauth_http.OAuthHttpResponse:
        recorded = OAuthRequest(
            url=url,
            method=method,
            headers=dict(headers or {}),
            json_body=json_body,
            form=dict(form or {}),
            timeout_ms=timeout_ms,
            cancel=cancel,
        )
        calls.append(recorded)
        result = handler(recorded)
        if inspect.isawaitable(result):
            result = await result
        return result

    oauth_http.request = request
    try:
        yield calls
    finally:
        oauth_http.request = original


class RecordingInteraction:
    """An `AuthInteraction` that records events and answers prompts from a callback."""

    def __init__(
        self,
        prompt: Callable[[AuthPrompt], Any] | None = None,
        cancel: CancelToken | None = None,
    ):
        # Provider logins receive a normalized interaction whose cancel is
        # always present (pi's `ProviderAuthInteraction`).
        self.cancel = cancel if cancel is not None else CancelToken()
        self.events: list[AuthEvent] = []
        self.prompts: list[AuthPrompt] = []
        self._prompt = prompt

    async def prompt(self, prompt: AuthPrompt) -> str:
        self.prompts.append(prompt)
        if self._prompt is None:
            raise AssertionError(f"Unexpected prompt: {prompt.type}")
        result = self._prompt(prompt)
        if inspect.isawaitable(result):
            result = await result
        return result

    def notify(self, event: AuthEvent) -> None:
        self.events.append(event)

    def events_of(self, event_type: str) -> list[AuthEvent]:
        return [event for event in self.events if event.type == event_type]
