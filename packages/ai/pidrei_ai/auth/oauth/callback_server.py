"""The provider flows' shared OAuth callback handler (port of pi's
packages/ai/src/auth/oauth/callback-server.ts).

pi's module has two layers. The lower one, what pi gets from `node:http`'s
`createServer`, is `pidrei_http.callback_server`. This module is the upper
one, `start_oauth_callback_server` + `wait_for_callback_or_manual_input`: one
handler over the lower layer shared by the Anthropic, OpenAI Codex and
OpenRouter flows. The ChatGPT flow keeps its own handler on the lower layer,
as pi keeps its own `createServer` there.
"""

import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

import tonio.colored as tonio

from pidrei_ai.auth.types import AuthPrompt, ProviderAuthInteraction
from pidrei_ai.utils.oauth_page import oauth_error_html, oauth_success_html
from pidrei_http.callback_server import (
    CallbackRequest,
    CallbackResponse,
    CallbackServer,
    OneShotValue,
    start_callback_server,
)
from pidrei_utils.cancel import CancelToken


async def keep_code(code: str) -> str:
    """`complete` for flows that exchange the code after the wait (pi's `async (code) => code`)."""
    return code


class OAuthCallbackServer[T]:
    """pi's `OAuthCallbackServer<T>`.

    pi's `claimed`/`settled` pair runs on one thread; here the connection
    handlers, `cancel()`, `close()`, the cancel listener and the deadline can
    run in parallel, so every claim and settle decision is taken under `_lock`.
    """

    __slots__ = ("_claimed", "_lock", "_result", "_server", "_unsubscribe", "redirect_uri")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._claimed = False
        self._result = OneShotValue()
        self._unsubscribe: Callable[[], None] | None = None
        self._server: CallbackServer | None = None
        self.redirect_uri = ""

    async def wait(self) -> T | None:
        """The result of `complete`, or None after `cancel()`. Raises when the
        provider redirects with an error, `complete` fails, the login is
        cancelled, or the timeout elapses."""
        outcome, value = await self._result.wait()
        if outcome == "error":
            raise value
        return value

    def cancel(self) -> None:
        """Stop waiting for the browser unless a callback is already being completed."""
        with self._lock:
            if self._claimed or self._result.settled:
                return
            self._result.settle(("ok", None))
        self._release()

    def close(self) -> None:
        self._finish("error", RuntimeError("OAuth callback server closed"))
        if self._server is not None:
            self._server.close()

    def _finish(self, outcome: Literal["ok", "error"], value: Any) -> None:
        with self._lock:
            if self._result.settled:
                return
            self._result.settle((outcome, value))
        self._release()

    def _claim(self, error: Exception | None, code: str | None) -> Literal["busy", "error", "missing", "claimed"]:
        """The handler's `claimed || settled` check and what follows it, as one step."""
        with self._lock:
            if self._claimed or self._result.settled:
                return "busy"
            if error is not None:
                self._result.settle(("error", error))
                verdict: Literal["busy", "error", "missing", "claimed"] = "error"
            elif not code:
                return "missing"
            else:
                self._claimed = True
                return "claimed"
        self._release()
        return verdict

    def _watch(self, unsubscribe: Callable[[], None]) -> None:
        with self._lock:
            if not self._result.settled:
                self._unsubscribe = unsubscribe
                return
        unsubscribe()

    def _release(self) -> None:
        """pi's `signal.removeEventListener` in `finish`."""
        with self._lock:
            unsubscribe, self._unsubscribe = self._unsubscribe, None
        if unsubscribe is not None:
            unsubscribe()


async def start_oauth_callback_server[T](
    *,
    provider_name: str,
    host: str,
    port: int,
    path: str,
    complete: Callable[[str], Awaitable[T]],
    redirect_host: str | None = None,
    state: str | None = None,
    cancel: CancelToken | None = None,
    timeout_ms: float | None = None,
) -> OAuthCallbackServer[T]:
    """pi's `startOAuthCallbackServer`.

    `port` 0 picks a free port; a taken port raises (the flows decide whether
    to fall back to the manual prompt). `state` is omitted for providers that
    send none. `complete` finishes the sign-in with the received code before
    the browser page is sent, so the page can show exchange failures.
    """
    if cancel is not None and cancel.cancelled:
        raise RuntimeError("Login cancelled")

    callback: OAuthCallbackServer[T] = OAuthCallbackServer()

    async def handle(request: CallbackRequest) -> CallbackResponse:
        if request.method != "GET" or request.path != path:
            return CallbackResponse(404, oauth_error_html("Callback route not found."))
        if state is not None and request.get("state") != state:
            return CallbackResponse(400, oauth_error_html("State mismatch."))
        error = request.get("error")
        description = (request.get("error_description") or error) if error else None
        code = request.get("code")
        verdict = callback._claim(
            RuntimeError(f"{provider_name} authorization failed: {description}") if error else None, code
        )
        if verdict == "busy":
            return CallbackResponse(409, oauth_error_html("This sign-in has already been handled."))
        if verdict == "error":
            return CallbackResponse(400, oauth_error_html(f"{provider_name} authorization failed.", description))
        if verdict == "missing" or not code:
            return CallbackResponse(400, oauth_error_html("Missing authorization code."))
        try:
            value = await complete(code)
        except Exception as failure:
            callback._finish("error", failure)
            return CallbackResponse(502, oauth_error_html(f"{provider_name} sign-in failed.", str(failure)))
        callback._finish("ok", value)
        return CallbackResponse(200, oauth_success_html(f"Signed in to {provider_name}. You may now close this page."))

    server = await start_callback_server(host=host, port=port, handle=handle)
    callback._server = server

    if cancel is not None:
        callback._watch(cancel.on_cancel(lambda _reason: callback._finish("error", RuntimeError("Login cancelled"))))
    if timeout_ms is not None:

        async def _deadline() -> None:
            if not await callback._result.wait_for(timeout_ms / 1000):
                callback._finish("error", RuntimeError(f"{provider_name} sign-in timed out"))

        tonio.spawn.without_tracking(_deadline())

    host_part = redirect_host if redirect_host is not None else host
    callback.redirect_uri = f"http://{f'[{host_part}]' if ':' in host_part else host_part}:{server.port}{path}"
    return callback


@dataclass(slots=True, frozen=True)
class CallbackOrManualInput[T]:
    """pi's `{ type: "callback"; value } | { type: "manual"; input }`."""

    type: Literal["callback", "manual"]
    value: T | None = None
    input: str = ""


async def wait_for_callback_or_manual_input[T](
    interaction: ProviderAuthInteraction,
    callback: OAuthCallbackServer[T] | None,
    *,
    message: str,
    placeholder: str,
) -> CallbackOrManualInput[T]:
    """Wait for the browser callback, or for the user to paste the code or
    redirect URL when the browser cannot reach the loopback server (for example
    over SSH). Without a callback server only the manual prompt is used."""
    manual_abort = CancelToken()
    manual = OneShotValue()

    async def run_manual_prompt() -> None:
        try:
            manual.settle(
                (
                    "ok",
                    await interaction.prompt(
                        AuthPrompt(type="manual_code", message=message, placeholder=placeholder, cancel=manual_abort)
                    ),
                )
            )
        except Exception as error:
            manual.settle(("error", error))
        if callback is not None:
            callback.cancel()

    tonio.spawn.without_tracking(run_manual_prompt())
    try:
        value = await callback.wait() if callback is not None else None
        if manual.settled:
            outcome, payload = await manual.wait()
            if outcome == "error":
                raise payload
        if value is not None:
            return CallbackOrManualInput(type="callback", value=value)
        outcome, payload = await manual.wait()
        if outcome == "error":
            raise payload
        return CallbackOrManualInput(type="manual", input=payload or "")
    finally:
        manual_abort.cancel()
