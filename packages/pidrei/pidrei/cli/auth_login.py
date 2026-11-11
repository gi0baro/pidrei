"""Port of pi's cli/auth-login.ts: `pidrei auth login <provider>`.

Not ported: the Radius MCP server setup after a Radius login (Radius is not a
PiDrei provider), and with it the command's `agentDir`/`cwd` options. pi
aborts the login on SIGINT; PiDrei handles no SIGINT (see `auth_terminal`), so
`cancel` is fired by the terminal's interrupt callback alone.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pidrei_ai.auth.oauth.urls import trusted_http_url
from pidrei_ai.auth.types import AuthEvent, AuthPrompt, AuthPromptOption, LoginOptions
from pidrei_ai.registry import ModelsRefreshOptions, Provider
from pidrei_tui import replace_control_characters
from pidrei_utils.cancel import CancelToken, combine_cancel_tokens
from pidrei_utils.timers import Timeout

from ..config import APP_NAME, get_auth_path
from ..core.model_runtime import CredentialSynchronizationError, ModelRuntime
from ..core.provider_matching import find_matching_providers
from ..utils.web_url import display_url
from .auth_terminal import LoginTerminal


@dataclass(slots=True)
class AuthLoginOptions:
    """Dependencies of `run_auth_login`, injected so tests can script the terminal."""

    model_runtime: ModelRuntime
    terminal: LoginTerminal
    get_device_id: Callable[[], str]
    # Cancels the login, e.g. on Ctrl+C in a secret prompt.
    cancel: CancelToken


AUTH_LOGIN_USAGE = f"{APP_NAME} auth login <provider> [--api-key] [--no-browser]"

_SELECTION = re.compile(r"[0-9]+")


@dataclass(slots=True, frozen=True)
class _LoginRequest:
    """Parsed `pidrei auth login` arguments."""

    provider_ref: str
    # Log in with an API key instead of an account.
    api_key: bool
    # Open the browser and take the browser sign-in method without asking.
    browser: bool


def provider_names(provider: Provider) -> tuple[str, str]:
    return provider.id, provider.name


async def run_auth_login(args: list[str], options: AuthLoginOptions) -> int:
    """Run `pidrei auth login <provider>` in the terminal and return the process exit code."""
    model_runtime, terminal, cancel = options.model_runtime, options.terminal, options.cancel
    request = _parse_login_request(args)
    if isinstance(request, str):
        terminal.error(request)
        return 1
    provider_ref = request.provider_ref

    matches = find_matching_providers(model_runtime.get_providers(), provider_ref, provider_names)
    provider = matches[0] if matches else None
    if provider is None:
        login_providers = sorted(
            candidate.id
            for candidate in model_runtime.get_providers()
            if candidate.auth.oauth or (candidate.auth.api_key is not None and candidate.auth.api_key.login)
        )
        terminal.error(f'Unknown provider "{replace_control_characters(provider_ref)}".')
        terminal.error(f"Providers: {', '.join(login_providers)}")
        return 1
    if not terminal.interactive:
        terminal.error(f'"{AUTH_LOGIN_USAGE}" requires an interactive terminal')
        return 1

    auth_type, unavailable = _select_auth_type(provider, request.api_key)
    if auth_type is None:
        terminal.error(unavailable)
        return 1
    try:
        await model_runtime.login(
            provider.id,
            auth_type,
            _LoginInteraction(provider.id, auth_type, request, terminal, cancel),
            LoginOptions(get_device_id=options.get_device_id),
        )
    except Exception as error:
        message = str(error)
        if isinstance(error, CredentialSynchronizationError):
            terminal.error(f"Logged in to {provider.name}, but local model state could not be synchronized: {message}")
        elif cancel.cancelled or message == "Login cancelled":
            terminal.error("Login cancelled.")
        else:
            terminal.error(f"Failed to log in to {provider.name}: {message}")
        return 1

    action = f"Logged in to {provider.name}" if auth_type == "oauth" else f"Saved API key for {provider.name}"
    terminal.write(f"{action}. Credentials saved to {get_auth_path()}")
    await _refresh_provider_catalog(provider, model_runtime, terminal, cancel)
    return 0


def _parse_login_request(args: list[str]) -> _LoginRequest | str:
    """The request, or the message to print for invalid arguments."""
    usage = f"Usage: {AUTH_LOGIN_USAGE}"
    provider_ref = ""
    api_key = False
    browser = True
    for arg in args:
        if arg == "--api-key":
            api_key = True
        elif arg == "--no-browser":
            browser = False
        elif arg.startswith("-") or provider_ref:
            return usage
        else:
            provider_ref = arg
    if not provider_ref:
        return usage
    if api_key and not browser:
        return "--no-browser only applies to account sign-in; omit it with --api-key."
    return _LoginRequest(provider_ref=provider_ref, api_key=api_key, browser=browser)


def _select_auth_type(provider: Provider, api_key: bool) -> tuple[str | None, str]:
    """Account login unless `api_key` asks for an API key. Providers without account login use their API key login,
    since that is the only way in. Returns the auth type, or None and the message saying why there is none."""
    has_api_key_login = provider.auth.api_key is not None and provider.auth.api_key.login is not None
    if not api_key and provider.auth.oauth:
        return "oauth", ""
    if has_api_key_login:
        return "api_key", ""
    if api_key and provider.auth.oauth:
        return None, f"{provider.name} has no API key login. Omit --api-key to sign in."
    method = provider.auth.api_key.name if provider.auth.api_key is not None else "ambient credentials"
    return None, f"{provider.name} is configured outside {APP_NAME} ({method})."


class _LoginInteraction:
    """Adapt provider prompts and events to the command's selected sign-in method."""

    def __init__(
        self, provider_id: str, auth_type: str, request: _LoginRequest, terminal: LoginTerminal, cancel: CancelToken
    ) -> None:
        self.cancel = cancel
        self._provider_id = provider_id
        self._request = request
        self._terminal = terminal
        # Account logins list the browser sign-in first, as /login; API key setup asks every selection.
        self._method_choice = "ask" if auth_type == "api_key" else "browser" if request.browser else "other"
        # Rebound whole: the flow's notify can run on another coroutine than its select prompt.
        self._skipped_methods: tuple[str, ...] = ()

    async def prompt(self, prompt: AuthPrompt) -> str:
        # The browser callback can finish a step before the pasted code, cancelling just that prompt.
        combined = combine_cancel_tokens(self.cancel, prompt.cancel)
        try:
            if prompt.type == "select":
                return await self._select_method(prompt, combined.token)
            placeholder = f" ({prompt.placeholder})" if prompt.placeholder else ""
            message = f"{prompt.message}{placeholder}: "
            if prompt.type == "secret":
                return await self._terminal.secret(message, combined.token)
            return await self._terminal.question(message, combined.token)
        finally:
            combined.cleanup()

    async def _select_method(self, prompt: AuthPrompt, cancel: CancelToken | None) -> str:
        options = prompt.options or []
        if self._method_choice == "ask" or not options:
            return await _select_in_terminal(prompt.message, options, self._terminal, cancel)
        first, *others = options
        if self._method_choice == "browser":
            self._skipped_methods = tuple(option.label for option in others)
            return first.id
        # A provider whose only method is the browser sign-in still gets it, without opening the browser.
        if not others:
            return first.id
        if len(others) > 1:
            return await _select_in_terminal(prompt.message, others, self._terminal, cancel)
        only_other = others[0]
        self._terminal.write(f"{prompt.message} {only_other.label}")
        return only_other.id

    def notify(self, event: AuthEvent) -> None:
        terminal = self._terminal
        if event.type == "auth_url":
            terminal.write(f"Open this URL to sign in:\n  {display_url(event.url)}")
            if event.instructions:
                terminal.write(event.instructions)
            target = trusted_http_url(event.url)
            if target is not None and self._request.browser:
                terminal.open_url(target)
            # Providers without a method choice have nothing else to offer, so they get no hint.
            rerun = f'"{APP_NAME} auth login {self._provider_id} --no-browser"'
            skipped = self._skipped_methods
            if len(skipped) == 1:
                terminal.write(f'To use "{skipped[0]}" instead, run {rerun}')
            elif skipped:
                labels = ", ".join(f'"{label}"' for label in skipped)
                terminal.write(f"Other sign-in methods: {labels}. Run {rerun} to choose one.")
        elif event.type == "device_code":
            terminal.write(f"Open {display_url(event.verification_uri)} and enter code: {event.user_code}")
            target = trusted_http_url(event.verification_uri)
            if target is not None and self._request.browser and event.open_browser_on_confirm is not False:
                terminal.open_url(target)
            terminal.write("Waiting for authentication...")
        elif event.type == "info":
            terminal.write(event.message)
            for link in event.links or []:
                label = f"{link.label}: " if link.label else ""
                terminal.write(f"  {label}{display_url(link.url)}")
        else:
            terminal.write(event.message)


async def _select_in_terminal(
    message: str, options: Sequence[AuthPromptOption], terminal: LoginTerminal, cancel: CancelToken | None
) -> str:
    terminal.write(message)
    for index, option in enumerate(options):
        description = f" - {option.description}" if option.description else ""
        terminal.write(f"  {index + 1}. {option.label}{description}")
    while True:
        answer = (await terminal.question(f"Select 1-{len(options)}: ", cancel)).strip()
        number = int(answer) if _SELECTION.fullmatch(answer) else 0
        if 1 <= number <= len(options):
            return options[number - 1].id
        terminal.write(f"Enter a number from 1 to {len(options)}.")


async def _refresh_provider_catalog(
    provider: Provider, model_runtime: ModelRuntime, terminal: LoginTerminal, cancel: CancelToken
) -> None:
    """Fetch the provider's model catalog now, since dynamic catalogs stay empty until the first authenticated
    refresh."""

    # The login is saved by now, so cancelling only skips the refresh.
    def cancelled() -> None:
        terminal.write(f"Skipped the {provider.name} model catalog refresh; {APP_NAME} refreshes it on next start.")

    timeout = CancelToken()
    timer = Timeout(15_000, timeout.cancel)
    combined = combine_cancel_tokens(cancel, timeout)
    try:
        # No `allow_network`: the runtime applies its own network switch (off in offline mode).
        result = await model_runtime.refresh(ModelsRefreshOptions(providers=[provider.id], cancel=combined.token))
        if cancel.cancelled:
            cancelled()
        elif result.aborted:
            terminal.write(
                f"The {provider.name} model catalog refresh timed out; {APP_NAME} refreshes it on next start."
            )
        elif result.errors:
            terminal.write(
                f"The {provider.name} model catalog could not be refreshed; {APP_NAME} retries on next start."
            )
    except Exception as error:
        if cancel.cancelled:
            cancelled()
            return
        terminal.write(f"The {provider.name} model catalog could not be refreshed: {error}")
    finally:
        timer.cancel()
        combined.cleanup()
