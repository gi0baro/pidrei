"""Mirror of pi coding-agent test/auth-login.test.ts.

Not mirrored: the two Radius cases ("keeps unrelated MCP servers when %s is
occupied", "configures the Radius MCP server without asking, once"); Radius
is not a PiDrei provider. pi's `signal` is the command's `cancel` token, and
pi's `vi.spyOn(modelRuntime, ...)` is an instance attribute set on the
runtime. pi's `toMatchObject` on a stored credential checks its type and the
one field the case names.
"""

import pytest
import tonio.colored as tonio

from pidrei.cli.auth_login import AuthLoginOptions, run_auth_login
from pidrei.core.auth_storage import AuthStorage
from pidrei.core.model_runtime import CredentialSynchronizationError, ModelRuntime
from pidrei_ai.auth.types import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthEvent,
    AuthPrompt,
    AuthPromptOption,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
    ProviderAuth,
)
from pidrei_ai.models_store import InMemoryModelsStore
from pidrei_ai.registry import ModelsRefreshOptions, ModelsRefreshResult
from pidrei_ai.types import Model, ModelCost
from pidrei_utils import clock
from pidrei_utils.cancel import CancelToken


PROVIDER_ID = "login-test"
USAGE = "Usage: pidrei auth login <provider> [--api-key] [--no-browser]"


class ScriptedTerminal:
    """Answers prompts in order; `secret` and `question` share one answer queue."""

    def __init__(self, answers: list[str], interactive: bool = True) -> None:
        self.interactive = interactive
        self.answers = list(answers)
        self.lines: list[str] = []
        self.errors: list[str] = []
        self.questions: list[str] = []
        self.opened: list[str] = []

    def write(self, line: str) -> None:
        self.lines.append(line)

    def error(self, line: str) -> None:
        self.errors.append(line)

    async def question(self, message: str, cancel: CancelToken | None) -> str:
        self.questions.append(message)
        if cancel is not None and cancel.cancelled:
            raise Exception("Login cancelled")
        if not self.answers:
            raise Exception(f"Unexpected prompt: {message}")
        return self.answers.pop(0)

    secret = question

    def open_url(self, url: str) -> None:
        self.opened.append(url)


def make_model(provider_id: str) -> Model:
    return Model(
        id="test-model",
        name="test-model",
        api="openai-completions",
        provider=provider_id,
        base_url="https://example.test/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=1000,
        max_tokens=100,
    )


class LoginTestProvider:
    def __init__(self, auth: ProviderAuth, provider_id: str) -> None:
        self.id = provider_id
        self.name = "Login Test"
        self.auth = auth
        self.base_url = None
        self.headers = None
        self.filter_models = None
        self.refresh_models_auth = None
        self._model = make_model(provider_id)

    has_dynamic_models = False

    def get_models(self):
        return [self._model]

    async def refresh_models(self, _context):
        return None

    def stream(self, model, context, options=None):
        raise RuntimeError("unused")

    def stream_simple(self, model, context, options=None):
        raise RuntimeError("unused")


def oauth_credential(access: str) -> OAuthCredential:
    return OAuthCredential(access=access, refresh="refresh", expires=clock.now_ms() + 3_600_000)


async def _same_credential(credential, _cancel):
    return credential


async def _to_auth(credential):
    return ModelAuth(api_key=credential.access)


def oauth_auth(login) -> OAuthAuth:
    return OAuthAuth(name="Login Test account", login=login, refresh=_same_credential, to_auth=_to_auth)


async def _both_methods_oauth_login(interaction, _options=None):
    method = await interaction.prompt(
        AuthPrompt(
            type="select",
            message="Sign in to Login Test:",
            options=[
                AuthPromptOption(id="browser", label="Sign in with browser"),
                AuthPromptOption(id="device", label="Sign in with device code"),
            ],
        )
    )
    if method == "device":
        interaction.notify(
            AuthEvent(type="device_code", user_code="ABCD-1234", verification_uri="https://example.test/device")
        )
        return oauth_credential("access-device")
    interaction.notify(
        AuthEvent(type="auth_url", url="https://example.test/authorize", instructions="Finish in the browser")
    )
    code = await interaction.prompt(AuthPrompt(type="manual_code", message="Paste the code"))
    return oauth_credential(f"access-{code}")


async def _api_key_login(interaction):
    return ApiKeyCredential(key=await interaction.prompt(AuthPrompt(type="secret", message="Enter Login Test API key")))


async def _resolve_nothing(_ctx, _credential, _cancel):
    return None


BOTH_OAUTH = oauth_auth(_both_methods_oauth_login)
BOTH_API_KEY = ApiKeyAuth(name="Login Test API key", login=_api_key_login, resolve=_resolve_nothing)


def both_methods() -> ProviderAuth:
    return ProviderAuth(oauth=BOTH_OAUTH, api_key=BOTH_API_KEY)


async def setup(auth: ProviderAuth, provider_id: str = PROVIDER_ID) -> tuple[AuthStorage, ModelRuntime]:
    credentials = AuthStorage.in_memory()
    model_runtime = await ModelRuntime(
        credentials=credentials,
        models_path=None,
        models_store=InMemoryModelsStore(),
        allow_model_network=False,
    )
    model_runtime.register_native_provider(LoginTestProvider(auth, provider_id))
    # Drain the full refresh `register_native_provider` requested (spec §3).
    await model_runtime.refresh(ModelsRefreshOptions(allow_network=False))
    return credentials, model_runtime


def run(args: list[str], model_runtime: ModelRuntime, terminal, cancel: CancelToken | None = None):
    return run_auth_login(
        args,
        AuthLoginOptions(
            model_runtime=model_runtime,
            terminal=terminal,
            get_device_id=lambda: "device",
            cancel=cancel if cancel is not None else CancelToken(),
        ),
    )


async def assert_stored(credentials: AuthStorage, credential_type: str, **fields) -> None:
    credential = await credentials.read(PROVIDER_ID)
    assert credential is not None and credential.type == credential_type
    for name, value in fields.items():
        assert getattr(credential, name) == value


@pytest.mark.tonio
async def test_requires_exactly_one_provider_and_known_options():
    _, model_runtime = await setup(both_methods())
    for args in ([], ["--api-key"], ["--provider", PROVIDER_ID], [PROVIDER_ID, "extra"]):
        terminal = ScriptedTerminal([])
        assert await run(args, model_runtime, terminal) == 1
        assert terminal.errors == [USAGE]


@pytest.mark.tonio
async def test_rejects_no_browser_with_api_key():
    _, model_runtime = await setup(both_methods())
    terminal = ScriptedTerminal([])
    assert await run([PROVIDER_ID, "--api-key", "--no-browser"], model_runtime, terminal) == 1
    assert terminal.errors == ["--no-browser only applies to account sign-in; omit it with --api-key."]


@pytest.mark.tonio
async def test_rejects_an_unknown_provider_and_lists_the_providers_with_a_login_flow():
    _, model_runtime = await setup(both_methods())
    terminal = ScriptedTerminal([])
    assert await run(["nope"], model_runtime, terminal) == 1
    assert terminal.errors[0] == 'Unknown provider "nope".'
    assert PROVIDER_ID in terminal.errors[1]


@pytest.mark.tonio
async def test_requires_an_interactive_terminal_after_checking_the_arguments():
    _, model_runtime = await setup(both_methods())
    terminal = ScriptedTerminal([], interactive=False)
    assert await run([PROVIDER_ID], model_runtime, terminal) == 1
    assert terminal.errors == [
        '"pidrei auth login <provider> [--api-key] [--no-browser]" requires an interactive terminal'
    ]


@pytest.mark.tonio
async def test_signs_in_with_an_account_in_the_browser_without_asking_for_a_method():
    credentials, model_runtime = await setup(both_methods())
    terminal = ScriptedTerminal(["the-code"])
    # Matches the provider name as well as the id.
    assert await run(["login test"], model_runtime, terminal) == 0
    assert terminal.questions == ["Paste the code: "]
    assert "Open this URL to sign in:\n  https://example.test/authorize" in terminal.lines
    assert "Finish in the browser" in terminal.lines
    assert terminal.opened == ["https://example.test/authorize"]
    assert (
        'To use "Sign in with device code" instead, run "pidrei auth login login-test --no-browser"' in terminal.lines
    )
    await assert_stored(credentials, "oauth", access="access-the-code")
    assert any(line.startswith("Logged in to Login Test.") for line in terminal.lines)


@pytest.mark.tonio
async def test_lists_several_other_sign_in_methods():
    async def login(interaction, _options=None):
        await interaction.prompt(
            AuthPrompt(
                type="select",
                message="Sign in to Login Test:",
                options=[
                    AuthPromptOption(id="browser", label="Browser"),
                    AuthPromptOption(id="device", label="Device code"),
                    AuthPromptOption(id="copy", label="Copy code"),
                ],
            )
        )
        interaction.notify(AuthEvent(type="auth_url", url="https://example.test/authorize"))
        return oauth_credential("access")

    _, model_runtime = await setup(ProviderAuth(oauth=oauth_auth(login)))
    terminal = ScriptedTerminal([])
    assert await run([PROVIDER_ID], model_runtime, terminal) == 0
    assert (
        'Other sign-in methods: "Device code", "Copy code". Run "pidrei auth login login-test --no-browser" to choose one.'
        in terminal.lines
    )


@pytest.mark.tonio
async def test_offers_no_other_sign_in_methods_when_the_provider_has_no_method_choice():
    async def login(interaction, _options=None):
        interaction.notify(AuthEvent(type="auth_url", url="https://example.test/authorize"))
        code = await interaction.prompt(AuthPrompt(type="manual_code", message="Paste the code"))
        return oauth_credential(f"access-{code}")

    credentials, model_runtime = await setup(ProviderAuth(oauth=oauth_auth(login)))
    terminal = ScriptedTerminal(["the-code"])
    assert await run([PROVIDER_ID], model_runtime, terminal) == 0
    assert terminal.opened == ["https://example.test/authorize"]
    assert not any("--no-browser" in line for line in terminal.lines)
    await assert_stored(credentials, "oauth", access="access-the-code")


@pytest.mark.tonio
async def test_no_browser_takes_the_only_other_sign_in_method_and_opens_nothing():
    credentials, model_runtime = await setup(both_methods())
    terminal = ScriptedTerminal([])
    assert await run([PROVIDER_ID, "--no-browser"], model_runtime, terminal) == 0
    assert terminal.questions == []
    assert terminal.lines[0] == "Sign in to Login Test: Sign in with device code"
    assert "Open https://example.test/device and enter code: ABCD-1234" in terminal.lines
    assert terminal.opened == []
    assert not any("--no-browser" in line for line in terminal.lines)
    await assert_stored(credentials, "oauth", access="access-device")


def three_method_login():
    async def login(interaction, _options=None):
        method = await interaction.prompt(
            AuthPrompt(
                type="select",
                message="Sign in to Login Test:",
                options=[
                    AuthPromptOption(id="browser", label="Browser"),
                    AuthPromptOption(id="device", label="Device code"),
                    AuthPromptOption(id="copy", label="Copy code"),
                ],
            )
        )
        return oauth_credential(f"access-{method}")

    return login


@pytest.mark.tonio
async def test_no_browser_asks_among_several_other_sign_in_methods_and_re_asks_on_an_invalid_selection():
    credentials, model_runtime = await setup(ProviderAuth(oauth=oauth_auth(three_method_login())))
    terminal = ScriptedTerminal(["7", "2"])
    assert await run(["--no-browser", PROVIDER_ID], model_runtime, terminal) == 0
    assert terminal.lines[0:3] == ["Sign in to Login Test:", "  1. Device code", "  2. Copy code"]
    assert "Enter a number from 1 to 2." in terminal.lines
    await assert_stored(credentials, "oauth", access="access-copy")


@pytest.mark.tonio
async def test_no_browser_keeps_a_providers_only_sign_in_method_without_opening_the_browser():
    async def login(interaction, _options=None):
        method = await interaction.prompt(
            AuthPrompt(
                type="select",
                message="Sign in to Login Test:",
                options=[AuthPromptOption(id="browser", label="Browser")],
            )
        )
        interaction.notify(AuthEvent(type="auth_url", url="https://example.test/authorize"))
        return oauth_credential(f"access-{method}")

    credentials, model_runtime = await setup(ProviderAuth(oauth=oauth_auth(login)))
    terminal = ScriptedTerminal([])
    assert await run([PROVIDER_ID, "--no-browser"], model_runtime, terminal) == 0
    assert terminal.questions == []
    assert terminal.opened == []
    await assert_stored(credentials, "oauth", access="access-browser")


@pytest.mark.tonio
async def test_asks_for_selections_during_api_key_setup():
    async def login(interaction):
        return ApiKeyCredential(
            key=await interaction.prompt(
                AuthPrompt(
                    type="select",
                    message="Choose credentials:",
                    options=[
                        AuthPromptOption(id="first", label="First"),
                        AuthPromptOption(id="second", label="Second", description="Other credentials"),
                    ],
                )
            )
        )

    credentials, model_runtime = await setup(
        ProviderAuth(api_key=ApiKeyAuth(name="Login Test API key", login=login, resolve=_resolve_nothing))
    )
    terminal = ScriptedTerminal(["2"])
    assert await run([PROVIDER_ID, "--api-key"], model_runtime, terminal) == 0
    assert terminal.questions == ["Select 1-2: "]
    assert "  2. Second - Other credentials" in terminal.lines
    await assert_stored(credentials, "api_key", key="second")


@pytest.mark.tonio
async def test_cancels_a_pending_prompt_without_cancelling_the_login_when_the_browser_callback_wins():
    prompt_cancel = CancelToken()
    question_asked = tonio.Event()
    observed: dict = {}

    async def login(interaction, _options=None):
        # pi's prompt runs up to its first await when called; here it is spawned
        # and cancelled once the terminal question is pending.
        pending = tonio.spawn(
            interaction.prompt(
                AuthPrompt(
                    type="manual_code", message="Paste the code", placeholder="redirect URL", cancel=prompt_cancel
                )
            )
        )
        await question_asked.wait(5)
        prompt_cancel.cancel()
        try:
            await pending
        except* Exception as group:
            observed["error"] = group.exceptions[0]
        observed["login_cancelled"] = interaction.cancel.cancelled
        return oauth_credential("callback")

    credentials, model_runtime = await setup(ProviderAuth(oauth=oauth_auth(login)))
    terminal = ScriptedTerminal([])
    calls: list[tuple[str, CancelToken | None]] = []

    async def question(message: str, cancel: CancelToken | None) -> str:
        # Parks until its token fires, then raises the token's reason.
        calls.append((message, cancel))
        question_asked.set()
        await cancel.wait(5)
        assert cancel.cancelled
        raise cancel.reason

    terminal.question = question
    assert await run([PROVIDER_ID], model_runtime, terminal) == 0
    assert [message for message, _ in calls] == ["Paste the code (redirect URL): "]
    assert isinstance(calls[0][1], CancelToken)
    assert observed["error"] is prompt_cancel.reason
    assert observed["login_cancelled"] is False
    await assert_stored(credentials, "oauth", access="callback")


@pytest.mark.tonio
async def test_reports_a_committed_login_separately_from_a_failed_login():
    _, model_runtime = await setup(both_methods())

    async def login(*_args, **_kwargs):
        raise CredentialSynchronizationError(
            PROVIDER_ID, "login", ApiKeyCredential(key="saved"), cause=Exception("snapshot failed")
        )

    model_runtime.login = login
    terminal = ScriptedTerminal([])
    assert await run([PROVIDER_ID], model_runtime, terminal) == 1
    assert "Logged in to Login Test, but local model state could not be synchronized:" in terminal.errors[0]


@pytest.mark.tonio
async def test_api_key_saves_an_api_key():
    credentials, model_runtime = await setup(both_methods())
    terminal = ScriptedTerminal(["sk-test"])
    assert await run([PROVIDER_ID, "--api-key"], model_runtime, terminal) == 0
    assert terminal.questions == ["Enter Login Test API key: "]
    await assert_stored(credentials, "api_key", key="sk-test")
    assert any(line.startswith("Saved API key for Login Test.") for line in terminal.lines)


@pytest.mark.tonio
async def test_uses_the_api_key_login_of_providers_without_account_sign_in():
    _, model_runtime = await setup(ProviderAuth(api_key=BOTH_API_KEY))
    terminal = ScriptedTerminal(["sk-test"])
    assert await run([PROVIDER_ID], model_runtime, terminal) == 0
    assert terminal.questions == ["Enter Login Test API key: "]


@pytest.mark.tonio
async def test_api_key_fails_for_providers_without_api_key_login():
    _, model_runtime = await setup(ProviderAuth(oauth=BOTH_OAUTH))
    terminal = ScriptedTerminal([])
    assert await run([PROVIDER_ID, "--api-key"], model_runtime, terminal) == 1
    assert terminal.errors == ["Login Test has no API key login. Omit --api-key to sign in."]


@pytest.mark.tonio
async def test_reports_providers_configured_outside_pidrei():
    _, model_runtime = await setup(ProviderAuth(api_key=ApiKeyAuth(name="AWS profile", resolve=_resolve_nothing)))
    terminal = ScriptedTerminal([])
    assert await run([PROVIDER_ID], model_runtime, terminal) == 1
    assert terminal.errors == ["Login Test is configured outside pidrei (AWS profile)."]


@pytest.mark.tonio
async def test_cancelling_during_the_catalog_refresh_skips_it_and_keeps_the_login():
    credentials, model_runtime = await setup(ProviderAuth(api_key=BOTH_API_KEY))
    cancel = CancelToken()

    # A refresh that only ends when cancelled, interrupted as soon as it starts.
    async def refresh(options=None, **_kwargs):
        cancel.cancel()
        await options.cancel.wait(5)
        return ModelsRefreshResult(aborted=options.cancel.cancelled, errors={})

    model_runtime.refresh = refresh
    terminal = ScriptedTerminal(["sk-test"])
    assert await run([PROVIDER_ID], model_runtime, terminal, cancel) == 0
    assert "Skipped the Login Test model catalog refresh; pidrei refreshes it on next start." in terminal.lines
    await assert_stored(credentials, "api_key", key="sk-test")


@pytest.mark.tonio
async def test_reports_cancellation():
    credentials, model_runtime = await setup(ProviderAuth(api_key=BOTH_API_KEY))
    cancel = CancelToken()
    cancel.cancel()
    terminal = ScriptedTerminal(["sk-test"])
    assert await run([PROVIDER_ID], model_runtime, terminal, cancel) == 1
    assert terminal.errors == ["Login cancelled."]
    assert await credentials.read(PROVIDER_ID) is None
