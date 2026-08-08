"""Mirror of pi coding-agent test/suite/regressions/7027-credential-refresh-hang.test.ts.

pi drives the bounded background refresh with fake timers; pidrei's timeout
seam is `interactive_mode._TimeoutCancel`, replaced with a manually-fired
fake for the second case.
"""

from types import SimpleNamespace
from typing import ClassVar

import pytest
import tonio.colored as tonio

import pidrei.modes.interactive.interactive_mode as interactive_mode_module
from pidrei.core.auth_storage import AuthStorage
from pidrei.core.model_runtime import ModelRuntime
from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei_ai.auth.types import ApiKeyAuth, ApiKeyCredential, AuthCheck, AuthResult, ModelAuth, ProviderAuth
from pidrei_ai.registry import ModelsRefreshOptions, ModelsRefreshResult
from pidrei_ai.types import Model, ModelCost
from pidrei_ai.utils.cancel import CancelToken

from .harness import create_harness


DYNAMIC_MODEL = Model(
    id="dynamic",
    name="Dynamic",
    api="openai-completions",
    provider="stalled-login",
    base_url="https://example.test/v1",
    reasoning=False,
    input=["text"],
    cost=ModelCost(),
    context_window=1000,
    max_tokens=100,
)


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


class StalledLoginProvider:
    def __init__(self, network_started: tonio.Event, network_release: tonio.Event):
        self.id = "stalled-login"
        self.name = "Stalled Login"
        self.base_url = None
        self.headers = None
        self.filter_models = None
        self._network_started = network_started
        self._network_release = network_release

        async def login(_interaction):
            return ApiKeyCredential(key="secret")

        async def check(_ctx, credential, _cancel):
            if credential is not None and credential.key:
                return AuthCheck(type="api_key", source="stored key")
            return None

        async def resolve(_ctx, credential, _cancel):
            key = credential.key if credential is not None and credential.key else "ambient-key"
            return AuthResult(
                auth=ModelAuth(api_key=key),
                source="stored key" if credential is not None and credential.key else "ambient key",
            )

        self.auth = ProviderAuth(api_key=ApiKeyAuth(name="API key", login=login, check=check, resolve=resolve))

    @property
    def has_dynamic_models(self) -> bool:
        return True

    def get_models(self):
        return [DYNAMIC_MODEL]

    async def refresh_models(self, context):
        if not context.allow_network:
            return
        self._network_started.set()
        await self._network_release.wait()

    def stream(self, model, context, options=None):
        raise RuntimeError("unused")

    def stream_simple(self, model, context, options=None):
        raise RuntimeError("unused")


class Interaction:
    cancel = None

    async def prompt(self, prompt):
        return "unused"

    def notify(self, event):
        pass


@pytest.mark.tonio
async def test_does_not_hold_login_behind_an_older_stalled_network_catalog_refresh():
    network_started = tonio.Event()
    network_release = tonio.Event()
    provider = StalledLoginProvider(network_started, network_release)
    credentials = AuthStorage.in_memory()
    runtime = await ModelRuntime.create(credentials=credentials, models_path=None, allow_model_network=False)
    runtime.register_native_provider(provider)
    await runtime.refresh(ModelsRefreshOptions(allow_network=False, providers=[provider.id]))

    stalled_outcome: dict = {}

    async def run_stalled_refresh() -> None:
        stalled_outcome["result"] = await runtime.refresh(
            ModelsRefreshOptions(allow_network=True, providers=[provider.id])
        )

    async def drive() -> None:
        await network_started.wait()
        credential = await runtime.login(provider.id, "api_key", Interaction())
        assert credential == ApiKeyCredential(key="secret")

        assert DYNAMIC_MODEL.id in [model.id for model in runtime.get_available_snapshot()]
        assert await credentials.read(provider.id) == ApiKeyCredential(key="secret")
        network_release.set()

    await tonio.spawn(run_stalled_refresh(), drive())
    assert stalled_outcome["result"].aborted is False


@pytest.mark.tonio
async def test_completes_interactive_login_before_its_bounded_background_refresh(harnesses):
    harness = await create_harness()
    harnesses.append(harness)
    runtime = harness.session.model_runtime
    refresh_options: list = []

    async def bounded_refresh(options=None):
        refresh_options.append(options)
        if options is None or options.cancel is None:
            # Leftover harness `_request_refresh` drain — not the bounded call.
            return ModelsRefreshResult(aborted=False, errors={})
        gate = tonio.Event()
        options.cancel.on_cancel(lambda _reason: gate.set())
        await gate.wait()
        return ModelsRefreshResult(aborted=True, errors={})

    runtime.refresh = bounded_refresh

    class FakeTimeout:
        instances: ClassVar[list] = []

        def __init__(self, _ms):
            self.token = CancelToken()
            self.timed_out = False
            FakeTimeout.instances.append(self)

    warning_calls: list = []

    async def noop_async(*_args):
        return None

    context = SimpleNamespace(
        session=harness.session,
        _update_available_provider_count=lambda: None,
        _footer=SimpleNamespace(invalidate=lambda: None),
        _update_editor_border_color=lambda: None,
        show_status=lambda message: None,
        show_error=lambda message: None,
        show_warning=warning_calls.append,
        _maybe_warn_about_anthropic_subscription_auth=noop_async,
        _check_daxnuts_easter_egg=lambda model: None,
        ui=SimpleNamespace(request_render=lambda force=False: None),
    )

    original_timeout = interactive_mode_module._TimeoutCancel
    interactive_mode_module._TimeoutCancel = FakeTimeout
    try:
        await InteractiveMode._complete_provider_authentication(
            context, DYNAMIC_MODEL.provider, "Stalled Login", "api_key", harness.get_model()
        )
        await tonio.time.sleep(0.01)
        scoped = [options for options in refresh_options if options is not None and options.providers]
        assert len(scoped) == 1
        assert scoped[0].providers == [DYNAMIC_MODEL.provider]
        assert isinstance(scoped[0].cancel, CancelToken)
        assert warning_calls == []

        timeout = FakeTimeout.instances[-1]
        timeout.timed_out = True
        timeout.token.cancel(TimeoutError("The operation timed out."))
        await tonio.time.sleep(0.01)
        assert warning_calls == [
            "Saved API key for Stalled Login, but its model catalog refresh timed out; using cached models."
        ]
    finally:
        interactive_mode_module._TimeoutCancel = original_timeout
