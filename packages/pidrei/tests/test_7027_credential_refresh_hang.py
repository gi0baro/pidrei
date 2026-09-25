"""Mirror of pi coding-agent test/suite/regressions/7027-credential-refresh-hang.test.ts.

pi drives the bounded background refresh with fake timers; pidrei's timeout
seam is `interactive_mode._TimeoutCancel`, replaced with a manually-fired
fake for the second case.
"""

import contextlib
from dataclasses import replace
from types import SimpleNamespace
from typing import ClassVar

import pytest
import tonio.colored as tonio
from tonio.colored import sync

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
    # Drain the detached full refresh `register_native_provider` requested
    # before racing anything against it (pidrei-only; pi's `void refresh` is
    # ordered by the single loop). Left running, its provider rebuild can
    # supersede the stalled refresh below between the cached and the network
    # phase: that refresh then returns `aborted=False` without ever calling
    # `refresh_models(allow_network=True)`, `network_started` never fires and
    # `drive()` waits forever — the macOS CI hang of 2026-09-05.
    await runtime.refresh(ModelsRefreshOptions(allow_network=False))
    await runtime.refresh(ModelsRefreshOptions(allow_network=False, providers=[provider.id]))

    stalled_outcome: dict = {}

    async def run_stalled_refresh() -> None:
        stalled_outcome["result"] = await runtime.refresh(
            ModelsRefreshOptions(allow_network=True, providers=[provider.id])
        )

    async def drive() -> None:
        await network_started.wait(10)
        assert network_started.is_set(), "the stalled refresh never reached its network phase"
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
    parked = tonio.Event()
    refreshed = tonio.Event()

    async def bounded_refresh(options=None, *, _requested_only=False):
        refresh_options.append(options)
        if options is None or options.cancel is None:
            # Leftover harness `_request_refresh` drain — not the bounded call.
            return ModelsRefreshResult(aborted=False, errors={})
        gate = tonio.Event()
        options.cancel.on_cancel(lambda _reason: gate.set())
        parked.set()
        await gate.wait(5)
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
        # The refresh continuation ends with the render request (the login flow's
        # own UI updates here are stubbed and do not render).
        ui=SimpleNamespace(request_render=lambda force=False: refreshed.set(), post_ui=lambda fn: fn()),
    )

    original_timeout = interactive_mode_module._TimeoutCancel
    interactive_mode_module._TimeoutCancel = FakeTimeout
    try:
        await InteractiveMode._complete_provider_authentication(
            context, DYNAMIC_MODEL.provider, "Stalled Login", "api_key", harness.get_model()
        )
        # The detached catalog refresh is parked on its cancel token.
        await parked.wait(5)
        assert parked.is_set()
        scoped = [options for options in refresh_options if options is not None and options.providers]
        assert len(scoped) == 1
        assert scoped[0].providers == [DYNAMIC_MODEL.provider]
        assert isinstance(scoped[0].cancel, CancelToken)
        assert warning_calls == []

        timeout = FakeTimeout.instances[-1]
        timeout.timed_out = True
        timeout.token.cancel(TimeoutError("The operation timed out."))
        await refreshed.wait(5)
        assert refreshed.is_set()
        assert warning_calls == [
            "Saved API key for Stalled Login, but its model catalog refresh timed out; using cached models."
        ]
    finally:
        interactive_mode_module._TimeoutCancel = original_timeout


@pytest.mark.tonio
async def test_a_credential_pass_overtaken_by_a_later_one_still_returns_with_its_state_live():
    """pidrei-specific. pi drops the older of two overlapping provider
    availability passes by seq, which on one loop can only ever drop the one
    that started first. On this runtime the login's own pass can bump *after*
    the unawaited tail of a refresh its recompose just cancelled, get dropped,
    and `login()` returns before anything reflecting the credential is
    published. Both passes are gated inside their catalog read so the first
    is still reading when the second starts; whichever order they run in,
    each must have its state live when it returns."""
    provider = StalledLoginProvider(tonio.Event(), tonio.Event())
    credentials = AuthStorage.in_memory()
    runtime = await ModelRuntime.create(credentials=credentials, models_path=None, allow_model_network=False)
    runtime.register_native_provider(provider)
    # Drain the detached refresh `register_native_provider` requested, so no
    # third pass can publish the credential behind the two under test.
    await runtime.refresh(ModelsRefreshOptions(allow_network=False))
    assert DYNAMIC_MODEL.id not in [model.id for model in runtime.get_available_snapshot()]

    async def _store(_current):
        return ApiKeyCredential(key="secret")

    await credentials.modify(provider.id, _store)

    gates = [tonio.Event(), tonio.Event()]
    entered = [tonio.Event(), tonio.Event()]
    # Set once the second pass can go no further while the first is held: it
    # is either parked in its own gated read, or (with per-provider
    # serialization) waiting for the lock the first pass holds.
    second_parked = tonio.Event()
    calls = 0
    original_get_available = runtime._models.get_available

    async def gated_get_available(provider_id=None, options=None):
        nonlocal calls
        index = min(calls, len(gates) - 1)
        calls += 1
        entered[index].set()
        if index == 1:
            second_parked.set()
        await gates[index].wait()
        return await original_get_available(provider_id, options)

    runtime._models.get_available = gated_get_available

    class ObservedLock:
        def __init__(self) -> None:
            self._inner = sync.Lock()
            self._attempts = 0

        async def __aenter__(self):
            self._attempts += 1
            if self._attempts == 2:
                second_parked.set()
            return await self._inner.__aenter__()

        async def __aexit__(self, *args):
            return await self._inner.__aexit__(*args)

    runtime._provider_availability_locks = {provider.id: ObservedLock()}

    async def run_pass(returned: tonio.Event) -> None:
        await runtime._refresh_provider_availability(provider.id, CancelToken())
        live = [model.id for model in runtime.get_available_snapshot()]
        returned.set()
        assert DYNAMIC_MODEL.id in live, "the pass returned before a snapshot with its credential state was published"

    first_returned = tonio.Event()
    second_returned = tonio.Event()

    async def drive() -> None:
        # The second pass starts only once the first is held in its read, so
        # it is the newer of the two by construction; its own read stays held
        # until the first has returned, so nothing else can publish for it.
        await entered[0].wait()
        async with tonio.scope() as scope:
            scope.spawn(run_pass(second_returned))
            await second_parked.wait()
            gates[0].set()
            await first_returned.wait()
            gates[1].set()

    await tonio.spawn(run_pass(first_returned), drive())


# -- post-login model discovery ------------------------------------------------
#
# pi's cases log in to Radius, whose catalog is empty until the first
# authenticated refresh; pidrei has no Radius provider, so the same deferral is
# driven through `openai` with an empty snapshot. pi's second "selects" case
# ("fast" from ["fast", "powerful"]) is the Radius-only catalog-order fallback
# and is not mirrored. The session is a stub: the flow only touches its
# `model`, `model_runtime` and `set_model`.

_POST_LOGIN_PROVIDER = "openai"
_POST_LOGIN_DEFAULT = interactive_mode_module.DEFAULT_MODEL_PER_PROVIDER[_POST_LOGIN_PROVIDER]


async def _start_login():
    unknown_model = replace(DYNAMIC_MODEL, id="unknown", provider="unknown", api="unknown")
    # Released by `discover` (pi's `finishRefresh`) or by the timeout token.
    # Created up front: the detached refresh task may not have started yet
    # when `discover` runs.
    refresh_gate = tonio.Event()
    refreshed = tonio.Event()
    available: list = []
    set_model_calls: list = []

    async def refresh(options=None):
        options.cancel.on_cancel(lambda _reason: refresh_gate.set())
        await refresh_gate.wait()
        return ModelsRefreshResult(aborted=options.cancel.cancelled, errors={})

    async def set_model(model, persist=False):
        set_model_calls.append((model, persist))

    session = SimpleNamespace(
        model=unknown_model,
        model_runtime=SimpleNamespace(get_available_snapshot=lambda: list(available), refresh=refresh),
        set_model=set_model,
    )

    async def noop_async(*_args):
        return None

    context = SimpleNamespace(
        session=session,
        status_calls=[],
        error_calls=[],
        warning_calls=[],
        _update_available_provider_count=lambda: None,
        _footer=SimpleNamespace(invalidate=lambda: None),
        _update_editor_border_color=lambda: None,
        _maybe_warn_about_anthropic_subscription_auth=noop_async,
        _check_daxnuts_easter_egg=lambda model: None,
        # The refresh continuation ends with the render request. No UI owner here:
        # posted UI updates apply inline.
        ui=SimpleNamespace(request_render=lambda force=False: refreshed.set(), post_ui=lambda fn: fn()),
    )
    context.show_status = context.status_calls.append
    context.show_error = context.error_calls.append
    context.show_warning = context.warning_calls.append

    await InteractiveMode._complete_provider_authentication(
        context, _POST_LOGIN_PROVIDER, "OpenAI", "oauth", unknown_model
    )
    assert any("Credentials saved" in message for message in context.status_calls)
    assert context.error_calls == []
    assert set_model_calls == []

    async def discover(ids: list[str]) -> None:
        available[:] = [replace(DYNAMIC_MODEL, provider=_POST_LOGIN_PROVIDER, id=model_id) for model_id in ids]
        refresh_gate.set()
        await refreshed.wait(5)
        assert refreshed.is_set(), "the refresh continuation never ran"

    return SimpleNamespace(context=context, session=session, set_model_calls=set_model_calls, discover=discover)


class _ManualTimeout:
    instances: ClassVar[list] = []

    def __init__(self, _ms):
        self.token = CancelToken()
        self.timed_out = False
        _ManualTimeout.instances.append(self)


@contextlib.contextmanager
def _manual_timeout():
    _ManualTimeout.instances = []
    original = interactive_mode_module._TimeoutCancel
    interactive_mode_module._TimeoutCancel = _ManualTimeout
    try:
        yield _ManualTimeout.instances
    finally:
        interactive_mode_module._TimeoutCancel = original


@pytest.mark.tonio
async def test_selects_the_default_model_from_the_refreshed_catalog():
    with _manual_timeout():
        login = await _start_login()
        await login.discover(["fast", _POST_LOGIN_DEFAULT])

    assert [(model.provider, model.id, persist) for model, persist in login.set_model_calls] == [
        (_POST_LOGIN_PROVIDER, _POST_LOGIN_DEFAULT, True)
    ]
    assert login.context.error_calls == []


@pytest.mark.tonio
async def test_reports_an_empty_catalog_only_after_refresh():
    with _manual_timeout():
        login = await _start_login()
        await login.discover([])

    assert login.set_model_calls == []
    assert any("no models are available" in message for message in login.context.error_calls)


@pytest.mark.tonio
async def test_preserves_a_model_selected_during_refresh():
    with _manual_timeout():
        login = await _start_login()
        login.session.model = DYNAMIC_MODEL
        await login.discover(["fast", _POST_LOGIN_DEFAULT])

    assert login.set_model_calls == []
    assert login.context.error_calls == []


@pytest.mark.tonio
async def test_bounds_refresh_to_15_seconds():
    with _manual_timeout() as timeouts:
        login = await _start_login()
        refreshed = tonio.Event()
        login.context.ui.request_render = lambda force=False: refreshed.set()
        timeouts[-1].timed_out = True
        timeouts[-1].token.cancel(TimeoutError("The operation timed out."))
        await refreshed.wait(5)
        assert refreshed.is_set()

    assert any("timed out" in message for message in login.context.warning_calls)
    assert any("no models are available" in message for message in login.context.error_calls)
    assert login.set_model_calls == []
