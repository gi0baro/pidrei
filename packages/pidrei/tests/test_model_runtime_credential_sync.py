"""Mirror of pi coding-agent test/model-runtime-credential-sync.test.ts.

pi's forever-pending promises (`new Promise(() => {})`) become gated
`tonio.Event`s released at test end so no parked task outlives its test
(a leftover would wedge interpreter shutdown on the shared test runtime).
"""

import pytest
import tonio.colored as tonio

from pidrei.core.auth_storage import AuthStorage
from pidrei.core.model_runtime import CredentialSynchronizationError, ModelRuntime
from pidrei_ai.auth.types import ApiKeyCredential, AuthCheck, AuthResult, ModelAuth, ProviderAuth
from pidrei_ai.registry import ModelsRefreshOptions
from pidrei_ai.types import Model, ModelCost
from pidrei_ai.utils.cancel import CancelToken


def make_model(provider: str) -> Model:
    return Model(
        id="dynamic",
        name="Dynamic",
        api="openai-completions",
        provider=provider,
        base_url="https://example.test/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=1000,
        max_tokens=100,
    )


class Interaction:
    def __init__(self, cancel: CancelToken | None = None):
        self.cancel = cancel

    async def prompt(self, prompt):
        return "unused"

    def notify(self, event):
        pass


class ProviderDouble:
    def __init__(self, id: str, *, login=None, refresh_models=None, check=None):
        self.id = id
        self.name = id
        self.base_url = None
        self.headers = None
        self.filter_models = None
        self._model = make_model(id)
        self._refresh_models = refresh_models

        async def default_login(_interaction):
            return ApiKeyCredential(key=f"{id}-key")

        async def default_check(_ctx, credential, _cancel):
            return AuthCheck(type="api_key", source="stored") if credential is not None else None

        async def resolve(_ctx, credential, _cancel):
            if credential is None:
                return None
            return AuthResult(auth=ModelAuth(api_key=credential.key), source="stored")

        from pidrei_ai.auth.types import ApiKeyAuth

        self.auth = ProviderAuth(
            api_key=ApiKeyAuth(
                name="API key",
                login=login if login is not None else default_login,
                check=check if check is not None else default_check,
                resolve=resolve,
            )
        )

    @property
    def has_dynamic_models(self) -> bool:
        return self._refresh_models is not None

    def get_models(self):
        return [self._model]

    async def refresh_models(self, context):
        if self._refresh_models is not None:
            await self._refresh_models(context)

    def stream(self, model, context, options=None):
        raise RuntimeError("unused")

    def stream_simple(self, model, context, options=None):
        raise RuntimeError("unused")


async def runtime_with_provider(registered, credentials: AuthStorage | None = None) -> ModelRuntime:
    runtime = await ModelRuntime.create(
        credentials=credentials if credentials is not None else AuthStorage.in_memory(),
        models_path=None,
        allow_model_network=False,
    )
    runtime.register_native_provider(registered)
    await runtime.refresh(ModelsRefreshOptions(allow_network=False, providers=[registered.id]))
    return runtime


@pytest.mark.tonio
async def test_publishes_locally_consistent_availability_before_login_and_logout_resolve():
    credentials = AuthStorage.in_memory()
    runtime = await runtime_with_provider(ProviderDouble("dynamic"), credentials)

    await runtime.login("dynamic", "api_key", Interaction())
    assert runtime.has_configured_auth("dynamic") is True
    assert "dynamic" in [entry.id for entry in runtime.get_available_snapshot()]
    assert await credentials.read("dynamic") == ApiKeyCredential(key="dynamic-key")

    await runtime.logout("dynamic")
    assert runtime.has_configured_auth("dynamic") is False
    assert not any(entry.provider == "dynamic" for entry in runtime.get_available_snapshot())
    assert await credentials.read("dynamic") is None


@pytest.mark.tonio
async def test_orders_same_provider_credential_operations_through_local_synchronization():
    login_started = tonio.Event()
    blocked_login = tonio.Event()
    credentials = AuthStorage.in_memory()

    async def blocked_login_fn(_interaction):
        login_started.set()
        await blocked_login.wait()
        return ApiKeyCredential(key="ordered-key")

    runtime = await runtime_with_provider(ProviderDouble("ordered", login=blocked_login_fn), credentials)

    async def run_login() -> None:
        await runtime.login("ordered", "api_key", Interaction())

    async def run_logout() -> None:
        await login_started.wait()
        logout = runtime.logout("ordered")
        await tonio.time.sleep(0.01)
        assert await credentials.read("ordered") is None
        blocked_login.set()
        await logout

    await tonio.spawn(run_login(), run_logout())
    assert await credentials.read("ordered") is None
    assert runtime.has_configured_auth("ordered") is False


@pytest.mark.tonio
async def test_allows_different_providers_to_run_credential_flows_concurrently():
    first_started = tonio.Event()
    second_started = tonio.Event()
    blocked = tonio.Event()

    async def login_one(_interaction):
        first_started.set()
        await blocked.wait()
        return ApiKeyCredential(key="one")

    async def login_two(_interaction):
        second_started.set()
        await blocked.wait()
        return ApiKeyCredential(key="two")

    runtime = await ModelRuntime.create(credentials=AuthStorage.in_memory(), models_path=None)
    runtime.register_native_provider(ProviderDouble("one", login=login_one))
    runtime.register_native_provider(ProviderDouble("two", login=login_two))
    await runtime.refresh(ModelsRefreshOptions(allow_network=False, providers=["one", "two"]))

    async def run_one() -> None:
        await runtime.login("one", "api_key", Interaction())

    async def run_two() -> None:
        await runtime.login("two", "api_key", Interaction())

    async def release() -> None:
        await first_started.wait()
        await second_started.wait()
        blocked.set()

    await tonio.spawn(run_one(), run_two(), release())


@pytest.mark.tonio
async def test_does_not_wait_for_unrelated_provider_availability_during_local_synchronization():
    state = {"stall": False}
    stall_release = tonio.Event()
    runtime = await ModelRuntime.create(credentials=AuthStorage.in_memory(), models_path=None)
    runtime.register_native_provider(ProviderDouble("target"))

    async def stalled_check(_ctx, _credential, _cancel):
        if state["stall"]:
            await stall_release.wait()

    runtime.register_native_provider(ProviderDouble("unrelated", check=stalled_check))
    await runtime.refresh(ModelsRefreshOptions(allow_network=False, providers=["target", "unrelated"]))
    state["stall"] = True

    try:
        await runtime.login("target", "api_key", Interaction())
        assert runtime.has_configured_auth("target") is True
        result = await runtime.refresh(ModelsRefreshOptions(allow_network=False, providers=["target"]))
        assert result.aborted is False
    finally:
        stall_release.set()


@pytest.mark.tonio
async def test_reports_cancellation_that_occurs_during_provider_scoped_availability():
    state = {"block": False}
    started = tonio.Event()
    release = tonio.Event()

    async def blocking_check(_ctx, credential, _cancel):
        if state["block"]:
            started.set()
            await release.wait()
        return AuthCheck(type="api_key", source="stored") if credential is not None else None

    registered = ProviderDouble("cancelled-availability", check=blocking_check)
    runtime = await runtime_with_provider(registered)
    await runtime.set_runtime_api_key(registered.id, "key")
    state["block"] = True
    controller = CancelToken()
    outcome: dict = {}

    async def run_refresh() -> None:
        outcome["result"] = await runtime.refresh(
            ModelsRefreshOptions(allow_network=False, providers=[registered.id], cancel=controller)
        )

    async def drive() -> None:
        await started.wait()
        controller.cancel()

    try:
        await tonio.spawn(run_refresh(), drive())
        assert outcome["result"].aborted is True
    finally:
        release.set()


@pytest.mark.tonio
async def test_does_not_run_network_refresh_inside_the_credential_operation_chain():
    network_calls = {"count": 0}
    network_release = tonio.Event()

    async def refresh_models(context):
        if context.allow_network:
            network_calls["count"] += 1
            await network_release.wait()

    runtime = await runtime_with_provider(ProviderDouble("local-only", refresh_models=refresh_models))

    try:
        await runtime.login("local-only", "api_key", Interaction())
        assert network_calls["count"] == 0
        assert runtime.has_configured_auth("local-only") is True
    finally:
        network_release.set()


@pytest.mark.tonio
async def test_keeps_provider_scoped_refreshes_from_superseding_unrelated_providers():
    started = tonio.Event()
    blocked = tonio.Event()
    received: dict = {}

    async def refresh_one_models(context):
        if not context.allow_network:
            return
        received["cancel"] = context.cancel
        started.set()
        await blocked.wait()

    runtime = await ModelRuntime.create(credentials=AuthStorage.in_memory(), models_path=None)
    runtime.register_native_provider(ProviderDouble("one", refresh_models=refresh_one_models))
    runtime.register_native_provider(ProviderDouble("two"))
    # Each registration spawned a detached full-refresh drain; only an
    # unscoped refresh clears the request flag, so run one now — otherwise a
    # lagging drain rebuilds every provider mid-test and supersedes (cancels)
    # exactly the refresh of "one" this test observes.
    await runtime.refresh(ModelsRefreshOptions(allow_network=False))
    await runtime.refresh(ModelsRefreshOptions(allow_network=False, providers=["one", "two"]))
    await runtime.set_runtime_api_key("one", "one-key")
    await runtime.set_runtime_api_key("two", "two-key")

    async def run_first() -> None:
        await runtime.refresh(ModelsRefreshOptions(allow_network=True, providers=["one"]))

    async def drive() -> None:
        await started.wait()
        await runtime.refresh(ModelsRefreshOptions(allow_network=True, providers=["two"]))
        assert received["cancel"].cancelled is False
        blocked.set()

    await tonio.spawn(run_first(), drive())


@pytest.mark.tonio
async def test_waits_for_a_committed_credential_mutation_to_settle_before_reporting_cancellation():
    committed = tonio.Event()
    mutation_finished = tonio.Event()
    state: dict = {"stored": None}

    class DelayedCommitStore:
        async def read(self, _provider_id, options=None):
            return state["stored"]

        async def list(self, options=None):
            from pidrei_ai.auth.types import CredentialInfo

            if state["stored"] is None:
                return []
            return [CredentialInfo(provider_id="delayed-commit", type=state["stored"].type)]

        async def modify(self, _provider_id, update, options=None):
            next_credential = await update(state["stored"])
            if next_credential is not None:
                state["stored"] = next_credential
            committed.set()
            await mutation_finished.wait()
            return state["stored"]

        async def delete(self, _provider_id, options=None):
            state["stored"] = None

    runtime = await ModelRuntime.create(credentials=DelayedCommitStore(), models_path=None)
    runtime.register_native_provider(ProviderDouble("delayed-commit"))
    await runtime.refresh(ModelsRefreshOptions(allow_network=False, providers=["delayed-commit"]))
    controller = CancelToken()
    outcome: dict = {"settled": False}

    async def run_login() -> None:
        try:
            await runtime.login("delayed-commit", "api_key", Interaction(cancel=controller))
            outcome["error"] = None
        except BaseException as error:
            outcome["error"] = error
        finally:
            outcome["settled"] = True

    async def drive() -> None:
        await committed.wait()
        controller.cancel()
        await tonio.time.sleep(0.01)
        assert outcome["settled"] is False
        mutation_finished.set()

    await tonio.spawn(run_login(), drive())
    assert isinstance(outcome["error"], CredentialSynchronizationError)
    assert outcome["error"].credential == ApiKeyCredential(key="delayed-commit-key")
    assert state["stored"] == ApiKeyCredential(key="delayed-commit-key")


@pytest.mark.tonio
async def test_reports_a_typed_error_when_cancellation_interrupts_post_commit_synchronization():
    state = {"block": False}
    cache_refresh_started = tonio.Event()
    release = tonio.Event()
    credentials = AuthStorage.in_memory()

    async def refresh_models(context):
        if not context.allow_network and state["block"]:
            cache_refresh_started.set()
            await release.wait()

    runtime = await runtime_with_provider(ProviderDouble("cancelled-sync", refresh_models=refresh_models), credentials)
    state["block"] = True
    controller = CancelToken()
    outcome: dict = {}

    async def run_login() -> None:
        try:
            await runtime.login("cancelled-sync", "api_key", Interaction(cancel=controller))
            outcome["error"] = None
        except BaseException as error:
            outcome["error"] = error

    async def drive() -> None:
        await cache_refresh_started.wait()
        controller.cancel()

    try:
        await tonio.spawn(run_login(), drive())
        error = outcome["error"]
        assert isinstance(error, CredentialSynchronizationError)
        assert error.provider_id == "cancelled-sync"
        assert error.operation == "login"
        assert error.credential == ApiKeyCredential(key="cancelled-sync-key")
        assert await credentials.read("cancelled-sync") == ApiKeyCredential(key="cancelled-sync-key")
    finally:
        release.set()


@pytest.mark.tonio
async def test_reports_committed_credentials_when_local_synchronization_fails():
    state = {"fail": False}
    credentials = AuthStorage.in_memory()

    async def refresh_models(context):
        if not context.allow_network and state["fail"]:
            raise RuntimeError("cache restore failed")

    runtime = await runtime_with_provider(ProviderDouble("broken-sync", refresh_models=refresh_models), credentials)
    state["fail"] = True

    with pytest.raises(CredentialSynchronizationError) as exc_info:
        await runtime.login("broken-sync", "api_key", Interaction())
    assert exc_info.value.provider_id == "broken-sync"
    assert exc_info.value.operation == "login"
    assert exc_info.value.credential == ApiKeyCredential(key="broken-sync-key")
    assert await credentials.read("broken-sync") == ApiKeyCredential(key="broken-sync-key")
