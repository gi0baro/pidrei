"""Mirror of pi's suite/regressions/11692-dispose-waits-for-oauth-refresh.test.ts.

pi's deferreds become `tonio.Event`s. pi sleeps 20 ms and asserts dispose has not
returned; here the token response is released only once dispose has reached the
credential settlement (the runtime's `settle_credential_writes`, wrapped to signal
it), and the assertion is that the rotated token was saved when dispose returned.
`register_native_provider` requests a detached full refresh, drained first (spec §3).
"""

import pytest
import tonio.colored as tonio

from pidrei.core.agent_session_runtime import AgentSessionRuntime
from pidrei.core.agent_session_services import AgentSessionServices
from pidrei.core.auth_storage import AuthStorage
from pidrei.core.model_runtime import ModelRuntime
from pidrei_ai.auth.types import ModelAuth, OAuthAuth, OAuthCredential, ProviderAuth
from pidrei_ai.registry import ModelsRefreshOptions
from pidrei_ai.types import Model, ModelCost
from pidrei_utils import clock

from .harness import create_harness


PROVIDER_ID = "rotating-oauth"

MODEL = Model(
    id="model",
    name="Model",
    api="openai-completions",
    provider=PROVIDER_ID,
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


class _RotatingOAuthProvider:
    def __init__(self, refresh):
        self.id = PROVIDER_ID
        self.name = PROVIDER_ID
        self.base_url = None
        self.headers = None
        self.filter_models = None

        async def login(_interaction, _options=None):
            raise Exception("unused")

        async def to_auth(credential):
            return ModelAuth(api_key=credential.access)

        self.auth = ProviderAuth(oauth=OAuthAuth(name="OAuth", login=login, refresh=refresh, to_auth=to_auth))

    @property
    def has_dynamic_models(self) -> bool:
        return True

    def get_models(self):
        return [MODEL]

    async def refresh_models(self, _context):
        # Like the built-in catalogs, the model list does not need the credential, but the
        # catalog refresh still resolves (and so refreshes) an expired one.
        pass

    def stream(self, model, context, options=None):
        raise RuntimeError("unused")

    def stream_simple(self, model, context, options=None):
        raise RuntimeError("unused")


@pytest.mark.tonio
async def test_dispose_waits_until_the_rotated_token_from_a_background_catalog_refresh_is_saved(harnesses):
    credentials = AuthStorage.in_memory(
        {PROVIDER_ID: OAuthCredential(access="old-access", refresh="old-refresh", expires=int(clock.now_ms()) - 1000)}
    )
    refresh_started = tonio.Event()
    response = tonio.Event()

    async def refresh(_credential, _cancel):
        # The token endpoint has already spent "old-refresh" when this request is in flight.
        refresh_started.set()
        await response.wait()
        return OAuthCredential(access="new-access", refresh="new-refresh", expires=int(clock.now_ms()) + 3_600_000)

    model_runtime = await ModelRuntime(credentials=credentials, models_path=None, allow_model_network=False)
    model_runtime.register_native_provider(_RotatingOAuthProvider(refresh))
    await model_runtime.refresh(ModelsRefreshOptions(allow_network=False))

    harness = await create_harness()
    harnesses.append(harness)
    services = AgentSessionServices(
        cwd=harness.temp_dir,
        agent_dir=harness.temp_dir,
        model_runtime=model_runtime,
        settings_manager=harness.settings_manager,
        resource_loader=harness.session.resource_loader,
    )

    async def unused_factory(**_kwargs):
        raise Exception("runtime replacement is not used in this test")

    runtime = AgentSessionRuntime(harness.session, services, unused_factory)
    settling = tonio.Event()
    settle_credential_writes = model_runtime.settle_credential_writes

    def signalling_settle():
        settling.set()
        return settle_credential_writes()

    model_runtime.settle_credential_writes = signalling_settle

    # The RPC startup catalog refresh runs in the background and is not awaited by shutdown.
    catalog_refresh = tonio.spawn(
        model_runtime.refresh(ModelsRefreshOptions(allow_network=True, providers=[PROVIDER_ID]))
    )
    await refresh_started.wait(5)
    assert refresh_started.is_set(), "the catalog refresh never refreshed the credential"

    saved_when_disposed: list = []
    released_when_disposed: list[bool] = []

    async def dispose() -> None:
        await runtime.dispose()
        released_when_disposed.append(response.is_set())
        saved_when_disposed.append(await credentials.read(PROVIDER_ID))

    disposing = tonio.spawn(dispose())
    try:
        await settling.wait(5)
        assert settling.is_set(), "dispose never waited for credential writes"
    finally:
        response.set()
    await disposing
    assert released_when_disposed == [True]
    assert len(saved_when_disposed) == 1
    assert saved_when_disposed[0].access == "new-access"
    assert saved_when_disposed[0].refresh == "new-refresh"
    await catalog_refresh
