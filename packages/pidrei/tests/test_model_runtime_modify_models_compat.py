"""Partial mirror of pi coding-agent test/model-runtime-modify-models-compat.test.ts.

Holds the 1.0.0 case (#9962); the rest of the suite is a PARITY GAP.
"""

import pytest
import tonio.colored as tonio

from pidrei.core.auth_storage import AuthStorage
from pidrei.core.model_runtime import ModelRuntime
from pidrei_ai.auth.types import ModelAuth, OAuthAuth, OAuthCredential, ProviderAuth
from pidrei_ai.registry import ModelsRefreshOptions
from pidrei_ai.utils import clock

from .model_runtime_helpers import UnusedStreams, make_model


class GatedCredentials:
    """The in-memory store, with `list()` held while `gate` is set. The
    availability refresh that registration requests reads `list()`, so the
    test sees the snapshot registration published before that refresh lands."""

    def __init__(self, store: AuthStorage) -> None:
        self._store = store
        self.gate: tonio.Event | None = None

    def read(self, provider_id, options=None):
        return self._store.read(provider_id, options)

    async def list(self, options=None):
        if self.gate is not None:
            await self.gate.wait(5)
        return await self._store.list(options)

    def modify(self, provider_id, fn, options=None):
        return self._store.modify(provider_id, fn, options)

    def delete(self, provider_id, options=None):
        return self._store.delete(provider_id, options)


class NativeProvider(UnusedStreams):
    def __init__(self) -> None:
        self.id = "extension-native"
        self.name = "Extension Native"
        self.base_url = None
        self.headers = None
        self.filter_models = None
        self.has_dynamic_models = False
        self._model = make_model("extension-native", "native")

        async def unused(*_args):
            raise RuntimeError("unused")

        async def refresh(credential, _cancel):
            return credential

        async def to_auth(credential):
            return ModelAuth(api_key=credential.access)

        self.auth = ProviderAuth(oauth=OAuthAuth(name="Native OAuth", login=unused, refresh=refresh, to_auth=to_auth))

    def get_models(self):
        return [self._model]


# Regression for #9962: initial model selection reads the snapshot before the async refresh finishes.
@pytest.mark.tonio
async def test_marks_a_native_provider_with_a_stored_credential_as_configured_when_it_registers():
    credentials = GatedCredentials(
        AuthStorage.in_memory(
            {
                "extension-native": OAuthCredential(
                    access="access", refresh="refresh", expires=clock.now_ms() + 3_600_000
                )
            }
        )
    )
    runtime = await ModelRuntime(credentials=credentials, models_path=None, allow_model_network=False)
    gate = credentials.gate = tonio.Event()
    try:
        runtime.register_native_provider(NativeProvider())

        assert runtime.has_configured_auth("extension-native") is True
        assert runtime.is_using_oauth("extension-native") is True
        assert "extension-native/native" in [f"{m.provider}/{m.id}" for m in runtime.get_available_snapshot()]
    finally:
        credentials.gate = None
        gate.set()
    await runtime.refresh(ModelsRefreshOptions(allow_network=False))
    assert runtime.has_configured_auth("extension-native") is True
