"""Not a pi mirror: pi composes providers on one event loop, so this class of
bug cannot exist there.

`ModelRuntime.register_provider()` / `register_native_provider()` /
`unregister_provider()` mutate the composition inputs synchronously and then
detach a `refresh()`, whose `_rebuild_providers()` walks *every* provider and
re-publishes each one. On free-threaded CPython that walk really does run
concurrently with the caller's thread, and `_recompose_provider()` composes a
provider and publishes it as two separate steps — so an `unregister_provider()`
landing between those two steps used to be silently undone, leaving `Models`
holding a provider the runtime had already dropped from its snapshot.

Found while wiring the extension loader in Phase 5e (`pi.unregister_provider()`
is the first caller outside tests). Fixed by `_composition_guard`.
"""

import contextlib

import pytest
import tonio.colored as tonio

from pidrei.core import model_runtime as model_runtime_module
from pidrei.core.auth_storage import AuthStorage
from pidrei.core.model_registry import ModelRegistry
from pidrei.core.model_runtime import ModelRuntime


PROVIDER_CONFIG = {
    "baseUrl": "https://provider.test/v1",
    "apiKey": "provider-test-key",
    "api": "openai-completions",
    "models": [
        {
            "id": "instant-model",
            "name": "Instant Model",
            "reasoning": False,
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 128000,
            "maxTokens": 4096,
        }
    ],
}


async def _make_runtime() -> ModelRuntime:
    return await ModelRuntime.create(credentials=AuthStorage.in_memory(), models_path=None, allow_model_network=False)


@pytest.mark.tonio
async def test_unregister_survives_the_refresh_the_register_detached():
    runtime = await _make_runtime()
    registry = ModelRegistry(runtime)

    # No await between these two: the detached refresh from the register is
    # still in flight, which is the whole point.
    runtime.register_provider("instant-provider", PROVIDER_CONFIG)
    runtime.unregister_provider("instant-provider")

    assert registry.find("instant-provider", "instant-model") is None
    # ...and it must not come back when the in-flight rebuilds land either.
    await tonio.sleep(0.05)
    assert registry.find("instant-provider", "instant-model") is None
    assert runtime.get_models("instant-provider") == []


@pytest.mark.tonio
async def test_repeated_register_unregister_pairs_never_leave_a_stray_provider():
    runtime = await _make_runtime()
    registry = ModelRegistry(runtime)

    for index in range(25):
        provider_id = f"churn-{index}"
        runtime.register_provider(provider_id, PROVIDER_CONFIG)
        assert registry.find(provider_id, "instant-model") is not None
        runtime.unregister_provider(provider_id)
        assert registry.find(provider_id, "instant-model") is None

    await tonio.sleep(0.05)
    for index in range(25):
        assert registry.find(f"churn-{index}", "instant-model") is None


@pytest.mark.tonio
async def test_the_model_collection_and_the_snapshot_agree_after_churn():
    """The old failure was observable precisely as these two disagreeing."""
    runtime = await _make_runtime()

    runtime.register_provider("instant-provider", PROVIDER_CONFIG)
    runtime.unregister_provider("instant-provider")
    await tonio.sleep(0.05)

    in_collection = {model.provider for model in runtime.get_models()}
    in_snapshot = {model.provider for model in runtime._snapshot.all}
    assert "instant-provider" not in in_collection
    assert in_collection == in_snapshot


@contextlib.contextmanager
def _blocking_compose(provider_id: str, entered, release):
    """Stall composition of one provider between composing and publishing —
    the exact window the old code let an unregister slip into."""
    original = model_runtime_module.compose_model_provider

    def stalled(target_id, *args, **kwargs):
        composed = original(target_id, *args, **kwargs)
        if target_id == provider_id:
            entered.set()
            release.wait_sync()
        return composed

    model_runtime_module.compose_model_provider = stalled
    try:
        yield
    finally:
        model_runtime_module.compose_model_provider = original


@pytest.mark.tonio
async def test_a_mutation_cannot_start_while_a_provider_is_mid_composition():
    """The probabilistic cases above only catch the regression when the
    threads happen to interleave; this pins the invariant that makes them
    safe."""
    runtime = await _make_runtime()
    entered = tonio.Event()
    release = tonio.Event()
    finished = tonio.Event()
    observed: list[bool] = []

    def wait_sync() -> None:
        while not release.is_set():
            pass

    release.wait_sync = wait_sync  # type: ignore[attr-defined]

    async def recompose() -> None:
        with _blocking_compose("instant-provider", entered, release):
            await tonio.spawn_blocking(lambda: runtime._recompose_provider("instant-provider"))

    async def unregister() -> None:
        await entered.wait()
        await tonio.spawn_blocking(lambda: runtime.unregister_provider("instant-provider"))
        finished.set()

    async def observe() -> None:
        # Record rather than assert, and always release: an assertion here
        # would leave the composing thread spinning forever on a failure.
        try:
            await entered.wait()
            await tonio.sleep(0.02)
            observed.append(finished.is_set())
        finally:
            release.set()

    runtime.register_provider("instant-provider", PROVIDER_CONFIG)
    await tonio.spawn(recompose(), unregister(), observe())

    # The unregister was queued behind the composition, not racing it.
    assert observed == [False]
    assert finished.is_set()
    assert ModelRegistry(runtime).find("instant-provider", "instant-model") is None
