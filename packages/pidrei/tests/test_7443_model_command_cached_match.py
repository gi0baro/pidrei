"""Mirror of pi coding-agent test/suite/regressions/7443-model-command-cached-match.test.ts."""

from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei_ai.registry import ModelsRefreshResult
from pidrei_ai.utils.cancel import CancelToken

from .harness import create_harness


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


def make_context(harness, status_calls, warning_calls):
    return SimpleNamespace(
        session=harness.session,
        show_status=status_calls.append,
        show_warning=warning_calls.append,
    )


@pytest.mark.tonio
async def test_matches_the_availability_snapshot_without_starting_a_catalog_refresh(harnesses):
    harness = await create_harness(models=[{"id": "cached", "name": "Cached"}])
    harnesses.append(harness)
    refresh_calls = {"count": 0}
    frozen = tonio.Event()

    async def frozen_refresh(_options=None):
        refresh_calls["count"] += 1
        await frozen.wait()

    harness.session.model_runtime.refresh = frozen_refresh
    status_calls: list = []
    warning_calls: list = []
    context = make_context(harness, status_calls, warning_calls)

    try:
        model = await InteractiveMode._find_exact_model_match(context, "cached")
        assert model is not None
        assert model.id == "cached"
        assert refresh_calls["count"] == 0
        assert status_calls == []
    finally:
        frozen.set()


@pytest.mark.tonio
async def test_uses_a_caller_owned_deadline_only_after_a_cache_miss(harnesses):
    harness = await create_harness(models=[{"id": "cached", "name": "Cached"}])
    harnesses.append(harness)
    refresh_options: list = []

    async def aborted_refresh(options=None):
        refresh_options.append(options)
        return ModelsRefreshResult(aborted=True, errors={})

    harness.session.model_runtime.refresh = aborted_refresh
    status_calls: list = []
    warning_calls: list = []
    context = make_context(harness, status_calls, warning_calls)

    assert await InteractiveMode._find_exact_model_match(context, "not-cached") is None

    assert len(refresh_options) == 1
    assert isinstance(refresh_options[0].cancel, CancelToken)
    assert status_calls == ["Refreshing model catalogs…"]
