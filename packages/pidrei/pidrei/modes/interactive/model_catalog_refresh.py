"""Shared interactive model-catalog refreshes (port of pi `modes/interactive/model-catalog-refresh.ts`).

pi shares one in-flight all-catalog refresh per ModelRuntime through a shared
promise raced against each caller's AbortSignal; the Python shape is the
established tonio Event + result/error box (see ModelRuntime's availability
runs) with a `threading.Lock` around the waiter bookkeeping — pi's counter
updates are event-loop-atomic, tonio tasks run on real threads.
"""

import threading
import weakref
from typing import Any

import tonio.colored as tonio

from pidrei_ai.registry import ModelsRefreshOptions
from pidrei_ai.utils.cancel import CancelToken

from ...utils.abort import race_with_cancel


class _ActiveModelCatalogRefresh:
    __slots__ = ("controller", "done", "error", "result", "waiters")

    def __init__(self) -> None:
        self.controller = CancelToken()
        self.done = tonio.Event()
        self.result: Any = None
        self.error: BaseException | None = None
        self.waiters = 0


class _ModelCatalogRefreshCoordinator:
    def __init__(self) -> None:
        self._active_by_runtime: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._guard = threading.Lock()

    async def refresh(self, model_runtime: Any, cancel: CancelToken) -> Any:
        cancel.raise_if_cancelled()
        with self._guard:
            active = self._active_by_runtime.get(model_runtime)
            # pi's finally-cleanup removes a settled entry within the same
            # microtask batch as its settlement; here `_run`'s cleanup runs on
            # another thread, so a settled-but-not-yet-removed entry must not
            # capture new callers.
            if active is not None and active.done.is_set():
                active = None
            if active is None:
                active = _ActiveModelCatalogRefresh()
                self._active_by_runtime[model_runtime] = active
                tonio.spawn.without_tracking(self._run(model_runtime, active))
            active.waiters += 1

        async def await_active() -> Any:
            await active.done.wait()
            if active.error is not None:
                raise active.error
            return active.result

        try:
            return await race_with_cancel(await_active(), cancel)
        finally:
            with self._guard:
                active.waiters -= 1
                should_abort = active.waiters == 0 and self._active_by_runtime.get(model_runtime) is active
            if should_abort:
                active.controller.cancel()

    async def _run(self, model_runtime: Any, active: _ActiveModelCatalogRefresh) -> None:
        try:
            active.result = await race_with_cancel(
                model_runtime.refresh(ModelsRefreshOptions(cancel=active.controller)),
                active.controller,
            )
        except BaseException as error:
            active.error = error
        finally:
            with self._guard:
                if self._active_by_runtime.get(model_runtime) is active:
                    del self._active_by_runtime[model_runtime]
            active.done.set()


_model_catalog_refresh_coordinator = _ModelCatalogRefreshCoordinator()


async def refresh_model_catalogs(model_runtime: Any, cancel: CancelToken) -> Any:
    """Share concurrent interactive all-catalog refreshes while keeping each
    caller's cancellation independent."""
    return await _model_catalog_refresh_coordinator.refresh(model_runtime, cancel)
