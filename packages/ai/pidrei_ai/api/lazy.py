"""Port of pi's lazy stream wrappers (packages/ai/src/api/lazy.ts).

`lazy_stream` returns a stream synchronously while async setup (auth
resolution, lazy loading) runs behind it on a detached tonio task; setup or
forwarding failures terminate the stream with an error event instead of
raising.
"""

import time
from collections.abc import AsyncIterable, Awaitable, Callable
from typing import Any

import tonio.colored as tonio

from pidrei_ai.types import AssistantMessage, AssistantMessageEvent, ErrorEvent, Model, Usage
from pidrei_ai.utils.event_stream import AssistantMessageEventStream


def _create_setup_error_message(model: Model, error: Any) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="error",
        error_message=str(error),
        timestamp=int(time.time() * 1000),
    )


async def _forward_stream(
    target: AssistantMessageEventStream,
    source: AsyncIterable[AssistantMessageEvent],
) -> None:
    async for event in source:
        target.push(event)
    result = getattr(source, "result", None)
    if callable(result):
        target.end(await result())
    else:
        target.end()


def lazy_stream(
    model: Model,
    setup: Callable[[], Awaitable[AsyncIterable[AssistantMessageEvent]]],
) -> AssistantMessageEventStream:
    outer = AssistantMessageEventStream()

    async def _run() -> None:
        try:
            inner = await setup()
            await _forward_stream(outer, inner)
        except Exception as error:
            message = _create_setup_error_message(model, error)
            outer.push(ErrorEvent(reason="error", error=message))
            outer.end(message)

    tonio.spawn.without_tracking(_run())
    return outer


class LazyApi:
    """Wraps a lazily loaded API implementation module as `ProviderStreams`.

    The module loads on first stream call; Python's import cache deduplicates
    loads. Load failures terminate the returned stream with an error event.
    """

    __slots__ = ("_load",)

    def __init__(self, load: Callable[[], Awaitable[Any]]):
        self._load = load

    def stream(self, model: Model, context: Any, options: Any = None) -> AssistantMessageEventStream:
        return lazy_stream(model, self._make_setup("stream", model, context, options))

    def stream_simple(self, model: Model, context: Any, options: Any = None) -> AssistantMessageEventStream:
        return lazy_stream(model, self._make_setup("stream_simple", model, context, options))

    def _make_setup(self, method: str, model: Model, context: Any, options: Any):
        async def _setup() -> AsyncIterable[AssistantMessageEvent]:
            module = await self._load()
            return getattr(module, method)(model, context, options)

        return _setup


def lazy_api(load: Callable[[], Awaitable[Any]]) -> LazyApi:
    return LazyApi(load)
