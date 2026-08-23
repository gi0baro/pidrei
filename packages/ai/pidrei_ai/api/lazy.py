"""Port of pi's lazy stream wrappers (packages/ai/src/api/lazy.ts).

`lazy_stream` returns a stream synchronously while async setup (auth
resolution, lazy loading) runs behind it; setup or forwarding failures
terminate the stream with an error event instead of raising.

Runtime shape (not pi's): the setup task is a child of a scope the stream
owns (`EventStream.spawn_producer`), so the request's `CancelToken` unwinds
it wherever it is parked. And where pi chains one stream per layer
(registry → provider → lazy api → adapter, each forwarding the next), here
the outermost stream is handed down as `into=` and every layer that accepts
it pushes straight into that one stream: one channel, zero forwarding hops
per event. A layer that returns a different stream (a provider that ignores
`into`) is forwarded the old way, so the protocol stays optional.
"""

import inspect
import time
from collections.abc import AsyncIterable, Awaitable, Callable
from typing import Any

from pidrei_ai.types import AssistantMessage, AssistantMessageEvent, ErrorEvent, Model, Usage
from pidrei_ai.utils.cancel import CancelToken
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
    setup: Callable[[AssistantMessageEventStream], Awaitable[AsyncIterable[AssistantMessageEvent] | None]],
    cancel: CancelToken | None = None,
    *,
    into: AssistantMessageEventStream | None = None,
) -> AssistantMessageEventStream:
    """Run `setup(stream)` behind a stream returned now.

    `setup` receives the stream to produce into. It may return that same
    stream (the next layer accepted `into=`: nothing to forward), `None`
    (it produced directly), or any other event source, which is forwarded.
    """
    outer = into if into is not None else AssistantMessageEventStream()
    if outer.partial is None:
        outer.partial = _create_setup_error_message(model, "Request was aborted")

    async def _run() -> None:
        try:
            inner = await setup(outer)
            if inner is None or inner is outer:
                return
            await _forward_stream(outer, inner)
        except Exception as error:
            message = _create_setup_error_message(model, error)
            outer.push(ErrorEvent(reason="error", error=message))
            outer.end(message)

    outer.spawn_producer(_run(), cancel)
    return outer


def _cancel_of(options: Any) -> CancelToken | None:
    return getattr(options, "cancel", None) if options is not None else None


def _accepts_into(fn: Callable[..., Any]) -> bool:
    try:
        parameters = inspect.signature(fn).parameters
    except TypeError, ValueError:
        return False
    return "into" in parameters or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def call_stream_into(fn: Callable[..., Any], *args: Any, into: AssistantMessageEventStream) -> Any:
    """Call a pi-style `stream(...)`, handing `into=` to implementations that take it.

    Providers written against pi's API (extensions, test doubles) return
    their own stream and are forwarded by `lazy_stream`; pidrei's own layers
    produce straight into `into`.
    """
    if _accepts_into(fn):
        return fn(*args, into=into)
    return fn(*args)


class LazyApi:
    """Wraps a lazily loaded API implementation module as `ProviderStreams`.

    The module loads on first stream call; Python's import cache deduplicates
    loads. Load failures terminate the returned stream with an error event.

    Deferred-response methods exist on the instance only when declared through
    `capabilities` — mirroring pi's conditional `api.fetchDeferred =` assignment
    — so `getattr(streams, "fetch_deferred", None)` probes work without loading
    the module.
    """

    __slots__ = ("_load", "cancel_deferred", "fetch_deferred")

    def __init__(self, load: Callable[[], Awaitable[Any]], capabilities: dict[str, bool] | None = None):
        self._load = load
        capabilities = capabilities or {}
        if capabilities.get("fetch_deferred"):

            def fetch_deferred(
                model: Model, handle: Any, options: Any = None, *, into: AssistantMessageEventStream | None = None
            ) -> AssistantMessageEventStream:
                async def _setup(stream: AssistantMessageEventStream) -> AsyncIterable[AssistantMessageEvent]:
                    module = await self._load()
                    implementation = getattr(module, "fetch_deferred", None)
                    if implementation is None:
                        raise RuntimeError("API does not support deferred responses")
                    return implementation(model, handle, options)

                return lazy_stream(model, _setup, _cancel_of(options), into=into)

            self.fetch_deferred = fetch_deferred
        if capabilities.get("cancel_deferred"):

            async def cancel_deferred(model: Model, handle: Any, options: Any = None) -> None:
                module = await self._load()
                implementation = getattr(module, "cancel_deferred", None)
                if implementation is None:
                    raise RuntimeError("API cannot cancel deferred responses")
                await implementation(model, handle, options)

            self.cancel_deferred = cancel_deferred

    def stream(
        self, model: Model, context: Any, options: Any = None, *, into: AssistantMessageEventStream | None = None
    ) -> AssistantMessageEventStream:
        return lazy_stream(model, self._make_setup("stream", model, context, options), _cancel_of(options), into=into)

    def stream_simple(
        self, model: Model, context: Any, options: Any = None, *, into: AssistantMessageEventStream | None = None
    ) -> AssistantMessageEventStream:
        return lazy_stream(
            model, self._make_setup("stream_simple", model, context, options), _cancel_of(options), into=into
        )

    def _make_setup(self, method: str, model: Model, context: Any, options: Any):
        async def _setup(stream: AssistantMessageEventStream) -> AsyncIterable[AssistantMessageEvent]:
            module = await self._load()
            return getattr(module, method)(model, context, options, into=stream)

        return _setup


def lazy_api(load: Callable[[], Awaitable[Any]], capabilities: dict[str, bool] | None = None) -> LazyApi:
    return LazyApi(load, capabilities)
