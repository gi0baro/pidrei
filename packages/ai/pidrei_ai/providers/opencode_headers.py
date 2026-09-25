"""Port of pi's providers/opencode-headers.ts."""

from dataclasses import replace
from typing import Any

from pidrei_ai.api.lazy import call_stream_into
from pidrei_ai.types import Context, Model, StreamOptions
from pidrei_ai.utils.event_stream import AssistantMessageEventStream


_OPENCODE_SESSION_HEADER = "x-opencode-session"


def _has_header(headers: dict[str, Any] | None, name: str) -> bool:
    expected = name.lower()
    return any(key.lower() == expected for key in (headers or {}))


def _with_session_header[TOptions: StreamOptions](options: TOptions | None) -> TOptions | None:
    if options is None or not options.session_id or _has_header(options.headers, _OPENCODE_SESSION_HEADER):
        return options
    return replace(options, headers={**(options.headers or {}), _OPENCODE_SESSION_HEADER: options.session_id})


class _OpenCodeSessionStreams:
    """`streams` with the session header applied; every other member (the
    optional deferred-response methods) passes through, like pi's spread."""

    __slots__ = ("_streams",)

    def __init__(self, streams: Any) -> None:
        self._streams = streams

    def stream(
        self, model: Model, context: Context, options: Any = None, *, into: AssistantMessageEventStream | None = None
    ) -> AssistantMessageEventStream:
        return call_stream_into(self._streams.stream, model, context, _with_session_header(options), into=into)

    def stream_simple(
        self, model: Model, context: Context, options: Any = None, *, into: AssistantMessageEventStream | None = None
    ) -> AssistantMessageEventStream:
        return call_stream_into(self._streams.stream_simple, model, context, _with_session_header(options), into=into)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._streams, name)


def with_opencode_session_header(streams: Any) -> Any:
    """Adds OpenCode's required per-conversation routing header before API dispatch."""
    return _OpenCodeSessionStreams(streams)
