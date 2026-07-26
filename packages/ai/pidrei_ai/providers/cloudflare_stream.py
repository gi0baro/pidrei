"""Port of pi's cloudflare-stream.ts: endpoint materialization before dispatch."""

from dataclasses import replace
from typing import Any

from pidrei_ai.types import Model, ProviderEnv
from pidrei_ai.utils.event_stream import AssistantMessageEventStream


CLOUDFLARE_ACCOUNT_ID = "CLOUDFLARE_ACCOUNT_ID"
CLOUDFLARE_GATEWAY_ID = "CLOUDFLARE_GATEWAY_ID"


def resolve_cloudflare_model(model: Model, env: ProviderEnv | None) -> Model:
    if not env:
        return model
    base_url = model.base_url.replace(
        f"{{{CLOUDFLARE_ACCOUNT_ID}}}", env.get(CLOUDFLARE_ACCOUNT_ID) or f"{{{CLOUDFLARE_ACCOUNT_ID}}}"
    ).replace(f"{{{CLOUDFLARE_GATEWAY_ID}}}", env.get(CLOUDFLARE_GATEWAY_ID) or f"{{{CLOUDFLARE_GATEWAY_ID}}}")
    return model if base_url == model.base_url else replace(model, base_url=base_url)


class CloudflareStreams:
    """Wrap an API implementation so Cloudflare account/gateway endpoint
    placeholders materialize from the resolved provider env before dispatch.
    """

    __slots__ = ("_streams",)

    def __init__(self, streams: Any):
        self._streams = streams

    def stream(self, model: Model, context: Any, options: Any = None) -> AssistantMessageEventStream:
        env = getattr(options, "env", None) if options is not None else None
        return self._streams.stream(resolve_cloudflare_model(model, env), context, options)

    def stream_simple(self, model: Model, context: Any, options: Any = None) -> AssistantMessageEventStream:
        env = getattr(options, "env", None) if options is not None else None
        return self._streams.stream_simple(resolve_cloudflare_model(model, env), context, options)


def cloudflare_streams(streams: Any) -> CloudflareStreams:
    return CloudflareStreams(streams)
