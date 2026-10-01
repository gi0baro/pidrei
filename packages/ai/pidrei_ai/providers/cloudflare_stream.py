"""Port of pi's cloudflare-stream.ts: endpoint materialization before dispatch."""

from dataclasses import replace
from typing import Any

from pidrei_ai.types import ClassifierModel, ClassifierResult, Model, ProviderEnv
from pidrei_ai.utils.event_stream import AssistantMessageEventStream


CLOUDFLARE_ACCOUNT_ID = "CLOUDFLARE_ACCOUNT_ID"
CLOUDFLARE_GATEWAY_ID = "CLOUDFLARE_GATEWAY_ID"


def resolve_cloudflare_model[TModel: Model | ClassifierModel](model: TModel, env: ProviderEnv | None) -> TModel:
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


class CloudflareClassifier:
    """Classifier counterpart of `CloudflareStreams`."""

    __slots__ = ("_classifier",)

    def __init__(self, classifier: Any):
        self._classifier = classifier

    async def classify(self, model: ClassifierModel, context: Any, options: Any = None) -> ClassifierResult:
        env = options.env if options is not None else None
        return await self._classifier.classify(resolve_cloudflare_model(model, env), context, options)


def cloudflare_classifier(classifier: Any) -> CloudflareClassifier:
    return CloudflareClassifier(classifier)
