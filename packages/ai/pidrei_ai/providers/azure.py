"""Port of pi's azure provider factory (packages/ai/src/providers/azure.ts).

`api` dispatches on `model.api`: azure-openai-responses, openai-completions
(Azure Foundry Chat Completions deployments).
"""

from collections.abc import AsyncIterable, Awaitable, Callable
from dataclasses import replace
from typing import Any

from pidrei_ai.api.azure_openai_config import resolve_azure_base_url, resolve_deployment_name
from pidrei_ai.api.azure_openai_responses_lazy import azure_openai_responses_api
from pidrei_ai.api.lazy import call_stream_into, lazy_stream
from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider
from pidrei_ai.types import AssistantMessageEvent, Model, SimpleStreamOptions, StreamOptions
from pidrei_ai.utils.callbacks import maybe_call
from pidrei_ai.utils.event_stream import AssistantMessageEventStream


def _resolve_azure_model(model: Model, options: StreamOptions | None) -> Model:
    return replace(model, base_url=resolve_azure_base_url(model, options))


def _with_deployment_name[TOptions: StreamOptions](
    model: Model, options: TOptions | None, empty: Callable[[], TOptions]
) -> TOptions | None:
    """Send the deployment name as the request's model, keeping `model.id` as the catalog id."""
    deployment_name = resolve_deployment_name(model, options)
    if deployment_name == model.id:
        return options
    caller_on_payload = options.on_payload if options is not None else None

    async def on_payload(payload: Any, payload_model: Model) -> Any:
        params = {**payload, "model": deployment_name}
        result = await maybe_call(caller_on_payload, params, payload_model)
        return result if result is not None else params

    return replace(options if options is not None else empty(), on_payload=on_payload)


class AzureStreams:
    """Resolve the Azure endpoint and deployment before dispatch, inside
    `lazy_stream` so an unconfigured endpoint errors on the stream instead of
    raising out of `stream()`."""

    __slots__ = ("_streams",)

    def __init__(self, streams: Any):
        self._streams = streams

    def stream(
        self,
        model: Model,
        context: Any,
        options: StreamOptions | None = None,
        *,
        into: AssistantMessageEventStream | None = None,
    ) -> AssistantMessageEventStream:
        return lazy_stream(
            model,
            self._setup(self._streams.stream, model, context, options, StreamOptions),
            options.cancel if options is not None else None,
            into=into,
        )

    def stream_simple(
        self,
        model: Model,
        context: Any,
        options: SimpleStreamOptions | None = None,
        *,
        into: AssistantMessageEventStream | None = None,
    ) -> AssistantMessageEventStream:
        return lazy_stream(
            model,
            self._setup(self._streams.stream_simple, model, context, options, SimpleStreamOptions),
            options.cancel if options is not None else None,
            into=into,
        )

    @staticmethod
    def _setup(
        dispatch: Callable[..., Any],
        model: Model,
        context: Any,
        options: StreamOptions | None,
        empty: Callable[[], StreamOptions],
    ) -> Callable[[AssistantMessageEventStream], Awaitable[AsyncIterable[AssistantMessageEvent] | None]]:
        async def setup(stream: AssistantMessageEventStream) -> AsyncIterable[AssistantMessageEvent] | None:
            return call_stream_into(
                dispatch,
                _resolve_azure_model(model, options),
                context,
                _with_deployment_name(model, options, empty),
                into=stream,
            )

        return setup


def azure_streams(streams: Any) -> AzureStreams:
    return AzureStreams(streams)


def azure_provider() -> Provider:
    return create_provider(
        id="azure",
        name="Azure",
        auth=ProviderAuth(api_key=env_api_key_auth("Azure OpenAI API key", ["AZURE_OPENAI_API_KEY"])),
        models=list(MODELS.get("azure", [])),
        api={
            "azure-openai-responses": azure_openai_responses_api(),
            "openai-completions": azure_streams(openai_completions_api()),
        },
    )
