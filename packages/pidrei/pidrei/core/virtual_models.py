"""Mirror of pi coding-agent src/core/virtual-models.ts.

Virtual models are catalog entries that route each request to a physical model.

The selection (`model_change`, `agent.state.model`, `ctx.model`) may name a
virtual model. Everything below the routing step only sees physical models:
providers stream them and assistant messages record them. A virtual model
never reaches a provider.

Virtual models belong to a provider id but are not provider models.
`ModelRuntime` keeps them separately and adds them to the provider's catalog
with `with_virtual_models()`, so any provider, including one with physical
models, can list several virtual models.

The API id and the state entry type keep pi's values: session files and
extensions written against pi compare against them.
"""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pidrei_ai.api.lazy import _cancel_of, call_stream_into, lazy_stream
from pidrei_ai.auth.types import ApiKeyAuth, AuthResult, ModelAuth, ProviderAuth
from pidrei_ai.registry import Provider, RefreshModelsContext
from pidrei_ai.types import AnyModel, AssistantMessage, Message, Model, ModelCost, ModelThinkingLevel
from pidrei_ai.utils.cancel import CancelToken
from pidrei_ai.utils.event_stream import AssistantMessageEventStream
from pidrei_ai.utils.model_operations import is_model_type


# API id of virtual catalog entries. Requests for it fail unless routed first.
VIRTUAL_MODEL_API = "pi-virtual"

# Custom entry type that stores router state on the session branch. Its data
# is `{"provider", "modelId", "state"}` (pi's `VirtualModelStateData`).
VIRTUAL_MODEL_STATE_ENTRY = "pi.virtual-model-state"

_THINKING_LEVELS: tuple[ModelThinkingLevel, ...] = ("off", "minimal", "low", "medium", "high", "xhigh", "max")

# Why a request is being routed.
# - "user": first request after a message the user wrote (prompt, steering, or follow-up)
# - "continuation": any other request in the agent loop, e.g. after tool results or extension messages
# - "retry": automatic retry after a failed request, including after compaction for a context overflow
# - "direct": a request outside the agent loop, e.g. a compaction summary or an extension call
type ModelRouteReason = Literal["user", "continuation", "retry", "direct"]


@dataclass(slots=True, frozen=True)
class RoutedResponse:
    """Physical model and thinking level of a response (pi: `previous`)."""

    model: Model
    thinking_level: ModelThinkingLevel | None = None


@dataclass(slots=True, frozen=True)
class FailedRoute:
    """The failed request a retry repeats (pi: `failed`)."""

    model: Model
    message: AssistantMessage
    thinking_level: ModelThinkingLevel | None = None


@dataclass(slots=True, frozen=True, kw_only=True)
class ModelRouteRequest:
    # The selected virtual model.
    model: Model
    # The selected thinking level. Its meaning is up to the router.
    thinking_level: ModelThinkingLevel
    reason: ModelRouteReason
    # Conversation for this request, including system messages.
    messages: Sequence[Message]
    # Physical model and thinking level of the latest successful response in `messages`.
    previous: RoutedResponse | None = None
    # For "retry": the failed request, which `messages` no longer contains.
    # `message` carries its `stop_reason` and `error_message`. None when the
    # router itself failed.
    failed: FailedRoute | None = None
    # Router state last returned on this session branch. None before the first
    # state and for "direct" requests.
    state: Any = None
    cancel: CancelToken | None = None


@dataclass(slots=True, frozen=True, kw_only=True)
class ModelRoute:
    """Physical model and thinking level for one request."""

    model: Model
    thinking_level: ModelThinkingLevel
    # New router state, stored on the session branch unless it is
    # `request.state` itself. Return `request.state` or None to keep the
    # current state. Must be JSON-serializable. Ignored for "direct" requests.
    state: Any = None


type RouteFn = Callable[[ModelRouteRequest], Awaitable[ModelRoute]]


@dataclass(slots=True, kw_only=True)
class VirtualModelDefinition:
    # Provider the virtual model is listed under. May be a provider with physical models.
    provider: str
    # Model id. Must not be the id of a physical model of `provider`.
    id: str
    name: str
    # Pick the physical model, which must have credentials, and thinking level
    # for one request. Async-only.
    route: RouteFn
    # Thinking levels offered for selection. Defaults to ["off"].
    thinking_levels: Sequence[ModelThinkingLevel] | None = None
    # Limits shown before the first response. Afterwards, pidrei uses the
    # limits of the physical model that answered. Unset limits are unknown (0).
    context_window: int | None = None
    max_tokens: int | None = None
    # Input types accepted for selection. Defaults to text and images; routed
    # models without image support get placeholders.
    input: list[Literal["text", "image"]] | None = None


def is_virtual_model(model: Any) -> bool:
    """Whether a model or message names a virtual model. Failed routing leaves
    the virtual model on its message."""
    return model.api == VIRTUAL_MODEL_API


def find_latest_response(messages: Sequence[Any]) -> AssistantMessage | None:
    """Latest successful response. Its model is physical: failed or aborted
    requests, including failed routing, are skipped."""
    for message in reversed(messages):
        if getattr(message, "role", None) == "assistant" and message.stop_reason not in ("error", "aborted"):
            return message
    return None


def get_branch_selection(
    branch: Sequence[dict[str, Any]],
    get_model: Callable[[str, str], Model | None],
) -> tuple[str, str] | None:
    """The `(provider, model_id)` selection a session branch records.

    A virtual `model_change` holds until the next `model_change`, because
    responses name the physical models it routed to. Otherwise the latest
    physical response wins, as in sessions without virtual models. A virtual
    model that is no longer registered does not hold, so the selection falls
    back to the physical model that answered last.
    """

    def is_virtual(provider: str, model_id: str) -> bool:
        model = get_model(provider, model_id)
        return model is not None and is_virtual_model(model)

    selection: tuple[str, str] | None = None
    for entry in branch:
        if entry["type"] == "model_change":
            selection = (entry["provider"], entry["modelId"])
        elif entry["type"] == "message":
            message = entry["message"]
            if (
                getattr(message, "role", None) == "assistant"
                and not is_virtual_model(message)
                and (selection is None or not is_virtual(*selection))
            ):
                selection = (message.provider, message.model)
    return selection


def is_same_state(a: Any, b: Any) -> bool:
    """pi compares a returned state with `!==`: scalars by value, anything else by identity."""
    if a is b:
        return True
    scalar = (str, int, float, bool)
    return isinstance(a, scalar) and type(a) is type(b) and a == b


def get_virtual_model_state(branch: Sequence[dict[str, Any]], provider: str, model_id: str) -> Any:
    """Latest router state a session branch stores for a virtual model."""
    for entry in reversed(branch):
        if entry["type"] != "custom" or entry.get("customType") != VIRTUAL_MODEL_STATE_ENTRY:
            continue
        data = entry.get("data")
        if isinstance(data, dict) and data.get("provider") == provider and data.get("modelId") == model_id:
            return data.get("state")
    return None


def create_virtual_model(definition: VirtualModelDefinition) -> Model:
    """Build the catalog entry of a virtual model."""
    levels = list(definition.thinking_levels) if definition.thinking_levels is not None else ["off"]
    return Model(
        id=definition.id,
        name=definition.name,
        api=VIRTUAL_MODEL_API,
        provider=definition.provider,
        base_url="",
        reasoning=any(level != "off" for level in levels),
        thinking_level_map={level: level if level in levels else None for level in _THINKING_LEVELS},
        input=list(definition.input) if definition.input is not None else ["text", "image"],
        cost=ModelCost(),
        context_window=definition.context_window if definition.context_window is not None else 0,
        max_tokens=definition.max_tokens if definition.max_tokens is not None else 0,
    )


def _unrouted_stream(model: Model, options: Any, into: AssistantMessageEventStream | None) -> Any:
    """Stream for a virtual model that was not routed, e.g. `stream()` with API-specific options."""

    async def fail(_stream: AssistantMessageEventStream) -> None:
        raise Exception(f"Virtual model {model.provider}/{model.id} must be routed before streaming")

    return lazy_stream(model, fail, _cancel_of(options), into=into)


async def _resolve_virtual_auth(_context: Any, _credential: Any, _cancel: Any) -> AuthResult:
    return AuthResult(auth=ModelAuth(), source="virtual")


class VirtualModelsProvider:
    """pi: the object `withVirtualModels()` returns. Without a base provider it
    is a keyless provider that only lists the virtual models; with one it
    spreads the base and adds the virtual models to its catalog."""

    def __init__(self, provider_id: str, provider: Any | None, virtual_models: list[Model]):
        self._base = provider
        self._virtual_models = list(virtual_models)
        self._ids = {model.id for model in virtual_models}
        self.id = provider_id
        if provider is None:
            self.name = provider_id
            self.base_url = None
            self.headers = None
            self.auth = ProviderAuth(api_key=ApiKeyAuth(name="Virtual model", resolve=_resolve_virtual_auth))
            self.filter_models = None
            self.filter_all_models = None
            self.generate_images = None
            self.classify = None
            return
        self.name = provider.name
        self.base_url = provider.base_url
        self.headers = provider.headers
        self.auth = provider.auth
        self.generate_images = provider.generate_images
        self.classify = provider.classify
        base_filter_models = provider.filter_models
        base_filter_all_models = provider.filter_all_models

        def filter_models(models: list[Model], credential: Any) -> list[Model]:
            real = self._physical(models)
            filtered = base_filter_models(real, credential) if base_filter_models is not None else real
            return [*filtered, *self._virtual(models)]

        self.filter_models = filter_models
        if base_filter_all_models is None:
            self.filter_all_models = None
        else:

            def filter_all_models(models: list[AnyModel], credential: Any) -> list[AnyModel]:
                return [*base_filter_all_models(self._physical(models), credential), *self._virtual(models)]

            self.filter_all_models = filter_all_models

    def _physical[TModel: AnyModel](self, models: Sequence[TModel]) -> list[TModel]:
        """A virtual model hides a physical chat model with the same id, which a
        catalog refresh can add after registration."""
        return [
            model
            for model in models
            if not is_virtual_model(model) and not (is_model_type(model, "chat") and model.id in self._ids)
        ]

    @staticmethod
    def _virtual[TModel: AnyModel](models: Sequence[TModel]) -> list[TModel]:
        return [model for model in models if is_virtual_model(model)]

    @property
    def has_dynamic_models(self) -> bool:
        return self._base is not None and self._base.has_dynamic_models

    def get_models(self) -> list[Model]:
        if self._base is None:
            return list(self._virtual_models)
        return [*self._physical(self._base.get_models()), *self._virtual_models]

    def get_all_models(self) -> list[AnyModel]:
        if self._base is None:
            return list(self._virtual_models)
        get_all_models = self._base.get_all_models
        base = get_all_models() if get_all_models is not None else self._base.get_models()
        return [*self._physical(base), *self._virtual_models]

    async def refresh_models(self, context: RefreshModelsContext) -> None:
        if self._base is not None:
            await self._base.refresh_models(context)

    def stream(self, model: Model, context: Any, options: Any = None, *, into: Any = None) -> Any:
        if self._base is None or is_virtual_model(model):
            return _unrouted_stream(model, options, into)
        if into is None:
            return self._base.stream(model, context, options)
        return call_stream_into(self._base.stream, model, context, options, into=into)

    def stream_simple(self, model: Model, context: Any, options: Any = None, *, into: Any = None) -> Any:
        if self._base is None or is_virtual_model(model):
            return _unrouted_stream(model, options, into)
        if into is None:
            return self._base.stream_simple(model, context, options)
        return call_stream_into(self._base.stream_simple, model, context, options, into=into)

    @property
    def supports_fetch_deferred(self) -> bool:
        return self._base is not None and self._base.supports_fetch_deferred

    @property
    def supports_cancel_deferred(self) -> bool:
        return self._base is not None and self._base.supports_cancel_deferred

    def fetch_deferred(self, model: Model, handle: Any, options: Any = None) -> Any:
        return self._base.fetch_deferred(model, handle, options)

    async def cancel_deferred(self, model: Model, handle: Any, options: Any = None) -> None:
        await self._base.cancel_deferred(model, handle, options)


def with_virtual_models(provider_id: str, provider: Provider | None, virtual_models: list[Model]) -> Provider:
    """Add virtual models to a provider's catalog. Without a provider, the
    result is a keyless provider that only lists the virtual models. A virtual
    model hides a physical chat model with the same id, which a catalog refresh
    can add after registration. Availability follows the provider's auth."""
    return VirtualModelsProvider(provider_id, provider, virtual_models)  # type: ignore[return-value]
