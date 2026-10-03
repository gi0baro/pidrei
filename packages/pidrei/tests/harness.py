"""Port of pi's test/suite/harness.ts.

An AgentSession wired to the faux provider, so a mirror can script provider
responses and inspect each request's context. pi registers the faux provider
globally through `registerFauxProvider`; pidrei's `faux_provider()` hands back
an explicit `Provider`, so the harness registers it on the model runtime
instead — the same result without the global.
"""

import shutil
import tempfile
from types import SimpleNamespace
from typing import Any

from pidrei.core.agent_session import AgentSession, AgentSessionConfig, ExtensionBindings
from pidrei.core.auth_storage import AuthStorage
from pidrei.core.event_bus import EventBus
from pidrei.core.extensions import LoadExtensionsResult
from pidrei.core.extensions.loader import create_extension_runtime, load_extension_from_factory
from pidrei.core.messages import convert_to_llm
from pidrei.core.model_runtime import ModelRuntime
from pidrei.core.session_manager import SessionManager
from pidrei.core.settings_manager import SettingsManager
from pidrei_agent.agent import Agent, AgentInitialState
from pidrei_ai.auth.types import ApiKeyCredential
from pidrei_ai.providers.faux import FauxModelDefinition, faux_provider
from pidrei_ai.registry import ModelsRefreshOptions

from .agent_session_helpers import create_test_resource_loader


class Harness:
    def __init__(self, *, session, session_manager, settings_manager, auth_storage, faux, temp_dir, events):
        self.session = session
        self.session_manager = session_manager
        self.settings_manager = settings_manager
        self.auth_storage = auth_storage
        self.faux = faux
        self.temp_dir = temp_dir
        self.events = events

    @property
    def models(self) -> list:
        return self.faux.models

    def get_model(self, model_id: str | None = None):
        return self.faux.get_model() if model_id is None else self.faux.get_model(model_id)

    def set_responses(self, responses) -> None:
        self.faux.set_responses(responses)

    def append_responses(self, responses) -> None:
        self.faux.append_responses(responses)

    def get_pending_response_count(self) -> int:
        return self.faux.get_pending_response_count()

    def events_of_type(self, type_name: str) -> list:
        return [event for event in self.events if getattr(event, "type", None) == type_name]

    def cleanup(self) -> None:
        self.session.dispose()
        shutil.rmtree(self.temp_dir, ignore_errors=True)


async def create_harness(
    *,
    settings: dict | None = None,
    tools: list | None = None,
    initial_active_tool_names: list[str] | None = None,
    allowed_tool_names: list[str] | None = None,
    excluded_tool_names: list[str] | None = None,
    resource_loader: Any = None,
    extension_factories: list | None = None,
    with_configured_auth: bool = True,
    models: list[dict] | None = None,
    session_manager: SessionManager | None = None,
    extension_bindings: ExtensionBindings | None = None,
) -> Harness:
    """`session_manager` is the session to continue, for example to test a
    resume. Default: a new in-memory session.

    The harness binds the extensions once, which emits `session_start`;
    `extension_bindings` are the bindings it binds with (pi's suites call
    `bindExtensions({ uiContext })` themselves, and a reload emits
    `session_start` again only to a session bound with a UI context)."""
    temp_dir = tempfile.mkdtemp(prefix="pidrei-suite-")
    faux = faux_provider(models=[FauxModelDefinition(**model) for model in models]) if models else faux_provider()
    faux.set_responses([])
    model = faux.get_model()

    session_manager = session_manager if session_manager is not None else SessionManager.in_memory()
    settings_manager = SettingsManager.in_memory(settings)

    auth_storage = AuthStorage.in_memory()
    if with_configured_auth:

        async def set_key(_credential):
            return ApiKeyCredential(key="faux-key")

        await auth_storage.modify(model.provider, set_key)

    model_runtime = await ModelRuntime(credentials=auth_storage, models_path=None, allow_model_network=False)
    if with_configured_auth:
        model_runtime.register_native_provider(faux.provider)
        # pi registers the faux provider before runtime creation, so the
        # availability snapshot already contains it; settle it here so
        # snapshot-dependent tests see the same state deterministically.
        await model_runtime.refresh(ModelsRefreshOptions(allow_network=False, providers=[faux.provider.id]))

    # AgentSession assigns `ref.current`, so this is an attribute holder.
    extension_runner_ref = SimpleNamespace(current=None)

    async def transform_context(messages, _cancel=None):
        runner = extension_runner_ref.current
        if runner is None:
            return messages
        return await runner.emit_context(messages)

    async def on_payload(payload, *_rest):
        runner = extension_runner_ref.current
        if runner is None or not runner.has_handlers("before_provider_request"):
            return payload
        return await runner.emit_before_provider_request(payload)

    # pidrei's providers call these with the model as a second argument.
    async def on_response(response, *_rest):
        runner = extension_runner_ref.current
        if runner is None or not runner.has_handlers("after_provider_response"):
            return
        await runner.emit(
            {
                "type": "after_provider_response",
                "status": getattr(response, "status", None),
                "headers": getattr(response, "headers", None),
            }
        )

    async def stream_fn(request_model, context, stream_options=None):
        # pi passes pi-ai's global `streamSimple`; pidrei routes through the
        # model runtime, which resolves auth and dispatches to the provider.
        return model_runtime.stream_simple(request_model, context, stream_options)

    async def get_api_key(_provider):
        return "faux-key" if with_configured_auth else None

    async def convert_context_to_llm(messages):
        return convert_to_llm(messages)

    agent = Agent(
        stream_fn=stream_fn,
        get_api_key=get_api_key,
        initial_state=AgentInitialState(model=model, system_prompt="", tools=[]),
        convert_to_llm=convert_context_to_llm,
        transform_context=transform_context,
        on_payload=on_payload,
        on_response=on_response,
    )

    if resource_loader is None:
        extensions_result = None
        if extension_factories:
            runtime = create_extension_runtime()
            extensions = [
                await load_extension_from_factory(factory, temp_dir, EventBus(), runtime, f"<inline:{index + 1}>")
                for index, factory in enumerate(extension_factories)
            ]
            extensions_result = LoadExtensionsResult(extensions=extensions, runtime=runtime)
        resource_loader = create_test_resource_loader(extensions_result)

    session = AgentSession(
        AgentSessionConfig(
            agent=agent,
            session_manager=session_manager,
            settings_manager=settings_manager,
            cwd=temp_dir,
            model_runtime=model_runtime,
            resource_loader=resource_loader,
            base_tools_override={tool.name: tool for tool in tools} if tools else None,
            initial_active_tool_names=initial_active_tool_names,
            allowed_tool_names=allowed_tool_names,
            excluded_tool_names=excluded_tool_names,
            extension_runner_ref=extension_runner_ref,
        )
    )
    await session.bind_extensions(extension_bindings if extension_bindings is not None else ExtensionBindings())

    events: list = []
    session.subscribe(events.append)

    return Harness(
        session=session,
        session_manager=session_manager,
        settings_manager=settings_manager,
        auth_storage=auth_storage,
        faux=faux,
        temp_dir=temp_dir,
        events=events,
    )


def get_message_text(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
        if part_type == "text":
            parts.append(part.get("text") if isinstance(part, dict) else part.text)
    return "\n".join(parts)


def get_user_texts(harness: Harness) -> list[str]:
    return [get_message_text(m) for m in harness.session.messages if getattr(m, "role", None) == "user"]


def get_assistant_texts(harness: Harness) -> list[str]:
    return [get_message_text(m) for m in harness.session.messages if getattr(m, "role", None) == "assistant"]


def get_tool_result(harness: Harness, tool_name: str) -> Any:
    """The latest result of `tool_name` in the session transcript."""
    for message in reversed(harness.session.messages):
        if message.role == "toolResult" and message.tool_name == tool_name:
            return message
    raise AssertionError(f"No {tool_name} tool result")


async def _answer(value: Any) -> Any:
    return value


def create_test_ui_context(**overrides: Any) -> SimpleNamespace:
    """pi's createTestUiContext: an extension UI context that does nothing,
    with `overrides` applied. Dialogs return an awaitable, as the real ones
    return a spawn handle."""
    ui = SimpleNamespace(
        select=lambda *_args, **_kwargs: _answer(None),
        confirm=lambda *_args, **_kwargs: _answer(False),
        input=lambda *_args, **_kwargs: _answer(None),
        editor=lambda *_args, **_kwargs: _answer(None),
        custom=lambda *_args, **_kwargs: _answer(None),
        notify=lambda *_args, **_kwargs: None,
        on_terminal_input=lambda *_args, **_kwargs: lambda: None,
        set_status=lambda *_args, **_kwargs: None,
        set_working_message=lambda *_args, **_kwargs: None,
        set_working_visible=lambda *_args, **_kwargs: None,
        set_working_indicator=lambda *_args, **_kwargs: None,
        set_hidden_thinking_label=lambda *_args, **_kwargs: None,
        set_widget=lambda *_args, **_kwargs: None,
        set_footer=lambda *_args, **_kwargs: None,
        set_header=lambda *_args, **_kwargs: None,
        set_title=lambda *_args, **_kwargs: None,
        paste_to_editor=lambda *_args, **_kwargs: None,
        set_editor_text=lambda *_args, **_kwargs: None,
        get_editor_text=lambda: "",
    )
    for name, value in overrides.items():
        setattr(ui, name, value)
    return ui


async def create_test_extensions_result(factories: list, cwd: str) -> LoadExtensionsResult:
    """pi's createTestExtensionsResult: load inline extension factories with a fresh runtime."""
    runtime = create_extension_runtime()
    event_bus = EventBus()
    extensions = [
        await load_extension_from_factory(factory, cwd, event_bus, runtime, f"<inline:{index + 1}>")
        for index, factory in enumerate(factories)
    ]
    return LoadExtensionsResult(extensions=extensions, runtime=runtime)


__all__ = [
    "Harness",
    "create_harness",
    "create_test_extensions_result",
    "create_test_ui_context",
    "get_assistant_texts",
    "get_message_text",
    "get_tool_result",
    "get_user_texts",
]
