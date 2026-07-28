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
from pidrei_ai.providers.faux import faux_provider

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
    system_prompt: str = "You are a test assistant.",
    tools: list | None = None,
    initial_active_tool_names: list[str] | None = None,
    allowed_tool_names: list[str] | None = None,
    excluded_tool_names: list[str] | None = None,
    resource_loader: Any = None,
    extension_factories: list | None = None,
    with_configured_auth: bool = True,
) -> Harness:
    temp_dir = tempfile.mkdtemp(prefix="pidrei-suite-")
    faux = faux_provider()
    faux.set_responses([])
    model = faux.get_model()

    session_manager = SessionManager.in_memory()
    settings_manager = SettingsManager.in_memory(settings)

    auth_storage = AuthStorage.in_memory()
    if with_configured_auth:

        async def set_key(_credential):
            return ApiKeyCredential(key="faux-key")

        await auth_storage.modify(model.provider, set_key)

    model_runtime = await ModelRuntime.create(credentials=auth_storage, models_path=None, allow_model_network=False)
    if with_configured_auth:
        model_runtime.register_native_provider(faux.provider)

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
        initial_state=AgentInitialState(model=model, system_prompt=system_prompt, tools=[]),
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
    await session.bind_extensions(ExtensionBindings())

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


__all__ = [
    "Harness",
    "create_harness",
    "get_assistant_texts",
    "get_message_text",
    "get_user_texts",
]
