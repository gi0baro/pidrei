"""Mirror of pi coding-agent src/core/agent-session-services.ts."""

import os
from dataclasses import dataclass, field
from typing import Any

from ..config import get_agent_dir
from ..utils.paths import resolve_path
from .model_runtime import ModelRuntime
from .resource_loader import DefaultResourceLoader
from .sdk import CreateAgentSessionOptions, CreateAgentSessionResult, create_agent_session
from .session_manager import SessionManager
from .settings_manager import SettingsManager


@dataclass(slots=True)
class AgentSessionRuntimeDiagnostic:
    """Non-fatal issue collected while creating services or sessions.

    Runtime creation returns diagnostics to the caller instead of printing or
    exiting; the app layer decides what to show and whether to abort."""

    type: str  # "info" | "warning" | "error"
    message: str


@dataclass(slots=True, kw_only=True)
class CreateAgentSessionServicesOptions:
    """Inputs for creating cwd-bound runtime services.

    These services are recreated whenever the effective session cwd changes.
    CLI-provided resource paths should be resolved to absolute paths before
    they reach this function."""

    cwd: str
    agent_dir: str | None = None
    settings_manager: SettingsManager | None = None
    model_runtime: ModelRuntime | None = None
    extension_flag_values: dict[str, Any] | None = None
    resource_loader_options: dict[str, Any] | None = None
    resource_loader_reload_options: dict[str, Any] | None = None


@dataclass(slots=True)
class AgentSessionServices:
    """Coherent cwd-bound runtime services for one effective session cwd.

    Infrastructure only; the AgentSession itself is created separately so
    session options can be resolved against these services first."""

    cwd: str
    agent_dir: str
    model_runtime: ModelRuntime
    settings_manager: SettingsManager
    resource_loader: Any
    diagnostics: list[AgentSessionRuntimeDiagnostic] = field(default_factory=list)


@dataclass(slots=True, kw_only=True)
class CreateAgentSessionFromServicesOptions:
    """Inputs for creating an AgentSession from already-created services."""

    services: AgentSessionServices
    session_manager: SessionManager
    session_start_event: dict[str, Any] | None = None
    model: Any = None
    thinking_level: str | None = None
    scoped_models: list[Any] | None = None
    tools: list[str] | None = None
    exclude_tools: list[str] | None = None
    no_tools: str | None = None
    custom_tools: list[Any] | None = None


def _apply_extension_flag_values(
    resource_loader: Any, extension_flag_values: dict[str, Any] | None
) -> list[AgentSessionRuntimeDiagnostic]:
    if not extension_flag_values:
        return []

    diagnostics: list[AgentSessionRuntimeDiagnostic] = []
    extensions_result = resource_loader.get_extensions()
    registered_flags: dict[str, Any] = {}
    for extension in extensions_result.extensions:
        for name, flag in extension.flags.items():
            registered_flags[name] = flag

    unknown_flags: list[str] = []
    for name, value in extension_flag_values.items():
        flag = registered_flags.get(name)
        if flag is None:
            unknown_flags.append(name)
            continue
        if flag.type == "boolean":
            extensions_result.runtime.flag_values[name] = True
            continue
        if isinstance(value, str):
            extensions_result.runtime.flag_values[name] = value
            continue
        diagnostics.append(
            AgentSessionRuntimeDiagnostic(type="error", message=f'Extension flag "--{name}" requires a value')
        )

    if unknown_flags:
        plural = "" if len(unknown_flags) == 1 else "s"
        flags = ", ".join(f"--{name}" for name in unknown_flags)
        diagnostics.append(AgentSessionRuntimeDiagnostic(type="error", message=f"Unknown option{plural}: {flags}"))

    return diagnostics


async def create_agent_session_services(options: CreateAgentSessionServicesOptions) -> AgentSessionServices:
    """Create cwd-bound runtime services. Returns services plus diagnostics.
    Does not create an AgentSession."""
    cwd = resolve_path(options.cwd)
    agent_dir = resolve_path(options.agent_dir) if options.agent_dir is not None else get_agent_dir()
    model_runtime = options.model_runtime
    if model_runtime is None:
        model_runtime = await ModelRuntime.create(
            auth_path=os.path.join(agent_dir, "auth.json"),
            models_path=os.path.join(agent_dir, "models.json"),
        )
    settings_manager = (
        options.settings_manager if options.settings_manager is not None else SettingsManager.create(cwd, agent_dir)
    )
    resource_loader = DefaultResourceLoader(
        **(options.resource_loader_options or {}),
        cwd=cwd,
        agent_dir=agent_dir,
        settings_manager=settings_manager,
    )
    await resource_loader.reload(**(options.resource_loader_reload_options or {}))

    diagnostics: list[AgentSessionRuntimeDiagnostic] = []
    extensions_result = resource_loader.get_extensions()
    for registration in extensions_result.runtime.pending_provider_registrations:
        try:
            model_runtime.register_provider(registration["name"], registration["config"])
        except Exception as error:
            diagnostics.append(
                AgentSessionRuntimeDiagnostic(
                    type="error",
                    message=f'Extension "{registration.get("extension_path")}" error: {error}',
                )
            )
    extensions_result.runtime.pending_provider_registrations = []
    for registration in extensions_result.runtime.pending_native_provider_registrations:
        try:
            model_runtime.register_native_provider(registration["provider"])
        except Exception as error:
            diagnostics.append(
                AgentSessionRuntimeDiagnostic(
                    type="error",
                    message=f'Extension "{registration.get("extension_path")}" error: {error}',
                )
            )
    extensions_result.runtime.pending_native_provider_registrations = []
    from .model_runtime import ModelsRefreshOptions

    await model_runtime.refresh(ModelsRefreshOptions(allow_network=False))
    diagnostics.extend(_apply_extension_flag_values(resource_loader, options.extension_flag_values))

    return AgentSessionServices(
        cwd=cwd,
        agent_dir=agent_dir,
        model_runtime=model_runtime,
        settings_manager=settings_manager,
        resource_loader=resource_loader,
        diagnostics=diagnostics,
    )


async def create_agent_session_from_services(
    options: CreateAgentSessionFromServicesOptions,
) -> CreateAgentSessionResult:
    """Create an AgentSession from previously created services. Keeps session
    creation separate from service creation so callers can resolve model,
    thinking, tools, and other session inputs against the target cwd first."""
    return await create_agent_session(
        CreateAgentSessionOptions(
            cwd=options.services.cwd,
            agent_dir=options.services.agent_dir,
            model_runtime=options.services.model_runtime,
            settings_manager=options.services.settings_manager,
            resource_loader=options.services.resource_loader,
            session_manager=options.session_manager,
            model=options.model,
            thinking_level=options.thinking_level,
            scoped_models=options.scoped_models,
            tools=options.tools,
            exclude_tools=options.exclude_tools,
            no_tools=options.no_tools,
            custom_tools=options.custom_tools,
            session_start_event=options.session_start_event,
        )
    )
