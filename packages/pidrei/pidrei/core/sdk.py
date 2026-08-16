"""Mirror of pi coding-agent src/core/sdk.ts.

createAgentSession and its option plumbing. pi's module-import side effect
`setDefaultStreamFn(streamSimple)` (a pre-0.81 compat fallback for extensions
constructing bare Agents) is not ported: pidrei never ported the pi-ai compat
global registry, and every Agent this package creates receives an explicit
runtime-bound stream function.
"""

import os
from dataclasses import dataclass, replace
from typing import Any

from pidrei_agent.agent import Agent, AgentInitialState
from pidrei_ai.registry import clamp_thinking_level
from pidrei_ai.types import Model, TextContent

from ..config import get_agent_dir
from ..utils.paths import resolve_path
from .agent_session import AgentSession, AgentSessionConfig, ScopedModel
from .auth_guidance import format_no_models_available_message
from .defaults import DEFAULT_THINKING_LEVEL
from .extensions import LoadExtensionsResult
from .messages import convert_to_llm
from .model_resolver import find_initial_model
from .model_runtime import ModelRuntime
from .provider_attribution import merge_provider_attribution_headers
from .resource_loader import DefaultResourceLoader
from .session_manager import SessionManager, get_default_session_dir
from .settings_manager import SettingsManager
from .timings import time
from .tools import ALL_TOOL_NAMES  # noqa: F401  (re-export surface parity)


@dataclass(slots=True, kw_only=True)
class CreateAgentSessionOptions:
    # Working directory for project-local discovery. Default: os.getcwd()
    cwd: str | None = None
    # Global config directory. Default: ~/.pidrei/agent
    agent_dir: str | None = None

    # Canonical model/auth runtime. Defaults to a runtime using agent_dir files.
    model_runtime: ModelRuntime | None = None

    # Model to use. Default: from settings, else first available
    model: Model | None = None
    # Thinking level. Default: from settings, else 'medium' (clamped to model capabilities)
    thinking_level: str | None = None
    # Models available for cycling
    scoped_models: list[ScopedModel] | None = None

    # Default tool suppression when no explicit allowlist is provided:
    # "all" starts with no tools enabled, "builtin" disables built-ins only.
    no_tools: str | None = None
    # Optional allowlist of tool names.
    tools: list[str] | None = None
    # Optional denylist of tool names (applies after `tools`).
    exclude_tools: list[str] | None = None
    # Custom tools to register (in addition to built-in tools).
    custom_tools: list[Any] | None = None

    # Resource loader. When omitted, DefaultResourceLoader is used.
    resource_loader: Any = None

    # Session manager. Default: SessionManager.create(cwd)
    session_manager: SessionManager | None = None

    # Settings manager. Default: SettingsManager.create(cwd, agent_dir)
    settings_manager: SettingsManager | None = None
    # Session start event metadata for extension runtime startup.
    session_start_event: dict[str, Any] | None = None


@dataclass(slots=True)
class CreateAgentSessionResult:
    # The created session
    session: AgentSession
    # Extensions result (for UI context setup in interactive mode)
    extensions_result: LoadExtensionsResult
    # Warning if session was restored with a different model than saved
    model_fallback_message: str | None = None


class _ExtensionRunnerRef:
    """Mutable ref used by the Agent stream fn to access the current runner."""

    __slots__ = ("current",)

    def __init__(self) -> None:
        self.current = None


def _convert_to_llm_with_block_images(settings_manager: SettingsManager):
    """convertToLlm wrapper that filters images if blockImages is enabled
    (defense in depth). Checks the setting dynamically so mid-session changes
    take effect."""

    async def convert(messages: list[Any]) -> list[Any]:
        converted = convert_to_llm(messages)
        if not settings_manager.get_block_images():
            return converted

        result = []
        for msg in converted:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", None)
            if role in ("user", "toolResult") and isinstance(content, list):
                has_images = any(getattr(block, "type", None) == "image" for block in content)
                if has_images:
                    replaced = [
                        TextContent(text="Image reading is disabled.")
                        if getattr(block, "type", None) == "image"
                        else block
                        for block in content
                    ]
                    # Dedupe consecutive "Image reading is disabled." texts
                    filtered: list[Any] = []
                    for block in replaced:
                        if (
                            getattr(block, "type", None) == "text"
                            and block.text == "Image reading is disabled."
                            and filtered
                            and getattr(filtered[-1], "type", None) == "text"
                            and filtered[-1].text == "Image reading is disabled."
                        ):
                            continue
                        filtered.append(block)
                    result.append(replace(msg, content=filtered))
                    continue
            result.append(msg)
        return result

    return convert


async def create_agent_session(options: CreateAgentSessionOptions | None = None) -> CreateAgentSessionResult:
    """Create an AgentSession with the specified options."""
    options = options if options is not None else CreateAgentSessionOptions()
    cwd = resolve_path(
        options.cwd
        if options.cwd is not None
        else (options.session_manager.get_cwd() if options.session_manager is not None else os.getcwd())
    )
    agent_dir = resolve_path(options.agent_dir) if options.agent_dir is not None else get_agent_dir()
    resource_loader = options.resource_loader

    auth_path = os.path.join(agent_dir, "auth.json") if options.agent_dir is not None else None
    models_path = os.path.join(agent_dir, "models.json") if options.agent_dir is not None else ...
    model_runtime = options.model_runtime
    if model_runtime is None:
        model_runtime = await ModelRuntime.create(auth_path=auth_path, models_path=models_path)

    settings_manager = (
        options.settings_manager
        if options.settings_manager is not None
        else await SettingsManager.create(cwd, agent_dir)
    )
    session_manager = (
        options.session_manager
        if options.session_manager is not None
        else await SessionManager.create(cwd, get_default_session_dir(cwd, agent_dir))
    )

    if resource_loader is None:
        resource_loader = DefaultResourceLoader(cwd=cwd, agent_dir=agent_dir, settings_manager=settings_manager)
        await resource_loader.reload()
        time("resourceLoader.reload")

    # Check if session has existing data to restore
    existing_session = session_manager.build_session_context()
    has_existing_session = len(existing_session.messages) > 0
    has_thinking_entry = any(entry.get("type") == "thinking_level_change" for entry in session_manager.get_branch())

    model = options.model
    model_fallback_message: str | None = None

    # If session has data, try to restore model from it
    if model is None and has_existing_session and existing_session.model is not None:
        restored_model = model_runtime.get_model(existing_session.model.provider, existing_session.model.model_id)
        if restored_model is not None and model_runtime.has_configured_auth(restored_model.provider):
            model = restored_model
        if model is None:
            model_fallback_message = (
                f"Could not restore model {existing_session.model.provider}/{existing_session.model.model_id}"
            )

    # If still no model, use find_initial_model (settings default, then provider defaults)
    if model is None:
        result = await find_initial_model(
            scoped_models=[],
            is_continuing=has_existing_session,
            default_provider=settings_manager.get_default_provider(),
            default_model_id=settings_manager.get_default_model(),
            default_thinking_level=settings_manager.get_default_thinking_level(),
            model_runtime=model_runtime,
        )
        model = result.model
        if model is None:
            model_fallback_message = format_no_models_available_message()
        elif model_fallback_message:
            model_fallback_message += f". Using {model.provider}/{model.id}"

    thinking_level = options.thinking_level

    # If session has data, restore thinking level from it
    if thinking_level is None and has_existing_session:
        if has_thinking_entry:
            thinking_level = existing_session.thinking_level
        else:
            default = settings_manager.get_default_thinking_level()
            thinking_level = default if default is not None else DEFAULT_THINKING_LEVEL

    # Fall back to settings default
    if thinking_level is None:
        default = settings_manager.get_default_thinking_level()
        thinking_level = default if default is not None else DEFAULT_THINKING_LEVEL

    # Clamp to model capabilities
    thinking_level = "off" if model is None else clamp_thinking_level(model, thinking_level)

    default_active_tool_names = ["read", "bash", "edit", "write"]
    configured_default_tool_names = settings_manager.get_default_tools()
    allowed_tool_names = options.tools if options.tools is not None else ([] if options.no_tools == "all" else None)
    excluded_tool_names = options.exclude_tools
    excluded_tool_name_set = set(excluded_tool_names) if excluded_tool_names is not None else None
    initial_active_tool_names = [
        name
        for name in (
            list(options.tools)
            if options.tools is not None
            else (
                []
                if options.no_tools
                else (
                    configured_default_tool_names
                    if configured_default_tool_names is not None
                    else default_active_tool_names
                )
            )
        )
        if excluded_tool_name_set is None or name not in excluded_tool_name_set
    ]

    extension_runner_ref = _ExtensionRunnerRef()

    async def stream_fn(request_model: Model, context: Any, stream_options: Any = None):
        provider_retry_settings = settings_manager.get_provider_retry_settings()
        http_idle_timeout_ms = settings_manager.get_http_idle_timeout_ms()
        # SDKs treat timeout=0 as 0ms (immediate timeout), not "no timeout".
        # Use max int32 to effectively disable the timeout.
        effective_timeout_ms = 2147483647 if http_idle_timeout_ms == 0 else http_idle_timeout_ms
        options_timeout = getattr(stream_options, "timeout_ms", None)
        settings_timeout = provider_retry_settings["timeout_ms"]
        timeout_ms = (
            options_timeout
            if options_timeout is not None
            else (settings_timeout if settings_timeout is not None else effective_timeout_ms)
        )
        options_ws_timeout = getattr(stream_options, "websocket_connect_timeout_ms", None)
        websocket_connect_timeout_ms = (
            options_ws_timeout
            if options_ws_timeout is not None
            else settings_manager.get_websocket_connect_timeout_ms()
        )
        header_runner = extension_runner_ref.current

        async def transform_headers(request_headers):
            headers = merge_provider_attribution_headers(
                request_model,
                settings_manager,
                getattr(stream_options, "session_id", None),
                request_headers,
            )
            if header_runner is not None and header_runner.has_handlers("before_provider_headers"):
                return await header_runner.emit_before_provider_headers(headers if headers is not None else {})
            return headers if headers is not None else {}

        options_max_retries = getattr(stream_options, "max_retries", None)
        options_max_retry_delay = getattr(stream_options, "max_retry_delay_ms", None)
        merged_options = replace(
            stream_options,
            timeout_ms=timeout_ms,
            websocket_connect_timeout_ms=websocket_connect_timeout_ms,
            max_retries=options_max_retries
            if options_max_retries is not None
            else provider_retry_settings["max_retries"],
            max_retry_delay_ms=(
                options_max_retry_delay
                if options_max_retry_delay is not None
                else provider_retry_settings["max_retry_delay_ms"]
            ),
            transform_headers=transform_headers,
        )
        return model_runtime.stream_simple(request_model, context, merged_options)

    async def on_payload(payload: Any, _model: Any = None) -> Any:
        runner = extension_runner_ref.current
        if runner is None or not runner.has_handlers("before_provider_request"):
            return payload
        return await runner.emit_before_provider_request(payload)

    async def on_response(response: Any, _model: Any = None) -> None:
        runner = extension_runner_ref.current
        if runner is None or not runner.has_handlers("after_provider_response"):
            return
        await runner.emit(
            {
                "type": "after_provider_response",
                "status": getattr(response, "status_code", getattr(response, "status", None)),
                "headers": getattr(response, "headers", None),
            }
        )

    async def transform_context(messages: list[Any], _cancel: Any = None) -> list[Any]:
        runner = extension_runner_ref.current
        if runner is None:
            return messages
        return await runner.emit_context(messages)

    agent = Agent(
        initial_state=AgentInitialState(
            system_prompt="",
            model=model,
            thinking_level=thinking_level,
            tools=[],
        ),
        convert_to_llm=_convert_to_llm_with_block_images(settings_manager),
        stream_fn=stream_fn,
        on_payload=on_payload,
        on_response=on_response,
        session_id=session_manager.get_session_id(),
        transform_context=transform_context,
        steering_mode=settings_manager.get_steering_mode(),
        follow_up_mode=settings_manager.get_follow_up_mode(),
        transport=settings_manager.get_transport(),
        thinking_budgets=settings_manager.get_thinking_budgets(),
        max_retry_delay_ms=settings_manager.get_provider_retry_settings()["max_retry_delay_ms"],
    )

    # Restore messages if session has existing data
    if has_existing_session:
        agent.state.messages = existing_session.messages
        if not has_thinking_entry:
            await session_manager.append_thinking_level_change(thinking_level)
    else:
        # Save initial model and thinking level for new sessions so they can be
        # restored on resume
        if model is not None:
            await session_manager.append_model_change(model.provider, model.id)
        await session_manager.append_thinking_level_change(thinking_level)

    session = AgentSession(
        AgentSessionConfig(
            agent=agent,
            session_manager=session_manager,
            settings_manager=settings_manager,
            cwd=cwd,
            scoped_models=options.scoped_models,
            resource_loader=resource_loader,
            custom_tools=options.custom_tools,
            model_runtime=model_runtime,
            initial_active_tool_names=initial_active_tool_names,
            allowed_tool_names=allowed_tool_names,
            excluded_tool_names=excluded_tool_names,
            extension_runner_ref=extension_runner_ref,
            session_start_event=options.session_start_event,
        )
    )
    extensions_result = resource_loader.get_extensions()

    return CreateAgentSessionResult(
        session=session,
        extensions_result=extensions_result,
        model_fallback_message=model_fallback_message,
    )
