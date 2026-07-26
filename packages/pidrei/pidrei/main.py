"""Mirror of pi coding-agent src/main.ts.

Main entry point for the coding agent CLI. This file handles CLI argument
parsing and translates it into createAgentSession() options; the SDK does
the heavy lifting.

Port notes (Phase 3):
- POSIX-only: pi's win32 branches (self-update quarantine cleanup) are not
  ported.
- The `install/remove/uninstall/update/list/config` subcommands live in
  `cli/package_commands.py` and are dispatched below, before the argument
  parse. pi's self-update half is deliberately absent (Phase 7 step 6).
- migrations.ts is not ported: all migrations are pi-version-legacy
  cleanups a fresh ~/.pidrei/ cannot contain (see PLAN); the interactive
  deprecation-warning display that reads its output is skipped with it.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Any

import tonio.colored as tonio

from .cli.args import Args, parse_args, print_help
from .cli.file_processor import process_file_arguments
from .cli.initial_message import InitialMessageResult, build_initial_message
from .cli.list_models import list_models
from .cli.package_commands import handle_config_command, handle_package_command
from .cli.project_trust import CreateProjectTrustContextOptions, create_project_trust_context
from .cli.session_picker import select_session
from .cli.startup_ui import should_run_first_time_setup, show_first_time_setup, show_startup_selector
from .config import ENV_SESSION_DIR, VERSION, expand_tilde_path, get_agent_dir
from .core.agent_session_runtime import CreateAgentSessionRuntimeResult, create_agent_session_runtime
from .core.agent_session_services import (
    AgentSessionRuntimeDiagnostic,
    CreateAgentSessionFromServicesOptions,
    CreateAgentSessionServicesOptions,
    create_agent_session_from_services,
    create_agent_session_services,
)
from .core.auth_guidance import format_no_models_available_message
from .core.http_config import apply_http_proxy_settings
from .core.model_resolver import resolve_cli_model, resolve_model_scope
from .core.output_guard import restore_stdout, take_over_stdout
from .core.project_trust import ResolveProjectTrustedOptions, resolve_project_trusted
from .core.sdk import CreateAgentSessionOptions
from .core.session_cwd import (
    MissingSessionCwdError,
    format_missing_session_cwd_prompt,
    get_missing_session_cwd_issue,
)
from .core.session_manager import SessionManager, assert_valid_session_id
from .core.settings_manager import SettingsManager
from .core.timings import print_timings, reset_timings, time
from .core.trust_manager import ProjectTrustStore, has_trust_requiring_project_resources
from .extensions import builtin_extensions
from .modes import run_print_mode, run_rpc_mode
from .modes.interactive.theme import init_theme, stop_theme_watcher
from .modes.print_mode import PrintModeOptions
from .utils.colors import dim, red, yellow
from .utils.paths import is_local_path, normalize_path, resolve_path


EXTENSION_LOAD_FAILURE_HINT = 'Hint: Start without extensions using "pidrei -ne".'

from pidrei_ai.registry import ModelsRefreshOptions, models_are_equal


async def _read_piped_stdin() -> str | None:
    """Read all content from piped stdin.

    Returns None if stdin is a TTY (interactive terminal)."""
    if sys.stdin.isatty():
        return None

    data = await tonio.spawn_blocking(sys.stdin.read)
    return data.strip() or None


def _collect_settings_diagnostics(
    settings_manager: SettingsManager, context: str
) -> list[AgentSessionRuntimeDiagnostic]:
    return [
        AgentSessionRuntimeDiagnostic(
            type="warning",
            message=f"({context}, {entry.scope} settings) {entry.error}",
        )
        for entry in settings_manager.drain_errors()
    ]


def _report_diagnostics(diagnostics: list[AgentSessionRuntimeDiagnostic]) -> None:
    for diagnostic in diagnostics:
        color = red if diagnostic.type == "error" else yellow if diagnostic.type == "warning" else dim
        prefix = "Error: " if diagnostic.type == "error" else "Warning: " if diagnostic.type == "warning" else ""
        print(color(f"{prefix}{diagnostic.message}"), file=sys.stderr)


def _is_truthy_env_flag(value: str | None) -> bool:
    if not value:
        return False
    return value == "1" or value.lower() in ("true", "yes")


def _resolve_app_mode(parsed: Args, stdin_is_tty: bool, stdout_is_tty: bool) -> str:
    if parsed.mode == "rpc":
        return "rpc"
    if parsed.mode == "json":
        return "json"
    if parsed.print or not stdin_is_tty or not stdout_is_tty:
        return "print"
    return "interactive"


def _to_print_output_mode(app_mode: str) -> str:
    return "json" if app_mode == "json" else "text"


def _is_plain_runtime_metadata_command(parsed: Args) -> bool:
    return not parsed.print and parsed.mode is None and (parsed.help is True or parsed.list_models is not None)


async def _prepare_initial_message(
    parsed: Args, auto_resize_images: bool, stdin_content: str | None = None
) -> InitialMessageResult:
    if not parsed.file_args:
        return build_initial_message(parsed=parsed, stdin_content=stdin_content)

    processed = await tonio.spawn_blocking(
        lambda: process_file_arguments(parsed.file_args, auto_resize_images=auto_resize_images)
    )
    return build_initial_message(
        parsed=parsed,
        file_text=processed.text,
        file_images=processed.images,
        stdin_content=stdin_content,
    )


async def _find_local_session_by_exact_id(session_id: str, cwd: str, session_dir: str | None) -> dict[str, str] | None:
    local_sessions = await SessionManager.list(cwd, session_dir)
    local_match = next((s for s in local_sessions if s.id == session_id), None)
    return {"type": "local", "path": local_match.path} if local_match else None


async def _resolve_session_path(session_arg: str, cwd: str, session_dir: str | None) -> dict[str, str]:
    """Resolve a session argument to a file path.

    If it looks like a path, use as-is. Otherwise try to match as a session
    ID prefix, locally first, then globally across projects."""
    if "/" in session_arg or "\\" in session_arg or session_arg.endswith(".jsonl"):
        return {"type": "path", "path": resolve_path(session_arg, cwd)}

    local_sessions = await SessionManager.list(cwd, session_dir)
    local_match = next((s for s in local_sessions if s.id == session_arg), None) or next(
        (s for s in local_sessions if s.id.startswith(session_arg)), None
    )

    if local_match:
        return {"type": "local", "path": local_match.path}

    all_sessions = await SessionManager.list_all(session_dir)
    global_match = next((s for s in all_sessions if s.id == session_arg), None) or next(
        (s for s in all_sessions if s.id.startswith(session_arg)), None
    )

    if global_match:
        return {"type": "global", "path": global_match.path, "cwd": global_match.cwd}

    return {"type": "not_found", "arg": session_arg}


async def _prompt_confirm(message: str) -> bool:
    """Prompt user for yes/no confirmation."""
    print(f"{message} [y/N] ", end="", flush=True)
    answer = (await tonio.spawn_blocking(sys.stdin.readline)).strip().lower()
    return answer in ("y", "yes")


def _validate_fork_flags(parsed: Args) -> None:
    if not parsed.fork:
        return

    conflicting_flags = [
        flag
        for flag, present in (
            ("--session", parsed.session),
            ("--continue", parsed.continue_),
            ("--resume", parsed.resume),
            ("--no-session", parsed.no_session),
        )
        if present
    ]

    if conflicting_flags:
        print(red(f"Error: --fork cannot be combined with {', '.join(conflicting_flags)}"), file=sys.stderr)
        raise SystemExit(1)


def _validate_session_id_flags(parsed: Args) -> None:
    if parsed.session_id is None:
        return

    conflicting_flags = [
        flag
        for flag, present in (
            ("--session", parsed.session),
            ("--continue", parsed.continue_),
            ("--resume", parsed.resume),
        )
        if present
    ]

    if conflicting_flags:
        print(red(f"Error: --session-id cannot be combined with {', '.join(conflicting_flags)}"), file=sys.stderr)
        raise SystemExit(1)

    try:
        assert_valid_session_id(parsed.session_id)
    except Exception as error:
        print(red(f"Error: {error}"), file=sys.stderr)
        raise SystemExit(1) from None


def _open_session_or_exit(path: str, session_dir: str | None) -> SessionManager:
    try:
        return SessionManager.open(path, session_dir)
    except Exception as error:
        print(red(f"Error: {error}"), file=sys.stderr)
        raise SystemExit(1) from None


def _fork_session_or_exit(
    source_path: str, cwd: str, session_dir: str | None, session_id: str | None = None
) -> SessionManager:
    try:
        return SessionManager.fork_from(source_path, cwd, session_dir, {"id": session_id})
    except Exception as error:
        print(red(f"Error: {error}"), file=sys.stderr)
        raise SystemExit(1) from None


async def _create_session_manager(
    parsed: Args, cwd: str, session_dir: str | None, settings_manager: SettingsManager
) -> SessionManager:
    if parsed.no_session or parsed.help or parsed.list_models is not None:
        return SessionManager.in_memory(cwd, {"id": parsed.session_id} if parsed.session_id is not None else None)

    if parsed.fork:
        if parsed.session_id:
            existing_target = await _find_local_session_by_exact_id(parsed.session_id, cwd, session_dir)
            if existing_target:
                print(red(f"Session already exists with id '{parsed.session_id}'"), file=sys.stderr)
                raise SystemExit(1)

        resolved = await _resolve_session_path(parsed.fork, cwd, session_dir)

        if resolved["type"] in ("path", "local", "global"):
            return _fork_session_or_exit(resolved["path"], cwd, session_dir, parsed.session_id)

        print(red(f"No session found matching '{resolved['arg']}'"), file=sys.stderr)
        raise SystemExit(1)

    if parsed.session:
        resolved = await _resolve_session_path(parsed.session, cwd, session_dir)

        if resolved["type"] in ("path", "local"):
            return _open_session_or_exit(resolved["path"], session_dir)

        if resolved["type"] == "global":
            print(yellow(f"Session found in different project: {resolved['cwd']}"))
            should_fork = await _prompt_confirm("Fork this session into current directory?")
            if not should_fork:
                print(dim("Aborted."))
                raise SystemExit(0)
            return _fork_session_or_exit(resolved["path"], cwd, session_dir)

        print(red(f"No session found matching '{resolved['arg']}'"), file=sys.stderr)
        raise SystemExit(1)

    if parsed.resume:
        selected_path = await select_session(
            lambda on_progress: SessionManager.list(cwd, session_dir, on_progress),
            lambda on_progress: SessionManager.list_all(session_dir, on_progress),
            settings_manager,
        )
        if not selected_path:
            print(dim("No session selected"))
            raise SystemExit(0)
        return SessionManager.open(selected_path, session_dir)

    if parsed.continue_:
        return SessionManager.continue_recent(cwd, session_dir)

    if parsed.session_id:
        existing_session = await _find_local_session_by_exact_id(parsed.session_id, cwd, session_dir)
        if existing_session:
            return SessionManager.open(existing_session["path"], session_dir)
        print(
            yellow(
                f"Warning: No project session found with id '{parsed.session_id}'; creating a new session with that id."
            ),
            file=sys.stderr,
        )

    return SessionManager.create(cwd, session_dir, {"id": parsed.session_id})


@dataclass(slots=True)
class _SessionOptionsResult:
    options: CreateAgentSessionOptions
    cli_thinking_from_model: bool
    diagnostics: list[AgentSessionRuntimeDiagnostic] = field(default_factory=list)


def _build_session_options(
    parsed: Args,
    scoped_models: list[Any],
    has_existing_session: bool,
    model_runtime: Any,
    settings_manager: SettingsManager,
) -> _SessionOptionsResult:
    options = CreateAgentSessionOptions()
    diagnostics: list[AgentSessionRuntimeDiagnostic] = []
    cli_thinking_from_model = False

    # Model from CLI
    # - supports --provider <name> --model <pattern>
    # - supports --model <provider>/<pattern>
    if parsed.model:
        resolved = resolve_cli_model(
            cli_provider=parsed.provider,
            cli_model=parsed.model,
            cli_thinking=parsed.thinking,
            model_runtime=model_runtime,
        )
        if resolved.warning:
            diagnostics.append(AgentSessionRuntimeDiagnostic(type="warning", message=resolved.warning))
        if resolved.error:
            diagnostics.append(AgentSessionRuntimeDiagnostic(type="error", message=resolved.error))
        if resolved.model is not None:
            options.model = resolved.model
            # Allow "--model <pattern>:<thinking>" as a shorthand.
            # Explicit --thinking still takes precedence (applied later).
            if not parsed.thinking and resolved.thinking_level:
                options.thinking_level = resolved.thinking_level
                cli_thinking_from_model = True

    if options.model is None and scoped_models and not has_existing_session:
        # Check if saved default is in scoped models - use it if so, otherwise first scoped model
        saved_provider = settings_manager.get_default_provider()
        saved_model_id = settings_manager.get_default_model()
        saved_model = (
            model_runtime.get_model(saved_provider, saved_model_id) if saved_provider and saved_model_id else None
        )
        saved_in_scope = (
            next((sm for sm in scoped_models if models_are_equal(sm.model, saved_model)), None)
            if saved_model is not None
            else None
        )

        if saved_in_scope is not None:
            options.model = saved_in_scope.model
            # Use thinking level from scoped model config if explicitly set
            if not parsed.thinking and saved_in_scope.thinking_level:
                options.thinking_level = saved_in_scope.thinking_level
        else:
            options.model = scoped_models[0].model
            # Use thinking level from first scoped model if explicitly set
            if not parsed.thinking and scoped_models[0].thinking_level:
                options.thinking_level = scoped_models[0].thinking_level

    # Thinking level from CLI (takes precedence over scoped model thinking levels set above)
    if parsed.thinking:
        options.thinking_level = parsed.thinking

    # Scoped models for Ctrl+P cycling
    # Keep thinking level unset when not explicitly set in the model pattern.
    # Unset means "inherit current session thinking level" during cycling.
    if scoped_models:
        options.scoped_models = list(scoped_models)

    # API key from CLI - set as a non-persistent runtime override
    # (handled by caller before createAgentSession)

    # Tools
    if parsed.no_tools:
        options.no_tools = "all"
    elif parsed.no_builtin_tools:
        options.no_tools = "builtin"
    if parsed.tools:
        options.tools = list(parsed.tools)
    if parsed.exclude_tools:
        options.exclude_tools = list(parsed.exclude_tools)

    return _SessionOptionsResult(
        options=options, cli_thinking_from_model=cli_thinking_from_model, diagnostics=diagnostics
    )


async def _prompt_for_missing_session_cwd(issue, settings_manager: SettingsManager) -> str | None:
    return await show_startup_selector(
        settings_manager,
        format_missing_session_cwd_prompt(issue),
        [
            {"label": "Continue", "value": issue.fallback_cwd},
            {"label": "Cancel", "value": None},
        ],
    )


def _resolve_cli_paths(cwd: str, paths: list[str] | None) -> list[str] | None:
    if paths is None:
        return None
    return [resolve_path(value, cwd) if is_local_path(value) else value for value in paths]


async def main(args: list[str], *, extension_factories: list[Any] | None = None) -> int:
    """Run the CLI. Returns the process exit code.

    pi calls process.exit() throughout main; the port raises SystemExit in
    the same places and converts it to a return value here so the exception
    never crosses the tonio runtime boundary."""
    try:
        await _main(args, extension_factories=extension_factories)
        return 0
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0
    except BaseExceptionGroup as group:
        system_exit = next((exc for exc in group.exceptions if isinstance(exc, SystemExit)), None)
        if system_exit is not None:
            return system_exit.code if isinstance(system_exit.code, int) else 0
        raise


async def _main(args: list[str], *, extension_factories: list[Any] | None = None) -> None:
    reset_timings()
    extension_factories = [*builtin_extensions(), *(extension_factories or [])]
    offline_mode = "--offline" in args or _is_truthy_env_flag(os.environ.get("PIDREI_OFFLINE"))
    if offline_mode:
        os.environ["PIDREI_OFFLINE"] = "1"

    cwd = os.getcwd()
    agent_dir = get_agent_dir()
    bootstrap_settings_manager = SettingsManager.create(cwd, agent_dir, project_trusted=False)
    apply_http_proxy_settings(bootstrap_settings_manager.get_global_settings().get("httpProxy"))

    # The package subcommands run and exit here, before the normal argument
    # parse, exactly as pi dispatches them (package-manager-cli.ts).
    handled = await handle_package_command(args, extension_factories=extension_factories)
    if handled is False:
        handled = await handle_config_command(args, extension_factories=extension_factories)
    if handled is not False:
        raise SystemExit(handled)

    parsed = parse_args(args)
    if parsed.diagnostics:
        for d in parsed.diagnostics:
            color = red if d["type"] == "error" else yellow
            label = "Error" if d["type"] == "error" else "Warning"
            print(color(f"{label}: {d['message']}"), file=sys.stderr)
        if any(d["type"] == "error" for d in parsed.diagnostics):
            raise SystemExit(1)
    time("parseArgs")

    if parsed.version:
        print(VERSION)
        raise SystemExit(0)

    if parsed.export:
        # lazy: import cycle within core
        from .core.export_html import export_from_file

        try:
            output_path = parsed.messages[0] if parsed.messages else None
            result = await export_from_file(parsed.export, output_path)
        except Exception as error:
            print(red(f"Error: {error}"), file=sys.stderr)
            raise SystemExit(1) from None
        print(f"Exported to: {result}")
        raise SystemExit(0)

    app_mode = _resolve_app_mode(parsed, sys.stdin.isatty(), sys.stdout.isatty())
    should_take_over_stdout = app_mode != "interactive" and not _is_plain_runtime_metadata_command(parsed)
    if should_take_over_stdout:
        take_over_stdout()

    if parsed.mode == "rpc" and parsed.file_args:
        print(red("Error: @file arguments are not supported in RPC mode"), file=sys.stderr)
        raise SystemExit(1)

    _validate_fork_flags(parsed)
    _validate_session_id_flags(parsed)

    # pi runs config migrations here; migrations.ts is not ported (a fresh
    # ~/.pidrei/ cannot contain the pi-version-legacy state they clean up).
    time("runMigrations")

    startup_settings_manager = SettingsManager.create(cwd, agent_dir)
    _report_diagnostics(_collect_settings_diagnostics(startup_settings_manager, "startup session lookup"))

    # Experimental first-time setup: theme choice and analytics opt-in.
    # Runs before any runtime services are created so the chosen settings
    # apply everywhere.
    if app_mode == "interactive" and not parsed.help and parsed.list_models is None and should_run_first_time_setup():
        await show_first_time_setup(startup_settings_manager)
        time("firstTimeSetup")

    # Decide the final runtime cwd before creating cwd-bound runtime services.
    # --session and --resume may select a session from another project, so project-local
    # settings, resources, provider registrations, and models must be resolved only after
    # the target session cwd is known. The startup-cwd settings manager is used only for
    # sessionDir lookup during session selection.
    env_session_dir = os.environ.get(ENV_SESSION_DIR)
    session_dir = (
        (normalize_path(parsed.session_dir) if parsed.session_dir else None)
        or (expand_tilde_path(env_session_dir) if env_session_dir else None)
        or startup_settings_manager.get_session_dir()
    )
    session_manager = await _create_session_manager(parsed, cwd, session_dir, startup_settings_manager)
    missing_session_cwd_issue = get_missing_session_cwd_issue(session_manager, cwd)
    if missing_session_cwd_issue:
        if app_mode == "interactive":
            selected_cwd = await _prompt_for_missing_session_cwd(missing_session_cwd_issue, startup_settings_manager)
            if not selected_cwd:
                raise SystemExit(0)
            session_manager = SessionManager.open(missing_session_cwd_issue.session_file, session_dir, selected_cwd)
        else:
            print(red(str(MissingSessionCwdError(missing_session_cwd_issue))), file=sys.stderr)
            raise SystemExit(1)
    if parsed.name is not None:
        name = parsed.name.strip()
        if not name:
            print(red("Error: --name requires a non-empty value"), file=sys.stderr)
            raise SystemExit(1)
        session_manager.append_session_info(name)
    time("createSessionManager")

    trust_store = ProjectTrustStore(agent_dir)
    session_cwd = session_manager.get_cwd()
    auto_trust_on_reload_cwd = (
        session_cwd
        if parsed.project_trust_override is None and not has_trust_requiring_project_resources(session_cwd)
        else None
    )
    trust_prompt_mode = "print" if parsed.help or parsed.list_models is not None else app_mode
    project_trust_by_cwd: dict[str, bool] = {}

    resolved_extension_paths = _resolve_cli_paths(cwd, parsed.extensions)
    resolved_skill_paths = _resolve_cli_paths(cwd, parsed.skills)
    resolved_prompt_template_paths = _resolve_cli_paths(cwd, parsed.prompt_templates)
    resolved_theme_paths = _resolve_cli_paths(cwd, parsed.themes)

    async def create_runtime(
        *,
        cwd: str,
        agent_dir: str,
        session_manager: SessionManager,
        session_start_event: dict[str, Any] | None = None,
        project_trust_context: Any = None,
    ):
        is_initial_runtime = session_start_event is None
        project_trust_diagnostics: list[AgentSessionRuntimeDiagnostic] = []
        cached_project_trust = project_trust_by_cwd.get(cwd)
        has_trust_requiring_resources = has_trust_requiring_project_resources(cwd)
        should_resolve_project_trust = (
            parsed.project_trust_override is None and cached_project_trust is None and has_trust_requiring_resources
        )
        if should_resolve_project_trust:
            project_trusted = False
        elif cached_project_trust is not None:
            project_trusted = cached_project_trust
        elif parsed.project_trust_override is not None:
            project_trusted = parsed.project_trust_override
        else:
            project_trusted = not has_trust_requiring_resources or trust_store.get(cwd) is True
        runtime_settings_manager = SettingsManager.create(cwd, agent_dir, project_trusted=project_trusted)

        async def resolve_trust(extensions_result=None) -> bool:
            trusted = await resolve_project_trusted(
                ResolveProjectTrustedOptions(
                    cwd=cwd,
                    trust_store=trust_store,
                    trust_override=parsed.project_trust_override,
                    default_project_trust=startup_settings_manager.get_default_project_trust(),
                    extensions_result=extensions_result,
                    project_trust_context=(
                        project_trust_context
                        or create_project_trust_context(
                            CreateProjectTrustContextOptions(
                                cwd=cwd,
                                mode=trust_prompt_mode if is_initial_runtime else app_mode,
                                settings_manager=startup_settings_manager,
                                has_ui=is_initial_runtime and trust_prompt_mode == "interactive",
                            )
                        )
                    ),
                    on_extension_error=lambda message: project_trust_diagnostics.append(
                        AgentSessionRuntimeDiagnostic(type="warning", message=message)
                    ),
                )
            )
            project_trust_by_cwd[cwd] = trusted
            return trusted

        services = await create_agent_session_services(
            CreateAgentSessionServicesOptions(
                cwd=cwd,
                agent_dir=agent_dir,
                settings_manager=runtime_settings_manager,
                extension_flag_values=dict(parsed.unknown_flags),
                resource_loader_reload_options=(
                    {"resolve_project_trust": resolve_trust} if should_resolve_project_trust else None
                ),
                resource_loader_options={
                    "extension_factories": extension_factories,
                    "additional_extension_paths": resolved_extension_paths,
                    "additional_skill_paths": resolved_skill_paths,
                    "additional_prompt_template_paths": resolved_prompt_template_paths,
                    "additional_theme_paths": resolved_theme_paths,
                    "no_extensions": bool(parsed.no_extensions),
                    "no_skills": bool(parsed.no_skills),
                    "no_prompt_templates": bool(parsed.no_prompt_templates),
                    "no_themes": bool(parsed.no_themes),
                    "no_context_files": bool(parsed.no_context_files),
                    "system_prompt": parsed.system_prompt,
                    "append_system_prompt": parsed.append_system_prompt,
                },
            )
        )
        settings_manager = services.settings_manager
        model_runtime = services.model_runtime
        resource_loader = services.resource_loader
        diagnostics: list[AgentSessionRuntimeDiagnostic] = [
            *project_trust_diagnostics,
            *services.diagnostics,
            *_collect_settings_diagnostics(settings_manager, "runtime creation"),
            *(
                AgentSessionRuntimeDiagnostic(
                    type="error", message=f'Failed to load extension "{err.path}": {err.error}'
                )
                for err in resource_loader.get_extensions().errors
            ),
        ]

        model_patterns = parsed.models if parsed.models is not None else settings_manager.get_enabled_models()
        scoped_models = await resolve_model_scope(model_patterns, model_runtime) if model_patterns else []
        session_options_result = _build_session_options(
            parsed,
            scoped_models,
            len(session_manager.build_session_context().messages) > 0,
            model_runtime,
            settings_manager,
        )
        session_options = session_options_result.options
        diagnostics.extend(session_options_result.diagnostics)

        if parsed.api_key:
            if session_options.model is None:
                diagnostics.append(
                    AgentSessionRuntimeDiagnostic(
                        type="error",
                        message="--api-key requires a model to be specified via --model, --provider/--model, or --models",
                    )
                )
            else:
                await model_runtime.set_runtime_api_key(
                    session_options.model.provider, parsed.api_key, ModelsRefreshOptions(allow_network=False)
                )
                await services.model_runtime.get_available()

        created = await create_agent_session_from_services(
            CreateAgentSessionFromServicesOptions(
                services=services,
                session_manager=session_manager,
                session_start_event=session_start_event,
                model=session_options.model,
                thinking_level=session_options.thinking_level,
                scoped_models=session_options.scoped_models,
                tools=session_options.tools,
                exclude_tools=session_options.exclude_tools,
                no_tools=session_options.no_tools,
                custom_tools=session_options.custom_tools,
            )
        )
        cli_thinking_override = parsed.thinking is not None or session_options_result.cli_thinking_from_model
        if created.session.model is not None and cli_thinking_override:
            created.session.set_thinking_level(created.session.thinking_level)

        return CreateAgentSessionRuntimeResult(
            session=created.session,
            services=services,
            extensions_result=created.extensions_result,
            model_fallback_message=created.model_fallback_message,
            diagnostics=diagnostics,
        )

    time("createRuntime")
    runtime = await create_agent_session_runtime(
        create_runtime,
        cwd=session_manager.get_cwd(),
        agent_dir=agent_dir,
        session_manager=session_manager,
    )
    time("createAgentSessionRuntime")
    services = runtime.services
    session = runtime.session
    settings_manager = services.settings_manager
    model_runtime = services.model_runtime
    resource_loader = services.resource_loader
    apply_http_proxy_settings(settings_manager.get_global_settings().get("httpProxy"))

    if parsed.help:
        extension_flags = [
            flag for extension in resource_loader.get_extensions().extensions for flag in extension.flags.values()
        ]
        print_help(extension_flags)
        raise SystemExit(0)

    if parsed.list_models is not None:
        search_pattern = parsed.list_models if isinstance(parsed.list_models, str) else None
        await list_models(model_runtime, search_pattern)
        raise SystemExit(0)

    # Read piped stdin content (if any) - skip for RPC mode which uses stdin for JSON-RPC
    stdin_content: str | None = None
    if app_mode != "rpc":
        stdin_content = await _read_piped_stdin()
        if stdin_content is not None and app_mode == "interactive":
            app_mode = "print"
    time("readPipedStdin")

    initial = await _prepare_initial_message(parsed, settings_manager.get_image_auto_resize(), stdin_content)
    time("prepareInitialMessage")
    init_theme(settings_manager.get_theme(), app_mode == "interactive")
    time("initTheme")

    # pi shows deprecation warnings from runMigrations here in interactive
    # mode; migrations.ts is not ported (see module docstring).

    time("resolveModelScope")
    _report_diagnostics(runtime.diagnostics)
    if any(diagnostic.type == "error" for diagnostic in runtime.diagnostics):
        if any("Failed to load extension" in diagnostic.message for diagnostic in runtime.diagnostics):
            print(yellow(EXTENSION_LOAD_FAILURE_HINT), file=sys.stderr)
        raise SystemExit(1)
    time("createAgentSession")

    if app_mode != "interactive" and session.model is None:
        print(red(format_no_models_available_message()), file=sys.stderr)
        raise SystemExit(1)

    startup_benchmark = _is_truthy_env_flag(os.environ.get("PIDREI_STARTUP_BENCHMARK"))
    if startup_benchmark and app_mode != "interactive":
        print(red("Error: PIDREI_STARTUP_BENCHMARK only supports interactive mode"), file=sys.stderr)
        raise SystemExit(1)

    # RPC refreshes catalogs here in the background; interactive mode starts
    # its refresh after TUI initialization.
    if not offline_mode and app_mode == "rpc":

        async def _background_refresh() -> None:
            try:
                await model_runtime.refresh()
            except Exception:
                pass

        tonio.spawn.without_tracking(_background_refresh())

    if app_mode == "rpc":
        print_timings()
        await run_rpc_mode(runtime)
    elif app_mode == "interactive":
        # lazy: core <-> modes import cycle (see modes/__init__.py)
        from .modes import InteractiveMode

        interactive_mode = InteractiveMode(
            runtime,
            {
                # migrations.ts is not ported, so there are never migrated
                # providers to announce.
                "modelFallbackMessage": runtime.model_fallback_message,
                "autoTrustOnReloadCwd": auto_trust_on_reload_cwd,
                "initialMessage": initial.initial_message,
                "initialImages": initial.initial_images,
                "initialMessages": parsed.messages,
                "verbose": parsed.verbose,
            },
        )
        if startup_benchmark:
            await interactive_mode.init()
            time("interactiveMode.init")
            # Give the TUI's stdin handler a brief chance to consume terminal
            # query replies (Kitty keyboard protocol, device attributes, cell
            # size) before restoring the terminal.
            await tonio.time.sleep(0.15)
            await interactive_mode.stop()
            stop_theme_watcher()
            print_timings()
            return

        print_timings()
        await interactive_mode.run()
    else:
        print_timings()
        exit_code = await run_print_mode(
            runtime,
            PrintModeOptions(
                mode=_to_print_output_mode(app_mode),
                messages=parsed.messages,
                initial_message=initial.initial_message,
                initial_images=initial.initial_images,
            ),
        )
        stop_theme_watcher()
        restore_stdout()
        if exit_code != 0:
            raise SystemExit(exit_code)
        return
