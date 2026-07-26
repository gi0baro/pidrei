"""Port of pi coding-agent src/package-manager-cli.ts.

`install` / `remove` / `uninstall` / `update` / `list` / `config`, dispatched
before the normal argument parse because they never start a session.

**pidrei does not self-update** (decided 2026-07-27). pi's `update` defaults to
updating pi itself, shelling out to whichever package manager installed it —
npm, pnpm, yarn or bun. pidrei installs from git or Homebrew, where "update"
means re-running the install command with a new tag, and `uv tool upgrade` on a
tag-pinned git install re-resolves the same tag and silently does nothing. So
roughly 250 lines of pi's file — `getSelfUpdatePlan`, `runSelfUpdate`, the
install-method detection, the npm/pnpm fallback hints and the Windows
special-casing — are not ported, and the update targets are reinterpreted:

| pi                    | pidrei                                            |
|-----------------------|---------------------------------------------------|
| `update` (bare)       | extensions (pi: self, with a "skipped" note)      |
| `update --self`       | refused, with the install command to run instead  |
| `update self` / `pi`  | same refusal                                      |
| `update --all`        | extensions + model catalogs (pi: self+extensions)  |
| `update --extensions` | unchanged                                         |
| `update --models`     | unchanged                                         |
| `update --force`      | refused: it only ever forced a self-reinstall     |

Everything else — the parser's error precedence, the project-trust handling,
the output shapes — mirrors pi.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Any

from ..config import APP_NAME, CONFIG_DIR_NAME, get_agent_dir, get_auth_path, get_models_path
from ..core.package_manager import DefaultPackageManager
from ..core.project_trust import ResolveProjectTrustedOptions, resolve_project_trusted
from ..core.settings_manager import SettingsManager
from ..core.trust_manager import ProjectTrustStore, has_trust_requiring_project_resources
from ..utils.colors import bold, dim, green, red


PACKAGE_COMMANDS = ("install", "remove", "uninstall", "update", "list")
CONFIG_COMMAND_USAGE = f"{APP_NAME} config [-l] [--approve|--no-approve]"

#: How to actually update pidrei, since `update --self` will not.
SELF_UPDATE_HINT = (
    f"{APP_NAME} does not update itself. Re-run the install command with the new version:\n"
    f"  uv tool install -p 3.14t git+https://github.com/gi0baro/pidrei@<version>\n"
    f"  brew upgrade pidrei"
)


@dataclass(slots=True)
class PackageCommandOptions:
    command: str
    source: str | None = None
    update_target: str | None = None  # "extensions" | "models" | "all"
    update_source: str | None = None
    local: bool = False
    project_trust_override: bool | None = None
    help: bool = False
    invalid_option: str | None = None
    invalid_argument: str | None = None
    missing_option_value: str | None = None
    conflicting_options: str | None = None
    self_update_requested: bool = False


def get_package_command_usage(command: str) -> str:
    if command == "install":
        return f"{APP_NAME} install <source> [-l] [--approve|--no-approve]"
    if command == "remove":
        return f"{APP_NAME} remove <source> [-l] [--approve|--no-approve]"
    if command == "update":
        return (
            f"{APP_NAME} update [source] [--extensions|--models|--all] [--extension <source>] [--approve|--no-approve]"
        )
    return f"{APP_NAME} list [--approve|--no-approve]"


def print_config_command_help() -> None:
    print(f"""{bold("Usage:")}
  {CONFIG_COMMAND_USAGE}

Open the resource configuration TUI to enable or disable package resources.
Without -l, starts in global settings (~/{CONFIG_DIR_NAME}/agent/settings.json).
Press Tab in the TUI to switch between global and project-local modes.

Options:
  -l, --local       Edit project overrides ({CONFIG_DIR_NAME}/settings.json)
  -a, --approve     Trust project-local files for this command with -l
  -na, --no-approve Ignore project-local files for this command with -l
""")


def print_package_command_help(command: str) -> None:
    if command == "install":
        print(f"""{bold("Usage:")}
  {get_package_command_usage("install")}

Install a package and add it to settings.

Options:
  -l, --local       Install project-locally ({CONFIG_DIR_NAME}/settings.json)
  -a, --approve     Trust project-local files for this command
  -na, --no-approve Ignore project-local files for this command

Examples:
  {APP_NAME} install git:github.com/user/repo
  {APP_NAME} install git:git@github.com:user/repo
  {APP_NAME} install https://github.com/user/repo
  {APP_NAME} install ssh://git@github.com/user/repo
  {APP_NAME} install ./local/path
""")
        return

    if command == "remove":
        print(f"""{bold("Usage:")}
  {get_package_command_usage("remove")}

Remove a package and its source from settings.
Alias: {APP_NAME} uninstall <source> [-l]

Options:
  -l, --local       Remove from project settings ({CONFIG_DIR_NAME}/settings.json)
  -a, --approve     Trust project-local files for this command
  -na, --no-approve Ignore project-local files for this command

Examples:
  {APP_NAME} remove git:github.com/user/repo
  {APP_NAME} uninstall ./local/path
""")
        return

    if command == "update":
        print(f"""{bold("Usage:")}
  {get_package_command_usage("update")}

Update installed packages or refresh model catalogs.

{APP_NAME} does not update itself; see "Updating {APP_NAME}" below.

Options:
  --extensions            Update installed packages (the default)
  --models                Refresh model catalogs only
  --all                   Update packages and refresh model catalogs
  --extension <source>    Update one package only
  -a, --approve           Trust project-local files for this command
  -na, --no-approve       Ignore project-local files for this command

Short forms:
  {APP_NAME} update                Update all installed packages
  {APP_NAME} update <source>       Update one package
  {APP_NAME} update --models       Refresh model catalogs only
  {APP_NAME} update --all          Both

Updating {APP_NAME}:
  Re-run the install command with the new version.
  uv tool install -p 3.14t git+https://github.com/gi0baro/pidrei@<version>
  brew upgrade pidrei
""")
        return

    print(f"""{bold("Usage:")}
  {get_package_command_usage("list")}

List installed packages from user and project settings.

Options:
  -a, --approve      Trust project-local files for this command
  -na, --no-approve  Ignore project-local files for this command
""")


def parse_package_command(args: list[str]) -> PackageCommandOptions | None:
    if not args:
        return None
    raw_command, rest = args[0], args[1:]
    command = "remove" if raw_command == "uninstall" else raw_command
    if command not in ("install", "remove", "update", "list"):
        return None

    options = PackageCommandOptions(command=command)
    extensions_flag = False
    models_flag = False
    all_flag = False
    extension_flag_source: str | None = None

    index = 0
    while index < len(rest):
        arg = rest[index]
        index += 1

        if arg in ("-h", "--help"):
            options.help = True
        elif arg in ("-l", "--local"):
            if command in ("install", "remove"):
                options.local = True
            else:
                options.invalid_option = options.invalid_option or arg
        elif arg in ("--approve", "-a"):
            options.project_trust_override = True
        elif arg in ("--no-approve", "-na"):
            options.project_trust_override = False
        elif arg in ("--self", "--force"):
            # pi's self-update flags. Refused by name rather than ignored, so a
            # habit carried over from pi gets an answer instead of a no-op.
            if command == "update":
                options.self_update_requested = True
            else:
                options.invalid_option = options.invalid_option or arg
        elif arg == "--extensions":
            if command == "update":
                extensions_flag = True
            else:
                options.invalid_option = options.invalid_option or arg
        elif arg == "--models":
            if command == "update":
                models_flag = True
            else:
                options.invalid_option = options.invalid_option or arg
        elif arg == "--all":
            if command == "update":
                all_flag = True
            else:
                options.invalid_option = options.invalid_option or arg
        elif arg == "--extension":
            if command != "update":
                options.invalid_option = options.invalid_option or arg
                continue
            value = rest[index] if index < len(rest) else None
            if not value or value.startswith("-"):
                options.missing_option_value = options.missing_option_value or arg
            elif extension_flag_source:
                options.conflicting_options = options.conflicting_options or "--extension can only be provided once"
                index += 1
            else:
                extension_flag_source = value
                index += 1
        elif arg.startswith("-"):
            options.invalid_option = options.invalid_option or arg
        elif options.source is None:
            options.source = arg
        else:
            options.invalid_argument = options.invalid_argument or arg

    if command == "update":
        _resolve_update_target(options, extensions_flag, models_flag, all_flag, extension_flag_source)
    return options


def _resolve_update_target(
    options: PackageCommandOptions,
    extensions_flag: bool,
    models_flag: bool,
    all_flag: bool,
    extension_flag_source: str | None,
) -> None:
    def conflict(message: str) -> None:
        options.conflicting_options = options.conflicting_options or message

    if all_flag and (extensions_flag or models_flag or extension_flag_source):
        conflict("--all cannot be combined with --extensions, --models, or --extension")
    if all_flag and options.source:
        conflict("--all cannot be combined with a positional source")

    if options.source in ("self", "pi", APP_NAME):
        options.self_update_requested = True
        options.source = None

    if models_flag:
        if extensions_flag or all_flag or extension_flag_source:
            conflict("--models cannot be combined with --extensions, --all, or --extension")
        if options.source:
            conflict("--models cannot be combined with a positional source")
        options.update_target = "models"
    elif extension_flag_source:
        if extensions_flag or all_flag:
            conflict("--extension cannot be combined with --extensions or --all")
        if options.source:
            conflict("--extension cannot be combined with a positional source")
        options.update_target = "extensions"
        options.update_source = extension_flag_source
    elif options.source:
        if extensions_flag or all_flag:
            conflict("positional update targets cannot be combined with --extensions or --all")
        options.update_target = "extensions"
        options.update_source = options.source
    elif all_flag:
        options.update_target = "all"
    else:
        # pi defaults to self here; without self-update, packages are what
        # `update` can actually do.
        options.update_target = "extensions"


@dataclass(slots=True)
class CommandSettings:
    settings_manager: SettingsManager
    project_trust_warnings: list[str] = field(default_factory=list)


async def create_command_settings_manager(
    *,
    cwd: str,
    agent_dir: str,
    project_trust_override: bool | None = None,
    use_saved_project_trust_only: bool = False,
    extension_factories: list[Any] | None = None,
) -> CommandSettings:
    settings_manager = SettingsManager.create(cwd, agent_dir, project_trusted=False)
    warnings: list[str] = []
    trust_store = ProjectTrustStore(agent_dir)

    if use_saved_project_trust_only:
        saved = trust_store.get(cwd) is True
        settings_manager.set_project_trusted(project_trust_override if project_trust_override is not None else saved)
        return CommandSettings(settings_manager=settings_manager, project_trust_warnings=warnings)

    extensions_result = None
    if project_trust_override is None and has_trust_requiring_project_resources(cwd):
        # lazy: core <-> modes import cycle (see modes/__init__.py)
        from ..core.resource_loader import DefaultResourceLoader

        loader = DefaultResourceLoader(
            cwd=cwd,
            agent_dir=agent_dir,
            settings_manager=settings_manager,
            extension_factories=extension_factories,
        )
        extensions_result = await loader.load_project_trust_extensions()
        for error in extensions_result.errors or []:
            warnings.append(f'Failed to load extension "{error.path}": {error.error}')

    from .project_trust import CreateProjectTrustContextOptions, create_project_trust_context

    project_trusted = await resolve_project_trusted(
        ResolveProjectTrustedOptions(
            cwd=cwd,
            trust_store=trust_store,
            trust_override=project_trust_override,
            default_project_trust=settings_manager.get_default_project_trust(),
            extensions_result=extensions_result,
            project_trust_context=create_project_trust_context(
                CreateProjectTrustContextOptions(
                    cwd=cwd,
                    mode="print",
                    settings_manager=settings_manager,
                    has_ui=False,
                )
            ),
            on_extension_error=warnings.append,
        )
    )
    settings_manager.set_project_trusted(project_trusted)
    return CommandSettings(settings_manager=settings_manager, project_trust_warnings=warnings)


def _report_settings_errors(settings_manager: SettingsManager, context: str) -> None:
    for entry in settings_manager.drain_errors():
        print(red(f"Warning: ({context}, {entry.scope} settings) {entry.error}"), file=sys.stderr)


def _report_project_trust_warnings(warnings: list[str]) -> None:
    for warning in warnings:
        print(red(f"Warning: {warning}"), file=sys.stderr)


async def _refresh_model_catalogs(agent_dir: str) -> None:
    from ..core.model_runtime import ModelRuntime, ModelsRefreshOptions

    runtime = await ModelRuntime.create(
        auth_path=get_auth_path(),
        models_path=get_models_path(),
        allow_model_network=False,
    )
    result = await runtime.refresh(ModelsRefreshOptions(allow_network=True, force=True))
    if getattr(result, "aborted", False):
        raise Exception("Model catalog refresh timed out.")
    errors = getattr(result, "errors", None) or {}
    if errors:
        details = "; ".join(f"{provider}: {error}" for provider, error in errors.items())
        raise Exception(f"Could not refresh model catalogs: {details}")
    print(green("Model catalogs refreshed"))


def _print_usage_error(message: str, command: str) -> None:
    print(red(message), file=sys.stderr)
    print(dim(f"Usage: {get_package_command_usage(command)}"), file=sys.stderr)


def _validate(options: PackageCommandOptions) -> bool:
    """Report the first problem, in pi's precedence order. True if handled."""
    if options.invalid_option:
        print(red(f'Unknown option {options.invalid_option} for "{options.command}".'), file=sys.stderr)
        print(
            dim(f'Use "{APP_NAME} --help" or "{get_package_command_usage(options.command)}".'),
            file=sys.stderr,
        )
        return True
    if options.missing_option_value:
        _print_usage_error(f"Missing value for {options.missing_option_value}.", options.command)
        return True
    if options.invalid_argument:
        _print_usage_error(f"Unexpected argument {options.invalid_argument}.", options.command)
        return True
    if options.conflicting_options:
        _print_usage_error(options.conflicting_options, options.command)
        return True
    if options.command in ("install", "remove") and not options.source:
        _print_usage_error(f"Missing {options.command} source.", options.command)
        return True
    return False


def _print_package_list(package_manager: DefaultPackageManager) -> None:
    configured = package_manager.list_configured_packages()
    if not configured:
        print(dim("No packages installed."))
        return

    def emit(package) -> None:
        print(f"  {package.source}")
        if package.installed_path:
            print(dim(f"    {package.installed_path}"))

    user_packages = [package for package in configured if package.scope == "user"]
    project_packages = [package for package in configured if package.scope == "project"]
    if user_packages:
        print(bold("User packages:"))
        for package in user_packages:
            emit(package)
    if project_packages:
        if user_packages:
            print()
        print(bold("Project packages:"))
        for package in project_packages:
            emit(package)


async def handle_package_command(args: list[str], *, extension_factories: list[Any] | None = None) -> bool | int:
    """False when `args` is not a package command; otherwise the exit code."""
    options = parse_package_command(args)
    if options is None:
        return False

    if options.help:
        print_package_command_help(options.command)
        return 0

    if _validate(options):
        return 1

    if options.self_update_requested:
        print(red(f"{APP_NAME} does not support self-update."), file=sys.stderr)
        print(dim(SELF_UPDATE_HINT), file=sys.stderr)
        return 1

    if options.command == "update" and options.update_target == "models":
        try:
            await _refresh_model_catalogs(get_agent_dir())
        except Exception as error:
            print(red(f"Error: {error or 'Unknown model catalog refresh error'}"), file=sys.stderr)
            return 1
        return 0

    cwd = os.getcwd()
    agent_dir = get_agent_dir()
    writes_project_config = options.command in ("install", "remove") and options.local
    command_settings = await create_command_settings_manager(
        cwd=cwd,
        agent_dir=agent_dir,
        project_trust_override=options.project_trust_override,
        use_saved_project_trust_only=options.command == "update",
        extension_factories=extension_factories,
    )
    settings_manager = command_settings.settings_manager
    _report_project_trust_warnings(command_settings.project_trust_warnings)
    if writes_project_config and not settings_manager.is_project_trusted():
        print(red("Project is not trusted. Use --approve to modify local package config."), file=sys.stderr)
        return 1
    _report_settings_errors(settings_manager, "package command")

    package_manager = DefaultPackageManager(cwd=cwd, agent_dir=agent_dir, settings_manager=settings_manager)
    package_manager.set_progress_callback(
        lambda event: print(dim(event.message)) if event.type == "start" and event.message else None
    )

    try:
        if options.command == "install":
            await package_manager.install_and_persist(options.source, local=options.local)
            print(green(f"Installed {options.source}"))
            return 0

        if options.command == "remove":
            if not await package_manager.remove_and_persist(options.source, local=options.local):
                print(red(f"No matching package found for {options.source}"), file=sys.stderr)
                return 1
            print(green(f"Removed {options.source}"))
            return 0

        if options.command == "list":
            _print_package_list(package_manager)
            return 0

        # update
        await package_manager.update(options.update_source)
        print(green(f"Updated {options.update_source}" if options.update_source else "Updated packages"))
        if options.update_target == "all":
            await _refresh_model_catalogs(agent_dir)
        return 0
    except Exception as error:
        print(red(f"Error: {error or 'Unknown package command error'}"), file=sys.stderr)
        return 1


async def handle_config_command(args: list[str], *, extension_factories: list[Any] | None = None) -> bool | int:
    """False when `args` is not the config command; otherwise the exit code."""
    if not args or args[0] != "config":
        return False
    rest = args[1:]

    if "-h" in rest or "--help" in rest:
        print_config_command_help()
        return 0

    local = False
    project_trust_override: bool | None = None
    for arg in rest:
        if arg in ("-l", "--local"):
            local = True
        elif arg in ("-a", "--approve"):
            project_trust_override = True
        elif arg in ("-na", "--no-approve"):
            project_trust_override = False
        elif arg.startswith("-"):
            print(red(f'Unknown option {arg} for "config".'), file=sys.stderr)
            print(dim(f'Use "{APP_NAME} --help" or "{CONFIG_COMMAND_USAGE}".'), file=sys.stderr)
            return 1
        else:
            print(red(f"Unexpected argument {arg}."), file=sys.stderr)
            print(dim(f"Usage: {CONFIG_COMMAND_USAGE}"), file=sys.stderr)
            return 1

    cwd = os.getcwd()
    agent_dir = get_agent_dir()
    command_settings = await create_command_settings_manager(
        cwd=cwd,
        agent_dir=agent_dir,
        project_trust_override=project_trust_override,
        extension_factories=extension_factories,
    )
    settings_manager = command_settings.settings_manager
    _report_project_trust_warnings(command_settings.project_trust_warnings)
    if local and not settings_manager.is_project_trusted():
        print(red("Project is not trusted. Use --approve to modify local resource config."), file=sys.stderr)
        return 1
    _report_settings_errors(settings_manager, "config command")

    global_settings_manager = SettingsManager.create(cwd, agent_dir, project_trusted=False)
    global_paths = await DefaultPackageManager(
        cwd=cwd, agent_dir=agent_dir, settings_manager=global_settings_manager
    ).resolve()
    project_paths = (
        await DefaultPackageManager(cwd=cwd, agent_dir=agent_dir, settings_manager=settings_manager).resolve()
        if settings_manager.is_project_trusted()
        else global_paths
    )

    # lazy: core <-> modes import cycle (see modes/__init__.py)
    from .config_selector import select_config

    await select_config(
        resolved_paths={"global": global_paths, "project": project_paths},
        settings_manager=settings_manager,
        cwd=cwd,
        agent_dir=agent_dir,
        write_scope="project" if local else "global",
        project_mode_available=settings_manager.is_project_trusted(),
    )
    return 0
