"""Mirror of pi coding-agent src/core/extensions/loader.ts.

**The extension ABI is the one deliberate redesign in the port**, because pi's
is TypeScript-shaped end to end:

- *An extension is a module, not a default export.* pi writes
  `export default function (pi) { ... }`; Python has no default export, so a
  pidrei extension is a `.py` module defining a module-level ``extension(pi)``
  callable (sync or async, exactly like pi's factory).
- *A package is a package.* pi's directory form is ``index.ts``/``index.js``
  plus a ``package.json`` carrying a ``pi.extensions`` manifest. Here it is
  ``__init__.py`` plus a ``pyproject.toml`` carrying ``[tool.pidrei]
  extensions = [...]`` — the same two rules against Python's own artifacts.
- *There is no module-alias table.* pi needs ~125 lines of `VIRTUAL_MODULES` /
  `getAliases()` so that an extension's `import "@earendil-works/pi-ai"`
  resolves both inside a Bun-compiled binary and in a dev checkout. A pidrei
  extension runs in the same interpreter with the same ``sys.path``, so it just
  writes ``import pidrei_ai`` and gets the very objects the agent is using.
  That whole block is deliberately not ported. Upstream fixes to it — such as
  pi's `c0613289`, which extends the virtual-module branch to Node SEA hosts —
  have no analogue here for the same reason.

Everything else is pi's: the cwd+generation-keyed factory cache, the
registration-writes-to-the-extension-object split, discovery order, and the
one-level-deep no-recursion rule.
"""

import importlib.machinery
import importlib.util
import itertools
import os
import re
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio

from pidrei.config import CONFIG_DIR_NAME, get_agent_dir
from pidrei.core.event_bus import EventBus
from pidrei.core.exec import exec_command
from pidrei.core.pidrei_manifest import read_pidrei_manifest
from pidrei.core.source_info import create_synthetic_source_info
from pidrei.core.timings import time as record_time
from pidrei.utils.paths import resolve_path

from .types import (
    RUNTIME_NOT_INITIALIZED,
    Extension,
    ExtensionFlag,
    ExtensionLoadError,
    ExtensionRuntime,
    ExtensionShortcut,
    LoadExtensionsResult,
    RegisteredCommand,
    RegisteredTool,
    ToolDefinition,
)


#: Module attribute holding the extension factory (pi: the default export).
FACTORY_ATTRIBUTE = "extension"


# -- factory cache ---------------------------------------------------------------

_extension_cache: dict[str, Any] = {}
_extension_cache_cwd: str | None = None
_extension_cache_generation = 0

# Guards module execution in `_load_extension_module`. Held only on a blocking
# pool thread, never across an await.
_import_lock = threading.RLock()

_module_counter = itertools.count()


@dataclass(slots=True, frozen=True)
class ExtensionCacheToken:
    cwd: str
    generation: int


def clear_extension_cache() -> None:
    global _extension_cache_cwd, _extension_cache_generation
    _extension_cache.clear()
    _extension_cache_cwd = None
    _extension_cache_generation += 1


def _use_extension_cache_cwd(cwd: str) -> ExtensionCacheToken:
    global _extension_cache_cwd
    resolved_cwd = resolve_path(cwd)
    if _extension_cache_cwd is not None and _extension_cache_cwd != resolved_cwd:
        clear_extension_cache()
    _extension_cache_cwd = resolved_cwd
    return ExtensionCacheToken(cwd=resolved_cwd, generation=_extension_cache_generation)


def _is_current_cache_token(token: ExtensionCacheToken | None) -> bool:
    return token is not None and _extension_cache_cwd == token.cwd and _extension_cache_generation == token.generation


# -- runtime ---------------------------------------------------------------------


def create_extension_runtime() -> ExtensionRuntime:
    """pi's createExtensionRuntime(): action methods stay unbound until the
    runner's bind_core() fills them in."""
    return ExtensionRuntime()


# -- the API handed to extension factories ---------------------------------------


class _ExtensionEventBus:
    """`pi.events` for one extension instance: staleness-checked and tracked."""

    __slots__ = ("_event_bus", "_runtime")

    def __init__(self, runtime: ExtensionRuntime, event_bus: EventBus):
        self._runtime = runtime
        self._event_bus = event_bus

    def emit(self, channel: str, data: Any) -> None:
        self._runtime.assert_active()
        self._event_bus.emit(channel, data)

    def on(self, channel: str, handler: Any) -> Callable[[], None]:
        self._runtime.assert_active()
        return self._runtime.track_event_bus_subscription(self._event_bus.on(channel, handler))


def _flag_value_type(value: Any) -> str:
    """JS `typeof` for a flag default, so a mismatch names the type the way pi does."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int | float):
        return "number"
    return type(value).__name__


class ExtensionAPI:
    """pi's ExtensionAPI: the `pi` object an extension factory receives.

    Registration methods write to the Extension record; action methods
    delegate to the shared runtime, which is unbound during loading.
    """

    __slots__ = ("_cwd", "_extension", "_runtime", "events")

    def __init__(self, extension: Extension, runtime: ExtensionRuntime, cwd: str, event_bus: EventBus):
        self._extension = extension
        self._runtime = runtime
        self._cwd = cwd
        #: Shared event bus for extension-to-extension communication. Wrapped so
        #: a stale ctx cannot use it and so subscriptions die with the runtime
        #: (pi #7193: a reload's old handlers kept receiving every event).
        self.events = _ExtensionEventBus(runtime, event_bus)

    def _action(self, name: str) -> Callable[..., Any]:
        action = getattr(self._runtime, name, None)
        if action is None:
            raise RuntimeError(RUNTIME_NOT_INITIALIZED)
        return action

    # -- registration ------------------------------------------------------------

    def on(self, event: str, handler: Any) -> None:
        self._runtime.assert_active()
        self._extension.handlers.setdefault(event, []).append(handler)

    def register_tool(self, tool: ToolDefinition) -> None:
        self._runtime.assert_active()
        self._extension.tools[tool.name] = RegisteredTool(definition=tool, source_info=self._extension.source_info)
        self._runtime.refresh_tools()

    def register_command(
        self,
        name: str,
        *,
        handler: Any,
        description: str | None = None,
        get_argument_completions: Any = None,
    ) -> None:
        self._runtime.assert_active()
        self._extension.commands[name] = RegisteredCommand(
            name=name,
            handler=handler,
            description=description,
            source_info=self._extension.source_info,
            get_argument_completions=get_argument_completions,
        )

    def register_shortcut(self, shortcut: str, *, handler: Any, description: str | None = None) -> None:
        self._runtime.assert_active()
        self._extension.shortcuts[shortcut] = ExtensionShortcut(
            shortcut=shortcut,
            handler=handler,
            description=description,
            extension_path=self._extension.path,
        )

    def register_flag(self, name: str, *, type: str, description: str | None = None, default: Any = None) -> None:
        self._runtime.assert_active()
        if default is not None and _flag_value_type(default) != type:
            raise ValueError(f'Invalid default for flag "{name}": expected {type}, got {_flag_value_type(default)}')
        self._extension.flags[name] = ExtensionFlag(
            type=type,
            description=description,
            name=name,
            extension_path=self._extension.path,
            default=default,
        )
        if default is not None and name not in self._runtime.flag_values:
            self._runtime.flag_values[name] = default

    def register_message_renderer(self, custom_type: str, renderer: Any) -> None:
        self._runtime.assert_active()
        self._extension.message_renderers[custom_type] = renderer

    def register_markdown_transformer(self, transformer: Any) -> None:
        self._runtime.assert_active()
        self._extension.markdown_transformer = transformer

    def register_entry_renderer(self, custom_type: str, renderer: Any) -> None:
        self._runtime.assert_active()
        self._extension.entry_renderers[custom_type] = renderer

    def get_flag(self, name: str) -> Any:
        self._runtime.assert_active()
        if name not in self._extension.flags:
            return None
        return self._runtime.flag_values.get(name)

    # -- actions -----------------------------------------------------------------

    def send_message(self, message: Any, options: Any = None) -> None:
        self._runtime.assert_active()
        self._action("send_message")(message, options)

    def send_user_message(self, content: Any, options: Any = None) -> None:
        self._runtime.assert_active()
        self._action("send_user_message")(content, options)

    async def append_entry(self, custom_type: str, data: Any = None) -> None:
        self._runtime.assert_active()
        await self._action("append_entry")(custom_type, data)

    async def set_session_name(self, name: str) -> None:
        self._runtime.assert_active()
        await self._action("set_session_name")(name)

    def get_session_name(self) -> Any:
        self._runtime.assert_active()
        return self._action("get_session_name")()

    async def set_label(self, entry_id: str, label: str | None) -> None:
        self._runtime.assert_active()
        await self._action("set_label")(entry_id, label)

    def exec(self, command: str, args: list[str], *, cwd: str | None = None, **options: Any) -> Any:
        """Returns the coroutine, as pi returns the promise: no await here, so
        the command starts only when the extension awaits it."""
        self._runtime.assert_active()
        return exec_command(command, args, cwd or self._cwd, **options)

    def get_active_tools(self) -> list[str]:
        self._runtime.assert_active()
        return self._action("get_active_tools")()

    def get_all_tools(self) -> list[Any]:
        self._runtime.assert_active()
        return self._action("get_all_tools")()

    def set_active_tools(self, tool_names: list[str]) -> None:
        self._runtime.assert_active()
        self._action("set_active_tools")(tool_names)

    def get_commands(self) -> list[Any]:
        self._runtime.assert_active()
        return self._action("get_commands")()

    def set_model(self, model: Any) -> Any:
        self._runtime.assert_active()
        return self._action("set_model")(model)

    def get_thinking_level(self) -> Any:
        self._runtime.assert_active()
        return self._action("get_thinking_level")()

    async def set_thinking_level(self, level: Any) -> None:
        self._runtime.assert_active()
        await self._action("set_thinking_level")(level)

    # -- providers ---------------------------------------------------------------

    def register_provider(self, provider_or_name: Any, config: Any = None) -> None:
        self._runtime.assert_active()
        if isinstance(provider_or_name, str):
            if config is None:
                raise ValueError("Provider config is required when registering by name")
            self._runtime.register_provider(provider_or_name, config, self._extension.path)
            return
        self._runtime.register_native_provider(provider_or_name, self._extension.path)

    def unregister_provider(self, name: str) -> None:
        self._runtime.assert_active()
        self._runtime.unregister_provider(name, self._extension.path)


# -- module loading --------------------------------------------------------------


def _identifier(name: str) -> str:
    return re.sub(r"\W", "_", os.path.splitext(name)[0])


def _import_module(resolved_path: str) -> Any:
    """Execute an extension module under a fresh synthetic name.

    pi passes `moduleCache: false` to jiti so every load re-evaluates; the
    unique name is how we opt out of the equivalent cache, sys.modules.
    """
    directory = os.path.dirname(resolved_path)
    basename = os.path.basename(resolved_path)
    label = os.path.basename(directory) if basename == "__init__.py" else basename
    root_name = f"_pidrei_extension_{next(_module_counter)}_{_identifier(label)}"

    if basename == "__init__.py":
        module_name = root_name
        spec = importlib.util.spec_from_file_location(
            module_name, resolved_path, submodule_search_locations=[directory]
        )
    else:
        # A standalone extension file gets a synthetic parent package rooted at
        # its own directory, so `from . import helper` resolves the way pi's
        # `import "./helper.ts"` does.
        package = importlib.util.module_from_spec(importlib.machinery.ModuleSpec(root_name, None, is_package=True))
        package.__path__ = [directory]
        sys.modules[root_name] = package
        module_name = f"{root_name}.{_identifier(basename)}"
        spec = importlib.util.spec_from_file_location(module_name, resolved_path)

    if spec is None or spec.loader is None:
        sys.modules.pop(root_name, None)
        raise ImportError(f"Cannot import extension module: {resolved_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        sys.modules.pop(root_name, None)
        raise
    return module


def _load_extension_module(resolved_path: str, cache_token: ExtensionCacheToken | None = None) -> Any:
    """Import an extension module and return its factory.

    Blocking: reads the source and lets CPython write `__pycache__`, so callers
    hand this to the blocking pool (see `_load_extension`).

    Serialised deliberately. Extension modules are arbitrary user code executed
    at import, and two of them may pull in the same dependency at once. Rather
    than reason about how the free-threaded interpreter locks imports — which is
    an implementation detail that can change between Python releases — only one
    module is executed at a time. Extension loading is a startup/reload step, so
    the lost concurrency costs nothing. Re-entrant because an extension could,
    in principle, trigger another extension import on the same thread.
    """
    with _import_lock:
        if _is_current_cache_token(cache_token):
            cached = _extension_cache.get(resolved_path)
            if cached is not None:
                return cached

        module = _import_module(resolved_path)
        factory = getattr(module, FACTORY_ATTRIBUTE, None)
        if not callable(factory):
            return None
        if _is_current_cache_token(cache_token):
            _extension_cache[resolved_path] = factory
        return factory


def _create_extension(extension_path: str, resolved_path: str) -> Extension:
    if extension_path.startswith("<") and extension_path.endswith(">"):
        source = extension_path[1:-1].split(":")[0] or "temporary"
    else:
        source = "local"
    base_dir = None if extension_path.startswith("<") else os.path.dirname(resolved_path)

    return Extension(
        path=extension_path,
        resolved_path=resolved_path,
        source_info=create_synthetic_source_info(extension_path, source=source, base_dir=base_dir),
    )


async def _call_factory(factory: Any, api: ExtensionAPI) -> None:
    result = factory(api)
    if hasattr(result, "__await__"):
        await result


async def _load_extension(
    extension_path: str,
    cwd: str,
    event_bus: EventBus,
    runtime: ExtensionRuntime,
    cache_token: ExtensionCacheToken | None = None,
) -> tuple[Extension | None, str | None]:
    resolved_path = resolve_path(extension_path, cwd, normalize_unicode_spaces=True)

    try:
        factory = await tonio.spawn_blocking(_load_extension_module, resolved_path, cache_token)
        record_time(f"{extension_path} module import", "extensions")
        if factory is None:
            return None, (f"Extension does not define a valid {FACTORY_ATTRIBUTE}() factory function: {extension_path}")

        extension = _create_extension(extension_path, resolved_path)
        api = ExtensionAPI(extension, runtime, cwd, event_bus)
        await _call_factory(factory, api)
        record_time(f"{extension_path} factory", "extensions")

        return extension, None
    except Exception as error:
        return None, f"Failed to load extension: {error}"


async def load_extension_from_factory(
    factory: Any,
    cwd: str,
    event_bus: EventBus,
    runtime: ExtensionRuntime,
    extension_path: str = "<inline>",
) -> Extension:
    """Build an Extension from an in-process factory (pi's inline extensions)."""
    extension = _create_extension(extension_path, extension_path)
    api = ExtensionAPI(extension, runtime, resolve_path(cwd), event_bus)
    await _call_factory(factory, api)
    record_time(f"{extension_path} factory", "extensions")
    return extension


async def _load_extensions_internal(
    paths: list[str],
    cwd: str,
    event_bus: EventBus | None = None,
    runtime: ExtensionRuntime | None = None,
    use_cache: bool = False,
) -> LoadExtensionsResult:
    extensions: list[Extension] = []
    errors: list[ExtensionLoadError] = []
    cache_token = _use_extension_cache_cwd(cwd) if use_cache else None
    resolved_cwd = cache_token.cwd if cache_token is not None else resolve_path(cwd)
    resolved_event_bus = event_bus if event_bus is not None else EventBus()
    resolved_runtime = runtime if runtime is not None else create_extension_runtime()

    for extension_path in paths:
        extension, error = await _load_extension(
            extension_path, resolved_cwd, resolved_event_bus, resolved_runtime, cache_token
        )
        if error is not None:
            errors.append(ExtensionLoadError(path=extension_path, error=error))
            continue
        if extension is not None:
            extensions.append(extension)

    return LoadExtensionsResult(extensions=extensions, errors=errors, runtime=resolved_runtime)


async def load_extensions(
    paths: list[str],
    cwd: str,
    event_bus: EventBus | None = None,
    runtime: ExtensionRuntime | None = None,
) -> LoadExtensionsResult:
    return await _load_extensions_internal(paths, cwd, event_bus, runtime)


async def load_extensions_cached(
    paths: list[str],
    cwd: str,
    event_bus: EventBus | None = None,
    runtime: ExtensionRuntime | None = None,
) -> LoadExtensionsResult:
    return await _load_extensions_internal(paths, cwd, event_bus, runtime, use_cache=True)


# -- discovery -------------------------------------------------------------------


def is_extension_file(name: str) -> bool:
    # Leading underscores mark a module as private in Python, which pi's
    # `.ts`/`.js` rule has no way to express; skipping them keeps helper
    # modules sitting next to an extension from being loaded as extensions.
    return name.endswith(".py") and not name.startswith("_")


def resolve_extension_entries(directory: str) -> list[str] | None:
    """Entry points declared by a directory, or None when it declares none.

    1. pyproject.toml with a `[tool.pidrei] extensions` list -> those paths
    2. __init__.py -> the package itself
    """
    pyproject_path = os.path.join(directory, "pyproject.toml")
    if os.path.exists(pyproject_path):
        manifest = read_pidrei_manifest(pyproject_path)
        declared = (manifest or {}).get("extensions")
        if declared:
            entries: list[str] = []
            for entry in declared:
                # os.path.abspath, not resolve_path: a declared "~entry.py"
                # stays package-relative rather than expanding to $HOME.
                resolved_entry = os.path.abspath(os.path.join(directory, entry))
                if os.path.exists(resolved_entry):
                    entries.append(resolved_entry)
            if entries:
                return entries

    package_init = os.path.join(directory, "__init__.py")
    if os.path.exists(package_init):
        return [package_init]

    return None


def discover_extensions_in_dir(directory: str) -> list[str]:
    """Discover extensions one level deep (pi's discoverExtensionsInDir).

    1. Direct files: `extensions/*.py`
    2. Subdirectory package: `extensions/*/__init__.py`
    3. Subdirectory manifest: `extensions/*/pyproject.toml` with a
       `[tool.pidrei] extensions` list

    No recursion beyond one level; deeper layouts must use the manifest.
    """
    if not os.path.exists(directory):
        return []

    discovered: list[str] = []
    try:
        for entry in sorted(os.scandir(directory), key=lambda item: item.name):
            entry_path = os.path.join(directory, entry.name)

            if entry.is_file() and is_extension_file(entry.name):
                discovered.append(entry_path)
                continue

            if entry.is_dir():
                entries = resolve_extension_entries(entry_path)
                if entries:
                    discovered.extend(entries)
    except OSError:
        return []

    return discovered


async def discover_and_load_extensions(
    configured_paths: list[str],
    cwd: str,
    agent_dir: str | None = None,
    event_bus: EventBus | None = None,
) -> LoadExtensionsResult:
    """Discover extensions in the standard locations, then load them."""
    resolved_cwd = resolve_path(cwd)
    resolved_agent_dir = resolve_path(agent_dir if agent_dir is not None else get_agent_dir())
    all_paths: list[str] = []
    seen: set[str] = set()

    def add_paths(paths: list[str]) -> None:
        for path in paths:
            resolved = os.path.abspath(path)
            if resolved not in seen:
                seen.add(resolved)
                all_paths.append(path)

    # Discovery is a directory scan plus a manifest read per candidate — one
    # blocking unit, so it goes to the pool whole rather than a hop per probe.
    def _discover() -> list[list[str]]:
        found: list[list[str]] = [
            # 1. Project-local extensions: cwd/<CONFIG_DIR_NAME>/extensions/
            discover_extensions_in_dir(os.path.join(resolved_cwd, CONFIG_DIR_NAME, "extensions")),
            # 2. Global extensions: agent_dir/extensions/
            discover_extensions_in_dir(os.path.join(resolved_agent_dir, "extensions")),
        ]
        # 3. Explicitly configured paths
        for path in configured_paths:
            resolved = resolve_path(path, resolved_cwd, normalize_unicode_spaces=True)
            if os.path.isdir(resolved):
                entries = resolve_extension_entries(resolved)
                found.append(entries if entries else discover_extensions_in_dir(resolved))
                continue
            found.append([resolved])
        return found

    for group in await tonio.spawn_blocking(_discover):
        add_paths(group)

    return await load_extensions(all_paths, resolved_cwd, event_bus)
