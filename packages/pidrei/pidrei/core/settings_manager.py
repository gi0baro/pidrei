"""Mirror of pi coding-agent src/core/settings-manager.ts.

Settings are represented as plain dicts with the on-disk camelCase keys
(pi's Settings interface is an open JS object: unknown keys must survive
load/merge/save round-trips, which a typed model would drop).

pi defers writes on a promise chain (`writeQueue`) drained by `flush()`;
here that chain is one writer task over an unbounded channel (see
`_enqueue_write`): sets stay ordered, write errors are recorded (not
raised) and surfaced via drain_errors(), and flush() awaits a ticket.

Loads (`reload`, `set_project_trusted`) open with a flush, as pi's
`await this.writeQueue`, but then read with awaits where pi reads
synchronously — so a setter can publish meanwhile. A load publishes only a
read no setter raced; otherwise it flushes again (the setter's write
included) and rereads.

Epoch discipline (PROPER_MT_DESIGN.md step 3): the published scope state
(`_settings`, `_global_settings`, `_project_settings`) is immutable —
setters deep-copy the scope, run pi's mutation lines on the private copy
(`_update_global_settings`), and publish by rebinding under `_write_lock`.
Readers pin one attribute read and never take a lock; a snapshot they hold
can never change under them. Values stored into a snapshot are held by
reference — callers do not mutate what they passed in (the same convention
step 2 keeps for user-owned dicts inside messages).
"""

import copy
import json
import math
import os
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

import tonio.colored as tonio
from tonio.colored import fs, sync
from tonio.colored.sync import channel

from pidrei_ai.utils.retry import DEFAULT_MAX_AGENT_RETRY_DELAY_MS

from ..config import CONFIG_DIR_NAME, get_agent_dir
from ..utils.lockfile import FileLock
from ..utils.paths import normalize_path, resolve_path
from ..utils.text import strip_bom
from .http_config import DEFAULT_HTTP_IDLE_TIMEOUT_MS, parse_http_idle_timeout_ms


if TYPE_CHECKING:
    from pidrei_tui import WheelScrollLines


type Settings = dict[str, Any]
type SettingsScope = Literal["global", "project"]

# Cache-warming profile. "idle" also warms between agent runs.
CACHE_WARMING_MODES = ("off", "streaming", "idle")
type CacheWarmingMode = Literal["off", "streaming", "idle"]


def _is_mergeable_object(value: Any) -> bool:
    return isinstance(value, dict)


_DEFAULT_COMPACTION_TOKEN_SETTINGS = {"reserveTokens": 16384, "keepRecentTokens": 20000}
_MAX_SAFE_INTEGER = 2**53 - 1
_MISSING = object()


def _js_string(value: Any) -> str:
    """JS `String(value)` for the JSON-shaped values a settings file can hold."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return str(int(value)) if value.is_integer() else repr(value)
    if isinstance(value, dict):
        return "[object Object]"
    if isinstance(value, list):
        return ",".join("" if item is None else _js_string(item) for item in value)
    return str(value)


def _is_non_negative_safe_integer(value: Any) -> bool:
    """pi's `typeof value === "number" && Number.isSafeInteger(value) && value >= 0`."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    if isinstance(value, float) and not value.is_integer():  # also rejects NaN and ±Infinity
        return False
    return 0 <= value <= _MAX_SAFE_INTEGER


def _compaction_token_setting(compaction: Any, field: str, model: Any) -> int:
    """Resolve one compaction token field: exact "provider/modelId" override,
    ordinary setting, then built-in default. A key that is present but invalid
    (including JSON null, which pi's `!== undefined` checks keep) raises."""
    compaction = compaction or {}
    ordinary = compaction.get(field, _MISSING)
    if ordinary is not _MISSING and not _is_non_negative_safe_integer(ordinary):
        raise Exception(
            f"Invalid compaction.{field} setting: {_js_string(ordinary)}. Expected a non-negative safe integer."
        )

    model_key = f"{model.provider}/{model.id}" if model is not None else None
    model_overrides = compaction.get("modelOverrides")
    entry = (
        model_overrides.get(model_key, _MISSING)
        if model_key is not None and _is_mergeable_object(model_overrides)
        else _MISSING
    )
    if entry is not _MISSING and not _is_mergeable_object(entry):
        raise Exception(
            f'Invalid compaction.modelOverrides["{model_key}"] setting: {_js_string(entry)}. Expected an object.'
        )
    override = entry.get(field, _MISSING) if entry is not _MISSING else _MISSING
    if override is not _MISSING and not _is_non_negative_safe_integer(override):
        raise Exception(
            f'Invalid compaction.modelOverrides["{model_key}"].{field} setting: {_js_string(override)}. '
            "Expected a non-negative safe integer."
        )
    if override is not _MISSING:
        return override
    if ordinary is not _MISSING:
        return ordinary
    return _DEFAULT_COMPACTION_TOKEN_SETTINGS[field]


def _deep_merge_objects(base: dict, overrides: dict) -> dict:
    result = dict(base)

    # pi skips `undefined` overrides here; Python has no undefined and a parsed
    # JSON `null` is a real value, so (as before this port's rewrite) every key
    # present in `overrides` wins.
    for key, override_value in overrides.items():
        base_value = base.get(key)
        result[key] = (
            _deep_merge_objects(base_value, override_value)
            if _is_mergeable_object(base_value) and _is_mergeable_object(override_value)
            else override_value
        )

    return result


# Tools enabled at startup when `defaultTools` does not change them.
DEFAULT_TOOL_NAMES: tuple[str, ...] = ("read", "bash", "edit", "write")


def _is_tool_modifier(entry: Any) -> bool:
    return isinstance(entry, str) and entry.startswith(("+", "-"))


def _merge_default_tools(base: Any, overrides: Any) -> Any:
    """Merge `defaultTools` of two settings layers. A list with plain tool names replaces the
    inherited one; a list of only `+name`/`-name` entries is appended, so it modifies the
    inherited selection."""
    # Settings files are not validated; a malformed value replaces instead of raising here.
    if not isinstance(base, list) or not isinstance(overrides, list) or not all(map(_is_tool_modifier, overrides)):
        return overrides
    return [*base, *overrides]


def _resolve_default_tools(entries: list[str]) -> list[str]:
    """Resolve a merged `defaultTools` list: plain names replace `DEFAULT_TOOL_NAMES`, then `+name`
    adds and `-name` removes a tool, in list order."""
    plain = [entry for entry in entries if not _is_tool_modifier(entry)]
    tools = plain if plain or not entries else list(DEFAULT_TOOL_NAMES)
    for entry in entries:
        if not _is_tool_modifier(entry):
            continue
        name = entry[1:]
        if entry.startswith("+") and name and name not in tools:
            tools.append(name)
        elif entry.startswith("-") and name in tools:
            tools.remove(name)
    return tools


def deep_merge_settings(base: Settings, overrides: Settings) -> Settings:
    """Deep merge settings: project/overrides take precedence, nested objects merge recursively."""
    merged = _deep_merge_objects(base, overrides)
    if "defaultTools" in overrides:
        merged["defaultTools"] = _merge_default_tools(base.get("defaultTools"), overrides["defaultTools"])
    return merged


def _parse_timeout_setting(value: Any, setting_name: str) -> int | None:
    timeout_ms = parse_http_idle_timeout_ms(value)
    if timeout_ms is not None:
        return timeout_ms
    if value is not None:
        raise Exception(f"Invalid {setting_name} setting: {value}")
    return None


@dataclass(slots=True)
class SettingsError:
    scope: SettingsScope
    error: Exception
    path: str | None = None


# Optional settings file path per scope, for reporting storage errors.
type SettingsPaths = dict[SettingsScope, str]


def _to_settings_error(scope: SettingsScope, error: Exception, path: str | None = None) -> SettingsError:
    return SettingsError(scope, error, path or None)


class SettingsStorage(Protocol):
    async def with_lock_async(self, scope: SettingsScope, fn: Any) -> None: ...


class FileSettingsStorage:
    def __init__(self, cwd: str, agent_dir: str):
        resolved_cwd = resolve_path(cwd)
        resolved_agent_dir = resolve_path(agent_dir)
        self._global_settings_path = os.path.join(resolved_agent_dir, "settings.json")
        self._project_settings_path = os.path.join(resolved_cwd, CONFIG_DIR_NAME, "settings.json")

    async def with_lock_async(self, scope: SettingsScope, fn: Any) -> None:
        path = fs.Path(self._global_settings_path if scope == "global" else self._project_settings_path)
        lock = FileLock(str(path))
        try:
            # Only create directory and lock if file exists or we need to write
            current = None
            if await path.exists():
                await lock.acquire()
                current = await path.read_text(encoding="utf-8")
            next_content = fn(current)
            if next_content is not None:
                # Only create directory when we actually need to write
                await path.parent.mkdir(parents=True, exist_ok=True)
                if not lock.held:
                    await lock.acquire()
                await path.write_text(next_content, encoding="utf-8")
        finally:
            await lock.release()


class InMemorySettingsStorage:
    """No I/O, so besides the storage interface it offers the sync `with_lock`
    that in-memory managers write through inline."""

    def __init__(self):
        self._global: str | None = None
        self._project: str | None = None
        self._lock = threading.Lock()

    def with_lock(self, scope: SettingsScope, fn: Any) -> None:
        with self._lock:
            current = self._global if scope == "global" else self._project
            next_content = fn(current)
            if next_content is not None:
                if scope == "global":
                    self._global = next_content
                else:
                    self._project = next_content

    async def with_lock_async(self, scope: SettingsScope, fn: Any) -> None:
        self.with_lock(scope, fn)


class SettingsManager:
    """`await SettingsManager(cwd, agent_dir)` loads the global and project
    `settings.json`; `await SettingsManager(storage=...)` loads an arbitrary
    backend. `__init__` does no I/O: until awaited, the settings are empty."""

    def __init__(
        self,
        cwd: str | None = None,
        agent_dir: str | None = None,
        *,
        project_trusted: bool = True,
        storage: SettingsStorage | None = None,
    ):
        settings_paths: SettingsPaths = {}
        if storage is None:
            if cwd is None:
                raise ValueError("pass cwd for file-backed settings, or a storage backend")
            resolved_cwd = resolve_path(cwd)
            resolved_agent_dir = resolve_path(agent_dir if agent_dir is not None else get_agent_dir())
            storage = FileSettingsStorage(resolved_cwd, resolved_agent_dir)
            settings_paths = {
                "global": os.path.join(resolved_agent_dir, "settings.json"),
                "project": os.path.join(resolved_cwd, CONFIG_DIR_NAME, "settings.json"),
            }
        elif cwd is not None or agent_dir is not None:
            raise ValueError("pass either cwd/agent_dir or a storage backend, not both")
        self._storage = storage
        self._global_settings: Settings = {}
        self._project_settings: Settings = {}
        self._project_trusted = project_trusted
        self._modified_fields: set[str] = set()
        self._modified_nested_fields: dict[str, set[str]] = {}
        self._modified_project_fields: set[str] = set()
        self._modified_project_nested_fields: dict[str, set[str]] = {}
        self._global_settings_load_error: Exception | None = None
        self._project_settings_load_error: Exception | None = None
        # Guards every writer-side section: scope-snapshot rebinds, the
        # modified-field sets, the error list, and the writer channel. Readers
        # never take it — they pin the published snapshots instead. Only ever
        # held by synchronous code: never across an await or I/O.
        self._write_lock = threading.RLock()
        # pi's `writeQueue` promise chain is one writer task over an unbounded
        # channel: writes run in enqueue order, a failed one never blocks the
        # next, and `flush()` is a ticket that the writer sets when it reaches
        # it. Started lazily by the first write; `None` until then.
        self._writes: Any = None
        # Bumped (under `_write_lock`) by every setter publish; a load that
        # sees it move while it read discards the read (see the module
        # docstring).
        self._mutations = 0
        # One load at a time (`reload`, `set_project_trusted`).
        self._load_lock = sync.Lock()
        self._errors: list[SettingsError] = []
        # File paths for reported storage errors; empty for other backends.
        self._settings_paths: SettingsPaths = settings_paths
        self._settings = deep_merge_settings(self._global_settings, self._project_settings)

    def __await__(self):
        return self._start().__await__()

    async def _start(self) -> SettingsManager:
        global_settings, global_error = await SettingsManager._try_load_from_storage(self._storage, "global")
        project_settings, project_error = await SettingsManager._try_load_from_storage(
            self._storage, "project", self._project_trusted
        )
        with self._write_lock:
            self._publish_reload(global_settings, global_error, project_settings, project_error)
        return self

    @staticmethod
    def in_memory(settings: Settings | None = None, *, project_trusted: bool = True) -> SettingsManager:
        """Create an in-memory SettingsManager (no file I/O, so no await: the
        seeded content is parsed back the way a load would)."""
        storage = InMemorySettingsStorage()
        initial_settings = SettingsManager._migrate_settings(copy.deepcopy(settings or {}))
        content = json.dumps(initial_settings, indent=2)
        storage.with_lock("global", lambda _current: content)
        manager = SettingsManager(storage=storage, project_trusted=project_trusted)
        with manager._write_lock:
            manager._publish_reload(SettingsManager._parse_settings(content), None, {}, None)
        return manager

    @staticmethod
    async def _load_from_storage(
        storage: SettingsStorage, scope: SettingsScope, project_trusted: bool = True
    ) -> Settings:
        if scope == "project" and not project_trusted:
            return {}

        content: str | None = None

        def read(current: str | None) -> None:
            nonlocal content
            content = current

        await storage.with_lock_async(scope, read)
        return SettingsManager._parse_settings(content)

    @staticmethod
    def _parse_settings(content: str | None) -> Settings:
        if not content:
            return {}
        settings = json.loads(strip_bom(content))
        return SettingsManager._migrate_settings(settings)

    @staticmethod
    async def _try_load_from_storage(
        storage: SettingsStorage, scope: SettingsScope, project_trusted: bool = True
    ) -> tuple[Settings, Exception | None]:
        try:
            return await SettingsManager._load_from_storage(storage, scope, project_trusted), None
        except Exception as error:
            return {}, error

    @staticmethod
    def _migrate_settings(settings: Settings) -> Settings:
        """Migrate old settings format to new format."""
        # Migrate queueMode -> steeringMode
        if "queueMode" in settings and "steeringMode" not in settings:
            settings["steeringMode"] = settings["queueMode"]
            del settings["queueMode"]

        # Migrate enableInstallTelemetry -> enableProviderAttribution. pi's key
        # named an install ping that Phase 7 step 1 removed; the toggle now
        # gates provider attribution headers only.
        if "enableInstallTelemetry" in settings and "enableProviderAttribution" not in settings:
            settings["enableProviderAttribution"] = settings["enableInstallTelemetry"]
        settings.pop("enableInstallTelemetry", None)

        # Migrate legacy websockets boolean -> transport enum
        if "transport" not in settings and isinstance(settings.get("websockets"), bool):
            settings["transport"] = "websocket" if settings["websockets"] else "sse"
            del settings["websockets"]

        # Migrate old skills object format to new array format
        if "skills" in settings and isinstance(settings["skills"], dict):
            skills_settings = settings["skills"]
            if skills_settings.get("enableSkillCommands") is not None and settings.get("enableSkillCommands") is None:
                settings["enableSkillCommands"] = skills_settings["enableSkillCommands"]
            custom_directories = skills_settings.get("customDirectories")
            if isinstance(custom_directories, list) and len(custom_directories) > 0:
                settings["skills"] = custom_directories
            else:
                del settings["skills"]

        # Migrate retry.maxDelayMs -> retry.provider.maxRetryDelayMs
        if isinstance(settings.get("retry"), dict):
            retry_settings = settings["retry"]
            provider_settings = (
                retry_settings.get("provider") if isinstance(retry_settings.get("provider"), dict) else None
            )
            max_delay = retry_settings.get("maxDelayMs")
            if (
                isinstance(max_delay, (int, float))
                and not isinstance(max_delay, bool)
                and (provider_settings is None or provider_settings.get("maxRetryDelayMs") is None)
            ):
                retry_settings["provider"] = {**(provider_settings or {}), "maxRetryDelayMs": max_delay}
            retry_settings.pop("maxDelayMs", None)

        return settings

    # -- scope state ----------------------------------------------------------

    def get_global_settings(self) -> Settings:
        return copy.deepcopy(self._global_settings)

    def get_project_settings(self) -> Settings:
        return copy.deepcopy(self._project_settings)

    def is_project_trusted(self) -> bool:
        return self._project_trusted

    async def set_project_trusted(self, trusted: bool) -> None:
        async with self._load_lock:
            with self._write_lock:
                if self._project_trusted == trusted:
                    return

                self._project_trusted = trusted
                self._modified_project_fields.clear()
                self._modified_project_nested_fields.clear()

                if not trusted:
                    self._project_settings = {}
                    self._project_settings_load_error = None
                    self._settings = deep_merge_settings(self._global_settings, self._project_settings)
                    return

            while True:
                await self.flush()
                mutations = self._mutations
                project_settings, project_error = await SettingsManager._try_load_from_storage(
                    self._storage, "project", trusted
                )
                with self._write_lock:
                    if self._mutations != mutations:
                        continue  # a setter raced the read
                    self._project_settings = project_settings
                    self._project_settings_load_error = project_error
                    if project_error is not None:
                        self._record_error("project", project_error)
                    self._settings = deep_merge_settings(self._global_settings, self._project_settings)
                    return

    async def reload(self) -> None:
        """pi: `async reload()` opens with `await this.writeQueue`.

        Draining first matters: a queued write that lands after the re-read
        would be invisible to the reloaded state.
        """
        async with self._load_lock:
            while True:
                await self.flush()
                mutations = self._mutations
                global_settings, global_error = await SettingsManager._try_load_from_storage(self._storage, "global")
                project_settings, project_error = await SettingsManager._try_load_from_storage(
                    self._storage, "project", self._project_trusted
                )
                with self._write_lock:
                    if self._mutations != mutations:
                        continue  # a setter raced the read
                    self._publish_reload(global_settings, global_error, project_settings, project_error)
                    return

    def _publish_reload(
        self,
        global_settings: Settings,
        global_error: Exception | None,
        project_settings: Settings,
        project_error: Exception | None,
    ) -> None:
        """Callers hold `_write_lock`."""
        if global_error is None:
            self._global_settings = global_settings
            self._global_settings_load_error = None
        else:
            self._global_settings_load_error = global_error
            self._record_error("global", global_error)

        self._modified_fields.clear()
        self._modified_nested_fields.clear()
        self._modified_project_fields.clear()
        self._modified_project_nested_fields.clear()

        if project_error is None:
            self._project_settings = project_settings
            self._project_settings_load_error = None
        else:
            self._project_settings_load_error = project_error
            self._record_error("project", project_error)

        self._settings = deep_merge_settings(self._global_settings, self._project_settings)

    def apply_overrides(self, overrides: Settings) -> None:
        """Apply additional overrides on top of current settings."""
        with self._write_lock:
            self._settings = deep_merge_settings(self._settings, overrides)

    # -- persistence ----------------------------------------------------------

    def _mark_modified(self, field: str, nested_key: str | None = None) -> None:
        self._modified_fields.add(field)
        if nested_key:
            self._modified_nested_fields.setdefault(field, set()).add(nested_key)

    def _mark_project_modified(self, field: str, nested_key: str | None = None) -> None:
        self._modified_project_fields.add(field)
        if nested_key:
            self._modified_project_nested_fields.setdefault(field, set()).add(nested_key)

    def _assert_project_trusted_for_write(self) -> None:
        if not self._project_trusted:
            raise Exception("Project is not trusted; refusing to write project settings")

    def _record_error(self, scope: SettingsScope, error: Exception) -> None:
        with self._write_lock:
            self._errors.append(_to_settings_error(scope, error, self._settings_paths.get(scope)))

    def _clear_modified_scope(self, scope: SettingsScope) -> None:
        with self._write_lock:
            if scope == "global":
                self._modified_fields.clear()
                self._modified_nested_fields.clear()
                return

            self._modified_project_fields.clear()
            self._modified_project_nested_fields.clear()

    def _enqueue_write(self, scope: SettingsScope, persist: Callable[[str | None], str]) -> None:
        """Append to the write chain and return, mirroring pi's
        `writeQueue = writeQueue.then(task).catch(recordError)`. `persist`
        maps the scope's current file content to the new one.

        Stays sync so setters stay sync: pi's setters do not await either, and
        the TUI invokes them from synchronous key handling. Errors are recorded
        rather than raised, exactly as pi's `.catch` does — the caller has never
        been able to observe a write failure, in pi or here.

        In-memory storage runs inline: there is no filesystem to get off, and
        spawning would demand a live runtime for what is a dict assignment.
        """
        if isinstance(self._storage, InMemorySettingsStorage):
            self._run_write(scope, persist)
            return
        self._ensure_writer().send((scope, persist, None))

    def _ensure_writer(self) -> Any:
        """Return the writer's channel, starting the writer task on first use."""
        with self._write_lock:
            if self._writes is None:
                sender, receiver = channel.unbounded()
                tonio.spawn.without_tracking(self._write_loop(receiver))
                self._writes = sender
            return self._writes

    def _run_write(self, scope: SettingsScope, persist: Callable[[str | None], str]) -> None:
        try:
            if scope == "project":
                self._assert_project_trusted_for_write()
            self._storage.with_lock(scope, persist)
            self._clear_modified_scope(scope)
        except Exception as error:
            self._record_error(scope, error)

    async def _write_loop(self, receiver: Any) -> None:
        """Items are `(scope, persist, done)`: a write carries `persist`, a
        `flush()` ticket carries `done` (set once everything before it is
        done)."""
        while True:
            scope, persist, done = await receiver.receive()
            if persist is not None:
                try:
                    if scope == "project":
                        self._assert_project_trusted_for_write()
                    await self._storage.with_lock_async(scope, persist)
                    self._clear_modified_scope(scope)
                except Exception as error:
                    self._record_error(scope, error)
            if done is not None:
                done.set()

    @staticmethod
    def _persist_fn(
        snapshot_settings: Settings,
        modified_fields: set[str],
        modified_nested_fields: dict[str, set[str]],
    ) -> Callable[[str | None], str]:
        """The write for a scope: its file content, with the modified fields
        taken from `snapshot_settings`."""

        def persist(current: str | None) -> str:
            current_file_settings: Settings = (
                SettingsManager._migrate_settings(json.loads(strip_bom(current))) if current else {}
            )
            merged_settings = dict(current_file_settings)
            for field in modified_fields:
                value = snapshot_settings.get(field)
                if field in modified_nested_fields and isinstance(value, dict):
                    nested_modified = modified_nested_fields[field]
                    base_nested = current_file_settings.get(field)
                    merged_nested = dict(base_nested) if isinstance(base_nested, dict) else {}
                    for nested_key in nested_modified:
                        merged_nested[nested_key] = value.get(nested_key)
                    merged_settings[field] = merged_nested
                else:
                    merged_settings[field] = value

            return json.dumps(merged_settings, indent=2)

        return persist

    def _save(self) -> None:
        """Publish the merged snapshot and enqueue persistence. Callers hold
        `_write_lock` (all mutation funnels through the epoch helpers)."""
        self._settings = deep_merge_settings(self._global_settings, self._project_settings)

        if self._global_settings_load_error is not None:
            return

        # The published epoch is immutable, so the write task can hold it
        # directly — pi's defensive copy protected against later mutation.
        snapshot_global_settings = self._global_settings
        modified_fields = set(self._modified_fields)
        modified_nested_fields = {key: set(value) for key, value in self._modified_nested_fields.items()}

        self._enqueue_write(
            "global", SettingsManager._persist_fn(snapshot_global_settings, modified_fields, modified_nested_fields)
        )

    def _save_project_settings(self, settings: Settings) -> None:
        with self._write_lock:
            self._assert_project_trusted_for_write()
            self._project_settings = copy.deepcopy(settings)
            self._mutations += 1
            self._settings = deep_merge_settings(self._global_settings, self._project_settings)

            if self._project_settings_load_error is not None:
                return

            # Immutable epoch — safe to hand to the write task as-is.
            snapshot_project_settings = self._project_settings
            modified_fields = set(self._modified_project_fields)
            modified_nested_fields = {key: set(value) for key, value in self._modified_project_nested_fields.items()}
            self._enqueue_write(
                "project",
                SettingsManager._persist_fn(snapshot_project_settings, modified_fields, modified_nested_fields),
            )

    def _update_project_settings(self, field: str, update: Any) -> None:
        with self._write_lock:
            self._assert_project_trusted_for_write()
            project_settings = copy.deepcopy(self._project_settings)
            update(project_settings)
            self._mark_project_modified(field)
            self._save_project_settings(project_settings)

    def _update_global_settings(self, update: Callable[[Settings], None]) -> None:
        """Epoch builder for the global scope: deep-copy the published
        snapshot, run pi's setter mutations on the private copy (`update`,
        which also marks modified fields), publish by rebinding, persist.
        The published dict is never mutated after this rebind."""
        with self._write_lock:
            settings = copy.deepcopy(self._global_settings)
            update(settings)
            self._global_settings = settings
            self._mutations += 1
            self._save()

    def _set_global(self, field: str, value: Any) -> None:
        def update(settings: Settings) -> None:
            settings[field] = value
            self._mark_modified(field)

        self._update_global_settings(update)

    def _set_global_nested(self, field: str, key: str, value: Any) -> None:
        def update(settings: Settings) -> None:
            if not isinstance(settings.get(field), dict):
                settings[field] = {}
            settings[field][key] = value
            self._mark_modified(field, key)

        self._update_global_settings(update)

    async def flush(self) -> None:
        """Wait for queued writes, mirroring pi's `await this.writeQueue`."""
        writes = self._writes
        if writes is None:
            return
        ticket = tonio.Event()
        writes.send((None, None, ticket))
        await ticket.wait()

    def drain_errors(self) -> list[SettingsError]:
        with self._write_lock:
            drained = list(self._errors)
            self._errors = []
        return drained

    # -- individual settings --------------------------------------------------

    def get_last_changelog_version(self) -> str | None:
        return self._settings.get("lastChangelogVersion")

    def set_last_changelog_version(self, version: str) -> None:
        self._set_global("lastChangelogVersion", version)

    def get_session_dir(self) -> str | None:
        session_dir = self._settings.get("sessionDir")
        return normalize_path(session_dir) if session_dir else session_dir

    def get_default_provider(self) -> str | None:
        return self._settings.get("defaultProvider")

    def get_default_model(self) -> str | None:
        return self._settings.get("defaultModel")

    def set_default_provider(self, provider: str) -> None:
        self._set_global("defaultProvider", provider)

    def set_default_model(self, model_id: str) -> None:
        self._set_global("defaultModel", model_id)

    def set_default_model_and_provider(self, provider: str, model_id: str) -> None:
        def update(settings: Settings) -> None:
            settings["defaultProvider"] = provider
            settings["defaultModel"] = model_id
            self._mark_modified("defaultProvider")
            self._mark_modified("defaultModel")

        self._update_global_settings(update)

    def get_steering_mode(self) -> str:
        return self._settings.get("steeringMode") or "one-at-a-time"

    def set_steering_mode(self, mode: str) -> None:
        self._set_global("steeringMode", mode)

    def get_follow_up_mode(self) -> str:
        return self._settings.get("followUpMode") or "one-at-a-time"

    def set_follow_up_mode(self, mode: str) -> None:
        self._set_global("followUpMode", mode)

    def get_theme_setting(self) -> str | None:
        value = self._settings.get("theme")
        return value if isinstance(value, str) else None

    def get_theme(self) -> str | None:
        theme = self.get_theme_setting()
        return None if theme is not None and "/" in theme else theme

    def set_theme(self, theme: str) -> None:
        self._set_global("theme", theme)

    def get_default_thinking_level(self) -> str | None:
        return self._settings.get("defaultThinkingLevel")

    def set_default_thinking_level(self, level: str) -> None:
        self._set_global("defaultThinkingLevel", level)

    def get_model_thinking_level(self, provider: str, model_id: str) -> str | None:
        """Per-model default thinking level override, keyed by "provider/modelId"."""
        return (self._settings.get("modelThinkingLevels") or {}).get(f"{provider}/{model_id}")

    def get_all_model_thinking_levels(self) -> dict[str, str]:
        return dict(self._settings.get("modelThinkingLevels") or {})

    def set_model_thinking_level(self, provider: str, model_id: str, level: str) -> None:
        def update(settings: Settings) -> None:
            overrides = settings.setdefault("modelThinkingLevels", {})
            overrides[f"{provider}/{model_id}"] = level
            self._mark_modified("modelThinkingLevels")

        self._update_global_settings(update)

    def remove_model_thinking_level(self, provider: str, model_id: str) -> None:
        if not self._global_settings.get("modelThinkingLevels"):
            return

        def update(settings: Settings) -> None:
            overrides = settings.get("modelThinkingLevels")
            if not overrides:
                return
            overrides.pop(f"{provider}/{model_id}", None)
            if not overrides:
                del settings["modelThinkingLevels"]
            self._mark_modified("modelThinkingLevels")

        self._update_global_settings(update)

    def get_transport(self) -> str:
        transport = self._settings.get("transport")
        return transport if transport is not None else "auto"

    def set_transport(self, transport: str) -> None:
        self._set_global("transport", transport)

    def get_compaction_enabled(self) -> bool:
        enabled = (self._settings.get("compaction") or {}).get("enabled")
        return enabled if enabled is not None else True

    def set_compaction_enabled(self, enabled: bool) -> None:
        self._set_global_nested("compaction", "enabled", enabled)

    def get_compaction_reserve_tokens(self, model: Any = None) -> int:
        return _compaction_token_setting(self._settings.get("compaction"), "reserveTokens", model)

    def get_compaction_keep_recent_tokens(self, model: Any = None) -> int:
        return _compaction_token_setting(self._settings.get("compaction"), "keepRecentTokens", model)

    def get_compaction_settings(self, model: Any = None) -> dict[str, Any]:
        """Resolve each token setting through model override, ordinary setting, then built-in default."""
        # One pinned snapshot read: the three values cannot mix epochs the way
        # delegating to the single-key getters would under a concurrent swap.
        compaction = self._settings.get("compaction")
        enabled = (compaction or {}).get("enabled")
        return {
            "enabled": enabled if enabled is not None else True,
            "reserve_tokens": _compaction_token_setting(compaction, "reserveTokens", model),
            "keep_recent_tokens": _compaction_token_setting(compaction, "keepRecentTokens", model),
        }

    def get_branch_summary_settings(self) -> dict[str, Any]:
        branch_summary = self._settings.get("branchSummary") or {}
        reserve = branch_summary.get("reserveTokens")
        skip = branch_summary.get("skipPrompt")
        return {
            "reserve_tokens": reserve if reserve is not None else 16384,
            "skip_prompt": skip if skip is not None else False,
        }

    def get_branch_summary_skip_prompt(self) -> bool:
        skip = (self._settings.get("branchSummary") or {}).get("skipPrompt")
        return skip if skip is not None else False

    def get_retry_enabled(self) -> bool:
        enabled = (self._settings.get("retry") or {}).get("enabled")
        return enabled if enabled is not None else True

    def set_retry_enabled(self, enabled: bool) -> None:
        self._set_global_nested("retry", "enabled", enabled)

    def get_retry_settings(self) -> dict[str, Any]:
        # One pinned snapshot read (see get_compaction_settings).
        retry = self._settings.get("retry") or {}
        enabled = retry.get("enabled")
        max_retries = retry.get("maxRetries")
        base_delay_ms = retry.get("baseDelayMs")
        max_agent_delay_ms = retry.get("maxAgentDelayMs")
        return {
            "enabled": enabled if enabled is not None else True,
            "max_retries": max_retries if max_retries is not None else 3,
            "base_delay_ms": base_delay_ms if base_delay_ms is not None else 2000,
            "max_agent_delay_ms": (
                max_agent_delay_ms if max_agent_delay_ms is not None else DEFAULT_MAX_AGENT_RETRY_DELAY_MS
            ),
        }

    def get_http_idle_timeout_ms(self) -> int:
        parsed = _parse_timeout_setting(self._settings.get("httpIdleTimeoutMs"), "httpIdleTimeoutMs")
        return parsed if parsed is not None else DEFAULT_HTTP_IDLE_TIMEOUT_MS

    def set_http_idle_timeout_ms(self, timeout_ms: float) -> None:
        if (
            not isinstance(timeout_ms, (int, float))
            or isinstance(timeout_ms, bool)
            or not math.isfinite(timeout_ms)
            or timeout_ms < 0
        ):
            raise Exception(f"Invalid httpIdleTimeoutMs setting: {timeout_ms}")
        self._set_global("httpIdleTimeoutMs", math.floor(timeout_ms))

    def get_cache_warming_mode(self) -> CacheWarmingMode:
        """Read from global settings only because warming costs money."""
        mode = self._global_settings.get("cacheWarming")
        return mode if mode in CACHE_WARMING_MODES else "streaming"

    def set_cache_warming_mode(self, mode: CacheWarmingMode) -> None:
        self._set_global("cacheWarming", mode)

    def get_provider_retry_settings(self) -> dict[str, Any]:
        provider = (self._settings.get("retry") or {}).get("provider") or {}
        max_retry_delay_ms = provider.get("maxRetryDelayMs")
        return {
            "timeout_ms": provider.get("timeoutMs"),
            "max_retries": provider.get("maxRetries"),
            "max_retry_delay_ms": max_retry_delay_ms if max_retry_delay_ms is not None else 60000,
        }

    def get_websocket_connect_timeout_ms(self) -> int | None:
        return _parse_timeout_setting(self._settings.get("websocketConnectTimeoutMs"), "websocketConnectTimeoutMs")

    def get_hide_thinking_block(self) -> bool:
        hide = self._settings.get("hideThinkingBlock")
        return hide if hide is not None else False

    def get_show_cache_miss_notices(self) -> bool:
        show = self._settings.get("showCacheMissNotices")
        return show if show is not None else False

    def get_external_editor_command(self) -> str:
        configured_editor = self._settings.get("externalEditor")
        if isinstance(configured_editor, str) and configured_editor.strip() != "":
            return configured_editor
        environment_editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if environment_editor:
            return environment_editor
        return "nano"

    def set_hide_thinking_block(self, hide: bool) -> None:
        self._set_global("hideThinkingBlock", hide)

    def set_show_cache_miss_notices(self, show: bool) -> None:
        self._set_global("showCacheMissNotices", show)

    def get_shell_path(self) -> str | None:
        shell_path = self._settings.get("shellPath")
        return normalize_path(shell_path) if shell_path else shell_path

    def set_shell_path(self, path: str | None) -> None:
        self._set_global("shellPath", path)

    def get_quiet_startup(self) -> bool:
        quiet = self._settings.get("quietStartup")
        return quiet if quiet is not None else False

    def set_quiet_startup(self, quiet: bool) -> None:
        self._set_global("quietStartup", quiet)

    def get_default_project_trust(self) -> str:
        value = self._global_settings.get("defaultProjectTrust")
        return value if value in ("always", "never") else "ask"

    def set_default_project_trust(self, default_project_trust: str) -> None:
        self._set_global("defaultProjectTrust", default_project_trust)

    def get_shell_command_prefix(self) -> str | None:
        return self._settings.get("shellCommandPrefix")

    def set_shell_command_prefix(self, prefix: str | None) -> None:
        self._set_global("shellCommandPrefix", prefix)

    def get_collapse_changelog(self) -> bool:
        collapse = self._settings.get("collapseChangelog")
        return collapse if collapse is not None else False

    def set_collapse_changelog(self, collapse: bool) -> None:
        self._set_global("collapseChangelog", collapse)

    def get_enable_provider_attribution(self) -> bool:
        enabled = self._settings.get("enableProviderAttribution")
        return enabled if enabled is not None else True

    def set_enable_provider_attribution(self, enabled: bool) -> None:
        self._set_global("enableProviderAttribution", enabled)

    def get_or_create_device_id(self) -> str:
        """Stable ID of this installation, e.g. sent to OpenAI as its agent host
        ID. Created on first use. Project settings are ignored so a committed
        project settings file cannot give every clone the same ID.

        The check and the create hold `_write_lock` together, so two logins
        racing on first use get the same ID.
        """
        with self._write_lock:
            device_id = self._global_settings.get("deviceId")
            if not device_id:
                device_id = str(uuid.uuid4())
                self._set_global("deviceId", device_id)
        return device_id

    def get_packages(self) -> list[Any]:
        return list(self._settings.get("packages") or [])

    def set_packages(self, packages: list[Any]) -> None:
        self._set_global("packages", packages)

    def set_project_packages(self, packages: list[Any]) -> None:
        def update(settings: Settings) -> None:
            settings["packages"] = packages

        self._update_project_settings("packages", update)

    def get_extension_paths(self) -> list[str]:
        return list(self._settings.get("extensions") or [])

    def set_extension_paths(self, paths: list[str]) -> None:
        self._set_global("extensions", paths)

    def set_project_extension_paths(self, paths: list[str]) -> None:
        def update(settings: Settings) -> None:
            settings["extensions"] = paths

        self._update_project_settings("extensions", update)

    def get_skill_paths(self) -> list[str]:
        return list(self._settings.get("skills") or [])

    def set_skill_paths(self, paths: list[str]) -> None:
        self._set_global("skills", paths)

    def set_project_skill_paths(self, paths: list[str]) -> None:
        def update(settings: Settings) -> None:
            settings["skills"] = paths

        self._update_project_settings("skills", update)

    def get_prompt_template_paths(self) -> list[str]:
        return list(self._settings.get("prompts") or [])

    def set_prompt_template_paths(self, paths: list[str]) -> None:
        self._set_global("prompts", paths)

    def set_project_prompt_template_paths(self, paths: list[str]) -> None:
        def update(settings: Settings) -> None:
            settings["prompts"] = paths

        self._update_project_settings("prompts", update)

    def get_theme_paths(self) -> list[str]:
        return list(self._settings.get("themes") or [])

    def set_theme_paths(self, paths: list[str]) -> None:
        self._set_global("themes", paths)

    def set_project_theme_paths(self, paths: list[str]) -> None:
        def update(settings: Settings) -> None:
            settings["themes"] = paths

        self._update_project_settings("themes", update)

    def get_enable_skill_commands(self) -> bool:
        enabled = self._settings.get("enableSkillCommands")
        return enabled if enabled is not None else True

    def set_enable_skill_commands(self, enabled: bool) -> None:
        self._set_global("enableSkillCommands", enabled)

    def get_thinking_budgets(self) -> dict[str, Any] | None:
        return self._settings.get("thinkingBudgets")

    def get_terminal_capability_overrides(self) -> dict[str, Any]:
        terminal = self._settings.get("terminal") or {}
        images = terminal.get("images")
        return {
            **({"images": images} if images in ("kitty", "iterm2") else {"images": None} if images is False else {}),
            **({"trueColor": terminal["trueColor"]} if isinstance(terminal.get("trueColor"), bool) else {}),
            **({"hyperlinks": terminal["hyperlinks"]} if isinstance(terminal.get("hyperlinks"), bool) else {}),
        }

    def get_show_images(self) -> bool:
        show = (self._settings.get("terminal") or {}).get("showImages")
        return show if show is not None else True

    def set_show_images(self, show: bool) -> None:
        self._set_global_nested("terminal", "showImages", show)

    def get_image_width_cells(self) -> int:
        width = (self._settings.get("terminal") or {}).get("imageWidthCells")
        if isinstance(width, bool) or not isinstance(width, (int, float)) or not math.isfinite(width):
            return 60
        return max(1, math.floor(width))

    def set_image_width_cells(self, width: float) -> None:
        self._set_global_nested("terminal", "imageWidthCells", max(1, math.floor(width)))

    def get_clear_on_shrink(self) -> bool:
        # Settings takes precedence, then env var, then default false
        clear_on_shrink = (self._settings.get("terminal") or {}).get("clearOnShrink")
        if clear_on_shrink is not None:
            return clear_on_shrink
        return os.environ.get("PIDREI_CLEAR_ON_SHRINK") == "1"

    def set_clear_on_shrink(self, enabled: bool) -> None:
        self._set_global_nested("terminal", "clearOnShrink", enabled)

    def get_show_terminal_progress(self) -> bool:
        show = (self._settings.get("terminal") or {}).get("showTerminalProgress")
        return show if show is not None else False

    def set_show_terminal_progress(self, enabled: bool) -> None:
        self._set_global_nested("terminal", "showTerminalProgress", enabled)

    def get_fullscreen_exit_output(self) -> str:
        return "resume-hint" if self._settings.get("fullscreenExitOutput") == "resume-hint" else "transcript"

    def set_fullscreen_exit_output(self, output: str) -> None:
        self._set_global("fullscreenExitOutput", output)

    def get_fullscreen_scrollbar(self) -> str:
        mode = self._settings.get("fullscreenScrollbar")
        return mode if mode in ("always", "hidden") else "auto"

    def set_fullscreen_scrollbar(self, mode: str) -> None:
        self._set_global("fullscreenScrollbar", mode)

    def get_fullscreen_copy_on_select(self) -> bool:
        enabled = self._settings.get("fullscreenCopyOnSelect")
        return enabled if enabled is not None else True

    def set_fullscreen_copy_on_select(self, enabled: bool) -> None:
        self._set_global("fullscreenCopyOnSelect", enabled)

    def get_fullscreen_wheel_scroll_lines(self) -> WheelScrollLines:
        lines = self._settings.get("fullscreenWheelScrollLines")
        # pi: `typeof lines === "number"` — a JSON bool is not a number there.
        if isinstance(lines, int | float) and not isinstance(lines, bool) and math.isfinite(lines):
            return max(1, min(100, math.floor(lines)))
        return "auto"

    def set_fullscreen_wheel_scroll_lines(self, lines: WheelScrollLines) -> None:
        self._set_global(
            "fullscreenWheelScrollLines", lines if lines == "auto" else max(1, min(100, math.floor(lines)))
        )

    def get_image_auto_resize(self) -> bool:
        auto_resize = (self._settings.get("images") or {}).get("autoResize")
        return auto_resize if auto_resize is not None else True

    def set_image_auto_resize(self, enabled: bool) -> None:
        self._set_global_nested("images", "autoResize", enabled)

    def get_block_images(self) -> bool:
        blocked = (self._settings.get("images") or {}).get("blockImages")
        return blocked if blocked is not None else False

    def set_block_images(self, blocked: bool) -> None:
        self._set_global_nested("images", "blockImages", blocked)

    def get_enabled_models(self) -> list[str] | None:
        return self._settings.get("enabledModels")

    def get_default_tools(self) -> list[str] | None:
        """The resolved `defaultTools` selection, or None when no settings layer sets it.

        pi returns undefined only for an absent key; a JSON `null` (like any
        non-list) resolves as an empty list."""
        settings = self._settings
        if "defaultTools" not in settings:
            return None
        tools = settings["defaultTools"]
        return _resolve_default_tools(
            [tool for tool in tools if isinstance(tool, str)] if isinstance(tools, list) else []
        )

    def set_enabled_models(self, patterns: list[str] | None) -> None:
        self._set_global("enabledModels", patterns)

    def get_double_escape_action(self) -> str:
        action = self._settings.get("doubleEscapeAction")
        return action if action is not None else "tree"

    def set_double_escape_action(self, action: str) -> None:
        self._set_global("doubleEscapeAction", action)

    def get_tree_filter_mode(self) -> str:
        mode = self._settings.get("treeFilterMode")
        valid = ["default", "no-tools", "user-only", "labeled-only", "all"]
        return mode if mode and mode in valid else "default"

    def set_tree_filter_mode(self, mode: str) -> None:
        self._set_global("treeFilterMode", mode)

    def get_show_hardware_cursor(self) -> bool:
        show = self._settings.get("showHardwareCursor")
        return show if show is not None else os.environ.get("PIDREI_HARDWARE_CURSOR") == "1"

    def set_show_hardware_cursor(self, enabled: bool) -> None:
        self._set_global("showHardwareCursor", enabled)

    def get_tui_mode(self) -> str:
        return "fullscreen" if self._settings.get("tuiMode") == "fullscreen" else "regular"

    def set_tui_mode(self, mode: str) -> None:
        self._set_global("tuiMode", mode)

    def get_editor_padding_x(self) -> int:
        padding = self._settings.get("editorPaddingX")
        return padding if padding is not None else 0

    def set_editor_padding_x(self, padding: float) -> None:
        self._set_global("editorPaddingX", max(0, min(3, math.floor(padding))))

    def get_output_pad(self) -> int:
        return 0 if self._settings.get("outputPad") == 0 else 1

    def set_output_pad(self, padding: int) -> None:
        self._set_global("outputPad", padding)

    def get_autocomplete_max_visible(self) -> int:
        max_visible = self._settings.get("autocompleteMaxVisible")
        return max_visible if max_visible is not None else 5

    def set_autocomplete_max_visible(self, max_visible: float) -> None:
        self._set_global("autocompleteMaxVisible", max(3, min(20, math.floor(max_visible))))

    def get_code_block_indent(self) -> str:
        indent = (self._settings.get("markdown") or {}).get("codeBlockIndent")
        return indent if indent is not None else "  "

    def get_warnings(self) -> dict[str, Any]:
        return dict(self._settings.get("warnings") or {})

    def set_warnings(self, warnings: dict[str, Any]) -> None:
        self._set_global("warnings", dict(warnings))
