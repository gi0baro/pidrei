"""Mirror of pi coding-agent src/core/settings-manager.ts.

Settings are represented as plain dicts with the on-disk camelCase keys
(pi's Settings interface is an open JS object: unknown keys must survive
load/merge/save round-trips, which a typed model would drop).

pi defers writes on a promise chain (`writeQueue`) drained by `flush()`.
Here writes execute synchronously under a lock, preserving the observable
contract: sets are ordered, write errors are recorded (not raised) and
surfaced via drain_errors(), and flush() is a synchronization no-op.
"""

import copy
import json
import math
import os
import threading
import uuid
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import tonio.colored as tonio
from tonio.exceptions import RuntimeNotInitializedError

from ..config import CONFIG_DIR_NAME, get_agent_dir
from ..utils.lockfile import acquire_lock_sync_with_retry
from ..utils.paths import normalize_path, resolve_path
from .http_config import DEFAULT_HTTP_IDLE_TIMEOUT_MS, parse_http_idle_timeout_ms


type Settings = dict[str, Any]
type SettingsScope = Literal["global", "project"]


def _is_mergeable_object(value: Any) -> bool:
    return isinstance(value, dict)


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


def deep_merge_settings(base: Settings, overrides: Settings) -> Settings:
    """Deep merge settings: project/overrides take precedence, nested objects merge recursively."""
    return _deep_merge_objects(base, overrides)


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


class SettingsStorage(Protocol):
    def with_lock(self, scope: SettingsScope, fn: Any) -> None: ...


class FileSettingsStorage:
    def __init__(self, cwd: str, agent_dir: str):
        resolved_cwd = resolve_path(cwd)
        resolved_agent_dir = resolve_path(agent_dir)
        self._global_settings_path = os.path.join(resolved_agent_dir, "settings.json")
        self._project_settings_path = os.path.join(resolved_cwd, CONFIG_DIR_NAME, "settings.json")

    def with_lock(self, scope: SettingsScope, fn: Any) -> None:
        path = self._global_settings_path if scope == "global" else self._project_settings_path
        directory = os.path.dirname(path)

        release = None
        try:
            # Only create directory and lock if file exists or we need to write
            file_exists = os.path.exists(path)
            if file_exists:
                release = acquire_lock_sync_with_retry(path)
            current = None
            if file_exists:
                with open(path, encoding="utf-8") as f:
                    current = f.read()
            next_content = fn(current)
            if next_content is not None:
                # Only create directory when we actually need to write
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                if release is None:
                    release = acquire_lock_sync_with_retry(path)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(next_content)
        finally:
            if release is not None:
                release()


class InMemorySettingsStorage:
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


class SettingsManager:
    def __init__(
        self,
        storage: SettingsStorage,
        initial_global: Settings,
        initial_project: Settings,
        global_load_error: Exception | None = None,
        project_load_error: Exception | None = None,
        initial_errors: list[SettingsError] | None = None,
        project_trusted: bool = True,
    ):
        self._storage = storage
        self._global_settings = initial_global
        self._project_settings = initial_project
        self._project_trusted = project_trusted
        self._modified_fields: set[str] = set()
        self._modified_nested_fields: dict[str, set[str]] = {}
        self._modified_project_fields: set[str] = set()
        self._modified_project_nested_fields: dict[str, set[str]] = {}
        self._global_settings_load_error = global_load_error
        self._project_settings_load_error = project_load_error
        self._write_lock = threading.RLock()
        # Tail of the write chain, mirroring pi's `writeQueue` promise.
        # `None` means nothing is pending. Guarded by `_write_lock`, which
        # is never held across an await.
        self._write_tail: tonio.Event | None = None
        self._errors: list[SettingsError] = list(initial_errors or [])
        self._settings = deep_merge_settings(self._global_settings, self._project_settings)

    # -- constructors ---------------------------------------------------------

    @staticmethod
    def create(cwd: str, agent_dir: str | None = None, *, project_trusted: bool = True) -> Awaitable[SettingsManager]:
        """Create a SettingsManager that loads from files.

        Construction reads both settings files under a lock, so it is one
        blocking unit handed to the pool — the same treatment the `SessionManager`
        factories get. Sync def returning the awaitable, per the standing rule.
        """
        return tonio.spawn_blocking(SettingsManager.create_sync, cwd, agent_dir, project_trusted=project_trusted)

    @staticmethod
    def create_sync(cwd: str, agent_dir: str | None = None, *, project_trusted: bool = True) -> SettingsManager:
        """Blocking construction. Only for callers already off the runtime —
        CLI code before `tonio.run`, and pool bodies. Everything on the runtime
        awaits `create()`."""
        storage = FileSettingsStorage(cwd, agent_dir if agent_dir is not None else get_agent_dir())
        return SettingsManager.from_storage(storage, project_trusted=project_trusted)

    @staticmethod
    def from_storage(storage: SettingsStorage, *, project_trusted: bool = True) -> SettingsManager:
        """Create a SettingsManager from an arbitrary storage backend."""
        global_settings, global_error = SettingsManager._try_load_from_storage(storage, "global")
        project_settings, project_error = SettingsManager._try_load_from_storage(storage, "project", project_trusted)
        initial_errors: list[SettingsError] = []
        if global_error is not None:
            initial_errors.append(SettingsError("global", global_error))
        if project_error is not None:
            initial_errors.append(SettingsError("project", project_error))

        return SettingsManager(
            storage,
            global_settings,
            project_settings,
            global_error,
            project_error,
            initial_errors,
            project_trusted,
        )

    @staticmethod
    def in_memory(settings: Settings | None = None, *, project_trusted: bool = True) -> SettingsManager:
        """Create an in-memory SettingsManager (no file I/O)."""
        storage = InMemorySettingsStorage()
        initial_settings = SettingsManager._migrate_settings(copy.deepcopy(settings or {}))
        storage.with_lock("global", lambda _current: json.dumps(initial_settings, indent=2))
        return SettingsManager.from_storage(storage, project_trusted=project_trusted)

    @staticmethod
    def _load_from_storage(storage: SettingsStorage, scope: SettingsScope, project_trusted: bool = True) -> Settings:
        if scope == "project" and not project_trusted:
            return {}

        content: str | None = None

        def read(current: str | None) -> None:
            nonlocal content
            content = current

        storage.with_lock(scope, read)

        if not content:
            return {}
        settings = json.loads(content)
        return SettingsManager._migrate_settings(settings)

    @staticmethod
    def _try_load_from_storage(
        storage: SettingsStorage, scope: SettingsScope, project_trusted: bool = True
    ) -> tuple[Settings, Exception | None]:
        try:
            return SettingsManager._load_from_storage(storage, scope, project_trusted), None
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

    def set_project_trusted(self, trusted: bool) -> None:
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

        project_settings, project_error = SettingsManager._try_load_from_storage(self._storage, "project", trusted)
        self._project_settings = project_settings
        self._project_settings_load_error = project_error
        if project_error is not None:
            self._record_error("project", project_error)
        self._settings = deep_merge_settings(self._global_settings, self._project_settings)

    async def reload(self) -> None:
        """pi: `async reload()` opens with `await this.writeQueue`.

        Draining first matters: a queued write that lands after the re-read
        would be invisible to the reloaded state.
        """
        await self.flush()
        with self._write_lock:
            global_settings, global_error = SettingsManager._try_load_from_storage(self._storage, "global")
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

            project_settings, project_error = SettingsManager._try_load_from_storage(
                self._storage, "project", self._project_trusted
            )
            if project_error is None:
                self._project_settings = project_settings
                self._project_settings_load_error = None
            else:
                self._project_settings_load_error = project_error
                self._record_error("project", project_error)

            self._settings = deep_merge_settings(self._global_settings, self._project_settings)

    def apply_overrides(self, overrides: Settings) -> None:
        """Apply additional overrides on top of current settings."""
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
        self._errors.append(SettingsError(scope, error))

    def _clear_modified_scope(self, scope: SettingsScope) -> None:
        if scope == "global":
            self._modified_fields.clear()
            self._modified_nested_fields.clear()
            return

        self._modified_project_fields.clear()
        self._modified_project_nested_fields.clear()

    def _enqueue_write(self, scope: SettingsScope, task: Any) -> None:
        """Append to the write chain and return, mirroring pi's
        `writeQueue = writeQueue.then(task).catch(recordError)`.

        Stays sync so setters stay sync: pi's setters do not await either, and
        the TUI invokes them from synchronous key handling. Errors are recorded
        rather than raised, exactly as pi's `.catch` does — the caller has never
        been able to observe a write failure, in pi or here.

        In-memory storage runs inline: there is no filesystem to get off, and
        spawning would demand a live runtime for what is a dict assignment.
        """
        if not isinstance(self._storage, FileSettingsStorage):
            self._run_write(scope, task)
            return

        with self._write_lock:
            previous, mine = self._write_tail, tonio.Event()
            self._write_tail = mine
        queued = self._run_queued_write(previous, mine, scope, task)
        try:
            tonio.spawn.without_tracking(queued)
        except RuntimeNotInitializedError:
            # No runtime means no worker to block, so blocking here cannot
            # violate the policy — the same boundary condition that puts
            # import-time code outside it. Reached from sync CLI paths and
            # from tests that drive the manager without `tonio.run`.
            queued.close()
            self._write_tail = previous
            self._run_write(scope, task)
            mine.set()

    def _run_write(self, scope: SettingsScope, task: Any) -> None:
        try:
            if scope == "project":
                self._assert_project_trusted_for_write()
            task()
            self._clear_modified_scope(scope)
        except Exception as error:
            self._record_error(scope, error)

    async def _run_queued_write(
        self, previous: tonio.Event | None, mine: tonio.Event, scope: SettingsScope, task: Any
    ) -> None:
        try:
            if previous is not None:
                await previous.wait()  # the `.then()`: writes stay ordered
            if scope == "project":
                self._assert_project_trusted_for_write()
            await tonio.spawn_blocking(task)
            self._clear_modified_scope(scope)
        except Exception as error:
            self._record_error(scope, error)
        finally:
            mine.set()

    def _persist_scoped_settings(
        self,
        scope: SettingsScope,
        snapshot_settings: Settings,
        modified_fields: set[str],
        modified_nested_fields: dict[str, set[str]],
    ) -> None:
        def persist(current: str | None) -> str:
            current_file_settings: Settings = SettingsManager._migrate_settings(json.loads(current)) if current else {}
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

        self._storage.with_lock(scope, persist)

    def _save(self) -> None:
        self._settings = deep_merge_settings(self._global_settings, self._project_settings)

        if self._global_settings_load_error is not None:
            return

        snapshot_global_settings = copy.deepcopy(self._global_settings)
        modified_fields = set(self._modified_fields)
        modified_nested_fields = {key: set(value) for key, value in self._modified_nested_fields.items()}

        self._enqueue_write(
            "global",
            lambda: self._persist_scoped_settings(
                "global", snapshot_global_settings, modified_fields, modified_nested_fields
            ),
        )

    def _save_project_settings(self, settings: Settings) -> None:
        self._assert_project_trusted_for_write()
        self._project_settings = copy.deepcopy(settings)
        self._settings = deep_merge_settings(self._global_settings, self._project_settings)

        if self._project_settings_load_error is not None:
            return

        snapshot_project_settings = copy.deepcopy(self._project_settings)
        modified_fields = set(self._modified_project_fields)
        modified_nested_fields = {key: set(value) for key, value in self._modified_project_nested_fields.items()}
        self._enqueue_write(
            "project",
            lambda: self._persist_scoped_settings(
                "project", snapshot_project_settings, modified_fields, modified_nested_fields
            ),
        )

    def _update_project_settings(self, field: str, update: Any) -> None:
        self._assert_project_trusted_for_write()
        project_settings = copy.deepcopy(self._project_settings)
        update(project_settings)
        self._mark_project_modified(field)
        self._save_project_settings(project_settings)

    async def flush(self) -> None:
        """Wait for queued writes, mirroring pi's `await this.writeQueue`."""
        tail = self._write_tail
        if tail is not None:
            await tail.wait()

    def drain_errors(self) -> list[SettingsError]:
        drained = list(self._errors)
        self._errors = []
        return drained

    # -- individual settings --------------------------------------------------

    def get_last_changelog_version(self) -> str | None:
        return self._settings.get("lastChangelogVersion")

    def set_last_changelog_version(self, version: str) -> None:
        self._global_settings["lastChangelogVersion"] = version
        self._mark_modified("lastChangelogVersion")
        self._save()

    def get_session_dir(self) -> str | None:
        session_dir = self._settings.get("sessionDir")
        return normalize_path(session_dir) if session_dir else session_dir

    def get_default_provider(self) -> str | None:
        return self._settings.get("defaultProvider")

    def get_default_model(self) -> str | None:
        return self._settings.get("defaultModel")

    def set_default_provider(self, provider: str) -> None:
        self._global_settings["defaultProvider"] = provider
        self._mark_modified("defaultProvider")
        self._save()

    def set_default_model(self, model_id: str) -> None:
        self._global_settings["defaultModel"] = model_id
        self._mark_modified("defaultModel")
        self._save()

    def set_default_model_and_provider(self, provider: str, model_id: str) -> None:
        self._global_settings["defaultProvider"] = provider
        self._global_settings["defaultModel"] = model_id
        self._mark_modified("defaultProvider")
        self._mark_modified("defaultModel")
        self._save()

    def get_steering_mode(self) -> str:
        return self._settings.get("steeringMode") or "one-at-a-time"

    def set_steering_mode(self, mode: str) -> None:
        self._global_settings["steeringMode"] = mode
        self._mark_modified("steeringMode")
        self._save()

    def get_follow_up_mode(self) -> str:
        return self._settings.get("followUpMode") or "one-at-a-time"

    def set_follow_up_mode(self, mode: str) -> None:
        self._global_settings["followUpMode"] = mode
        self._mark_modified("followUpMode")
        self._save()

    def get_theme_setting(self) -> str | None:
        value = self._settings.get("theme")
        return value if isinstance(value, str) else None

    def get_theme(self) -> str | None:
        theme = self.get_theme_setting()
        return None if theme is not None and "/" in theme else theme

    def set_theme(self, theme: str) -> None:
        self._global_settings["theme"] = theme
        self._mark_modified("theme")
        self._save()

    def get_default_thinking_level(self) -> str | None:
        return self._settings.get("defaultThinkingLevel")

    def set_default_thinking_level(self, level: str) -> None:
        self._global_settings["defaultThinkingLevel"] = level
        self._mark_modified("defaultThinkingLevel")
        self._save()

    def get_transport(self) -> str:
        transport = self._settings.get("transport")
        return transport if transport is not None else "auto"

    def set_transport(self, transport: str) -> None:
        self._global_settings["transport"] = transport
        self._mark_modified("transport")
        self._save()

    def get_compaction_enabled(self) -> bool:
        enabled = (self._settings.get("compaction") or {}).get("enabled")
        return enabled if enabled is not None else True

    def set_compaction_enabled(self, enabled: bool) -> None:
        if not isinstance(self._global_settings.get("compaction"), dict):
            self._global_settings["compaction"] = {}
        self._global_settings["compaction"]["enabled"] = enabled
        self._mark_modified("compaction", "enabled")
        self._save()

    def get_compaction_reserve_tokens(self) -> int:
        reserve = (self._settings.get("compaction") or {}).get("reserveTokens")
        return reserve if reserve is not None else 16384

    def get_compaction_keep_recent_tokens(self) -> int:
        keep = (self._settings.get("compaction") or {}).get("keepRecentTokens")
        return keep if keep is not None else 20000

    def get_compaction_settings(self) -> dict[str, Any]:
        return {
            "enabled": self.get_compaction_enabled(),
            "reserve_tokens": self.get_compaction_reserve_tokens(),
            "keep_recent_tokens": self.get_compaction_keep_recent_tokens(),
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
        if not isinstance(self._global_settings.get("retry"), dict):
            self._global_settings["retry"] = {}
        self._global_settings["retry"]["enabled"] = enabled
        self._mark_modified("retry", "enabled")
        self._save()

    def get_retry_settings(self) -> dict[str, Any]:
        retry = self._settings.get("retry") or {}
        max_retries = retry.get("maxRetries")
        base_delay_ms = retry.get("baseDelayMs")
        return {
            "enabled": self.get_retry_enabled(),
            "max_retries": max_retries if max_retries is not None else 3,
            "base_delay_ms": base_delay_ms if base_delay_ms is not None else 2000,
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
        self._global_settings["httpIdleTimeoutMs"] = math.floor(timeout_ms)
        self._mark_modified("httpIdleTimeoutMs")
        self._save()

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
        self._global_settings["hideThinkingBlock"] = hide
        self._mark_modified("hideThinkingBlock")
        self._save()

    def set_show_cache_miss_notices(self, show: bool) -> None:
        self._global_settings["showCacheMissNotices"] = show
        self._mark_modified("showCacheMissNotices")
        self._save()

    def get_shell_path(self) -> str | None:
        shell_path = self._settings.get("shellPath")
        return normalize_path(shell_path) if shell_path else shell_path

    def set_shell_path(self, path: str | None) -> None:
        self._global_settings["shellPath"] = path
        self._mark_modified("shellPath")
        self._save()

    def get_quiet_startup(self) -> bool:
        quiet = self._settings.get("quietStartup")
        return quiet if quiet is not None else False

    def set_quiet_startup(self, quiet: bool) -> None:
        self._global_settings["quietStartup"] = quiet
        self._mark_modified("quietStartup")
        self._save()

    def get_default_project_trust(self) -> str:
        value = self._global_settings.get("defaultProjectTrust")
        return value if value in ("always", "never") else "ask"

    def set_default_project_trust(self, default_project_trust: str) -> None:
        self._global_settings["defaultProjectTrust"] = default_project_trust
        self._mark_modified("defaultProjectTrust")
        self._save()

    def get_shell_command_prefix(self) -> str | None:
        return self._settings.get("shellCommandPrefix")

    def set_shell_command_prefix(self, prefix: str | None) -> None:
        self._global_settings["shellCommandPrefix"] = prefix
        self._mark_modified("shellCommandPrefix")
        self._save()

    def get_collapse_changelog(self) -> bool:
        collapse = self._settings.get("collapseChangelog")
        return collapse if collapse is not None else False

    def set_collapse_changelog(self, collapse: bool) -> None:
        self._global_settings["collapseChangelog"] = collapse
        self._mark_modified("collapseChangelog")
        self._save()

    def get_enable_provider_attribution(self) -> bool:
        enabled = self._settings.get("enableProviderAttribution")
        return enabled if enabled is not None else True

    def set_enable_provider_attribution(self, enabled: bool) -> None:
        self._global_settings["enableProviderAttribution"] = enabled
        self._mark_modified("enableProviderAttribution")
        self._save()

    def get_enable_analytics(self) -> bool:
        enabled = self._settings.get("enableAnalytics")
        return enabled if enabled is not None else False

    def get_tracking_id(self) -> str | None:
        return self._settings.get("trackingId")

    def set_enable_analytics(self, enabled: bool) -> None:
        """Set the analytics opt-in preference; generates a tracking identifier on first opt-in."""
        self._global_settings["enableAnalytics"] = enabled
        self._mark_modified("enableAnalytics")
        if enabled and not self._global_settings.get("trackingId"):
            self._global_settings["trackingId"] = str(uuid.uuid4())
            self._mark_modified("trackingId")
        self._save()

    def get_packages(self) -> list[Any]:
        return list(self._settings.get("packages") or [])

    def set_packages(self, packages: list[Any]) -> None:
        self._global_settings["packages"] = packages
        self._mark_modified("packages")
        self._save()

    def set_project_packages(self, packages: list[Any]) -> None:
        def update(settings: Settings) -> None:
            settings["packages"] = packages

        self._update_project_settings("packages", update)

    def get_extension_paths(self) -> list[str]:
        return list(self._settings.get("extensions") or [])

    def set_extension_paths(self, paths: list[str]) -> None:
        self._global_settings["extensions"] = paths
        self._mark_modified("extensions")
        self._save()

    def set_project_extension_paths(self, paths: list[str]) -> None:
        def update(settings: Settings) -> None:
            settings["extensions"] = paths

        self._update_project_settings("extensions", update)

    def get_skill_paths(self) -> list[str]:
        return list(self._settings.get("skills") or [])

    def set_skill_paths(self, paths: list[str]) -> None:
        self._global_settings["skills"] = paths
        self._mark_modified("skills")
        self._save()

    def set_project_skill_paths(self, paths: list[str]) -> None:
        def update(settings: Settings) -> None:
            settings["skills"] = paths

        self._update_project_settings("skills", update)

    def get_prompt_template_paths(self) -> list[str]:
        return list(self._settings.get("prompts") or [])

    def set_prompt_template_paths(self, paths: list[str]) -> None:
        self._global_settings["prompts"] = paths
        self._mark_modified("prompts")
        self._save()

    def set_project_prompt_template_paths(self, paths: list[str]) -> None:
        def update(settings: Settings) -> None:
            settings["prompts"] = paths

        self._update_project_settings("prompts", update)

    def get_theme_paths(self) -> list[str]:
        return list(self._settings.get("themes") or [])

    def set_theme_paths(self, paths: list[str]) -> None:
        self._global_settings["themes"] = paths
        self._mark_modified("themes")
        self._save()

    def set_project_theme_paths(self, paths: list[str]) -> None:
        def update(settings: Settings) -> None:
            settings["themes"] = paths

        self._update_project_settings("themes", update)

    def get_enable_skill_commands(self) -> bool:
        enabled = self._settings.get("enableSkillCommands")
        return enabled if enabled is not None else True

    def set_enable_skill_commands(self, enabled: bool) -> None:
        self._global_settings["enableSkillCommands"] = enabled
        self._mark_modified("enableSkillCommands")
        self._save()

    def get_thinking_budgets(self) -> dict[str, Any] | None:
        return self._settings.get("thinkingBudgets")

    def get_show_images(self) -> bool:
        show = (self._settings.get("terminal") or {}).get("showImages")
        return show if show is not None else True

    def set_show_images(self, show: bool) -> None:
        if not isinstance(self._global_settings.get("terminal"), dict):
            self._global_settings["terminal"] = {}
        self._global_settings["terminal"]["showImages"] = show
        self._mark_modified("terminal", "showImages")
        self._save()

    def get_image_width_cells(self) -> int:
        width = (self._settings.get("terminal") or {}).get("imageWidthCells")
        if isinstance(width, bool) or not isinstance(width, (int, float)) or not math.isfinite(width):
            return 60
        return max(1, math.floor(width))

    def set_image_width_cells(self, width: float) -> None:
        if not isinstance(self._global_settings.get("terminal"), dict):
            self._global_settings["terminal"] = {}
        self._global_settings["terminal"]["imageWidthCells"] = max(1, math.floor(width))
        self._mark_modified("terminal", "imageWidthCells")
        self._save()

    def get_clear_on_shrink(self) -> bool:
        # Settings takes precedence, then env var, then default false
        clear_on_shrink = (self._settings.get("terminal") or {}).get("clearOnShrink")
        if clear_on_shrink is not None:
            return clear_on_shrink
        return os.environ.get("PIDREI_CLEAR_ON_SHRINK") == "1"

    def set_clear_on_shrink(self, enabled: bool) -> None:
        if not isinstance(self._global_settings.get("terminal"), dict):
            self._global_settings["terminal"] = {}
        self._global_settings["terminal"]["clearOnShrink"] = enabled
        self._mark_modified("terminal", "clearOnShrink")
        self._save()

    def get_show_terminal_progress(self) -> bool:
        show = (self._settings.get("terminal") or {}).get("showTerminalProgress")
        return show if show is not None else False

    def set_show_terminal_progress(self, enabled: bool) -> None:
        if not isinstance(self._global_settings.get("terminal"), dict):
            self._global_settings["terminal"] = {}
        self._global_settings["terminal"]["showTerminalProgress"] = enabled
        self._mark_modified("terminal", "showTerminalProgress")
        self._save()

    def get_fullscreen_scrollbar(self) -> str:
        mode = self._settings.get("fullscreenScrollbar")
        return mode if mode in ("always", "hidden") else "auto"

    def set_fullscreen_scrollbar(self, mode: str) -> None:
        self._global_settings["fullscreenScrollbar"] = mode
        self._mark_modified("fullscreenScrollbar")
        self._save()

    def get_image_auto_resize(self) -> bool:
        auto_resize = (self._settings.get("images") or {}).get("autoResize")
        return auto_resize if auto_resize is not None else True

    def set_image_auto_resize(self, enabled: bool) -> None:
        if not isinstance(self._global_settings.get("images"), dict):
            self._global_settings["images"] = {}
        self._global_settings["images"]["autoResize"] = enabled
        self._mark_modified("images", "autoResize")
        self._save()

    def get_block_images(self) -> bool:
        blocked = (self._settings.get("images") or {}).get("blockImages")
        return blocked if blocked is not None else False

    def set_block_images(self, blocked: bool) -> None:
        if not isinstance(self._global_settings.get("images"), dict):
            self._global_settings["images"] = {}
        self._global_settings["images"]["blockImages"] = blocked
        self._mark_modified("images", "blockImages")
        self._save()

    def get_enabled_models(self) -> list[str] | None:
        return self._settings.get("enabledModels")

    def set_enabled_models(self, patterns: list[str] | None) -> None:
        self._global_settings["enabledModels"] = patterns
        self._mark_modified("enabledModels")
        self._save()

    def get_double_escape_action(self) -> str:
        action = self._settings.get("doubleEscapeAction")
        return action if action is not None else "tree"

    def set_double_escape_action(self, action: str) -> None:
        self._global_settings["doubleEscapeAction"] = action
        self._mark_modified("doubleEscapeAction")
        self._save()

    def get_tree_filter_mode(self) -> str:
        mode = self._settings.get("treeFilterMode")
        valid = ["default", "no-tools", "user-only", "labeled-only", "all"]
        return mode if mode and mode in valid else "default"

    def set_tree_filter_mode(self, mode: str) -> None:
        self._global_settings["treeFilterMode"] = mode
        self._mark_modified("treeFilterMode")
        self._save()

    def get_show_hardware_cursor(self) -> bool:
        show = self._settings.get("showHardwareCursor")
        return show if show is not None else os.environ.get("PIDREI_HARDWARE_CURSOR") == "1"

    def set_show_hardware_cursor(self, enabled: bool) -> None:
        self._global_settings["showHardwareCursor"] = enabled
        self._mark_modified("showHardwareCursor")
        self._save()

    def get_tui_mode(self) -> str:
        return "fullscreen" if self._settings.get("tuiMode") == "fullscreen" else "regular"

    def set_tui_mode(self, mode: str) -> None:
        self._global_settings["tuiMode"] = mode
        self._mark_modified("tuiMode")
        self._save()

    def get_editor_padding_x(self) -> int:
        padding = self._settings.get("editorPaddingX")
        return padding if padding is not None else 0

    def set_editor_padding_x(self, padding: float) -> None:
        self._global_settings["editorPaddingX"] = max(0, min(3, math.floor(padding)))
        self._mark_modified("editorPaddingX")
        self._save()

    def get_output_pad(self) -> int:
        return 0 if self._settings.get("outputPad") == 0 else 1

    def set_output_pad(self, padding: int) -> None:
        self._global_settings["outputPad"] = padding
        self._mark_modified("outputPad")
        self._save()

    def get_autocomplete_max_visible(self) -> int:
        max_visible = self._settings.get("autocompleteMaxVisible")
        return max_visible if max_visible is not None else 5

    def set_autocomplete_max_visible(self, max_visible: float) -> None:
        self._global_settings["autocompleteMaxVisible"] = max(3, min(20, math.floor(max_visible)))
        self._mark_modified("autocompleteMaxVisible")
        self._save()

    def get_code_block_indent(self) -> str:
        indent = (self._settings.get("markdown") or {}).get("codeBlockIndent")
        return indent if indent is not None else "  "

    def get_warnings(self) -> dict[str, Any]:
        return dict(self._settings.get("warnings") or {})

    def set_warnings(self, warnings: dict[str, Any]) -> None:
        self._global_settings["warnings"] = dict(warnings)
        self._mark_modified("warnings")
        self._save()
