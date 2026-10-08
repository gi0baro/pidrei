"""Mirror of pi coding-agent src/modes/interactive/interactive-mode.ts.

Interactive mode for the coding agent. Handles TUI rendering and user
interaction, delegating business logic to AgentSession.

Parallelism/async deltas vs pi's single-threaded runtime:
- fire-and-forget promises become spawned tasks (``_spawn_flow`` for UI flows)
- pi async methods whose synchronous prefix is observable are split into a
  synchronous handler that runs the prefix and spawns the rest
- UI state is guarded by the TUI's state lock (spec/ui-island.md): helpers
  take it around their bodies, flows around each stretch between awaits
- ``ui.start()``/``ui.stop()`` are awaited (the tonio TUI driver is async)
"""

import contextlib
import errno
import json
import os
import posixpath
import re
import signal
import subprocess
import termios
import threading
import traceback
import unicodedata
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import replace as dataclass_replace
from types import SimpleNamespace
from typing import Protocol, runtime_checkable

import tonio.colored as tonio
from tonio.colored import fs, signals as tonio_signals

from pidrei_ai.auth.types import AuthOperationOptions, LoginOptions
from pidrei_ai.registry import ModelsRefreshOptions
from pidrei_tui import (
    CombinedAutocompleteProvider,
    Container,
    Markdown,
    Spacer,
    Text,
    TruncatedText,
    TuiAltScreen,
    TuiMainScreen,
    fuzzy_filter,
    get_capabilities,
    hyperlink,
    is_viewport_tui,
    matches_key,
    prime_capabilities,
    set_capability_overrides,
    set_keybindings,
    visible_width,
)
from pidrei_tui.tui import call_sync
from pidrei_utils import clock
from pidrei_utils.cancel import CancelToken as AiCancelToken

from ...config import (
    APP_NAME,
    APP_TITLE,
    CONFIG_DIR_NAME,
    TEMP_DIR,
    VERSION,
    get_agent_dir,
    get_auth_path,
    get_changelog_path,
    get_debug_log_path,
    get_docs_path,
    get_share_viewer_url,
)
from ...core.agent_session import PromptOptions, ScopedModel, parse_skill_block
from ...core.agent_session_runtime import SessionImportFileNotFoundError
from ...core.bash_executor import BashResult
from ...core.cache_stats import (
    CACHE_TTL_MS,
    collect_cache_misses,
    compute_cache_waste,
    detect_cache_miss,
)
from ...core.cache_warmer import format_cache_warming_status, format_cache_warming_usage
from ...core.defaults import DEFAULT_THINKING_LEVEL, THINKING_LEVEL_OPTIONS
from ...core.exec import exec_command
from ...core.extensions.types import ExtensionContext, ProjectTrustContext
from ...core.footer_data_provider import FooterDataProvider
from ...core.http_config import format_http_idle_timeout_ms
from ...core.keybindings import KeybindingsManager
from ...core.message_wire import serialize_message
from ...core.messages import create_compaction_summary_message, create_custom_message
from ...core.model_resolver import (
    DEFAULT_MODEL_PER_PROVIDER,
    find_exact_model_reference_match,
    resolve_model_scope_from_models,
)
from ...core.model_runtime import CredentialSynchronizationError
from ...core.output_guard import (
    attach_terminal,
    detach_terminal,
    drain_output,
    stdout_isatty,
    write_stderr,
    write_stdout,
)
from ...core.package_manager import DefaultPackageManager
from ...core.session_cwd import MissingSessionCwdError, format_missing_session_cwd_prompt
from ...core.session_manager import SessionManager, session_entry_to_context_messages
from ...core.slash_commands import BUILTIN_SLASH_COMMANDS
from ...core.tools.renderers import with_built_in_renderers
from ...core.tools.truncate import TruncationResult
from ...core.trust_manager import (
    ProjectTrustStore,
    get_project_trust_options_blocking,
    has_trust_requiring_project_resources_blocking,
)
from ...core.usage_totals import get_usage_cost_breakdown
from ...utils.changelog import get_new_entries, normalize_changelog_links, parse_changelog
from ...utils.clipboard import copy_to_clipboard, read_clipboard_file_paths, read_clipboard_text
from ...utils.clipboard_image import extension_for_image_mime_type, read_clipboard_image
from ...utils.colors import dim
from ...utils.fd_io import hard_exit
from ...utils.git import parse_git_url
from ...utils.image_process import convert_image_to_png_base64
from ...utils.paths import get_cwd_relative_path
from ...utils.process import probe_tmux_hyperlinks, run_command
from ...utils.shell import kill_tracked_detached_children
from ...utils.temp_file_writer import discard_temp_file
from ...utils.tools_manager import ensure_tool
from ...utils.version_check import RELEASES_URL, check_for_new_version
from .chat_viewport import create_chat_viewport
from .components import (
    ArminComponent,
    AssistantMessageComponent,
    BashExecutionComponent,
    BorderedLoader,
    BranchSummaryMessageComponent,
    BranchSummaryStatusIndicator,
    CompactionStatusIndicator,
    CompactionSummaryMessageComponent,
    CustomEditor,
    CustomEntryComponent,
    CustomMessageComponent,
    DynamicBorder,
    EarendilAnnouncementComponent,
    ExtensionEditorComponent,
    ExtensionInputComponent,
    ExtensionSelectorComponent,
    FooterComponent,
    IdleStatus,
    LoginDialogComponent,
    ModelSelectorComponent,
    OAuthSelectorComponent,
    RetryStatusIndicator,
    ScopedModelsSelectorComponent,
    SessionSelectorComponent,
    SettingsSelectorComponent,
    SkillInvocationMessageComponent,
    ThinkingSelectorComponent,
    ToolExecutionComponent,
    TreeSelectorComponent,
    TrustSelectorComponent,
    UserMessageComponent,
    UserMessageSelectorComponent,
    WorkingStatusIndicator,
    format_auth_selector_provider_type,
    format_key_text,
    format_tokens,
    key_display_text,
    key_hint,
    key_text,
    load_earendil_image_base64,
    raw_key_hint,
)
from .components.pi_logo import pi_logo_lines, pi_wordmark, supports_pi_logo
from .components.themed_text import ThemedText
from .extension_tui import ExtensionTui, guard_overlay_handle
from .external_editor import edit_in_external_editor
from .model_catalog_refresh import refresh_model_catalogs
from .model_search import get_model_search_text
from .theme import (
    SYSTEM_THEME_NAME,
    InteractiveThemeController,
    get_available_themes,
    get_available_themes_with_paths,
    get_editor_theme,
    get_markdown_theme,
    get_theme_by_name,
    on_theme_change,
    set_registered_themes,
    stop_theme_watcher,
    theme,
)
from .tui_renderer import create_interactive_tui, create_interactive_tui_reference


@runtime_checkable
class _OutputPadded(Protocol):
    """Transcript components that follow the outputPad setting (pi's `"setOutputPad" in child`)."""

    def set_output_pad(self, output_pad: int) -> None: ...


class _TimeoutCancel:
    """Mirror of pi's `setTimeout(() => controller.abort(), ms)` pattern for
    bounding catalog refreshes; `timed_out` reports whether the timer (rather
    than a caller) fired."""

    __slots__ = ("timed_out", "token")

    def __init__(self, ms: float):
        self.token = AiCancelToken()
        self.timed_out = False

        async def _expire() -> None:
            await tonio.time.sleep(ms / 1000)
            self.timed_out = True
            self.token.cancel(TimeoutError("The operation timed out."))

        tonio.spawn.without_tracking(_expire())


def _timeout_cancel(ms: float) -> AiCancelToken:
    return _TimeoutCancel(ms).token


async def _is_yes(choice: Awaitable) -> bool:
    return await choice == "Yes"


async def _answer(value):
    return value


async def _wait_for_answer(done, outcome: dict):
    """The wait an extension dialog's handle runs: the dialog's answer, once
    it is settled."""
    await done.wait(None)
    return outcome["value"]


# UI state discipline (spec/ui-island.md). What a call site keeps from pi is
# whether pi runs the call to completion before moving on or fires it and
# forgets it — not its colour.
# - A UI-mutating helper is one synchronous function that takes the UI state
#   lock (`self.ui.state_lock`, reentrant) around its body; any task calls it
#   directly. Nothing awaits under the lock.
# - Task code (flows that await) takes the lock for each stretch between two
#   of its awaits: one hold per stretch, so a frame never shows half of it.
# - A handler pi calls without awaiting still runs its part up to its first
#   await inside the caller: a synchronous handler does that part and spawns
#   the rest with `_spawn_flow` (escapes reach the crash handler).


def is_expandable(obj) -> bool:
    """Whether a component can be expanded/collapsed."""
    return callable(getattr(obj, "set_expanded", None))


def is_working_status_editor(editor) -> bool:
    """Whether the editor opted into rendering the working status in its border
    (pi's WorkingStatusEditor structural check)."""
    return getattr(editor, "embed_working_status", None) is True and callable(
        getattr(editor, "set_working_status_indicator", None)
    )


class ExpandableText(ThemedText):
    def __init__(self, get_collapsed_text, get_expanded_text, expanded=False, padding_x=0, padding_y=0) -> None:
        state = {"expanded": expanded}
        super().__init__(
            lambda: get_expanded_text() if state["expanded"] else get_collapsed_text(), padding_x, padding_y
        )
        self._state = state

    def set_expanded(self, expanded: bool) -> None:
        self._state["expanded"] = expanded
        self.invalidate()


def _is_custom_session_entry(item) -> bool:
    return isinstance(item, dict) and item.get("type") == "custom"


def _is_compaction_cost_notice(item) -> bool:
    """pi's `CompactionCostNotice` is a plain object in the render item union;
    here it stays a dict, like the custom session entries alongside it."""
    return isinstance(item, dict) and item.get("type") == "compaction_cost"


def _is_usage_session_entry(item) -> bool:
    return isinstance(item, dict) and item.get("type") == "usage"


# EIO: tty reads/ioctls from an orphaned background process group, or writes after hangup.
# ENOTTY: the tty was revoked (macOS) and stdin is no longer a terminal.
_DEAD_TERMINAL_ERRNOS = {errno.EIO, errno.EPIPE, errno.ENOTCONN, errno.ENOTTY}


def is_dead_terminal_error(error) -> bool:
    if isinstance(error, OSError):
        return error.errno in _DEAD_TERMINAL_ERRNOS
    # Raw mode fails with `termios.error`, which is not an `OSError`; its first
    # argument is the errno.
    if isinstance(error, termios.error):
        return bool(error.args) and error.args[0] in _DEAD_TERMINAL_ERRNOS
    return False


def _partial_truncation_result(content: str) -> TruncationResult:
    """pi casts ``{truncated: true, content}`` to TruncationResult for display;
    the bash component only reads ``truncated``, so the counters are zeroed."""
    return TruncationResult(
        content=content,
        truncated=True,
        truncated_by=None,
        total_lines=0,
        total_bytes=0,
        output_lines=0,
        output_bytes=0,
        last_line_partial=False,
        first_line_exceeds_limit=False,
        max_lines=0,
        max_bytes=0,
    )


ANTHROPIC_SUBSCRIPTION_AUTH_WARNING = (
    "Anthropic subscription auth is active. Third-party harness usage draws from extra usage and is "
    "billed per token, not your Claude plan limits. Manage extra usage at "
    "https://claude.ai/settings/usage. Disable this warning in /settings."
)


def is_anthropic_subscription_auth_key(api_key) -> bool:
    return isinstance(api_key, str) and api_key.startswith("sk-ant-oat")


def _is_unknown_model(model) -> bool:
    return model is not None and model.provider == "unknown" and model.id == "unknown" and model.api == "unknown"


def _llama_cpp_post_login_guidance(action_label: str, loaded_model_count: int) -> str:
    return (
        f"{action_label}. No llama.cpp models are loaded. Use /llama to load a model, then /model to select it."
        if loaded_model_count == 0
        else f"{action_label}. Use /model to select a loaded llama.cpp model, or /llama to manage models."
    )


_SAFE_SHELL_VALUE_RE = re.compile(r"[^a-zA-Z0-9_\-./~:@]")


def _quote_if_needed(value: str) -> str:
    if len(value) > 0 and not _SAFE_SHELL_VALUE_RE.search(value):
        return value
    escaped = value.replace("'", "'\\''")
    return f"'{escaped}'"


async def format_resume_command(session_manager) -> str | None:
    if not stdout_isatty():
        return None
    if not session_manager.is_persisted():
        return None

    session_file = session_manager.get_session_file()
    if not session_file or not await fs.Path(session_file).exists():
        return None

    args = [APP_NAME]
    if not session_manager.uses_default_session_dir():
        args.extend(["--session-dir", _quote_if_needed(session_manager.get_session_dir())])
    args.extend(["--session", session_manager.get_session_id()])
    return " ".join(args)


_AUTH_TYPE_ORDER = {"oauth": 0, "api_key": 1}


def _create_fuzzy_autocomplete_items(items, prefix, get_search_text, to_autocomplete_item):
    filtered = fuzzy_filter(items, prefix, get_search_text)
    if not filtered:
        return None
    return [to_autocomplete_item(item) for item in filtered]


def _get_login_provider_completion_options(provider_options: list) -> list:
    by_id: dict = {}
    for provider in provider_options:
        existing = by_id.get(provider["id"])
        if existing is not None:
            if provider["authType"] not in existing["authTypes"]:
                existing["authTypes"].append(provider["authType"])
                existing["authTypes"].sort(key=lambda auth_type: _AUTH_TYPE_ORDER[auth_type])
            continue
        by_id[provider["id"]] = {
            "id": provider["id"],
            "name": provider["name"],
            "authTypes": [provider["authType"]],
            "subscription": provider.get("subscription"),
        }
    return sorted(by_id.values(), key=lambda p: (p["name"].lower(), p["name"]))


def _get_login_provider_search_text(provider: dict) -> str:
    auth_types = " ".join(
        f"{auth_type} {format_auth_selector_provider_type(auth_type, provider['subscription'])}"
        for auth_type in provider["authTypes"]
    )
    return f"{provider['id']} {provider['name']} {auth_types}"


def _format_login_provider_completion_description(provider: dict) -> str:
    auth_types = "/".join(
        format_auth_selector_provider_type(auth_type, provider["subscription"]) for auth_type in provider["authTypes"]
    )
    return auth_types if provider["name"] == provider["id"] else f"{provider['name']} · {auth_types}"


class ExtensionUIContext:
    """`ctx.ui` surface for extensions in the TUI (decided 2026-07-28).

    A snake_case object, matching `docs/extensions.md`, the shipped examples
    and `_NoOpUIContext` — pi's counterpart is a camelCase JS object; the
    dict-of-callbacks it was ported as matched neither. The shape is part of
    the contract and identical across the real, no-op and RPC contexts
    (spec/ui-island.md, `ctx.ui`):

    - Setters and getters are synchronous, each one call through the UI
      state lock: the change is whole, and the caller's next line sees it.
    - `apply(fn)` runs a synchronous `fn` under the lock, to group several
      changes into one frame.
    - The waiting methods (`select`, `confirm`, `input`, `editor`,
      `custom`) mount at call time and return an awaitable handle for the
      answer, which may also be dropped. `get_all_themes`, `get_theme` and
      `set_theme` are awaitable (theme loading reads files).

    Callable from anywhere: extension handlers on their own coroutines, and
    component code under the lock (it is reentrant).
    """

    def __init__(self, mode: InteractiveMode) -> None:
        self._mode = mode

    def select(self, title, options, opts=None):
        return self._mode._show_extension_selector(title, options, opts)

    def confirm(self, title, message, opts=None):
        return self._mode._show_extension_confirm(title, message, opts)

    def input(self, title, placeholder=None, opts=None):
        return self._mode._show_extension_input(title, placeholder, opts)

    def notify(self, message, type=None) -> None:
        self._mode._show_extension_notify(message, type)

    def on_terminal_input(self, handler):
        return self._mode._add_extension_terminal_input_listener(handler)

    def set_status(self, key, text) -> None:
        self._mode._set_extension_status(key, text)

    def apply(self, fn):
        """pidrei-only: run the synchronous `fn` under the UI state lock, as
        one whole change, and return its result. Coroutines are refused."""
        return self._mode.ui.apply(fn)

    def set_working_message(self, message=None) -> None:
        mode = self._mode
        with mode.ui.state_lock:
            mode._working_message = message
            if mode._active_status_indicator is not None and mode._active_status_indicator.kind == "working":
                mode._active_status_indicator.set_message(
                    message if message is not None else mode._default_working_message
                )

    def set_working_visible(self, visible) -> None:
        self._mode._set_working_visible(visible)

    def set_working_indicator(self, options=None) -> None:
        self._mode._set_working_indicator(options)

    def set_hidden_thinking_label(self, label=None) -> None:
        self._mode._set_hidden_thinking_label(label)

    def set_widget(self, key, content, options=None) -> None:
        self._mode._set_extension_widget(key, content, options)

    def set_footer(self, factory) -> None:
        """Set a custom footer component, or None to restore the built-in footer.

        The factory receives a FooterDataProvider for data not otherwise
        accessible: git branch and extension statuses from set_status().
        Context usage is on `ctx.get_context_usage()`, token stats on
        `ctx.session_manager.get_entries()`, model info on `ctx.model`.
        """
        self._mode._set_extension_footer(factory)

    def set_header(self, factory) -> None:
        self._mode._set_extension_header(factory)

    def set_title(self, title) -> None:
        self._mode.ui.terminal.set_title(title)

    def write_terminal(self, sequence: str) -> None:
        # pidrei-only (pi's extensions write `process.stdout` directly, safe on
        # its one thread): the TUI's output pump is the terminal's only
        # writer, so raw sequences queue behind it, never inside a frame.
        self._mode.ui.terminal.write_sync(sequence)

    def custom(self, factory, options=None):
        return self._mode._show_extension_custom(factory, options)

    def paste_to_editor(self, text: str) -> None:
        # pi's direct `editor.handleInput(paste)`, whatever has focus, as one
        # hold of the lock (the editor is resolved inside it, behind any
        # editor swap). Not through the input stream (spec/ui-island.md).
        data = f"\x1b[200~{text}\x1b[201~"
        with self._mode.ui.state_lock:
            self._mode.editor.handle_input(data)

    def set_editor_text(self, text: str) -> None:
        self._mode._set_editor_text(text)

    def get_editor_text(self) -> str:
        with self._mode.ui.state_lock:
            editor = self._mode.editor
            get_expanded = getattr(editor, "get_expanded_text", None)
            return get_expanded() if get_expanded is not None else editor.get_text()

    def editor(self, title, prefill=None):
        return self._mode._show_extension_editor(title, prefill)

    def add_autocomplete_provider(self, factory) -> None:
        mode = self._mode
        with mode._extension_registry_guard:
            mode._autocomplete_provider_wrappers = (*mode._autocomplete_provider_wrappers, factory)
        mode._setup_autocomplete_provider()

    def set_editor_component(self, factory) -> None:
        self._mode._set_custom_editor_component(factory)

    def get_editor_component(self):
        with self._mode.ui.state_lock:
            return self._mode._editor_component_factory

    @property
    def theme(self):
        return theme

    def get_all_themes(self):
        return get_available_themes_with_paths()

    def get_theme(self, name):
        return get_theme_by_name(name)

    async def set_theme(self, theme_or_name):
        # lazy: core <-> modes import cycle (see modes/__init__.py)
        from .theme import Theme as ThemeClass

        mode = self._mode
        if isinstance(theme_or_name, ThemeClass):
            return mode._theme_controller.set_theme_instance(theme_or_name)
        result = await mode._theme_controller.set_theme_name(theme_or_name)
        if result["success"] and mode.settings_manager.get_theme() != theme_or_name:
            mode.settings_manager.set_theme(theme_or_name)
        return result

    def get_tools_expanded(self) -> bool:
        with self._mode.ui.state_lock:
            return self._mode._tool_output_expanded

    def set_tools_expanded(self, expanded) -> None:
        self._mode.set_tools_expanded(expanded)


class _TerminalInputSubscription:
    """An extension input listener plus its current unsubscribe handle.

    Switching renderers moves every listener to the new one, so the handle has
    to be replaceable while the identity the caller holds stays the same.
    """

    __slots__ = ("handler", "unsubscribe")

    def __init__(self, handler, unsubscribe) -> None:
        self.handler = handler
        self.unsubscribe = unsubscribe


class InteractiveMode:
    """Options: ``{"migratedProviders"?, "startupDiagnostics"?,
    "modelFallbackMessage"?, "autoTrustOnReloadCwd"?, "initialMessage"?,
    "initialImages"?, "initialMessages"?, "verbose"?, "tuiMode"?, "terminal"?}``.

    ``startupDiagnostics`` are the diagnostics collected before the TUI was
    initialized, replayed into the transcript. ``terminal`` is the terminal
    implementation; it defaults to the current process terminal."""

    def __init__(self, runtime_host, options: dict | None = None) -> None:
        options = options or {}
        self.runtime_host = runtime_host
        set_capability_overrides(self.settings_manager.get_terminal_capability_overrides())
        tui_mode = options.get("tuiMode") or self.settings_manager.get_tui_mode()
        self._options = {**options, "tuiMode": tui_mode}
        self._auto_trust_on_reload_cwd = options.get("autoTrustOnReloadCwd")
        self.runtime_host.set_before_session_invalidate(lambda: self._reset_extension_ui())

        async def _rebind_session(new_session, swap) -> None:
            await self._rebind_current_session({"renderBeforeBind": True}, new_session, swap)
            await self._theme_controller.apply_from_settings()

        self.runtime_host.set_rebind_session(_rebind_session)
        self._version = VERSION
        self._renderer = create_interactive_tui(
            tui_mode=tui_mode,
            show_hardware_cursor=self.settings_manager.get_show_hardware_cursor(),
            log_directory=get_agent_dir(),
            terminal=options.get("terminal"),
            fullscreen_copy_on_select=self.settings_manager.get_fullscreen_copy_on_select(),
            fullscreen_wheel_scroll_lines=self.settings_manager.get_fullscreen_wheel_scroll_lines(),
        )
        self._main_screen_render_state = None
        self._fullscreen_layout_root = None
        self.ui = create_interactive_tui_reference(lambda: self._renderer)
        # What extension factories receive as `tui` (spec/ui-island.md).
        self._extension_tui = ExtensionTui(self.ui)
        self.ui.set_clear_on_shrink(self.settings_manager.get_clear_on_shrink())
        # Kitty-protocol terminals accept PNG only: images convert through it, extension images included.
        self.ui.set_image_converter(convert_image_to_png_base64)
        self.ui.set_render_error_handler(self._uncaught_crash)
        self._header_container = Container()
        self._loaded_resources_container = Container()
        self._chat_container = Container()
        # Keep loaded resources before chat so restored session messages never
        # precede them; the whole document is what the alt screen scrolls.
        self._document_container = Container()
        self._document_container.add_child(self._header_container)
        self._document_container.add_child(self._loaded_resources_container)
        self._document_container.add_child(self._chat_container)
        self._transcript_scroll_view = None
        self._pending_messages_container = Container()
        self._status_container = Container()
        self._widget_container_above = Container()
        self._widget_container_below = Container()
        # Not awaited: defaults only for now. The editors and `set_keybindings`
        # need the object here; the user's keybindings.json is read into it in
        # `init()` (`reload()`).
        self._keybindings = KeybindingsManager()
        set_keybindings(self._keybindings)
        editor_padding_x = self.settings_manager.get_editor_padding_x()
        autocomplete_max_visible = self.settings_manager.get_autocomplete_max_visible()
        self._default_editor = CustomEditor(
            self.ui,
            get_editor_theme(),
            self._keybindings,
            {
                "paddingX": editor_padding_x,
                "autocompleteMaxVisible": autocomplete_max_visible,
                "embedWorkingStatus": True,
            },
        )
        self.editor = self._default_editor
        self._editor_component_factory = None
        self._autocomplete_provider = None
        # Extension registrations, written from extension tasks and read or
        # reset from others: the wrappers are a copy-on-write tuple, the
        # terminal-input subscriptions change only under this guard.
        self._extension_registry_guard = threading.Lock()
        self._autocomplete_provider_wrappers: tuple = ()
        self._fd_path: str | None = None
        self._editor_container = Container()
        self._editor_container.add_child(self.editor)
        self._footer_data_provider = FooterDataProvider(self.session_manager.get_cwd())
        self._footer = FooterComponent(self.session, self._footer_data_provider)
        self._footer.set_auto_compact_enabled(self.session.auto_compaction_enabled)
        self._footer_container = Container()
        self._footer_container.add_child(self._footer)

        self._is_initialized = False
        # Submit tasks hand texts to the main loop: a submit takes the
        # installed callback or queues its text, the loop pops a queued text
        # or installs its callback — each under the guard, so no text waits
        # in the queue while the loop waits for one.
        self._on_input_callback = None
        self._pending_user_inputs: list = []
        self._user_input_guard = threading.Lock()
        self._active_status_indicator = None
        self._active_working_indicator_embedded = False
        self._idle_status = IdleStatus()
        self._working_message: str | None = None
        self._working_visible = True
        self._working_indicator_options = None
        self._default_working_message = "Working"
        self._default_hidden_thinking_label = "Thinking..."
        self._hidden_thinking_label = self._default_hidden_thinking_label

        self._last_sigint_time = 0.0
        self._last_escape_time = 0.0
        self._is_shutting_down = False
        # Check-and-set of `_is_shutting_down` for the paths that claim the one
        # shutdown: they run on different tasks (spawned key actions, the
        # signal watcher, /quit, extensions, the crash guard).
        self._shutdown_guard = threading.Lock()
        # Bumped on unregister so stale signal watchers turn inert. tonio's
        # signal receiver cannot be cancelled from outside while blocked, but
        # every unregister call site exits the process right after, so an
        # inert watcher never outlives anything that matters.
        self._signal_watch_generation = 0
        self._changelog_markdown: str | None = None
        self._startup_notices_shown = False
        self._anthropic_subscription_warning_shown = False
        self._anthropic_subscription_warning_guard = threading.Lock()

        # Status line tracking (for mutating immediately-sequential status
        # updates)
        self._last_status_spacer = None
        self._last_status_text = None
        self._last_status_message = ""
        self._managed_tool_status_started = False

        # Streaming message tracking
        self._streaming_component = None
        # Entries a boundary compaction already rendered; their own entry_appended is skipped.
        self._entries_rendered_by_boundary_compaction: set[str] = set()
        self._streaming_message = None

        # Tool execution tracking: tool_call_id -> component
        self._pending_tools: dict = {}

        # Tool output expansion state, under the UI state lock (set from the
        # keybinding and from extension tasks).
        self._tool_output_expanded = False

        # Thinking block visibility state
        self._hide_thinking_block = self.settings_manager.get_hide_thinking_block()
        self._output_pad = self.settings_manager.get_output_pad()

        # Skill commands: command name -> skill file path
        self._skill_commands: dict = {}

        # Agent subscription unsubscribe function
        self._unsubscribe = None
        self._signal_cleanup_handlers: list = []

        # Track if editor is in bash mode (text starts with !)
        self._is_bash_mode = False
        # An editor `!` command is claimed from its submit until its flow
        # ends (under the UI state lock): see `_handle_editor_submit`.
        self._bash_claimed = False

        # (pi's current-bash-component field is a local of
        # `_handle_bash_command` here: bash commands run concurrently.)

        # Track pending bash components (shown in pending area, moved to
        # chat on submit)
        self._pending_bash_components: list = []

        # Active editor-area selector (disposed when replaced or on stop)
        self._active_selector_token: object | None = None
        self._active_selector_dispose = None

        # Auto-compaction / auto-retry state
        self._auto_compaction_escape_handler = None
        self._retry_escape_handler = None

        # Messages queued while compaction is running:
        # {"text", "mode": "steer" | "followUp"} records
        self._compaction_queued_messages: list = []
        # Submit handlers append while flushes (from any task) take it:
        # every read-modify-write of the list takes this.
        self._compaction_queue_guard = threading.Lock()

        # Shutdown state
        self._shutdown_requested = False

        # Extension UI state
        self._extension_selector = None
        self._extension_input = None
        self._extension_editor = None
        self._extension_terminal_input_subscriptions: set = set()

        # Extension widgets (components rendered above/below the editor)
        self._extension_widgets_above: dict = {}
        self._extension_widgets_below: dict = {}

        # Custom footer/header from extension (None = use built-in)
        self._custom_footer = None
        self._built_in_header = None
        self._custom_header = None

        # Register themes from resource loader and initialize
        set_registered_themes(self.session.resource_loader.get_themes()["themes"])
        self._theme_controller = InteractiveThemeController(
            self.ui,
            {
                "getSettingsManager": lambda: self.settings_manager,
                "showError": lambda message: self.show_error(message),
                "onChanged": self._on_theme_changed,
                "initialThemeSetting": options.get("initialThemeSetting"),
            },
        )

    # Convenience accessors
    @property
    def session(self):
        return self.runtime_host.session

    @property
    def agent(self):
        return self.session.agent

    @property
    def session_manager(self):
        return self.session.session_manager

    @property
    def settings_manager(self):
        return self.session.settings_manager

    # =========================================================================
    # Autocomplete
    # =========================================================================

    def _get_autocomplete_source_tag(self, source_info=None) -> str | None:
        # Built-in extension commands are untagged, like built-in commands.
        if source_info is None or source_info.source == "builtin":
            return None

        if source_info.scope == "user":
            scope_prefix = "u"
        elif source_info.scope == "project":
            scope_prefix = "p"
        else:
            scope_prefix = "t"
        source = source_info.source.strip()

        if source in ("auto", "local", "cli"):
            return scope_prefix

        if source.startswith("npm:"):
            return f"{scope_prefix}:{source}"

        git_source = parse_git_url(source)
        if git_source:
            ref = f"@{git_source['ref']}" if git_source["ref"] else ""
            return f"{scope_prefix}:git:{git_source['host']}/{git_source['path']}{ref}"

        return scope_prefix

    def _prefix_autocomplete_description(self, description, source_info=None):
        source_tag = self._get_autocomplete_source_tag(source_info)
        if not source_tag:
            return description
        return f"[{source_tag}] {description}" if description else f"[{source_tag}]"

    def _get_built_in_command_conflict_diagnostics(self, extension_runner) -> list:
        builtin_names = {command.name for command in BUILTIN_SLASH_COMMANDS}
        diagnostics = []
        for command in extension_runner.get_registered_commands():
            if command.name not in builtin_names:
                continue
            if command.invocation_name == command.name:
                message = (
                    f"Extension command '/{command.name}' conflicts with built-in interactive command. "
                    "Skipping in autocomplete."
                )
            else:
                message = (
                    f"Extension command '/{command.name}' conflicts with built-in interactive command. "
                    f"Available as '/{command.invocation_name}'."
                )
            diagnostics.append(
                {"type": "warning", "message": message, "path": getattr(command.source_info, "path", None)}
            )
        return diagnostics

    def _create_base_autocomplete_provider(self):
        # Define commands for autocomplete
        slash_commands = []
        for command in BUILTIN_SLASH_COMMANDS:
            entry = {"name": command.name, "description": command.description}
            if command.argument_hint:
                entry["argumentHint"] = command.argument_hint
            slash_commands.append(entry)

        model_command = next((command for command in slash_commands if command["name"] == "model"), None)
        if model_command is not None:

            async def get_model_completions(prefix: str):
                if self.session.scoped_models:
                    models = [s["model"] if isinstance(s, dict) else s.model for s in self.session.scoped_models]
                else:
                    models = self.session.model_runtime.get_available_snapshot()

                if not models:
                    return None

                # Create items with provider/id format
                items = [
                    {"id": m.id, "provider": m.provider, "name": m.name, "label": f"{m.provider}/{m.id}"}
                    for m in models
                ]

                return _create_fuzzy_autocomplete_items(
                    items,
                    prefix,
                    get_model_search_text,
                    lambda item: {"value": item["label"], "label": item["id"], "description": item["provider"]},
                )

            model_command["getArgumentCompletions"] = get_model_completions

        thinking_command = next((command for command in slash_commands if command["name"] == "thinking"), None)
        if thinking_command is not None:

            async def get_thinking_completions(prefix: str):
                return _create_fuzzy_autocomplete_items(
                    self.session.get_available_thinking_levels(),
                    prefix,
                    lambda level: level,
                    lambda level: {"value": level, "label": level},
                )

            thinking_command["getArgumentCompletions"] = get_thinking_completions

        login_command = next((command for command in slash_commands if command["name"] == "login"), None)
        if login_command is not None:

            async def get_login_completions(prefix: str):
                providers = _get_login_provider_completion_options(self.get_login_provider_options())
                return _create_fuzzy_autocomplete_items(
                    providers,
                    prefix,
                    _get_login_provider_search_text,
                    lambda provider: {
                        "value": provider["id"],
                        "label": provider["id"],
                        "description": _format_login_provider_completion_description(provider),
                    },
                )

            login_command["getArgumentCompletions"] = get_login_completions

        # Convert prompt templates to SlashCommand format for autocomplete
        template_commands = []
        for cmd in self.session.prompt_templates:
            entry = {
                "name": cmd.name,
                "description": self._prefix_autocomplete_description(cmd.description, cmd.source_info),
            }
            if getattr(cmd, "argument_hint", None):
                entry["argumentHint"] = cmd.argument_hint
            template_commands.append(entry)

        # Convert extension commands to SlashCommand format
        builtin_command_names = {c["name"] for c in slash_commands}
        extension_commands = [
            {
                "name": cmd.invocation_name,
                "description": self._prefix_autocomplete_description(cmd.description, cmd.source_info),
                "getArgumentCompletions": getattr(cmd, "get_argument_completions", None),
            }
            for cmd in self.session.extension_runner.get_registered_commands()
            if cmd.name not in builtin_command_names
        ]

        # Build skill commands from session skills (if enabled)
        self._skill_commands.clear()
        skill_command_list = []
        if self.settings_manager.get_enable_skill_commands():
            for skill in self.session.resource_loader.get_skills().skills:
                command_name = f"skill:{skill.name}"
                self._skill_commands[command_name] = skill.file_path
                skill_command_list.append(
                    {
                        "name": command_name,
                        "description": self._prefix_autocomplete_description(skill.description, skill.source_info),
                    }
                )

        return CombinedAutocompleteProvider(
            [*slash_commands, *template_commands, *extension_commands, *skill_command_list],
            self.session_manager.get_cwd(),
            self._fd_path,
        )

    def _setup_autocomplete_provider(self) -> None:
        """Rebuild the autocomplete provider and install it on the editors
        (extension registration, session bind, reload, settings)."""
        with self.ui.state_lock:
            provider = self._create_base_autocomplete_provider()
            trigger_characters: list = []
            for wrap_provider in self._autocomplete_provider_wrappers:
                provider = wrap_provider(provider)
                trigger_characters.extend(getattr(provider, "trigger_characters", None) or [])
            if trigger_characters:
                provider.trigger_characters = list(dict.fromkeys(trigger_characters))

            self._autocomplete_provider = provider
            self._default_editor.set_autocomplete_provider(provider)
            if self.editor is not self._default_editor:
                set_provider = getattr(self.editor, "set_autocomplete_provider", None)
                if set_provider is not None:
                    set_provider(provider)

    # =========================================================================
    # Startup
    # =========================================================================

    def _show_startup_notices_if_needed(self) -> None:
        # The check-and-set is under the lock too: every session bind calls this.
        with self.ui.state_lock:
            if self._startup_notices_shown:
                return
            self._startup_notices_shown = True

            if not self._changelog_markdown:
                return

            if self._chat_container.children:
                self._chat_container.add_child(Spacer(1))
            self._chat_container.add_child(DynamicBorder())
            if self.settings_manager.get_collapse_changelog():
                version_match = re.search(r"##\s+\[?(\d+\.\d+\.\d+(?:\.\d+)?)\]?", self._changelog_markdown)
                latest_version = version_match.group(1) if version_match else self._version
                condensed_text = f"Updated to v{latest_version}. Use {theme.bold('/changelog')} to view full changelog."
                self._chat_container.add_child(Text(condensed_text, 1, 0))
            else:
                self._chat_container.add_child(ThemedText(lambda: theme.bold(theme.fg("accent", "What's New")), 1, 0))
                self._chat_container.add_child(Spacer(1))
                self._chat_container.add_child(
                    Markdown(self._changelog_markdown.strip(), 1, 0, self._get_markdown_theme_with_settings())
                )
                self._chat_container.add_child(Spacer(1))
            self._chat_container.add_child(DynamicBorder())

    def _mount_interactive_tui(self, tui, components) -> None:
        for component in components:
            tui.add_child(component)
        if is_viewport_tui(tui):
            if self._fullscreen_layout_root is None:
                raise RuntimeError("Fullscreen layout is not initialized")
            tui.set_layout_root(self._fullscreen_layout_root)

    async def _stop_interactive_tui(self, fullscreen_exit_output: str) -> None:
        if self._renderer.mode == "fullscreen" and fullscreen_exit_output == "transcript":
            renderer = self._renderer

            # Under the UI state lock, like the switch's own overlay check.
            with self.ui.state_lock:
                while renderer.has_overlay_entries:
                    renderer.hide_overlay()
            await self._switch_tui_mode("regular", restore_progress=False, start_renderer=False)
            # The fullscreen renderer released the terminal when it stopped. The transcript
            # below and the regular screen's stop both write to it, and that stop releases it.
            self._renderer.terminal.arm()
            await self._renderer.render_now()
        await self.ui.stop({"preserveScreen": self._renderer.mode == "fullscreen"})

    async def _switch_tui_mode(self, mode: str, restore_progress: bool = True, start_renderer: bool = True) -> bool:
        """Task code: stops and starts renderers. Input handling runs it as a
        completion (the settings switch)."""
        previous_ui = self._renderer
        if mode == previous_ui.mode:
            return True

        await previous_ui.stop({"preserveScreen": True})

        # The tree is checked and moved to the next renderer in one hold of
        # the UI state lock, which both renderers share (one terminal).
        # Overlays are checked there, not before the stop — one opened in
        # between could not be carried over.
        next_ui = None
        with previous_ui.state_lock:
            if not previous_ui.has_overlay_entries:
                components = list(previous_ui.children)
                focus = previous_ui.get_focused_component()
                if isinstance(previous_ui, TuiMainScreen):
                    self._main_screen_render_state = previous_ui.capture_render_state()

                previous_ui.set_focus(None)
                previous_ui.clear()
                if is_viewport_tui(previous_ui):
                    previous_ui.set_layout_root(None)

                next_ui = create_interactive_tui(
                    tui_mode=mode,
                    show_hardware_cursor=previous_ui.get_show_hardware_cursor(),
                    log_directory=get_agent_dir(),
                    terminal=previous_ui.terminal,
                    fullscreen_copy_on_select=self.settings_manager.get_fullscreen_copy_on_select(),
                    fullscreen_wheel_scroll_lines=self.settings_manager.get_fullscreen_wheel_scroll_lines(),
                )
                next_ui.set_clear_on_shrink(previous_ui.get_clear_on_shrink())
                next_ui.set_image_conversions(previous_ui.image_conversions)
                next_ui.set_render_error_handler(self._uncaught_crash)
                next_ui.on_debug = previous_ui.on_debug
                if isinstance(next_ui, TuiMainScreen) and self._main_screen_render_state is not None:
                    next_ui.restore_render_state(self._main_screen_render_state)
                self._renderer = next_ui
                self._options["tuiMode"] = mode
                self._mount_interactive_tui(next_ui, components)
                next_ui.invalidate()
                next_ui.set_focus(focus)

        if next_ui is None:
            if start_renderer:
                await previous_ui.start()
                previous_ui.request_render(True)
            return False
        if not start_renderer:
            return True
        await next_ui.start()
        await self._theme_controller.rebind_tui()
        self._rebind_extension_terminal_input_listeners()
        if (
            restore_progress
            and self.settings_manager.get_show_terminal_progress()
            and (self.session.is_streaming or self.session.is_compacting)
        ):
            next_ui.terminal.set_progress(True)
        return True

    async def init(self) -> None:
        if self._is_initialized:
            return

        self._register_signal_handlers()

        # Deferred out of __init__: reads the user's keybindings.json.
        await self._keybindings.reload()

        # Deferred out of __init__: git metadata for the footer branch. After
        # this, `get_git_branch()` on the render path is a pure cache read.
        await self._footer_data_provider.prime()

        # Deferred out of __init__: init_theme reads theme files from disk.
        await self._theme_controller.prime()

        # Load changelog (only show new entries, skip for resumed sessions)
        self._changelog_markdown = await self._get_changelog_for_display()

        if self.session.scoped_models and self._should_show_startup_details():
            model_parts = []
            for sm in self.session.scoped_models:
                model = sm["model"] if isinstance(sm, dict) else sm.model
                thinking_level = (
                    sm.get("thinkingLevel") if isinstance(sm, dict) else getattr(sm, "thinking_level", None)
                )
                thinking_str = f":{thinking_level}" if thinking_level else ""
                model_parts.append(f"{model.id}{thinking_str}")
            model_list = ", ".join(model_parts)
            cycle_keys = self._keybindings.get_keys("app.model.cycleForward")
            cycle_hint = (
                theme.fg("muted", f" ({format_key_text('/'.join(cycle_keys), {'capitalize': True})} to cycle)")
                if cycle_keys
                else ""
            )
            write_stdout(theme.fg("dim", f"Model scope: {model_list}{cycle_hint}") + "\n")

        # Keep one component tree and remount it when changing renderers.
        self._render_widgets()  # Initialize with default spacer
        viewport = create_chat_viewport(
            document=self._document_container,
            pending_messages=self._pending_messages_container,
            status=self._status_container,
            widgets_above=self._widget_container_above,
            editor=self._editor_container,
            widgets_below=self._widget_container_below,
            footer=self._footer_container,
            scrollbar=self.settings_manager.get_fullscreen_scrollbar(),
            scrollbar_track_style=lambda text: theme.fg("scrollbarTrack", text),
            scrollbar_thumb_style=lambda text: theme.fg("scrollbarThumb", text),
        )
        self._transcript_scroll_view = viewport.transcript
        self._fullscreen_layout_root = viewport.root
        self._mount_interactive_tui(
            self._renderer,
            [
                self._document_container,
                self._pending_messages_container,
                self._status_container,
                self._widget_container_above,
                self._editor_container,
                self._widget_container_below,
                self._footer_container,
            ],
        )
        self.ui.set_focus(self.editor)

        # Accept text while startup completes, but only enable interrupt,
        # exit, and submission feedback.
        self._setup_startup_input_handlers()

        # Render paths read the capabilities: settle them (under tmux, a
        # subprocess probe) before the first frame.
        await prime_capabilities(probe_tmux_hyperlinks)

        # The terminal is the tty's one writer until `stop()` closes it:
        # stdout/stderr writes go through its queue too.
        await attach_terminal(self.ui.terminal)

        # Start the UI before initializing extensions so session_start
        # handlers can use interactive dialogs
        await self.ui.start()
        self._is_initialized = True

        await self._theme_controller.apply_from_settings()
        # The header and startup notices bake theme colors into their text, so
        # build them once the terminal reported its colors. This ends at the
        # terminal's DA1 reply, or after 100 ms if it answers nothing.
        await self._theme_controller.wait_for_terminal_colors()

        # Add header with keybindings from config (unless silenced)
        if self._should_show_startup_header():
            show_details = self._should_show_startup_details()
            # Built on demand so the header follows theme changes. The logo's
            # first line carries the version, its second line the first line
            # of key hints. Terminals that cannot render the logo get a
            # "PiDrei vX" line instead, with the key hints below it.
            show_logo = supports_pi_logo()

            def with_logo(hints: str) -> str:
                if not show_logo:
                    return f"{pi_wordmark()} {theme.fg('dim', f'v{self._version}')}\n{hints}"
                top, bottom = pi_logo_lines()
                return f"{top} {theme.fg('dim', f'v{self._version}')}\n{bottom} {hints}"

            def expanded_instructions() -> str:
                return "\n".join(
                    [
                        key_hint("app.interrupt", "to interrupt"),
                        key_hint("app.clear", "to clear"),
                        raw_key_hint(f"{key_text('app.clear')} twice", "to exit"),
                        key_hint("app.exit", "to exit (empty)"),
                        key_hint("app.suspend", "to suspend"),
                        key_hint("tui.editor.deleteToLineEnd", "to delete to end"),
                        key_hint("app.thinking.cycle", "to cycle thinking level"),
                        raw_key_hint(
                            f"{key_text('app.model.cycleForward')}/{key_text('app.model.cycleBackward')}",
                            "to cycle models",
                        ),
                        key_hint("app.model.select", "to select model"),
                        key_hint("app.tools.expand", "to expand tools"),
                        key_hint("app.thinking.toggle", "to expand thinking"),
                        key_hint("app.editor.external", "for external editor"),
                        raw_key_hint("/", "for commands"),
                        raw_key_hint("!", "to run bash"),
                        raw_key_hint("!!", "to run bash (no context)"),
                        key_hint("app.message.followUp", "to queue follow-up"),
                        key_hint("app.message.dequeue", "to edit all queued messages"),
                        key_hint("app.clipboard.pasteImage", "to paste files on macOS, images, or text"),
                        raw_key_hint("drop files", "to attach"),
                    ]
                )

            def compact_instructions() -> str:
                return theme.fg("muted", " · ").join(
                    [
                        key_hint("app.interrupt", "interrupt"),
                        raw_key_hint(f"{key_text('app.clear')}/{key_text('app.exit')}", "clear/exit"),
                        raw_key_hint("/", "commands"),
                        raw_key_hint("!", "bash"),
                        key_hint("app.tools.expand", "more"),
                    ]
                )

            def compact_onboarding() -> str:
                return theme.fg(
                    "dim",
                    f"Press {key_text('app.tools.expand')} to show full startup help"
                    f"{' and loaded resources' if show_details else ''}.",
                )

            def onboarding() -> str:
                return theme.fg(
                    "dim",
                    "PiDrei can explain its own features and look up its docs. Ask it how to use or extend PiDrei.",
                )

            self._built_in_header = ExpandableText(
                lambda: f"{with_logo(compact_instructions())}\n{compact_onboarding()}\n\n{onboarding()}",
                lambda: f"{with_logo(expanded_instructions())}\n\n{onboarding()}",
                self._get_startup_expansion_state(),
                1,
                0,
            )
            # Setup UI layout
            header_children = [Spacer(1), self._built_in_header, Spacer(1)]
        else:
            # Minimal header when silenced
            self._built_in_header = Text("", 0, 0)
            header_children = [self._built_in_header]

        # The UI is started: the mount is one hold of the UI state lock.
        with self.ui.state_lock:
            for child in header_children:
                self._header_container.add_child(child)
            self.ui.request_render()

        # Resolve fd and rg after mounting the TUI (pi also downloads them
        # here; pidrei only looks them up — see utils/tools_manager.py — so
        # `on_status` never fires but the staged-startup shape is pi's).
        # Both are needed: fd for autocomplete, rg for the grep tool and bash
        # commands.
        async def _ensure(tool: str) -> str | None:
            return await ensure_tool(tool, self._show_managed_tool_status)

        fd_path, _ = await tonio.spawn(_ensure("fd"), _ensure("rg"))
        self._fd_path = fd_path

        # Enable the remaining input handlers only after managed-tool setup
        # completes — in one hold, as startup input is dispatched through
        # the handlers being replaced.
        with self.ui.state_lock:
            self._setup_key_handlers()
            self._setup_editor_submit_handler()
            self.ui.request_render()

        # Initialize extensions first so resources are shown before messages
        await self._rebind_current_session()

        # Render initial messages AFTER showing loaded resources
        self._render_initial_messages(await self._needs_project_trust_warning())

        # Set up theme file watcher. Every theme change (the file reload's
        # task, `set_theme`, an in-memory theme set through the extension
        # API) arrives on the changing task: the swap and the refresh are one
        # hold, so a frame never mixes two themes (§4.5c).
        def handle_theme_change(apply_theme) -> None:
            with self.ui.state_lock:
                if apply_theme():
                    self.ui.invalidate()
                    self._update_editor_border_color()
                    self.ui.request_render()

        on_theme_change(handle_theme_change)

        # Set up git branch watcher (uses provider instead of footer)
        self._footer_data_provider.on_branch_change(lambda: self.ui.request_render())

        # Initialize available provider count for footer display
        self._update_available_provider_count()

    def _update_terminal_title(self) -> None:
        """Update terminal title with session name and cwd."""
        cwd_basename = os.path.basename(self.session_manager.get_cwd())
        session_name = self.session_manager.get_session_name()
        if session_name:
            self.ui.terminal.set_title(f"{APP_TITLE} - {session_name} - {cwd_basename}")
        else:
            self.ui.terminal.set_title(f"{APP_TITLE} - {cwd_basename}")

    async def run(self) -> None:
        """Run the interactive mode. This is the main entry point.

        Initializes the UI, shows warnings, processes initial messages, and
        starts the interactive loop.
        """
        await self.init()

        if not os.environ.get("PIDREI_OFFLINE"):

            async def refresh_models() -> None:
                with contextlib.suppress(Exception):
                    await refresh_model_catalogs(self.session.model_runtime, _timeout_cancel(15_000))
                    self._update_available_provider_count()

            self._spawn_flow(refresh_models())

        # Start version check asynchronously
        async def version_check() -> None:
            new_release = await check_for_new_version(self._version)
            if new_release:
                self.show_new_version_notification(new_release)

        self._spawn_flow(version_check())

        # Start package update check asynchronously
        async def package_update_check() -> None:
            updates = await self._check_for_package_updates()
            if updates:
                self.show_package_update_notification(updates)

        self._spawn_flow(package_update_check())

        # Check tmux keyboard setup asynchronously
        async def tmux_check() -> None:
            warning = await self._check_tmux_keyboard_setup()
            if warning:
                self.show_warning(warning)

        self._spawn_flow(tmux_check())

        # Show startup warnings
        migrated_providers = self._options.get("migratedProviders")
        startup_diagnostics = self._options.get("startupDiagnostics")
        model_fallback_message = self._options.get("modelFallbackMessage")
        initial_message = self._options.get("initialMessage")
        initial_images = self._options.get("initialImages")
        initial_messages = self._options.get("initialMessages")

        for diagnostic in startup_diagnostics or []:
            if diagnostic.type == "error":
                self.show_error(diagnostic.message)
            elif diagnostic.type == "warning":
                self.show_warning(diagnostic.message)
            else:
                self.show_status(diagnostic.message)

        if migrated_providers:
            self.show_warning(f"Migrated credentials to auth.json: {', '.join(migrated_providers)}")

        models_json_error = self.session.model_runtime.get_error()
        if models_json_error:
            self.show_error(f"models.json error: {models_json_error}")

        if model_fallback_message:
            self.show_warning(model_fallback_message)

        self._spawn_flow(self._maybe_warn_about_anthropic_subscription_auth())

        # Process initial messages
        if initial_message:
            try:
                await self.session.prompt(initial_message, PromptOptions(images=initial_images))
            except Exception as error:
                self.show_error(str(error) or "Unknown error occurred")

        if initial_messages:
            for message in initial_messages:
                try:
                    await self.session.prompt(message)
                except Exception as error:
                    self.show_error(str(error) or "Unknown error occurred")

        # Main interactive loop
        while True:
            user_input = await self._get_user_input()
            try:
                await self.session.prompt(user_input)
            except Exception as error:
                self.show_error(str(error) or "Unknown error occurred")

    async def _check_for_package_updates(self) -> list:
        if os.environ.get("PIDREI_OFFLINE"):
            return []

        try:
            package_manager = DefaultPackageManager(
                cwd=self.session_manager.get_cwd(),
                agent_dir=get_agent_dir(),
                settings_manager=self.settings_manager,
            )
            updates = await package_manager.check_for_available_updates()
            return [update.display_name for update in updates]
        except Exception:
            return []

    async def _check_tmux_keyboard_setup(self) -> str | None:
        if not os.environ.get("TMUX"):
            return None

        async def run_tmux_show(option: str) -> str | None:
            try:
                result = await run_command(
                    ["tmux", "show", "-gv", option],  # PATH lookup, like pi's spawn
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                )
            except OSError, subprocess.TimeoutExpired:
                return None
            if result.returncode != 0:
                return None
            return result.stdout.decode("utf-8", "replace").strip()

        # Both options at once, under one 2s budget each, as pi does with
        # `Promise.all` — sequential awaits doubled the worst case.
        extended_keys, extended_keys_format = await tonio.spawn(
            run_tmux_show("extended-keys"),
            run_tmux_show("extended-keys-format"),
        )

        # If we couldn't query tmux (timeout, sandbox, etc.), don't warn
        if extended_keys is None:
            return None

        if extended_keys not in ("on", "always"):
            return (
                "tmux extended-keys is off. Modified Enter keys may not work. "
                "Add `set -g extended-keys on` to ~/.tmux.conf and restart tmux."
            )

        if extended_keys_format == "xterm":
            return (
                "tmux extended-keys-format is xterm. PiDrei works best with csi-u. "
                "Add `set -g extended-keys-format csi-u` to ~/.tmux.conf and restart tmux."
            )

        return None

    async def _get_changelog_for_display(self) -> str | None:
        """Get changelog entries to display on startup.

        Only shows new entries since the last seen version; skips resumed
        sessions.
        """
        # Skip changelog for resumed/continued sessions (already have messages)
        if self.session.state.messages:
            return None

        last_version = self.settings_manager.get_last_changelog_version()
        changelog_path = get_changelog_path()
        entries = await parse_changelog(changelog_path)

        if not last_version:
            # Fresh install - record the version, don't show changelog
            self.settings_manager.set_last_changelog_version(VERSION)
            return None

        new_entries = get_new_entries(entries, last_version)
        if new_entries:
            self.settings_manager.set_last_changelog_version(VERSION)
            return "\n\n".join(normalize_changelog_links(e["content"], e) for e in new_entries)

        return None

    def _get_markdown_theme_with_settings(self) -> dict:
        return {
            **get_markdown_theme(),
            "codeBlockIndent": self.settings_manager.get_code_block_indent(),
        }

    # =========================================================================
    # Resource display helpers
    # =========================================================================

    def _format_display_path(self, p: str) -> str:
        home = os.path.expanduser("~")
        result = p

        # Replace home directory with ~
        if result.startswith(home):
            result = f"~{result[len(home) :]}"

        return result

    def _format_extension_display_path(self, path: str) -> str:
        result = self._format_display_path(path)
        result = re.sub(r"/index\.ts$", "", result)
        result = re.sub(r"/index\.js$", "", result)
        return result

    def _format_context_path(self, p: str) -> str:
        cwd = os.path.abspath(self.session_manager.get_cwd())
        absolute_path = os.path.abspath(p) if os.path.isabs(p) else os.path.abspath(os.path.join(cwd, p))
        relative_path = get_cwd_relative_path(absolute_path, cwd)
        if relative_path is not None:
            return relative_path

        return self._format_display_path(absolute_path)

    def _get_startup_expansion_state(self) -> bool:
        return bool(self._options.get("verbose")) or self._tool_output_expanded

    def _should_show_startup_header(self) -> bool:
        """Startup header (logo, version, key hints). Hidden only by quietStartup: true."""
        return self._options.get("verbose") is True or self.settings_manager.get_quiet_startup() is not True

    def _should_show_startup_details(self) -> bool:
        """Startup details (model scope, loaded resources). Hidden by quietStartup: true or "header"."""
        return self._options.get("verbose") is True or self.settings_manager.get_quiet_startup() is False

    def _get_short_path(self, full_path: str, source_info=None) -> str:
        """Get a short path relative to the package root for display."""
        normalized_full_path = full_path.replace("\\", "/")
        base_dir = getattr(source_info, "base_dir", None) if source_info is not None else None
        if base_dir and self._is_package_source(source_info):
            normalized_base_dir = base_dir.replace("\\", "/")
            npm_root_match = re.match(r"^(.*/node_modules)/(@?[^/]+(?:/[^/]+)?)$", normalized_base_dir)
            # If full_path is under the same node_modules root as base_dir,
            # preserve that relative topology.
            if npm_root_match and normalized_full_path.startswith(f"{npm_root_match.group(1)}/"):
                return posixpath.relpath(normalized_full_path, normalized_base_dir)

            relative_path = os.path.relpath(os.path.abspath(full_path), os.path.abspath(base_dir))
            if (
                relative_path
                and relative_path != "."
                and not relative_path.startswith("..")
                and not os.path.isabs(relative_path)
            ):
                return relative_path.replace("\\", "/")

        source = getattr(source_info, "source", None) or "" if source_info is not None else ""
        npm_match = re.search(r"node_modules/(@?[^/]+(?:/[^/]+)?)/(.*)", normalized_full_path)
        if npm_match and source.startswith("npm:"):
            return npm_match.group(2)

        git_match = re.search(r"git/[^/]+/[^/]+/(.*)", normalized_full_path)
        if git_match and source.startswith("git:"):
            return git_match.group(1)

        return self._format_display_path(full_path)

    def _get_compact_path_label(self, resource_path: str, source_info=None) -> str:
        short_path = self._get_short_path(resource_path, source_info)
        normalized_path = short_path.replace("\\", "/")
        segments = [segment for segment in normalized_path.split("/") if segment and segment != "~"]
        if segments:
            return segments[-1]
        return short_path

    def _get_compact_package_source_label(self, source_info=None) -> str:
        source = getattr(source_info, "source", None) or "" if source_info is not None else ""
        if source.startswith("npm:"):
            return source[len("npm:") :] or source

        git_source = parse_git_url(source)
        if git_source:
            return git_source["path"] or source

        return source

    def _get_compact_extension_label(self, resource_path: str, source_info=None) -> str:

        if not self._is_package_source(source_info):
            return self._get_compact_path_label(resource_path, source_info)

        source_label = self._get_compact_package_source_label(source_info)
        if not source_label:
            return self._get_compact_path_label(resource_path, source_info)

        short_path = self._get_short_path(resource_path, source_info).replace("\\", "/")
        package_path = short_path.removeprefix("extensions/")
        parsed_dir, parsed_base = posixpath.split(package_path)
        parsed_name = posixpath.splitext(parsed_base)[0]

        if parsed_name == "index":
            return source_label if not parsed_dir or parsed_dir == "." else f"{source_label}:{parsed_dir}"

        return f"{source_label}:{package_path}"

    def _get_compact_display_path_segments(self, resource_path: str) -> list:
        return [
            segment
            for segment in self._format_display_path(resource_path).replace("\\", "/").split("/")
            if segment and segment != "~"
        ]

    def _get_compact_non_package_extension_label(self, resource_path: str, index: int, all_paths: list) -> str:
        segments = all_paths[index]["segments"] if 0 <= index < len(all_paths) else None
        if not segments:
            return self._get_compact_path_label(resource_path)

        for segment_count in range(1, len(segments) + 1):
            candidate = "/".join(segments[-segment_count:])
            is_unique = all(
                item_index == index or "/".join(item["segments"][-segment_count:]) != candidate
                for item_index, item in enumerate(all_paths)
            )

            if is_unique:
                return candidate

        return "/".join(segments)

    def _get_compact_extension_labels(self, extensions: list) -> list:
        non_package_extensions = []
        for extension in extensions:
            if self._is_package_source(extension.get("sourceInfo")):
                continue
            segments = self._get_compact_display_path_segments(extension["path"])
            if len(segments) > 1 and segments[-1] in ("index.ts", "index.js"):
                segments.pop()
            non_package_extensions.append(
                {"path": extension["path"], "sourceInfo": extension.get("sourceInfo"), "segments": segments}
            )

        labels = []
        for extension in extensions:
            if self._is_package_source(extension.get("sourceInfo")):
                labels.append(self._get_compact_extension_label(extension["path"], extension.get("sourceInfo")))
                continue

            non_package_index = next(
                (i for i, item in enumerate(non_package_extensions) if item["path"] == extension["path"]), -1
            )
            if non_package_index == -1:
                labels.append(self._get_compact_path_label(extension["path"], extension.get("sourceInfo")))
                continue

            labels.append(
                self._get_compact_non_package_extension_label(
                    extension["path"], non_package_index, non_package_extensions
                )
            )
        return labels

    def _get_display_source_info(self, source_info=None) -> dict:
        source = (getattr(source_info, "source", None) or "local") if source_info is not None else "local"
        scope = (getattr(source_info, "scope", None) or "project") if source_info is not None else "project"
        if source == "local":
            if scope == "user":
                return {"label": "user", "scopeLabel": None, "color": "muted"}
            if scope == "project":
                return {"label": "project", "scopeLabel": None, "color": "muted"}
            if scope == "temporary":
                return {"label": "path", "scopeLabel": "temp", "color": "muted"}
            return {"label": "path", "scopeLabel": None, "color": "muted"}

        if source == "cli":
            return {"label": "path", "scopeLabel": "temp" if scope == "temporary" else None, "color": "muted"}

        if scope == "user":
            scope_label = "user"
        elif scope == "project":
            scope_label = "project"
        elif scope == "temporary":
            scope_label = "temp"
        else:
            scope_label = None
        return {"label": source, "scopeLabel": scope_label, "color": "accent"}

    def _get_scope_group(self, source_info=None) -> str:
        source = (getattr(source_info, "source", None) or "local") if source_info is not None else "local"
        scope = (getattr(source_info, "scope", None) or "project") if source_info is not None else "project"
        if source == "cli" or scope == "temporary":
            return "path"
        if scope == "user":
            return "user"
        if scope == "project":
            return "project"
        return "path"

    def _is_package_source(self, source_info=None) -> bool:
        source = (getattr(source_info, "source", None) or "") if source_info is not None else ""
        return source.startswith(("npm:", "git:"))

    def _build_scope_groups(self, items: list) -> list:
        groups = {
            "user": {"scope": "user", "paths": [], "packages": {}},
            "project": {"scope": "project", "paths": [], "packages": {}},
            "path": {"scope": "path", "paths": [], "packages": {}},
        }

        for item in items:
            group_key = self._get_scope_group(item.get("sourceInfo"))
            group = groups[group_key]
            source_info = item.get("sourceInfo")
            source = (getattr(source_info, "source", None) or "local") if source_info is not None else "local"

            if self._is_package_source(source_info):
                group["packages"].setdefault(source, []).append(item)
            else:
                group["paths"].append(item)

        return [
            group
            for group in (groups["project"], groups["user"], groups["path"])
            if group["paths"] or group["packages"]
        ]

    def _format_scope_groups(self, groups: list, options: dict) -> str:
        lines: list = []

        for group in groups:
            lines.append(f"  {theme.fg('accent', group['scope'])}")

            sorted_paths = sorted(group["paths"], key=lambda item: (item["path"].lower(), item["path"]))
            for item in sorted_paths:
                lines.append(theme.fg("dim", f"    {options['formatPath'](item)}"))

            sorted_packages = sorted(group["packages"].items(), key=lambda kv: (kv[0].lower(), kv[0]))
            for source, package_items in sorted_packages:
                lines.append(f"    {theme.fg('mdLink', source)}")
                sorted_package_paths = sorted(package_items, key=lambda item: (item["path"].lower(), item["path"]))
                for item in sorted_package_paths:
                    lines.append(theme.fg("dim", f"      {options['formatPackagePath'](item, source)}"))

        return "\n".join(lines)

    def _find_source_info_for_path(self, p: str, source_infos: dict):
        exact = source_infos.get(p)
        if exact is not None:
            return exact

        current = p
        while "/" in current:
            current = current[: current.rfind("/")]
            parent = source_infos.get(current)
            if parent is not None:
                return parent

        return None

    def _format_path_with_source(self, p: str, source_info=None) -> str:
        if source_info is not None:
            short_path = self._get_short_path(p, source_info)
            display = self._get_display_source_info(source_info)
            label = display["label"]
            scope_label = display["scopeLabel"]
            label_text = f"{label} ({scope_label})" if scope_label else label
            return f"{label_text} {short_path}"
        return self._format_display_path(p)

    def _format_diagnostics(self, diagnostics: list, source_infos: dict) -> str:
        lines: list = []

        # Group collision diagnostics by name
        collisions: dict = {}
        other_diagnostics: list = []

        for d in diagnostics:
            d_type = d.get("type") if isinstance(d, dict) else d.type
            d_collision = d.get("collision") if isinstance(d, dict) else getattr(d, "collision", None)
            if d_type == "collision" and d_collision is not None:
                name = d_collision.get("name") if isinstance(d_collision, dict) else d_collision.name
                collisions.setdefault(name, []).append(d)
            else:
                other_diagnostics.append(d)

        # Format collision diagnostics grouped by name
        for name, collision_list in collisions.items():
            first_entry = collision_list[0]
            first = (
                first_entry.get("collision")
                if isinstance(first_entry, dict)
                else getattr(first_entry, "collision", None)
            )
            if first is None:
                continue
            winner_path = first.get("winnerPath") if isinstance(first, dict) else first.winner_path
            lines.append(theme.fg("warning", f'  "{name}" collision:'))
            lines.append(
                theme.fg(
                    "dim",
                    f"    {theme.fg('success', '✓')} "
                    f"{self._format_path_with_source(winner_path, self._find_source_info_for_path(winner_path, source_infos))}",
                )
            )
            for d in collision_list:
                d_collision = d.get("collision") if isinstance(d, dict) else getattr(d, "collision", None)
                if d_collision is not None:
                    loser_path = (
                        d_collision.get("loserPath") if isinstance(d_collision, dict) else d_collision.loser_path
                    )
                    lines.append(
                        theme.fg(
                            "dim",
                            f"    {theme.fg('warning', '✗')} "
                            f"{self._format_path_with_source(loser_path, self._find_source_info_for_path(loser_path, source_infos))} (skipped)",
                        )
                    )

        for d in other_diagnostics:
            d_type = d.get("type") if isinstance(d, dict) else d.type
            d_path = d.get("path") if isinstance(d, dict) else getattr(d, "path", None)
            d_message = d.get("message") if isinstance(d, dict) else d.message
            color = "error" if d_type == "error" else "warning"
            if d_path:
                formatted_path = self._format_path_with_source(
                    d_path, self._find_source_info_for_path(d_path, source_infos)
                )
                lines.append(theme.fg(color, f"  {formatted_path}"))
                lines.append(theme.fg(color, f"    {d_message}"))
            else:
                lines.append(theme.fg(color, f"  {d_message}"))

        return "\n".join(lines)

    def _show_loaded_resources(self, options: dict | None = None) -> None:
        """From session bind and /reload."""
        with self.ui.state_lock:
            options = options or {}
            # Resource rendering is idempotent; chat clears no longer clear this
            # separate container.
            self._loaded_resources_container.clear()

            show_listing = options.get("force") or self._should_show_startup_details()
            show_diagnostics = show_listing or options.get("showDiagnosticsWhenQuiet") is True
            if not show_listing and not show_diagnostics:
                return

            def section_header(name: str, color: str = "mdHeading") -> str:
                return theme.fg(color, f"[{name}]")

            def format_compact_list(items: list, list_options: dict | None = None) -> str:
                labels = [item.strip() for item in items if item.strip()]
                if (list_options or {}).get("sort") is not False:
                    labels.sort(key=lambda label: (label.lower(), label))
                return theme.fg("dim", f"  {', '.join(labels)}")

            # Bodies are built on demand so the listing follows theme changes.
            def add_loaded_section(name: str, collapsed_body, expanded_body=None, color: str = "mdHeading") -> None:
                expanded = expanded_body if expanded_body is not None else collapsed_body
                section = ExpandableText(
                    lambda name=name, body=collapsed_body, color=color: f"{section_header(name, color)}\n{body()}",
                    lambda name=name, body=expanded, color=color: f"{section_header(name, color)}\n{body()}",
                    self._get_startup_expansion_state(),
                    0,
                    0,
                )
                self._loaded_resources_container.add_child(section)
                self._loaded_resources_container.add_child(Spacer(1))

            skills_result = self.session.resource_loader.get_skills()
            prompts_result = self.session.resource_loader.get_prompts()
            themes_result = self.session.resource_loader.get_themes()
            if options.get("extensions") is not None:
                extensions = options["extensions"]
            else:
                extensions = [
                    {"path": extension.path, "sourceInfo": extension.source_info}
                    for extension in self.session.resource_loader.get_extensions().extensions
                    if not getattr(extension, "hidden", False)
                ]
            source_infos: dict = {}
            for extension in extensions:
                if extension.get("sourceInfo") is not None:
                    source_infos[extension["path"]] = extension["sourceInfo"]
            for skill in skills_result.skills:
                if skill.source_info is not None:
                    source_infos[skill.file_path] = skill.source_info
            for prompt in prompts_result.prompts:
                if prompt.source_info is not None:
                    source_infos[prompt.file_path] = prompt.source_info
            for loaded_theme in themes_result["themes"]:
                if loaded_theme.source_path and loaded_theme.source_info is not None:
                    source_infos[loaded_theme.source_path] = loaded_theme.source_info

            if show_listing:
                system_prompt_source = self.session.resource_loader.get_system_prompt_source()
                context_files = [
                    *([system_prompt_source] if system_prompt_source is not None else []),
                    *self.session.resource_loader.get_append_system_prompt_sources(),
                    *self.session.resource_loader.get_agents_files(),
                ]
                if context_files:
                    self._loaded_resources_container.add_child(Spacer(1))

                    def context_list() -> str:
                        return "\n".join(
                            theme.fg("dim", f"  {self._format_display_path(f.path)}") for f in context_files
                        )

                    def context_compact_list() -> str:
                        return format_compact_list(
                            [self._format_context_path(context_file.path) for context_file in context_files],
                            {"sort": False},
                        )

                    add_loaded_section("Context", context_compact_list, context_list)

                skills = skills_result.skills
                if skills:
                    groups = self._build_scope_groups(
                        [{"path": skill.file_path, "sourceInfo": skill.source_info} for skill in skills]
                    )

                    def skill_list(groups=groups) -> str:
                        return self._format_scope_groups(
                            groups,
                            {
                                "formatPath": lambda item: self._format_display_path(item["path"]),
                                "formatPackagePath": lambda item, source: self._get_short_path(
                                    item["path"], item.get("sourceInfo")
                                ),
                            },
                        )

                    def skill_compact_list() -> str:
                        return format_compact_list([skill.name for skill in skills])

                    add_loaded_section("Skills", skill_compact_list, skill_list)

                templates = self.session.prompt_templates
                if templates:
                    groups = self._build_scope_groups(
                        [{"path": template.file_path, "sourceInfo": template.source_info} for template in templates]
                    )
                    template_by_path = {t.file_path: t for t in templates}

                    def format_template(item, _source=None):
                        template = template_by_path.get(item["path"])
                        return f"/{template.name}" if template else self._format_display_path(item["path"])

                    def template_list(groups=groups) -> str:
                        return self._format_scope_groups(
                            groups,
                            {
                                "formatPath": format_template,
                                "formatPackagePath": format_template,
                            },
                        )

                    def prompt_compact_list() -> str:
                        return format_compact_list([f"/{template.name}" for template in templates])

                    add_loaded_section("Prompts", prompt_compact_list, template_list)

                if extensions:
                    groups = self._build_scope_groups(extensions)

                    def ext_list(groups=groups) -> str:
                        return self._format_scope_groups(
                            groups,
                            {
                                "formatPath": lambda item: self._format_extension_display_path(item["path"]),
                                "formatPackagePath": lambda item, source: self._format_extension_display_path(
                                    self._get_short_path(item["path"], item.get("sourceInfo"))
                                ),
                            },
                        )

                    extension_labels = self._get_compact_extension_labels(extensions)

                    def extension_compact_list() -> str:
                        return format_compact_list(extension_labels)

                    add_loaded_section("Extensions", extension_compact_list, ext_list, "mdHeading")

            if show_diagnostics:
                skill_diagnostics = skills_result.diagnostics
                if skill_diagnostics:
                    self._loaded_resources_container.add_child(
                        ThemedText(
                            lambda: (
                                f"{theme.fg('warning', '[Skill conflicts]')}\n"
                                f"{self._format_diagnostics(skill_diagnostics, source_infos)}"
                            ),
                            0,
                            0,
                        )
                    )
                    self._loaded_resources_container.add_child(Spacer(1))

                prompt_diagnostics = prompts_result.diagnostics
                if prompt_diagnostics:
                    self._loaded_resources_container.add_child(
                        ThemedText(
                            lambda: (
                                f"{theme.fg('warning', '[Prompt conflicts]')}\n"
                                f"{self._format_diagnostics(prompt_diagnostics, source_infos)}"
                            ),
                            0,
                            0,
                        )
                    )
                    self._loaded_resources_container.add_child(Spacer(1))

                extension_diagnostics: list = []
                extensions_result = self.session.resource_loader.get_extensions()
                for error in extensions_result.errors:
                    extension_diagnostics.append({"type": "error", "message": error.error, "path": error.path})
                for warning in extensions_result.warnings:
                    extension_diagnostics.append({"type": "warning", "message": warning.warning, "path": warning.path})

                runner = self.session.extension_runner
                get_command_diagnostics = getattr(runner, "get_command_diagnostics", None)
                if get_command_diagnostics is not None:
                    extension_diagnostics.extend(get_command_diagnostics())
                extension_diagnostics.extend(self._get_built_in_command_conflict_diagnostics(runner))

                get_shortcut_diagnostics = getattr(runner, "get_shortcut_diagnostics", None)
                if get_shortcut_diagnostics is not None:
                    extension_diagnostics.extend(get_shortcut_diagnostics())

                if extension_diagnostics:
                    self._loaded_resources_container.add_child(
                        ThemedText(
                            lambda: (
                                f"{theme.fg('warning', '[Extension issues]')}\n"
                                f"{self._format_diagnostics(extension_diagnostics, source_infos)}"
                            ),
                            0,
                            0,
                        )
                    )
                    self._loaded_resources_container.add_child(Spacer(1))

                theme_diagnostics = themes_result["diagnostics"]
                if theme_diagnostics:
                    self._loaded_resources_container.add_child(
                        ThemedText(
                            lambda: (
                                f"{theme.fg('warning', '[Theme conflicts]')}\n"
                                f"{self._format_diagnostics(theme_diagnostics, source_infos)}"
                            ),
                            0,
                            0,
                        )
                    )
                    self._loaded_resources_container.add_child(Spacer(1))

    async def _bind_current_session_extensions(self) -> None:
        """Initialize the extension system with TUI-based UI context."""
        # lazy: core <-> modes import cycle (see modes/__init__.py)
        from ...core.agent_session import ExtensionBindings

        ui_context = self._create_extension_ui_context()

        async def new_session_action(options=None):
            self._clear_status_indicator()
            try:
                return await self.runtime_host.new_session(**(options or {}))
            except Exception as error:
                return await self._handle_fatal_runtime_error("Failed to create session", error)

        async def fork_action(entry_id, options=None):
            try:
                result = await self.runtime_host.fork(entry_id, **(options or {}))
                if not result["cancelled"]:
                    self._set_editor_text(result.get("selectedText") or "")
                    self.show_status("Forked to new session")
                return {"cancelled": result["cancelled"]}
            except Exception as error:
                return await self._handle_fatal_runtime_error("Failed to fork session", error)

        async def navigate_tree_action(target_id, options=None):
            options = options or {}
            result = await self.session.navigate_tree(
                target_id,
                {
                    "summarize": options.get("summarize"),
                    "custom_instructions": options.get("customInstructions"),
                    "replace_instructions": options.get("replaceInstructions"),
                    "label": options.get("label"),
                },
            )
            if result.cancelled:
                return {"cancelled": True}

            self._rerender_initial_messages(await self._needs_project_trust_warning())
            if result.editor_text:
                self._fill_empty_editor(result.editor_text)
            self.show_status("Navigated to selected point")
            self._flush_compaction_queue({"willRetry": False})
            return {"cancelled": False}

        async def switch_session_action(session_path, options=None):
            return await self._handle_resume_session(session_path, options)

        async def reload_action():
            await self._handle_reload_command()

        def shutdown_handler() -> None:
            self._request_shutdown()
            if self.session.is_idle:
                tonio.spawn.without_tracking(self.shutdown())

        await self.session.bind_extensions(
            ExtensionBindings(
                ui_context=ui_context,
                mode="tui",
                abort_handler=lambda: self._restore_queued_messages_to_editor({"abort": True}),
                command_context_actions={
                    "wait_for_idle": lambda: self.session.wait_for_idle(),
                    "new_session": new_session_action,
                    "fork": fork_action,
                    "navigate_tree": navigate_tree_action,
                    "switch_session": switch_session_action,
                    "reload": reload_action,
                },
                shutdown_handler=shutdown_handler,
                on_error=lambda error: self._show_extension_error(
                    error.extension_path, error.error, getattr(error, "stack", None)
                ),
            )
        )

        with self.ui.state_lock:
            set_registered_themes(self.session.resource_loader.get_themes()["themes"])
            self._setup_autocomplete_provider()

            extension_runner = self.session.extension_runner
            self._setup_extension_shortcuts(extension_runner)
            self._show_loaded_resources({"force": False, "showDiagnosticsWhenQuiet": True})
            self._show_startup_notices_if_needed()

    def _apply_fullscreen_scrollbar_setting(self) -> None:
        """Apply the fullscreen scrollbar setting (under the UI state lock)."""
        if self._transcript_scroll_view is not None:
            self._transcript_scroll_view.set_scrollbar(self.settings_manager.get_fullscreen_scrollbar())

    def _apply_runtime_settings(self, resolved_cwd: dict) -> bool:
        """pi's applyRuntimeSettings, one synchronous block under the UI state
        lock. Its cwd I/O is prefetched (`footer_data_provider.resolve_cwd`,
        §4.5b). Returns whether the cwd changed; the caller then starts its
        watcher (`watch_cwd`) after the hold."""
        with self.ui.state_lock:
            set_capability_overrides(self.settings_manager.get_terminal_capability_overrides())
            # pi configures the undici HTTP dispatcher here; pidrei's HTTP
            # transport is punkreq's concern (see core/http_config.py).
            self._apply_fullscreen_scrollbar_setting()
            if isinstance(self._renderer, TuiAltScreen):
                self._renderer.set_copy_on_select(self.settings_manager.get_fullscreen_copy_on_select())
                self._renderer.set_wheel_scroll_lines(self.settings_manager.get_fullscreen_wheel_scroll_lines())
            self._footer.set_session(self.session)
            self._footer.set_auto_compact_enabled(self.session.auto_compaction_enabled)
            cwd_changed = self._footer_data_provider.apply_cwd(resolved_cwd)
            self._hide_thinking_block = self.settings_manager.get_hide_thinking_block()
            self._output_pad = self.settings_manager.get_output_pad()
            self.ui.set_show_hardware_cursor(self.settings_manager.get_show_hardware_cursor())
            clear_on_shrink = self.settings_manager.get_clear_on_shrink()
            self.ui.set_clear_on_shrink(clear_on_shrink)
            if not clear_on_shrink and self._active_status_indicator is None:
                self._status_container.clear()
            editor_padding_x = self.settings_manager.get_editor_padding_x()
            autocomplete_max_visible = self.settings_manager.get_autocomplete_max_visible()
            self._default_editor.set_padding_x(editor_padding_x)
            self._default_editor.set_autocomplete_max_visible(autocomplete_max_visible)
            if self.editor is not self._default_editor:
                set_padding = getattr(self.editor, "set_padding_x", None)
                if set_padding is not None:
                    set_padding(editor_padding_x)
                set_max_visible = getattr(self.editor, "set_autocomplete_max_visible", None)
                if set_max_visible is not None:
                    set_max_visible(autocomplete_max_visible)
            return cwd_changed

    async def _rebind_current_session(self, options: dict | None = None, new_session=None, swap=None) -> None:
        """pi's rebindCurrentSession. On a session replacement the runtime
        host hands over `new_session` and its `swap`: the swap and pi's first
        synchronous block (runtime settings, the chat redraw, the
        subscription) are one hold of the UI state lock, so a frame shows the
        old session or the new one, never a mix (§4.5c). The block's I/O (the
        cwd, the trust warning's check) is prefetched for the new session
        before the hold (§4.5b)."""
        options = options or {}
        target = new_session if new_session is not None else self.session
        resolved_cwd = await self._footer_data_provider.resolve_cwd(target.session_manager.get_cwd())
        trust_warning = options.get("renderBeforeBind") and await self._needs_project_trust_warning(target)
        with self.ui.state_lock:
            if swap is not None:
                swap()
            session = self.session
            if self._unsubscribe is not None:
                self._unsubscribe()
            self._unsubscribe = None
            cwd_changed = self._apply_runtime_settings(resolved_cwd)
            if options.get("renderBeforeBind"):
                self.render_current_session_state(trust_warning)
                self._subscribe_to_agent()
        if cwd_changed:
            await self._footer_data_provider.watch_cwd()

        await self._bind_current_session_extensions()

        with self.ui.state_lock:
            if self.session is not session:
                return

            if not options.get("renderBeforeBind"):
                self._subscribe_to_agent()

            self._update_available_provider_count()
            self._update_editor_border_color()
            self._update_terminal_title()

    async def _handle_fatal_runtime_error(self, prefix: str, error) -> None:
        self.show_error(f"{prefix}: {error}")
        stop_theme_watcher()
        await self.stop("transcript")
        hard_exit(1)

    def render_current_session_state(self, trust_warning: bool) -> None:
        with self.ui.state_lock:
            self._loaded_resources_container.clear()
            self._chat_container.clear()
            self._pending_messages_container.clear()
            with self._compaction_queue_guard:
                self._compaction_queued_messages = []
            self._streaming_component = None
            self._streaming_message = None
            self._pending_tools.clear()
            self._render_initial_messages(trust_warning)

    def _get_registered_tool_definition(self, tool_name: str):
        """Extension-registered definition, falling back to the built-in one.

        The renderer components take whatever this returns, so they never
        reach into the tool registry themselves.
        """
        return self.session.extension_runner.resolve_tool_renderers(
            tool_name, lambda: with_built_in_renderers(tool_name, self.session.get_tool_definition(tool_name))
        )

    def _get_markdown_transformers(self) -> list:
        return self.session.extension_runner.get_markdown_transformers()

    def _setup_extension_shortcuts(self, extension_runner) -> None:
        """Set up keyboard shortcuts registered by extensions."""
        get_shortcuts = getattr(extension_runner, "get_shortcuts", None)
        shortcuts = get_shortcuts(self._keybindings.get_effective_config()) if get_shortcuts is not None else {}
        if not shortcuts:
            return

        # Create a context for shortcut handlers (pi's ExtensionContext object
        # literal: attribute access, pidrei's snake_case names — the same
        # surface handlers get from the runner's context).
        def create_context() -> ExtensionContext:
            def compact(options=None):
                options = options or {}

                async def run_compact() -> None:
                    try:
                        result = await self.session.compact(options.get("custom_instructions"))
                        on_complete = options.get("on_complete")
                        if on_complete is not None:
                            on_complete(result)
                    except Exception as error:
                        on_error = options.get("on_error")
                        if on_error is not None:
                            on_error(error)

                tonio.spawn.without_tracking(run_compact())

            return ExtensionContext(
                ui=self._create_extension_ui_context(),
                mode="tui",
                has_ui=True,
                cwd=self.session_manager.get_cwd(),
                session_manager=self.session_manager,
                model_registry=extension_runner.get_model_registry(),
                model=self.session.model,
                scoped_models=self.session.scoped_models,
                thinking_level=self.session.thinking_level,
                is_idle=lambda: self.session.is_idle,
                is_project_trusted=lambda: self.settings_manager.is_project_trusted(),
                signal=self.session.agent.signal,
                abort=lambda: self._restore_queued_messages_to_editor({"abort": True}),
                has_pending_messages=lambda: self.session.pending_message_count > 0,
                shutdown=lambda: self._request_shutdown(),
                get_context_usage=lambda: self.session.get_context_usage(),
                compact=compact,
                get_system_prompt=lambda: self.session.system_prompt,
            )

        def on_extension_shortcut(data: str) -> bool:
            for shortcut_str, shortcut in shortcuts.items():
                if matches_key(data, shortcut_str):

                    async def run_handler(handler=shortcut) -> None:
                        try:
                            # Async-only, like command handlers (docs/extensions.md).
                            await handler.handler(create_context())
                        except Exception as err:
                            self.show_error(f"Shortcut handler error: {err}")

                    tonio.spawn.without_tracking(run_handler())
                    return True
            return False

        with self.ui.state_lock:
            self._default_editor.on_extension_shortcut = on_extension_shortcut

    def _set_extension_status(self, key: str, text) -> None:
        """Set extension status text in the footer."""
        self._footer_data_provider.set_extension_status(key, text)
        self.ui.request_render()

    # The status-indicator state machine (`_active_status_indicator` and the
    # working-message knobs) changes only under the UI state lock, so
    # kind-guards and set_message updates read consistent state whether the
    # caller is the agent listener, a detached command handler, or an
    # extension.

    def _set_editor_working_status_indicator(self, indicator) -> bool:
        """Hand the status indicator to the editor's border; False when the
        active editor has not opted in (the standalone status row stays)."""
        self._default_editor.set_working_status_indicator(None)
        if not is_working_status_editor(self.editor):
            return False
        self.editor.set_working_status_indicator(indicator)
        return True

    def _show_status_indicator(self, indicator) -> None:
        with self.ui.state_lock:
            if self._active_status_indicator is not None:
                self._active_status_indicator.dispose()
            self._active_status_indicator = indicator
            self._active_working_indicator_embedded = False
            self._status_container.clear()
            self._set_editor_working_status_indicator(None)
            if self._set_editor_working_status_indicator(indicator):
                self._active_working_indicator_embedded = True
                return
            self._status_container.set_children([indicator])

    def _clear_status_indicator(self, kind: str | None = None) -> None:
        with self.ui.state_lock:
            if kind and (self._active_status_indicator is None or self._active_status_indicator.kind != kind):
                return
            cleared_indicator = self._active_status_indicator
            cleared_indicator_was_embedded = self._active_working_indicator_embedded
            if cleared_indicator is not None:
                cleared_indicator.dispose()
            self._active_status_indicator = None
            self._active_working_indicator_embedded = False
            self._set_editor_working_status_indicator(None)
            if (
                cleared_indicator is not None
                and not cleared_indicator_was_embedded
                and self._options.get("tuiMode") == "regular"
                and self.ui.get_clear_on_shrink()
            ):
                self._status_container.set_children([self._idle_status])
            else:
                self._status_container.clear()

    def _show_working_status_indicator(self) -> None:
        """Under the UI state lock (the event handler, `_set_working_visible`)."""
        color_fn = None
        if is_working_status_editor(self.editor):

            def color_fn(text: str) -> str:
                border_color = self.editor.border_color or theme.get_thinking_border_color(
                    self.session.thinking_level or "off"
                )
                return border_color(text)

        self._show_status_indicator(
            WorkingStatusIndicator(
                self.ui,
                self._working_message if self._working_message is not None else self._default_working_message,
                self._working_indicator_options,
                color_fn,
            )
        )

    def _set_working_visible(self, visible: bool) -> None:
        with self.ui.state_lock:
            self._working_visible = visible
            if not visible:
                self._clear_status_indicator("working")
                self.ui.request_render()
                return
            if self.session.is_streaming and (
                self._active_status_indicator is None or self._active_status_indicator.kind != "working"
            ):
                self._show_working_status_indicator()
            self.ui.request_render()

    def _set_working_indicator(self, options=None) -> None:
        with self.ui.state_lock:
            self._working_indicator_options = options
            if self._active_status_indicator is not None and self._active_status_indicator.kind == "working":
                self._active_status_indicator.set_indicator(options)
            self.ui.request_render()

    def _set_hidden_thinking_label(self, label=None) -> None:
        with self.ui.state_lock:
            self._hidden_thinking_label = label if label is not None else self._default_hidden_thinking_label
            for child in self._chat_container.children:
                if isinstance(child, AssistantMessageComponent):
                    child.set_hidden_thinking_label(self._hidden_thinking_label)
            if self._streaming_component is not None:
                self._streaming_component.set_hidden_thinking_label(self._hidden_thinking_label)
            self.ui.request_render()

    # Maximum total widget lines to prevent viewport overflow
    MAX_WIDGET_LINES = 10

    def _set_extension_widget(self, key: str, content, options: dict | None = None) -> None:
        """Set an extension widget (list of strings or a component factory)."""
        placement = (options or {}).get("placement") or "aboveEditor"

        def remove_existing(widget_map: dict) -> None:
            existing = widget_map.get(key)
            if existing is not None and getattr(existing, "dispose", None) is not None:
                existing.dispose()
            widget_map.pop(key, None)

        with self.ui.state_lock:
            remove_existing(self._extension_widgets_above)
            remove_existing(self._extension_widgets_below)

            if content is None:
                self._render_widgets()
                return

            if isinstance(content, list):
                # Wrap string list in a Container with Text components
                container = Container()
                for line in content[: InteractiveMode.MAX_WIDGET_LINES]:
                    container.add_child(Text(line, 1, 0))
                if len(content) > InteractiveMode.MAX_WIDGET_LINES:
                    container.add_child(ThemedText(lambda: theme.fg("muted", "... (widget truncated)"), 1, 0))
                component = container
            else:
                # Factory function - create component
                component = content(self._extension_tui, theme)

            target_map = self._extension_widgets_below if placement == "belowEditor" else self._extension_widgets_above
            target_map[key] = component
            self._render_widgets()

    def _clear_extension_widgets(self) -> None:
        """Dispose and drop every extension widget."""
        with self.ui.state_lock:
            for widget in self._extension_widgets_above.values():
                dispose = getattr(widget, "dispose", None)
                if dispose is not None:
                    dispose()
            for widget in self._extension_widgets_below.values():
                dispose = getattr(widget, "dispose", None)
                if dispose is not None:
                    dispose()
            self._extension_widgets_above.clear()
            self._extension_widgets_below.clear()
            self._render_widgets()

    def _reset_extension_ui(self) -> None:
        # From session invalidation, the extension `reload` action and
        # /reload: the whole reset is one change under the UI state lock.
        with self.ui.state_lock:
            self._clear_extension_terminal_input_listeners()
            self._footer_data_provider.clear_extension_statuses()
            with self._extension_registry_guard:
                self._autocomplete_provider_wrappers = ()
            self._set_custom_editor_component(None)
            self._default_editor.on_extension_shortcut = None
            self._working_message = None
            self._working_visible = True
            if self._extension_selector is not None:
                self._hide_extension_selector(self._extension_selector)
            if self._extension_input is not None:
                self._hide_extension_input(self._extension_input)
            if self._extension_editor is not None:
                self._hide_extension_editor(self._extension_editor)
            self.ui.hide_overlay()
            self._set_extension_footer(None)
            self._set_extension_header(None)
            self._clear_extension_widgets()
            self._footer.invalidate()
            self._setup_autocomplete_provider()
            self._update_terminal_title()
            self._set_working_indicator()
            if self._active_status_indicator is not None and self._active_status_indicator.kind == "working":
                self._active_status_indicator.set_message(
                    f"{self._default_working_message} ({key_text('app.interrupt')} to interrupt)"
                )
            self._set_hidden_thinking_label()

    def _render_widgets(self) -> None:
        """Render all extension widgets to the widget containers."""
        if self._widget_container_above is None or self._widget_container_below is None:
            return
        self._render_widget_container(self._widget_container_above, self._extension_widgets_above, True, True)
        self._render_widget_container(self._widget_container_below, self._extension_widgets_below, False, False)
        self.ui.request_render()

    def _render_widget_container(self, container, widgets: dict, spacer_when_empty: bool, leading_spacer: bool) -> None:
        container.clear()

        if not widgets:
            if spacer_when_empty:
                container.add_child(Spacer(1))
            return

        if leading_spacer:
            container.add_child(Spacer(1))
        for component in widgets.values():
            container.add_child(component)

    def _set_extension_footer(self, factory) -> None:
        """Set a custom footer component, or restore the built-in footer."""
        with self.ui.state_lock:
            # Dispose existing custom footer
            if self._custom_footer is not None and getattr(self._custom_footer, "dispose", None) is not None:
                self._custom_footer.dispose()

            self._footer_container.clear()
            if factory is not None:
                # Create and add custom footer, passing the data provider
                self._custom_footer = factory(self._extension_tui, theme, self._footer_data_provider)
                self._footer_container.add_child(self._custom_footer)
            else:
                # Restore built-in footer
                self._custom_footer = None
                self._footer_container.add_child(self._footer)

            self.ui.request_render()

    def _set_extension_header(self, factory) -> None:
        """Set a custom header component, or restore the built-in header."""
        with self.ui.state_lock:
            # Header may not be initialized yet if called during early
            # initialization
            if self._built_in_header is None:
                return

            # Dispose existing custom header
            if self._custom_header is not None and getattr(self._custom_header, "dispose", None) is not None:
                self._custom_header.dispose()

            # Find the index of the current header in the header container
            current_header = self._custom_header if self._custom_header is not None else self._built_in_header
            try:
                index = self._header_container.children.index(current_header)
            except ValueError:
                index = -1

            if factory is not None:
                # Create and add custom header
                self._custom_header = factory(self._extension_tui, theme)
                if is_expandable(self._custom_header):
                    self._custom_header.set_expanded(self._tool_output_expanded)
                if index != -1:
                    self._header_container.children[index] = self._custom_header
                else:
                    # If not found (e.g. built-in header was never added), add
                    # at the top
                    self._header_container.children.insert(0, self._custom_header)
            else:
                # Restore built-in header
                self._custom_header = None
                if is_expandable(self._built_in_header):
                    self._built_in_header.set_expanded(self._tool_output_expanded)
                if index != -1:
                    self._header_container.children[index] = self._built_in_header

            self.ui.request_render()

    # Extension tasks add and remove these while the renderer switch rebinds
    # and reset/stop clear them: every step runs under the registry guard, so
    # a subscription is never rebound after its removal (leaking a listener)
    # or iterated while the set changes.

    def _add_extension_terminal_input_listener(self, handler):
        with self._extension_registry_guard:
            subscription = _TerminalInputSubscription(handler, self.ui.add_input_listener(handler))
            self._extension_terminal_input_subscriptions.add(subscription)

        def remove() -> None:
            with self._extension_registry_guard:
                if subscription not in self._extension_terminal_input_subscriptions:
                    return  # already cleared
                subscription.unsubscribe()
                self._extension_terminal_input_subscriptions.discard(subscription)

        return remove

    def _rebind_extension_terminal_input_listeners(self) -> None:
        with self._extension_registry_guard:
            for subscription in self._extension_terminal_input_subscriptions:
                subscription.unsubscribe()
                subscription.unsubscribe = self.ui.add_input_listener(subscription.handler)

    def _clear_extension_terminal_input_listeners(self) -> None:
        with self._extension_registry_guard:
            for subscription in self._extension_terminal_input_subscriptions:
                subscription.unsubscribe()
            self._extension_terminal_input_subscriptions.clear()

    def _create_project_trust_context(self, cwd: str) -> ProjectTrustContext:
        ui = self._create_extension_ui_context()
        return ProjectTrustContext(
            cwd=cwd,
            mode="tui",
            has_ui=True,
            # pi narrows the UI to these four; an attribute object, as the
            # trust flow and project_trust handlers read `ctx.ui.select` etc.
            ui=SimpleNamespace(select=ui.select, confirm=ui.confirm, input=ui.input, notify=ui.notify),
        )

    def _create_extension_ui_context(self) -> ExtensionUIContext:
        """Create the ExtensionUIContext object for extensions."""
        return ExtensionUIContext(self)

    def _show_extension_selector(self, title: str, options: list, opts: dict | None = None):
        """Show a selector for extensions: mounted before this returns (pi's
        synchronous mount, so a key that opens it sends the next key to it);
        returns the spawn handle of the wait for the pick, which the caller
        may await or drop (spec/ui-island.md, `ctx.ui`)."""
        opts = opts or {}
        done = tonio.Event()
        # Settled by a pick, the countdown or an abort callback on any task:
        # the first settle claims under the guard.
        settle_guard = threading.Lock()
        outcome: dict = {"value": None, "settled": False}
        remove_abort = None

        signal = opts.get("signal")
        if signal is not None and signal.cancelled:
            return tonio.spawn(_answer(None))

        def settle(value, hide) -> None:
            with settle_guard:
                if outcome["settled"]:
                    return
                outcome["settled"] = True
            outcome["value"] = value
            if remove_abort is not None:
                remove_abort()
            hide(component)
            done.set()

        component = ExtensionSelectorComponent(
            title,
            options,
            lambda option: settle(option, self._hide_extension_selector),
            lambda: settle(None, self._hide_extension_selector),
            {
                "tui": self.ui,
                "timeout": opts.get("timeout"),
                "onToggleToolsExpanded": lambda: self._toggle_tool_output_expansion(),
            },
        )
        if signal is not None:
            remove_abort = signal.on_cancel(lambda _reason: settle(None, self._hide_extension_selector))

        # The field is claimed at mount: a hide for the previous dialog must
        # find that dialog there, not this one.
        with self.ui.state_lock:
            if not outcome["settled"]:  # else aborted before it was shown
                self._dispose_active_selector()
                self._extension_selector = component
                self._editor_container.clear()
                self._editor_container.add_child(component)
                self.ui.set_focus(component)
                self.ui.request_render()

        return tonio.spawn(_wait_for_answer(done, outcome))

    def _hide_extension_selector(self, component) -> None:
        with self.ui.state_lock:
            component.dispose()
            if self._extension_selector is not component:
                return  # never mounted, or already replaced
            self._editor_container.clear()
            self._editor_container.add_child(self.editor)
            self._extension_selector = None
            self.ui.set_focus(self.editor)
            self.ui.request_render()

    def _show_extension_confirm(self, title: str, message: str, opts: dict | None = None) -> Awaitable[bool]:
        """Show a confirmation dialog for extensions (see `_show_extension_selector`)."""
        return tonio.spawn(_is_yes(self._show_extension_selector(f"{title}\n{message}", ["Yes", "No"], opts)))

    async def _prompt_for_missing_session_cwd(self, error) -> str | None:
        confirmed = await self._show_extension_confirm(
            "Session cwd not found", format_missing_session_cwd_prompt(error.issue)
        )
        return error.issue.fallback_cwd if confirmed else None

    def _show_extension_input(self, title: str, placeholder=None, opts: dict | None = None):
        """Show a text input for extensions (see `_show_extension_selector`)."""
        opts = opts or {}
        done = tonio.Event()
        # First settle claims: see `_show_extension_selector`.
        settle_guard = threading.Lock()
        outcome: dict = {"value": None, "settled": False}
        remove_abort = None

        signal = opts.get("signal")
        if signal is not None and signal.cancelled:
            return tonio.spawn(_answer(None))

        def settle(value, hide) -> None:
            with settle_guard:
                if outcome["settled"]:
                    return
                outcome["settled"] = True
            outcome["value"] = value
            if remove_abort is not None:
                remove_abort()
            hide(component)
            done.set()

        # Submit, cancel and the countdown come from the component; an abort
        # from any task (see `_show_extension_selector`).
        component = ExtensionInputComponent(
            title,
            placeholder,
            lambda value: settle(value, self._hide_extension_input),
            lambda: settle(None, self._hide_extension_input),
            {"tui": self.ui, "timeout": opts.get("timeout")},
        )
        if signal is not None:
            remove_abort = signal.on_cancel(lambda _reason: settle(None, self._hide_extension_input))

        # Field claimed at mount: see `_show_extension_selector`.
        with self.ui.state_lock:
            if not outcome["settled"]:  # else aborted before it was shown
                self._dispose_active_selector()
                self._extension_input = component
                self._editor_container.clear()
                self._editor_container.add_child(component)
                self.ui.set_focus(component)
                self.ui.request_render()

        return tonio.spawn(_wait_for_answer(done, outcome))

    def _hide_extension_input(self, component) -> None:
        with self.ui.state_lock:
            component.dispose()
            if self._extension_input is not component:
                return  # never mounted, or already replaced
            self._editor_container.clear()
            self._editor_container.add_child(self.editor)
            self._extension_input = None
            self.ui.set_focus(self.editor)
            self.ui.request_render()

    def _show_extension_editor(self, title: str, prefill=None):
        """Show a multi-line editor for extensions (with Ctrl+G support)."""
        done = tonio.Event()
        # First settle claims: see `_show_extension_selector`.
        settle_guard = threading.Lock()
        outcome: dict = {"value": None, "settled": False}

        def settle(value) -> None:
            with settle_guard:
                if outcome["settled"]:
                    return
                outcome["settled"] = True
            outcome["value"] = value
            self._hide_extension_editor(component)
            done.set()

        component = ExtensionEditorComponent(
            self.ui,
            self._keybindings,
            title,
            prefill,
            lambda value: settle(value),
            lambda: settle(None),
            None,
            self.settings_manager.get_external_editor_command(),
        )

        # Field claimed at mount: see `_show_extension_selector`.
        with self.ui.state_lock:
            self._dispose_active_selector()
            self._extension_editor = component
            self._editor_container.clear()
            self._editor_container.add_child(component)
            self.ui.set_focus(component)
            self.ui.request_render()

        return tonio.spawn(_wait_for_answer(done, outcome))

    def _hide_extension_editor(self, component) -> None:
        """Put the editor back in place of the extension editor dialog."""
        with self.ui.state_lock:
            if self._extension_editor is not component:
                return  # already replaced
            self._editor_container.clear()
            self._editor_container.add_child(self.editor)
            self._extension_editor = None
            self.ui.set_focus(self.editor)
            self.ui.request_render()

    def _set_custom_editor_component(self, factory) -> None:
        """Set a custom editor component from an extension.

        Pass None to restore the default editor. The factory is published and
        the editor swapped in one hold of the UI state lock, so
        `get_editor_component()` sees it immediately, as in pi.
        """
        with self.ui.state_lock:
            self._editor_component_factory = factory

            # Save text from current editor before switching
            current_text = self.editor.get_text()

            self._dispose_active_selector()
            self._editor_container.clear()

            if factory is not None:
                # Create the custom editor with tui, theme, and keybindings
                new_editor = factory(self._extension_tui, get_editor_theme(), self._keybindings)

                # Wire up callbacks from the default editor
                new_editor.on_submit = self._default_editor.on_submit
                new_editor.on_change = self._default_editor.on_change

                # Copy text from previous editor
                new_editor.set_text(current_text)

                # Copy appearance settings if supported
                if getattr(new_editor, "border_color", None) is not None:
                    new_editor.border_color = self._default_editor.border_color
                set_padding = getattr(new_editor, "set_padding_x", None)
                if set_padding is not None:
                    set_padding(self._default_editor.get_padding_x())
                set_max_visible = getattr(new_editor, "set_autocomplete_max_visible", None)
                if set_max_visible is not None:
                    set_max_visible(self._default_editor.get_autocomplete_max_visible())

                # Set autocomplete if supported
                set_provider = getattr(new_editor, "set_autocomplete_provider", None)
                if set_provider is not None and self._autocomplete_provider is not None:
                    set_provider(self._autocomplete_provider)

                # If extending CustomEditor, copy app-level handlers (duck typed)
                if isinstance(getattr(new_editor, "action_handlers", None), dict):
                    if not getattr(new_editor, "on_escape", None):

                        def forward_escape():
                            if self._default_editor.on_escape:
                                self._default_editor.on_escape()

                        new_editor.on_escape = forward_escape
                    if not getattr(new_editor, "on_ctrl_d", None):

                        def forward_ctrl_d():
                            if self._default_editor.on_ctrl_d:
                                self._default_editor.on_ctrl_d()

                        new_editor.on_ctrl_d = forward_ctrl_d
                    if not getattr(new_editor, "on_paste_image", None):
                        new_editor.on_paste_image = lambda: (
                            self._default_editor.on_paste_image() if self._default_editor.on_paste_image else None
                        )
                    if not getattr(new_editor, "on_extension_shortcut", None):
                        new_editor.on_extension_shortcut = lambda data: (
                            self._default_editor.on_extension_shortcut(data)
                            if self._default_editor.on_extension_shortcut
                            else False
                        )
                    # Copy action handlers (clear, suspend, model switching, ...)
                    for action, handler in self._default_editor.action_handlers.items():
                        new_editor.action_handlers[action] = handler

                self.editor = new_editor
            else:
                # Restore default editor with text from custom editor
                self._default_editor.set_text(current_text)
                self.editor = self._default_editor

            self._editor_container.add_child(self.editor)
            if self._active_status_indicator is not None:
                self._status_container.clear()
                self._active_working_indicator_embedded = self._set_editor_working_status_indicator(
                    self._active_status_indicator
                )
                if not self._active_working_indicator_embedded:
                    self._status_container.set_children([self._active_status_indicator])
            self.ui.set_focus(self.editor)
            self.ui.request_render()

    def _show_extension_notify(self, message: str, type=None) -> None:
        """Show a notification for extensions."""
        if type == "error":
            self.show_error(message)
        elif type == "warning":
            self.show_warning(message)
        else:
            self.show_status(message)

    def _show_extension_custom(self, factory, options: dict | None = None):
        """Show a custom component with keyboard focus.

        Overlay mode renders on top of existing content.

        The factory is synchronous and runs in one hold of the UI state lock
        with the editor-text snapshot (pi takes it at call time) and the
        mount (spec/ui-island.md, `ctx.ui`), so no frame or key lands in between.
        Returns the spawn handle of the wait for the result, which the caller
        may await or drop. `done` (`close`) takes the lock itself: callable
        from input handling, a timer, spawned work or the factory itself.
        """
        options = options or {}
        is_overlay = bool(options.get("overlay"))
        done = tonio.Event()
        # Written under the UI state lock (the mount hold and `close`); the
        # waiting handle reads `value` after `done`.
        state: dict = {"component": None, "closed": False, "value": None, "editor_text": ""}

        def restore_editor() -> None:
            self._editor_container.clear()
            self._editor_container.add_child(self.editor)
            self.editor.set_text(state["editor_text"])
            self.ui.set_focus(self.editor)
            self.ui.request_render()

        def close(result=None) -> None:
            with self.ui.state_lock:
                if state["closed"]:
                    return
                state["closed"] = True
                state["value"] = result
                component = state["component"]
                if not is_overlay:
                    restore_editor()
                elif component is not None:
                    self.ui.hide_overlay()
                # Note: both branches above already request a render
                with contextlib.suppress(Exception):
                    if component is not None and getattr(component, "dispose", None) is not None:
                        component.dispose()
            done.set()

        def mount(component) -> None:
            if is_overlay:
                overlay_options = options.get("overlayOptions")
                if overlay_options is not None:
                    resolved_options = overlay_options() if callable(overlay_options) else overlay_options
                else:
                    # Fallback: use component's width property if available
                    width = getattr(component, "width", None)
                    resolved_options = {"width": width} if width else None
                handle = self.ui.show_overlay(component, resolved_options)
                # Expose handle to caller for visibility control
                on_handle = options.get("onHandle")
                if on_handle is not None:
                    on_handle(guard_overlay_handle(self.ui, handle))
            else:
                self._dispose_active_selector()
                self._editor_container.clear()
                self._editor_container.add_child(component)
                self.ui.set_focus(component)
                self.ui.request_render()

        with self.ui.state_lock:
            state["editor_text"] = self.editor.get_text()
            try:
                # Synchronous only: an `async def` factory is refused.
                component = call_sync(lambda: factory(self._extension_tui, theme, self._keybindings, close))
            except Exception:
                if not state["closed"]:
                    if not is_overlay:
                        restore_editor()
                    raise
                component = None  # closed from inside the factory: the result stands
            if not state["closed"] and component is not None:
                state["component"] = component
                mount(component)

        return tonio.spawn(_wait_for_answer(done, state))

    def _show_extension_error(self, extension_path: str, error: str, stack=None) -> None:
        """Show an extension error in the UI."""
        error_msg = f'Extension "{extension_path}" error: {error}'
        error_text = ThemedText(lambda: theme.fg("error", error_msg), 1, 0)
        # Show stack trace in dim color, indented (skip first line, it
        # duplicates the error message)
        stack_lines = stack.split("\n")[1:] if stack else []

        def render_stack() -> str:
            return "\n".join(theme.fg("dim", f"  {line.strip()}") for line in stack_lines)

        # Raised on whatever task hit the extension error.
        with self.ui.state_lock:
            self._chat_container.add_child(error_text)
            if stack_lines:
                self._chat_container.add_child(ThemedText(render_stack, 1, 0))
            self.ui.request_render()

    # =========================================================================
    # Key Handlers
    # =========================================================================

    def _setup_key_handlers(self) -> None:
        # Set up handlers on the default editor - they use self.editor for
        # text access so they work correctly regardless of active editor
        # Escape, from the editor's input handling.
        def on_escape() -> None:
            if self.session.is_streaming:
                self._restore_queued_messages_to_editor({"abort": True})
            elif self.session.is_bash_running:
                self.session.abort_bash()
            elif self._is_bash_mode:
                self._set_editor_text("")
                self._is_bash_mode = False
                self._update_editor_border_color()
            elif not self.editor.get_text().strip():
                # Double-escape with empty editor triggers /tree, /fork, or
                # nothing based on setting
                action = self.settings_manager.get_double_escape_action()
                if action != "none":
                    now = clock.monotonic() * 1000
                    if now - self._last_escape_time < 500:
                        if action == "tree":
                            self._show_tree_selector()
                        else:
                            self._show_user_message_selector()
                        self._last_escape_time = 0
                    else:
                        self._last_escape_time = now

        self._default_editor.on_escape = on_escape

        # Register app action handlers (synchronous, as in pi)
        self._default_editor.on_action("app.clear", self._handle_ctrl_c)
        self._default_editor.on_ctrl_d = self._handle_ctrl_d
        # The UI stops before the next key, as pi's (synchronous) handler does.
        self._default_editor.on_action("app.suspend", lambda: self._finish_before_next_input(self._handle_ctrl_z()))
        self._default_editor.on_action(
            "app.thinking.cycle", lambda: self._finish_before_next_input(self._cycle_thinking_level())
        )
        self._default_editor.on_action("app.model.cycleForward", lambda: self._spawn_flow(self._cycle_model("forward")))
        self._default_editor.on_action(
            "app.model.cycleBackward", lambda: self._spawn_flow(self._cycle_model("backward"))
        )

        # Global debug handler on TUI (works regardless of focus)
        self.ui.on_debug = lambda: self._spawn_flow(self._handle_debug_command())
        self._default_editor.on_action("app.model.select", self._show_model_selector)
        self._default_editor.on_action("app.tools.expand", self._toggle_tool_output_expansion)
        self._default_editor.on_action("app.thinking.toggle", self._toggle_thinking_block_visibility)
        self._default_editor.on_action(
            "app.editor.external", lambda: self._finish_before_next_input(self._open_external_editor())
        )
        self._default_editor.on_action(
            "app.message.copy",
            lambda: self._handle_copy_command({"flashConfirmation": True, "preferSelection": True}),
        )
        self._default_editor.on_action("app.message.followUp", self._handle_follow_up)
        self._default_editor.on_action("app.message.dequeue", self._handle_dequeue)
        self._default_editor.on_action("app.session.new", self._handle_clear_command)
        self._default_editor.on_action("app.session.tree", self._show_tree_selector)
        self._default_editor.on_action("app.session.fork", self._show_user_message_selector)
        self._default_editor.on_action("app.session.resume", self._show_session_selector)

        # From editor mutations.
        def on_change(text: str) -> None:
            was_bash_mode = self._is_bash_mode
            self._is_bash_mode = text.lstrip().startswith("!")
            if was_bash_mode != self._is_bash_mode:
                self._update_editor_border_color()

        self._default_editor.on_change = on_change

        # Handle clipboard paste (triggered on Ctrl+V). Images are attached
        # by path; otherwise, paste plain text from the system clipboard.
        self._default_editor.on_paste_image = lambda: self._spawn_flow(self._handle_clipboard_paste())

    def _spawn_flow(self, flow: Awaitable[None]) -> None:
        """Run `flow` on its own task, fire-and-forget (pi's `void` call).
        Whatever escapes it goes to the crash handler, as an unhandled
        rejection does in pi (spec/ui-island.md, errors)."""

        async def run() -> None:
            try:
                await flow
            except Exception as error:
                await self._uncaught_crash(error)

        tonio.spawn.without_tracking(run())

    def _finish_before_next_input(self, work: Awaitable[None]) -> None:
        """Spawn a key's slow work and hold the next key until it finishes
        (pi does that work synchronously). The input consumer waits for it
        before the next item, and the work applies its UI changes in place."""
        self.ui.finish_before_next_input(tonio.spawn(work))

    async def _handle_clipboard_paste(self) -> None:
        try:
            file_paths = await read_clipboard_file_paths()
            if file_paths:
                if any(unicodedata.category(char) == "Cc" for file_path in file_paths for char in file_path):
                    raise Exception("Clipboard file path contains control characters")
                paths = (
                    " ".join(_quote_if_needed(file_path) for file_path in file_paths)
                    if self._is_bash_mode
                    else "\n".join(file_paths)
                )
                # The cursor read and the insert are one hold, so a key typed
                # meanwhile cannot move the cursor between them.
                with self.ui.state_lock:
                    # pi: `this.editor.getCursor?.()` — custom editors may not have one.
                    get_cursor = getattr(self.editor, "get_cursor", None)
                    cursor = get_cursor() if get_cursor is not None else None
                    character_before_cursor = character_after_cursor = ""
                    if cursor is not None:
                        lines = self.editor.get_text().split("\n")
                        current_line = lines[cursor["line"]] if cursor["line"] < len(lines) else ""
                        col = cursor["col"]
                        character_before_cursor = current_line[col - 1] if 0 < col <= len(current_line) else ""
                        character_after_cursor = current_line[col] if col < len(current_line) else ""
                    leading_space = " " if character_before_cursor and not character_before_cursor.isspace() else ""
                    trailing_space = " " if character_after_cursor and not character_after_cursor.isspace() else ""
                    insert = getattr(self.editor, "insert_text_at_cursor", None)
                    if insert is not None:
                        insert(f"{leading_space}{paths}{trailing_space}")
                self.ui.request_render()
                return

            image = await read_clipboard_image()
            if image:
                ext = extension_for_image_mime_type(image["mimeType"]) or "png"
                file_name = f"{APP_NAME}-clipboard-{uuid.uuid4()}.{ext}"
                file_path = TEMP_DIR / file_name
                await file_path.write_bytes(image["bytes"])

                self._insert_into_editor(str(file_path))
                self.ui.request_render()
                return

            text = await read_clipboard_text()
            if text:
                self._insert_into_editor(text)
                self.ui.request_render()
        except Exception as error:
            self.show_error(f"Failed to paste from clipboard: {error}")

    # pi's `this.editor.setText(...)` and friends: like pi's, they request no
    # render of their own (their callers do, or a render follows anyway).

    def _set_editor_text(self, text: str) -> None:
        with self.ui.state_lock:
            self.editor.set_text(text)

    def _insert_into_editor(self, text: str) -> None:
        with self.ui.state_lock:
            insert = getattr(self.editor, "insert_text_at_cursor", None)
            if insert is not None:
                insert(text)

    def _fill_empty_editor(self, text: str) -> None:
        """Put `text` in the editor unless it holds something already: the
        check and the write are one hold, so keys typed meanwhile win."""
        with self.ui.state_lock:
            if not self.editor.get_text().strip():
                self.editor.set_text(text)

    def _apply_editor_history(self, text: str) -> None:
        """Add ``text`` to the current editor's history."""
        with self.ui.state_lock:
            add_to_history = getattr(self.editor, "add_to_history", None)
            if add_to_history is not None:
                add_to_history(text)

    def _setup_startup_input_handlers(self) -> None:
        """The startup-window bindings (pi wires these inline in `init()`;
        extracted so the wiring is unit-testable like `_setup_editor_submit_handler`).

        `on_submit` is the editor's `(text) -> None` callback, not an action
        handler: a submit landing before `_setup_editor_submit_handler` ran
        crashed the input pump (reachable on a slow runner — macOS CI, 0.85.1)."""
        self._default_editor.on_action("app.clear", self._handle_ctrl_c)
        self._default_editor.on_ctrl_d = self._handle_ctrl_d
        self._default_editor.on_submit = self._handle_startup_submit

    def _setup_editor_submit_handler(self) -> None:
        self._default_editor.on_submit = self._handle_editor_submit

    async def _then_clear_editor(self, command: Awaitable[None]) -> None:
        """pi's submit branch `await this.handleX(); this.editor.setText("")`."""
        await command
        self._set_editor_text("")

    def _handle_editor_submit(self, text: str) -> None:
        """The editor's submit (its input handling, or the follow-up action).
        pi's handler is called without awaiting: its part up to the first
        await — clearing the editor, mounting a selector, queueing — runs in
        the keypress, before the next key. That part runs here; the rest of a
        command is spawned."""
        text = text.strip()
        if not text:
            return

        # Handle commands
        # Commands that are synchronous in pi but start with I/O here: the
        # next key waits for them (and for the editor clear after them).
        if text == "/settings":
            self._finish_before_next_input(self._then_clear_editor(self._show_settings_selector()))
            return
        if text == "/scoped-models":
            self._set_editor_text("")
            self._show_models_selector()
            return
        if text == "/model" or text.startswith("/model "):
            search_term = text[7:].strip() if text.startswith("/model ") else None
            self._set_editor_text("")
            self._handle_model_command(search_term)
            return
        if text == "/thinking" or text.startswith("/thinking "):
            search_term = text[10:].strip() if text.startswith("/thinking ") else None
            self._set_editor_text("")
            self._handle_thinking_command(search_term)
            return
        if text == "/export" or text.startswith("/export "):
            self._spawn_flow(self._then_clear_editor(self._handle_export_command(text)))
            return
        if text == "/import" or text.startswith("/import "):
            self.handle_import_command(text, and_then=lambda: self._set_editor_text(""))
            return
        if text == "/share":
            self._spawn_flow(self._then_clear_editor(self._handle_share_command()))
            return
        if text == "/copy":
            self._handle_copy_command(and_then=lambda: self._set_editor_text(""))
            return
        if text == "/name" or text.startswith("/name "):
            self._handle_name_command(text)
            return
        if text == "/session":
            self.handle_session_command()
            self._set_editor_text("")
            return
        if text == "/changelog":
            self._finish_before_next_input(self._then_clear_editor(self._handle_changelog_command()))
            return
        if text == "/hotkeys":
            self._handle_hotkeys_command()
            self._set_editor_text("")
            return
        if text == "/fork":
            self._show_user_message_selector()
            self._set_editor_text("")
            return
        if text == "/clone":
            self._set_editor_text("")
            self.handle_clone_command()
            return
        if text == "/tree":
            self._show_tree_selector()
            self._set_editor_text("")
            return
        if text == "/trust":
            self._finish_before_next_input(self._then_clear_editor(self._show_trust_selector()))
            return
        if text == "/login" or text.startswith("/login "):
            provider_ref = text[7:].strip() if text.startswith("/login ") else None
            self._set_editor_text("")
            self._handle_login_command(provider_ref)
            return
        if text == "/logout":
            self._show_oauth_selector("logout")
            self._set_editor_text("")
            return
        if text == "/new":
            self._set_editor_text("")
            self._handle_clear_command()
            return
        if text == "/compact" or text.startswith("/compact "):
            custom_instructions = text[9:].strip() if text.startswith("/compact ") else None
            self._set_editor_text("")
            self.handle_compact_command(custom_instructions)
            return
        if text == "/reload":
            self._set_editor_text("")
            previous_editor = self._start_reload()
            if previous_editor is not None:
                self._spawn_flow(self._reload(previous_editor))
            return
        if text == "/debug":
            self._finish_before_next_input(self._then_clear_editor(self._handle_debug_command()))
            return
        if text == "/arminsayshi":
            self._handle_armin_says_hi()
            self._set_editor_text("")
            return
        if text == "/dementedelves":
            self._finish_before_next_input(self._then_clear_editor(self._handle_demented_elves()))
            return
        if text == "/resume":
            self._show_session_selector()
            self._set_editor_text("")
            return
        if text == "/quit":
            self._set_editor_text("")
            self._spawn_flow(self.shutdown())
            return

        # Handle bash command (! for normal, !! for excluded from context)
        if text.startswith("!"):
            is_excluded = text.startswith("!!")
            command = text[2:].strip() if is_excluded else text[1:].strip()
            if command:
                # "Running" is set only once the flow reaches the executor,
                # after the extensions' hook: the claim, taken here with the
                # check, closes that window (§7.3).
                if self.session.is_bash_running or self._bash_claimed:
                    self.show_warning("A bash command is already running. Press Esc to cancel it first.")
                    self._set_editor_text(text)
                    return
                self._bash_claimed = True
                self._apply_editor_history(text)
                self._spawn_flow(self._run_editor_bash_command(command, is_excluded))
                return

        # Queue input during compaction (extension commands run immediately)
        if self.session.is_compacting:
            if self._is_extension_command(text):
                self._apply_editor_history(text)
                self._set_editor_text("")
                self._spawn_flow(self.session.prompt(text))
                return
            self._queue_compaction_message(text, "steer")
            return

        # If streaming, use prompt() with steer behavior. This handles
        # extension commands (execute immediately), prompt template
        # expansion, and queueing
        if self.session.is_streaming:
            self._apply_editor_history(text)
            self._set_editor_text("")
            self._spawn_flow(self._steer(text))
            return

        # Normal message submission. First, move any pending bash components
        # to chat
        self._flush_pending_bash_components()

        with self._user_input_guard:
            on_input, self._on_input_callback = self._on_input_callback, None
            if on_input is None:
                self._pending_user_inputs.append(text)
        if on_input is not None:
            on_input(text)
        self._apply_editor_history(text)

    async def _run_editor_bash_command(self, command: str, is_excluded: bool) -> None:
        try:
            await self._handle_bash_command(command, is_excluded)
        finally:
            with self.ui.state_lock:
                self._bash_claimed = False
        with self.ui.state_lock:
            self._is_bash_mode = False
            self._update_editor_border_color()

    async def _steer(self, text: str) -> None:
        await self.session.prompt(text, PromptOptions(streaming_behavior="steer"))
        self._update_pending_messages_display()
        self.ui.request_render()

    def _subscribe_to_agent(self) -> None:
        # The fused emit contract (spec/ui-island.md, agent events): the session's
        # listener applies each event in place, under the UI state lock, as
        # synchronously as pi's listener — the UI reflects an event when its
        # emit returns, and an apply error propagates to the emitter (for
        # agent events the run fails, as in pi). An event from a session that
        # is no longer current is dropped (pi's `this.session !== session`).
        session = self.session

        def apply_event(event) -> None:
            with self.ui.state_lock:
                if self.session is not session:
                    return
                self._handle_event(event)

        self._unsubscribe = session.subscribe(apply_event)

    def _handle_event(self, event) -> None:  # noqa: C901
        # pi lazily awaits init() here; pidrei always subscribes after init.
        if not self._is_initialized:
            return

        self._footer.invalidate()
        event_type = getattr(event, "type", None)

        if event_type == "agent_start":
            self._pending_tools.clear()
            # Restore main escape handler if retry handler is still active
            # (retry success event fires later, but we need the main handler
            # now)
            if self._retry_escape_handler is not None:
                self._default_editor.on_escape = self._retry_escape_handler
                self._retry_escape_handler = None

        elif event_type == "turn_start":
            if self.settings_manager.get_show_terminal_progress():
                self.ui.terminal.set_progress(True)
            if self._working_visible:
                if self._active_status_indicator is None or self._active_status_indicator.kind != "working":
                    self._show_working_status_indicator()
            else:
                self._clear_status_indicator()
            self.ui.request_render()

        elif event_type == "queue_update":
            self._update_pending_messages_display()
            self.ui.request_render()

        elif event_type == "entry_appended":
            entry = event.entry
            if entry["id"] in self._entries_rendered_by_boundary_compaction:
                self._entries_rendered_by_boundary_compaction.discard(entry["id"])
            elif entry.get("type") == "custom":
                self._add_custom_entry_to_chat(entry)
                self.ui.request_render()
            elif entry.get("type") == "usage" and entry.get("kind") == "cache_warm":
                self._add_cache_warming_usage(entry)
                self.ui.request_render()
            elif entry.get("type") == "custom_message" and entry.get("display"):
                self._add_message_to_chat(
                    create_custom_message(
                        entry["customType"],
                        entry.get("content"),
                        entry["display"],
                        entry.get("details"),
                        entry.get("timestamp"),
                    )
                )
                self.ui.request_render()
            elif entry.get("type") == "compaction":
                self._render_boundary_compaction(entry)

        elif event_type == "session_info_changed":
            self._update_terminal_title()
            self._footer.invalidate()
            self.ui.request_render()

        elif event_type == "thinking_level_changed":
            self._footer.invalidate()
            self._update_editor_border_color()

        elif event_type == "message_start":
            if event.message.role == "custom":
                self._add_message_to_chat(event.message)
                self.ui.request_render()
            elif event.message.role == "user":
                self._add_message_to_chat(event.message)
                self._update_pending_messages_display()
                self.ui.request_render()
            elif event.message.role == "assistant":
                self._streaming_component = AssistantMessageComponent(
                    None,
                    self._hide_thinking_block,
                    self._get_markdown_theme_with_settings(),
                    self._hidden_thinking_label,
                    self._output_pad,
                    self._get_markdown_transformers(),
                )
                self._streaming_message = event.message
                self._chat_container.add_child(self._streaming_component)
                self._streaming_component.update_content(self._streaming_message, True)
                self.ui.request_render()

        elif event_type == "message_update":
            if self._streaming_component is not None and event.message.role == "assistant":
                self._streaming_message = event.message
                self._streaming_component.update_content(self._streaming_message, True)

                for content in self._streaming_message.content:
                    if content.type == "toolCall":
                        if content.id not in self._pending_tools:
                            component = ToolExecutionComponent(
                                content.name,
                                content.id,
                                content.arguments,
                                {
                                    "showImages": self.settings_manager.get_show_images(),
                                    "imageWidthCells": self.settings_manager.get_image_width_cells(),
                                    "outputPad": self._output_pad,
                                },
                                self._get_registered_tool_definition(content.name),
                                self.ui,
                                self.session_manager.get_cwd(),
                            )
                            component.set_expanded(self._tool_output_expanded)
                            self._chat_container.add_child(component)
                            self._pending_tools[content.id] = component
                        else:
                            component = self._pending_tools.get(content.id)
                            if component is not None:
                                component.update_args(content.arguments)
                self.ui.request_render()

        elif event_type == "message_end":
            if event.message.role == "user":
                return
            if self._streaming_component is not None and event.message.role == "assistant":
                self._streaming_message = event.message
                error_message = None
                if self._streaming_message.stop_reason == "aborted":
                    retry_attempt = self.session.retry_attempt
                    error_message = (
                        f"Aborted after {retry_attempt} retry attempt{'s' if retry_attempt > 1 else ''}"
                        if retry_attempt > 0
                        else "Operation aborted"
                    )
                    # Frozen messages (spec/concurrency.md): messages are frozen values,
                    # so the abort decoration is a display-only copy. pi mutates
                    # the shared message here, which also lands in the session
                    # file; the persisted message now keeps the provider's
                    # original error text (the rebuild path recomputes the
                    # decoration for tools either way, mirroring pi's).
                    self._streaming_message = dataclass_replace(self._streaming_message, error_message=error_message)
                self._streaming_component.update_content(self._streaming_message, False)

                if self._streaming_message.stop_reason in ("aborted", "error"):
                    if not error_message:
                        error_message = self._streaming_message.error_message or "Error"
                    for component in self._pending_tools.values():
                        component.update_result({"content": [{"type": "text", "text": error_message}], "isError": True})
                    self._pending_tools.clear()
                else:
                    # Args are now complete - trigger diff computation for
                    # edit tools
                    for component in self._pending_tools.values():
                        component.set_args_complete()
                    self._maybe_show_thinking_drop_notice(self._streaming_message)
                    self._maybe_show_cache_miss_notice(self._streaming_message)
                self._streaming_component = None
                self._streaming_message = None
                self._footer.invalidate()
            self.ui.request_render()

        elif event_type == "bash_execution_update":
            # The bash execution callback handles TUI output rendering.
            pass

        elif event_type == "tool_execution_start":
            # Nested calls (from other tools, e.g. codemode scripts) are shown inside their parent's row.
            if event.parent_tool_call_id:
                return
            component = self._pending_tools.get(event.tool_call_id)
            if component is None:
                component = ToolExecutionComponent(
                    event.tool_name,
                    event.tool_call_id,
                    event.args,
                    {
                        "showImages": self.settings_manager.get_show_images(),
                        "imageWidthCells": self.settings_manager.get_image_width_cells(),
                        "outputPad": self._output_pad,
                    },
                    self._get_registered_tool_definition(event.tool_name),
                    self.ui,
                    self.session_manager.get_cwd(),
                )
                component.set_expanded(self._tool_output_expanded)
                self._chat_container.add_child(component)
                self._pending_tools[event.tool_call_id] = component
            component.mark_execution_started()
            self.ui.request_render()

        elif event_type == "tool_execution_update":
            component = self._pending_tools.get(event.tool_call_id)
            if component is not None:
                partial = event.partial_result
                component.update_result(
                    {
                        "content": partial.content if partial is not None else [],
                        "details": getattr(partial, "details", None) if partial is not None else None,
                        "isError": False,
                    },
                    True,
                )
                self.ui.request_render()

        elif event_type == "tool_execution_end":
            component = self._pending_tools.get(event.tool_call_id)
            if component is not None:
                result = event.result
                component.update_result(
                    {
                        "content": result.content if result is not None else [],
                        "details": getattr(result, "details", None) if result is not None else None,
                        "isError": event.is_error,
                        "durationMs": event.duration_ms,
                    }
                )
                self._pending_tools.pop(event.tool_call_id, None)
                self.ui.request_render()

        elif event_type == "agent_end":
            if self.settings_manager.get_show_terminal_progress():
                self.ui.terminal.set_progress(False)
            self._clear_status_indicator("working")
            if self._streaming_component is not None:
                self._chat_container.remove_child(self._streaming_component)
                self._streaming_component = None
                self._streaming_message = None
            self._pending_tools.clear()

            self.ui.request_render()

        elif event_type == "agent_settled":
            self._spawn_flow(self._check_shutdown_requested())

        elif event_type == "compaction_start":
            if self.settings_manager.get_show_terminal_progress():
                self.ui.terminal.set_progress(True)
            # Keep editor active; submissions are queued during compaction.
            self._auto_compaction_escape_handler = self._default_editor.on_escape
            self._default_editor.on_escape = self.session.abort_compaction
            self._show_status_indicator(CompactionStatusIndicator(self.ui, event.reason))
            self.ui.request_render()

        elif event_type == "compaction_end":
            if self.settings_manager.get_show_terminal_progress():
                self.ui.terminal.set_progress(False)
            if self._auto_compaction_escape_handler is not None:
                self._default_editor.on_escape = self._auto_compaction_escape_handler
                self._auto_compaction_escape_handler = None
            self._clear_status_indicator("compaction")
            if event.aborted:
                if event.reason == "manual":
                    self.show_error("Compaction cancelled")
                else:
                    self.show_status("Auto-compaction cancelled")
            elif event.result is not None:
                entries = self.session_manager.build_context_entries()
                if not entries or entries[0].get("type") != "compaction":
                    raise Exception("Completed compaction is missing from the session context")
                self._chat_container.clear()
                # The latest compaction is prepended for model context; append it below at its chronological position.
                self._render_session_entries(entries[1:])
                self._add_message_to_chat(
                    create_compaction_summary_message(
                        event.result.summary,
                        event.result.tokens_before,
                        clock.now_iso(),
                    )
                )
                if event.result.usage:
                    self._add_compaction_cost_notice(
                        {"type": "compaction_cost", "kind": "compaction", "usage": event.result.usage}
                    )
                self._footer.invalidate()
            elif event.error_message:
                if event.reason == "manual":
                    self.show_error(event.error_message)
                else:
                    self._chat_container.add_child(Spacer(1))
                    error_message = event.error_message
                    self._chat_container.add_child(ThemedText(lambda: theme.fg("error", error_message), 1, 0))
            self._flush_compaction_queue({"willRetry": event.will_retry})
            self.ui.request_render()

        elif event_type == "auto_retry_start":
            # Set up escape to abort retry
            self._retry_escape_handler = self._default_editor.on_escape
            self._default_editor.on_escape = self.session.abort_retry
            self._show_status_indicator(
                RetryStatusIndicator(self.ui, event.attempt, event.max_attempts, event.delay_ms)
            )
            self.ui.request_render()

        elif event_type == "auto_retry_end":
            # Restore escape handler
            if self._retry_escape_handler is not None:
                self._default_editor.on_escape = self._retry_escape_handler
                self._retry_escape_handler = None
            self._clear_status_indicator("retry")
            # Show error only on final failure (success shows normal
            # response)
            if not event.success:
                self.show_error(f"Retry failed after {event.attempt} attempts: {event.final_error or 'Unknown error'}")
            self.ui.request_render()

        elif event_type == "summarization_retry_scheduled":
            self.show_error(event.error_message)
            self._show_status_indicator(
                RetryStatusIndicator(self.ui, event.attempt, event.max_attempts, event.delay_ms)
            )
            self.ui.request_render()

        elif event_type == "summarization_retry_attempt_start":
            self._clear_status_indicator("retry")
            if event.source == "branchSummary":
                self._show_status_indicator(BranchSummaryStatusIndicator(self.ui))
            else:
                self._show_status_indicator(CompactionStatusIndicator(self.ui, event.reason))
            self.ui.request_render()

        elif event_type == "summarization_retry_finished":
            self._clear_status_indicator("retry")
            self.ui.request_render()

    def _get_user_message_text(self, message) -> str:
        """Extract text content from a user message."""
        if message.role != "user":
            return ""
        if isinstance(message.content, str):
            return message.content
        return "".join(
            (c.get("text") if isinstance(c, dict) else getattr(c, "text", ""))
            for c in message.content
            if (c.get("type") if isinstance(c, dict) else getattr(c, "type", None)) == "text"
        )

    def _handle_startup_submit(self, text: str) -> None:
        # The editor's submit.
        self.editor.set_text(text)
        self.show_status("Startup is still in progress")

    def _show_managed_tool_status(self, status: dict) -> None:
        """Show a managed-tool status update in the chat."""

        with self.ui.state_lock:
            if not self._managed_tool_status_started:
                self._chat_container.add_child(Spacer(1))
                self._managed_tool_status_started = True
            message = f"Warning: {status['message']}" if status["type"] == "warning" else status["message"]
            color = "warning" if status["type"] == "warning" else "dim"
            self._chat_container.add_child(ThemedText(lambda: theme.fg(color, message), 1, 0))
            self._last_status_spacer = None
            self._last_status_text = None
            self.ui.request_render()

    def _append_to_chat(self, *components) -> None:
        """Append components to the chat, in order, in one hold."""
        with self.ui.state_lock:
            for component in components:
                self._chat_container.add_child(component)
            self.ui.request_render()

    def show_status(self, message: str) -> None:
        """Show a status message in the chat.

        Back-to-back status messages update the previous status line instead
        of appending new ones, to avoid log spam.
        """
        with self.ui.state_lock:
            children = self._chat_container.children
            last = children[-1] if children else None
            second_last = children[-2] if len(children) > 1 else None

            if (
                last is not None
                and second_last is not None
                and last is self._last_status_text
                and second_last is self._last_status_spacer
            ):
                self._last_status_message = message
                self._last_status_text.invalidate()
                self.ui.request_render()
                return

            spacer = Spacer(1)
            self._last_status_message = message
            text = ThemedText(lambda: theme.fg("dim", self._last_status_message), 1, 0)
            self._chat_container.add_child(spacer)
            self._chat_container.add_child(text)
            self._last_status_spacer = spacer
            self._last_status_text = text
            self.ui.request_render()

    def _add_custom_entry_to_chat(self, entry: dict) -> None:
        renderer = self.session.extension_runner.get_entry_renderer(entry.get("customType"))
        if renderer is None:
            return
        component = CustomEntryComponent(entry, renderer, self._output_pad)
        component.set_expanded(self._tool_output_expanded)
        if not component.has_content():
            return

        if self._streaming_component is not None:
            try:
                streaming_index = self._chat_container.children.index(self._streaming_component)
            except ValueError:
                streaming_index = -1
            if streaming_index >= 0:
                self._chat_container.children.insert(streaming_index, component)
                return

        self._chat_container.add_child(component)

    def _add_message_to_chat(self, message, options: dict | None = None) -> None:
        options = options or {}
        role = message.role
        if role == "bashExecution":
            component = BashExecutionComponent(message.command, self.ui, message.exclude_from_context, self._output_pad)
            if message.output:
                component.append_output(message.output)

            component.set_complete(
                message.exit_code,
                message.cancelled,
                SimpleNamespace(truncated=True) if message.truncated else None,
                message.full_output_path,
            )
            self._chat_container.add_child(component)
        elif role == "custom":
            if message.display:
                renderer = self.session.extension_runner.get_message_renderer(message.custom_type)
                component = CustomMessageComponent(
                    message, renderer, self._get_markdown_theme_with_settings(), self._output_pad
                )
                component.set_expanded(self._tool_output_expanded)
                self._chat_container.add_child(component)
        elif role == "compactionSummary":
            self._chat_container.add_child(Spacer(1))
            component = CompactionSummaryMessageComponent(
                message, self._get_markdown_theme_with_settings(), self._output_pad
            )
            component.set_expanded(self._tool_output_expanded)
            self._chat_container.add_child(component)
        elif role == "branchSummary":
            self._chat_container.add_child(Spacer(1))
            component = BranchSummaryMessageComponent(
                message, self._get_markdown_theme_with_settings(), self._output_pad
            )
            component.set_expanded(self._tool_output_expanded)
            self._chat_container.add_child(component)
        elif role == "system":
            pass
        elif role == "user":
            text_content = self._get_user_message_text(message)
            if text_content:
                if self._chat_container.children:
                    self._chat_container.add_child(Spacer(1))
                skill_block = parse_skill_block(text_content)
                if skill_block is not None:
                    # Render skill block (collapsible)
                    component = SkillInvocationMessageComponent(
                        skill_block, self._get_markdown_theme_with_settings(), self._output_pad
                    )
                    component.set_expanded(self._tool_output_expanded)
                    self._chat_container.add_child(component)
                    # Render user message separately if present
                    if skill_block.user_message:
                        self._chat_container.add_child(Spacer(1))
                        user_component = UserMessageComponent(
                            skill_block.user_message,
                            self._get_markdown_theme_with_settings(),
                            self._output_pad,
                            self._get_markdown_transformers(),
                        )
                        self._chat_container.add_child(user_component)
                else:
                    user_component = UserMessageComponent(
                        text_content,
                        self._get_markdown_theme_with_settings(),
                        self._output_pad,
                        self._get_markdown_transformers(),
                    )
                    self._chat_container.add_child(user_component)
                if options.get("populateHistory"):
                    self._apply_editor_history(text_content)
        elif role == "assistant":
            assistant_component = AssistantMessageComponent(
                message,
                self._hide_thinking_block,
                self._get_markdown_theme_with_settings(),
                self._hidden_thinking_label,
                self._output_pad,
                self._get_markdown_transformers(),
            )
            self._chat_container.add_child(assistant_component)
        elif role == "toolResult":
            # Tool results are rendered inline with tool calls, handled
            # separately
            pass

    def _render_session_items(self, items: list, options: dict | None = None) -> None:
        options = options or {}
        self._pending_tools.clear()
        rendered_pending_tools: dict = {}
        # Cache misses are not persisted, unlike successful cache-warming
        # usage. Re-derive them and inject them after the assistant messages
        # that paid for them.
        cache_misses = (
            collect_cache_misses(self.session_manager.get_entries(), self.session.model_runtime)
            if self.settings_manager.get_show_cache_miss_notices()
            else {}
        )

        if options.get("updateFooter"):
            self._footer.invalidate()
            self._update_editor_border_color()

        for item in items:
            if _is_custom_session_entry(item):
                self._add_custom_entry_to_chat(item)
                continue
            if _is_usage_session_entry(item):
                self._add_cache_warming_usage(item)
                continue
            if _is_compaction_cost_notice(item):
                self._add_compaction_cost_notice(item)
                continue

            message = item
            # Assistant messages need special handling for tool calls
            if message.role == "assistant":
                self._add_message_to_chat(message)
                # Render tool call components
                for content in message.content:
                    if content.type == "toolCall":
                        component = ToolExecutionComponent(
                            content.name,
                            content.id,
                            content.arguments,
                            {
                                "showImages": self.settings_manager.get_show_images(),
                                "imageWidthCells": self.settings_manager.get_image_width_cells(),
                                "outputPad": self._output_pad,
                            },
                            self._get_registered_tool_definition(content.name),
                            self.ui,
                            self.session_manager.get_cwd(),
                        )
                        component.set_expanded(self._tool_output_expanded)
                        self._chat_container.add_child(component)

                        if message.stop_reason in ("aborted", "error"):
                            if message.stop_reason == "aborted":
                                retry_attempt = self.session.retry_attempt
                                error_message = (
                                    f"Aborted after {retry_attempt} retry attempt{'s' if retry_attempt > 1 else ''}"
                                    if retry_attempt > 0
                                    else "Operation aborted"
                                )
                            else:
                                error_message = message.error_message or "Error"
                            component.update_result(
                                {"content": [{"type": "text", "text": error_message}], "isError": True}
                            )
                        else:
                            rendered_pending_tools[content.id] = component
                if message.stop_reason not in ("aborted", "error"):
                    # collect_cache_misses keys by id() of the assistant
                    # message (pi keys a Map by object reference)
                    miss = cache_misses.get(id(message))
                    if miss:
                        self._add_cache_miss_notice(miss)
            elif message.role == "toolResult":
                # Match tool results to pending tool components
                component = rendered_pending_tools.get(message.tool_call_id)
                if component is not None:
                    # pi passes the message itself, so its durationMs reaches the renderer after a reload.
                    component.update_result(
                        {
                            "content": message.content,
                            "details": message.details,
                            "isError": message.is_error,
                            "durationMs": message.duration_ms,
                        }
                    )
                    rendered_pending_tools.pop(message.tool_call_id, None)
            else:
                # All other messages use standard rendering
                self._add_message_to_chat(message, options)

        for tool_call_id, component in rendered_pending_tools.items():
            self._pending_tools[tool_call_id] = component
        self.ui.request_render()

    def _render_boundary_compaction(self, entry: dict) -> None:
        """Re-render the transcript around a compaction a boundary handler appended."""
        entries = self.session_manager.build_context_entries()
        if not entries or entries[0].get("id") != entry["id"]:
            return
        self._chat_container.clear()
        branch = self.session_manager.get_branch()
        compaction_index = next((i for i, candidate in enumerate(branch) if candidate["id"] == entry["id"]), -1)
        entries_after_compaction = {candidate["id"] for candidate in branch[compaction_index + 1 :]}
        retained_entries = entries[1:]
        self._render_session_entries(
            [candidate for candidate in retained_entries if candidate["id"] not in entries_after_compaction]
        )
        self._add_message_to_chat(
            create_compaction_summary_message(entry.get("summary"), entry.get("tokensBefore"), entry.get("timestamp"))
        )
        if entry.get("usage"):
            self._add_compaction_cost_notice({"type": "compaction_cost", "kind": "compaction", "usage": entry["usage"]})
        self._render_session_entries(
            [candidate for candidate in retained_entries if candidate["id"] in entries_after_compaction]
        )
        self._entries_rendered_by_boundary_compaction.update(entries_after_compaction)
        self._footer.invalidate()
        self.ui.request_render()

    def _render_session_entries(self, entries: list, options: dict | None = None) -> None:
        """Render session entries to chat (initial load and post-compaction).

        options: {"updateFooter"?, "populateHistory"?}
        """
        # Selection coordinates point into the transcript being replaced (pi #9311).
        if isinstance(self._renderer, TuiAltScreen):
            self._renderer.reset_text_selection()
        items: list = []
        for entry in entries:
            if entry.get("type") == "custom" or (entry.get("type") == "usage" and entry.get("kind") == "cache_warm"):
                items.append(entry)
                continue
            messages = session_entry_to_context_messages(entry)
            items.extend(messages)
            if entry.get("type") in ("compaction", "branch_summary") and entry.get("usage") and messages:
                items.append({"type": "compaction_cost", "kind": entry["type"], "usage": entry["usage"]})
        self._render_session_items(items, options)

    def _add_cache_warming_usage(self, entry: dict) -> None:
        if not self.settings_manager.get_show_cache_miss_notices():
            return
        self._chat_container.add_child(Spacer(1))
        usage = format_cache_warming_usage(entry)
        self._chat_container.add_child(ThemedText(lambda: theme.fg("dim", usage), 1, 0))

    def _add_compaction_cost_notice(self, notice: dict) -> None:
        """Render billing usage for a compaction or branch summary. The notice is derived
        from persisted summary usage and is not stored as a separate session entry."""
        if not self.settings_manager.get_show_cache_miss_notices():
            return

        usage = notice["usage"]
        tokens = usage.input + usage.output + usage.cache_read + usage.cache_write
        cost = f" (~${usage.cost.total:.2f})" if usage.cost.total >= 0.01 else ""
        label = "Compaction" if notice["kind"] == "compaction" else "Branch summary"
        self._chat_container.add_child(Spacer(1))
        self._chat_container.add_child(
            ThemedText(lambda: theme.fg("warning", f"{label}: {format_tokens(tokens)} tokens billed{cost}"), 1, 0)
        )

    @staticmethod
    def _count_dropped_thinking_blocks(message) -> int:
        count = 0
        for diagnostic in message.diagnostics or []:
            if diagnostic.type != "anthropic_input_transformations":
                continue
            transformations = (diagnostic.details or {}).get("transformations")
            if not isinstance(transformations, list):
                continue
            count += sum(
                1
                for transformation in transformations
                if isinstance(transformation, dict) and transformation.get("type") == "thinking_dropped"
            )
        return count

    def _maybe_show_thinking_drop_notice(self, message) -> None:
        if not self.settings_manager.get_show_cache_miss_notices():
            return

        dropped_count = InteractiveMode._count_dropped_thinking_blocks(message)
        if dropped_count == 0:
            return

        # Compare against the previous response on the branch. pi relies on message_end
        # reaching the UI before persistence; here the UI update can run after the session
        # persisted the message, so the current message is skipped by identity.
        previous_dropped_count = 0
        for entry in reversed(self.session_manager.get_branch()):
            if entry.get("type") != "message":
                continue
            entry_message = entry.get("message")
            if getattr(entry_message, "role", None) != "assistant" or entry_message is message:
                continue
            previous_dropped_count = InteractiveMode._count_dropped_thinking_blocks(entry_message)
            break
        if dropped_count <= previous_dropped_count:
            return

        noun = "thinking block" if dropped_count == 1 else "thinking blocks"
        self._chat_container.add_child(Spacer(1))
        self._chat_container.add_child(
            ThemedText(
                lambda: theme.fg("warning", f"Anthropic dropped {dropped_count} {noun} (details in session)"), 1, 0
            )
        )

    def _maybe_show_cache_miss_notice(self, message) -> None:
        """Show a transcript notice for a significant prompt-cache miss.

        Only states observable facts: the miss itself, a model switch, or an
        idle gap past the cache TTL.
        """
        if not self.settings_manager.get_show_cache_miss_notices():
            return

        # Entries don't contain `message` yet: message_end fires before
        # persistence.
        miss = detect_cache_miss(self.session_manager.get_entries(), message, self.session.model_runtime)
        if miss:
            self._add_cache_miss_notice(miss)

    def _add_cache_miss_notice(self, miss) -> None:
        if miss.missed_tokens < 20_000 and miss.missed_cost < 0.1:
            return

        cost = f" (~${miss.missed_cost:.2f})" if miss.missed_cost >= 0.01 else ""
        re_billed = f"{format_tokens(miss.missed_tokens)} tokens re-billed{cost}"
        label = "Cache miss"
        if miss.model_changed:
            label = "Cache miss after model switch"
        elif miss.idle_ms >= CACHE_TTL_MS:
            label = f"Cache miss after {round(miss.idle_ms / 60_000)}m idle"
        self._chat_container.add_child(Spacer(1))
        self._chat_container.add_child(ThemedText(lambda: theme.fg("warning", f"{label}: {re_billed}"), 1, 0))

    def _render_initial_messages(self, trust_warning: bool) -> None:
        """`trust_warning` is `_needs_project_trust_warning()`, awaited by the
        caller before this hold."""
        with self.ui.state_lock:
            entries = self.session_manager.build_context_entries()
            self._render_session_entries(entries, {"updateFooter": True, "populateHistory": True})
            if trust_warning:
                self._render_project_trust_warning()

            # Show compaction info if session was compacted
            all_entries = self.session_manager.get_entries()
            compaction_count = sum(1 for entry in all_entries if entry.get("type") == "compaction")
            if compaction_count > 0:
                times = "1 time" if compaction_count == 1 else f"{compaction_count} times"
                self.show_status(f"Session compacted {times}")

    def _rerender_initial_messages(self, trust_warning: bool) -> None:
        """Clear the chat and render the session again, in one hold: nothing
        lands between the clear and the render."""
        with self.ui.state_lock:
            self._chat_container.clear()
            self._render_initial_messages(trust_warning)

    async def _needs_project_trust_warning(self, session=None) -> bool:
        """Whether `session` (the current one by default) gets the
        project-trust warning. Its resource check is file I/O, so the render
        paths take the answer from here, awaited before their hold of the UI
        state lock."""
        session = session if session is not None else self.session
        if session.settings_manager.is_project_trusted():
            return False
        return await tonio.spawn_blocking(
            has_trust_requiring_project_resources_blocking, session.session_manager.get_cwd()
        )

    def _render_project_trust_warning(self) -> None:
        if self._chat_container.children:
            self._chat_container.add_child(Spacer(1))
        self._chat_container.add_child(
            ThemedText(
                lambda: theme.fg(
                    "warning",
                    f"This project is not trusted. Project {CONFIG_DIR_NAME} resources and packages "
                    f"are ignored. Use /trust to save a trust decision, then restart {APP_NAME}.",
                ),
                1,
                0,
            )
        )

    async def _get_user_input(self) -> str:
        received = tonio.Event()
        outcome: dict = {"text": ""}

        def on_input(text: str) -> None:
            # (The submit took the callback out under the guard.)
            outcome["text"] = text
            received.set()

        with self._user_input_guard:
            if self._pending_user_inputs:
                return self._pending_user_inputs.pop(0)
            self._on_input_callback = on_input
        await received.wait(None)
        return outcome["text"]

    def _rebuild_chat_from_messages(self) -> None:
        with self.ui.state_lock:
            self._chat_container.clear()
            self._render_session_entries(self.session_manager.build_context_entries())

    # =========================================================================
    # Key handlers
    # =========================================================================

    def _handle_ctrl_c(self) -> None:
        # The clear action.
        now = clock.monotonic() * 1000
        if now - self._last_sigint_time < 500:
            self._spawn_flow(self.shutdown())
        else:
            self._set_editor_text("")
            self._last_sigint_time = now

    def _handle_ctrl_d(self) -> None:
        # Only called when editor is empty (enforced by CustomEditor)
        self._spawn_flow(self.shutdown())

    async def shutdown(self, options: dict | None = None) -> None:
        """Gracefully shutdown the agent.

        Stops the TUI before emitting shutdown events so extension UI cleanup
        cannot repaint the final frame while the process is exiting.
        """
        with self._shutdown_guard:
            if self._is_shutting_down:
                return
            self._is_shutting_down = True
        # Keep signal handlers registered until terminal cleanup has completed
        # (pi's signal-exit re-send concern does not apply here, but the
        # watcher also guards on _is_shutting_down via this flag).

        # pi routes dead-terminal EIO from the restore writes through the
        # stdout/stderr "error" listeners; Python surfaces it as OSError at
        # the write site, so the restore sequence is guarded directly.
        if options and options.get("fromSignal"):
            # Signal-triggered shutdown (SIGTERM/SIGHUP). Emit extension
            # cleanup (session_shutdown) BEFORE touching the terminal.
            # Extension teardown such as removing sockets does not write to
            # the tty, so it must not be skipped if a later terminal-restore
            # write fails on a dead or stalled terminal (see pi #4144).
            await self.runtime_host.dispose()
            await self._theme_controller.disable_auto_sync()
            try:
                await self.ui.terminal.drain_input(1000)
                await self.stop()
            except OSError as error:
                if is_dead_terminal_error(error):
                    self._emergency_terminal_exit()
                raise
            # `hard_exit` skips everything still queued.
            await drain_output()
            hard_exit(0)

        # Interactive quit (Ctrl+D, Ctrl+C, /quit, extension shutdown()).
        # Stop the TUI before emitting shutdown events so extension UI cleanup
        # cannot repaint the final frame while the process is exiting.
        # Drain any in-flight Kitty key release events before stopping.
        # This prevents escape sequences from leaking to the parent shell over
        # slow SSH.
        await self._theme_controller.disable_auto_sync()
        try:
            await self.ui.terminal.drain_input(1000)
            await self.stop()
        except OSError as error:
            if is_dead_terminal_error(error):
                self._emergency_terminal_exit()
            raise
        await self.runtime_host.dispose()

        resume_command = await format_resume_command(self.session_manager)
        if resume_command:
            write_stdout(f"{dim('To resume this session:')} {resume_command}\n")
            # `hard_exit` skips everything still queued.
            await drain_output()

        hard_exit(0)

    def _emergency_terminal_exit(self) -> None:
        with self._shutdown_guard:
            self._is_shutting_down = True
        self._unregister_signal_handlers()
        kill_tracked_detached_children()
        # The terminal is gone. Do not run normal shutdown because TUI and
        # extension cleanup can write restore sequences and re-trigger EIO.
        hard_exit(129)

    async def _uncaught_crash(self, error: BaseException) -> None:
        """Last-resort handler for uncaught exceptions. The TUI puts stdin into
        raw mode and hides the cursor; without this handler, an uncaught throw
        tears down the process while leaving the terminal in raw mode with no
        cursor, requiring ``stty sane && reset`` to recover.

        Unlike _emergency_terminal_exit, the terminal is still alive here, so
        ui.stop() restores cooked mode, the cursor, and disables bracketed
        paste / Kitty / modifyOtherKeys sequences.

        Deviation: Node registers this for the process-wide uncaughtException
        event; Python has no equivalent hook with the runtime still usable, so
        the interactive run loop calls it from its own crash guard instead.
        """
        # A dead terminal is not a pidrei crash. Do not try to restore it.
        if is_dead_terminal_error(error):
            self._emergency_terminal_exit()
        with self._shutdown_guard:
            already_shutting_down = self._is_shutting_down
            self._is_shutting_down = True
        if already_shutting_down:
            hard_exit(1)
        with contextlib.suppress(Exception):
            self._unregister_signal_handlers()
        with contextlib.suppress(Exception):
            kill_tracked_detached_children()
        with contextlib.suppress(Exception):
            await self.ui.stop()
        # The report goes straight to stderr, after what the terminal holds.
        detach_terminal()
        with contextlib.suppress(Exception):
            await self.ui.close()
        write_stderr(f"{APP_NAME} exiting due to uncaught exception:\n")
        write_stderr("".join(traceback.format_exception(error)))
        await drain_output()
        hard_exit(1)

    def _request_shutdown(self) -> None:
        with self._shutdown_guard:
            self._shutdown_requested = True

    async def _check_shutdown_requested(self) -> None:
        """Check if shutdown was requested and perform shutdown if so."""
        with self._shutdown_guard:
            requested = self._shutdown_requested
        if not requested:
            return
        await self.shutdown()

    def _register_signal_handlers(self) -> None:
        """Turn SIGTERM/SIGHUP into a graceful shutdown.

        SIGHUP does not hard-exit: graceful shutdown emits session_shutdown
        first, then attempts terminal restore. A genuinely dead terminal
        surfaces as OSError(EIO) on the restore writes, which shutdown()
        converts into _emergency_terminal_exit (see pi #4144, #5080).

        Deviation: pi prepends process listeners and hooks stdout/stderr
        "error" events plus uncaughtException; here a tonio signal receiver
        feeds a background watcher, write errors surface at the write site,
        and the run loop's crash guard covers _uncaught_crash.
        """
        self._unregister_signal_handlers()
        with self._shutdown_guard:
            generation = self._signal_watch_generation

        async def watch_signals() -> None:
            try:
                with tonio_signals.signal_receiver(signal.SIGTERM, signal.SIGHUP) as receiver:
                    async for _sig in receiver:
                        with self._shutdown_guard:
                            superseded = self._signal_watch_generation != generation
                        if superseded:
                            return
                        kill_tracked_detached_children()
                        await self.shutdown({"fromSignal": True})
            except ValueError, RuntimeError:
                # Signals can only be watched from the main thread (tests).
                return

        tonio.spawn.without_tracking(watch_signals())

    def _unregister_signal_handlers(self) -> None:
        with self._shutdown_guard:
            self._signal_watch_generation += 1

    async def _handle_ctrl_z(self) -> None:
        """Suspend to background (Ctrl+Z): the suspend action's work, which
        the next key waits for.

        pi's synchronous handler: the SIGCONT handler is installed, the TUI
        stopped and SIGTSTP sent. The resume is its own task, as pi's
        `process.once("SIGCONT")` handler is: it restores the TUI when
        SIGCONT comes, and input handling never waits for it
        (spec/ui-island.md).

        Deviations: Python processes stay alive without pi's event-loop
        keep-alive timer, and SIGCONT is awaited through a tonio signal
        receiver instead of process.once. SIGINT is ignored while suspended
        so Ctrl+C in the terminal does not kill the backgrounded process.
        """
        try:
            previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        except ValueError:
            previous_sigint = None

        # Undone here if the suspend fails, else by the resume.
        with contextlib.ExitStack() as suspended:
            if previous_sigint is not None:
                suspended.callback(signal.signal, signal.SIGINT, previous_sigint)
            sigcont = suspended.enter_context(tonio_signals.signal_receiver(signal.SIGCONT))

            # Stop the TUI (restore terminal to normal mode)
            await self.ui.stop()

            # Send SIGTSTP to the process group (pid 0 means all
            # processes in the group)
            os.kill(0, signal.SIGTSTP)

            self._spawn_flow(self._resume_after_suspend(sigcont, suspended.pop_all()))

    async def _resume_after_suspend(self, sigcont, suspended: contextlib.ExitStack) -> None:
        with suspended:
            # Stopped until `fg`; SIGCONT wakes the receiver.
            async for _sig in sigcont:
                break
        await self.ui.start()
        self.ui.request_render(True)

    def _handle_follow_up(self) -> None:
        # The keybinding action: the editor is read and cleared in one step,
        # as pi's sync handler does (a later clear would land after keys typed
        # meanwhile, and wipe them); only the prompt goes to a detached task.
        get_expanded = getattr(self.editor, "get_expanded_text", None)
        text = (get_expanded() if get_expanded is not None else self.editor.get_text()).strip()
        if not text:
            return

        # Queue input during compaction (extension commands execute immediately)
        if self.session.is_compacting:
            if self._is_extension_command(text):
                self._apply_editor_history(text)
                self._set_editor_text("")
                self._spawn_flow(self.session.prompt(text))
            else:
                self._queue_compaction_message(text, "followUp")
            return

        # Alt+Enter queues a follow-up message (waits until agent finishes).
        # This handles extension commands (execute immediately), prompt
        # template expansion, and queueing
        if self.session.is_streaming:
            self._apply_editor_history(text)
            self._set_editor_text("")
            self._spawn_flow(self._queue_follow_up(text))
        # If not streaming, Alt+Enter acts like regular Enter (trigger on_submit)
        elif self.editor.on_submit:
            self._set_editor_text("")
            self.editor.on_submit(text)

    async def _queue_follow_up(self, text: str) -> None:
        await self.session.prompt(text, PromptOptions(streaming_behavior="followUp"))
        self._update_pending_messages_display()
        self.ui.request_render()

    def _handle_dequeue(self) -> None:
        # The dequeue action.
        restored = self._restore_queued_messages_to_editor()
        if restored == 0:
            self.show_status("No queued messages to restore")
        else:
            self.show_status(f"Restored {restored} queued message{'s' if restored > 1 else ''} to editor")

    def _on_theme_changed(self) -> None:
        """The theme controller's change hook."""
        self._update_editor_border_color()

    def _update_editor_border_color(self) -> None:
        with self.ui.state_lock:
            if self._is_bash_mode:
                self.editor.border_color = theme.get_bash_mode_border_color()
            else:
                level = self.session.thinking_level or "off"
                self.editor.border_color = theme.get_thinking_border_color(level)
            if self._active_status_indicator is not None:
                self._active_status_indicator.invalidate()
            self.ui.request_render()

    async def _cycle_thinking_level(self) -> None:
        # The thinking-cycle action's work, which the next key waits for (pi
        # persists synchronously): the UI updates apply in place.
        new_level = await self.session.cycle_thinking_level()
        if new_level is None:
            self.show_status("Current model does not support thinking")
        else:
            self._footer.invalidate()
            self._update_editor_border_color()
            self.show_status(f"Thinking level: {new_level}")

    async def _cycle_model(self, direction: str) -> None:
        try:
            result = await self.session.cycle_model(direction)
            with self.ui.state_lock:
                if result is None:
                    msg = "Only one model in scope" if self.session.scoped_models else "Only one model available"
                    self.show_status(msg)
                else:
                    self._footer.invalidate()
                    self._update_editor_border_color()
                    thinking_str = (
                        f" (thinking: {result.thinking_level})"
                        if result.model.reasoning and result.thinking_level != "off"
                        else ""
                    )
                    self.show_status(f"Switched to {result.model.name or result.model.id}{thinking_str}")
                    self._spawn_flow(self._maybe_warn_about_anthropic_subscription_auth(result.model))
        except Exception as error:
            self.show_error(str(error))

    def _toggle_tool_output_expansion(self) -> None:
        # The expand action and the extension selector's toggle.
        with self.ui.state_lock:
            self.set_tools_expanded(not self._tool_output_expanded)

    def set_tools_expanded(self, expanded: bool) -> None:
        # The flag, the children and the status change in one hold.
        with self.ui.state_lock:
            if expanded == self._tool_output_expanded:
                return
            self._tool_output_expanded = expanded
            active_header = self._custom_header if self._custom_header is not None else self._built_in_header
            if is_expandable(active_header):
                active_header.set_expanded(expanded)
            for container in (self._loaded_resources_container, self._chat_container):
                for child in container.children:
                    if is_expandable(child):
                        child.set_expanded(expanded)
            self.show_status(f"Tool output: {'expanded' if expanded else 'collapsed'}")

    def _update_thinking_block_visibility(self) -> None:
        """Update rendered assistant messages without rebuilding live tool components."""
        for child in self._chat_container.children:
            if isinstance(child, AssistantMessageComponent):
                child.set_hide_thinking_block(self._hide_thinking_block)
        self.ui.request_render()

    def _toggle_thinking_block_visibility(self) -> None:
        self._hide_thinking_block = not self._hide_thinking_block
        self.settings_manager.set_hide_thinking_block(self._hide_thinking_block)
        self._update_thinking_block_visibility()
        self.show_status(f"Thinking blocks: {'hidden' if self._hide_thinking_block else 'visible'}")

    async def _open_external_editor(self) -> None:
        """The external-editor action's work, which the next key waits for:
        the text is read and the TUI stopped here, as pi's handler does before
        its first await; the edit (the user's editor, then the restart) runs
        on its own task, so the agent keeps running meanwhile, as in pi
        (spec/ui-island.md)."""
        editor_cmd = self.settings_manager.get_external_editor_command()
        get_expanded = getattr(self.editor, "get_expanded_text", None)
        content = get_expanded() if get_expanded is not None else self.editor.get_text()
        await self.ui.stop()
        self._spawn_flow(self._edit_in_external_editor(editor_cmd, content))

    async def _edit_in_external_editor(self, editor_cmd, content: str) -> None:
        try:
            result = await edit_in_external_editor({"command": editor_cmd, "content": content})
            if result["status"] == "complete":
                self._set_editor_text(result["content"])
        finally:
            await self.ui.start()
            self.ui.request_render(True)

    # =========================================================================
    # UI helpers
    # =========================================================================

    def show_error(self, error_message: str) -> None:
        with self.ui.state_lock:
            self._chat_container.add_child(Spacer(1))
            self._chat_container.add_child(
                ThemedText(lambda: theme.fg("error", f"Error: {error_message}"), self._output_pad, 0)
            )
            self.ui.request_render()

    def show_warning(self, warning_message: str) -> None:
        with self.ui.state_lock:
            self._chat_container.add_child(Spacer(1))
            self._chat_container.add_child(ThemedText(lambda: theme.fg("warning", f"Warning: {warning_message}"), 1, 0))
            self.ui.request_render()

    def show_new_version_notification(self, release: dict) -> None:
        def update_instruction() -> str:
            return theme.fg("muted", f"New version {release['version']} is available. Run ") + theme.fg(
                "accent", f"{APP_NAME} update"
            )

        # The release's own page, from the same record the check produced.
        changelog_url = release.get("url") or RELEASES_URL

        def changelog_line() -> str:
            changelog_link = (
                hyperlink(theme.fg("accent", changelog_url), changelog_url)
                if get_capabilities()["hyperlinks"]
                else theme.fg("accent", changelog_url)
            )
            return theme.fg("muted", "Release notes: ") + changelog_link

        note = (release.get("note") or "").strip()

        with self.ui.state_lock:
            self._chat_container.add_child(Spacer(1))
            self._chat_container.add_child(DynamicBorder(lambda text: theme.fg("warning", text)))
            self._chat_container.add_child(
                ThemedText(
                    lambda: f"{theme.bold(theme.fg('warning', 'Update Available'))}\n{update_instruction()}", 1, 0
                )
            )
            if note:
                self._chat_container.add_child(Spacer(1))
                self._chat_container.add_child(
                    Markdown(
                        note,
                        1,
                        0,
                        self._get_markdown_theme_with_settings(),
                        {"color": lambda text: theme.fg("muted", text)},
                    )
                )
                self._chat_container.add_child(Spacer(1))
            self._chat_container.add_child(ThemedText(changelog_line, 1, 0))
            self._chat_container.add_child(DynamicBorder(lambda text: theme.fg("warning", text)))
            self.ui.request_render()

    def show_package_update_notification(self, packages: list) -> None:
        def update_instruction() -> str:
            return theme.fg("muted", "Package updates are available. Run ") + theme.fg(
                "accent", f"{APP_NAME} update --extensions"
            )

        package_lines = "\n".join(f"- {pkg}" for pkg in packages)

        with self.ui.state_lock:
            self._chat_container.add_child(Spacer(1))
            self._chat_container.add_child(DynamicBorder(lambda text: theme.fg("warning", text)))
            self._chat_container.add_child(
                ThemedText(
                    lambda: (
                        f"{theme.bold(theme.fg('warning', 'Package Updates Available'))}\n{update_instruction()}\n"
                        f"{theme.fg('muted', 'Packages:')}\n{package_lines}"
                    ),
                    1,
                    0,
                )
            )
            self._chat_container.add_child(DynamicBorder(lambda text: theme.fg("warning", text)))
            self.ui.request_render()

    def _get_all_queued_messages(self) -> dict:
        """Get all queued messages (read-only).

        Combines session queue and compaction queue.
        """
        with self._compaction_queue_guard:
            compaction_queue = list(self._compaction_queued_messages)
        return {
            "steering": [
                *self.session.get_steering_messages(),
                *(msg["text"] for msg in compaction_queue if msg["mode"] == "steer"),
            ],
            "followUp": [
                *self.session.get_follow_up_messages(),
                *(msg["text"] for msg in compaction_queue if msg["mode"] == "followUp"),
            ],
        }

    def _clear_all_queues(self) -> dict:
        """Clear all queued messages and return their contents.

        Clears both session queue and compaction queue.
        """
        cleared = self.session.clear_queue()
        with self._compaction_queue_guard:
            compaction_queue, self._compaction_queued_messages = self._compaction_queued_messages, []
        compaction_steering = [msg["text"] for msg in compaction_queue if msg["mode"] == "steer"]
        compaction_follow_up = [msg["text"] for msg in compaction_queue if msg["mode"] == "followUp"]
        return {
            "steering": [*cleared["steering"], *compaction_steering],
            "followUp": [*cleared["followUp"], *compaction_follow_up],
        }

    def _update_pending_messages_display(self) -> None:
        with self.ui.state_lock:
            self._pending_messages_container.clear()
            queued = self._get_all_queued_messages()
            steering_messages = queued["steering"]
            follow_up_messages = queued["followUp"]
            if steering_messages or follow_up_messages:
                self._pending_messages_container.add_child(Spacer(1))
                for message in steering_messages:
                    text = theme.fg("dim", f"Steering: {message}")
                    self._pending_messages_container.add_child(TruncatedText(text, 1, 0))
                for message in follow_up_messages:
                    text = theme.fg("dim", f"Follow-up: {message}")
                    self._pending_messages_container.add_child(TruncatedText(text, 1, 0))
                dequeue_hint = self._get_app_key_display("app.message.dequeue")
                hint_text = theme.fg("dim", f"↳ {dequeue_hint} to edit all queued messages")
                self._pending_messages_container.add_child(TruncatedText(hint_text, 1, 0))

    def _restore_queued_messages_to_editor(self, options: dict | None = None) -> int:
        """options: {"abort"?, "currentText"?}

        The queues move into the editor and the pending display updates in one
        hold (callers include extension `ctx.abort()` on its own task)."""
        with self.ui.state_lock:
            cleared = self._clear_all_queues()
            all_queued = [*cleared["steering"], *cleared["followUp"]]
            if all_queued:
                queued_text = "\n\n".join(all_queued)
                explicit_text = options.get("currentText") if options else None
                current_text = explicit_text if explicit_text is not None else self.editor.get_text()
                combined_text = "\n\n".join(t for t in (queued_text, current_text) if t.strip())
                self.editor.set_text(combined_text)
                self.ui.request_render()
            self._update_pending_messages_display()
        if options and options.get("abort"):
            # pi: `void this.session.abort()` — the cancel lands synchronously
            # (an extension's `ctx.abort()` routes here, regression #8935);
            # only the idle wait is discarded.
            self.session._request_abort()
        return len(all_queued)

    def _queue_compaction_message(self, text: str, mode: str) -> None:
        """From the submit handler and the follow-up action: the editor is
        cleared right away rather than behind queued keys. The queue itself is
        guarded (the flush takes it from any task)."""
        with self._compaction_queue_guard:
            self._compaction_queued_messages.append({"text": text, "mode": mode})
        self._apply_editor_history(text)
        self._set_editor_text("")
        self._show_compaction_queue()

    def _show_compaction_queue(self) -> None:
        """From `_queue_compaction_message`."""
        self._update_pending_messages_display()
        self.show_status("Queued message for after compaction")

    def _is_extension_command(self, text: str) -> bool:
        if not text.startswith("/"):
            return False

        extension_runner = self.session.extension_runner

        space_index = text.find(" ")
        command_name = text[1:] if space_index == -1 else text[1:space_index]
        return extension_runner.get_command(command_name) is not None

    def _flush_compaction_queue(self, options: dict | None = None) -> None:
        """pi's part before its first await (take the queued messages, refresh
        the pending display) runs here; sending them is spawned."""
        with self.ui.state_lock:
            with self._compaction_queue_guard:
                queued_messages, self._compaction_queued_messages = self._compaction_queued_messages, []
            if not queued_messages:
                return
            self._update_pending_messages_display()
        self._spawn_flow(self._send_compaction_queue(queued_messages, options))

    async def _send_compaction_queue(self, queued_messages: list, options: dict | None) -> None:
        def restore_queue(error) -> None:
            with self.ui.state_lock:
                self.session.clear_queue()
                with self._compaction_queue_guard:
                    # Ahead of anything queued since the flush took the list (pi
                    # restores the list as it was; nothing else could have queued).
                    self._compaction_queued_messages = [*queued_messages, *self._compaction_queued_messages]
                self._update_pending_messages_display()
                self.show_error(f"Failed to send queued message{'s' if len(queued_messages) > 1 else ''}: {error}")

        try:
            if options and options.get("willRetry"):
                # When retry is pending, queue messages for the retry turn
                for message in queued_messages:
                    if self._is_extension_command(message["text"]):
                        await self.session.prompt(message["text"])
                    elif message["mode"] == "followUp":
                        await self.session.follow_up(message["text"])
                    else:
                        await self.session.steer(message["text"])
                self._update_pending_messages_display()
                return

            # Find first non-extension-command message to use as prompt
            first_prompt_index = next(
                (
                    index
                    for index, message in enumerate(queued_messages)
                    if not self._is_extension_command(message["text"])
                ),
                -1,
            )
            if first_prompt_index == -1:
                # All extension commands - execute them all
                for message in queued_messages:
                    await self.session.prompt(message["text"])
                return

            # Execute any extension commands before the first prompt
            pre_commands = queued_messages[:first_prompt_index]
            first_prompt = queued_messages[first_prompt_index]
            rest = queued_messages[first_prompt_index + 1 :]

            for message in pre_commands:
                await self.session.prompt(message["text"])

            # Start a prompt when idle, or queue it into a run still
            # finishing compaction.
            async def send_first_prompt() -> None:
                try:
                    await self.session.prompt(
                        first_prompt["text"], PromptOptions(streaming_behavior=first_prompt["mode"])
                    )
                except Exception as error:
                    restore_queue(error)

            self._spawn_flow(send_first_prompt())

            # Queue remaining messages
            for message in rest:
                if self._is_extension_command(message["text"]):
                    await self.session.prompt(message["text"])
                elif message["mode"] == "followUp":
                    await self.session.follow_up(message["text"])
                else:
                    await self.session.steer(message["text"])
            self._update_pending_messages_display()
        except Exception as error:
            restore_queue(error)

    def _flush_pending_bash_components(self) -> None:
        """Move pending bash components from pending area to chat; under the UI
        state lock (the submit handler)."""
        for component in self._pending_bash_components:
            self._pending_messages_container.remove_child(component)
            self._chat_container.add_child(component)
        self._pending_bash_components = []

    # =========================================================================
    # Selectors
    # =========================================================================

    def _dispose_active_selector(self) -> None:
        with self.ui.state_lock:
            dispose = self._active_selector_dispose
            self._active_selector_token = None
            self._active_selector_dispose = None
            if dispose is not None:
                dispose()

    def _show_selector(self, create) -> None:
        """Show a selector component in place of the editor.

        ``create`` receives a ``done`` callback and returns a
        ``{"component", "focus"}`` record with an optional ``"dispose"``.

        Like pi's (sync) showSelector, the selector is mounted before this
        returns (from input handling: before the next key). ``done`` restores
        the editor right away; both run under the UI state lock, from any task.
        """
        token = object()

        def done() -> None:
            with self.ui.state_lock:
                if dispose is not None:
                    dispose()
                if self._active_selector_token is not token:
                    return
                self._active_selector_token = None
                self._active_selector_dispose = None
                self._editor_container.clear()
                self._editor_container.add_child(self.editor)
                self.ui.set_focus(self.editor)
                self.ui.request_render()

        with self.ui.state_lock:
            created = create(done)
            dispose = created.get("dispose")
            self._dispose_active_selector()
            self._active_selector_token = token
            self._active_selector_dispose = dispose
            self._editor_container.clear()
            self._editor_container.add_child(created["component"])
            self.ui.set_focus(created["focus"])
            self.ui.request_render()

    async def _show_settings_selector(self) -> None:
        # Resolved before `create` runs: listing themes reads the custom-theme
        # directory, and `create` is a sync factory.
        available_themes = await get_available_themes()

        def create(done):
            default_provider = self.settings_manager.get_default_provider()
            default_model_id = self.settings_manager.get_default_model()
            default_model = (
                f"{default_provider}/{default_model_id}" if default_provider and default_model_id else "not set"
            )

            def on_auto_compact_change(enabled: bool) -> None:
                self.session.set_auto_compaction_enabled(enabled)
                self._footer.set_auto_compact_enabled(enabled)

            def on_show_images_change(enabled: bool) -> None:
                self.settings_manager.set_show_images(enabled)
                for child in self._chat_container.children:
                    if isinstance(child, ToolExecutionComponent):
                        child.set_show_images(enabled)

            def on_image_width_cells_change(width: int) -> None:
                self.settings_manager.set_image_width_cells(width)
                for child in self._chat_container.children:
                    if isinstance(child, ToolExecutionComponent):
                        child.set_image_width_cells(width)

            def on_enable_skill_commands_change(enabled: bool) -> None:
                self.settings_manager.set_enable_skill_commands(enabled)
                self._setup_autocomplete_provider()

            def on_transport_change(transport: str) -> None:
                self.settings_manager.set_transport(transport)
                self.session.agent.transport = transport

            def on_http_idle_timeout_ms_change(timeout_ms: int) -> None:
                # The undici-dispatcher reconfiguration has no pidrei
                # counterpart (HTTP transport is punkreq's concern; see
                # core/http_config.py).
                self.settings_manager.set_http_idle_timeout_ms(timeout_ms)
                self.show_status(f"HTTP idle timeout: {format_http_idle_timeout_ms(timeout_ms)}")

            def on_cache_warming_mode_change(mode: str) -> None:
                self.session.set_cache_warming_mode(mode)
                self.show_status(f"Cache warming: {mode}")

            async def set_session_thinking_level(level: str) -> None:
                # The next key waits for this (pi's setThinkingLevel is
                # synchronous), so the UI updates apply in place.
                await self.session.set_thinking_level(level)
                self._footer.invalidate()
                self._update_editor_border_color()

            def on_model_thinking_level_change(provider: str, model_id: str, level: str) -> None:
                self.settings_manager.set_model_thinking_level(provider, model_id, level)
                # If the override is for the current model, apply it to the session too
                current = self.session.model
                if current is not None and current.provider == provider and current.id == model_id:
                    self._finish_before_next_input(set_session_thinking_level(level))

            def on_model_thinking_level_remove(provider: str, model_id: str) -> None:
                self.settings_manager.remove_model_thinking_level(provider, model_id)
                # If the override was for the current model, revert to global default
                current = self.session.model
                if current is not None and current.provider == provider and current.id == model_id:
                    global_default = self.settings_manager.get_default_thinking_level() or DEFAULT_THINKING_LEVEL
                    self._finish_before_next_input(set_session_thinking_level(global_default))

            def on_theme_change(theme_setting: str) -> None:
                self.settings_manager.set_theme(theme_setting)
                self._spawn_flow(self._theme_controller.set_theme_setting(theme_setting))

            def on_hide_thinking_block_change(hidden: bool) -> None:
                self._hide_thinking_block = hidden
                self.settings_manager.set_hide_thinking_block(hidden)
                self._update_thinking_block_visibility()

            def on_show_cache_miss_notices_change(shown: bool) -> None:
                self.settings_manager.set_show_cache_miss_notices(shown)
                self._rebuild_chat_from_messages()

            def on_show_hardware_cursor_change(enabled: bool) -> None:
                self.settings_manager.set_show_hardware_cursor(enabled)
                self.ui.set_show_hardware_cursor(enabled)

            def on_editor_padding_x_change(padding: int) -> None:
                self.settings_manager.set_editor_padding_x(padding)
                self._default_editor.set_padding_x(padding)
                if self.editor is not self._default_editor:
                    set_padding = getattr(self.editor, "set_padding_x", None)
                    if set_padding is not None:
                        set_padding(padding)

            def on_output_pad_change(padding: int) -> None:
                self.settings_manager.set_output_pad(padding)
                self._output_pad = padding
                for container in (self._chat_container, self._pending_messages_container):
                    for child in container.children:
                        if isinstance(child, _OutputPadded):
                            child.set_output_pad(padding)
                self.ui.request_render()

            def on_autocomplete_max_visible_change(max_visible: int) -> None:
                self.settings_manager.set_autocomplete_max_visible(max_visible)
                self._default_editor.set_autocomplete_max_visible(max_visible)
                if self.editor is not self._default_editor:
                    set_max_visible = getattr(self.editor, "set_autocomplete_max_visible", None)
                    if set_max_visible is not None:
                        set_max_visible(max_visible)

            def on_clear_on_shrink_change(enabled: bool) -> None:
                self.settings_manager.set_clear_on_shrink(enabled)
                self.ui.set_clear_on_shrink(enabled)
                if not enabled and self._active_status_indicator is None:
                    self._status_container.clear()

            def on_fullscreen_scrollbar_change(mode: str) -> None:
                self.settings_manager.set_fullscreen_scrollbar(mode)
                self._apply_fullscreen_scrollbar_setting()

            def on_fullscreen_copy_on_select_change(enabled: bool) -> None:
                self.settings_manager.set_fullscreen_copy_on_select(enabled)
                if isinstance(self._renderer, TuiAltScreen):
                    self._renderer.set_copy_on_select(enabled)

            def on_fullscreen_wheel_scroll_lines_change(lines) -> None:
                self.settings_manager.set_fullscreen_wheel_scroll_lines(lines)
                if isinstance(self._renderer, TuiAltScreen):
                    self._renderer.set_wheel_scroll_lines(lines)

            def on_cancel() -> None:
                done()
                self.ui.request_render()

            selector = SettingsSelectorComponent(
                {
                    "autoCompact": self.session.auto_compaction_enabled,
                    "defaultModel": default_model,
                    "currentModel": self.session.model,
                    "availableDefaultModels": self.session.model_runtime.get_available_snapshot(),
                    "showImages": self.settings_manager.get_show_images(),
                    "imageWidthCells": self.settings_manager.get_image_width_cells(),
                    "autoResizeImages": self.settings_manager.get_image_auto_resize(),
                    "blockImages": self.settings_manager.get_block_images(),
                    "enableSkillCommands": self.settings_manager.get_enable_skill_commands(),
                    "steeringMode": self.session.steering_mode,
                    "followUpMode": self.session.follow_up_mode,
                    "transport": self.settings_manager.get_transport(),
                    "httpIdleTimeoutMs": self.settings_manager.get_http_idle_timeout_ms(),
                    "cacheWarmingMode": self.settings_manager.get_cache_warming_mode(),
                    "thinkingLevel": self.settings_manager.get_default_thinking_level() or DEFAULT_THINKING_LEVEL,
                    "availableThinkingLevels": list(THINKING_LEVEL_OPTIONS),
                    "modelThinkingLevels": self.settings_manager.get_all_model_thinking_levels(),
                    "currentTheme": self._theme_controller.get_theme_selection() or SYSTEM_THEME_NAME,
                    "terminalTheme": self._theme_controller.get_terminal_theme(),
                    "availableThemes": available_themes,
                    "hideThinkingBlock": self._hide_thinking_block,
                    "collapseChangelog": self.settings_manager.get_collapse_changelog(),
                    "enableProviderAttribution": self.settings_manager.get_enable_provider_attribution(),
                    "doubleEscapeAction": self.settings_manager.get_double_escape_action(),
                    "treeFilterMode": self.settings_manager.get_tree_filter_mode(),
                    "showHardwareCursor": self.settings_manager.get_show_hardware_cursor(),
                    "showCacheMissNotices": self.settings_manager.get_show_cache_miss_notices(),
                    "defaultProjectTrust": self.settings_manager.get_default_project_trust(),
                    "editorPaddingX": self.settings_manager.get_editor_padding_x(),
                    "outputPad": self.settings_manager.get_output_pad(),
                    "autocompleteMaxVisible": self.settings_manager.get_autocomplete_max_visible(),
                    "quietStartup": self.settings_manager.get_quiet_startup(),
                    "clearOnShrink": self.settings_manager.get_clear_on_shrink(),
                    "showTerminalProgress": self.settings_manager.get_show_terminal_progress(),
                    "tuiMode": self._renderer.mode,
                    "fullscreenExitOutput": self.settings_manager.get_fullscreen_exit_output(),
                    "fullscreenScrollbar": self.settings_manager.get_fullscreen_scrollbar(),
                    "fullscreenCopyOnSelect": self.settings_manager.get_fullscreen_copy_on_select(),
                    "fullscreenWheelScrollLines": self.settings_manager.get_fullscreen_wheel_scroll_lines(),
                    "warnings": self.settings_manager.get_warnings(),
                },
                {
                    "onAutoCompactChange": on_auto_compact_change,
                    "onShowImagesChange": on_show_images_change,
                    "onImageWidthCellsChange": on_image_width_cells_change,
                    "onAutoResizeImagesChange": lambda enabled: self.settings_manager.set_image_auto_resize(enabled),
                    "onBlockImagesChange": lambda blocked: self.settings_manager.set_block_images(blocked),
                    "onEnableSkillCommandsChange": on_enable_skill_commands_change,
                    "onSteeringModeChange": lambda mode: self.session.set_steering_mode(mode),
                    "onFollowUpModeChange": lambda mode: self.session.set_follow_up_mode(mode),
                    "onTransportChange": on_transport_change,
                    "onHttpIdleTimeoutMsChange": on_http_idle_timeout_ms_change,
                    "onCacheWarmingModeChange": on_cache_warming_mode_change,
                    "onModelThinkingLevelChange": on_model_thinking_level_change,
                    "onModelThinkingLevelRemove": on_model_thinking_level_remove,
                    "onThemeChange": on_theme_change,
                    "onThemePreview": lambda theme_name: self._finish_before_next_input(
                        self._theme_controller.preview(theme_name)
                    ),
                    "onHideThinkingBlockChange": on_hide_thinking_block_change,
                    "onShowCacheMissNoticesChange": on_show_cache_miss_notices_change,
                    "onCollapseChangelogChange": lambda collapsed: self.settings_manager.set_collapse_changelog(
                        collapsed
                    ),
                    "onEnableProviderAttributionChange": lambda enabled: (
                        self.settings_manager.set_enable_provider_attribution(enabled)
                    ),
                    "onQuietStartupChange": lambda quiet: self.settings_manager.set_quiet_startup(quiet),
                    "onDefaultProjectTrustChange": lambda default_project_trust: (
                        self.settings_manager.set_default_project_trust(default_project_trust)
                    ),
                    "onDoubleEscapeActionChange": lambda action: self.settings_manager.set_double_escape_action(action),
                    "onTreeFilterModeChange": lambda mode: self.settings_manager.set_tree_filter_mode(mode),
                    "onShowHardwareCursorChange": on_show_hardware_cursor_change,
                    "onEditorPaddingXChange": on_editor_padding_x_change,
                    "onOutputPadChange": on_output_pad_change,
                    "onAutocompleteMaxVisibleChange": on_autocomplete_max_visible_change,
                    "onClearOnShrinkChange": on_clear_on_shrink_change,
                    "onShowTerminalProgressChange": lambda enabled: self.settings_manager.set_show_terminal_progress(
                        enabled
                    ),
                    "onTuiModeChange": lambda mode: self._on_settings_tui_mode_change(mode, selector),
                    "onFullscreenExitOutputChange": lambda output: self.settings_manager.set_fullscreen_exit_output(
                        output
                    ),
                    "onFullscreenScrollbarChange": on_fullscreen_scrollbar_change,
                    "onFullscreenCopyOnSelectChange": on_fullscreen_copy_on_select_change,
                    "onFullscreenWheelScrollLinesChange": on_fullscreen_wheel_scroll_lines_change,
                    "onWarningsChange": lambda warnings: self.settings_manager.set_warnings(warnings),
                    "onCancel": on_cancel,
                },
            )
            return {"component": selector, "focus": selector.get_settings_list()}

        self._show_selector(create)

    def _on_settings_tui_mode_change(self, mode: str, selector) -> None:
        # From the settings input. pi switches in place; here the switch is
        # the key's completion, so the next key goes to the new renderer
        # (spec/ui-island.md).
        self._finish_before_next_input(self._switch_tui_mode_from_settings(mode, selector))

    async def _switch_tui_mode_from_settings(self, mode: str, selector) -> None:
        switched = await self._switch_tui_mode(mode)
        if not switched:
            with self.ui.state_lock:
                selector.get_settings_list().update_value("tui-mode", self._renderer.mode)
                self.show_status("Close active overlays before changing TUI mode")
            return
        self.settings_manager.set_tui_mode(mode)
        with self.ui.state_lock:
            if self._active_status_indicator is None:
                self._status_container.clear()
            self.show_status(f"TUI mode: {mode}")

    def _handle_thinking_command(self, search_term: str | None = None) -> None:
        """From the submit handler: the selector or the error is shown here;
        setting a level is spawned."""
        available_levels = self.session.get_available_thinking_levels()
        if not search_term:
            self._show_thinking_selector()
            return

        normalized = search_term.strip().lower()
        level = next((candidate for candidate in available_levels if candidate.lower() == normalized), None)
        if level is None:
            self.show_error(f'Unknown thinking level "{search_term}". Available levels: {", ".join(available_levels)}.')
            return

        self._spawn_flow(self._select_thinking_level(level, False))

    async def _select_thinking_level(self, level: str, persist: bool) -> None:
        try:
            await self.session.set_thinking_level(level, persist=persist)
            message = f"Default thinking level: {level}" if persist else f"Thinking level: {level}"
            with self.ui.state_lock:
                self._footer.invalidate()
                self._update_editor_border_color()
                self.show_status(message)
        except Exception as error:
            self.show_error(str(error))

    def _show_thinking_selector(self) -> None:
        def create(done):
            # From the selector's input handling: pi's selectThinkingLevel is
            # synchronous, so the next key waits for the level and `done()`.
            def select_level(level: str, persist: bool) -> None:
                async def select() -> None:
                    await self._select_thinking_level(level, persist)
                    done()

                self._finish_before_next_input(select())

            def on_cancel() -> None:
                done()
                self.ui.request_render()

            selector = ThinkingSelectorComponent(
                self.session.thinking_level or DEFAULT_THINKING_LEVEL,
                self.session.get_available_thinking_levels(),
                lambda level: select_level(level, False),
                on_cancel,
                lambda level: select_level(level, True),
                self.settings_manager.get_default_thinking_level() or DEFAULT_THINKING_LEVEL,
            )
            return {"component": selector, "focus": selector}

        self._show_selector(create)

    def _handle_model_command(self, search_term: str | None = None) -> None:
        """From the submit handler: no search term shows the selector here
        (pi's part before its first await); a search is spawned."""
        if not search_term:
            self._show_model_selector()
            return
        self._spawn_flow(self._select_model_by_search(search_term))

    async def _select_model_by_search(self, search_term: str) -> None:
        model = await self._find_exact_model_match(search_term)
        if model is not None:
            try:
                await self.session.set_model(model, persist=False)
                with self.ui.state_lock:
                    self._footer.invalidate()
                    self._update_editor_border_color()
                    self.show_status(f"Model: {model.id}")
                self._spawn_flow(self._maybe_warn_about_anthropic_subscription_auth(model))
            except Exception as error:
                self.show_error(str(error))
            return

        self._show_model_selector(search_term)

    async def _find_exact_model_match(self, search_term: str):
        cached_models = (
            [scoped.model for scoped in self.session.scoped_models]
            if self.session.scoped_models
            else list(self.session.model_runtime.get_available_snapshot())
        )
        cached_match = find_exact_model_reference_match(search_term, cached_models)
        if cached_match is not None or self.session.scoped_models:
            return cached_match

        self.show_status("Refreshing model catalogs…")
        timeout = _TimeoutCancel(15_000)
        try:
            result = await refresh_model_catalogs(self.session.model_runtime, timeout.token)
            if result.aborted and timeout.timed_out:
                self.show_warning("Model refresh timed out; searching cached models.")
            elif result.errors:
                self.show_warning(f"Could not refresh {', '.join(result.errors)}; searching cached models.")
        except Exception as error:
            self.show_warning(
                "Model refresh timed out; searching cached models."
                if timeout.timed_out
                else f"Could not refresh model catalogs: {error}"
            )
        return find_exact_model_reference_match(search_term, list(self.session.model_runtime.get_available_snapshot()))

    def _update_available_provider_count(self) -> None:
        """Update the footer's available provider count from the current
        snapshot without refreshing catalogs."""
        models = (
            [scoped.model for scoped in self.session.scoped_models]
            if self.session.scoped_models
            else self.session.model_runtime.get_available_snapshot()
        )
        unique_providers = {model.provider for model in models}
        self._footer_data_provider.set_available_provider_count(len(unique_providers))

    def _show_anthropic_subscription_warning_once(self) -> None:
        # The checks run detached and can overlap (startup, a model switch):
        # the first to get here claims the warning.
        with self._anthropic_subscription_warning_guard:
            if self._anthropic_subscription_warning_shown:
                return
            self._anthropic_subscription_warning_shown = True
        self.show_warning(ANTHROPIC_SUBSCRIPTION_AUTH_WARNING)

    async def _maybe_warn_about_anthropic_subscription_auth(self, model=None) -> None:
        if model is None:
            model = self.session.model
        if self.settings_manager.get_warnings().get("anthropicExtraUsage") is False:
            return
        if self._anthropic_subscription_warning_shown:
            return
        if model is None or model.provider != "anthropic":
            return

        try:
            auth_check = await self.session.model_runtime.check_auth("anthropic")
            if auth_check is not None and auth_check.type == "oauth":
                self._show_anthropic_subscription_warning_once()
                return
            auth_result = await self.session.model_runtime.get_auth(model.provider)
            api_key = auth_result.auth.api_key if auth_result is not None else None
            if not is_anthropic_subscription_auth_key(api_key):
                return
            self._show_anthropic_subscription_warning_once()
        except Exception:
            # Ignore auth lookup failures for warning-only checks.
            return

    async def _maybe_save_implicit_project_trust_after_reload(self) -> tuple[bool, str | None]:
        """Whether project trust was saved, and the warning to show if saving
        failed. The caller shows it where pi does (after the resource
        listing): this I/O runs ahead of that block (§4.5b)."""
        cwd = self.session_manager.get_cwd()
        if self._auto_trust_on_reload_cwd != cwd:
            return False, None
        if not self.settings_manager.is_project_trusted():
            return False, None
        if not await tonio.spawn_blocking(has_trust_requiring_project_resources_blocking, cwd):
            return False, None

        trust_store = ProjectTrustStore(self.runtime_host.services.agent_dir)
        try:
            if await trust_store.get(cwd) is not None:
                self._auto_trust_on_reload_cwd = None
                return False, None
            await trust_store.set(cwd, True)
            self._auto_trust_on_reload_cwd = None
            return True, None
        except Exception as error:
            return False, f"Could not save project trust after reload: {error}"

    async def _show_trust_selector(self) -> None:
        cwd = self.session_manager.get_cwd()
        trust_store = ProjectTrustStore(self.runtime_host.services.agent_dir)
        saved_decision = await trust_store.get_entry(cwd)
        trust_options = await tonio.spawn_blocking(get_project_trust_options_blocking, cwd)

        def create(done):
            # From the selector's input handling: pi saves synchronously, so
            # the next key waits for the save, `done()` and the status.
            def on_select(selection: dict) -> None:
                async def save() -> None:
                    await trust_store.set_many(selection["updates"])
                    done()
                    self.show_status(
                        f"Saved trust decision: {'trusted' if selection['trusted'] else 'untrusted'}. "
                        f"Restart {APP_NAME} for this to take effect."
                    )

                self._finish_before_next_input(save())

            def on_cancel() -> None:
                done()
                self.ui.request_render()

            selector = TrustSelectorComponent(
                {
                    "cwd": cwd,
                    "trustOptions": trust_options,
                    "savedDecision": saved_decision,
                    "projectTrusted": self.settings_manager.is_project_trusted(),
                    "onSelect": on_select,
                    "onCancel": on_cancel,
                }
            )
            return {"component": selector, "focus": selector}

        self._show_selector(create)

    def _show_model_selector(self, initial_search_input: str | None = None) -> None:
        def create(done):
            # Spawned from the selector's callbacks; after the await, the UI
            # changes are one hold.
            async def select_model(model, persist: bool) -> None:
                try:
                    await self.session.set_model(model, persist=persist)
                    with self.ui.state_lock:
                        self._update_available_provider_count()
                        self._footer.invalidate()
                        self._update_editor_border_color()
                        done()
                        self.show_status(
                            f"Default model: {model.provider}/{model.id}" if persist else f"Model: {model.id}"
                        )
                    self._spawn_flow(self._maybe_warn_about_anthropic_subscription_auth(model))
                except Exception as error:
                    with self.ui.state_lock:
                        done()
                        self.show_error(str(error))

            def on_cancel() -> None:
                done()
                self.ui.request_render()

            default_provider = self.settings_manager.get_default_provider()
            default_model = self.settings_manager.get_default_model()
            selector = ModelSelectorComponent(
                self.ui,
                self.session.model,
                self.session.model_runtime,
                self.session.scoped_models,
                lambda model: self._spawn_flow(select_model(model, False)),
                on_cancel,
                initial_search_input,
                lambda model: self._spawn_flow(select_model(model, True)),
                {"provider": default_provider, "id": default_model} if default_provider and default_model else None,
            )
            return {"component": selector, "focus": selector, "dispose": selector.dispose}

        self._show_selector(create)

    def _show_models_selector(self) -> None:
        state = {
            "availableModels": list(self.session.model_runtime.get_available_snapshot()),
            "selectionChanged": False,
        }
        state["availableModelIds"] = {f"{model.provider}/{model.id}" for model in state["availableModels"]}
        configured_patterns = self.settings_manager.get_enabled_models()
        session_scoped_models = self.session.scoped_models

        def configured_enabled_ids(models) -> list | None:
            if not configured_patterns:
                return None
            resolved = resolve_model_scope_from_models(configured_patterns, list(models))
            ids = [f"{scoped.model.provider}/{scoped.model.id}" for scoped in resolved.scoped_models]
            # Configured patterns that matched nothing stay listed (and
            # editable) as unavailable entries.
            for diagnostic in resolved.diagnostics:
                if diagnostic.code == "no-match" and diagnostic.pattern not in ids:
                    ids.append(diagnostic.pattern)
            return ids

        state["enabledIds"] = (
            [f"{scoped.model.provider}/{scoped.model.id}" for scoped in session_scoped_models]
            if session_scoped_models
            else configured_enabled_ids(state["availableModels"])
        )

        # Helper to update session's scoped models (session-only, no persist)
        def update_session_models(enabled_ids) -> None:
            state["enabledIds"] = None if enabled_ids is None else list(enabled_ids)
            available_model_ids = state["availableModelIds"]
            has_enabled_available_model = any(model_id in available_model_ids for model_id in enabled_ids or [])
            all_available_models_enabled = enabled_ids is not None and all(
                model_id in enabled_ids for model_id in available_model_ids
            )
            if enabled_ids and has_enabled_available_model and not all_available_models_enabled:
                new_scoped_models = resolve_model_scope_from_models(enabled_ids, state["availableModels"]).scoped_models
                self.session.set_scoped_models(
                    [ScopedModel(model=sm.model, thinking_level=sm.thinking_level) for sm in new_scoped_models]
                )
            else:
                self.session.set_scoped_models([])
            self._update_available_provider_count()
            self.ui.request_render()

        def create(done):
            disposed = {"value": False}
            timeout = _TimeoutCancel(15_000)

            def on_change(enabled_ids) -> None:
                state["selectionChanged"] = True
                update_session_models(enabled_ids)

            def on_persist(enabled_ids) -> None:
                available_models = state["availableModels"]
                available_model_ids = state["availableModelIds"]
                all_enabled = (
                    enabled_ids is not None
                    and len(enabled_ids) == len(available_models)
                    and all(model_id in available_model_ids for model_id in enabled_ids)
                )
                new_patterns = None if enabled_ids is None or all_enabled else enabled_ids
                self.settings_manager.set_enabled_models(list(new_patterns) if new_patterns else None)
                self.show_status("Model selection saved to settings")

            def on_cancel() -> None:
                done()
                self.ui.request_render()

            selector = ScopedModelsSelectorComponent(
                {
                    "allModels": state["availableModels"],
                    "enabledModelIds": state["enabledIds"],
                    "refreshStatus": "Refreshing model catalogs…",
                },
                {
                    "onChange": on_change,
                    "onPersist": on_persist,
                    "onCancel": on_cancel,
                },
            )

            async def refresh_catalogs() -> None:
                # Spawned: the outcome is applied under the UI state lock,
                # which also guards `state` (written by `on_change`) and the
                # disposed check (set by the selector's dispose).
                try:
                    result = await refresh_model_catalogs(self.session.model_runtime, timeout.token)
                except Exception as error:
                    failure = (
                        "Model refresh timed out; showing cached models."
                        if timeout.timed_out
                        else f"Could not refresh model catalogs: {error}"
                    )
                    with self.ui.state_lock:
                        if not disposed["value"]:
                            selector.set_refresh_status(failure, "warning")
                            self.ui.request_render()
                    return

                with self.ui.state_lock:
                    if disposed["value"]:
                        return
                    state["availableModels"] = list(self.session.model_runtime.get_available_snapshot())
                    state["availableModelIds"] = {f"{model.provider}/{model.id}" for model in state["availableModels"]}
                    if not state["selectionChanged"] and not session_scoped_models:
                        state["enabledIds"] = configured_enabled_ids(state["availableModels"])
                        selector.update_models(state["availableModels"], state["enabledIds"])
                    else:
                        selector.update_models(state["availableModels"])
                    if state["enabledIds"] is not None:
                        update_session_models(state["enabledIds"])
                    if result.aborted and timeout.timed_out:
                        selector.set_refresh_status("Model refresh timed out; showing cached models.", "warning")
                    elif result.errors:
                        selector.set_refresh_status(
                            f"Could not refresh {', '.join(result.errors)}; showing cached models.", "warning"
                        )
                    else:
                        selector.set_refresh_status("Model catalogs refreshed.", "success")
                    self.ui.request_render()

            self._spawn_flow(refresh_catalogs())

            def dispose() -> None:
                disposed["value"] = True
                timeout.token.cancel()

            return {"component": selector, "focus": selector, "dispose": dispose}

        self._show_selector(create)

    def _show_user_message_selector(self) -> None:
        user_messages = self.session.get_user_messages_for_forking()

        if not user_messages:
            self.show_status("No messages to fork from")
            return

        initial_selected_id = user_messages[-1]["entryId"]

        def create(done):
            # From the selector's input handling: the selector closes there;
            # the fork runs on its own task.
            def select_entry(entry_id: str) -> None:
                done()
                self._spawn_flow(fork(entry_id))

            async def fork(entry_id: str) -> None:
                try:
                    result = await self.runtime_host.fork(entry_id)
                    if result.get("cancelled"):
                        self.ui.request_render()
                        return

                    with self.ui.state_lock:
                        self._set_editor_text(result.get("selectedText") or "")
                        self.show_status("Forked to new session")
                except Exception as error:
                    self.show_error(str(error))

            def on_cancel() -> None:
                done()
                self.ui.request_render()

            selector = UserMessageSelectorComponent(
                [{"id": message["entryId"], "text": message["text"]} for message in user_messages],
                select_entry,
                on_cancel,
                initial_selected_id,
            )
            return {"component": selector, "focus": selector.get_message_list()}

        self._show_selector(create)

    def handle_clone_command(self) -> None:
        """From the submit handler: the nothing-to-clone status is shown
        here; the clone is spawned."""
        leaf_id = self.session_manager.get_leaf_id()
        if not leaf_id:
            self.show_status("Nothing to clone yet")
            return
        self._spawn_flow(self._clone_session(leaf_id))

    async def _clone_session(self, leaf_id: str) -> None:
        try:
            result = await self.runtime_host.fork(leaf_id, position="at")
            if result.get("cancelled"):
                self.ui.request_render()
                return

            with self.ui.state_lock:
                self._set_editor_text("")
                self.show_status("Cloned to new session")
        except Exception as error:
            self.show_error(str(error))

    def _show_tree_selector(self, initial_selected_id: str | None = None) -> None:
        tree = self.session_manager.get_tree()
        real_leaf_id = self.session_manager.get_leaf_id()
        initial_filter_mode = self.settings_manager.get_tree_filter_mode()

        if not tree:
            self.show_status("No entries in session")
            return

        summary_title = "Summarize branch?"
        summary_options = ["No summary", "Summarize", "Summarize with custom prompt"]

        def create(done):
            # From the selector's input handling: pi closes the selector (and
            # shows the first summary prompt) before its first await; the
            # navigation runs on its own task.
            def select_entry(entry_id: str) -> None:
                # Selecting the current leaf is a no-op (already there)
                if entry_id == self.session_manager.get_leaf_id():
                    done()
                    self.show_status("Already at this point")
                    return

                # Ask about summarization
                done()  # Close selector first

                # Check if we should skip the prompt (user preference to
                # always default to no summary)
                first_choice = None
                if not self.settings_manager.get_branch_summary_skip_prompt():
                    first_choice = self._show_extension_selector(summary_title, summary_options)
                self._spawn_flow(navigate(entry_id, first_choice))

            async def navigate(entry_id: str, first_choice) -> None:
                # Loop until user makes a complete choice or cancels to tree
                wants_summary = False
                custom_instructions: str | None = None

                if first_choice is not None:
                    pending_choice = first_choice
                    while True:
                        summary_choice = await pending_choice

                        if summary_choice is None:
                            # User pressed escape - re-show tree selector with
                            # same selection
                            self._show_tree_selector(entry_id)
                            return

                        wants_summary = summary_choice != "No summary"

                        if summary_choice == "Summarize with custom prompt":
                            custom_instructions = await self._show_extension_editor("Custom summarization instructions")
                            if custom_instructions is None:
                                # User cancelled - loop back to summary selector
                                pending_choice = self._show_extension_selector(summary_title, summary_options)
                                continue

                        # User made a complete choice
                        break

                # The user committed to navigating: stop the active response first.
                if self.session.is_streaming:
                    self._restore_queued_messages_to_editor()
                    await self.session.abort()

                # Recheck after the dialogs and streaming abort, before
                # replacing another operation's UI. The escape handler slot is
                # UI state (`_handle_event` swaps it too): saved and restored
                # under the UI state lock. Each stretch is one hold.
                showing_summary_indicator = False
                escape_handler: dict = {"original": None}
                with self.ui.state_lock:
                    if self.session.is_compacting:
                        self.show_error(
                            "Wait for the current compaction or tree navigation to finish before navigating the session tree."
                        )
                        return
                    escape_handler["original"] = self._default_editor.on_escape
                    if wants_summary:
                        self._default_editor.on_escape = self.session.abort_branch_summary
                        self._append_to_chat(Spacer(1))
                        self._show_status_indicator(BranchSummaryStatusIndicator(self.ui))
                        showing_summary_indicator = True

                failure = None
                try:
                    result = await self.session.navigate_tree(
                        entry_id,
                        {"summarize": wants_summary, "custom_instructions": custom_instructions},
                    )
                except Exception as error:
                    failure = error

                trust_warning = failure is None and await self._needs_project_trust_warning()
                with self.ui.state_lock:
                    try:
                        if failure is not None:
                            self.show_error(str(failure))
                        elif result.aborted:
                            # Summarization aborted - re-show tree selector with
                            # same selection
                            self.show_status("Branch summarization cancelled")
                            self._show_tree_selector(entry_id)
                        elif result.cancelled:
                            self.show_status("Navigation cancelled")
                        else:
                            # Update UI
                            self._rerender_initial_messages(trust_warning)
                            if result.editor_text:
                                self._fill_empty_editor(result.editor_text)
                            self.show_status("Navigated to selected point")
                            self._flush_compaction_queue({"willRetry": False})
                    except Exception as error:
                        self.show_error(str(error))
                    finally:
                        if showing_summary_indicator:
                            self._clear_status_indicator("branchSummary")
                        self._default_editor.on_escape = escape_handler["original"]

            # From the selector's input handling (see select_entry).
            def copy_entry(text) -> None:
                if not text:
                    self.show_error("Selected entry has no text to copy")
                    return
                self._spawn_flow(copy(text))

            async def copy(text) -> None:
                try:
                    await copy_to_clipboard(text, self.ui.terminal.write_sync)
                    self.show_status("Copied selected message to clipboard")
                except Exception as error:
                    self.show_error(str(error))

            def on_cancel() -> None:
                done()
                self.ui.request_render()

            # From the label input's submit: pi appends synchronously, so the
            # next key waits for the write.
            def on_label_edit(entry_id: str, label) -> None:
                async def append() -> None:
                    await self.session_manager.append_label_change(entry_id, label)
                    self.ui.request_render()

                self._finish_before_next_input(append())

            selector = TreeSelectorComponent(
                tree,
                real_leaf_id,
                self.ui.terminal.rows,
                select_entry,
                on_cancel,
                on_label_edit,
                initial_selected_id,
                initial_filter_mode,
            )
            selector.on_copy = copy_entry
            return {"component": selector, "focus": selector}

        self._show_selector(create)

    def _show_session_selector(self) -> None:
        def create(done):
            # From the selector's input handling: the selector closes there;
            # the resume runs on its own task.
            def select_session(session_path: str) -> None:
                done()
                self._spawn_flow(self._handle_resume_session(session_path))

            def on_cancel() -> None:
                done()
                self.ui.request_render()

            async def rename_session(session_file_path: str, next_name) -> None:
                next_value = (next_name or "").strip()
                if not next_value:
                    return
                mgr = await SessionManager(session_file=session_file_path)
                await mgr.append_session_info(next_value)

            selector = SessionSelectorComponent(
                lambda on_progress, cancel: SessionManager.list(
                    self.session_manager.get_cwd(), self.session_manager.get_session_dir(), on_progress, cancel
                ),
                lambda on_progress, cancel: (
                    SessionManager.list_all(on_progress=on_progress, cancel=cancel)
                    if self.session_manager.uses_default_session_dir()
                    else SessionManager.list_all(self.session_manager.get_session_dir(), on_progress, cancel)
                ),
                select_session,
                on_cancel,
                lambda: self._spawn_flow(self.shutdown()),
                lambda: self.ui.request_render(),
                {
                    "renameSession": rename_session,
                    "showRenameHint": True,
                    "keybindings": self._keybindings,
                },
                self.session_manager.get_session_file(),
                state_lock=self.ui.state_lock,
                finish_before_next_input=self.ui.finish_before_next_input,
            )
            return {"component": selector, "focus": selector}

        self._show_selector(create)

    async def _handle_resume_session(self, session_path: str, options: dict | None = None) -> dict:
        self._clear_status_indicator()
        with_session = options.get("withSession") if options else None
        try:
            result = await self.runtime_host.switch_session(
                session_path,
                with_session=with_session,
                project_trust_context_factory=lambda cwd: self._create_project_trust_context(cwd),
            )
            if result.get("cancelled"):
                return result
            self.show_status("Resumed session")
            return result
        except MissingSessionCwdError as error:
            selected_cwd = await self._prompt_for_missing_session_cwd(error)
            if not selected_cwd:
                self.show_status("Resume cancelled")
                return {"cancelled": True}
            result = await self.runtime_host.switch_session(
                session_path,
                cwd_override=selected_cwd,
                with_session=with_session,
                project_trust_context_factory=lambda cwd: self._create_project_trust_context(cwd),
            )
            if result.get("cancelled"):
                return result
            self.show_status("Resumed session in current cwd")
            return result
        except Exception as error:
            return await self._handle_fatal_runtime_error("Failed to resume session", error)

    def get_login_provider_options(self, auth_type: str | None = None) -> list:
        options: list = []
        for provider in self.session.model_runtime.get_providers():
            auth_status = self.session.model_runtime.get_provider_auth_status(provider.id)
            status = (
                {
                    "type": ("oauth" if self.session.model_runtime.is_using_oauth(provider.id) else "api_key"),
                    "source": auth_status.label if auth_status.label is not None else auth_status.source,
                }
                if auth_status.configured
                else None
            )
            subscription = provider.auth.oauth is not None and provider.auth.oauth.is_subscription is True
            if (not auth_type or auth_type == "oauth") and provider.auth.oauth:
                options.append(
                    {
                        "id": provider.id,
                        "name": provider.name,
                        "authType": "oauth",
                        "method": provider.auth.oauth,
                        "status": status,
                        "subscription": subscription,
                    }
                )
            if (not auth_type or auth_type == "api_key") and provider.auth.api_key:
                options.append(
                    {
                        "id": provider.id,
                        "name": provider.name,
                        "authType": "api_key",
                        "method": provider.auth.api_key,
                        "status": status,
                        "subscription": subscription,
                    }
                )
        return sorted(options, key=lambda option: (option["name"].lower(), option["name"]))

    async def _get_logout_provider_options(self) -> list:
        options = []
        for credential in await self.session.model_runtime.list_credentials(
            AuthOperationOptions(cancel=_timeout_cancel(15_000))
        ):
            provider = self.session.model_runtime.get_provider(credential.provider_id)
            options.append(
                {
                    "id": credential.provider_id,
                    "name": provider.name if provider is not None else credential.provider_id,
                    "authType": credential.type,
                    "status": {"type": credential.type, "source": "stored credential"},
                    "subscription": (
                        provider is not None
                        and provider.auth.oauth is not None
                        and provider.auth.oauth.is_subscription is True
                    ),
                }
            )
        return sorted(options, key=lambda option: (option["name"].lower(), option["name"]))

    def _find_login_provider_options(self, provider_ref: str) -> list:
        normalized_provider_ref = provider_ref.strip().lower()
        if not normalized_provider_ref:
            return []

        return [
            provider
            for provider in self.get_login_provider_options()
            if provider["id"].lower() == normalized_provider_ref or provider["name"].lower() == normalized_provider_ref
        ]

    def _handle_login_command(self, provider_ref: str | None = None) -> None:
        """From the submit handler: the selector or the login dialog is shown
        here; a login flow is spawned."""
        if not provider_ref:
            self._show_login_auth_type_selector()
            return

        provider_options = self._find_login_provider_options(provider_ref)
        if len(provider_options) == 1:
            self._start_provider_login(provider_options[0])
            return

        if len(provider_options) > 1:
            provider_ids = {provider["id"] for provider in provider_options}
            if len(provider_ids) == 1:
                self._show_login_auth_type_selector(provider_options)
                return

        self._show_login_provider_selector(None, provider_ref)

    def _start_provider_login(self, provider_option: dict, on_back: Callable[[], None] | None = None) -> None:
        """From the login selectors and /login: the dialog is mounted before
        this returns, as pi's is before its first await; the login flow is
        spawned. `on_back` reopens the selector the login was started from
        when the user cancels it."""
        if provider_option["authType"] == "oauth":
            self._show_login_dialog(provider_option["id"], provider_option["name"], on_back)
            return
        if getattr(provider_option.get("method"), "login", None):
            self._show_api_key_login_dialog(provider_option["id"], provider_option["name"], on_back)
            return
        self._show_ambient_auth_dialog(provider_option, on_back)

    def _show_login_auth_type_selector(self, provider_options: list | None = None) -> None:
        oauth_provider = (
            next((provider for provider in provider_options if provider["authType"] == "oauth"), None)
            if provider_options
            else None
        )
        oauth_login_label = (
            getattr(oauth_provider["method"], "login_label", None)
            if oauth_provider is not None and oauth_provider.get("method") is not None
            else None
        )
        subscription_label = oauth_login_label if oauth_login_label is not None else "Sign in with an account"
        api_key_label = "Sign in with an API key"
        available_auth_types = (
            {provider["authType"] for provider in provider_options} if provider_options else {"oauth", "api_key"}
        )
        options: list = []
        if "oauth" in available_auth_types:
            options.append(subscription_label)
        if "api_key" in available_auth_types:
            options.append(api_key_label)

        if not options:
            self.show_status("No login methods available.")
            return

        if provider_options and len(options) == 1:
            provider_option = provider_options[0]
            if provider_option:
                self._start_provider_login(provider_option)
            return

        title = (
            f"Select authentication method for {provider_options[0]['name']}:"
            if provider_options
            else "Select authentication method:"
        )

        def create(done):
            def on_select(option: str) -> None:
                done()
                auth_type = "oauth" if option == subscription_label else "api_key"
                if provider_options:
                    provider_option = next(
                        (provider for provider in provider_options if provider["authType"] == auth_type), None
                    )
                    if provider_option:
                        self._start_provider_login(
                            provider_option, lambda: self._show_login_auth_type_selector(provider_options)
                        )
                    return
                self._show_login_provider_selector(auth_type)

            def on_cancel() -> None:
                done()
                self.ui.request_render()

            selector = ExtensionSelectorComponent(title, options, on_select, on_cancel)
            return {"component": selector, "focus": selector, "dispose": selector.dispose}

        self._show_selector(create)

    def _show_login_provider_selector(
        self, auth_type: str | None = None, initial_search_input: str | None = None
    ) -> None:
        provider_options = self.get_login_provider_options(auth_type)
        if not provider_options:
            if auth_type == "oauth":
                message = "No account providers available."
            elif auth_type == "api_key":
                message = "No API key providers available."
            else:
                message = "No login providers available."
            self.show_status(message)
            return

        def create(done):
            # From the selector's input handling: the selector closes and the
            # login dialog mounts there.
            def select_provider(provider_id: str, selected_auth_type: str) -> None:
                done()

                provider_option = next(
                    (
                        provider
                        for provider in provider_options
                        if provider["id"] == provider_id and provider["authType"] == selected_auth_type
                    ),
                    None,
                )
                if provider_option is not None:
                    self._start_provider_login(
                        provider_option, lambda: self._show_login_provider_selector(auth_type, initial_search_input)
                    )

            def on_cancel() -> None:
                done()
                if auth_type:
                    self._show_login_auth_type_selector()
                else:
                    self.ui.request_render()

            selector = OAuthSelectorComponent(
                "login",
                provider_options,
                select_provider,
                on_cancel,
                initial_search_input,
            )
            return {"component": selector, "focus": selector}

        self._show_selector(create)

    def _show_oauth_selector(self, mode: str) -> None:
        """pi's showOAuthSelector is async; the dispatch call sites treat it as
        fire-and-forget, so the logout path spawns its async body."""
        if mode == "login":
            self._show_login_auth_type_selector()
            return

        self._spawn_flow(self._show_logout_selector(mode))

    async def _show_logout_selector(self, mode: str) -> None:
        try:
            provider_options = await self._get_logout_provider_options()
        except Exception as error:
            self.show_error(f"Could not read stored credentials: {error}")
            return
        if not provider_options:
            self.show_status(
                "No stored credentials to remove. /logout only removes credentials saved by /login; "
                "environment variables and models.json config are unchanged."
            )
            return

        def create(done):
            # From the selector's input handling: the selector closes there;
            # the logout runs on its own task.
            def select_provider(provider_id: str) -> None:
                done()

                provider_option = next(
                    (provider for provider in provider_options if provider["id"] == provider_id), None
                )
                if provider_option is not None:
                    self._spawn_flow(logout(provider_option))

            async def logout(provider_option: dict) -> None:
                try:
                    await self.session.model_runtime.logout(
                        provider_option["id"], AuthOperationOptions(cancel=_timeout_cancel(15_000))
                    )
                    self._update_available_provider_count()
                    message = (
                        f"Logged out of {provider_option['name']}"
                        if provider_option["authType"] == "oauth"
                        else f"Removed stored API key for {provider_option['name']}. "
                        "Environment variables and models.json config are unchanged."
                    )
                    self.show_status(message)
                except Exception as error:
                    self.show_error(
                        f"Credentials removed for {provider_option['name']}, "
                        f"but local model state could not be synchronized: {error}"
                        if isinstance(error, CredentialSynchronizationError)
                        else f"Logout failed: {error}"
                    )

            def on_cancel() -> None:
                done()
                self.ui.request_render()

            selector = OAuthSelectorComponent(
                mode,
                provider_options,
                lambda provider_id, _auth_type: select_provider(provider_id),
                on_cancel,
            )
            return {"component": selector, "focus": selector}

        self._show_selector(create)

    async def _complete_provider_authentication(
        self,
        provider_id: str,
        provider_name: str,
        auth_type: str,
        previous_model,
        first_step: Callable[[], None] | None = None,
    ) -> None:
        """`first_step` is the caller's UI step that pi runs in the same
        synchronous stretch as this flow's start (the login flows restore the
        editor slot): it applies in this flow's first hold."""
        action_label = f"Logged in to {provider_name}" if auth_type == "oauth" else f"Saved API key for {provider_name}"
        # UI steps of the current stretch, applied at the start of its hold (or
        # just before the stretch's await): see `first_step`.
        pending_steps: list = [first_step] if first_step is not None else []

        def apply_pending_steps() -> None:
            while pending_steps:
                pending_steps.pop(0)()

        session = self.session
        # Dynamic catalogs may be empty until the first authenticated network refresh.
        defer_selection = (
            _is_unknown_model(previous_model)
            and provider_id in DEFAULT_MODEL_PER_PROVIDER
            and not any(
                model.provider == provider_id and model.id == DEFAULT_MODEL_PER_PROVIDER[provider_id]
                for model in session.model_runtime.get_available_snapshot()
            )
        )

        async def finish_authentication(then: Callable[[], None] | None = None) -> None:
            selected_model = None
            selection_error: str | None = None
            if _is_unknown_model(previous_model):
                available_models = self.session.model_runtime.get_available_snapshot()
                provider_models = [model for model in available_models if model.provider == provider_id]
                # Matches LLAMA_PROVIDER_ID from pi's built-in llama extension, which pidrei does not
                # ship; kept inline so a provider registered under that id gets the same guidance.
                if provider_id == "llama.cpp":
                    selection_error = _llama_cpp_post_login_guidance(action_label, len(provider_models))
                elif provider_id not in DEFAULT_MODEL_PER_PROVIDER:
                    selection_error = (
                        f'{action_label}, but no default model is configured for provider "{provider_id}". '
                        "Use /model to select a model."
                    )
                elif not provider_models:
                    selection_error = (
                        f"{action_label}, but no models are available for that provider. Use /model to select a model."
                    )
                else:
                    default_model_id = DEFAULT_MODEL_PER_PROVIDER[provider_id]
                    # pi falls back to catalog order for Radius here; pidrei has no Radius provider.
                    selected_model = next((model for model in provider_models if model.id == default_model_id), None)
                    if selected_model is None:
                        selection_error = (
                            f'{action_label}, but its default model "{default_model_id}" is not available. '
                            "Use /model to select a model."
                        )
                    else:
                        with self.ui.state_lock:
                            apply_pending_steps()  # the stretch ends at this await
                        try:
                            await self.session.set_model(selected_model, persist=True)
                        except Exception as error:
                            selected_model = None
                            selection_error = (
                                f"{action_label}, but selecting its default model failed: {error}. "
                                "Use /model to select a model."
                            )

            # Reached from the login flow and from the spawned catalog refresh
            # below: the UI updates are one hold.
            with self.ui.state_lock:
                apply_pending_steps()
                self._update_available_provider_count()
                self._footer.invalidate()
                self._update_editor_border_color()
                if selected_model is not None:
                    self.show_status(
                        f"{action_label}. Selected {selected_model.id}. Credentials saved to {get_auth_path()}"
                    )
                    self._spawn_flow(self._maybe_warn_about_anthropic_subscription_auth(selected_model))
                else:
                    self.show_status(f"{action_label}. Credentials saved to {get_auth_path()}")
                    if selection_error:
                        self.show_error(selection_error)
                    else:
                        self._spawn_flow(self._maybe_warn_about_anthropic_subscription_auth())
                if then is not None:
                    then()

        if defer_selection:
            with self.ui.state_lock:
                apply_pending_steps()
                self.show_status(f"{action_label}. Credentials saved to {get_auth_path()}. Refreshing model catalog…")
        else:
            await finish_authentication()

        timeout = _TimeoutCancel(15_000)

        async def refresh_provider_catalog() -> None:
            try:
                result = await session.model_runtime.refresh(
                    ModelsRefreshOptions(providers=[provider_id], cancel=timeout.token)
                )
            except Exception as error:
                self.show_warning(f"{action_label}, but its model catalog could not be refreshed: {error}")
                return
            if result.aborted:
                warning = f"{action_label}, but its model catalog refresh timed out; using cached models."
                pending_steps.append(lambda: self.show_warning(warning))
            elif result.errors:
                warning = f"{action_label}, but its model catalog could not be refreshed; using cached models."
                pending_steps.append(lambda: self.show_warning(warning))

            def refreshed() -> None:
                self._update_available_provider_count()
                self._footer.invalidate()
                self.ui.request_render()

            # Do not replace a model or session selected while the refresh was running.
            if defer_selection and self.session is session and session.model is previous_model:
                await finish_authentication(then=refreshed)
                return
            with self.ui.state_lock:
                apply_pending_steps()
                refreshed()

        self._spawn_flow(refresh_provider_catalog())

    def _show_in_editor_slot(self, component) -> None:
        """Swap `component` into the editor slot and focus it (the login
        flows drive their dialogs from the login task)."""
        with self.ui.state_lock:
            self._editor_container.clear()
            self._editor_container.add_child(component)
            self.ui.set_focus(component)
            self.ui.request_render()

    def _restore_editor_slot(self) -> None:
        """Put the editor back in the editor slot."""
        with self.ui.state_lock:
            self._editor_container.clear()
            self._editor_container.add_child(self.editor)
            self.ui.set_focus(self.editor)
            self.ui.request_render()

    def _show_ambient_auth_dialog(self, provider_option: dict, on_back: Callable[[], None] | None = None) -> None:
        """Reached from `_start_provider_login`."""

        # The dialog's only completion is its cancel key.
        def on_complete(*_args) -> None:
            self._restore_editor_slot()
            if on_back is not None:
                on_back()

        dialog = LoginDialogComponent(
            self.ui,
            provider_option["id"],
            on_complete,
            provider_option["name"],
            f"{provider_option['name']} setup",
        )
        method_name = getattr(provider_option.get("method"), "name", None)
        dialog.show_info(
            f"{method_name if method_name is not None else 'Authentication'} is configured outside {APP_NAME}.",
            [],
            True,
        )

        self._show_in_editor_slot(dialog)

    def _show_api_key_login_dialog(
        self, provider_id: str, provider_name: str, on_back: Callable[[], None] | None = None
    ) -> None:
        """Reached from `_start_provider_login`: the dialog is set up and
        mounted here, as pi's is before its first await; the login flow is
        spawned."""
        previous_model = self.session.model

        dialog = LoginDialogComponent(
            self.ui,
            provider_id,
            lambda *_args: None,  # Completion handled below
            provider_name,
        )

        if provider_id == "amazon-bedrock":
            dialog.show_details(
                [
                    theme.fg("text", "You can also use an AWS profile, IAM keys, or role-based credentials."),
                    theme.fg("muted", "See:"),
                    theme.fg("accent", f"  {os.path.join(get_docs_path(), 'providers.md')}"),
                ]
            )

        self._show_in_editor_slot(dialog)
        self._spawn_flow(self._api_key_login(dialog, provider_id, provider_name, previous_model, on_back))

    async def _api_key_login(
        self, dialog, provider_id: str, provider_name: str, previous_model, on_back: Callable[[], None] | None
    ) -> None:
        try:
            await self._login_provider(dialog, provider_id, "api_key")
            # pi restores the editor in the stretch that starts completing the
            # login: it applies in that flow's first hold.
            await self._complete_provider_authentication(
                provider_id, provider_name, "api_key", previous_model, self._restore_editor_slot
            )
        except Exception as error:
            error_msg = str(error)
            with self.ui.state_lock:
                self._restore_editor_slot()
                if isinstance(error, CredentialSynchronizationError):
                    self.show_error(
                        f"Saved API key for {provider_name}, but local model state could not be synchronized: {error_msg}"
                    )
                elif error_msg == "Login cancelled":
                    if on_back is not None:
                        on_back()
                else:
                    self.show_error(f"Failed to save API key for {provider_name}: {error_msg}")

    async def _show_auth_select(self, dialog, prompt) -> str:
        done = tonio.Event()
        outcome: dict = {}

        labels = [option.label for option in prompt.options]

        # From the selector's input handling: the dialog is back in the slot
        # before the next key.
        def on_select(option_label: str) -> None:
            self._show_in_editor_slot(dialog)
            option_id = next((option.id for option in prompt.options if option.label == option_label), None)
            if option_id:
                outcome["value"] = option_id
            done.set()

        def on_cancel() -> None:
            self._show_in_editor_slot(dialog)
            done.set()

        selector = ExtensionSelectorComponent(prompt.message, labels, on_select, on_cancel)
        self._show_in_editor_slot(selector)

        await done.wait(None)
        if "value" in outcome:
            return outcome["value"]
        raise Exception("Login cancelled")

    async def _show_auth_prompt(self, dialog, prompt) -> str:
        if prompt.type == "select":
            response = self._show_auth_select(dialog, prompt)
        elif prompt.type == "manual_code":
            response = dialog.show_manual_input(prompt.message)
        else:
            response = dialog.show_prompt(prompt.message, prompt.placeholder)

        cancel = prompt.cancel
        if cancel is None:
            return await response
        if cancel.cancelled:
            raise Exception("Login cancelled")

        # Race the response against out-of-band cancellation. Like pi's
        # Promise.race, a cancelled prompt leaves the response awaitable
        # unresolved behind the dialog that is being torn down.
        settled = tonio.Event()
        outcome = tonio.Result()

        async def run_response() -> None:
            try:
                outcome.store(("value", await response))
            except Exception as error:
                outcome.store(("error", error))
            finally:
                settled.set()

        tonio.spawn.without_tracking(run_response())
        await tonio.Waiter.any(settled, cancel.event)
        stored = outcome.fetch()
        if stored is None:
            raise Exception("Login cancelled")
        kind, payload = stored
        if kind == "error":
            raise payload
        return payload

    def _notify_auth_dialog(self, dialog, event) -> None:
        with self.ui.state_lock:
            if event.type == "auth_url":
                dialog.show_auth(event.url, event.instructions)
            elif event.type == "device_code":
                dialog.show_device_code({"verificationUri": event.verification_uri, "userCode": event.user_code})
                dialog.show_waiting("Waiting for authentication...")
            elif event.type == "info":
                dialog.show_info(event.message, event.links)
            else:
                dialog.show_progress(event.message)

    async def _login_provider(self, dialog, provider_id: str, method: str) -> None:
        mode = self

        class DialogInteraction:
            cancel = dialog.signal

            async def prompt(self, prompt) -> str:
                return await mode._show_auth_prompt(dialog, prompt)

            def notify(self, event) -> None:
                mode._notify_auth_dialog(dialog, event)

        await self.session.model_runtime.login(
            provider_id,
            method,
            DialogInteraction(),
            LoginOptions(get_device_id=self.settings_manager.get_or_create_device_id),
        )

    def _show_login_dialog(
        self, provider_id: str, provider_name: str, on_back: Callable[[], None] | None = None
    ) -> None:
        """Reached from `_start_provider_login`: mounted here; the OAuth flow
        is spawned."""
        previous_model = self.session.model
        dialog = LoginDialogComponent(self.ui, provider_id, lambda *_args: None, provider_name)
        self._show_in_editor_slot(dialog)
        self._spawn_flow(self._oauth_login(dialog, provider_id, provider_name, previous_model, on_back))

    async def _oauth_login(
        self, dialog, provider_id: str, provider_name: str, previous_model, on_back: Callable[[], None] | None
    ) -> None:
        try:
            await self._login_provider(dialog, provider_id, "oauth")
            # pi restores the editor in the stretch that starts completing the
            # login: it applies in that flow's first hold.
            await self._complete_provider_authentication(
                provider_id, provider_name, "oauth", previous_model, self._restore_editor_slot
            )
        except Exception as error:
            error_msg = str(error)
            with self.ui.state_lock:
                self._restore_editor_slot()
                if isinstance(error, CredentialSynchronizationError):
                    self.show_error(
                        f"Logged in to {provider_name}, but local model state could not be synchronized: {error_msg}"
                    )
                elif error_msg == "Login cancelled":
                    if on_back is not None:
                        on_back()
                else:
                    self.show_error(f"Failed to login to {provider_name}: {error_msg}")

    # =========================================================================
    # Command handlers
    # =========================================================================

    async def _handle_reload_command(self) -> None:
        """The extension `reload` action's entry: the reload's start (one hold
        of the UI state lock), then the reload itself."""
        previous_editor = self._start_reload()
        if previous_editor is not None:
            await self._reload(previous_editor)

    def _start_reload(self):
        """pi's part before its first await (the guards' warnings, the
        extension-UI reset and the reload box), in one hold. Returns the
        editor to restore when the reload ends, or None when the reload was
        refused; the reload itself is `_reload`."""
        with self.ui.state_lock:
            if self.session.is_streaming:
                self.show_warning("Wait for the current response to finish before reloading.")
                return None
            if self.session.is_compacting:
                self.show_warning("Wait for compaction to finish before reloading.")
                return None

            self._reset_extension_ui()

            def border_color(s: str) -> str:
                return theme.fg("border", s)

            reload_box = Container()
            reload_box.add_child(DynamicBorder(border_color))
            reload_box.add_child(Spacer(1))
            reload_box.add_child(
                ThemedText(
                    lambda: theme.fg(
                        "muted", "Reloading keybindings, extensions, skills, prompts, themes, and context files..."
                    ),
                    1,
                    0,
                )
            )
            reload_box.add_child(Spacer(1))
            reload_box.add_child(DynamicBorder(border_color))

            # The box is up before the reload work starts, which runs on its
            # own task (pi yields a tick here so the box can paint).
            previous_editor = self.editor
            self._editor_container.clear()
            self._editor_container.add_child(reload_box)
            self.ui.set_focus(reload_box)
            self.ui.request_render(True)
            return previous_editor

    async def _reload(self, previous_editor) -> None:
        def dismiss_reload_box(restore_previous: bool) -> None:
            with self.ui.state_lock:
                editor = previous_editor if restore_previous else self.editor
                self._editor_container.clear()
                self._editor_container.add_child(editor)
                self.ui.set_focus(editor)
                self.ui.request_render()

        chat_restored_before_session_start = False
        reload_box_dismissed = False

        async def restore_chat_before_session_start() -> None:
            nonlocal chat_restored_before_session_start
            if chat_restored_before_session_start:
                return
            with self.ui.state_lock:
                self._hide_thinking_block = self.settings_manager.get_hide_thinking_block()
                self._output_pad = self.settings_manager.get_output_pad()
                self._rebuild_chat_from_messages()
            chat_restored_before_session_start = True

        try:
            await self.session.reload(restore_chat_before_session_start)
            # pi's next two blocks are synchronous: their pidrei-only I/O (the
            # keybindings file, the cwd, the trust save) runs first, then
            # each block applies in one hold (§4.5b). `applyFromSettings` is
            # awaited in pi too.
            await self._keybindings.reload()
            resolved_cwd = await self._footer_data_provider.resolve_cwd(self.session_manager.get_cwd())
            with self.ui.state_lock:
                if not chat_restored_before_session_start:
                    self._hide_thinking_block = self.settings_manager.get_hide_thinking_block()
                    self._output_pad = self.settings_manager.get_output_pad()
                    self._rebuild_chat_from_messages()
                    chat_restored_before_session_start = True
                active_header = self._custom_header if self._custom_header is not None else self._built_in_header
                if is_expandable(active_header):
                    active_header.set_expanded(self._tool_output_expanded)
                set_registered_themes(self.session.resource_loader.get_themes()["themes"])
                cwd_changed = self._apply_runtime_settings(resolved_cwd)
            if cwd_changed:
                await self._footer_data_provider.watch_cwd()
            await self._theme_controller.apply_from_settings()

            saved_implicit_project_trust, trust_warning = await self._maybe_save_implicit_project_trust_after_reload()
            with self.ui.state_lock:
                self._setup_autocomplete_provider()
                runner = self.session.extension_runner
                self._setup_extension_shortcuts(runner)
                self._show_loaded_resources({"force": False, "showDiagnosticsWhenQuiet": True})
                if trust_warning is not None:
                    self.show_warning(trust_warning)
                models_json_error = self.session.model_runtime.get_error()
                if models_json_error:
                    self.show_error(f"models.json error: {models_json_error}")
                self.show_status(
                    "Reloaded keybindings, extensions, skills, prompts, themes, and context files; saved project trust"
                    if saved_implicit_project_trust
                    else "Reloaded keybindings, extensions, skills, prompts, themes, and context files"
                )
                dismiss_reload_box(restore_previous=False)
            reload_box_dismissed = True
        except Exception as error:
            if not reload_box_dismissed:
                dismiss_reload_box(restore_previous=True)
            self.show_error(f"Reload failed: {error}")

    async def _handle_export_command(self, text: str) -> None:
        output_path = self._get_path_command_argument(text, "/export")

        try:
            if output_path is not None and output_path.endswith(".jsonl"):
                file_path = await self.session.export_to_jsonl(output_path)
                self.show_status(f"Session exported to: {file_path}")
            else:
                file_path = await self.session.export_to_html(output_path, {"themeName": theme.name})
                self.show_status(f"Session exported to: {file_path}")
        except Exception as error:
            self.show_error(f"Failed to export session: {error if str(error) else 'Unknown error'}")

    def _get_path_command_argument(self, text: str, command: str) -> str | None:
        if text == command:
            return None
        if not text.startswith(f"{command} "):
            return None

        args_string = text[len(command) + 1 :].lstrip()
        if not args_string:
            return None

        first_char = args_string[0]
        if first_char in ('"', "'"):
            closing_quote_index = args_string.find(first_char, 1)
            if closing_quote_index < 0:
                return None
            return args_string[1:closing_quote_index]

        whitespace_match = re.search(r"\s", args_string)
        if whitespace_match is None:
            return args_string
        return args_string[: whitespace_match.start()]

    def handle_import_command(self, text: str, and_then: Callable[[], None] | None = None) -> None:
        """From the submit handler: the usage error or the confirmation
        dialog is shown here; the import is spawned. `and_then` runs once the
        command is done (pi's submit clears the editor after it)."""
        input_path = self._get_path_command_argument(text, "/import")
        if not input_path:
            self.show_error("Usage: /import <path.jsonl>")
            if and_then is not None:
                and_then()
            return

        confirmed = self._show_extension_confirm("Import session", f"Replace current session with {input_path}?")

        async def run() -> None:
            await self._import_session(input_path, confirmed)
            if and_then is not None:
                and_then()

        self._spawn_flow(run())

    async def _import_session(self, input_path: str, confirmed: Awaitable[bool]) -> None:
        if not await confirmed:
            self.show_status("Import cancelled")
            return

        try:
            self._clear_status_indicator()
            result = await self.runtime_host.import_from_jsonl(input_path)
            if result.get("cancelled"):
                self.show_status("Import cancelled")
                return
            self.show_status(f"Session imported from: {input_path}")
        except MissingSessionCwdError as error:
            selected_cwd = await self._prompt_for_missing_session_cwd(error)
            if not selected_cwd:
                self.show_status("Import cancelled")
                return
            result = await self.runtime_host.import_from_jsonl(input_path, selected_cwd)
            if result.get("cancelled"):
                self.show_status("Import cancelled")
                return
            self.show_status(f"Session imported from: {input_path}")
        except SessionImportFileNotFoundError as error:
            self.show_error(f"Failed to import session: {error}")
        except Exception as error:
            await self._handle_fatal_runtime_error("Failed to import session", error)

    async def _handle_share_command(self) -> None:
        """Share the session as a secret gist.

        pi also tries a Radius artifact upload first and falls back to the gist
        path; Radius is dropped surface here (initial port), so only the
        gist half exists and the JSONL export that feeds Radius is not made.
        pi later moved both halves into `modes/interactive/session-share.ts`
        (upstream 460191cf); that file is dropped with the Radius flow, so the
        gist half stays here and the reusable export is core/session_export.py.
        """
        # Check if gh is available and logged in
        try:
            auth_result = await run_command(
                ["gh", "auth", "status"],  # PATH lookup, like pi's spawnSync
                stdin=subprocess.DEVNULL,
                capture_output=True,
            )
        except OSError:
            self.show_error("GitHub CLI (gh) is not installed. Install it from https://cli.github.com/")
            return
        if auth_result.returncode != 0:
            self.show_error("GitHub CLI is not logged in. Run 'gh auth login' first.")
            return

        # Export to a temp file
        tmp_file = TEMP_DIR / "session.html"
        try:
            await self._export_and_share(tmp_file)
        except BaseException:
            # An await is not served once this task is cancelled, so the cleanup is detached.
            tonio.spawn.without_tracking(discard_temp_file(tmp_file))
            raise
        await discard_temp_file(tmp_file)

    async def _export_and_share(self, tmp_file: fs.Path) -> None:
        try:
            await self.session.export_to_html(str(tmp_file), {"themeName": theme.name})
        except Exception as error:
            self.show_error(f"Failed to export session: {error if str(error) else 'Unknown error'}")
            return
        await self._share_via_gist(str(tmp_file))

    async def _share_via_gist(self, tmp_file: str) -> None:
        # Show cancellable loader, replacing the editor
        loader = BorderedLoader(self.ui, theme, "Creating gist...")
        self._show_in_editor_slot(loader)

        # Create a secret gist asynchronously. exec_command kills the process
        # when the cancel token fires (pi kills the spawned gh directly).
        gist_cancel = AiCancelToken()

        # The loader's abort key.
        def on_abort() -> None:
            gist_cancel.cancel()
            with self.ui.state_lock:
                self._restore_share_editor(loader)
                self.show_status("Share cancelled")

        loader.on_abort = on_abort

        try:
            result = await exec_command(
                "gh",
                ["gist", "create", "--public=false", tmp_file],
                self.session_manager.get_cwd(),
                cancel=gist_cancel,
            )

            # One hold: the cancelled check and the restore are atomic
            # against the loader's abort key, which restores under the lock.
            with self.ui.state_lock:
                if loader.signal.cancelled:
                    return

                self._restore_share_editor(loader)

                if result.code != 0:
                    error_msg = result.stderr.strip() or "Unknown error"
                    self.show_error(f"Failed to create gist: {error_msg}")
                    return

                # Extract gist ID from the URL returned by gh
                # gh returns something like: https://gist.github.com/username/GIST_ID
                gist_url = result.stdout.strip()
                gist_id = gist_url.split("/")[-1] if gist_url else None
                if not gist_id:
                    self.show_error("Failed to parse gist ID from gh output")
                    return

                # The gist URL is the share URL; a viewer link is added only when
                # one is configured (PIDREI_SHARE_VIEWER_URL).
                preview_url = get_share_viewer_url(gist_id)
                gist_link = hyperlink(gist_url, gist_url)
                self.show_status(
                    f"Share URL: {hyperlink(preview_url, preview_url)}\nGist: {gist_link}"
                    if preview_url
                    else f"Gist: {gist_link}"
                )
        except Exception as error:
            with self.ui.state_lock:
                if not loader.signal.cancelled:
                    self._restore_share_editor(loader)
                    self.show_error(f"Failed to create gist: {error if str(error) else 'Unknown error'}")

    def _restore_share_editor(self, loader: BorderedLoader) -> None:
        # From the share task and from the loader's abort: both can land, so
        # the restore acts only while this loader holds the slot.
        with self.ui.state_lock:
            loader.dispose()
            if loader not in self._editor_container.children:
                return
            self._editor_container.clear()
            self._editor_container.add_child(self.editor)
            self.ui.set_focus(self.editor)
            self.ui.request_render()

    def _handle_copy_command(self, options: dict | None = None, and_then: Callable[[], None] | None = None) -> None:
        """The copy action and /copy: the selection is read and the no-text
        error shown here, as pi does before its first await; the clipboard
        write is spawned. `and_then` runs once the command is done (pi's
        submit clears the editor after it)."""
        options = options or {}
        with self.ui.state_lock:
            # pi narrows with `instanceof TuiAltScreen`; the reference proxy is
            # deliberately not isinstance-transparent, so ask the renderer.
            renderer = self._renderer
            if (
                options.get("preferSelection")
                and isinstance(renderer, TuiAltScreen)
                and not renderer.get_copy_on_select()
                and renderer.has_active_selection()
            ):
                copy = renderer.copy_active_selection_to_clipboard()
            else:
                text = self.session.get_last_assistant_text()
                if not text:
                    self.show_error("No agent messages to copy yet.")
                    if and_then is not None:
                        and_then()
                    return
                copy = self._copy_last_message(text, options)

        async def run() -> None:
            await copy
            if and_then is not None:
                and_then()

        self._spawn_flow(run())

    async def _copy_last_message(self, text: str, options: dict) -> None:
        try:
            await copy_to_clipboard(text, self.ui.terminal.write_sync)
            # pi narrows with `instanceof TuiAltScreen`; the reference proxy is
            # deliberately not isinstance-transparent, so ask the renderer.
            if options.get("flashConfirmation") and isinstance(self._renderer, TuiAltScreen):
                self.ui.flash("Copied!")
            else:
                self.show_status("Copied last agent message to clipboard")
        except Exception as error:
            self.show_error(str(error))

    def _handle_name_command(self, text: str) -> None:
        """From the submit handler: showing the name (or the usage) happens
        here. Setting it is synchronous in pi, so the next key waits for it;
        the editor clears after, as pi's submit does."""
        name = re.sub(r"^/name\s*", "", text).strip()
        if not name:
            with self.ui.state_lock:
                current_name = self.session_manager.get_session_name()
                if current_name:
                    self._append_to_chat(
                        Spacer(1), ThemedText(lambda: theme.fg("dim", f"Session name: {current_name}"), 1, 0)
                    )
                else:
                    self.show_warning("Usage: /name <name>")
                self._set_editor_text("")
            return
        self._finish_before_next_input(self._then_clear_editor(self._set_session_name(name)))

    async def _set_session_name(self, name: str) -> None:
        await self.session.set_session_name(name)
        session_name = self.session_manager.get_session_name()
        if session_name != name:
            self.show_warning(f"Session name was normalized from {json.dumps(name)} to {json.dumps(session_name)}")
        display_name = session_name if session_name is not None else name
        self._append_to_chat(
            Spacer(1),
            ThemedText(lambda: theme.fg("dim", f"Session name set: {display_name}"), 1, 0),
        )

    def handle_session_command(self) -> None:
        stats = self.session.get_session_stats()
        session_name = self.session_manager.get_session_name()
        entries = self.session_manager.get_entries()
        cache_waste = compute_cache_waste(entries, self.session.model_runtime)

        # Cost/token totals per provider/model actually used (e.g. OpenRouter
        # `auto` resolves to a concrete responseModel). Usage without model
        # attribution is grouped separately so the breakdown reconciles with
        # the session total.
        usage_breakdown = get_usage_cost_breakdown(entries)

        # Snapshot the stats; the text is built on demand so it follows theme changes.
        cache_warming_status = self.session.cache_warming_status
        cache_warming_mode = self.settings_manager.get_cache_warming_mode()
        model = self.session.model
        selected_model_key = f"{model.provider}/{model.id}" if model is not None else None

        def render_info() -> str:
            info = f"{theme.bold('Session Info')}\n\n"
            if session_name:
                info += f"{theme.fg('dim', 'Name:')} {session_name}\n"
            session_file = stats.session_file if stats.session_file is not None else "In-memory"
            info += f"{theme.fg('dim', 'File:')} {session_file}\n"
            info += f"{theme.fg('dim', 'ID:')} {stats.session_id}\n\n"
            info += f"{theme.bold('Messages')}\n"
            info += f"{theme.fg('dim', 'Total:')} {stats.total_messages}\n"
            info += f"{theme.fg('dim', 'User:')} {stats.user_messages}\n"
            info += f"{theme.fg('dim', 'Assistant:')} {stats.assistant_messages}\n"
            info += f"{theme.fg('dim', 'Tools:')} {stats.tool_calls} calls, {stats.tool_results} results\n\n"
            info += f"{theme.bold('Tokens')}\n"
            # "Input" is the full prompt volume. With cache activity, split it
            # into cached (served from cache) vs uncached (everything else) -
            # the only provider-independent split. Cache writes, where
            # reported, are a detail of the uncached portion.
            input_tokens = stats.tokens.input
            cache_read = stats.tokens.cache_read
            cache_write = stats.tokens.cache_write
            prompt_tokens = input_tokens + cache_read + cache_write
            info += f"{theme.fg('dim', 'Input:')} {prompt_tokens:,}\n"
            if prompt_tokens > 0 and (cache_read > 0 or cache_write > 0):
                hit_rate = theme.fg("dim", f"({cache_read / prompt_tokens * 100:.1f}%)")
                info += f"  {theme.fg('dim', 'Cached:')} {cache_read:,} {hit_rate}\n"
                written = f" {theme.fg('dim', f'({cache_write:,} written to cache)')}" if cache_write > 0 else ""
                info += f"  {theme.fg('dim', 'Uncached:')} {input_tokens + cache_write:,}{written}\n"
            info += f"{theme.fg('dim', 'Output:')} {stats.tokens.output:,}\n"
            info += f"{theme.fg('dim', 'Total:')} {stats.tokens.total:,}\n"

            info += f"\n{theme.bold('Cache Warming')}\n"
            info += f"{theme.fg('dim', 'Mode:')} {cache_warming_mode}\n"
            status_text = (
                format_cache_warming_status(cache_warming_status)
                if cache_warming_status is not None
                else "Inactive (cache warming unavailable)"
            )
            info += f"{theme.fg('dim', 'Status:')} {status_text}\n"
            decision = cache_warming_status.decision if cache_warming_status is not None else None
            if decision is not None and decision.economics_available:
                info += f"{theme.fg('dim', 'Cache miss penalty:')} ${decision.miss_cost:.3f}\n"
                info += f"{theme.fg('dim', 'Refresh cost:')} ${decision.warm_cost:.3f}\n"

            if stats.cost > 0 or cache_waste.missed_tokens > 0:
                info += f"\n{theme.bold('Cost')}\n"
                info += f"{theme.fg('dim', 'Total:')} ${stats.cost:.3f}"
                # A single entry repeats the total, unless it names a model other than the selected one.
                if len(usage_breakdown) > 1 or (usage_breakdown and usage_breakdown[0].key != selected_model_key):
                    for entry in usage_breakdown:
                        info += (
                            f"\n  {theme.fg('dim', f'{entry.key}:')} ${entry.cost:.3f} "
                            f"{theme.fg('dim', f'({format_tokens(entry.tokens)} tokens)')}"
                        )
                if cache_waste.missed_tokens > 0:
                    miss_label = "1 miss" if cache_waste.miss_count == 1 else f"{cache_waste.miss_count} misses"
                    detail = f"{cache_waste.missed_tokens:,} tokens, {miss_label}"
                    info += (
                        f"\n{theme.fg('dim', 'Cache Re-billed:')} ${cache_waste.missed_cost:.3f} "
                        f"{theme.fg('dim', f'({detail})')}"
                        if cache_waste.missed_cost >= 0.0001
                        else f"\n{theme.fg('dim', 'Cache Re-billed:')} {detail}"
                    )
            return info

        self._append_to_chat(Spacer(1), ThemedText(render_info, 1, 0))

    async def _handle_changelog_command(self) -> None:
        changelog_path = get_changelog_path()
        all_entries = await parse_changelog(changelog_path)

        changelog_markdown = (
            "\n\n".join(normalize_changelog_links(entry["content"], entry) for entry in reversed(all_entries))
            if all_entries
            else "No changelog entries found."
        )

        self._append_to_chat(
            Spacer(1),
            DynamicBorder(),
            ThemedText(lambda: theme.bold(theme.fg("accent", "What's New")), 1, 0),
            Spacer(1),
            Markdown(changelog_markdown, 1, 1, self._get_markdown_theme_with_settings()),
            DynamicBorder(),
        )

    def _get_app_key_display(self, action: str) -> str:
        """Get capitalized display string for an app keybinding action."""
        return key_display_text(action)

    def _get_editor_key_display(self, action: str) -> str:
        """Get capitalized display string for an editor keybinding action."""
        return key_display_text(action)

    def _handle_hotkeys_command(self) -> None:
        # Navigation keybindings
        cursor_up = self._get_editor_key_display("tui.editor.cursorUp")
        cursor_down = self._get_editor_key_display("tui.editor.cursorDown")
        cursor_left = self._get_editor_key_display("tui.editor.cursorLeft")
        cursor_right = self._get_editor_key_display("tui.editor.cursorRight")
        cursor_word_left = self._get_editor_key_display("tui.editor.cursorWordLeft")
        cursor_word_right = self._get_editor_key_display("tui.editor.cursorWordRight")
        cursor_line_start = self._get_editor_key_display("tui.editor.cursorLineStart")
        cursor_line_end = self._get_editor_key_display("tui.editor.cursorLineEnd")
        jump_forward = self._get_editor_key_display("tui.editor.jumpForward")
        jump_backward = self._get_editor_key_display("tui.editor.jumpBackward")
        page_up = self._get_editor_key_display("tui.editor.pageUp")
        page_down = self._get_editor_key_display("tui.editor.pageDown")

        # Editing keybindings
        submit = self._get_editor_key_display("tui.input.submit")
        new_line = self._get_editor_key_display("tui.input.newLine")
        delete_word_backward = self._get_editor_key_display("tui.editor.deleteWordBackward")
        delete_word_forward = self._get_editor_key_display("tui.editor.deleteWordForward")
        delete_to_line_start = self._get_editor_key_display("tui.editor.deleteToLineStart")
        delete_to_line_end = self._get_editor_key_display("tui.editor.deleteToLineEnd")
        yank = self._get_editor_key_display("tui.editor.yank")
        yank_pop = self._get_editor_key_display("tui.editor.yankPop")
        undo = self._get_editor_key_display("tui.editor.undo")
        tab = self._get_editor_key_display("tui.input.tab")

        # App keybindings
        interrupt = self._get_app_key_display("app.interrupt")
        clear = self._get_app_key_display("app.clear")
        exit_key = self._get_app_key_display("app.exit")
        suspend = self._get_app_key_display("app.suspend")
        cycle_thinking_level = self._get_app_key_display("app.thinking.cycle")
        cycle_model_forward = self._get_app_key_display("app.model.cycleForward")
        select_model = self._get_app_key_display("app.model.select")
        expand_tools = self._get_app_key_display("app.tools.expand")
        toggle_thinking = self._get_app_key_display("app.thinking.toggle")
        external_editor = self._get_app_key_display("app.editor.external")
        cycle_model_backward = self._get_app_key_display("app.model.cycleBackward")
        copy_message = self._get_app_key_display("app.message.copy")
        follow_up = self._get_app_key_display("app.message.followUp")
        dequeue = self._get_app_key_display("app.message.dequeue")
        paste_image = self._get_app_key_display("app.clipboard.pasteImage")

        hotkeys = f"""
**Navigation**
| Key | Action |
|-----|--------|
| `{cursor_up}` / `{cursor_down}` / `{cursor_left}` / `{cursor_right}` | Move cursor / browse history |
| `{cursor_word_left}` / `{cursor_word_right}` | Move by word |
| `{cursor_line_start}` | Start of line |
| `{cursor_line_end}` | End of line |
| `{jump_forward}` | Jump forward to character |
| `{jump_backward}` | Jump backward to character |
| `{page_up}` / `{page_down}` | Scroll by page |

**Editing**
| Key | Action |
|-----|--------|
| `{submit}` | Send message |
| `{new_line}` | New line |
| `{delete_word_backward}` | Delete word backwards |
| `{delete_word_forward}` | Delete word forwards |
| `{delete_to_line_start}` | Delete to start of line |
| `{delete_to_line_end}` | Delete to end of line |
| `{yank}` | Paste the most-recently-deleted text |
| `{yank_pop}` | Cycle through the deleted text after pasting |
| `{undo}` | Undo |

**Other**
| Key | Action |
|-----|--------|
| `{tab}` | Path completion / accept autocomplete |
| `{interrupt}` | Cancel autocomplete / abort streaming |
| `{clear}` | Clear editor (first) / exit (second) |
| `{exit_key}` | Exit (when editor is empty) |
| `{suspend}` | Suspend to background |
| `{cycle_thinking_level}` | Cycle thinking level |
| `{cycle_model_forward}` / `{cycle_model_backward}` | Cycle models |
| `{select_model}` | Open model selector |
| `{expand_tools}` | Toggle tool output expansion |
| `{toggle_thinking}` | Toggle thinking block visibility |
| `{external_editor}` | Edit message in external editor |
| `{copy_message}` | Copy selection or last assistant message |
| `{follow_up}` | Queue follow-up message |
| `{dequeue}` | Restore queued messages |
| `{paste_image}` | Paste files on macOS, images, or text from clipboard |
| `/` | Slash commands |
| `!` | Run bash command |
| `!!` | Run bash command (excluded from context) |
"""

        # Add extension-registered shortcuts
        extension_runner = self.session.extension_runner
        get_shortcuts = getattr(extension_runner, "get_shortcuts", None)
        shortcuts = get_shortcuts(self._keybindings.get_effective_config()) if get_shortcuts is not None else {}
        if shortcuts:
            hotkeys += "\n**Extensions**\n| Key | Action |\n|-----|--------|\n"
            for key, shortcut in shortcuts.items():
                description = shortcut.description if shortcut.description is not None else shortcut.extension_path
                key_display = format_key_text(key, {"capitalize": True})
                hotkeys += f"| `{key_display}` | {description} |\n"

        self._append_to_chat(
            Spacer(1),
            DynamicBorder(),
            ThemedText(lambda: theme.bold(theme.fg("accent", "Keyboard Shortcuts")), 1, 0),
            Spacer(1),
            Markdown(hotkeys.strip(), 1, 1, self._get_markdown_theme_with_settings()),
            DynamicBorder(),
        )

    def _handle_clear_command(self) -> None:
        """The new-session action and /new: the status indicator is cleared
        here; the new session is spawned."""
        self._clear_status_indicator()
        self._spawn_flow(self._new_session())

    async def _new_session(self) -> None:
        try:
            result = await self.runtime_host.new_session()
            if result.get("cancelled"):
                return
            # After the new session's chat reset (the rebind, inside `new_session`).
            self._append_to_chat(Spacer(1), ThemedText(lambda: theme.fg("accent", "✓ New session started"), 1, 1))
        except Exception as error:
            await self._handle_fatal_runtime_error("Failed to create session", error)

    async def _handle_debug_command(self) -> None:
        with self.ui.state_lock:
            width = self.ui.terminal.columns
            height = self.ui.terminal.rows
            all_lines = self.ui.render(width)

        debug_log_path = get_debug_log_path()
        debug_lines = [
            f"Debug output at {clock.now_iso()}",
            f"Terminal: {width}x{height}",
            f"Total lines: {len(all_lines)}",
            "",
            "=== All rendered lines with visible widths ===",
        ]
        for idx, line in enumerate(all_lines):
            vw = visible_width(line)
            escaped = json.dumps(line, ensure_ascii=False)
            debug_lines.append(f"[{idx}] (w={vw}) {escaped}")
        debug_lines.extend(["", "=== Agent messages (JSONL) ==="])
        debug_lines.extend(
            json.dumps(serialize_message(msg), ensure_ascii=False, separators=(",", ":"))
            for msg in self.session.messages
        )
        debug_lines.append("")
        debug_data = "\n".join(debug_lines)

        debug_log_file = fs.Path(debug_log_path)
        await debug_log_file.parent.mkdir(parents=True, exist_ok=True)
        await debug_log_file.write_text(debug_data, encoding="utf-8")

        self._append_to_chat(
            Spacer(1),
            ThemedText(
                lambda: f"{theme.fg('accent', '✓ Debug log written')}\n{theme.fg('muted', debug_log_path)}", 1, 1
            ),
        )

    def _handle_armin_says_hi(self) -> None:
        self._append_to_chat(Spacer(1), ArminComponent(self.ui))

    async def _handle_demented_elves(self) -> None:
        self._append_to_chat(Spacer(1), EarendilAnnouncementComponent(self.ui, await load_earendil_image_base64()))

    async def _handle_bash_command(self, command: str, exclude_from_context: bool = False) -> None:
        extension_runner = self.session.extension_runner

        # Emit user_bash event to let extensions intercept
        try:
            event_result = await extension_runner.emit_user_bash(
                {
                    "type": "user_bash",
                    "command": command,
                    "excludeFromContext": exclude_from_context,
                    "cwd": self.session_manager.get_cwd(),
                }
            )
        except Exception:
            # The extension runner already reported the error. Do not fall back to local execution.
            return

        # If extension returned a full result, use it directly
        if event_result and event_result.get("result"):
            result = event_result["result"]

            # Create UI component for display, show output and complete
            component = BashExecutionComponent(command, self.ui, exclude_from_context, self._output_pad)
            with self.ui.state_lock:
                self._mount_bash_component(component, self.session.is_streaming)
                if result.get("output"):
                    component.append_output(result["output"])
                component.set_complete(
                    result.get("exitCode"),
                    bool(result.get("cancelled")),
                    _partial_truncation_result(result.get("output") or "") if result.get("truncated") else None,
                    result.get("fullOutputPath"),
                )

            # Record the result in session
            await self.session.record_bash_result(
                command,
                BashResult(
                    output=result.get("output") or "",
                    exit_code=result.get("exitCode"),
                    cancelled=bool(result.get("cancelled")),
                    truncated=bool(result.get("truncated")),
                    full_output_path=result.get("fullOutputPath"),
                ),
                {"excludeFromContext": exclude_from_context},
            )
            self.ui.request_render()
            return

        # Normal execution path (possibly with custom operations)
        # (pi keeps the component in a field; a local, since two `!` commands
        # run concurrently here and each must keep its own output.)
        component = BashExecutionComponent(command, self.ui, exclude_from_context, self._output_pad)
        self._mount_bash_component(component, self.session.is_streaming)
        self.ui.request_render()

        def on_chunk(chunk: str) -> None:
            # On the executor's task, in order.
            with self.ui.state_lock:
                component.append_output(chunk)
            self.ui.request_render()

        try:
            result = await self.session.execute_bash(
                command,
                on_chunk,
                {
                    "excludeFromContext": exclude_from_context,
                    "operations": event_result.get("operations") if event_result else None,
                },
            )

            with self.ui.state_lock:
                component.set_complete(
                    result.exit_code,
                    result.cancelled,
                    _partial_truncation_result(result.output) if result.truncated else None,
                    result.full_output_path,
                )
        except Exception as error:
            with self.ui.state_lock:
                component.set_complete(None, False)
                self.show_error(f"Bash command failed: {error if str(error) else 'Unknown error'}")

        self.ui.request_render()

    def _mount_bash_component(self, component, deferred: bool) -> None:
        with self.ui.state_lock:
            if deferred:
                # Show in pending area when agent is streaming
                # (`_flush_pending_bash_components` moves the pending list)
                self._pending_messages_container.add_child(component)
                self._pending_bash_components.append(component)
            else:
                # Show in chat immediately when agent is idle
                self._chat_container.add_child(component)

    def handle_compact_command(self, custom_instructions: str | None = None) -> None:
        """From the submit handler: the status indicator is cleared here; the
        compaction is spawned."""
        self._clear_status_indicator()
        self._spawn_flow(self._compact(custom_instructions))

    async def _compact(self, custom_instructions: str | None) -> None:
        with contextlib.suppress(Exception):
            # Ignore, will be emitted as an event
            await self.session.compact(custom_instructions)

    async def stop(self, fullscreen_exit_output: str | None = None) -> None:
        if fullscreen_exit_output is None:
            fullscreen_exit_output = self.settings_manager.get_fullscreen_exit_output()
        with self.ui.state_lock:
            self._dispose_active_selector()
            if self.settings_manager.get_show_terminal_progress():
                self.ui.terminal.set_progress(False)
            self._clear_status_indicator()
        await self._theme_controller.disable_auto_sync()
        with self.ui.state_lock:
            self._clear_extension_terminal_input_listeners()
            self._footer.dispose()
            self._footer_data_provider.dispose()
            if self._unsubscribe:
                self._unsubscribe()
        if self._is_initialized:
            await self._stop_interactive_tui(fullscreen_exit_output)
            self._is_initialized = False
        # The terminal's queues outlive renderer stops (suspend, switches);
        # this is the app's shutdown, so they end here, once, putting out what
        # the terminal still holds. stdout/stderr writes go straight to the
        # fds again from here.
        detach_terminal()
        await self.ui.close()
        self._unregister_signal_handlers()
