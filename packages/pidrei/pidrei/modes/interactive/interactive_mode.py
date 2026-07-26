"""Mirror of pi coding-agent src/modes/interactive/interactive-mode.ts.

Interactive mode for the coding agent. Handles TUI rendering and user
interaction, delegating business logic to AgentSession.

Parallelism/async deltas vs pi's single-threaded runtime:
- fire-and-forget promises become ``tonio.spawn.without_tracking`` tasks
- pi async methods whose synchronous prefix is observable are split into a
  sync prologue returning the awaitable remainder
- ``ui.start()``/``ui.stop()`` are awaited (the tonio TUI driver is async)
"""

import contextlib
import errno
import json
import os
import re
import signal
import sys
import tempfile
import time
import traceback
import uuid
from datetime import UTC, datetime

import tonio.colored as tonio
from tonio.colored import signals as tonio_signals

from pidrei_agent.harness.session.serde import serialize_message
from pidrei_ai.utils.cancel import CancelToken as AiCancelToken
from pidrei_tui import (
    TUI,
    CombinedAutocompleteProvider,
    Container,
    Markdown,
    ProcessTerminal,
    Spacer,
    Text,
    TruncatedText,
    fuzzy_filter,
    get_capabilities,
    hyperlink,
    matches_key,
    set_keybindings,
    visible_width,
)

from ...config import (
    APP_NAME,
    APP_TITLE,
    CONFIG_DIR_NAME,
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
from ...core.exec import exec_command
from ...core.footer_data_provider import FooterDataProvider
from ...core.http_config import format_http_idle_timeout_ms
from ...core.keybindings import KeybindingsManager
from ...core.messages import create_compaction_summary_message
from ...core.model_resolver import (
    DEFAULT_MODEL_PER_PROVIDER,
    find_exact_model_reference_match,
    resolve_model_scope,
)
from ...core.package_manager import DefaultPackageManager
from ...core.session_cwd import MissingSessionCwdError, format_missing_session_cwd_prompt
from ...core.session_manager import SessionManager, session_entry_to_context_messages
from ...core.slash_commands import BUILTIN_SLASH_COMMANDS
from ...core.telemetry import is_install_telemetry_enabled
from ...core.tools.truncate import TruncationResult
from ...core.trust_manager import ProjectTrustStore, has_trust_requiring_project_resources
from ...core.usage_totals import get_usage_cost_breakdown
from ...utils.changelog import get_new_entries, normalize_changelog_links, parse_changelog
from ...utils.clipboard import copy_to_clipboard, read_clipboard_text
from ...utils.clipboard_image import extension_for_image_mime_type, read_clipboard_image
from ...utils.colors import dim
from ...utils.git import parse_git_url
from ...utils.paths import get_cwd_relative_path
from ...utils.shell import kill_tracked_detached_children
from ...utils.tools_manager import ensure_tool
from ...utils.user_agent import get_pidrei_user_agent
from ...utils.version_check import check_for_new_version
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
    DaxnutsComponent,
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
    raw_key_hint,
)
from .external_editor import edit_in_external_editor
from .model_search import get_model_search_text
from .theme import (
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


def is_expandable(obj) -> bool:
    """Whether a component can be expanded/collapsed."""
    return callable(getattr(obj, "set_expanded", None))


class ExpandableText(Text):
    def __init__(self, get_collapsed_text, get_expanded_text, expanded=False, padding_x=0, padding_y=0) -> None:
        super().__init__(get_expanded_text() if expanded else get_collapsed_text(), padding_x, padding_y)
        self._get_collapsed_text = get_collapsed_text
        self._get_expanded_text = get_expanded_text

    def set_expanded(self, expanded: bool) -> None:
        self.set_text(self._get_expanded_text() if expanded else self._get_collapsed_text())


def _is_custom_session_entry(item) -> bool:
    return isinstance(item, dict) and item.get("type") == "custom"


_DEAD_TERMINAL_ERRNOS = {errno.EIO, errno.EPIPE, errno.ENOTCONN}


def is_dead_terminal_error(error) -> bool:
    return isinstance(error, OSError) and error.errno in _DEAD_TERMINAL_ERRNOS


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


_SAFE_SHELL_VALUE_RE = re.compile(r"[^a-zA-Z0-9_\-./~:@]")


def _quote_if_needed(value: str) -> str:
    if len(value) > 0 and not _SAFE_SHELL_VALUE_RE.search(value):
        return value
    escaped = value.replace("'", "'\\''")
    return f"'{escaped}'"


def format_resume_command(session_manager) -> str | None:
    if not sys.stdout.isatty():
        return None
    if not session_manager.is_persisted():
        return None

    session_file = session_manager.get_session_file()
    if not session_file or not os.path.exists(session_file):
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
        }
    return sorted(by_id.values(), key=lambda p: (p["name"].lower(), p["name"]))


def _get_login_provider_search_text(provider: dict) -> str:
    auth_types = " ".join(
        f"{auth_type} {format_auth_selector_provider_type(auth_type)}" for auth_type in provider["authTypes"]
    )
    return f"{provider['id']} {provider['name']} {auth_types}"


def _format_login_provider_completion_description(provider: dict) -> str:
    auth_types = "/".join(format_auth_selector_provider_type(auth_type) for auth_type in provider["authTypes"])
    return auth_types if provider["name"] == provider["id"] else f"{provider['name']} · {auth_types}"


class InteractiveMode:
    """Options: ``{"migratedProviders"?, "modelFallbackMessage"?,
    "autoTrustOnReloadCwd"?, "initialMessage"?, "initialImages"?,
    "initialMessages"?, "verbose"?}``."""

    def __init__(self, runtime_host, options: dict | None = None) -> None:
        options = options or {}
        self.runtime_host = runtime_host
        self._options = options
        self._auto_trust_on_reload_cwd = options.get("autoTrustOnReloadCwd")
        self.runtime_host.set_before_session_invalidate(lambda: self._reset_extension_ui())
        self.runtime_host.set_rebind_session(
            lambda _session=None: self._rebind_current_session({"renderBeforeBind": True})
        )
        self._version = VERSION
        self.ui = TUI(ProcessTerminal(), self.settings_manager.get_show_hardware_cursor(), get_agent_dir())
        self.ui.set_clear_on_shrink(self.settings_manager.get_clear_on_shrink())
        self._header_container = Container()
        self._loaded_resources_container = Container()
        self._chat_container = Container()
        self._pending_messages_container = Container()
        self._status_container = Container()
        self._widget_container_above = Container()
        self._widget_container_below = Container()
        self._keybindings = KeybindingsManager.create()
        set_keybindings(self._keybindings)
        editor_padding_x = self.settings_manager.get_editor_padding_x()
        autocomplete_max_visible = self.settings_manager.get_autocomplete_max_visible()
        self._default_editor = CustomEditor(
            self.ui,
            get_editor_theme(),
            self._keybindings,
            {"paddingX": editor_padding_x, "autocompleteMaxVisible": autocomplete_max_visible},
        )
        self.editor = self._default_editor
        self._editor_component_factory = None
        self._autocomplete_provider = None
        self._autocomplete_provider_wrappers: list = []
        self._fd_path: str | None = None
        self._editor_container = Container()
        self._editor_container.add_child(self.editor)
        self._footer_data_provider = FooterDataProvider(self.session_manager.get_cwd())
        self._footer = FooterComponent(self.session, self._footer_data_provider)
        self._footer.set_auto_compact_enabled(self.session.auto_compaction_enabled)

        self._is_initialized = False
        self._on_input_callback = None
        self._pending_user_inputs: list = []
        self._active_status_indicator = None
        self._idle_status = IdleStatus()
        self._working_message: str | None = None
        self._working_visible = True
        self._working_indicator_options = None
        self._default_working_message = "Working..."
        self._default_hidden_thinking_label = "Thinking..."
        self._hidden_thinking_label = self._default_hidden_thinking_label

        self._last_sigint_time = 0.0
        self._last_escape_time = 0.0
        self._is_shutting_down = False
        # Bumped on unregister so stale signal watchers turn inert. tonio's
        # signal receiver cannot be cancelled from outside while blocked, but
        # every unregister call site exits the process right after, so an
        # inert watcher never outlives anything that matters.
        self._signal_watch_generation = 0
        self._changelog_markdown: str | None = None
        self._startup_notices_shown = False
        self._anthropic_subscription_warning_shown = False

        # Status line tracking (for mutating immediately-sequential status
        # updates)
        self._last_status_spacer = None
        self._last_status_text = None

        # Streaming message tracking
        self._streaming_component = None
        self._streaming_message = None

        # Tool execution tracking: tool_call_id -> component
        self._pending_tools: dict = {}

        # Tool output expansion state
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

        # Track current bash execution component
        self._bash_component = None

        # Track pending bash components (shown in pending area, moved to
        # chat on submit)
        self._pending_bash_components: list = []

        # Auto-compaction / auto-retry state
        self._auto_compaction_escape_handler = None
        self._retry_escape_handler = None

        # Messages queued while compaction is running:
        # {"text", "mode": "steer" | "followUp"} records
        self._compaction_queued_messages: list = []

        # Shutdown state
        self._shutdown_requested = False

        # Extension UI state
        self._extension_selector = None
        self._extension_input = None
        self._extension_editor = None
        self._extension_terminal_input_unsubscribers: set = set()

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
            self.settings_manager,
            lambda message: self.show_error(message),
            lambda: self._update_editor_border_color(),
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
        if source_info is None:
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
                # Get available models (scoped or from registry)
                if self.session.scoped_models:
                    models = [s["model"] if isinstance(s, dict) else s.model for s in self.session.scoped_models]
                else:
                    models = await self.session.model_runtime.get_available()

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

        login_command = next((command for command in slash_commands if command["name"] == "login"), None)
        if login_command is not None:

            def get_login_completions(prefix: str):
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
        if self._startup_notices_shown:
            return
        self._startup_notices_shown = True

        if not self._changelog_markdown:
            return

        if self._chat_container.children:
            self._chat_container.add_child(Spacer(1))
        self._chat_container.add_child(DynamicBorder())
        if self.settings_manager.get_collapse_changelog():
            version_match = re.search(r"##\s+\[?(\d+\.\d+\.\d+)\]?", self._changelog_markdown)
            latest_version = version_match.group(1) if version_match else self._version
            condensed_text = (
                f"Updated to v{latest_version}. Use {theme.bold('/changelog')} to view full changelog."
            )
            self._chat_container.add_child(Text(condensed_text, 1, 0))
        else:
            self._chat_container.add_child(Text(theme.bold(theme.fg("accent", "What's New")), 1, 0))
            self._chat_container.add_child(Spacer(1))
            self._chat_container.add_child(
                Markdown(self._changelog_markdown.strip(), 1, 0, self._get_markdown_theme_with_settings())
            )
            self._chat_container.add_child(Spacer(1))
        self._chat_container.add_child(DynamicBorder())

    async def init(self) -> None:
        if self._is_initialized:
            return

        self._register_signal_handlers()

        # Load changelog (only show new entries, skip for resumed sessions)
        self._changelog_markdown = self._get_changelog_for_display()

        # Ensure fd and rg are available. Both are needed: fd for
        # autocomplete, rg for the grep tool and bash commands.
        fd_path = ensure_tool("fd")
        ensure_tool("rg")
        self._fd_path = fd_path

        if self.session.scoped_models and (self._options.get("verbose") or not self.settings_manager.get_quiet_startup()):
            model_parts = []
            for sm in self.session.scoped_models:
                model = sm["model"] if isinstance(sm, dict) else sm.model
                thinking_level = sm.get("thinkingLevel") if isinstance(sm, dict) else getattr(sm, "thinking_level", None)
                thinking_str = f":{thinking_level}" if thinking_level else ""
                model_parts.append(f"{model.id}{thinking_str}")
            model_list = ", ".join(model_parts)
            cycle_keys = self._keybindings.get_keys("app.model.cycleForward")
            cycle_hint = (
                theme.fg("muted", f" ({format_key_text('/'.join(cycle_keys), {'capitalize': True})} to cycle)")
                if cycle_keys
                else ""
            )
            print(theme.fg("dim", f"Model scope: {model_list}{cycle_hint}"))

        # Add header container as first child. Populate it after applying
        # theme settings. Keep loaded resources before chat so restored
        # session messages never precede them.
        self.ui.add_child(self._header_container)
        self.ui.add_child(self._loaded_resources_container)

        self.ui.add_child(self._chat_container)
        self.ui.add_child(self._pending_messages_container)
        self.ui.add_child(self._status_container)
        self._render_widgets()  # Initialize with default spacer
        self.ui.add_child(self._widget_container_above)
        self.ui.add_child(self._editor_container)
        self.ui.add_child(self._widget_container_below)
        self.ui.add_child(self._footer)
        self.ui.set_focus(self.editor)

        self._setup_key_handlers()
        self._setup_editor_submit_handler()

        # Start the UI before initializing extensions so session_start
        # handlers can use interactive dialogs
        await self.ui.start()
        self._is_initialized = True

        await self._theme_controller.apply_from_settings()

        # Add header with keybindings from config (unless silenced)
        if self._options.get("verbose") or not self.settings_manager.get_quiet_startup():
            logo = theme.bold(theme.fg("accent", APP_NAME)) + theme.fg("dim", f" v{self._version}")

            expanded_instructions = "\n".join(
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
                    key_hint("app.clipboard.pasteImage", "to paste image (with text fallback)"),
                    raw_key_hint("drop files", "to attach"),
                ]
            )
            compact_instructions = theme.fg("muted", " · ").join(
                [
                    key_hint("app.interrupt", "interrupt"),
                    raw_key_hint(f"{key_text('app.clear')}/{key_text('app.exit')}", "clear/exit"),
                    raw_key_hint("/", "commands"),
                    raw_key_hint("!", "bash"),
                    key_hint("app.tools.expand", "more"),
                ]
            )
            compact_onboarding = theme.fg(
                "dim",
                f"Press {key_text('app.tools.expand')} to show full startup help and loaded resources.",
            )
            onboarding = theme.fg(
                "dim",
                f"{APP_NAME} can explain its own features and look up its docs. "
                f"Ask it how to use or extend {APP_NAME}.",
            )
            self._built_in_header = ExpandableText(
                lambda: f"{logo}\n{compact_instructions}\n{compact_onboarding}\n\n{onboarding}",
                lambda: f"{logo}\n{expanded_instructions}\n\n{onboarding}",
                self._get_startup_expansion_state(),
                1,
                0,
            )

            # Setup UI layout
            self._header_container.add_child(Spacer(1))
            self._header_container.add_child(self._built_in_header)
            self._header_container.add_child(Spacer(1))
        else:
            # Minimal header when silenced
            self._built_in_header = Text("", 0, 0)
            self._header_container.add_child(self._built_in_header)
        self.ui.request_render()

        # Initialize extensions first so resources are shown before messages
        await self._rebind_current_session()

        # Render initial messages AFTER showing loaded resources
        self._render_initial_messages()

        # Set up theme file watcher
        def handle_theme_change() -> None:
            self.ui.invalidate()
            self._update_editor_border_color()
            self.ui.request_render()

        on_theme_change(handle_theme_change)

        # Set up git branch watcher (uses provider instead of footer)
        self._footer_data_provider.on_branch_change(lambda: self.ui.request_render())

        # Initialize available provider count for footer display
        await self._update_available_provider_count()

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
                    await self.session.model_runtime.refresh()
                    await self._update_available_provider_count()

            tonio.spawn.without_tracking(refresh_models())

        # Start version check asynchronously
        async def version_check() -> None:
            new_release = await check_for_new_version(self._version)
            if new_release:
                self._show_new_version_notification(new_release)

        tonio.spawn.without_tracking(version_check())

        # Start package update check asynchronously
        async def package_update_check() -> None:
            updates = await self._check_for_package_updates()
            if updates:
                self._show_package_update_notification(updates)

        tonio.spawn.without_tracking(package_update_check())

        # Check tmux keyboard setup asynchronously
        async def tmux_check() -> None:
            warning = await self._check_tmux_keyboard_setup()
            if warning:
                self.show_warning(warning)

        tonio.spawn.without_tracking(tmux_check())

        # Show startup warnings
        migrated_providers = self._options.get("migratedProviders")
        model_fallback_message = self._options.get("modelFallbackMessage")
        initial_message = self._options.get("initialMessage")
        initial_images = self._options.get("initialImages")
        initial_messages = self._options.get("initialMessages")

        if migrated_providers:
            self.show_warning(f"Migrated credentials to auth.json: {', '.join(migrated_providers)}")

        models_json_error = self.session.model_runtime.get_error()
        if models_json_error:
            self.show_error(f"models.json error: {models_json_error}")

        if model_fallback_message:
            self.show_warning(model_fallback_message)

        tonio.spawn.without_tracking(self._maybe_warn_about_anthropic_subscription_auth())

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
            check = getattr(package_manager, "check_for_available_updates", None)
            if check is None:
                # Package installation/update checking is Phase 5.
                return []
            updates = await check()
            return [update["displayName"] if isinstance(update, dict) else update.display_name for update in updates]
        except Exception:
            return []

    async def _check_tmux_keyboard_setup(self) -> str | None:
        if not os.environ.get("TMUX"):
            return None

        import subprocess

        def run_tmux_show(option: str) -> str | None:
            try:
                result = subprocess.run(  # noqa: S603
                    ["tmux", "show", "-gv", option],  # noqa: S607 - PATH lookup like pi's spawn
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    encoding="utf-8",
                    timeout=2,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                return None
            return result.stdout.strip() if result.returncode == 0 else None

        extended_keys = await tonio.spawn_blocking(run_tmux_show, "extended-keys")
        extended_keys_format = await tonio.spawn_blocking(run_tmux_show, "extended-keys-format")

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
                f"tmux extended-keys-format is xterm. {APP_NAME} works best with csi-u. "
                "Add `set -g extended-keys-format csi-u` to ~/.tmux.conf and restart tmux."
            )

        return None

    def _get_changelog_for_display(self) -> str | None:
        """Get changelog entries to display on startup.

        Only shows new entries since the last seen version; skips resumed
        sessions.
        """
        # Skip changelog for resumed/continued sessions (already have messages)
        if self.session.state.messages:
            return None

        last_version = self.settings_manager.get_last_changelog_version()
        changelog_path = get_changelog_path()
        entries = parse_changelog(changelog_path)

        if not last_version:
            # Fresh install - record the version, send telemetry, don't show
            # changelog
            self.settings_manager.set_last_changelog_version(VERSION)
            self._report_install_telemetry(VERSION)
            return None

        new_entries = get_new_entries(entries, last_version)
        if new_entries:
            self.settings_manager.set_last_changelog_version(VERSION)
            self._report_install_telemetry(VERSION)
            return "\n\n".join(normalize_changelog_links(e["content"], e) for e in new_entries)

        return None

    def _report_install_telemetry(self, version: str) -> None:
        if os.environ.get("PIDREI_OFFLINE"):
            return

        if not is_install_telemetry_enabled(self.settings_manager):
            return

        async def report() -> None:
            import urllib.parse

            from pidrei_ai.utils.http import request_timeout, shared_client

            with contextlib.suppress(Exception):
                await shared_client().get(
                    f"https://pi.dev/api/report-install?version={urllib.parse.quote(version)}",
                    headers={"User-Agent": get_pidrei_user_agent(version)},
                    timeout=request_timeout(5000),
                )

        tonio.spawn.without_tracking(report())

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
            result = f"~{result[len(home):]}"

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
                import posixpath

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
        import posixpath

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
                first_entry.get("collision") if isinstance(first_entry, dict) else getattr(first_entry, "collision", None)
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
        options = options or {}
        # Resource rendering is idempotent; chat clears no longer clear this
        # separate container.
        self._loaded_resources_container.clear()

        show_listing = (
            options.get("force") or self._options.get("verbose") or not self.settings_manager.get_quiet_startup()
        )
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

        def add_loaded_section(name: str, collapsed_body: str, expanded_body=None, color: str = "mdHeading") -> None:
            expanded = expanded_body if expanded_body is not None else collapsed_body
            section = ExpandableText(
                lambda name=name, body=collapsed_body, color=color: f"{section_header(name, color)}\n{body}",
                lambda name=name, body=expanded, color=color: f"{section_header(name, color)}\n{body}",
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
            context_files = self.session.resource_loader.get_agents_files()
            if context_files:
                self._loaded_resources_container.add_child(Spacer(1))
                context_list = "\n".join(
                    theme.fg("dim", f"  {self._format_display_path(f.path)}") for f in context_files
                )
                context_compact_list = format_compact_list(
                    [self._format_context_path(context_file.path) for context_file in context_files],
                    {"sort": False},
                )
                add_loaded_section("Context", context_compact_list, context_list)

            skills = skills_result.skills
            if skills:
                groups = self._build_scope_groups(
                    [{"path": skill.file_path, "sourceInfo": skill.source_info} for skill in skills]
                )
                skill_list = self._format_scope_groups(
                    groups,
                    {
                        "formatPath": lambda item: self._format_display_path(item["path"]),
                        "formatPackagePath": lambda item, source: self._get_short_path(
                            item["path"], item.get("sourceInfo")
                        ),
                    },
                )
                skill_compact_list = format_compact_list([skill.name for skill in skills])
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

                template_list = self._format_scope_groups(
                    groups,
                    {
                        "formatPath": format_template,
                        "formatPackagePath": format_template,
                    },
                )
                prompt_compact_list = format_compact_list([f"/{template.name}" for template in templates])
                add_loaded_section("Prompts", prompt_compact_list, template_list)

            if extensions:
                groups = self._build_scope_groups(extensions)
                ext_list = self._format_scope_groups(
                    groups,
                    {
                        "formatPath": lambda item: self._format_extension_display_path(item["path"]),
                        "formatPackagePath": lambda item, source: self._format_extension_display_path(
                            self._get_short_path(item["path"], item.get("sourceInfo"))
                        ),
                    },
                )
                extension_compact_list = format_compact_list(self._get_compact_extension_labels(extensions))
                add_loaded_section("Extensions", extension_compact_list, ext_list, "mdHeading")

            # Show loaded themes (excluding built-in)
            custom_themes = [t for t in themes_result["themes"] if t.source_path]
            if custom_themes:
                groups = self._build_scope_groups(
                    [
                        {"path": loaded_theme.source_path, "sourceInfo": loaded_theme.source_info}
                        for loaded_theme in custom_themes
                    ]
                )
                theme_list = self._format_scope_groups(
                    groups,
                    {
                        "formatPath": lambda item: self._format_display_path(item["path"]),
                        "formatPackagePath": lambda item, source: self._get_short_path(
                            item["path"], item.get("sourceInfo")
                        ),
                    },
                )
                theme_compact_list = format_compact_list(
                    [
                        loaded_theme.name
                        if loaded_theme.name is not None
                        else self._get_compact_path_label(loaded_theme.source_path, loaded_theme.source_info)
                        for loaded_theme in custom_themes
                    ]
                )
                add_loaded_section("Themes", theme_compact_list, theme_list)

        if show_diagnostics:
            skill_diagnostics = skills_result.diagnostics
            if skill_diagnostics:
                warning_lines = self._format_diagnostics(skill_diagnostics, source_infos)
                self._loaded_resources_container.add_child(
                    Text(f"{theme.fg('warning', '[Skill conflicts]')}\n{warning_lines}", 0, 0)
                )
                self._loaded_resources_container.add_child(Spacer(1))

            prompt_diagnostics = prompts_result.diagnostics
            if prompt_diagnostics:
                warning_lines = self._format_diagnostics(prompt_diagnostics, source_infos)
                self._loaded_resources_container.add_child(
                    Text(f"{theme.fg('warning', '[Prompt conflicts]')}\n{warning_lines}", 0, 0)
                )
                self._loaded_resources_container.add_child(Spacer(1))

            extension_diagnostics: list = []
            extension_errors = self.session.resource_loader.get_extensions().errors
            for error in extension_errors:
                extension_diagnostics.append({"type": "error", "message": error.error, "path": error.path})

            runner = self.session.extension_runner
            get_command_diagnostics = getattr(runner, "get_command_diagnostics", None)
            if get_command_diagnostics is not None:
                extension_diagnostics.extend(get_command_diagnostics())
            extension_diagnostics.extend(self._get_built_in_command_conflict_diagnostics(runner))

            get_shortcut_diagnostics = getattr(runner, "get_shortcut_diagnostics", None)
            if get_shortcut_diagnostics is not None:
                extension_diagnostics.extend(get_shortcut_diagnostics())

            if extension_diagnostics:
                warning_lines = self._format_diagnostics(extension_diagnostics, source_infos)
                self._loaded_resources_container.add_child(
                    Text(f"{theme.fg('warning', '[Extension issues]')}\n{warning_lines}", 0, 0)
                )
                self._loaded_resources_container.add_child(Spacer(1))

            theme_diagnostics = themes_result["diagnostics"]
            if theme_diagnostics:
                warning_lines = self._format_diagnostics(theme_diagnostics, source_infos)
                self._loaded_resources_container.add_child(
                    Text(f"{theme.fg('warning', '[Theme conflicts]')}\n{warning_lines}", 0, 0)
                )
                self._loaded_resources_container.add_child(Spacer(1))

    async def _bind_current_session_extensions(self) -> None:
        """Initialize the extension system with TUI-based UI context."""
        from ...core.agent_session import ExtensionBindings

        ui_context = self._create_extension_ui_context()

        async def new_session_action(options=None):
            self._clear_status_indicator()
            try:
                return await self.runtime_host.new_session(options)
            except Exception as error:
                return await self._handle_fatal_runtime_error("Failed to create session", error)

        async def fork_action(entry_id, options=None):
            try:
                result = await self.runtime_host.fork(entry_id, options)
                if not result["cancelled"]:
                    self.editor.set_text(result.get("selectedText") or "")
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

            self._chat_container.clear()
            self._render_initial_messages()
            if result.editor_text and not self.editor.get_text().strip():
                self.editor.set_text(result.editor_text)
            self.show_status("Navigated to selected point")
            tonio.spawn.without_tracking(self._flush_compaction_queue({"willRetry": False}))
            return {"cancelled": False}

        async def switch_session_action(session_path, options=None):
            return await self._handle_resume_session(session_path, options)

        async def reload_action():
            await self._handle_reload_command()

        def shutdown_handler() -> None:
            self._shutdown_requested = True
            if self.session.is_idle:
                tonio.spawn.without_tracking(self.shutdown())

        await self.session.bind_extensions(
            ExtensionBindings(
                ui_context=ui_context,
                mode="tui",
                abort_handler=lambda: self._restore_queued_messages_to_editor({"abort": True}),
                command_context_actions={
                    "waitForIdle": lambda: self.session.wait_for_idle(),
                    "newSession": new_session_action,
                    "fork": fork_action,
                    "navigateTree": navigate_tree_action,
                    "switchSession": switch_session_action,
                    "reload": reload_action,
                },
                shutdown_handler=shutdown_handler,
                on_error=lambda error: self._show_extension_error(
                    error.extension_path, error.error, getattr(error, "stack", None)
                ),
            )
        )

        set_registered_themes(self.session.resource_loader.get_themes()["themes"])
        self._setup_autocomplete_provider()

        extension_runner = self.session.extension_runner
        self._setup_extension_shortcuts(extension_runner)
        self._show_loaded_resources({"force": False, "showDiagnosticsWhenQuiet": True})
        self._show_startup_notices_if_needed()

    def _apply_runtime_settings(self) -> None:
        # pi configures the undici HTTP dispatcher here; pidrei's HTTP
        # transport is punkreq's concern (see core/http_config.py).
        self._footer.set_session(self.session)
        self._footer.set_auto_compact_enabled(self.session.auto_compaction_enabled)
        self._footer_data_provider.set_cwd(self.session_manager.get_cwd())
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

    async def _rebind_current_session(self, options: dict | None = None) -> None:
        options = options or {}
        if self._unsubscribe is not None:
            self._unsubscribe()
        self._unsubscribe = None
        self._apply_runtime_settings()
        if options.get("renderBeforeBind"):
            self.render_current_session_state()
            self._subscribe_to_agent()
            await self._bind_current_session_extensions()
        else:
            await self._bind_current_session_extensions()
            self._subscribe_to_agent()
        await self._update_available_provider_count()
        self._update_editor_border_color()
        self._update_terminal_title()

    async def _handle_fatal_runtime_error(self, prefix: str, error) -> None:
        self.show_error(f"{prefix}: {error}")
        stop_theme_watcher()
        await self.stop()
        os._exit(1)

    def render_current_session_state(self) -> None:
        self._loaded_resources_container.clear()
        self._chat_container.clear()
        self._pending_messages_container.clear()
        self._compaction_queued_messages = []
        self._streaming_component = None
        self._streaming_message = None
        self._pending_tools.clear()
        self._render_initial_messages()

    def _get_registered_tool_definition(self, tool_name: str):
        """Get a registered tool definition by name (for custom rendering)."""
        return self.session.get_tool_definition(tool_name)

    def _setup_extension_shortcuts(self, extension_runner) -> None:
        """Set up keyboard shortcuts registered by extensions."""
        get_shortcuts = getattr(extension_runner, "get_shortcuts", None)
        shortcuts = get_shortcuts(self._keybindings.get_effective_config()) if get_shortcuts is not None else {}
        if not shortcuts:
            return

        # Create a context for shortcut handlers
        def create_context() -> dict:
            def compact(options=None):
                options = options or {}

                async def run_compact() -> None:
                    try:
                        result = await self.session.compact(options.get("customInstructions"))
                        on_complete = options.get("onComplete")
                        if on_complete is not None:
                            on_complete(result)
                    except Exception as error:
                        on_error = options.get("onError")
                        if on_error is not None:
                            on_error(error)

                tonio.spawn.without_tracking(run_compact())

            return {
                "ui": self._create_extension_ui_context(),
                "mode": "tui",
                "hasUI": True,
                "cwd": self.session_manager.get_cwd(),
                "sessionManager": self.session_manager,
                "modelRegistry": extension_runner.get_model_registry(),
                "model": self.session.model,
                "thinkingLevel": self.session.thinking_level,
                "isIdle": lambda: self.session.is_idle,
                "isProjectTrusted": lambda: self.settings_manager.is_project_trusted(),
                "signal": self.session.agent.signal,
                "abort": lambda: self._restore_queued_messages_to_editor({"abort": True}),
                "hasPendingMessages": lambda: self.session.pending_message_count > 0,
                "shutdown": lambda: setattr(self, "_shutdown_requested", True),
                "getContextUsage": lambda: self.session.get_context_usage,
                "compact": compact,
                "getSystemPrompt": lambda: self.session.system_prompt,
            }

        def on_extension_shortcut(data: str) -> bool:
            for shortcut_str, shortcut in shortcuts.items():
                if matches_key(data, shortcut_str):

                    async def run_handler(handler=shortcut) -> None:
                        try:
                            result = handler.handler(create_context())
                            if hasattr(result, "__await__"):
                                await result
                        except Exception as err:
                            self.show_error(f"Shortcut handler error: {err}")

                    tonio.spawn.without_tracking(run_handler())
                    return True
            return False

        self._default_editor.on_extension_shortcut = on_extension_shortcut

    def _set_extension_status(self, key: str, text) -> None:
        """Set extension status text in the footer."""
        self._footer_data_provider.set_extension_status(key, text)
        self.ui.request_render()

    def _show_status_indicator(self, indicator) -> None:
        if self._active_status_indicator is not None:
            self._active_status_indicator.dispose()
        self._active_status_indicator = indicator
        self._status_container.clear()
        self._status_container.add_child(indicator)

    def _clear_status_indicator(self, kind: str | None = None) -> None:
        if kind and (self._active_status_indicator is None or self._active_status_indicator.kind != kind):
            return
        had_active_status_indicator = self._active_status_indicator is not None
        if self._active_status_indicator is not None:
            self._active_status_indicator.dispose()
        self._active_status_indicator = None
        self._status_container.clear()
        if had_active_status_indicator and self.ui.get_clear_on_shrink():
            self._status_container.add_child(self._idle_status)

    def _set_working_visible(self, visible: bool) -> None:
        self._working_visible = visible
        if not visible:
            self._clear_status_indicator("working")
            self.ui.request_render()
            return
        if self.session.is_streaming and (
            self._active_status_indicator is None or self._active_status_indicator.kind != "working"
        ):
            self._show_status_indicator(
                WorkingStatusIndicator(
                    self.ui,
                    self._working_message if self._working_message is not None else self._default_working_message,
                    self._working_indicator_options,
                )
            )
        self.ui.request_render()

    def _set_working_indicator(self, options=None) -> None:
        self._working_indicator_options = options
        if self._active_status_indicator is not None and self._active_status_indicator.kind == "working":
            self._active_status_indicator.set_indicator(options)
        self.ui.request_render()

    def _set_hidden_thinking_label(self, label=None) -> None:
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
                container.add_child(Text(theme.fg("muted", "... (widget truncated)"), 1, 0))
            component = container
        else:
            # Factory function - create component
            component = content(self.ui, theme)

        target_map = self._extension_widgets_below if placement == "belowEditor" else self._extension_widgets_above
        target_map[key] = component
        self._render_widgets()

    def _clear_extension_widgets(self) -> None:
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
        if self._extension_selector is not None:
            self._hide_extension_selector()
        if self._extension_input is not None:
            self._hide_extension_input()
        if self._extension_editor is not None:
            self._hide_extension_editor()
        self.ui.hide_overlay()
        self._clear_extension_terminal_input_listeners()
        self._set_extension_footer(None)
        self._set_extension_header(None)
        self._clear_extension_widgets()
        self._footer_data_provider.clear_extension_statuses()
        self._footer.invalidate()
        self._autocomplete_provider_wrappers = []
        self._set_custom_editor_component(None)
        self._setup_autocomplete_provider()
        self._default_editor.on_extension_shortcut = None
        self._update_terminal_title()
        self._working_message = None
        self._working_visible = True
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
        # Dispose existing custom footer
        if self._custom_footer is not None and getattr(self._custom_footer, "dispose", None) is not None:
            self._custom_footer.dispose()

        # Remove current footer from UI
        if self._custom_footer is not None:
            self.ui.remove_child(self._custom_footer)
        else:
            self.ui.remove_child(self._footer)

        if factory is not None:
            # Create and add custom footer, passing the data provider
            self._custom_footer = factory(self.ui, theme, self._footer_data_provider)
            self.ui.add_child(self._custom_footer)
        else:
            # Restore built-in footer
            self._custom_footer = None
            self.ui.add_child(self._footer)

        self.ui.request_render()

    def _set_extension_header(self, factory) -> None:
        """Set a custom header component, or restore the built-in header."""
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
            self._custom_header = factory(self.ui, theme)
            if is_expandable(self._custom_header):
                self._custom_header.set_expanded(self._tool_output_expanded)
            if index != -1:
                self._header_container.children[index] = self._custom_header
            else:
                # If not found (e.g. built-in header was never added), add at
                # the top
                self._header_container.children.insert(0, self._custom_header)
        else:
            # Restore built-in header
            self._custom_header = None
            if is_expandable(self._built_in_header):
                self._built_in_header.set_expanded(self._tool_output_expanded)
            if index != -1:
                self._header_container.children[index] = self._built_in_header

        self.ui.request_render()

    def _add_extension_terminal_input_listener(self, handler):
        unsubscribe = self.ui.add_input_listener(handler)
        self._extension_terminal_input_unsubscribers.add(unsubscribe)

        def remove() -> None:
            unsubscribe()
            self._extension_terminal_input_unsubscribers.discard(unsubscribe)

        return remove

    def _clear_extension_terminal_input_listeners(self) -> None:
        for unsubscribe in self._extension_terminal_input_unsubscribers:
            unsubscribe()
        self._extension_terminal_input_unsubscribers.clear()

    def _create_project_trust_context(self, cwd: str) -> dict:
        ui = self._create_extension_ui_context()
        return {
            "cwd": cwd,
            "mode": "tui",
            "hasUI": True,
            "ui": {
                "select": ui["select"],
                "confirm": ui["confirm"],
                "input": ui["input"],
                "notify": ui["notify"],
            },
        }

    def _create_extension_ui_context(self) -> dict:
        """Create the ExtensionUIContext record for extensions."""

        def set_working_message(message=None) -> None:
            self._working_message = message
            if self._active_status_indicator is not None and self._active_status_indicator.kind == "working":
                self._active_status_indicator.set_message(
                    message if message is not None else self._default_working_message
                )

        def add_autocomplete_provider(factory) -> None:
            self._autocomplete_provider_wrappers.append(factory)
            self._setup_autocomplete_provider()

        def get_editor_text() -> str:
            get_expanded = getattr(self.editor, "get_expanded_text", None)
            return get_expanded() if get_expanded is not None else self.editor.get_text()

        def set_theme_from_extension(theme_or_name):
            from .theme import Theme as ThemeClass

            if isinstance(theme_or_name, ThemeClass):
                return self._theme_controller.set_theme_instance(theme_or_name)
            result = self._theme_controller.set_theme_name(theme_or_name)
            if result["success"] and self.settings_manager.get_theme() != theme_or_name:
                self.settings_manager.set_theme(theme_or_name)
            return result

        return {
            "select": lambda title, options, opts=None: self._show_extension_selector(title, options, opts),
            "confirm": lambda title, message, opts=None: self._show_extension_confirm(title, message, opts),
            "input": lambda title, placeholder=None, opts=None: self._show_extension_input(title, placeholder, opts),
            "notify": lambda message, type=None: self._show_extension_notify(message, type),
            "onTerminalInput": lambda handler: self._add_extension_terminal_input_listener(handler),
            "setStatus": lambda key, text: self._set_extension_status(key, text),
            "setWorkingMessage": set_working_message,
            "setWorkingVisible": lambda visible: self._set_working_visible(visible),
            "setWorkingIndicator": lambda options=None: self._set_working_indicator(options),
            "setHiddenThinkingLabel": lambda label=None: self._set_hidden_thinking_label(label),
            "setWidget": lambda key, content, options=None: self._set_extension_widget(key, content, options),
            "setFooter": lambda factory: self._set_extension_footer(factory),
            "setHeader": lambda factory: self._set_extension_header(factory),
            "setTitle": lambda title: self.ui.terminal.set_title(title),
            "custom": lambda factory, options=None: self._show_extension_custom(factory, options),
            "pasteToEditor": lambda text: self.editor.handle_input(f"\x1b[200~{text}\x1b[201~"),
            "setEditorText": lambda text: self.editor.set_text(text),
            "getEditorText": get_editor_text,
            "editor": lambda title, prefill=None: self._show_extension_editor(title, prefill),
            "addAutocompleteProvider": add_autocomplete_provider,
            "setEditorComponent": lambda factory: self._set_custom_editor_component(factory),
            "getEditorComponent": lambda: self._editor_component_factory,
            "theme": theme,
            "getAllThemes": lambda: get_available_themes_with_paths(),
            "getTheme": lambda name: get_theme_by_name(name),
            "setTheme": set_theme_from_extension,
            "getToolsExpanded": lambda: self._tool_output_expanded,
            "setToolsExpanded": lambda expanded: self.set_tools_expanded(expanded),
        }

    def _show_extension_selector(self, title: str, options: list, opts: dict | None = None):
        """Show a selector for extensions; returns an awaitable."""
        opts = opts or {}
        done = tonio.Event()
        outcome: dict = {"value": None}
        remove_abort = None

        signal = opts.get("signal")
        if signal is not None and signal.cancelled:

            async def already_aborted():
                return None

            return already_aborted()

        def settle(value) -> None:
            if done.is_set():
                return
            outcome["value"] = value
            if remove_abort is not None:
                remove_abort()
            self._hide_extension_selector()
            done.set()

        if signal is not None:
            remove_abort = signal.on_cancel(lambda: settle(None))

        self._extension_selector = ExtensionSelectorComponent(
            title,
            options,
            lambda option: settle(option),
            lambda: settle(None),
            {
                "tui": self.ui,
                "timeout": opts.get("timeout"),
                "onToggleToolsExpanded": lambda: self._toggle_tool_output_expansion(),
            },
        )

        self._editor_container.clear()
        self._editor_container.add_child(self._extension_selector)
        self.ui.set_focus(self._extension_selector)
        self.ui.request_render()

        async def wait():
            await done.wait(None)
            return outcome["value"]

        return wait()

    def _hide_extension_selector(self) -> None:
        if self._extension_selector is not None:
            self._extension_selector.dispose()
        self._editor_container.clear()
        self._editor_container.add_child(self.editor)
        self._extension_selector = None
        self.ui.set_focus(self.editor)
        self.ui.request_render()

    async def _show_extension_confirm(self, title: str, message: str, opts: dict | None = None) -> bool:
        """Show a confirmation dialog for extensions."""
        result = await self._show_extension_selector(f"{title}\n{message}", ["Yes", "No"], opts)
        return result == "Yes"

    async def _prompt_for_missing_session_cwd(self, error) -> str | None:
        confirmed = await self._show_extension_confirm(
            "Session cwd not found", format_missing_session_cwd_prompt(error.issue)
        )
        return error.issue.fallback_cwd if confirmed else None

    def _show_extension_input(self, title: str, placeholder=None, opts: dict | None = None):
        """Show a text input for extensions; returns an awaitable."""
        opts = opts or {}
        done = tonio.Event()
        outcome: dict = {"value": None}
        remove_abort = None

        signal = opts.get("signal")
        if signal is not None and signal.cancelled:

            async def already_aborted():
                return None

            return already_aborted()

        def settle(value) -> None:
            if done.is_set():
                return
            outcome["value"] = value
            if remove_abort is not None:
                remove_abort()
            self._hide_extension_input()
            done.set()

        if signal is not None:
            remove_abort = signal.on_cancel(lambda: settle(None))

        self._extension_input = ExtensionInputComponent(
            title,
            placeholder,
            lambda value: settle(value),
            lambda: settle(None),
            {"tui": self.ui, "timeout": opts.get("timeout")},
        )

        self._editor_container.clear()
        self._editor_container.add_child(self._extension_input)
        self.ui.set_focus(self._extension_input)
        self.ui.request_render()

        async def wait():
            await done.wait(None)
            return outcome["value"]

        return wait()

    def _hide_extension_input(self) -> None:
        if self._extension_input is not None:
            self._extension_input.dispose()
        self._editor_container.clear()
        self._editor_container.add_child(self.editor)
        self._extension_input = None
        self.ui.set_focus(self.editor)
        self.ui.request_render()

    def _show_extension_editor(self, title: str, prefill=None):
        """Show a multi-line editor for extensions (with Ctrl+G support)."""
        done = tonio.Event()
        outcome: dict = {"value": None}

        def settle(value) -> None:
            if done.is_set():
                return
            outcome["value"] = value
            self._hide_extension_editor()
            done.set()

        self._extension_editor = ExtensionEditorComponent(
            self.ui,
            self._keybindings,
            title,
            prefill,
            lambda value: settle(value),
            lambda: settle(None),
            None,
            self.settings_manager.get_external_editor_command(),
        )

        self._editor_container.clear()
        self._editor_container.add_child(self._extension_editor)
        self.ui.set_focus(self._extension_editor)
        self.ui.request_render()

        async def wait():
            await done.wait(None)
            return outcome["value"]

        return wait()

    def _hide_extension_editor(self) -> None:
        self._editor_container.clear()
        self._editor_container.add_child(self.editor)
        self._extension_editor = None
        self.ui.set_focus(self.editor)
        self.ui.request_render()

    def _set_custom_editor_component(self, factory) -> None:
        """Set a custom editor component from an extension.

        Pass None to restore the default editor.
        """
        self._editor_component_factory = factory

        # Save text from current editor before switching
        current_text = self.editor.get_text()

        self._editor_container.clear()

        if factory is not None:
            # Create the custom editor with tui, theme, and keybindings
            new_editor = factory(self.ui, get_editor_theme(), self._keybindings)

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

            # Set autocomplete if supported
            set_provider = getattr(new_editor, "set_autocomplete_provider", None)
            if set_provider is not None and self._autocomplete_provider is not None:
                set_provider(self._autocomplete_provider)

            # If extending CustomEditor, copy app-level handlers (duck typed)
            if isinstance(getattr(new_editor, "action_handlers", None), dict):
                if not getattr(new_editor, "on_escape", None):
                    new_editor.on_escape = lambda: (
                        self._default_editor.on_escape() if self._default_editor.on_escape else None
                    )
                if not getattr(new_editor, "on_ctrl_d", None):
                    new_editor.on_ctrl_d = lambda: (
                        self._default_editor.on_ctrl_d() if self._default_editor.on_ctrl_d else None
                    )
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

    async def _show_extension_custom(self, factory, options: dict | None = None):
        """Show a custom component with keyboard focus.

        Overlay mode renders on top of existing content.
        """
        options = options or {}
        saved_text = self.editor.get_text()
        is_overlay = bool(options.get("overlay"))

        def restore_editor() -> None:
            self._editor_container.clear()
            self._editor_container.add_child(self.editor)
            self.editor.set_text(saved_text)
            self.ui.set_focus(self.editor)
            self.ui.request_render()

        done = tonio.Event()
        outcome: dict = {"value": None}
        state: dict = {"component": None, "closed": False}

        def close(result=None) -> None:
            if state["closed"]:
                return
            state["closed"] = True
            if is_overlay:
                self.ui.hide_overlay()
            else:
                restore_editor()
            # Note: both branches above already request a render
            outcome["value"] = result
            with contextlib.suppress(Exception):
                component = state["component"]
                if component is not None and getattr(component, "dispose", None) is not None:
                    component.dispose()
            done.set()

        try:
            component = factory(self.ui, theme, self._keybindings, close)
            if hasattr(component, "__await__"):
                component = await component
        except Exception:
            if not state["closed"]:
                if not is_overlay:
                    restore_editor()
                raise
            component = None

        if not state["closed"] and component is not None:
            state["component"] = component
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
                    on_handle(handle)
            else:
                self._editor_container.clear()
                self._editor_container.add_child(component)
                self.ui.set_focus(component)
                self.ui.request_render()

        await done.wait(None)
        return outcome["value"]

    def _show_extension_error(self, extension_path: str, error: str, stack=None) -> None:
        """Show an extension error in the UI."""
        error_msg = f'Extension "{extension_path}" error: {error}'
        error_text = Text(theme.fg("error", error_msg), 1, 0)
        self._chat_container.add_child(error_text)
        if stack:
            # Show stack trace in dim color, indented (skip first line, it
            # duplicates the error message)
            stack_lines = "\n".join(
                theme.fg("dim", f"  {line.strip()}") for line in stack.split("\n")[1:]
            )
            if stack_lines:
                self._chat_container.add_child(Text(stack_lines, 1, 0))
        self.ui.request_render()

    # =========================================================================
    # Key Handlers
    # =========================================================================

    def _setup_key_handlers(self) -> None:
        # Set up handlers on the default editor - they use self.editor for
        # text access so they work correctly regardless of active editor
        def on_escape() -> None:
            if self.session.is_streaming:
                self._restore_queued_messages_to_editor({"abort": True})
            elif self.session.is_bash_running:
                self.session.abort_bash()
            elif self._is_bash_mode:
                self.editor.set_text("")
                self._is_bash_mode = False
                self._update_editor_border_color()
            elif not self.editor.get_text().strip():
                # Double-escape with empty editor triggers /tree, /fork, or
                # nothing based on setting
                action = self.settings_manager.get_double_escape_action()
                if action != "none":
                    now = time.time() * 1000
                    if now - self._last_escape_time < 500:
                        if action == "tree":
                            self._show_tree_selector()
                        else:
                            self._show_user_message_selector()
                        self._last_escape_time = 0
                    else:
                        self._last_escape_time = now

        self._default_editor.on_escape = on_escape

        # Register app action handlers
        self._default_editor.on_action("app.clear", lambda: self._handle_ctrl_c())
        self._default_editor.on_ctrl_d = lambda: self._handle_ctrl_d()
        self._default_editor.on_action(
            "app.suspend", lambda: tonio.spawn.without_tracking(self._handle_ctrl_z())
        )
        self._default_editor.on_action("app.thinking.cycle", lambda: self._cycle_thinking_level())
        self._default_editor.on_action(
            "app.model.cycleForward", lambda: tonio.spawn.without_tracking(self._cycle_model("forward"))
        )
        self._default_editor.on_action(
            "app.model.cycleBackward", lambda: tonio.spawn.without_tracking(self._cycle_model("backward"))
        )

        # Global debug handler on TUI (works regardless of focus)
        self.ui.on_debug = lambda: self._handle_debug_command()
        self._default_editor.on_action("app.model.select", lambda: self._show_model_selector())
        self._default_editor.on_action("app.tools.expand", lambda: self._toggle_tool_output_expansion())
        self._default_editor.on_action("app.thinking.toggle", lambda: self._toggle_thinking_block_visibility())
        self._default_editor.on_action(
            "app.editor.external", lambda: tonio.spawn.without_tracking(self._handle_open_external_editor())
        )
        self._default_editor.on_action(
            "app.message.copy", lambda: tonio.spawn.without_tracking(self._handle_copy_command())
        )
        self._default_editor.on_action(
            "app.message.followUp", lambda: tonio.spawn.without_tracking(self._handle_follow_up())
        )
        self._default_editor.on_action("app.message.dequeue", lambda: self._handle_dequeue())
        self._default_editor.on_action(
            "app.session.new", lambda: tonio.spawn.without_tracking(self._handle_clear_command())
        )
        self._default_editor.on_action("app.session.tree", lambda: self._show_tree_selector())
        self._default_editor.on_action("app.session.fork", lambda: self._show_user_message_selector())
        self._default_editor.on_action("app.session.resume", lambda: self._show_session_selector())

        def on_change(text: str) -> None:
            was_bash_mode = self._is_bash_mode
            self._is_bash_mode = text.lstrip().startswith("!")
            if was_bash_mode != self._is_bash_mode:
                self._update_editor_border_color()

        self._default_editor.on_change = on_change

        # Handle clipboard paste (triggered on Ctrl+V). Images are attached
        # by path; otherwise, paste plain text from the system clipboard.
        self._default_editor.on_paste_image = lambda: tonio.spawn.without_tracking(self._handle_clipboard_paste())

    async def _handle_clipboard_paste(self) -> None:
        try:
            image = await read_clipboard_image()
            if image:
                import tempfile

                ext = extension_for_image_mime_type(image["mimeType"]) or "png"
                file_name = f"{APP_NAME}-clipboard-{uuid.uuid4()}.{ext}"
                file_path = os.path.join(tempfile.gettempdir(), file_name)
                with open(file_path, "wb") as f:  # noqa: ASYNC230
                    f.write(image["bytes"])

                insert = getattr(self.editor, "insert_text_at_cursor", None)
                if insert is not None:
                    insert(file_path)
                self.ui.request_render()
                return

            text = await read_clipboard_text()
            if text:
                insert = getattr(self.editor, "insert_text_at_cursor", None)
                if insert is not None:
                    insert(text)
                self.ui.request_render()
        except Exception:
            # Silently ignore clipboard errors (permissions etc.)
            pass

    def _setup_editor_submit_handler(self) -> None:
        def on_submit(text: str) -> None:
            tonio.spawn.without_tracking(self._handle_editor_submit(text))

        self._default_editor.on_submit = on_submit

    async def _handle_editor_submit(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        # Handle commands
        if text == "/settings":
            self._show_settings_selector()
            self.editor.set_text("")
            return
        if text == "/scoped-models":
            self.editor.set_text("")
            await self._show_models_selector()
            return
        if text == "/model" or text.startswith("/model "):
            search_term = text[7:].strip() if text.startswith("/model ") else None
            self.editor.set_text("")
            await self._handle_model_command(search_term)
            return
        if text == "/export" or text.startswith("/export "):
            await self._handle_export_command(text)
            self.editor.set_text("")
            return
        if text == "/import" or text.startswith("/import "):
            await self.handle_import_command(text)
            self.editor.set_text("")
            return
        if text == "/share":
            await self._handle_share_command()
            self.editor.set_text("")
            return
        if text == "/copy":
            await self._handle_copy_command()
            self.editor.set_text("")
            return
        if text == "/name" or text.startswith("/name "):
            self._handle_name_command(text)
            self.editor.set_text("")
            return
        if text == "/session":
            self.handle_session_command()
            self.editor.set_text("")
            return
        if text == "/changelog":
            self._handle_changelog_command()
            self.editor.set_text("")
            return
        if text == "/hotkeys":
            self._handle_hotkeys_command()
            self.editor.set_text("")
            return
        if text == "/fork":
            self._show_user_message_selector()
            self.editor.set_text("")
            return
        if text == "/clone":
            self.editor.set_text("")
            await self.handle_clone_command()
            return
        if text == "/tree":
            self._show_tree_selector()
            self.editor.set_text("")
            return
        if text == "/trust":
            self._show_trust_selector()
            self.editor.set_text("")
            return
        if text == "/login" or text.startswith("/login "):
            provider_ref = text[7:].strip() if text.startswith("/login ") else None
            self.editor.set_text("")
            await self._handle_login_command(provider_ref)
            return
        if text == "/logout":
            self._show_oauth_selector("logout")
            self.editor.set_text("")
            return
        if text == "/new":
            self.editor.set_text("")
            await self._handle_clear_command()
            return
        if text == "/compact" or text.startswith("/compact "):
            custom_instructions = text[9:].strip() if text.startswith("/compact ") else None
            self.editor.set_text("")
            await self.handle_compact_command(custom_instructions)
            return
        if text == "/reload":
            self.editor.set_text("")
            await self._handle_reload_command()
            return
        if text == "/debug":
            self._handle_debug_command()
            self.editor.set_text("")
            return
        if text == "/arminsayshi":
            self._handle_armin_says_hi()
            self.editor.set_text("")
            return
        if text == "/dementedelves":
            self._handle_demented_elves()
            self.editor.set_text("")
            return
        if text == "/resume":
            self._show_session_selector()
            self.editor.set_text("")
            return
        if text == "/quit":
            self.editor.set_text("")
            await self.shutdown()
            return

        # Handle bash command (! for normal, !! for excluded from context)
        if text.startswith("!"):
            is_excluded = text.startswith("!!")
            command = text[2:].strip() if is_excluded else text[1:].strip()
            if command:
                if self.session.is_bash_running:
                    self.show_warning("A bash command is already running. Press Esc to cancel it first.")
                    self.editor.set_text(text)
                    return
                add_to_history = getattr(self.editor, "add_to_history", None)
                if add_to_history is not None:
                    add_to_history(text)
                await self._handle_bash_command(command, is_excluded)
                self._is_bash_mode = False
                self._update_editor_border_color()
                return

        # Queue input during compaction (extension commands run immediately)
        if self.session.is_compacting:
            if self._is_extension_command(text):
                add_to_history = getattr(self.editor, "add_to_history", None)
                if add_to_history is not None:
                    add_to_history(text)
                self.editor.set_text("")
                await self.session.prompt(text)
            else:
                self._queue_compaction_message(text, "steer")
            return

        # If streaming, use prompt() with steer behavior. This handles
        # extension commands (execute immediately), prompt template
        # expansion, and queueing
        if self.session.is_streaming:
            add_to_history = getattr(self.editor, "add_to_history", None)
            if add_to_history is not None:
                add_to_history(text)
            self.editor.set_text("")
            await self.session.prompt(text, PromptOptions(streaming_behavior="steer"))
            self._update_pending_messages_display()
            self.ui.request_render()
            return

        # Normal message submission. First, move any pending bash components
        # to chat
        self._flush_pending_bash_components()

        if self._on_input_callback is not None:
            self._on_input_callback(text)
        else:
            self._pending_user_inputs.append(text)
        add_to_history = getattr(self.editor, "add_to_history", None)
        if add_to_history is not None:
            add_to_history(text)

    def _subscribe_to_agent(self) -> None:
        # pidrei emits session events synchronously; handling them inline
        # preserves event ordering (pi awaits an async handler per event).
        self._unsubscribe = self.session.subscribe(self._handle_event)

    def _handle_event(self, event) -> None:  # noqa: C901
        # pi lazily awaits init() here; pidrei always subscribes after init.
        if not self._is_initialized:
            return

        self._footer.invalidate()
        event_type = getattr(event, "type", None)

        if event_type == "agent_start":
            self._pending_tools.clear()
            if self.settings_manager.get_show_terminal_progress():
                self.ui.terminal.set_progress(True)
            # Restore main escape handler if retry handler is still active
            # (retry success event fires later, but we need the main handler
            # now)
            if self._retry_escape_handler is not None:
                self._default_editor.on_escape = self._retry_escape_handler
                self._retry_escape_handler = None
            if self._working_visible:
                self._show_status_indicator(
                    WorkingStatusIndicator(
                        self.ui,
                        self._working_message if self._working_message is not None else self._default_working_message,
                        self._working_indicator_options,
                    )
                )
            else:
                self._clear_status_indicator()
            self.ui.request_render()

        elif event_type == "queue_update":
            self._update_pending_messages_display()
            self.ui.request_render()

        elif event_type == "entry_appended":
            if event.entry.get("type") == "custom":
                self._add_custom_entry_to_chat(event.entry)
                self.ui.request_render()

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
                )
                self._streaming_message = event.message
                self._chat_container.add_child(self._streaming_component)
                self._streaming_component.update_content(self._streaming_message)
                self.ui.request_render()

        elif event_type == "message_update":
            if self._streaming_component is not None and event.message.role == "assistant":
                self._streaming_message = event.message
                self._streaming_component.update_content(self._streaming_message)

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
                    self._streaming_message.error_message = error_message
                self._streaming_component.update_content(self._streaming_message)

                if self._streaming_message.stop_reason in ("aborted", "error"):
                    if not error_message:
                        error_message = self._streaming_message.error_message or "Error"
                    for component in self._pending_tools.values():
                        component.update_result(
                            {"content": [{"type": "text", "text": error_message}], "isError": True}
                        )
                    self._pending_tools.clear()
                else:
                    # Args are now complete - trigger diff computation for
                    # edit tools
                    for component in self._pending_tools.values():
                        component.set_args_complete()
                    self._maybe_show_cache_miss_notice(self._streaming_message)
                self._streaming_component = None
                self._streaming_message = None
                self._footer.invalidate()
            self.ui.request_render()

        elif event_type == "bash_execution_update":
            # The bash execution callback handles TUI output rendering.
            pass

        elif event_type == "tool_execution_start":
            component = self._pending_tools.get(event.tool_call_id)
            if component is None:
                component = ToolExecutionComponent(
                    event.tool_name,
                    event.tool_call_id,
                    event.args,
                    {
                        "showImages": self.settings_manager.get_show_images(),
                        "imageWidthCells": self.settings_manager.get_image_width_cells(),
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
            tonio.spawn.without_tracking(self._check_shutdown_requested())

        elif event_type == "compaction_start":
            if self.settings_manager.get_show_terminal_progress():
                self.ui.terminal.set_progress(True)
            # Keep editor active; submissions are queued during compaction.
            self._auto_compaction_escape_handler = self._default_editor.on_escape
            self._default_editor.on_escape = lambda: self.session.abort_compaction()
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
                from datetime import UTC, datetime

                self._chat_container.clear()
                self._rebuild_chat_from_messages()
                self._add_message_to_chat(
                    create_compaction_summary_message(
                        event.result.summary,
                        event.result.tokens_before,
                        datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
                    )
                )
                self._footer.invalidate()
            elif event.error_message:
                if event.reason == "manual":
                    self.show_error(event.error_message)
                else:
                    self._chat_container.add_child(Spacer(1))
                    self._chat_container.add_child(Text(theme.fg("error", event.error_message), 1, 0))
            tonio.spawn.without_tracking(self._flush_compaction_queue({"willRetry": event.will_retry}))
            self.ui.request_render()

        elif event_type == "auto_retry_start":
            # Set up escape to abort retry
            self._retry_escape_handler = self._default_editor.on_escape
            self._default_editor.on_escape = lambda: self.session.abort_retry()
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
                self.show_error(
                    f"Retry failed after {event.attempt} attempts: {event.final_error or 'Unknown error'}"
                )
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

    def show_status(self, message: str) -> None:
        """Show a status message in the chat.

        Back-to-back status messages update the previous status line instead
        of appending new ones, to avoid log spam.
        """
        children = self._chat_container.children
        last = children[-1] if children else None
        second_last = children[-2] if len(children) > 1 else None

        if (
            last is not None
            and second_last is not None
            and last is self._last_status_text
            and second_last is self._last_status_spacer
        ):
            self._last_status_text.set_text(theme.fg("dim", message))
            self.ui.request_render()
            return

        spacer = Spacer(1)
        text = Text(theme.fg("dim", message), 1, 0)
        self._chat_container.add_child(spacer)
        self._chat_container.add_child(text)
        self._last_status_spacer = spacer
        self._last_status_text = text
        self.ui.request_render()

    def _add_custom_entry_to_chat(self, entry: dict) -> None:
        renderer = self.session.extension_runner.get_entry_renderer(entry.get("customType"))
        if renderer is None:
            return
        component = CustomEntryComponent(entry, renderer)
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
            component = BashExecutionComponent(message.command, self.ui, message.exclude_from_context)
            if message.output:
                component.append_output(message.output)
            from types import SimpleNamespace

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
                component = CustomMessageComponent(message, renderer, self._get_markdown_theme_with_settings())
                component.set_expanded(self._tool_output_expanded)
                self._chat_container.add_child(component)
        elif role == "compactionSummary":
            self._chat_container.add_child(Spacer(1))
            component = CompactionSummaryMessageComponent(message, self._get_markdown_theme_with_settings())
            component.set_expanded(self._tool_output_expanded)
            self._chat_container.add_child(component)
        elif role == "branchSummary":
            self._chat_container.add_child(Spacer(1))
            component = BranchSummaryMessageComponent(message, self._get_markdown_theme_with_settings())
            component.set_expanded(self._tool_output_expanded)
            self._chat_container.add_child(component)
        elif role == "user":
            text_content = self._get_user_message_text(message)
            if text_content:
                if self._chat_container.children:
                    self._chat_container.add_child(Spacer(1))
                skill_block = parse_skill_block(text_content)
                if skill_block is not None:
                    # Render skill block (collapsible)
                    component = SkillInvocationMessageComponent(
                        skill_block, self._get_markdown_theme_with_settings()
                    )
                    component.set_expanded(self._tool_output_expanded)
                    self._chat_container.add_child(component)
                    # Render user message separately if present
                    if skill_block.user_message:
                        self._chat_container.add_child(Spacer(1))
                        user_component = UserMessageComponent(
                            skill_block.user_message, self._get_markdown_theme_with_settings(), self._output_pad
                        )
                        self._chat_container.add_child(user_component)
                else:
                    user_component = UserMessageComponent(
                        text_content, self._get_markdown_theme_with_settings(), self._output_pad
                    )
                    self._chat_container.add_child(user_component)
                if options.get("populateHistory"):
                    add_to_history = getattr(self.editor, "add_to_history", None)
                    if add_to_history is not None:
                        add_to_history(text_content)
        elif role == "assistant":
            assistant_component = AssistantMessageComponent(
                message,
                self._hide_thinking_block,
                self._get_markdown_theme_with_settings(),
                self._hidden_thinking_label,
                self._output_pad,
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
        # Cache-miss notices are not persisted; re-derive them from the full
        # entry list and re-inject them after the assistant messages that
        # paid for them.
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
                    component.update_result(
                        {"content": message.content, "details": message.details, "isError": message.is_error}
                    )
                    rendered_pending_tools.pop(message.tool_call_id, None)
            else:
                # All other messages use standard rendering
                self._add_message_to_chat(message, options)

        for tool_call_id, component in rendered_pending_tools.items():
            self._pending_tools[tool_call_id] = component
        self.ui.request_render()

    def _render_session_entries(self, entries: list, options: dict | None = None) -> None:
        """Render session entries to chat (initial load and post-compaction).

        options: {"updateFooter"?, "populateHistory"?}
        """
        items: list = []
        for entry in entries:
            if entry.get("type") == "custom":
                items.append(entry)
            else:
                items.extend(session_entry_to_context_messages(entry))
        self._render_session_items(items, options)

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
        text = theme.fg("warning", f"{label}: {re_billed}")
        self._chat_container.add_child(Spacer(1))
        self._chat_container.add_child(Text(text, 1, 0))

    def _render_initial_messages(self) -> None:
        entries = self.session_manager.build_context_entries()
        self._render_session_entries(entries, {"updateFooter": True, "populateHistory": True})
        self._render_project_trust_warning_if_needed()

        # Show compaction info if session was compacted
        all_entries = self.session_manager.get_entries()
        compaction_count = sum(1 for entry in all_entries if entry.get("type") == "compaction")
        if compaction_count > 0:
            times = "1 time" if compaction_count == 1 else f"{compaction_count} times"
            self.show_status(f"Session compacted {times}")

    def _render_project_trust_warning_if_needed(self) -> None:
        if self.settings_manager.is_project_trusted() or not has_trust_requiring_project_resources(
            self.session_manager.get_cwd()
        ):
            return

        if self._chat_container.children:
            self._chat_container.add_child(Spacer(1))
        self._chat_container.add_child(
            Text(
                theme.fg(
                    "warning",
                    f"This project is not trusted. Project {CONFIG_DIR_NAME} resources and packages "
                    f"are ignored. Use /trust to save a trust decision, then restart {APP_NAME}.",
                ),
                1,
                0,
            )
        )

    async def _get_user_input(self) -> str:
        if self._pending_user_inputs:
            return self._pending_user_inputs.pop(0)

        received = tonio.Event()
        outcome: dict = {"text": ""}

        def on_input(text: str) -> None:
            self._on_input_callback = None
            outcome["text"] = text
            received.set()

        self._on_input_callback = on_input
        await received.wait(None)
        return outcome["text"]

    def _rebuild_chat_from_messages(self) -> None:
        self._chat_container.clear()
        self._render_session_entries(self.session_manager.build_context_entries())

    # =========================================================================
    # Key handlers
    # =========================================================================

    def _handle_ctrl_c(self) -> None:
        now = time.time() * 1000
        if now - self._last_sigint_time < 500:
            tonio.spawn.without_tracking(self.shutdown())
        else:
            self.clear_editor()
            self._last_sigint_time = now

    def _handle_ctrl_d(self) -> None:
        # Only called when editor is empty (enforced by CustomEditor)
        tonio.spawn.without_tracking(self.shutdown())

    async def shutdown(self, options: dict | None = None) -> None:
        """Gracefully shutdown the agent.

        Stops the TUI before emitting shutdown events so extension UI cleanup
        cannot repaint the final frame while the process is exiting.
        """
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
            self._theme_controller.disable_auto_sync()
            try:
                await self.ui.terminal.drain_input(1000)
                await self.stop()
            except OSError as error:
                if is_dead_terminal_error(error):
                    self._emergency_terminal_exit()
                raise
            os._exit(0)

        # Interactive quit (Ctrl+D, Ctrl+C, /quit, extension shutdown()).
        # Stop the TUI before emitting shutdown events so extension UI cleanup
        # cannot repaint the final frame while the process is exiting.
        # Drain any in-flight Kitty key release events before stopping.
        # This prevents escape sequences from leaking to the parent shell over
        # slow SSH.
        self._theme_controller.disable_auto_sync()
        try:
            await self.ui.terminal.drain_input(1000)
            await self.stop()
        except OSError as error:
            if is_dead_terminal_error(error):
                self._emergency_terminal_exit()
            raise
        await self.runtime_host.dispose()

        resume_command = format_resume_command(self.session_manager)
        if resume_command:
            sys.stdout.write(f"{dim('To resume this session:')} {resume_command}\n")
            sys.stdout.flush()

        os._exit(0)

    def _emergency_terminal_exit(self) -> None:
        self._is_shutting_down = True
        self._unregister_signal_handlers()
        kill_tracked_detached_children()
        # The terminal is gone. Do not run normal shutdown because TUI and
        # extension cleanup can write restore sequences and re-trigger EIO.
        os._exit(129)

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
        if self._is_shutting_down:
            os._exit(1)
        self._is_shutting_down = True
        with contextlib.suppress(Exception):
            self._unregister_signal_handlers()
        with contextlib.suppress(Exception):
            kill_tracked_detached_children()
        with contextlib.suppress(Exception):
            await self.ui.stop()
        print(f"{APP_NAME} exiting due to uncaught exception:", file=sys.stderr)
        traceback.print_exception(error, file=sys.stderr)
        os._exit(1)

    async def _check_shutdown_requested(self) -> None:
        """Check if shutdown was requested and perform shutdown if so."""
        if not self._shutdown_requested:
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
        generation = self._signal_watch_generation

        async def watch_signals() -> None:
            try:
                with tonio_signals.signal_receiver(signal.SIGTERM, signal.SIGHUP) as receiver:
                    async for _sig in receiver:
                        if self._signal_watch_generation != generation:
                            return
                        kill_tracked_detached_children()
                        await self.shutdown({"fromSignal": True})
            except (ValueError, RuntimeError):
                # Signals can only be watched from the main thread (tests).
                return

        tonio.spawn.without_tracking(watch_signals())

    def _unregister_signal_handlers(self) -> None:
        self._signal_watch_generation += 1

    async def _handle_ctrl_z(self) -> None:
        """Suspend to background (Ctrl+Z) and restore the TUI on resume.

        Deviations: Python processes stay alive without pi's event-loop
        keep-alive timer, and SIGCONT is awaited through a tonio signal
        receiver instead of process.once. SIGINT is ignored while suspended
        so Ctrl+C in the terminal does not kill the backgrounded process.
        """
        try:
            previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        except ValueError:
            previous_sigint = None

        try:
            with tonio_signals.signal_receiver(signal.SIGCONT) as sigcont:
                # Stop the TUI (restore terminal to normal mode)
                await self.ui.stop()

                # Send SIGTSTP to the process group (pid 0 means all
                # processes in the group)
                os.kill(0, signal.SIGTSTP)

                # Stopped here until `fg`; SIGCONT resumes and wakes the
                # receiver so the TUI can be restored.
                async for _sig in sigcont:
                    break
        finally:
            if previous_sigint is not None:
                signal.signal(signal.SIGINT, previous_sigint)

        await self.ui.start()
        self.ui.request_render(True)

    async def _handle_follow_up(self) -> None:
        get_expanded = getattr(self.editor, "get_expanded_text", None)
        text = (get_expanded() if get_expanded is not None else self.editor.get_text()).strip()
        if not text:
            return

        add_to_history = getattr(self.editor, "add_to_history", None)

        # Queue input during compaction (extension commands execute immediately)
        if self.session.is_compacting:
            if self._is_extension_command(text):
                if add_to_history is not None:
                    add_to_history(text)
                self.editor.set_text("")
                await self.session.prompt(text)
            else:
                self._queue_compaction_message(text, "followUp")
            return

        # Alt+Enter queues a follow-up message (waits until agent finishes).
        # This handles extension commands (execute immediately), prompt
        # template expansion, and queueing
        if self.session.is_streaming:
            if add_to_history is not None:
                add_to_history(text)
            self.editor.set_text("")
            await self.session.prompt(text, PromptOptions(streaming_behavior="followUp"))
            self._update_pending_messages_display()
            self.ui.request_render()
        # If not streaming, Alt+Enter acts like regular Enter (trigger on_submit)
        elif self.editor.on_submit:
            self.editor.set_text("")
            self.editor.on_submit(text)

    def _handle_dequeue(self) -> None:
        restored = self._restore_queued_messages_to_editor()
        if restored == 0:
            self.show_status("No queued messages to restore")
        else:
            self.show_status(f"Restored {restored} queued message{'s' if restored > 1 else ''} to editor")

    def _update_editor_border_color(self) -> None:
        if self._is_bash_mode:
            self.editor.border_color = theme.get_bash_mode_border_color()
        else:
            level = self.session.thinking_level or "off"
            self.editor.border_color = theme.get_thinking_border_color(level)
        self.ui.request_render()

    def _cycle_thinking_level(self) -> None:
        new_level = self.session.cycle_thinking_level()
        if new_level is None:
            self.show_status("Current model does not support thinking")
        else:
            self._footer.invalidate()
            self._update_editor_border_color()
            self.show_status(f"Thinking level: {new_level}")

    async def _cycle_model(self, direction: str) -> None:
        try:
            result = await self.session.cycle_model(direction)
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
                tonio.spawn.without_tracking(self._maybe_warn_about_anthropic_subscription_auth(result.model))
        except Exception as error:
            self.show_error(str(error))

    def _toggle_tool_output_expansion(self) -> None:
        self.set_tools_expanded(not self._tool_output_expanded)

    def set_tools_expanded(self, expanded: bool) -> None:
        self._tool_output_expanded = expanded
        active_header = self._custom_header if self._custom_header is not None else self._built_in_header
        if is_expandable(active_header):
            active_header.set_expanded(expanded)
        for container in (self._loaded_resources_container, self._chat_container):
            for child in container.children:
                if is_expandable(child):
                    child.set_expanded(expanded)
        self.ui.request_render()

    def _toggle_thinking_block_visibility(self) -> None:
        self._hide_thinking_block = not self._hide_thinking_block
        self.settings_manager.set_hide_thinking_block(self._hide_thinking_block)

        # Rebuild chat from session messages
        self._chat_container.clear()
        self._rebuild_chat_from_messages()

        # If streaming, re-add the streaming component with updated visibility
        # and re-render
        if self._streaming_component is not None and self._streaming_message is not None:
            self._streaming_component.set_hide_thinking_block(self._hide_thinking_block)
            self._streaming_component.update_content(self._streaming_message)
            self._chat_container.add_child(self._streaming_component)

        self.show_status(f"Thinking blocks: {'hidden' if self._hide_thinking_block else 'visible'}")

    async def _handle_open_external_editor(self) -> None:
        editor_cmd = self.settings_manager.get_external_editor_command()
        get_expanded = getattr(self.editor, "get_expanded_text", None)
        content = get_expanded() if get_expanded is not None else self.editor.get_text()
        await self.ui.stop()
        try:
            result = await edit_in_external_editor({"command": editor_cmd, "content": content})
            if result["status"] == "complete":
                self.editor.set_text(result["content"])
        finally:
            await self.ui.start()
            self.ui.request_render(True)

    # =========================================================================
    # UI helpers
    # =========================================================================

    def clear_editor(self) -> None:
        self.editor.set_text("")
        self.ui.request_render()

    def show_error(self, error_message: str) -> None:
        self._chat_container.add_child(Spacer(1))
        self._chat_container.add_child(Text(theme.fg("error", f"Error: {error_message}"), 1, 0))
        self.ui.request_render()

    def show_warning(self, warning_message: str) -> None:
        self._chat_container.add_child(Spacer(1))
        self._chat_container.add_child(Text(theme.fg("warning", f"Warning: {warning_message}"), 1, 0))
        self.ui.request_render()

    def show_new_version_notification(self, release: dict) -> None:
        action = theme.fg("accent", f"{APP_NAME} update")
        update_instruction = theme.fg("muted", f"New version {release['version']} is available. Run ") + action
        changelog_url = "https://pi.dev/changelog"
        changelog_link = (
            hyperlink(theme.fg("accent", changelog_url), changelog_url)
            if get_capabilities()["hyperlinks"]
            else theme.fg("accent", changelog_url)
        )
        changelog_line = theme.fg("muted", "Changelog: ") + changelog_link
        note = (release.get("note") or "").strip()

        self._chat_container.add_child(Spacer(1))
        self._chat_container.add_child(DynamicBorder(lambda text: theme.fg("warning", text)))
        self._chat_container.add_child(
            Text(f"{theme.bold(theme.fg('warning', 'Update Available'))}\n{update_instruction}", 1, 0)
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
        self._chat_container.add_child(Text(changelog_line, 1, 0))
        self._chat_container.add_child(DynamicBorder(lambda text: theme.fg("warning", text)))
        self.ui.request_render()

    def show_package_update_notification(self, packages: list) -> None:
        action = theme.fg("accent", f"{APP_NAME} update --extensions")
        update_instruction = theme.fg("muted", "Package updates are available. Run ") + action
        package_lines = "\n".join(f"- {pkg}" for pkg in packages)

        self._chat_container.add_child(Spacer(1))
        self._chat_container.add_child(DynamicBorder(lambda text: theme.fg("warning", text)))
        self._chat_container.add_child(
            Text(
                f"{theme.bold(theme.fg('warning', 'Package Updates Available'))}\n{update_instruction}\n"
                f"{theme.fg('muted', 'Packages:')}\n{package_lines}",
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
        return {
            "steering": [
                *self.session.get_steering_messages(),
                *(msg["text"] for msg in self._compaction_queued_messages if msg["mode"] == "steer"),
            ],
            "followUp": [
                *self.session.get_follow_up_messages(),
                *(msg["text"] for msg in self._compaction_queued_messages if msg["mode"] == "followUp"),
            ],
        }

    def _clear_all_queues(self) -> dict:
        """Clear all queued messages and return their contents.

        Clears both session queue and compaction queue.
        """
        cleared = self.session.clear_queue()
        compaction_steering = [msg["text"] for msg in self._compaction_queued_messages if msg["mode"] == "steer"]
        compaction_follow_up = [msg["text"] for msg in self._compaction_queued_messages if msg["mode"] == "followUp"]
        self._compaction_queued_messages = []
        return {
            "steering": [*cleared["steering"], *compaction_steering],
            "followUp": [*cleared["followUp"], *compaction_follow_up],
        }

    def _update_pending_messages_display(self) -> None:
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
        """options: {"abort"?, "currentText"?}"""
        cleared = self._clear_all_queues()
        all_queued = [*cleared["steering"], *cleared["followUp"]]
        if not all_queued:
            self._update_pending_messages_display()
            if options and options.get("abort"):
                self.agent.abort()
            return 0
        queued_text = "\n\n".join(all_queued)
        current_text = options.get("currentText") if options else None
        if current_text is None:
            current_text = self.editor.get_text()
        combined_text = "\n\n".join(t for t in (queued_text, current_text) if t.strip())
        self.editor.set_text(combined_text)
        self._update_pending_messages_display()
        if options and options.get("abort"):
            self.agent.abort()
        return len(all_queued)

    def _queue_compaction_message(self, text: str, mode: str) -> None:
        self._compaction_queued_messages.append({"text": text, "mode": mode})
        add_to_history = getattr(self.editor, "add_to_history", None)
        if add_to_history is not None:
            add_to_history(text)
        self.editor.set_text("")
        self._update_pending_messages_display()
        self.show_status("Queued message for after compaction")

    def _is_extension_command(self, text: str) -> bool:
        if not text.startswith("/"):
            return False

        extension_runner = self.session.extension_runner

        space_index = text.find(" ")
        command_name = text[1:] if space_index == -1 else text[1:space_index]
        return extension_runner.get_command(command_name) is not None

    async def _flush_compaction_queue(self, options: dict | None = None) -> None:
        if not self._compaction_queued_messages:
            return

        queued_messages = list(self._compaction_queued_messages)
        self._compaction_queued_messages = []
        self._update_pending_messages_display()

        def restore_queue(error) -> None:
            self.session.clear_queue()
            self._compaction_queued_messages = queued_messages
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

            tonio.spawn.without_tracking(send_first_prompt())

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
        """Move pending bash components from pending area to chat"""
        for component in self._pending_bash_components:
            self._pending_messages_container.remove_child(component)
            self._chat_container.add_child(component)
        self._pending_bash_components = []

    # =========================================================================
    # Selectors
    # =========================================================================

    def _show_selector(self, create) -> None:
        """Show a selector component in place of the editor.

        ``create`` receives a ``done`` callback and returns a
        ``{"component", "focus"}`` record.
        """

        def done() -> None:
            self._editor_container.clear()
            self._editor_container.add_child(self.editor)
            self.ui.set_focus(self.editor)

        created = create(done)
        self._editor_container.clear()
        self._editor_container.add_child(created["component"])
        self.ui.set_focus(created["focus"])
        self.ui.request_render()

    def _show_settings_selector(self) -> None:
        def create(done):
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

            def on_thinking_level_change(level: str) -> None:
                self.session.set_thinking_level(level)
                self._footer.invalidate()
                self._update_editor_border_color()

            def on_theme_change(theme_setting: str) -> None:
                self.settings_manager.set_theme(theme_setting)
                tonio.spawn.without_tracking(self._theme_controller.apply_from_settings())

            def on_hide_thinking_block_change(hidden: bool) -> None:
                self._hide_thinking_block = hidden
                self.settings_manager.set_hide_thinking_block(hidden)
                for child in self._chat_container.children:
                    if isinstance(child, AssistantMessageComponent):
                        child.set_hide_thinking_block(hidden)
                self._chat_container.clear()
                self._rebuild_chat_from_messages()

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
                if self._streaming_component is not None or self.session.is_streaming:
                    for child in self._chat_container.children:
                        if isinstance(child, (AssistantMessageComponent, UserMessageComponent)):
                            child.set_output_pad(padding)
                    if self._streaming_component is not None:
                        self._streaming_component.set_output_pad(padding)
                    self.ui.request_render()
                    return
                self._rebuild_chat_from_messages()

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

            def on_cancel() -> None:
                done()
                self.ui.request_render()

            selector = SettingsSelectorComponent(
                {
                    "autoCompact": self.session.auto_compaction_enabled,
                    "showImages": self.settings_manager.get_show_images(),
                    "imageWidthCells": self.settings_manager.get_image_width_cells(),
                    "autoResizeImages": self.settings_manager.get_image_auto_resize(),
                    "blockImages": self.settings_manager.get_block_images(),
                    "enableSkillCommands": self.settings_manager.get_enable_skill_commands(),
                    "steeringMode": self.session.steering_mode,
                    "followUpMode": self.session.follow_up_mode,
                    "transport": self.settings_manager.get_transport(),
                    "httpIdleTimeoutMs": self.settings_manager.get_http_idle_timeout_ms(),
                    "thinkingLevel": self.session.thinking_level,
                    "availableThinkingLevels": self.session.get_available_thinking_levels(),
                    "currentTheme": self.settings_manager.get_theme_setting() or "dark",
                    "terminalTheme": self._theme_controller.get_terminal_theme(),
                    "availableThemes": get_available_themes(),
                    "hideThinkingBlock": self._hide_thinking_block,
                    "collapseChangelog": self.settings_manager.get_collapse_changelog(),
                    "enableInstallTelemetry": self.settings_manager.get_enable_install_telemetry(),
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
                    "onThinkingLevelChange": on_thinking_level_change,
                    "onThemeChange": on_theme_change,
                    "onThemePreview": lambda theme_name: self._theme_controller.preview(theme_name),
                    "onHideThinkingBlockChange": on_hide_thinking_block_change,
                    "onShowCacheMissNoticesChange": on_show_cache_miss_notices_change,
                    "onCollapseChangelogChange": lambda collapsed: self.settings_manager.set_collapse_changelog(
                        collapsed
                    ),
                    "onEnableInstallTelemetryChange": lambda enabled: self.settings_manager.set_enable_install_telemetry(
                        enabled
                    ),
                    "onQuietStartupChange": lambda enabled: self.settings_manager.set_quiet_startup(enabled),
                    "onDefaultProjectTrustChange": lambda default_project_trust: self.settings_manager.set_default_project_trust(
                        default_project_trust
                    ),
                    "onDoubleEscapeActionChange": lambda action: self.settings_manager.set_double_escape_action(
                        action
                    ),
                    "onTreeFilterModeChange": lambda mode: self.settings_manager.set_tree_filter_mode(mode),
                    "onShowHardwareCursorChange": on_show_hardware_cursor_change,
                    "onEditorPaddingXChange": on_editor_padding_x_change,
                    "onOutputPadChange": on_output_pad_change,
                    "onAutocompleteMaxVisibleChange": on_autocomplete_max_visible_change,
                    "onClearOnShrinkChange": on_clear_on_shrink_change,
                    "onShowTerminalProgressChange": lambda enabled: self.settings_manager.set_show_terminal_progress(
                        enabled
                    ),
                    "onWarningsChange": lambda warnings: self.settings_manager.set_warnings(warnings),
                    "onCancel": on_cancel,
                },
            )
            return {"component": selector, "focus": selector.get_settings_list()}

        self._show_selector(create)

    async def _handle_model_command(self, search_term: str | None = None) -> None:
        if not search_term:
            self._show_model_selector()
            return

        model = await self._find_exact_model_match(search_term)
        if model is not None:
            try:
                await self.session.set_model(model)
                self._footer.invalidate()
                self._update_editor_border_color()
                self.show_status(f"Model: {model.id}")
                tonio.spawn.without_tracking(self._maybe_warn_about_anthropic_subscription_auth(model))
                self._check_daxnuts_easter_egg(model)
            except Exception as error:
                self.show_error(str(error))
            return

        self._show_model_selector(search_term)

    async def _find_exact_model_match(self, search_term: str):
        models = await self._get_model_candidates()
        return find_exact_model_reference_match(search_term, models)

    async def _get_model_candidates(self) -> list:
        if self.session.scoped_models:
            return [scoped.model for scoped in self.session.scoped_models]

        try:
            await self.session.model_runtime.refresh()
            return list(await self.session.model_runtime.get_available())
        except Exception:
            return []

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
                self._anthropic_subscription_warning_shown = True
                self.show_warning(ANTHROPIC_SUBSCRIPTION_AUTH_WARNING)
                return
            auth_result = await self.session.model_runtime.get_auth(model.provider)
            api_key = auth_result.auth.api_key if auth_result is not None else None
            if not is_anthropic_subscription_auth_key(api_key):
                return
            self._anthropic_subscription_warning_shown = True
            self.show_warning(ANTHROPIC_SUBSCRIPTION_AUTH_WARNING)
        except Exception:
            # Ignore auth lookup failures for warning-only checks.
            return

    def _maybe_save_implicit_project_trust_after_reload(self) -> bool:
        cwd = self.session_manager.get_cwd()
        if self._auto_trust_on_reload_cwd != cwd:
            return False
        if not self.settings_manager.is_project_trusted() or not has_trust_requiring_project_resources(cwd):
            return False

        trust_store = ProjectTrustStore(self.runtime_host.services.agent_dir)
        try:
            if trust_store.get(cwd) is not None:
                self._auto_trust_on_reload_cwd = None
                return False
            trust_store.set(cwd, True)
            self._auto_trust_on_reload_cwd = None
            return True
        except Exception as error:
            self.show_warning(f"Could not save project trust after reload: {error}")
            return False

    def _show_trust_selector(self) -> None:
        cwd = self.session_manager.get_cwd()
        trust_store = ProjectTrustStore(self.runtime_host.services.agent_dir)
        saved_decision = trust_store.get_entry(cwd)

        def create(done):
            def on_select(selection: dict) -> None:
                trust_store.set_many(selection["updates"])
                done()
                self.show_status(
                    f"Saved trust decision: {'trusted' if selection['trusted'] else 'untrusted'}. "
                    f"Restart {APP_NAME} for this to take effect."
                )

            def on_cancel() -> None:
                done()
                self.ui.request_render()

            selector = TrustSelectorComponent(
                {
                    "cwd": cwd,
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
            async def select_model(model) -> None:
                try:
                    await self.session.set_model(model)
                    self._footer.invalidate()
                    self._update_editor_border_color()
                    done()
                    self.show_status(f"Model: {model.id}")
                    tonio.spawn.without_tracking(self._maybe_warn_about_anthropic_subscription_auth(model))
                    self._check_daxnuts_easter_egg(model)
                except Exception as error:
                    done()
                    self.show_error(str(error))

            def on_cancel() -> None:
                done()
                self.ui.request_render()

            selector = ModelSelectorComponent(
                self.ui,
                self.session.model,
                self.settings_manager,
                self.session.model_runtime,
                self.session.scoped_models,
                lambda model: tonio.spawn.without_tracking(select_model(model)),
                on_cancel,
                initial_search_input,
            )
            return {"component": selector, "focus": selector}

        self._show_selector(create)

    async def _show_models_selector(self) -> None:
        # Get all available models
        await self.session.model_runtime.refresh()
        all_models = list(await self.session.model_runtime.get_available())

        if not all_models:
            self.show_status("No models available")
            return

        # Check if session has scoped models (from previous session-only
        # changes or CLI --models)
        session_scoped_models = self.session.scoped_models
        has_session_scope = len(session_scoped_models) > 0

        # Build enabled model IDs from session state or settings
        current_enabled_ids: list | None = None

        if has_session_scope:
            # Use current session's scoped models
            current_enabled_ids = [f"{scoped.model.provider}/{scoped.model.id}" for scoped in session_scoped_models]
        else:
            # Fall back to settings
            patterns = self.settings_manager.get_enabled_models()
            if patterns:
                scoped_models = await resolve_model_scope(patterns, self.session.model_runtime)
                current_enabled_ids = [f"{scoped.model.provider}/{scoped.model.id}" for scoped in scoped_models]

        state = {"enabledIds": current_enabled_ids}

        # Helper to update session's scoped models (session-only, no persist)
        async def update_session_models(enabled_ids) -> None:
            state["enabledIds"] = None if enabled_ids is None else list(enabled_ids)
            if enabled_ids and len(enabled_ids) < len(all_models):
                new_scoped_models = await resolve_model_scope(enabled_ids, self.session.model_runtime)
                self.session.set_scoped_models(
                    [ScopedModel(model=sm.model, thinking_level=sm.thinking_level) for sm in new_scoped_models]
                )
            else:
                # All enabled or none enabled = no filter
                self.session.set_scoped_models([])
            self._update_available_provider_count()
            self.ui.request_render()

        def create(done):
            def on_persist(enabled_ids) -> None:
                # Persist to settings
                new_patterns = (
                    None  # All enabled = clear filter
                    if enabled_ids is None or len(enabled_ids) == len(all_models)
                    else enabled_ids
                )
                self.settings_manager.set_enabled_models(list(new_patterns) if new_patterns else None)
                self.show_status("Model selection saved to settings")

            def on_cancel() -> None:
                done()
                self.ui.request_render()

            selector = ScopedModelsSelectorComponent(
                {
                    "allModels": all_models,
                    "enabledModelIds": state["enabledIds"],
                },
                {
                    "onChange": lambda enabled_ids: tonio.spawn.without_tracking(update_session_models(enabled_ids)),
                    "onPersist": on_persist,
                    "onCancel": on_cancel,
                },
            )
            return {"component": selector, "focus": selector}

        self._show_selector(create)

    def _show_user_message_selector(self) -> None:
        user_messages = self.session.get_user_messages_for_forking()

        if not user_messages:
            self.show_status("No messages to fork from")
            return

        initial_selected_id = user_messages[-1]["entryId"]

        def create(done):
            async def select_entry(entry_id: str) -> None:
                done()
                try:
                    result = await self.runtime_host.fork(entry_id)
                    if result.get("cancelled"):
                        self.ui.request_render()
                        return

                    self.editor.set_text(result.get("selectedText") or "")
                    self.show_status("Forked to new session")
                except Exception as error:
                    self.show_error(str(error))

            def on_cancel() -> None:
                done()
                self.ui.request_render()

            selector = UserMessageSelectorComponent(
                [{"id": message["entryId"], "text": message["text"]} for message in user_messages],
                lambda entry_id: tonio.spawn.without_tracking(select_entry(entry_id)),
                on_cancel,
                initial_selected_id,
            )
            return {"component": selector, "focus": selector.get_message_list()}

        self._show_selector(create)

    async def handle_clone_command(self) -> None:
        leaf_id = self.session_manager.get_leaf_id()
        if not leaf_id:
            self.show_status("Nothing to clone yet")
            return

        try:
            result = await self.runtime_host.fork(leaf_id, position="at")
            if result.get("cancelled"):
                self.ui.request_render()
                return

            self.editor.set_text("")
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

        def create(done):
            async def select_entry(entry_id: str) -> None:
                # Selecting the current leaf is a no-op (already there)
                if entry_id == real_leaf_id:
                    done()
                    self.show_status("Already at this point")
                    return

                # Ask about summarization
                done()  # Close selector first

                # Loop until user makes a complete choice or cancels to tree
                wants_summary = False
                custom_instructions: str | None = None

                # Check if we should skip the prompt (user preference to
                # always default to no summary)
                if not self.settings_manager.get_branch_summary_skip_prompt():
                    while True:
                        summary_choice = await self._show_extension_selector(
                            "Summarize branch?",
                            ["No summary", "Summarize", "Summarize with custom prompt"],
                        )

                        if summary_choice is None:
                            # User pressed escape - re-show tree selector with
                            # same selection
                            self._show_tree_selector(entry_id)
                            return

                        wants_summary = summary_choice != "No summary"

                        if summary_choice == "Summarize with custom prompt":
                            custom_instructions = await self._show_extension_editor(
                                "Custom summarization instructions"
                            )
                            if custom_instructions is None:
                                # User cancelled - loop back to summary selector
                                continue

                        # User made a complete choice
                        break

                # Set up escape handler and status indicator if summarizing
                showing_summary_indicator = False
                original_on_escape = self._default_editor.on_escape

                if wants_summary:
                    self._default_editor.on_escape = lambda: self.session.abort_branch_summary()
                    self._chat_container.add_child(Spacer(1))
                    self._show_status_indicator(BranchSummaryStatusIndicator(self.ui))
                    showing_summary_indicator = True
                    self.ui.request_render()

                try:
                    result = await self.session.navigate_tree(
                        entry_id,
                        {"summarize": wants_summary, "custom_instructions": custom_instructions},
                    )

                    if result.aborted:
                        # Summarization aborted - re-show tree selector with
                        # same selection
                        self.show_status("Branch summarization cancelled")
                        self._show_tree_selector(entry_id)
                        return
                    if result.cancelled:
                        self.show_status("Navigation cancelled")
                        return

                    # Update UI
                    self._chat_container.clear()
                    self._render_initial_messages()
                    if result.editor_text and not self.editor.get_text().strip():
                        self.editor.set_text(result.editor_text)
                    self.show_status("Navigated to selected point")
                    tonio.spawn.without_tracking(self._flush_compaction_queue({"willRetry": False}))
                except Exception as error:
                    self.show_error(str(error))
                finally:
                    if showing_summary_indicator:
                        self._clear_status_indicator("branchSummary")
                    self._default_editor.on_escape = original_on_escape

            async def copy_entry(text) -> None:
                if not text:
                    self.show_error("Selected entry has no text to copy")
                    return
                try:
                    await copy_to_clipboard(text)
                    self.show_status("Copied selected message to clipboard")
                except Exception as error:
                    self.show_error(str(error))

            def on_cancel() -> None:
                done()
                self.ui.request_render()

            def on_label_edit(entry_id: str, label) -> None:
                self.session_manager.append_label_change(entry_id, label)
                self.ui.request_render()

            selector = TreeSelectorComponent(
                tree,
                real_leaf_id,
                self.ui.terminal.rows,
                initial_selected_id,
                initial_filter_mode,
            )
            selector.on_select = lambda entry_id: tonio.spawn.without_tracking(select_entry(entry_id))
            selector.on_cancel = on_cancel
            selector.on_label_edit = on_label_edit
            selector.on_copy = lambda text: tonio.spawn.without_tracking(copy_entry(text))
            return {"component": selector, "focus": selector}

        self._show_selector(create)

    def _show_session_selector(self) -> None:
        def create(done):
            async def select_session(session_path: str) -> None:
                done()
                await self._handle_resume_session(session_path)

            def on_cancel() -> None:
                done()
                self.ui.request_render()

            async def rename_session(session_file_path: str, next_name) -> None:
                next_value = (next_name or "").strip()
                if not next_value:
                    return
                mgr = SessionManager.open(session_file_path)
                mgr.append_session_info(next_value)

            selector = SessionSelectorComponent(
                lambda on_progress: SessionManager.list(
                    self.session_manager.get_cwd(), self.session_manager.get_session_dir(), on_progress
                ),
                lambda on_progress: (
                    SessionManager.list_all(on_progress=on_progress)
                    if self.session_manager.uses_default_session_dir()
                    else SessionManager.list_all(self.session_manager.get_session_dir(), on_progress)
                ),
                lambda session_path: tonio.spawn.without_tracking(select_session(session_path)),
                on_cancel,
                lambda: tonio.spawn.without_tracking(self.shutdown()),
                lambda: self.ui.request_render(),
                {
                    "renameSession": rename_session,
                    "showRenameHint": True,
                    "keybindings": self._keybindings,
                },
                self.session_manager.get_session_file(),
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
            if (not auth_type or auth_type == "oauth") and provider.auth.oauth:
                options.append(
                    {
                        "id": provider.id,
                        "name": provider.name,
                        "authType": "oauth",
                        "method": provider.auth.oauth,
                        "status": status,
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
                    }
                )
        return sorted(options, key=lambda option: (option["name"].lower(), option["name"]))

    async def _get_logout_provider_options(self) -> list:
        options = []
        for credential in await self.session.model_runtime.list_credentials():
            provider = self.session.model_runtime.get_provider(credential.provider_id)
            options.append(
                {
                    "id": credential.provider_id,
                    "name": provider.name if provider is not None else credential.provider_id,
                    "authType": credential.type,
                    "status": {"type": credential.type, "source": "stored credential"},
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
            if provider["id"].lower() == normalized_provider_ref
            or provider["name"].lower() == normalized_provider_ref
        ]

    async def _handle_login_command(self, provider_ref: str | None = None) -> None:
        await self.session.model_runtime.get_available()
        if not provider_ref:
            self._show_login_auth_type_selector()
            return

        provider_options = self._find_login_provider_options(provider_ref)
        if len(provider_options) == 1:
            await self._start_provider_login(provider_options[0])
            return

        if len(provider_options) > 1:
            provider_ids = {provider["id"] for provider in provider_options}
            if len(provider_ids) == 1:
                self._show_login_auth_type_selector(provider_options)
                return

        self._show_login_provider_selector(None, provider_ref)

    async def _start_provider_login(self, provider_option: dict) -> None:
        if provider_option["authType"] == "oauth":
            await self._show_login_dialog(provider_option["id"], provider_option["name"])
        elif getattr(provider_option.get("method"), "login", None):
            await self._show_api_key_login_dialog(provider_option["id"], provider_option["name"])
        else:
            self._show_ambient_auth_dialog(provider_option)

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
                tonio.spawn.without_tracking(self._start_provider_login(provider_option))
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
                        tonio.spawn.without_tracking(self._start_provider_login(provider_option))
                    return
                self._show_login_provider_selector(auth_type)

            def on_cancel() -> None:
                done()
                self.ui.request_render()

            selector = ExtensionSelectorComponent(title, options, on_select, on_cancel)
            return {"component": selector, "focus": selector}

        self._show_selector(create)

    def _show_login_provider_selector(
        self, auth_type: str | None = None, initial_search_input: str | None = None
    ) -> None:
        provider_options = self.get_login_provider_options(auth_type)
        if not provider_options:
            if auth_type == "oauth":
                message = "No subscription providers available."
            elif auth_type == "api_key":
                message = "No API key providers available."
            else:
                message = "No login providers available."
            self.show_status(message)
            return

        def create(done):
            async def select_provider(provider_id: str, selected_auth_type: str) -> None:
                done()

                provider_option = next(
                    (
                        provider
                        for provider in provider_options
                        if provider["id"] == provider_id and provider["authType"] == selected_auth_type
                    ),
                    None,
                )
                if provider_option is None:
                    return

                await self._start_provider_login(provider_option)

            def on_cancel() -> None:
                done()
                if auth_type:
                    self._show_login_auth_type_selector()
                else:
                    self.ui.request_render()

            selector = OAuthSelectorComponent(
                "login",
                provider_options,
                lambda provider_id, selected_auth_type: tonio.spawn.without_tracking(
                    select_provider(provider_id, selected_auth_type)
                ),
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

        tonio.spawn.without_tracking(self._show_logout_selector(mode))

    async def _show_logout_selector(self, mode: str) -> None:
        provider_options = await self._get_logout_provider_options()
        if not provider_options:
            self.show_status(
                "No stored credentials to remove. /logout only removes credentials saved by /login; "
                "environment variables and models.json config are unchanged."
            )
            return

        def create(done):
            async def select_provider(provider_id: str) -> None:
                done()

                provider_option = next(
                    (provider for provider in provider_options if provider["id"] == provider_id), None
                )
                if provider_option is None:
                    return

                try:
                    await self.session.model_runtime.logout(provider_option["id"])
                    self._update_available_provider_count()
                    message = (
                        f"Logged out of {provider_option['name']}"
                        if provider_option["authType"] == "oauth"
                        else f"Removed stored API key for {provider_option['name']}. "
                        "Environment variables and models.json config are unchanged."
                    )
                    self.show_status(message)
                except Exception as error:
                    self.show_error(f"Logout failed: {error}")

            def on_cancel() -> None:
                done()
                self.ui.request_render()

            selector = OAuthSelectorComponent(
                mode,
                provider_options,
                lambda provider_id, _auth_type: tonio.spawn.without_tracking(select_provider(provider_id)),
                on_cancel,
            )
            return {"component": selector, "focus": selector}

        self._show_selector(create)

    async def _complete_provider_authentication(
        self, provider_id: str, provider_name: str, auth_type: str, previous_model
    ) -> None:
        await self.session.model_runtime.get_available()

        action_label = (
            f"Logged in to {provider_name}" if auth_type == "oauth" else f"Saved API key for {provider_name}"
        )

        selected_model = None
        selection_error: str | None = None
        if _is_unknown_model(previous_model):
            available_models = await self.session.model_runtime.get_available()
            provider_models = [model for model in available_models if model.provider == provider_id]
            if provider_id not in DEFAULT_MODEL_PER_PROVIDER:
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
                selected_model = next((model for model in provider_models if model.id == default_model_id), None)
                if selected_model is None:
                    selection_error = (
                        f'{action_label}, but its default model "{default_model_id}" is not available. '
                        "Use /model to select a model."
                    )
                else:
                    try:
                        await self.session.set_model(selected_model)
                    except Exception as error:
                        selected_model = None
                        selection_error = (
                            f"{action_label}, but selecting its default model failed: {error}. "
                            "Use /model to select a model."
                        )

        self._update_available_provider_count()
        self._footer.invalidate()
        self._update_editor_border_color()
        if selected_model is not None:
            self.show_status(f"{action_label}. Selected {selected_model.id}. Credentials saved to {get_auth_path()}")
            tonio.spawn.without_tracking(self._maybe_warn_about_anthropic_subscription_auth(selected_model))
            self._check_daxnuts_easter_egg(selected_model)
        else:
            self.show_status(f"{action_label}. Credentials saved to {get_auth_path()}")
            if selection_error:
                self.show_error(selection_error)
            else:
                tonio.spawn.without_tracking(self._maybe_warn_about_anthropic_subscription_auth())

    def _show_ambient_auth_dialog(self, provider_option: dict) -> None:
        def restore_editor() -> None:
            self._editor_container.clear()
            self._editor_container.add_child(self.editor)
            self.ui.set_focus(self.editor)
            self.ui.request_render()

        dialog = LoginDialogComponent(
            self.ui,
            provider_option["id"],
            lambda *_args: restore_editor(),
            provider_option["name"],
            f"{provider_option['name']} setup",
        )
        method_name = getattr(provider_option.get("method"), "name", None)
        dialog.show_info(
            f"{method_name if method_name is not None else 'Authentication'} is configured outside {APP_NAME}.",
            [],
            True,
        )

        self._editor_container.clear()
        self._editor_container.add_child(dialog)
        self.ui.set_focus(dialog)
        self.ui.request_render()

    async def _show_api_key_login_dialog(self, provider_id: str, provider_name: str) -> None:
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

        self._editor_container.clear()
        self._editor_container.add_child(dialog)
        self.ui.set_focus(dialog)
        self.ui.request_render()

        def restore_editor() -> None:
            self._editor_container.clear()
            self._editor_container.add_child(self.editor)
            self.ui.set_focus(self.editor)
            self.ui.request_render()

        try:
            await self._login_provider(dialog, provider_id, "api_key")
            restore_editor()
            await self._complete_provider_authentication(provider_id, provider_name, "api_key", previous_model)
        except Exception as error:
            restore_editor()
            error_msg = str(error)
            if error_msg != "Login cancelled":
                self.show_error(f"Failed to save API key for {provider_name}: {error_msg}")

    async def _show_auth_select(self, dialog, prompt) -> str:
        done = tonio.Event()
        outcome: dict = {}

        def restore_dialog() -> None:
            self._editor_container.clear()
            self._editor_container.add_child(dialog)
            self.ui.set_focus(dialog)
            self.ui.request_render()

        labels = [option.label for option in prompt.options]

        def on_select(option_label: str) -> None:
            restore_dialog()
            option_id = next((option.id for option in prompt.options if option.label == option_label), None)
            if option_id:
                outcome["value"] = option_id
            done.set()

        def on_cancel() -> None:
            restore_dialog()
            done.set()

        selector = ExtensionSelectorComponent(prompt.message, labels, on_select, on_cancel)
        self._editor_container.clear()
        self._editor_container.add_child(selector)
        self.ui.set_focus(selector)
        self.ui.request_render()

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
        outcome: dict = {}

        async def run_response() -> None:
            try:
                outcome["value"] = await response
            except Exception as error:
                outcome["error"] = error
            finally:
                settled.set()

        remove_abort = cancel.on_cancel(lambda: settled.set())
        tonio.spawn.without_tracking(run_response())
        try:
            await settled.wait(None)
        finally:
            remove_abort()
        if "error" in outcome:
            raise outcome["error"]
        if "value" in outcome:
            return outcome["value"]
        raise Exception("Login cancelled")

    def _notify_auth_dialog(self, dialog, event) -> None:
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

        await self.session.model_runtime.login(provider_id, method, DialogInteraction())

    async def _show_login_dialog(self, provider_id: str, provider_name: str) -> None:
        previous_model = self.session.model
        dialog = LoginDialogComponent(self.ui, provider_id, lambda *_args: None, provider_name)
        self._editor_container.clear()
        self._editor_container.add_child(dialog)
        self.ui.set_focus(dialog)
        self.ui.request_render()

        def restore_editor() -> None:
            self._editor_container.clear()
            self._editor_container.add_child(self.editor)
            self.ui.set_focus(self.editor)
            self.ui.request_render()

        try:
            await self._login_provider(dialog, provider_id, "oauth")
            restore_editor()
            await self._complete_provider_authentication(provider_id, provider_name, "oauth", previous_model)
        except Exception as error:
            restore_editor()
            error_msg = str(error)
            if error_msg != "Login cancelled":
                self.show_error(f"Failed to login to {provider_name}: {error_msg}")

    # =========================================================================
    # Command handlers
    # =========================================================================

    async def _handle_reload_command(self) -> None:
        if self.session.is_streaming:
            self.show_warning("Wait for the current response to finish before reloading.")
            return
        if self.session.is_compacting:
            self.show_warning("Wait for compaction to finish before reloading.")
            return

        self._reset_extension_ui()

        def border_color(s: str) -> str:
            return theme.fg("border", s)

        reload_box = Container()
        reload_box.add_child(DynamicBorder(border_color))
        reload_box.add_child(Spacer(1))
        reload_box.add_child(
            Text(
                theme.fg("muted", "Reloading keybindings, extensions, skills, prompts, themes, and context files..."),
                1,
                0,
            )
        )
        reload_box.add_child(Spacer(1))
        reload_box.add_child(DynamicBorder(border_color))

        previous_editor = self.editor
        self._editor_container.clear()
        self._editor_container.add_child(reload_box)
        self.ui.set_focus(reload_box)
        self.ui.request_render(True)
        await tonio.time.sleep(0)

        def dismiss_reload_box(editor) -> None:
            self._editor_container.clear()
            self._editor_container.add_child(editor)
            self.ui.set_focus(editor)
            self.ui.request_render()

        chat_restored_before_session_start = False
        reload_box_dismissed = False

        def restore_chat_before_session_start() -> None:
            nonlocal chat_restored_before_session_start
            if chat_restored_before_session_start:
                return
            self._hide_thinking_block = self.settings_manager.get_hide_thinking_block()
            self._output_pad = self.settings_manager.get_output_pad()
            self._rebuild_chat_from_messages()
            chat_restored_before_session_start = True

        try:
            await self.session.reload(restore_chat_before_session_start)
            restore_chat_before_session_start()
            # pi reconfigures the undici HTTP dispatcher here; punkreq owns
            # HTTP transport in pidrei (see core/http_config.py).
            self._keybindings.reload()
            active_header = self._custom_header if self._custom_header is not None else self._built_in_header
            if is_expandable(active_header):
                active_header.set_expanded(self._tool_output_expanded)
            set_registered_themes(self.session.resource_loader.get_themes()["themes"])
            await self._theme_controller.apply_from_settings()
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
            self.ui.set_show_hardware_cursor(self.settings_manager.get_show_hardware_cursor())
            clear_on_shrink = self.settings_manager.get_clear_on_shrink()
            self.ui.set_clear_on_shrink(clear_on_shrink)
            if not clear_on_shrink and self._active_status_indicator is None:
                self._status_container.clear()
            self._setup_autocomplete_provider()
            runner = self.session.extension_runner
            self._setup_extension_shortcuts(runner)
            self._show_loaded_resources({"force": False, "showDiagnosticsWhenQuiet": True})
            saved_implicit_project_trust = self._maybe_save_implicit_project_trust_after_reload()
            models_json_error = self.session.model_runtime.get_error()
            if models_json_error:
                self.show_error(f"models.json error: {models_json_error}")
            self.show_status(
                "Reloaded keybindings, extensions, skills, prompts, themes, and context files; saved project trust"
                if saved_implicit_project_trust
                else "Reloaded keybindings, extensions, skills, prompts, themes, and context files"
            )
            dismiss_reload_box(self.editor)
            reload_box_dismissed = True
        except Exception as error:
            if not reload_box_dismissed:
                dismiss_reload_box(previous_editor)
            self.show_error(f"Reload failed: {error}")

    async def _handle_export_command(self, text: str) -> None:
        output_path = self._get_path_command_argument(text, "/export")

        try:
            if output_path is not None and output_path.endswith(".jsonl"):
                file_path = self.session.export_to_jsonl(output_path)
                self.show_status(f"Session exported to: {file_path}")
            else:
                file_path = await self.session.export_to_html(output_path)
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

    async def handle_import_command(self, text: str) -> None:
        input_path = self._get_path_command_argument(text, "/import")
        if not input_path:
            self.show_error("Usage: /import <path.jsonl>")
            return

        confirmed = await self._show_extension_confirm(
            "Import session", f"Replace current session with {input_path}?"
        )
        if not confirmed:
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
        import subprocess

        # Check if gh is available and logged in
        def run_gh_auth_status():
            return subprocess.run(
                ["gh", "auth", "status"],  # noqa: S607 - PATH lookup like pi's spawnSync
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
            )

        try:
            auth_result = await tonio.spawn_blocking(run_gh_auth_status)
        except OSError:
            self.show_error("GitHub CLI (gh) is not installed. Install it from https://cli.github.com/")
            return
        if auth_result.returncode != 0:
            self.show_error("GitHub CLI is not logged in. Run 'gh auth login' first.")
            return

        # Export to a temp file
        tmp_file = os.path.join(tempfile.gettempdir(), "session.html")
        try:
            await self.session.export_to_html(tmp_file)
        except Exception as error:
            self.show_error(f"Failed to export session: {error if str(error) else 'Unknown error'}")
            return

        # Show cancellable loader, replacing the editor
        loader = BorderedLoader(self.ui, theme, "Creating gist...")
        self._editor_container.clear()
        self._editor_container.add_child(loader)
        self.ui.set_focus(loader)
        self.ui.request_render()

        def restore_editor() -> None:
            loader.dispose()
            self._editor_container.clear()
            self._editor_container.add_child(self.editor)
            self.ui.set_focus(self.editor)
            with contextlib.suppress(OSError):
                os.unlink(tmp_file)

        # Create a secret gist asynchronously. exec_command kills the process
        # when the cancel token fires (pi kills the spawned gh directly).
        gist_cancel = AiCancelToken()

        def on_abort() -> None:
            gist_cancel.cancel()
            restore_editor()
            self.show_status("Share cancelled")

        loader.on_abort = on_abort

        try:
            result = await exec_command(
                "gh",
                ["gist", "create", "--public=false", tmp_file],
                self.session_manager.get_cwd(),
                cancel=gist_cancel,
            )

            if loader.signal.cancelled:
                return

            restore_editor()

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

            # Create the preview URL
            preview_url = get_share_viewer_url(gist_id)
            self.show_status(f"Share URL: {preview_url}\nGist: {gist_url}")
        except Exception as error:
            if not loader.signal.cancelled:
                restore_editor()
                self.show_error(f"Failed to create gist: {error if str(error) else 'Unknown error'}")

    async def _handle_copy_command(self) -> None:
        text = self.session.get_last_assistant_text()
        if not text:
            self.show_error("No agent messages to copy yet.")
            return

        try:
            await copy_to_clipboard(text)
            self.show_status("Copied last agent message to clipboard")
        except Exception as error:
            self.show_error(str(error))

    def _handle_name_command(self, text: str) -> None:
        name = re.sub(r"^/name\s*", "", text).strip()
        if not name:
            current_name = self.session_manager.get_session_name()
            if current_name:
                self._chat_container.add_child(Spacer(1))
                self._chat_container.add_child(Text(theme.fg("dim", f"Session name: {current_name}"), 1, 0))
            else:
                self.show_warning("Usage: /name <name>")
            self.ui.request_render()
            return

        self.session.set_session_name(name)
        session_name = self.session_manager.get_session_name()
        if session_name != name:
            self.show_warning(
                f"Session name was normalized from {json.dumps(name)} to {json.dumps(session_name)}"
            )
        self._chat_container.add_child(Spacer(1))
        self._chat_container.add_child(
            Text(theme.fg("dim", f"Session name set: {session_name if session_name is not None else name}"), 1, 0)
        )
        self.ui.request_render()

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

        info = f"{theme.bold('Session Info')}\n\n"
        if session_name:
            info += f"{theme.fg('dim', 'Name:')} {session_name}\n"
        info += f"{theme.fg('dim', 'File:')} {stats.session_file if stats.session_file is not None else 'In-memory'}\n"
        info += f"{theme.fg('dim', 'ID:')} {stats.session_id}\n\n"
        info += f"{theme.bold('Messages')}\n"
        info += f"{theme.fg('dim', 'Total:')} {stats.total_messages}\n"
        info += f"{theme.fg('dim', 'User:')} {stats.user_messages}\n"
        info += f"{theme.fg('dim', 'Assistant:')} {stats.assistant_messages}\n"
        info += f"{theme.fg('dim', 'Tools:')} {stats.tool_calls} calls, {stats.tool_results} results\n\n"
        info += f"{theme.bold('Tokens')}\n"
        # "Input" is the full prompt volume. With cache activity, split it into
        # cached (served from cache) vs uncached (everything else) - the only
        # provider-independent split. Cache writes, where reported, are a
        # detail of the uncached portion.
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

        if stats.cost > 0 or cache_waste.missed_tokens > 0:
            info += f"\n{theme.bold('Cost')}\n"
            info += f"{theme.fg('dim', 'Total:')} ${stats.cost:.3f}"
            if len(usage_breakdown) > 1:
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

        self._chat_container.add_child(Spacer(1))
        self._chat_container.add_child(Text(info, 1, 0))
        self.ui.request_render()

    def _handle_changelog_command(self) -> None:
        changelog_path = get_changelog_path()
        all_entries = parse_changelog(changelog_path)

        changelog_markdown = (
            "\n\n".join(normalize_changelog_links(entry["content"], entry) for entry in reversed(all_entries))
            if all_entries
            else "No changelog entries found."
        )

        self._chat_container.add_child(Spacer(1))
        self._chat_container.add_child(DynamicBorder())
        self._chat_container.add_child(Text(theme.bold(theme.fg("accent", "What's New")), 1, 0))
        self._chat_container.add_child(Spacer(1))
        self._chat_container.add_child(Markdown(changelog_markdown, 1, 1, self._get_markdown_theme_with_settings()))
        self._chat_container.add_child(DynamicBorder())
        self.ui.request_render()

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
| `{copy_message}` | Copy last assistant message |
| `{follow_up}` | Queue follow-up message |
| `{dequeue}` | Restore queued messages |
| `{paste_image}` | Paste image or text from clipboard |
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
                description = (
                    shortcut.description if shortcut.description is not None else shortcut.extension_path
                )
                key_display = format_key_text(key, {"capitalize": True})
                hotkeys += f"| `{key_display}` | {description} |\n"

        self._chat_container.add_child(Spacer(1))
        self._chat_container.add_child(DynamicBorder())
        self._chat_container.add_child(Text(theme.bold(theme.fg("accent", "Keyboard Shortcuts")), 1, 0))
        self._chat_container.add_child(Spacer(1))
        self._chat_container.add_child(Markdown(hotkeys.strip(), 1, 1, self._get_markdown_theme_with_settings()))
        self._chat_container.add_child(DynamicBorder())
        self.ui.request_render()

    async def _handle_clear_command(self) -> None:
        self._clear_status_indicator()
        try:
            result = await self.runtime_host.new_session()
            if result.get("cancelled"):
                return
            self._chat_container.add_child(Spacer(1))
            self._chat_container.add_child(Text(theme.fg("accent", "✓ New session started"), 1, 1))
            self.ui.request_render()
        except Exception as error:
            await self._handle_fatal_runtime_error("Failed to create session", error)

    def _handle_debug_command(self) -> None:
        width = self.ui.terminal.columns
        height = self.ui.terminal.rows
        all_lines = self.ui.render(width)

        debug_log_path = get_debug_log_path()
        debug_lines = [
            f"Debug output at {datetime.now(UTC).isoformat(timespec='milliseconds').replace('+00:00', 'Z')}",
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

        os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)
        with open(debug_log_path, "w", encoding="utf-8") as handle:
            handle.write(debug_data)

        self._chat_container.add_child(Spacer(1))
        self._chat_container.add_child(
            Text(f"{theme.fg('accent', '✓ Debug log written')}\n{theme.fg('muted', debug_log_path)}", 1, 1)
        )
        self.ui.request_render()

    def _handle_armin_says_hi(self) -> None:
        self._chat_container.add_child(Spacer(1))
        self._chat_container.add_child(ArminComponent(self.ui))
        self.ui.request_render()

    def _handle_demented_elves(self) -> None:
        self._chat_container.add_child(Spacer(1))
        self._chat_container.add_child(EarendilAnnouncementComponent())
        self.ui.request_render()

    def _handle_daxnuts(self) -> None:
        self._chat_container.add_child(Spacer(1))
        self._chat_container.add_child(DaxnutsComponent(self.ui))
        self.ui.request_render()

    def _check_daxnuts_easter_egg(self, model) -> None:
        if model.provider == "opencode" and "kimi-k2.5" in model.id.lower():
            self._handle_daxnuts()

    async def _handle_bash_command(self, command: str, exclude_from_context: bool = False) -> None:
        extension_runner = self.session.extension_runner

        # Emit user_bash event to let extensions intercept
        event_result = await extension_runner.emit_user_bash(
            {
                "type": "user_bash",
                "command": command,
                "excludeFromContext": exclude_from_context,
                "cwd": self.session_manager.get_cwd(),
            }
        )

        # If extension returned a full result, use it directly
        if event_result and event_result.get("result"):
            result = event_result["result"]

            # Create UI component for display
            self._bash_component = BashExecutionComponent(command, self.ui, exclude_from_context)
            if self.session.is_streaming:
                self._pending_messages_container.add_child(self._bash_component)
                self._pending_bash_components.append(self._bash_component)
            else:
                self._chat_container.add_child(self._bash_component)

            # Show output and complete
            if result.get("output"):
                self._bash_component.append_output(result["output"])
            self._bash_component.set_complete(
                result.get("exitCode"),
                bool(result.get("cancelled")),
                _partial_truncation_result(result.get("output") or "") if result.get("truncated") else None,
                result.get("fullOutputPath"),
            )

            # Record the result in session
            self.session.record_bash_result(
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
            self._bash_component = None
            self.ui.request_render()
            return

        # Normal execution path (possibly with custom operations)
        is_deferred = self.session.is_streaming
        self._bash_component = BashExecutionComponent(command, self.ui, exclude_from_context)

        if is_deferred:
            # Show in pending area when agent is streaming
            self._pending_messages_container.add_child(self._bash_component)
            self._pending_bash_components.append(self._bash_component)
        else:
            # Show in chat immediately when agent is idle
            self._chat_container.add_child(self._bash_component)
        self.ui.request_render()

        def on_chunk(chunk: str) -> None:
            if self._bash_component is not None:
                self._bash_component.append_output(chunk)
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

            if self._bash_component is not None:
                self._bash_component.set_complete(
                    result.exit_code,
                    result.cancelled,
                    _partial_truncation_result(result.output) if result.truncated else None,
                    result.full_output_path,
                )
        except Exception as error:
            if self._bash_component is not None:
                self._bash_component.set_complete(None, False)
            self.show_error(f"Bash command failed: {error if str(error) else 'Unknown error'}")

        self._bash_component = None
        self.ui.request_render()

    async def handle_compact_command(self, custom_instructions: str | None = None) -> None:
        self._clear_status_indicator()

        with contextlib.suppress(Exception):
            # Ignore, will be emitted as an event
            await self.session.compact(custom_instructions)

    async def stop(self) -> None:
        if self.settings_manager.get_show_terminal_progress():
            self.ui.terminal.set_progress(False)
        self._clear_status_indicator()
        self._theme_controller.disable_auto_sync()
        self._clear_extension_terminal_input_listeners()
        self._footer.dispose()
        self._footer_data_provider.dispose()
        if self._unsubscribe:
            self._unsubscribe()
        if self._is_initialized:
            await self.ui.stop()
            self._is_initialized = False
        self._unregister_signal_handlers()
