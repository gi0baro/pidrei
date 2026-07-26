"""Mirror of pi coding-agent test/interactive-mode-status.test.ts.

pi drives InteractiveMode.prototype methods with fake `this` objects; here
the unbound methods run against bare InteractiveMode instances created with
``__new__`` (real helper methods stay available) or SimpleNamespace fakes.
The Windows npm-sibling labeling test is not ported (POSIX-only port).
"""

import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.core.source_info import SourceInfo
from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei.modes.interactive.theme import init_theme
from pidrei_tui import TUI, CombinedAutocompleteProvider, Container


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tui" / "tests"))
from virtual_terminal import VirtualTerminal


@pytest.fixture(autouse=True)
def _theme():
    # showStatus and showLoadedResources use the global theme instance
    init_theme("dark")


class _BareInteractiveMode(InteractiveMode):
    """InteractiveMode with the runtime-delegating properties replaced by
    plain attributes, so fake-this tests can assign them directly (pi's fake
    `this` objects override the same accessors implicitly)."""

    session = None
    session_manager = None
    settings_manager = None

    def __init__(self) -> None:
        pass


def render_last_line(container: Container, width: int = 120) -> str:
    if not container.children:
        return ""
    return "\n".join(container.children[-1].render(width))


def render_all(container: Container, width: int = 120) -> str:
    return "\n".join(line for child in container.children for line in child.render(width))


def normalize_rendered_output(container: Container, width: int = 220) -> str:
    output = re.sub(r"\x1b\[[0-9;]*m", "", render_all(container, width))
    return "\n".join(line.rstrip() for line in output.split("\n")).strip()


class FakeFocusableComponent:
    def __init__(self, label: str) -> None:
        self.focused = False
        self.inputs: list = []
        self._label = label
        self._text = ""

    def handle_input(self, data: str) -> None:
        self.inputs.append(data)

    def get_text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text

    def render(self, _width=None) -> list:
        return [self._label]

    def invalidate(self) -> None:
        pass


async def flush_tui(ui: TUI, terminal: VirtualTerminal) -> None:
    ui.request_render(True)
    await terminal.wait_for_render()


class _OtherComponent:
    def render(self, _width=None) -> list:
        return ["OTHER"]

    def invalidate(self) -> None:
        pass


class TestShowStatus:
    def _create_fake(self):
        fake = SimpleNamespace(
            _chat_container=Container(),
            request_render_calls=[],
            _last_status_spacer=None,
            _last_status_text=None,
        )
        fake.ui = SimpleNamespace(request_render=lambda force=False: fake.request_render_calls.append(force))
        return fake

    def test_coalesces_immediately_sequential_status_messages(self):
        fake = self._create_fake()

        InteractiveMode.show_status(fake, "STATUS_ONE")
        assert len(fake._chat_container.children) == 2
        assert "STATUS_ONE" in render_last_line(fake._chat_container)

        InteractiveMode.show_status(fake, "STATUS_TWO")
        # second status updates the previous line instead of appending
        assert len(fake._chat_container.children) == 2
        assert "STATUS_TWO" in render_last_line(fake._chat_container)
        assert "STATUS_ONE" not in render_last_line(fake._chat_container)

    def test_appends_a_new_status_line_if_something_else_was_added_in_between(self):
        fake = self._create_fake()

        InteractiveMode.show_status(fake, "STATUS_ONE")
        assert len(fake._chat_container.children) == 2

        # Something else gets added to the chat in between status updates
        fake._chat_container.add_child(_OtherComponent())
        assert len(fake._chat_container.children) == 3

        InteractiveMode.show_status(fake, "STATUS_TWO")
        # adds spacer + text
        assert len(fake._chat_container.children) == 5
        assert "STATUS_TWO" in render_last_line(fake._chat_container)


class TestSetToolsExpanded:
    def test_applies_expansion_state_to_the_active_header_and_chat_entries(self):
        header_calls: list = []
        loaded_calls: list = []
        chat_calls: list = []
        request_render_calls: list = []
        fake = SimpleNamespace(
            _tool_output_expanded=False,
            _custom_header=None,
            _built_in_header=SimpleNamespace(set_expanded=header_calls.append),
            _loaded_resources_container=SimpleNamespace(
                children=[SimpleNamespace(set_expanded=loaded_calls.append)]
            ),
            _chat_container=SimpleNamespace(children=[SimpleNamespace(set_expanded=chat_calls.append)]),
            ui=SimpleNamespace(request_render=lambda force=False: request_render_calls.append(force)),
        )

        InteractiveMode.set_tools_expanded(fake, True)

        assert fake._tool_output_expanded is True
        assert header_calls == [True]
        assert loaded_calls == [True]
        assert chat_calls == [True]
        assert len(request_render_calls) == 1


class TestCreateExtensionUIContextSetTheme:
    def test_persists_theme_changes_to_settings_manager(self):
        state = {"theme": "dark"}
        set_theme_calls: list = []
        settings_manager = SimpleNamespace(
            get_theme=lambda: state["theme"],
            set_theme=lambda name: (set_theme_calls.append(name), state.update(theme=name)),
        )
        set_theme_name_calls: list = []
        request_render_calls: list = []
        fake = SimpleNamespace(
            session=SimpleNamespace(settings_manager=settings_manager),
            settings_manager=settings_manager,
        )
        fake.ui = SimpleNamespace(request_render=lambda force=False: request_render_calls.append(force))

        def set_theme_name(name):
            set_theme_name_calls.append(name)
            fake.ui.request_render()
            return {"success": True}

        fake._theme_controller = SimpleNamespace(
            set_theme_instance=lambda instance: {"success": True},
            set_theme_name=set_theme_name,
        )

        ui_context = InteractiveMode._create_extension_ui_context(fake)
        result = ui_context["setTheme"]("light")

        assert result["success"] is True
        assert set_theme_name_calls == ["light"]
        assert set_theme_calls == ["light"]
        assert state["theme"] == "light"
        assert len(request_render_calls) == 1

    def test_does_not_persist_invalid_theme_names(self):
        set_theme_calls: list = []
        settings_manager = SimpleNamespace(
            get_theme=lambda: "dark",
            set_theme=set_theme_calls.append,
        )
        set_theme_name_calls: list = []
        request_render_calls: list = []
        fake = SimpleNamespace(
            session=SimpleNamespace(settings_manager=settings_manager),
            settings_manager=settings_manager,
            ui=SimpleNamespace(request_render=lambda force=False: request_render_calls.append(force)),
        )

        def set_theme_name(name):
            set_theme_name_calls.append(name)
            return {"success": False, "error": "Theme not found"}

        fake._theme_controller = SimpleNamespace(
            set_theme_instance=lambda instance: {"success": True},
            set_theme_name=set_theme_name,
        )

        ui_context = InteractiveMode._create_extension_ui_context(fake)
        result = ui_context["setTheme"]("__missing_theme__")

        assert result["success"] is False
        assert set_theme_name_calls == ["__missing_theme__"]
        assert set_theme_calls == []
        assert request_render_calls == []


@pytest.mark.tonio
async def test_overlay_custom_ui_reclaims_input_after_non_overlay_custom_ui_closes():
    init_theme("dark")
    terminal = VirtualTerminal(80, 24)
    ui = TUI(terminal)
    editor_container = Container()
    editor = FakeFocusableComponent("EDITOR")
    palette = FakeFocusableComponent("PALETTE")
    overlay = FakeFocusableComponent("OVERLAY")
    replacement = FakeFocusableComponent("REPLACEMENT")
    closers: dict = {}
    results: dict = {}
    fake = SimpleNamespace(
        editor=editor,
        _editor_container=editor_container,
        _keybindings={},
        ui=ui,
    )

    def show_extension_custom(key, factory, options=None):
        done = tonio.Event()

        async def run() -> None:
            results[key] = await InteractiveMode._show_extension_custom(fake, factory, options)
            done.set()

        tonio.spawn.without_tracking(run())
        return done

    editor_container.add_child(editor)
    ui.add_child(editor_container)
    ui.add_child(palette)
    ui.set_focus(palette)
    await ui.start()
    try:
        def overlay_factory(_tui, _theme, _keybindings, done):
            closers["overlay"] = done
            return overlay

        overlay_done = show_extension_custom("overlay", overlay_factory, {"overlay": True})
        await flush_tui(ui, terminal)
        assert overlay.focused is True

        def replacement_factory(_tui, _theme, _keybindings, done):
            closers["replacement"] = done
            return replacement

        replacement_done = show_extension_custom("replacement", replacement_factory)
        await flush_tui(ui, terminal)
        assert replacement.focused is True

        closers["replacement"]("done")
        await replacement_done.wait(None)
        await flush_tui(ui, terminal)
        terminal.send_input("x")
        await flush_tui(ui, terminal)

        assert overlay.inputs == ["x"]
        assert editor.inputs == []
        assert overlay.focused is True

        closers["overlay"]("closed")
        await overlay_done.wait(None)
        assert results == {"overlay": "closed", "replacement": "done"}
    finally:
        await ui.stop()


class TestCreateExtensionUIContextAddAutocompleteProvider:
    def test_stores_wrapper_factories_and_rebuilds_autocomplete_immediately(self):
        def wrapper(current):
            return current

        setup_calls: list = []
        fake = SimpleNamespace(_autocomplete_provider_wrappers=[])
        fake._setup_autocomplete_provider = lambda: setup_calls.append(True)

        ui_context = InteractiveMode._create_extension_ui_context(fake)
        ui_context["addAutocompleteProvider"](wrapper)

        assert fake._autocomplete_provider_wrappers == [wrapper]
        assert setup_calls == [True]


class TestSetupAutocompleteProvider:
    def test_stacks_wrapper_factories_over_a_fresh_base_provider(self):
        default_editor_providers: list = []
        custom_editor_providers: list = []
        calls: list = []

        def make_wrap(tag):
            def wrap(current):
                async def get_suggestions(lines, cursor_line, cursor_col, options):
                    calls.append(f"getSuggestions:{tag}")
                    return await current.get_suggestions(lines, cursor_line, cursor_col, options)

                def apply_completion(lines, cursor_line, cursor_col, item, prefix):
                    calls.append(f"applyCompletion:{tag}")
                    return current.apply_completion(lines, cursor_line, cursor_col, item, prefix)

                def should_trigger_file_completion(lines, cursor_line, cursor_col):
                    calls.append(f"shouldTrigger:{tag}")
                    inner = getattr(current, "should_trigger_file_completion", None)
                    return inner(lines, cursor_line, cursor_col) if inner is not None else True

                return SimpleNamespace(
                    get_suggestions=get_suggestions,
                    apply_completion=apply_completion,
                    should_trigger_file_completion=should_trigger_file_completion,
                )

            return wrap

        fake = SimpleNamespace(
            _autocomplete_provider_wrappers=[make_wrap("wrap1"), make_wrap("wrap2")],
        )
        fake._create_base_autocomplete_provider = lambda: CombinedAutocompleteProvider([], "/tmp/project", None)
        fake._default_editor = SimpleNamespace(set_autocomplete_provider=default_editor_providers.append)
        fake.editor = SimpleNamespace(set_autocomplete_provider=custom_editor_providers.append)

        InteractiveMode._setup_autocomplete_provider(fake)

        assert len(default_editor_providers) == 1
        assert len(custom_editor_providers) == 1
        provider = default_editor_providers[0]
        assert provider is custom_editor_providers[0]
        assert provider.should_trigger_file_completion(["foo"], 0, 3) is True
        assert calls == ["shouldTrigger:wrap2", "shouldTrigger:wrap1"]

    def test_merges_trigger_characters_from_wrapper_factories(self):
        default_editor_providers: list = []

        def pass_through(trigger_characters):
            def wrap(current):
                return SimpleNamespace(
                    trigger_characters=trigger_characters,
                    get_suggestions=current.get_suggestions,
                    apply_completion=current.apply_completion,
                )

            return wrap

        fake = SimpleNamespace(
            _autocomplete_provider_wrappers=[pass_through(["$"]), pass_through(["!"])],
        )
        fake._create_base_autocomplete_provider = lambda: CombinedAutocompleteProvider([], "/tmp/project", None)
        fake._default_editor = SimpleNamespace(set_autocomplete_provider=default_editor_providers.append)
        fake.editor = SimpleNamespace(set_autocomplete_provider=lambda provider: None)

        InteractiveMode._setup_autocomplete_provider(fake)

        provider = default_editor_providers[0]
        assert provider.trigger_characters == ["$", "!"]


def _create_base_provider_fake(models=None, login_provider_options=None):
    async def get_available():
        return models or []

    fake = _BareInteractiveMode()
    fake.session = SimpleNamespace(
        scoped_models=[],
        model_runtime=SimpleNamespace(get_available=get_available),
        prompt_templates=[],
        extension_runner=SimpleNamespace(get_registered_commands=list),
        resource_loader=SimpleNamespace(get_skills=lambda: SimpleNamespace(skills=[])),
    )
    fake.settings_manager = SimpleNamespace(get_enable_skill_commands=lambda: False)
    fake._skill_commands = {}
    fake.session_manager = SimpleNamespace(get_cwd=lambda: "/tmp")
    fake._fd_path = None
    if login_provider_options is not None:
        fake.get_login_provider_options = lambda auth_type=None: login_provider_options
    return fake


class TestCreateBaseAutocompleteProvider:
    @pytest.mark.tonio
    async def test_matches_model_command_arguments_across_provider_model_order(self):
        models = [
            SimpleNamespace(id="gpt-5.2-codex", provider="github-copilot", name="GPT-5.2 Codex"),
            SimpleNamespace(id="gpt-5.5", provider="openai-codex", name="GPT-5.5"),
        ]
        fake = _create_base_provider_fake(models=models)

        provider = InteractiveMode._create_base_autocomplete_provider(fake)
        line = "/model codexgpt"
        suggestions = await provider.get_suggestions([line], 0, len(line), {})

        assert [item["value"] for item in suggestions["items"]] == [
            "openai-codex/gpt-5.5",
            "github-copilot/gpt-5.2-codex",
        ]

    @pytest.mark.tonio
    async def test_matches_login_command_arguments_by_provider_id_and_name(self):
        fake = _create_base_provider_fake(
            login_provider_options=[
                {"id": "anthropic", "name": "Anthropic", "authType": "oauth"},
                {"id": "anthropic", "name": "Anthropic", "authType": "api_key"},
                {"id": "openai", "name": "OpenAI", "authType": "api_key"},
            ]
        )

        provider = InteractiveMode._create_base_autocomplete_provider(fake)
        line = "/login subscription anthrop"
        suggestions = await provider.get_suggestions([line], 0, len(line), {})

        assert suggestions["items"] == [
            {
                "value": "anthropic",
                "label": "anthropic",
                "description": "Anthropic · subscription/API key",
            }
        ]


def create_source_info(file_path, *, source, scope, origin, base_dir=None):
    return SourceInfo(path=file_path, source=source, scope=scope, origin=origin, base_dir=base_dir)


def create_show_loaded_resources_fake(
    *,
    quiet_startup,
    verbose=False,
    tool_output_expanded=False,
    cwd=None,
    context_files=None,
    extensions=None,
    skills=None,
    skill_diagnostics=None,
    use_real_scope_groups=False,
):
    fake = _BareInteractiveMode()
    fake._options = {"verbose": verbose}
    fake._tool_output_expanded = tool_output_expanded
    fake._loaded_resources_container = Container()
    fake._chat_container = Container()
    fake.settings_manager = SimpleNamespace(get_quiet_startup=lambda: quiet_startup)
    fake.session_manager = SimpleNamespace(get_cwd=lambda: cwd if cwd is not None else "/tmp/project")
    fake.session = SimpleNamespace(
        prompt_templates=[],
        extension_runner=SimpleNamespace(
            get_command_diagnostics=list,
            get_shortcut_diagnostics=list,
        ),
        resource_loader=SimpleNamespace(
            get_agents_files=lambda: [
                SimpleNamespace(path=item["path"]) for item in (context_files or [])
            ],
            get_skills=lambda: SimpleNamespace(
                skills=[
                    SimpleNamespace(file_path=skill["filePath"], name=skill["name"], source_info=None)
                    for skill in (skills or [])
                ],
                diagnostics=list(skill_diagnostics or []),
            ),
            get_prompts=lambda: SimpleNamespace(prompts=[], diagnostics=[]),
            get_extensions=lambda: SimpleNamespace(
                extensions=[
                    SimpleNamespace(path=ext["path"], source_info=ext.get("sourceInfo"), hidden=False)
                    for ext in (extensions or [])
                ],
                errors=[],
            ),
            get_themes=lambda: {"themes": [], "diagnostics": []},
        ),
    )
    if not use_real_scope_groups:
        fake._build_scope_groups = lambda items: []
        fake._format_scope_groups = lambda groups, options=None: "resource-list"
    fake._format_diagnostics = lambda diagnostics, source_infos: "diagnostics"
    fake._get_built_in_command_conflict_diagnostics = lambda runner: []
    return fake


def create_extension_fixtures():
    return [
        {
            "path": "/tmp/project/.pi/extensions/answer.ts",
            "sourceInfo": create_source_info(
                "/tmp/project/.pi/extensions/answer.ts",
                source="local",
                scope="project",
                origin="top-level",
                base_dir="/tmp/project/.pi/extensions",
            ),
        },
        {
            "path": "/tmp/project/.pi/extensions/local-index/index.ts",
            "sourceInfo": create_source_info(
                "/tmp/project/.pi/extensions/local-index/index.ts",
                source="local",
                scope="project",
                origin="top-level",
                base_dir="/tmp/project/.pi/extensions",
            ),
        },
        {
            "path": "/tmp/agent/extensions/user-index/index.ts",
            "sourceInfo": create_source_info(
                "/tmp/agent/extensions/user-index/index.ts",
                source="local",
                scope="user",
                origin="top-level",
                base_dir="/tmp/agent/extensions",
            ),
        },
        {
            "path": "/tmp/project/.pi/npm/node_modules/pi-markdown-preview/extensions/index.ts",
            "sourceInfo": create_source_info(
                "/tmp/project/.pi/npm/node_modules/pi-markdown-preview/extensions/index.ts",
                source="npm:pi-markdown-preview",
                scope="project",
                origin="package",
                base_dir="/tmp/project/.pi/npm/node_modules/pi-markdown-preview",
            ),
        },
        {
            "path": "/tmp/project/.pi/npm/node_modules/@scope/pi-scoped/extensions/index.ts",
            "sourceInfo": create_source_info(
                "/tmp/project/.pi/npm/node_modules/@scope/pi-scoped/extensions/index.ts",
                source="npm:@scope/pi-scoped",
                scope="project",
                origin="package",
                base_dir="/tmp/project/.pi/npm/node_modules/@scope/pi-scoped",
            ),
        },
        {
            "path": "/tmp/project/.pi/git/github.com/HazAT/pi-interactive-subagents/extensions/index.ts",
            "sourceInfo": create_source_info(
                "/tmp/project/.pi/git/github.com/HazAT/pi-interactive-subagents/extensions/index.ts",
                source="git:github.com/HazAT/pi-interactive-subagents",
                scope="project",
                origin="package",
                base_dir="/tmp/project/.pi/git/github.com/HazAT/pi-interactive-subagents",
            ),
        },
        {
            "path": "/tmp/project/.pi/git/github.com/HazAT/pi-interactive-subagents/extensions/subagents/index.ts",
            "sourceInfo": create_source_info(
                "/tmp/project/.pi/git/github.com/HazAT/pi-interactive-subagents/extensions/subagents/index.ts",
                source="git:github.com/HazAT/pi-interactive-subagents",
                scope="project",
                origin="package",
                base_dir="/tmp/project/.pi/git/github.com/HazAT/pi-interactive-subagents",
            ),
        },
        {
            "path": "/tmp/temp/cli-extension.ts",
            "sourceInfo": create_source_info(
                "/tmp/temp/cli-extension.ts",
                source="cli",
                scope="temporary",
                origin="top-level",
                base_dir="/tmp/temp",
            ),
        },
    ]


class TestShowLoadedResources:
    def test_shows_a_compact_resource_listing_by_default(self):
        fake = create_show_loaded_resources_fake(
            quiet_startup=False,
            skills=[{"filePath": "/tmp/skill/SKILL.md", "name": "commit"}],
        )

        InteractiveMode._show_loaded_resources(fake, {"force": False})

        output = render_all(fake._loaded_resources_container)
        assert "[Skills]" in output
        assert "commit" in output
        assert "resource-list" not in output

    def test_shows_full_resource_listing_when_expanded(self):
        fake = create_show_loaded_resources_fake(
            quiet_startup=False,
            tool_output_expanded=True,
            skills=[{"filePath": "/tmp/skill/SKILL.md", "name": "commit"}],
        )

        InteractiveMode._show_loaded_resources(fake, {"force": False})

        output = render_all(fake._loaded_resources_container)
        assert "[Skills]" in output
        assert "resource-list" in output
        assert "commit" not in output

    def test_shows_full_resource_listing_on_verbose_startup_even_when_tool_output_is_collapsed(self):
        fake = create_show_loaded_resources_fake(
            quiet_startup=True,
            verbose=True,
            tool_output_expanded=False,
            skills=[{"filePath": "/tmp/skill/SKILL.md", "name": "commit"}],
        )

        InteractiveMode._show_loaded_resources(fake, {"force": False})

        output = render_all(fake._loaded_resources_container)
        assert "[Skills]" in output
        assert "resource-list" in output
        assert "commit" not in output

    def test_abbreviates_extensions_in_compact_listing(self):
        fake = create_show_loaded_resources_fake(
            quiet_startup=False,
            extensions=[{"path": "/tmp/extensions/answer.ts"}, {"path": "/tmp/extensions/btw.ts"}],
        )

        InteractiveMode._show_loaded_resources(fake, {"force": False})

        output = render_all(fake._loaded_resources_container)
        assert "[Extensions]" in output
        assert "answer.ts, btw.ts" in output
        assert "extensions/answer.ts" not in output

    def test_captures_mixed_extension_layouts_in_compact_output(self):
        fake = create_show_loaded_resources_fake(
            quiet_startup=False,
            extensions=create_extension_fixtures(),
            use_real_scope_groups=True,
        )

        InteractiveMode._show_loaded_resources(fake, {"force": False})

        assert normalize_rendered_output(fake._loaded_resources_container) == (
            "[Extensions]\n"
            "  @scope/pi-scoped, answer.ts, cli-extension.ts, HazAT/pi-interactive-subagents, "
            "HazAT/pi-interactive-subagents:subagents, local-index, pi-markdown-preview, user-index"
        )

    def test_adds_more_parent_folders_until_local_extension_labels_are_unique(self):
        extensions = [
            {
                "path": f"/tmp/{name}/one/index.ts",
                "sourceInfo": create_source_info(
                    f"/tmp/{name}/one/index.ts",
                    source="cli",
                    scope="temporary",
                    origin="top-level",
                    base_dir=f"/tmp/{name}",
                ),
            }
            for name in ("alpha", "beta", "gamma")
        ]

        fake = create_show_loaded_resources_fake(
            quiet_startup=False,
            extensions=extensions,
            use_real_scope_groups=True,
        )

        InteractiveMode._show_loaded_resources(fake, {"force": False})

        assert normalize_rendered_output(fake._loaded_resources_container) == (
            "[Extensions]\n  alpha/one, beta/one, gamma/one"
        )

    def test_strips_index_ts_from_local_extension_label_showing_parent_dir(self):
        extensions = [
            {
                "path": "/tmp/extensions/plan-mode/index.ts",
                "sourceInfo": create_source_info(
                    "/tmp/extensions/plan-mode/index.ts",
                    source="local",
                    scope="project",
                    origin="top-level",
                    base_dir="/tmp/extensions",
                ),
            }
        ]

        fake = create_show_loaded_resources_fake(
            quiet_startup=False, extensions=extensions, use_real_scope_groups=True
        )

        InteractiveMode._show_loaded_resources(fake, {"force": False})

        assert normalize_rendered_output(fake._loaded_resources_container) == "[Extensions]\n  plan-mode"

    def test_strips_index_js_from_local_extension_label_showing_parent_dir(self):
        extensions = [
            {
                "path": "/tmp/extensions/plan-mode/index.js",
                "sourceInfo": create_source_info(
                    "/tmp/extensions/plan-mode/index.js",
                    source="local",
                    scope="project",
                    origin="top-level",
                    base_dir="/tmp/extensions",
                ),
            }
        ]

        fake = create_show_loaded_resources_fake(
            quiet_startup=False, extensions=extensions, use_real_scope_groups=True
        )

        InteractiveMode._show_loaded_resources(fake, {"force": False})

        assert normalize_rendered_output(fake._loaded_resources_container) == "[Extensions]\n  plan-mode"

    def test_mixed_single_file_and_subdirectory_index_ts_extensions_strip_index_ts(self):
        extensions = [
            {
                "path": "/tmp/extensions/webfetch.ts",
                "sourceInfo": create_source_info(
                    "/tmp/extensions/webfetch.ts",
                    source="local",
                    scope="project",
                    origin="top-level",
                    base_dir="/tmp/extensions",
                ),
            },
            {
                "path": "/tmp/extensions/plan-mode/index.ts",
                "sourceInfo": create_source_info(
                    "/tmp/extensions/plan-mode/index.ts",
                    source="local",
                    scope="project",
                    origin="top-level",
                    base_dir="/tmp/extensions",
                ),
            },
        ]

        fake = create_show_loaded_resources_fake(
            quiet_startup=False, extensions=extensions, use_real_scope_groups=True
        )

        InteractiveMode._show_loaded_resources(fake, {"force": False})

        assert normalize_rendered_output(fake._loaded_resources_container) == (
            "[Extensions]\n  plan-mode, webfetch.ts"
        )

    def test_multiple_index_ts_with_unique_parent_dirs_need_no_disambiguation(self):
        extensions = [
            {
                "path": f"/tmp/extensions/{name}/index.ts",
                "sourceInfo": create_source_info(
                    f"/tmp/extensions/{name}/index.ts",
                    source="local",
                    scope="project",
                    origin="top-level",
                    base_dir="/tmp/extensions",
                ),
            }
            for name in ("foo", "bar")
        ]

        fake = create_show_loaded_resources_fake(
            quiet_startup=False, extensions=extensions, use_real_scope_groups=True
        )

        InteractiveMode._show_loaded_resources(fake, {"force": False})

        assert normalize_rendered_output(fake._loaded_resources_container) == "[Extensions]\n  bar, foo"

    def test_multiple_index_ts_with_same_parent_dir_name_disambiguated_with_grandparent(self):
        extensions = [
            {
                "path": f"/tmp/{name}/tools/index.ts",
                "sourceInfo": create_source_info(
                    f"/tmp/{name}/tools/index.ts",
                    source="cli",
                    scope="temporary",
                    origin="top-level",
                    base_dir=f"/tmp/{name}",
                ),
            }
            for name in ("alpha", "beta")
        ]

        fake = create_show_loaded_resources_fake(
            quiet_startup=False, extensions=extensions, use_real_scope_groups=True
        )

        InteractiveMode._show_loaded_resources(fake, {"force": False})

        assert normalize_rendered_output(fake._loaded_resources_container) == (
            "[Extensions]\n  alpha/tools, beta/tools"
        )

    def test_non_index_file_in_subdirectory_stays_as_filename(self):
        extensions = [
            {
                "path": "/tmp/extensions/my-ext/main.ts",
                "sourceInfo": create_source_info(
                    "/tmp/extensions/my-ext/main.ts",
                    source="local",
                    scope="project",
                    origin="top-level",
                    base_dir="/tmp/extensions",
                ),
            }
        ]

        fake = create_show_loaded_resources_fake(
            quiet_startup=False, extensions=extensions, use_real_scope_groups=True
        )

        InteractiveMode._show_loaded_resources(fake, {"force": False})

        assert normalize_rendered_output(fake._loaded_resources_container) == "[Extensions]\n  main.ts"

    def test_package_extensions_still_strip_index_ts_correctly_regression_guard(self):
        extensions = [
            {
                "path": "/tmp/project/.pi/npm/node_modules/pi-markdown-preview/extensions/index.ts",
                "sourceInfo": create_source_info(
                    "/tmp/project/.pi/npm/node_modules/pi-markdown-preview/extensions/index.ts",
                    source="npm:pi-markdown-preview",
                    scope="project",
                    origin="package",
                    base_dir="/tmp/project/.pi/npm/node_modules/pi-markdown-preview",
                ),
            }
        ]

        fake = create_show_loaded_resources_fake(
            quiet_startup=False, extensions=extensions, use_real_scope_groups=True
        )

        InteractiveMode._show_loaded_resources(fake, {"force": False})

        assert normalize_rendered_output(fake._loaded_resources_container) == (
            "[Extensions]\n  pi-markdown-preview"
        )

    def test_labels_npm_sibling_extensions_relative_to_the_declaring_package(self):
        extensions = [
            {
                "path": "/tmp/project/.pi/npm/node_modules/primary-package/index.ts",
                "sourceInfo": create_source_info(
                    "/tmp/project/.pi/npm/node_modules/primary-package/index.ts",
                    source="npm:primary-package",
                    scope="project",
                    origin="package",
                    base_dir="/tmp/project/.pi/npm/node_modules/primary-package",
                ),
            },
            {
                "path": "/tmp/project/.pi/npm/node_modules/sibling-package/index.ts",
                "sourceInfo": create_source_info(
                    "/tmp/project/.pi/npm/node_modules/sibling-package/index.ts",
                    source="npm:primary-package",
                    scope="project",
                    origin="package",
                    base_dir="/tmp/project/.pi/npm/node_modules/primary-package",
                ),
            },
        ]

        fake = create_show_loaded_resources_fake(
            quiet_startup=False, extensions=extensions, use_real_scope_groups=True
        )

        InteractiveMode._show_loaded_resources(fake, {"force": False})

        assert normalize_rendered_output(fake._loaded_resources_container) == (
            "[Extensions]\n  primary-package, primary-package:../sibling-package"
        )

    def test_captures_mixed_extension_layouts_in_expanded_output(self):
        fake = create_show_loaded_resources_fake(
            quiet_startup=False,
            tool_output_expanded=True,
            extensions=create_extension_fixtures(),
            use_real_scope_groups=True,
        )

        InteractiveMode._show_loaded_resources(fake, {"force": False})

        assert normalize_rendered_output(fake._loaded_resources_container) == (
            "[Extensions]\n"
            "  project\n"
            "    /tmp/project/.pi/extensions/answer.ts\n"
            "    /tmp/project/.pi/extensions/local-index\n"
            "    git:github.com/HazAT/pi-interactive-subagents\n"
            "      extensions\n"
            "      extensions/subagents\n"
            "    npm:@scope/pi-scoped\n"
            "      extensions\n"
            "    npm:pi-markdown-preview\n"
            "      extensions\n"
            "  user\n"
            "    /tmp/agent/extensions/user-index\n"
            "  path\n"
            "    /tmp/temp/cli-extension.ts"
        )

    def test_shows_context_paths_relative_to_cwd_while_preserving_full_external_paths(self):
        home = os.path.expanduser("~")
        cwd = os.path.join(home, "Development", "pi-mono")
        fake = create_show_loaded_resources_fake(
            quiet_startup=False,
            cwd=cwd,
            context_files=[
                {"path": os.path.join(home, ".pi", "agent", "AGENTS.md")},
                {"path": os.path.join(cwd, "AGENTS.md")},
            ],
        )

        InteractiveMode._show_loaded_resources(fake, {"force": False})

        output = render_all(fake._loaded_resources_container)
        assert "[Context]" in output
        assert "~/.pi/agent/AGENTS.md, AGENTS.md" in output
        assert f"{cwd}/AGENTS.md" not in output

    def test_shows_full_context_paths_when_expanded(self):
        home = os.path.expanduser("~")
        cwd = os.path.join(home, "Development", "pi-mono")
        fake = create_show_loaded_resources_fake(
            quiet_startup=False,
            tool_output_expanded=True,
            cwd=cwd,
            context_files=[
                {"path": os.path.join(home, ".pi", "agent", "AGENTS.md")},
                {"path": os.path.join(cwd, "AGENTS.md")},
            ],
        )

        InteractiveMode._show_loaded_resources(fake, {"force": False})

        output = render_all(fake._loaded_resources_container)
        assert "[Context]" in output
        assert "~/.pi/agent/AGENTS.md" in output
        assert "~/Development/pi-mono/AGENTS.md" in output
        assert "~/.pi/agent/AGENTS.md, AGENTS.md" not in output

    def test_does_not_show_verbose_listing_on_quiet_startup_during_reload(self):
        fake = create_show_loaded_resources_fake(
            quiet_startup=True,
            skills=[{"filePath": "/tmp/skill/SKILL.md", "name": "commit"}],
        )

        InteractiveMode._show_loaded_resources(
            fake,
            {"extensions": [{"path": "/tmp/ext/index.ts"}], "force": False, "showDiagnosticsWhenQuiet": True},
        )

        assert len(fake._loaded_resources_container.children) == 0

    def test_still_shows_diagnostics_on_quiet_startup_when_requested(self):
        fake = create_show_loaded_resources_fake(
            quiet_startup=True,
            skills=[{"filePath": "/tmp/skill/SKILL.md", "name": "commit"}],
            skill_diagnostics=[{"type": "warning", "message": "duplicate skill name"}],
        )

        InteractiveMode._show_loaded_resources(fake, {"force": False, "showDiagnosticsWhenQuiet": True})

        output = render_all(fake._loaded_resources_container)
        assert "[Skill conflicts]" in output
        assert "[Skills]" not in output
