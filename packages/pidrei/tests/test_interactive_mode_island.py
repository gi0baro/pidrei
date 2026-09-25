"""pidrei-specific: the TUI-island wiring of interactive mode's agent listener
(PROPER_MT_DESIGN.md step 1).

No pi counterpart: pi's single JS thread makes every listener owner-side for
free. Here `_route_event` posts each session event's application to the UI
owner (`TuiBase.post_ui`) and `_settle_ui_after_agent_event` is the
per-agent-event owner barrier that keeps the fused emit contract
("listeners settled" ⇒ UI updated).
"""

import threading
from types import SimpleNamespace

import pytest
import tonio.colored as tonio
from tonio.colored import sync as tonio_sync

from pidrei.core.keybindings import KeybindingsManager
from pidrei.modes.interactive import interactive_mode
from pidrei.modes.interactive.components.extension_selector import ExtensionSelectorComponent
from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei.modes.interactive.theme import init_theme_sync
from pidrei_tui import Container, set_keybindings
from pidrei_tui._owner import OwnerTask
from pidrei_tui.tui import TuiBase

from .ui_timer_helpers import manual_ui_timers


class _ManualOwner:
    """A UI owner queue nothing applies until the test drains it — the window
    in which off-owner code runs ahead of the owner."""

    def __init__(self) -> None:
        self.jobs: list = []

    def post_ui(self, fn) -> None:
        self.jobs.append(fn)

    def drain(self) -> None:
        while self.jobs:  # applies may post more
            self.jobs.pop(0)()


class _Ui:
    """The slice of a TUI the routing touches, with the real `post_ui`."""

    post_ui = TuiBase.post_ui

    def __init__(self, owner: OwnerTask) -> None:
        self.input_owner = owner


def _fake_mode(owner: OwnerTask):
    applied: list = []
    fake = SimpleNamespace(applied=applied, ui=_Ui(owner))
    fake._handle_event = applied.append
    return fake


@pytest.mark.tonio
async def test_routed_events_apply_on_the_owner_in_order_before_the_barrier_settles():
    owner = OwnerTask()
    async with tonio.scope() as scope:
        owner.start(scope)
        fake = _fake_mode(owner)

        events = [f"event-{i}" for i in range(5)]
        for event in events:
            InteractiveMode._route_event(fake, event)
        # (No "nothing applied yet" assert here: `post` guarantees FIFO, not
        # delayed application — the owner may legally apply an event before
        # this task reaches the barrier.)

        # The awaited barrier means everything routed during the emit
        # has been applied when it returns — the fused contract.
        await InteractiveMode._settle_ui_after_agent_event(fake, None)
        assert fake.applied == events

        owner.close()


@pytest.mark.tonio
async def test_events_routed_before_the_owner_starts_apply_at_start_in_order():
    # `post` always enqueues: work posted before start() runs when the owner
    # starts, still in order.
    owner = OwnerTask()
    fake = _fake_mode(owner)
    InteractiveMode._route_event(fake, "early-one")
    InteractiveMode._route_event(fake, "early-two")
    assert fake.applied == []

    async with tonio.scope() as scope:
        owner.start(scope)
        await InteractiveMode._settle_ui_after_agent_event(fake, None)
        assert fake.applied == ["early-one", "early-two"]
        owner.close()


@pytest.mark.tonio
async def test_barrier_orders_after_mutations_posted_by_helper_wrappers():
    """A wrapped helper (`ui.post_ui`) called while an event applies posts
    behind the event's job; the barrier still settles after it — order is
    the post order, on one task."""
    owner = OwnerTask()
    async with tonio.scope() as scope:
        owner.start(scope)
        order: list = []
        fake = SimpleNamespace(ui=_Ui(owner))

        def handle_event(event) -> None:
            order.append(("event", event))
            if event == "first":
                fake.ui.post_ui(lambda: order.append(("helper", event)))

        fake._handle_event = handle_event

        InteractiveMode._route_event(fake, "first")
        InteractiveMode._route_event(fake, "second")
        await InteractiveMode._settle_ui_after_agent_event(fake, None)
        assert order == [("event", "first"), ("event", "second"), ("helper", "first")]

        owner.close()


def test_extension_state_registered_after_a_reset_survives_the_posted_reset():
    """`_reset_extension_ui` runs off the owner (session invalidation, /reload)
    and posts its component work. Extension state the next extensions write
    directly (here: an autocomplete wrapper) resets synchronously, so the
    posted reset, landing after the registration, does not wipe it."""
    posted: list = []  # a manual owner queue: nothing applies until drained
    installed: list = []
    base = SimpleNamespace(name="base")
    default_editor = SimpleNamespace(
        set_autocomplete_provider=installed.append,
        on_extension_shortcut=lambda data: False,
    )
    noop = lambda *args, **kwargs: None
    fake = SimpleNamespace(
        ui=SimpleNamespace(post_ui=posted.append, hide_overlay=noop),
        _autocomplete_provider_wrappers=(lambda current: SimpleNamespace(stale=current),),
        _extension_registry_guard=threading.Lock(),
        _editor_component_factory=None,
        _default_editor=default_editor,
        editor=default_editor,
        _create_base_autocomplete_provider=lambda: base,
        _clear_extension_terminal_input_listeners=noop,
        _footer_data_provider=SimpleNamespace(clear_extension_statuses=noop),
        _extension_selector=None,
        _extension_input=None,
        _extension_editor=None,
        _set_extension_footer=noop,
        _set_extension_header=noop,
        _clear_extension_widgets=noop,
        _footer=SimpleNamespace(invalidate=noop),
        _update_terminal_title=noop,
        _set_working_indicator=noop,
        _active_status_indicator=None,
        _set_hidden_thinking_label=noop,
        _apply_custom_editor_component=noop,
    )
    fake._set_custom_editor_component = lambda factory: InteractiveMode._set_custom_editor_component(fake, factory)
    fake._setup_autocomplete_provider = lambda: InteractiveMode._setup_autocomplete_provider(fake)
    fake._apply_autocomplete_provider = lambda: InteractiveMode._apply_autocomplete_provider(fake)

    InteractiveMode._reset_extension_ui(fake)
    # The next session's extension registers before the owner ran the reset.
    InteractiveMode._create_extension_ui_context(fake).add_autocomplete_provider(
        lambda current: SimpleNamespace(wrapped=current)
    )
    for job in posted:
        job()

    assert len(fake._autocomplete_provider_wrappers) == 1
    assert installed[-1].wrapped is base


def _noop(*_args, **_kwargs) -> None:
    return None


def _editor_slot_mode(owner: _ManualOwner):
    """Fake `this` for the extension dialogs and the editor-slot helpers."""
    fake = SimpleNamespace(
        ui=SimpleNamespace(post_ui=owner.post_ui, set_focus=_noop, request_render=_noop),
        editor=SimpleNamespace(name="editor"),
        _editor_container=Container(),
        _extension_selector=None,
        _dispose_active_selector=_noop,
        _toggle_tool_output_expansion=_noop,
    )
    fake._hide_extension_selector = lambda component: InteractiveMode._hide_extension_selector(fake, component)
    return fake


class _RecordingSelector(ExtensionSelectorComponent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.disposed = False

    def dispose(self) -> None:
        self.disposed = True
        super().dispose()


@pytest.mark.tonio
async def test_a_dialog_opened_before_the_previous_one_is_hidden_survives_that_hide(monkeypatch):
    # The extension resumes as soon as dialog A settles, and may open dialog
    # B before the owner applied A's hide: the hide must act on A, not on
    # whichever dialog the field names by then (it used to dispose B and
    # clear the field, leaving a dead dialog mounted).
    init_theme_sync("dark")
    set_keybindings(KeybindingsManager())
    monkeypatch.setattr(interactive_mode, "ExtensionSelectorComponent", _RecordingSelector)
    owner = _ManualOwner()
    fake = _editor_slot_mode(owner)

    first_wait = InteractiveMode._show_extension_selector(fake, "first", ["Yes", "No"])
    owner.drain()
    first = fake._extension_selector
    await first.handle_input("\n")  # settles A: its hide is posted
    assert await first_wait == "Yes"

    second_wait = InteractiveMode._show_extension_selector(fake, "second", ["Yes", "No"])
    owner.drain()
    second_wait.close()

    second = fake._extension_selector
    assert second is not first
    assert first.disposed and not second.disposed
    assert fake._editor_container.children == [second]


def test_a_share_restore_landing_after_another_component_took_the_slot_leaves_it():
    # Esc on the gist loader restores the editor on the owner while the share
    # task may be restoring too: the late restore must not evict whatever
    # the user opened in between.
    owner = _ManualOwner()
    fake = _editor_slot_mode(owner)
    loader = SimpleNamespace(dispose=_noop)
    fake._editor_container.add_child(loader)

    InteractiveMode._restore_share_editor(fake, loader)  # Esc
    owner.drain()
    selector = SimpleNamespace(name="selector")
    fake._editor_container.clear()
    fake._editor_container.add_child(selector)  # the user opens a selector
    InteractiveMode._restore_share_editor(fake, loader)  # the share task finishes
    owner.drain()

    assert fake._editor_container.children == [selector]


@pytest.mark.tonio
async def test_the_new_session_notice_lands_after_the_posted_chat_reset():
    # `new_session()` rebinds the session, which posts the chat reset; the
    # notice appended directly used to land first and be wiped by it.
    owner = _ManualOwner()
    chat = Container()
    fake = SimpleNamespace(ui=SimpleNamespace(post_ui=owner.post_ui, request_render=_noop), _chat_container=chat)
    fake._clear_status_indicator = _noop
    fake._append_to_chat = lambda *components: InteractiveMode._append_to_chat(fake, *components)

    async def new_session() -> dict:
        owner.post_ui(chat.clear)  # the rebind's posted `render_current_session_state`
        return {"cancelled": False}

    fake.runtime_host = SimpleNamespace(new_session=new_session)
    init_theme_sync("dark")
    await InteractiveMode._handle_clear_command(fake)
    owner.drain()

    assert len(chat.children) == 2  # spacer + "New session started"


def test_tools_expansion_is_readable_at_once_and_applied_on_the_owner():
    # `ctx.ui.set_tools_expanded` runs on the extension's task and
    # `get_tools_expanded` reads the flag synchronously right after.
    owner = _ManualOwner()
    expanded_children: list = []
    fake = SimpleNamespace(
        _tool_output_expanded=False,
        _tool_output_expanded_guard=threading.Lock(),
        _custom_header=None,
        _built_in_header=None,
        _loaded_resources_container=SimpleNamespace(children=[]),
        _chat_container=SimpleNamespace(children=[SimpleNamespace(set_expanded=expanded_children.append)]),
        ui=SimpleNamespace(post_ui=owner.post_ui),
        show_status=_noop,
    )

    InteractiveMode.set_tools_expanded(fake, True)
    assert fake._tool_output_expanded is True
    assert expanded_children == []  # component work waits for the owner
    owner.drain()
    assert expanded_children == [True]


@pytest.mark.tonio
async def test_concurrent_bash_commands_keep_their_own_output():
    # pi keeps the running command's component in one field; two `!`
    # commands run concurrently here, and the second used to take the
    # first's output chunks.
    init_theme_sync("dark")
    set_keybindings(KeybindingsManager())
    owner = _ManualOwner()
    chat = Container()
    both_started = tonio.Event()
    started: list = []
    chunks_sent = tonio.Event()

    async def execute_bash(command, on_chunk, _options):
        started.append(command)
        if len(started) == 2:
            both_started.set()
        await both_started.wait(5)
        on_chunk(f"{command}-out\n")
        if command == "first":
            chunks_sent.set()
        else:
            await chunks_sent.wait(5)
        return SimpleNamespace(exit_code=0, cancelled=False, truncated=False, output="", full_output_path=None)

    async def emit_user_bash(_event):
        return None

    session = SimpleNamespace(
        extension_runner=SimpleNamespace(emit_user_bash=emit_user_bash),
        is_streaming=False,
        execute_bash=execute_bash,
    )
    fake = SimpleNamespace(
        ui=SimpleNamespace(post_ui=owner.post_ui, request_render=_noop),
        session=session,
        session_manager=SimpleNamespace(get_cwd=lambda: "/tmp"),
        _chat_container=chat,
        _pending_messages_container=Container(),
        _pending_bash_components=[],
        show_error=_noop,
    )
    fake._mount_bash_component = lambda component, deferred: InteractiveMode._mount_bash_component(
        fake, component, deferred
    )

    with manual_ui_timers():
        async with tonio.scope() as scope:
            scope.spawn(InteractiveMode._handle_bash_command(fake, "first"))
            scope.spawn(InteractiveMode._handle_bash_command(fake, "second"))
        owner.drain()

    outputs = {component.get_command(): component.get_output() for component in chat.children}
    assert outputs == {"first": "first-out\n", "second": "second-out\n"}


class _TextEditor:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.on_submit = None

    def get_text(self) -> str:
        return self.text

    def set_text(self, text: str) -> None:
        self.text = text


@pytest.mark.tonio
async def test_follow_up_keeps_keys_typed_right_after_it():
    # The Alt+Enter action runs on the owner: the editor is read and cleared
    # there and then. A clear posted for later would land behind the next
    # keys and wipe them.
    owner = _ManualOwner()
    prompted = tonio.Event()
    editor = _TextEditor("queued question")

    async def prompt(_text, _options=None) -> None:
        prompted.set()

    fake = SimpleNamespace(
        editor=editor,
        session=SimpleNamespace(is_compacting=False, is_streaming=True, prompt=prompt),
        ui=SimpleNamespace(post_ui=owner.post_ui, request_render=_noop, input_owner=OwnerTask()),
        _add_editor_history=_noop,
        _update_pending_messages_display=_noop,
    )
    fake._set_editor_text = lambda text: InteractiveMode._set_editor_text(fake, text)
    fake._post_editor_mutation = lambda fn: owner.post_ui(fn)
    fake._clear_editor_on_owner = lambda: InteractiveMode._clear_editor_on_owner(fake)
    fake._queue_follow_up = lambda text: InteractiveMode._queue_follow_up(fake, text)

    InteractiveMode._handle_follow_up(fake)
    editor.text += "next"  # typed right after Alt+Enter
    owner.drain()
    await prompted.wait(5)

    assert prompted.is_set()
    assert editor.text == "next"


def test_a_posted_editor_update_reaches_the_editor_current_when_it_applies():
    # A custom-editor swap queued ahead of the update must not leave it
    # writing into the editor that was current when it was posted.
    owner = _ManualOwner()
    previous, current = _TextEditor(), _TextEditor()
    fake = SimpleNamespace(editor=previous)
    fake._post_editor_mutation = owner.post_ui

    InteractiveMode._set_editor_text(fake, "restored")
    InteractiveMode._insert_into_editor(fake, "!")  # no insert API: skipped

    def swap() -> None:
        fake.editor = current

    owner.jobs.insert(0, swap)  # the swap was queued first
    owner.drain()

    assert (previous.text, current.text) == ("", "restored")


def test_extension_statuses_a_reader_holds_do_not_change_under_it():
    # The footer iterates the statuses on the owner while extensions set
    # them from their own tasks.
    from pidrei.core.footer_data_provider import FooterDataProvider

    provider = FooterDataProvider("/tmp")
    provider.set_extension_status("first", "one")
    held = provider.get_extension_statuses()
    provider.set_extension_status("second", "two")
    provider.set_extension_status("first", None)
    provider.clear_extension_statuses()

    assert held == {"first": "one"}
    assert provider.get_extension_statuses() == {}


@pytest.mark.tonio
async def test_overlapping_subscription_auth_checks_warn_once():
    # The check runs detached at startup and on every model switch; two in
    # flight both pass the early "already shown" test.
    both_waiting = tonio_sync.Barrier(2)
    warnings: list = []

    async def check_auth(_provider):
        await both_waiting.wait()
        return SimpleNamespace(type="oauth")

    fake = SimpleNamespace(
        _anthropic_subscription_warning_shown=False,
        _anthropic_subscription_warning_guard=threading.Lock(),
        settings_manager=SimpleNamespace(get_warnings=dict),
        session=SimpleNamespace(model_runtime=SimpleNamespace(check_auth=check_auth)),
        show_warning=warnings.append,
    )
    fake._show_anthropic_subscription_warning_once = lambda: InteractiveMode._show_anthropic_subscription_warning_once(
        fake
    )
    model = SimpleNamespace(provider="anthropic")

    async with tonio.scope() as scope:
        scope.spawn(InteractiveMode._maybe_warn_about_anthropic_subscription_auth(fake, model))
        scope.spawn(InteractiveMode._maybe_warn_about_anthropic_subscription_auth(fake, model))

    assert len(warnings) == 1
