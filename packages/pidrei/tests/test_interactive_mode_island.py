"""pidrei-specific: the TUI-island wiring of interactive mode.

No pi counterpart: pi's single JS thread makes every listener UI-side for
free. Here the session listener applies each event in place under the UI
state lock (spec/ui-island.md, "Agent events"), and flows apply each stretch
between their awaits in one hold of that lock (spec/ui-island.md, "Whole
changes").
"""

import threading
from functools import partial
from types import SimpleNamespace

import pytest
import tonio.colored as tonio
from tonio.colored import sync as tonio_sync

from pidrei.core.keybindings import KeybindingsManager
from pidrei.modes.interactive import interactive_mode
from pidrei.modes.interactive.components.extension_selector import ExtensionSelectorComponent
from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei.modes.interactive.theme import init_theme
from pidrei_tui import Container, set_keybindings
from pidrei_utils.cancel import CancelToken

from .agent_session_helpers import create_assistant_message
from .ui_timer_helpers import manual_ui_timers


def _held(lock) -> bool:
    """Whether `lock` is held: a probe thread cannot take it without waiting."""
    taken: list[bool] = []

    def probe() -> None:
        acquired = lock.acquire(blocking=False)
        if acquired:
            lock.release()
        taken.append(acquired)

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    return not taken[0]


class _Session:
    """The slice of an AgentSession the listener wiring touches: its `_emit`
    fan-out calls the listeners in order, synchronously, and lets their
    errors propagate (as pi's)."""

    def __init__(self, retry_attempt: int = 0) -> None:
        self.retry_attempt = retry_attempt
        self._listeners: list = []

    def subscribe(self, listener):
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def emit(self, event) -> None:
        for listener in list(self._listeners):
            listener(event)


def _subscribed_mode(session: _Session, **attributes):
    fake = SimpleNamespace(
        session=session,
        ui=SimpleNamespace(state_lock=threading.RLock(), request_render=lambda force=False: None),
        _unsubscribe=None,
        **attributes,
    )
    if "_handle_event" not in attributes:
        fake._handle_event = partial(InteractiveMode._handle_event, fake)
    InteractiveMode._subscribe_to_agent(fake)
    return fake


def test_message_end_is_applied_before_the_session_moves_on():
    """Regression (spec/ui-island.md, "Agent events"): the session emits `message_end`, then
    persists the message and resets its retry counter. The UI applies the
    event inside the emit, so an aborted retry reads the counter the event
    belongs to ("Aborted after 2 retry attempts"); applied later (posted to
    the UI owner), it read the reset counter and said "Operation aborted"."""
    session = _Session(retry_attempt=2)
    shown: list = []
    _subscribed_mode(
        session,
        _is_initialized=True,
        _footer=SimpleNamespace(invalidate=lambda: None),
        _streaming_component=SimpleNamespace(
            update_content=lambda message, _partial: shown.append(message.error_message)
        ),
        _streaming_message=None,
        _pending_tools={},
    )

    session.emit(SimpleNamespace(type="message_end", message=create_assistant_message("", stop_reason="aborted")))
    session.retry_attempt = 0  # what `_handle_agent_event` does right after the emit

    assert shown == ["Aborted after 2 retry attempts"]


def test_events_apply_inside_the_emit_in_order_under_the_ui_state_lock():
    session = _Session()
    applied: list = []
    fake = _subscribed_mode(session, _handle_event=lambda event: applied.append((event, _held(fake.ui.state_lock))))

    for index in range(3):
        session.emit(f"event-{index}")
        assert applied[-1] == (f"event-{index}", True)
    assert [event for event, _held_then in applied] == ["event-0", "event-1", "event-2"]


def test_events_from_a_replaced_session_are_dropped():
    session = _Session()
    applied: list = []
    fake = _subscribed_mode(session, _handle_event=applied.append)

    fake.session = _Session()
    session.emit("stale")

    assert applied == []


def test_an_apply_error_propagates_to_the_emitter():
    """pi's listener has no try/catch: the error reaches the emit's caller
    (for agent events, the dispatcher fails the run)."""
    session = _Session()

    def fail(_event) -> None:
        raise RuntimeError("apply failed")

    _subscribed_mode(session, _handle_event=fail)

    with pytest.raises(RuntimeError, match="apply failed"):
        session.emit("event")


def test_extension_state_registered_after_a_reset_survives_it():
    """`_reset_extension_ui` runs on session invalidation and /reload, one
    hold of the UI state lock: the previous extensions' state goes, and what
    the next extensions register afterwards (here: an autocomplete wrapper)
    stays."""
    installed: list = []
    base = SimpleNamespace(name="base")
    default_editor = SimpleNamespace(
        set_autocomplete_provider=installed.append,
        on_extension_shortcut=lambda data: False,
    )
    noop = lambda *args, **kwargs: None
    fake = SimpleNamespace(
        ui=SimpleNamespace(state_lock=threading.RLock(), hide_overlay=noop),
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
        _set_custom_editor_component=noop,
    )
    fake._setup_autocomplete_provider = lambda: InteractiveMode._setup_autocomplete_provider(fake)

    InteractiveMode._reset_extension_ui(fake)
    assert fake._autocomplete_provider_wrappers == ()
    InteractiveMode._create_extension_ui_context(fake).add_autocomplete_provider(
        lambda current: SimpleNamespace(wrapped=current)
    )

    assert len(fake._autocomplete_provider_wrappers) == 1
    assert installed[-1].wrapped is base


def _noop(*_args, **_kwargs) -> None:
    return None


def _editor_slot_mode():
    """Fake `this` for the extension dialogs and the editor-slot helpers."""
    fake = SimpleNamespace(
        ui=SimpleNamespace(state_lock=threading.RLock(), set_focus=_noop, request_render=_noop),
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
async def test_an_aborted_dialog_is_hidden_before_the_next_one_opens(monkeypatch):
    # The extension resumes as soon as dialog A settles and may open dialog B
    # right away: A's abort (from any task) hides A itself, so B is left
    # mounted and alive.
    await init_theme("dark")
    set_keybindings(KeybindingsManager.in_memory())
    monkeypatch.setattr(interactive_mode, "ExtensionSelectorComponent", _RecordingSelector)
    fake = _editor_slot_mode()

    abort = CancelToken()
    first_wait = InteractiveMode._show_extension_selector(fake, "first", ["Yes", "No"], {"signal": abort})
    first = fake._extension_selector
    abort.cancel()
    assert await first_wait is None

    close_second = CancelToken()
    second_wait = InteractiveMode._show_extension_selector(fake, "second", ["Yes", "No"], {"signal": close_second})

    second = fake._extension_selector
    assert second is not first
    assert first.disposed and not second.disposed
    assert fake._editor_container.children == [second]
    close_second.cancel()
    assert await second_wait is None


def test_a_share_restore_landing_after_another_component_took_the_slot_leaves_it():
    # Esc on the gist loader restores the editor while the share task may be
    # restoring too: the late restore must not evict whatever the user opened
    # in between.
    fake = _editor_slot_mode()
    loader = SimpleNamespace(dispose=_noop)
    fake._editor_container.add_child(loader)

    InteractiveMode._restore_share_editor(fake, loader)  # Esc
    selector = SimpleNamespace(name="selector")
    fake._editor_container.clear()
    fake._editor_container.add_child(selector)  # the user opens a selector
    InteractiveMode._restore_share_editor(fake, loader)  # the share task finishes

    assert fake._editor_container.children == [selector]


@pytest.mark.tonio
async def test_the_new_session_notice_lands_after_the_chat_reset():
    # `new_session()` rebinds the session, which resets the chat; the notice
    # must land after that reset, not be wiped by it.
    chat = Container()
    fake = SimpleNamespace(
        ui=SimpleNamespace(state_lock=threading.RLock(), request_render=_noop),
        _chat_container=chat,
    )
    fake._append_to_chat = lambda *components: InteractiveMode._append_to_chat(fake, *components)

    async def new_session() -> dict:
        chat.clear()  # the rebind's `render_current_session_state`
        return {"cancelled": False}

    fake.runtime_host = SimpleNamespace(new_session=new_session)
    await init_theme("dark")
    await InteractiveMode._new_session(fake)

    assert len(chat.children) == 2  # spacer + "New session started"


def test_tools_expansion_applies_the_flag_and_the_children_together():
    # `ctx.ui.set_tools_expanded` runs on the extension's task and
    # `get_tools_expanded` reads the flag synchronously right after.
    expanded_children: list = []
    fake = SimpleNamespace(
        _tool_output_expanded=False,
        _custom_header=None,
        _built_in_header=None,
        _loaded_resources_container=SimpleNamespace(children=[]),
        _chat_container=SimpleNamespace(children=[SimpleNamespace(set_expanded=expanded_children.append)]),
        ui=SimpleNamespace(state_lock=threading.RLock()),
        show_status=_noop,
    )

    InteractiveMode.set_tools_expanded(fake, True)

    assert fake._tool_output_expanded is True
    assert expanded_children == [True]


@pytest.mark.tonio
async def test_concurrent_bash_commands_keep_their_own_output():
    # pi keeps the running command's component in one field; two `!`
    # commands run concurrently here, and the second used to take the
    # first's output chunks.
    await init_theme("dark")
    set_keybindings(KeybindingsManager.in_memory())
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
        ui=SimpleNamespace(state_lock=threading.RLock(), request_render=_noop),
        session=session,
        session_manager=SimpleNamespace(get_cwd=lambda: "/tmp"),
        _chat_container=chat,
        _pending_messages_container=Container(),
        _pending_bash_components=[],
        _output_pad=1,
        show_error=_noop,
    )
    fake._mount_bash_component = lambda component, deferred: InteractiveMode._mount_bash_component(
        fake, component, deferred
    )

    with manual_ui_timers():
        async with tonio.scope() as scope:
            scope.spawn(InteractiveMode._handle_bash_command(fake, "first"))
            scope.spawn(InteractiveMode._handle_bash_command(fake, "second"))

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
    # The Alt+Enter action reads and clears the editor there and then: a
    # clear applied later would land behind the next keys and wipe them.
    prompted = tonio.Event()
    editor = _TextEditor("queued question")
    flows: list = []

    async def prompt(_text, _options=None) -> None:
        prompted.set()

    fake = SimpleNamespace(
        editor=editor,
        session=SimpleNamespace(is_compacting=False, is_streaming=True, prompt=prompt),
        ui=SimpleNamespace(state_lock=threading.RLock(), request_render=_noop),
        _apply_editor_history=_noop,
        _update_pending_messages_display=_noop,
        _spawn_flow=flows.append,
    )
    fake._set_editor_text = lambda text: InteractiveMode._set_editor_text(fake, text)
    fake._queue_follow_up = lambda text: InteractiveMode._queue_follow_up(fake, text)

    InteractiveMode._handle_follow_up(fake)
    editor.text += "next"  # typed right after Alt+Enter
    for flow in flows:
        await flow

    assert prompted.is_set()
    assert editor.text == "next"


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


class _HoldRecorder:
    """Stands in for `TUI.state_lock`: a reentrant lock that numbers each
    outermost hold, so a test can check which steps shared one."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._depth = 0
        self.holds = 0
        self.current: int | None = None

    def __enter__(self) -> None:
        self._lock.acquire()
        if self._depth == 0:
            self.holds += 1
            self.current = self.holds
        self._depth += 1

    def __exit__(self, *_exc) -> None:
        self._depth -= 1
        if self._depth == 0:
            self.current = None
        self._lock.release()


@pytest.mark.tonio
async def test_a_session_swap_and_the_rebinds_first_block_are_one_hold():
    """spec/ui-island.md, "Flows that await several times": pi's swap and the rebind's first synchronous
    block (runtime settings, the chat redraw, the subscription) never show
    apart; the block's I/O (the cwd, the trust warning's check) runs before
    the hold, for the new session."""
    lock = _HoldRecorder()
    steps: list = []
    old_session = SimpleNamespace(session_manager=SimpleNamespace(get_cwd=lambda: "/old"))
    new_session = SimpleNamespace(session_manager=SimpleNamespace(get_cwd=lambda: "/new"))

    async def resolve_cwd(cwd: str) -> dict:
        steps.append(("resolve", cwd, lock.current))
        return {"cwd": cwd}

    async def bind_current_session_extensions() -> None:
        steps.append(("bind", lock.current))

    async def needs_project_trust_warning(session) -> bool:
        steps.append(("trust", session.session_manager.get_cwd(), lock.current))
        return True

    fake = SimpleNamespace(
        session=old_session,
        ui=SimpleNamespace(state_lock=lock),
        _footer_data_provider=SimpleNamespace(resolve_cwd=resolve_cwd),
        _needs_project_trust_warning=needs_project_trust_warning,
        _unsubscribe=lambda: steps.append(("unsubscribe", lock.current)),
        _apply_runtime_settings=lambda resolved: steps.append(("settings", resolved["cwd"], lock.current)) or False,
        render_current_session_state=lambda trust_warning: steps.append(("render", trust_warning, lock.current)),
        _subscribe_to_agent=lambda: steps.append(("subscribe", lock.current)),
        _bind_current_session_extensions=bind_current_session_extensions,
        _update_available_provider_count=_noop,
        _update_editor_border_color=_noop,
        _update_terminal_title=_noop,
    )

    def swap() -> None:
        steps.append(("swap", lock.current))
        fake.session = new_session

    await InteractiveMode._rebind_current_session(fake, {"renderBeforeBind": True}, new_session, swap)

    hold = steps[2][-1]
    assert hold is not None
    assert steps[:7] == [
        ("resolve", "/new", None),
        ("trust", "/new", None),
        ("swap", hold),
        ("unsubscribe", hold),
        ("settings", "/new", hold),
        ("render", True, hold),
        ("subscribe", hold),
    ]
    assert steps[7] == ("bind", None)


def test_a_bash_command_submitted_while_another_starts_gets_the_busy_warning():
    """spec/ui-island.md, "The guards": "running" is set only once the first command's
    flow reaches the executor (after the extensions' hook); the claim taken
    with the check at submit closes that window, and pi's warning shows."""
    flows: list = []
    warnings: list = []
    editor_texts: list = []
    fake = SimpleNamespace(
        session=SimpleNamespace(is_bash_running=False),
        ui=SimpleNamespace(state_lock=threading.RLock()),
        _bash_claimed=False,
        show_warning=warnings.append,
        _set_editor_text=editor_texts.append,
        _apply_editor_history=_noop,
        _spawn_flow=flows.append,
        _run_editor_bash_command=lambda command, excluded: command,
    )

    InteractiveMode._handle_editor_submit(fake, "!first")
    InteractiveMode._handle_editor_submit(fake, "!second")

    assert flows == ["first"]
    assert warnings == ["A bash command is already running. Press Esc to cancel it first."]
    assert editor_texts == ["!second"]


@pytest.mark.tonio
async def test_the_bash_claim_is_released_when_the_command_ends_even_by_an_error():
    async def failing_bash_command(_command, _excluded) -> None:
        raise RuntimeError("executor failed")

    fake = SimpleNamespace(
        ui=SimpleNamespace(state_lock=threading.RLock()),
        _bash_claimed=True,
        _is_bash_mode=True,
        _handle_bash_command=failing_bash_command,
        _update_editor_border_color=_noop,
    )

    with pytest.raises(RuntimeError, match="executor failed"):
        await InteractiveMode._run_editor_bash_command(fake, "ls", False)

    assert fake._bash_claimed is False
