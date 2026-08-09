"""Mirror of pi tui test/overlay-non-capturing.test.ts."""

import pytest

from pidrei_tui.tui import Container
from pidrei_tui.tui_main_screen import TuiMainScreen

from .virtual_terminal import VirtualTerminal


class StaticOverlay:
    def __init__(self, lines):
        self.lines = lines

    def render(self, width):
        return self.lines

    def invalidate(self):
        pass


class EmptyContent:
    def render(self, width):
        return []

    def invalidate(self):
        pass


class FocusableOverlay:
    def __init__(self, lines):
        self.focused = False
        self.inputs = []
        self.lines = lines
        self._on_input = None

    def set_input_handler(self, handler):
        """Override the input behavior (pi tests reassign handleInput)."""
        self._on_input = handler

    async def handle_input(self, data):
        if self._on_input is not None:
            self._on_input(data)
        else:
            self.inputs.append(data)

    def render(self, width):
        return self.lines

    def invalidate(self):
        pass


async def render_and_flush(tui, terminal):
    """Render and wait until the frame on screen reflects the current state.

    Two round-trips, not one. Waiting for a single frame is not enough: if the
    render loop is already inside `_do_render()` when the test mutates focus or
    overlay order, it writes a *stale* frame, the counter advances and the wait
    returns early — the assertion then reads the previous frame. The second
    request cannot start until the first has completed, so the frame it
    produces is guaranteed to have begun after the mutation.

    This replaced `await tonio.sleep(0.05)`, which waited for nothing at all
    and was the cause of a long-standing load-dependent flake across the focus
    and overlay suites.
    """
    for _ in range(2):
        before = terminal.frames
        tui.request_render(True)
        await terminal.wait_for_render(before)


def _make(columns=80, rows=24):
    terminal = VirtualTerminal(columns, rows)
    return terminal, TuiMainScreen(terminal)


# focus management


@pytest.mark.tonio
async def test_non_capturing_overlay_preserves_focus_on_creation():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    overlay = FocusableOverlay(["OVERLAY"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        tui.show_overlay(overlay, {"nonCapturing": True})
        await render_and_flush(tui, terminal)
        assert editor.focused is True
        assert overlay.focused is False
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_focus_transfers_focus_to_the_overlay():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    overlay = FocusableOverlay(["OVERLAY"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        handle = tui.show_overlay(overlay, {"nonCapturing": True})
        handle.focus()
        await render_and_flush(tui, terminal)
        assert editor.focused is False
        assert overlay.focused is True
        assert handle.is_focused() is True
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_unfocus_restores_previous_focus():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    overlay = FocusableOverlay(["OVERLAY"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        handle = tui.show_overlay(overlay, {"nonCapturing": True})
        handle.focus()
        handle.unfocus()
        await render_and_flush(tui, terminal)
        assert editor.focused is True
        assert overlay.focused is False
        assert handle.is_focused() is False
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_set_hidden_false_on_non_capturing_overlay_does_not_auto_focus():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    overlay = FocusableOverlay(["OVERLAY"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        handle = tui.show_overlay(overlay, {"nonCapturing": True})
        handle.set_hidden(True)
        handle.set_hidden(False)
        await render_and_flush(tui, terminal)
        assert editor.focused is True
        assert overlay.focused is False
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_hide_when_overlay_is_not_focused_does_not_change_focus():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    overlay = FocusableOverlay(["OVERLAY"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        handle = tui.show_overlay(overlay, {"nonCapturing": True})
        handle.hide()
        await render_and_flush(tui, terminal)
        assert editor.focused is True
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_hide_when_focused_restores_focus_correctly():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    overlay = FocusableOverlay(["OVERLAY"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        handle = tui.show_overlay(overlay, {"nonCapturing": True})
        handle.focus()
        handle.hide()
        await render_and_flush(tui, terminal)
        assert editor.focused is True
        assert overlay.focused is False
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_capturing_overlay_removed_with_non_capturing_below_restores_focus_to_editor():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    non_capturing = FocusableOverlay(["NC"])
    capturing = FocusableOverlay(["CAP"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        tui.show_overlay(non_capturing, {"nonCapturing": True})
        handle = tui.show_overlay(capturing)
        assert capturing.focused is True
        handle.hide()
        await render_and_flush(tui, terminal)
        assert editor.focused is True
        assert non_capturing.focused is False
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_sub_overlay_cleanup_then_hide_overlay_restores_focus_and_input_to_editor():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    timer = FocusableOverlay(["TIMER"])
    controller = FocusableOverlay(["CTRL"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        timer_handle = tui.show_overlay(timer, {"nonCapturing": True})
        tui.show_overlay(controller)
        assert controller.focused is True
        assert editor.focused is False
        timer_handle.hide()
        tui.hide_overlay()
        await render_and_flush(tui, terminal)
        assert editor.focused is True
        assert controller.focused is False
        assert timer.focused is False
        await terminal.send_input("x")
        await render_and_flush(tui, terminal)
        assert editor.inputs == ["x"]
        assert controller.inputs == []
        assert timer.inputs == []
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_removed_focused_child_overlay_does_not_become_parent_overlay_fallback():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    child = FocusableOverlay(["CHILD"])
    parent = FocusableOverlay(["PARENT"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        child_handle = tui.show_overlay(child, {"nonCapturing": True})
        child_handle.focus()
        parent_handle = tui.show_overlay(parent)
        assert parent.focused is True

        child_handle.hide()
        parent_handle.hide()
        await terminal.send_input("x")
        await render_and_flush(tui, terminal)

        assert editor.inputs == ["x"]
        assert child.inputs == []
        assert parent.inputs == []
        assert editor.focused is True
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_deferred_sub_overlay_pattern_restores_focus():
    # pi defers the controller overlay through a microtask (.then); a tonio
    # spawned task plays the same role here.
    import tonio.colored as tonio

    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    timer = FocusableOverlay(["TIMER"])
    controller = FocusableOverlay(["CTRL"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        timer_handle = tui.show_overlay(timer, {"nonCapturing": True})

        async def push_controller():
            tui.show_overlay(controller)

        await tonio.spawn(push_controller())
        await render_and_flush(tui, terminal)

        assert controller.focused is True
        assert editor.focused is False

        # Simulate Esc: cleanup + close
        timer_handle.hide()
        tui.hide_overlay()
        await render_and_flush(tui, terminal)

        assert editor.focused is True, "editor should regain focus"
        assert controller.focused is False
        assert timer.focused is False

        await terminal.send_input("x")
        await render_and_flush(tui, terminal)
        assert editor.inputs == ["x"], "editor should receive input after close"
        assert controller.inputs == []
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_handle_input_redirection_skips_non_capturing_overlays_when_focused_overlay_becomes_invisible():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    fallback_capturing = FocusableOverlay(["FALLBACK"])
    non_capturing = FocusableOverlay(["NC"])
    primary = FocusableOverlay(["PRIMARY"])
    state = {"visible": True}
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        tui.show_overlay(fallback_capturing)
        tui.show_overlay(non_capturing, {"nonCapturing": True})
        tui.show_overlay(primary, {"visible": lambda w, h: state["visible"]})
        assert primary.focused is True
        state["visible"] = False
        await terminal.send_input("x")
        await render_and_flush(tui, terminal)
        assert primary.inputs == []
        assert non_capturing.inputs == []
        assert fallback_capturing.inputs == ["x"]
        assert fallback_capturing.focused is True
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_active_base_focus_replacement_receives_close_input_before_overlay_restore():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    replacement = FocusableOverlay(["REPLACEMENT"])
    overlay = FocusableOverlay(["OVERLAY"])

    def overlay_input(data):
        overlay.inputs.append(data)
        if data == "b":
            tui.set_focus(replacement)

    def replacement_input(data):
        replacement.inputs.append(data)
        if data == "\r":
            tui.set_focus(editor)

    overlay.set_input_handler(overlay_input)
    replacement.set_input_handler(replacement_input)
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        tui.show_overlay(overlay)
        assert overlay.focused is True
        await terminal.send_input("b")
        await render_and_flush(tui, terminal)
        assert replacement.focused is True

        await terminal.send_input("\r")
        await render_and_flush(tui, terminal)
        assert replacement.inputs == ["\r"]
        assert overlay.inputs == ["b"]
        assert overlay.focused is True

        await terminal.send_input("x")
        await render_and_flush(tui, terminal)
        assert overlay.inputs == ["b", "x"]
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_active_replacement_still_receives_input_when_it_is_another_overlay_pre_focus():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    replacement = FocusableOverlay(["REPLACEMENT"])
    passive = FocusableOverlay(["PASSIVE"])
    overlay = FocusableOverlay(["OVERLAY"])

    def overlay_input(data):
        overlay.inputs.append(data)
        if data == "b":
            tui.set_focus(replacement)

    def replacement_input(data):
        replacement.inputs.append(data)
        if data == "\r":
            tui.set_focus(editor)

    overlay.set_input_handler(overlay_input)
    replacement.set_input_handler(replacement_input)
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        tui.set_focus(replacement)
        tui.show_overlay(passive, {"nonCapturing": True})
        tui.set_focus(editor)
        tui.show_overlay(overlay)
        await terminal.send_input("b")
        await render_and_flush(tui, terminal)
        assert replacement.focused is True

        await terminal.send_input("1")
        await terminal.send_input("\r")
        await render_and_flush(tui, terminal)
        assert replacement.inputs == ["1", "\r"]
        assert overlay.inputs == ["b"]
        assert overlay.focused is True
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_blocked_replacement_can_move_focus_internally_before_overlay_restore():
    terminal, tui = _make()
    base = Container()
    editor = FocusableOverlay(["EDITOR"])
    first_replacement = FocusableOverlay(["FIRST"])
    second_replacement = FocusableOverlay(["SECOND"])
    overlay = FocusableOverlay(["OVERLAY"])

    def overlay_input(data):
        overlay.inputs.append(data)
        if data == "b":
            tui.set_focus(first_replacement)

    def first_input(data):
        first_replacement.inputs.append(data)
        if data == "n":
            tui.set_focus(second_replacement)

    def second_input(data):
        second_replacement.inputs.append(data)
        if data == "\r":
            base.clear()
            base.add_child(editor)
            tui.set_focus(editor)

    overlay.set_input_handler(overlay_input)
    first_replacement.set_input_handler(first_input)
    second_replacement.set_input_handler(second_input)
    base.add_child(editor)
    base.add_child(first_replacement)
    base.add_child(second_replacement)
    tui.add_child(base)
    tui.set_focus(editor)
    await tui.start()
    try:
        tui.show_overlay(overlay)
        await terminal.send_input("b")
        await render_and_flush(tui, terminal)
        await terminal.send_input("n")
        await render_and_flush(tui, terminal)
        await terminal.send_input("2")
        await terminal.send_input("\r")
        await render_and_flush(tui, terminal)

        assert overlay.inputs == ["b"]
        assert first_replacement.inputs == ["n"]
        assert second_replacement.inputs == ["2", "\r"]
        assert overlay.focused is True
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_removed_replacement_restores_overlay_even_when_overlay_pre_focus_differs_from_next_focus():
    terminal, tui = _make()
    base = Container()
    editor = FocusableOverlay(["EDITOR"])
    palette = FocusableOverlay(["PALETTE"])
    replacement = FocusableOverlay(["REPLACEMENT"])
    overlay = FocusableOverlay(["OVERLAY"])

    def overlay_input(data):
        overlay.inputs.append(data)
        if data == "b":
            tui.set_focus(replacement)

    def replacement_input(data):
        replacement.inputs.append(data)
        if data == "\r":
            base.clear()
            base.add_child(editor)
            tui.set_focus(editor)

    overlay.set_input_handler(overlay_input)
    replacement.set_input_handler(replacement_input)
    base.add_child(editor)
    base.add_child(palette)
    base.add_child(replacement)
    tui.add_child(base)
    tui.set_focus(palette)
    await tui.start()
    try:
        tui.show_overlay(overlay)
        await terminal.send_input("b")
        await render_and_flush(tui, terminal)
        await terminal.send_input("\r")
        await terminal.send_input("x")
        await render_and_flush(tui, terminal)

        assert overlay.inputs == ["b", "x"]
        assert replacement.inputs == ["\r"]
        assert editor.inputs == []
        assert overlay.focused is True
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_unfocus_target_releases_a_blocked_overlay_while_replacement_remains_focused():
    terminal, tui = _make()
    fallback = FocusableOverlay(["FALLBACK"])
    target = FocusableOverlay(["TARGET"])
    replacement = FocusableOverlay(["REPLACEMENT"])
    overlay = FocusableOverlay(["OVERLAY"])

    def replacement_input(data):
        replacement.inputs.append(data)
        if data == "\r":
            tui.set_focus(fallback)

    replacement.set_input_handler(replacement_input)
    tui.add_child(EmptyContent())
    await tui.start()
    try:
        overlay_handle = tui.show_overlay(overlay)

        def overlay_input(data):
            overlay.inputs.append(data)
            if data == "b":
                tui.set_focus(replacement)
                overlay_handle.unfocus({"target": target})

        overlay.set_input_handler(overlay_input)

        await terminal.send_input("b")
        await render_and_flush(tui, terminal)
        assert replacement.focused is True
        await terminal.send_input("\r")
        await terminal.send_input("x")
        await render_and_flush(tui, terminal)

        assert overlay.inputs == ["b"]
        assert replacement.inputs == ["\r"]
        assert fallback.inputs == []
        assert target.inputs == ["x"]
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_handle_input_restores_focus_to_a_visible_focused_overlay_after_base_focus_steal():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    replacement = FocusableOverlay(["REPLACEMENT"])
    overlay = FocusableOverlay(["OVERLAY"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        tui.show_overlay(overlay)
        assert overlay.focused is True
        tui.set_focus(replacement)
        tui.set_focus(editor)
        await terminal.send_input("x")
        await render_and_flush(tui, terminal)
        assert overlay.inputs == ["x"]
        assert editor.inputs == []
        assert overlay.focused is True
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_handle_input_restores_focus_to_explicitly_focused_raw_sub_overlay_after_base_focus_steal():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    controller = FocusableOverlay(["CONTROLLER"])
    sub_overlay = FocusableOverlay(["SUB"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        tui.show_overlay(controller)
        sub_handle = tui.show_overlay(sub_overlay, {"nonCapturing": True})
        sub_handle.focus()
        tui.set_focus(editor)
        await terminal.send_input("x")
        await render_and_flush(tui, terminal)
        assert sub_overlay.inputs == ["x"]
        assert controller.inputs == []
        assert editor.inputs == []
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_passive_non_capturing_overlay_does_not_regain_input_after_base_focus():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    passive = FocusableOverlay(["PASSIVE"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        tui.show_overlay(passive, {"nonCapturing": True})
        await terminal.send_input("x")
        await render_and_flush(tui, terminal)
        assert editor.inputs == ["x"]
        assert passive.inputs == []
        assert editor.focused is True
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_explicitly_focused_non_capturing_overlay_regains_input_after_base_focus_steal():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    overlay = FocusableOverlay(["NC"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        handle = tui.show_overlay(overlay, {"nonCapturing": True})
        handle.focus()
        tui.set_focus(editor)
        await terminal.send_input("x")
        await render_and_flush(tui, terminal)
        assert overlay.inputs == ["x"]
        assert editor.inputs == []
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_unfocus_prevents_visible_overlay_from_regaining_input():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    overlay = FocusableOverlay(["OVERLAY"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        handle = tui.show_overlay(overlay)
        handle.unfocus()
        await terminal.send_input("x")
        await render_and_flush(tui, terminal)
        assert editor.inputs == ["x"]
        assert overlay.inputs == []
        assert editor.focused is True
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_set_focus_none_explicitly_clears_visible_overlay_restore():
    terminal, tui = _make()
    overlay = FocusableOverlay(["OVERLAY"])
    tui.add_child(EmptyContent())
    await tui.start()
    try:
        tui.show_overlay(overlay)
        tui.set_focus(None)
        await terminal.send_input("x")
        await render_and_flush(tui, terminal)
        assert overlay.inputs == []
        assert overlay.focused is False
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_blocked_replacement_set_focus_none_resumes_the_visible_overlay():
    terminal, tui = _make()
    replacement = FocusableOverlay(["REPLACEMENT"])
    overlay = FocusableOverlay(["OVERLAY"])

    def replacement_input(data):
        replacement.inputs.append(data)
        if data == "\r":
            tui.set_focus(None)

    def overlay_input(data):
        overlay.inputs.append(data)
        if data == "b":
            tui.set_focus(replacement)

    replacement.set_input_handler(replacement_input)
    overlay.set_input_handler(overlay_input)
    tui.add_child(EmptyContent())
    await tui.start()
    try:
        tui.show_overlay(overlay)
        await terminal.send_input("b")
        await render_and_flush(tui, terminal)
        await terminal.send_input("\r")
        await terminal.send_input("x")
        await render_and_flush(tui, terminal)
        assert replacement.inputs == ["\r"]
        assert overlay.inputs == ["b", "x"]
        assert overlay.focused is True
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_temporarily_invisible_focused_overlay_falls_back_without_losing_restore_eligibility():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    overlay = FocusableOverlay(["OVERLAY"])
    state = {"visible": True}
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        tui.show_overlay(overlay, {"visible": lambda w, h: state["visible"]})
        tui.set_focus(editor)
        state["visible"] = False
        await terminal.send_input("x")
        await render_and_flush(tui, terminal)
        assert editor.inputs == ["x"]
        assert overlay.inputs == []
        state["visible"] = True
        await terminal.send_input("y")
        await render_and_flush(tui, terminal)
        assert editor.inputs == ["x"]
        assert overlay.inputs == ["y"]
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_temporarily_invisible_focused_overlay_with_none_pre_focus_restores_when_visible_again():
    terminal, tui = _make()
    overlay = FocusableOverlay(["OVERLAY"])
    state = {"visible": True}
    tui.add_child(EmptyContent())
    await tui.start()
    try:
        tui.show_overlay(overlay, {"visible": lambda w, h: state["visible"]})
        state["visible"] = False
        await terminal.send_input("x")
        await render_and_flush(tui, terminal)
        assert overlay.inputs == []
        state["visible"] = True
        await terminal.send_input("y")
        await render_and_flush(tui, terminal)
        assert overlay.inputs == ["y"]
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_cyclic_overlay_pre_focus_ancestry_does_not_hang_focus_changes():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    overlay = FocusableOverlay(["OVERLAY"])
    tui.add_child(EmptyContent())
    tui.set_focus(overlay)
    await tui.start()
    try:
        handle = tui.show_overlay(overlay, {"nonCapturing": True})
        handle.focus()
        tui.set_focus(editor)
        await terminal.send_input("x")
        await render_and_flush(tui, terminal)
        assert editor.inputs == ["x"]
        assert overlay.inputs == []
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_handle_input_restores_the_focus_order_top_overlay_after_base_focus_steal():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    lower = FocusableOverlay(["LOWER"])
    upper = FocusableOverlay(["UPPER"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        lower_handle = tui.show_overlay(lower)
        tui.show_overlay(upper)
        lower_handle.focus()
        tui.set_focus(editor)
        await terminal.send_input("x")
        await render_and_flush(tui, terminal)
        assert lower.inputs == ["x"]
        assert upper.inputs == []
        assert editor.inputs == []
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_hide_overlay_does_not_reassign_focus_when_topmost_overlay_is_non_capturing():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    capturing = FocusableOverlay(["CAP"])
    non_capturing = FocusableOverlay(["NC"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        tui.show_overlay(capturing)
        tui.show_overlay(non_capturing, {"nonCapturing": True})
        assert capturing.focused is True
        tui.hide_overlay()
        await render_and_flush(tui, terminal)
        assert capturing.focused is True
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_multiple_capturing_and_non_capturing_overlays_restore_focus_through_removals():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    c1 = FocusableOverlay(["C1"])
    n1 = FocusableOverlay(["N1"])
    c2 = FocusableOverlay(["C2"])
    n2 = FocusableOverlay(["N2"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        c1_handle = tui.show_overlay(c1)
        tui.show_overlay(n1, {"nonCapturing": True})
        c2_handle = tui.show_overlay(c2)
        tui.show_overlay(n2, {"nonCapturing": True})
        assert c2.focused is True
        c2_handle.hide()
        await render_and_flush(tui, terminal)
        assert c1.focused is True
        c1_handle.hide()
        await render_and_flush(tui, terminal)
        assert editor.focused is True
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_capturing_overlay_unfocus_on_topmost_capturing_overlay_falls_back_to_pre_focus():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    capturing = FocusableOverlay(["CAP"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        handle = tui.show_overlay(capturing)
        assert capturing.focused is True
        handle.unfocus()
        await render_and_flush(tui, terminal)
        assert editor.focused is True
        assert capturing.focused is False
    finally:
        await tui.stop()


# no-op guards


@pytest.mark.tonio
async def test_focus_on_hidden_overlay_is_a_no_op():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    overlay = FocusableOverlay(["OVERLAY"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        handle = tui.show_overlay(overlay, {"nonCapturing": True})
        handle.set_hidden(True)
        handle.focus()
        await render_and_flush(tui, terminal)
        assert editor.focused is True
        assert handle.is_focused() is False
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_focus_after_hide_is_a_no_op():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    overlay = FocusableOverlay(["OVERLAY"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        handle = tui.show_overlay(overlay, {"nonCapturing": True})
        handle.hide()
        handle.focus()
        await render_and_flush(tui, terminal)
        assert editor.focused is True
        assert handle.is_focused() is False
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_unfocus_when_overlay_does_not_have_focus_is_a_no_op():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    overlay = FocusableOverlay(["OVERLAY"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        handle = tui.show_overlay(overlay, {"nonCapturing": True})
        handle.unfocus()
        await render_and_flush(tui, terminal)
        assert editor.focused is True
        assert overlay.focused is False
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_unfocus_with_none_pre_focus_clears_focus_and_does_not_route_input_back_to_overlay():
    terminal, tui = _make()
    overlay = FocusableOverlay(["OVERLAY"])
    tui.add_child(EmptyContent())
    await tui.start()
    try:
        handle = tui.show_overlay(overlay)
        assert overlay.focused is True
        handle.unfocus()
        assert overlay.focused is False
        await terminal.send_input("x")
        await render_and_flush(tui, terminal)
        assert overlay.inputs == []
        assert handle.is_focused() is False
    finally:
        await tui.stop()


# focus cycle prevention


@pytest.mark.tonio
async def test_toggle_focus_between_non_capturing_overlays_then_unfocus_returns_to_editor():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    a = FocusableOverlay(["A"])
    b = FocusableOverlay(["B"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        a_handle = tui.show_overlay(a, {"nonCapturing": True})
        b_handle = tui.show_overlay(b, {"nonCapturing": True})
        a_handle.focus()
        b_handle.focus()
        a_handle.focus()
        a_handle.unfocus()
        await render_and_flush(tui, terminal)
        assert editor.focused is True
        assert a.focused is False
        assert b.focused is False
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_explicit_unfocus_target_supports_cycling_between_three_overlays_and_editor():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    a = FocusableOverlay(["A"])
    b = FocusableOverlay(["B"])
    c = FocusableOverlay(["C"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        a_handle = tui.show_overlay(a)
        b_handle = tui.show_overlay(b)
        c_handle = tui.show_overlay(c)

        a_handle.focus()
        await terminal.send_input("a")
        await render_and_flush(tui, terminal)
        b_handle.focus()
        await terminal.send_input("b")
        await render_and_flush(tui, terminal)
        c_handle.focus()
        await terminal.send_input("c")
        await render_and_flush(tui, terminal)
        c_handle.unfocus({"target": editor})
        await terminal.send_input("e")
        await render_and_flush(tui, terminal)
        a_handle.focus()
        await terminal.send_input("A")
        await render_and_flush(tui, terminal)
        a_handle.unfocus({"target": editor})
        await terminal.send_input("E")
        await render_and_flush(tui, terminal)

        assert a.inputs == ["a", "A"]
        assert b.inputs == ["b"]
        assert c.inputs == ["c"]
        assert editor.inputs == ["e", "E"]
        assert editor.focused is True
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_explicit_none_unfocus_target_clears_focus_without_restoring_overlays():
    terminal, tui = _make()
    overlay = FocusableOverlay(["OVERLAY"])
    tui.add_child(EmptyContent())
    await tui.start()
    try:
        handle = tui.show_overlay(overlay)
        handle.unfocus({"target": None})
        await terminal.send_input("x")
        await render_and_flush(tui, terminal)
        assert overlay.inputs == []
        assert handle.is_focused() is False
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_hiding_focused_overlay_falls_back_to_next_visual_frontmost_overlay():
    terminal, tui = _make()
    editor = FocusableOverlay(["EDITOR"])
    a = FocusableOverlay(["A"])
    b = FocusableOverlay(["B"])
    c = FocusableOverlay(["C"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        a_handle = tui.show_overlay(a)
        b_handle = tui.show_overlay(b)
        tui.show_overlay(c)
        a_handle.focus()
        b_handle.focus()
        b_handle.set_hidden(True)
        await terminal.send_input("x")
        await render_and_flush(tui, terminal)
        assert a.inputs == ["x"]
        assert c.inputs == []
        assert a.focused is True
    finally:
        await tui.stop()


# rendering order


@pytest.mark.tonio
async def test_focus_on_already_focused_overlay_bumps_visual_order():
    terminal, tui = _make(20, 6)
    editor = FocusableOverlay(["EDITOR"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        a_handle = tui.show_overlay(StaticOverlay(["A"]), {"row": 0, "col": 0, "width": 1, "nonCapturing": True})
        tui.show_overlay(StaticOverlay(["B"]), {"row": 0, "col": 0, "width": 1, "nonCapturing": True})
        a_handle.focus()
        tui.show_overlay(StaticOverlay(["C"]), {"row": 0, "col": 0, "width": 1, "nonCapturing": True})
        await render_and_flush(tui, terminal)
        assert terminal.get_viewport()[0][:1] == "C"
        a_handle.focus()
        await render_and_flush(tui, terminal)
        assert terminal.get_viewport()[0][:1] == "A"
        assert a_handle.is_focused() is True
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_default_rendering_order_for_overlapping_overlays_follows_creation_order():
    terminal, tui = _make(20, 6)
    tui.add_child(EmptyContent())
    await tui.start()
    try:
        tui.show_overlay(StaticOverlay(["A"]), {"row": 0, "col": 0, "width": 1, "nonCapturing": True})
        tui.show_overlay(StaticOverlay(["B"]), {"row": 0, "col": 0, "width": 1, "nonCapturing": True})
        await render_and_flush(tui, terminal)
        assert terminal.get_viewport()[0][:1] == "B"
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_focus_on_lower_overlay_renders_it_on_top():
    terminal, tui = _make(20, 6)
    tui.add_child(EmptyContent())
    await tui.start()
    try:
        lower = tui.show_overlay(StaticOverlay(["A"]), {"row": 0, "col": 0, "width": 1, "nonCapturing": True})
        tui.show_overlay(StaticOverlay(["B"]), {"row": 0, "col": 0, "width": 1, "nonCapturing": True})
        await render_and_flush(tui, terminal)
        assert terminal.get_viewport()[0][:1] == "B"
        lower.focus()
        await render_and_flush(tui, terminal)
        assert terminal.get_viewport()[0][:1] == "A"
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_focusing_middle_overlay_places_it_on_top_while_preserving_others_relative_order():
    terminal, tui = _make(20, 6)
    tui.add_child(EmptyContent())
    await tui.start()
    try:
        tui.show_overlay(StaticOverlay(["A"]), {"row": 0, "col": 0, "width": 1, "nonCapturing": True})
        middle = tui.show_overlay(StaticOverlay(["B"]), {"row": 0, "col": 0, "width": 1, "nonCapturing": True})
        top = tui.show_overlay(StaticOverlay(["C"]), {"row": 0, "col": 0, "width": 1, "nonCapturing": True})
        await render_and_flush(tui, terminal)
        assert terminal.get_viewport()[0][:1] == "C"
        middle.focus()
        await render_and_flush(tui, terminal)
        assert terminal.get_viewport()[0][:1] == "B"
        middle.hide()
        await render_and_flush(tui, terminal)
        assert terminal.get_viewport()[0][:1] == "C"
        top.hide()
        await render_and_flush(tui, terminal)
        assert terminal.get_viewport()[0][:1] == "A"
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_capturing_overlay_hidden_and_shown_again_renders_on_top_after_unhide():
    terminal, tui = _make(20, 6)
    tui.add_child(EmptyContent())
    await tui.start()
    try:
        tui.show_overlay(StaticOverlay(["A"]), {"row": 0, "col": 0, "width": 1, "nonCapturing": True})
        capturing = tui.show_overlay(StaticOverlay(["B"]), {"row": 0, "col": 0, "width": 1})
        await render_and_flush(tui, terminal)
        assert terminal.get_viewport()[0][:1] == "B"
        capturing.set_hidden(True)
        tui.show_overlay(StaticOverlay(["C"]), {"row": 0, "col": 0, "width": 1, "nonCapturing": True})
        await render_and_flush(tui, terminal)
        assert terminal.get_viewport()[0][:1] == "C"
        capturing.set_hidden(False)
        await render_and_flush(tui, terminal)
        assert terminal.get_viewport()[0][:1] == "B"
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_unfocus_does_not_change_visual_order_until_another_overlay_is_focused():
    terminal, tui = _make(20, 6)
    editor = FocusableOverlay(["EDITOR"])
    tui.add_child(EmptyContent())
    tui.set_focus(editor)
    await tui.start()
    try:
        a = tui.show_overlay(StaticOverlay(["A"]), {"row": 0, "col": 0, "width": 1, "nonCapturing": True})
        b = tui.show_overlay(StaticOverlay(["B"]), {"row": 0, "col": 0, "width": 1, "nonCapturing": True})
        await render_and_flush(tui, terminal)
        assert terminal.get_viewport()[0][:1] == "B"
        a.focus()
        await render_and_flush(tui, terminal)
        assert terminal.get_viewport()[0][:1] == "A"
        a.unfocus()
        await render_and_flush(tui, terminal)
        assert terminal.get_viewport()[0][:1] == "A"
        b.focus()
        await render_and_flush(tui, terminal)
        assert terminal.get_viewport()[0][:1] == "B"
    finally:
        await tui.stop()
