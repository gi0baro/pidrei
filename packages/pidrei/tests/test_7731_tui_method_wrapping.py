"""Mirror of pi's suite/regressions/7731-tui-method-wrapping.test.ts."""

from types import SimpleNamespace

from pidrei.modes.interactive.interactive_mode import create_interactive_tui_reference


def test_calls_the_method_captured_before_a_replacement():
    renderer = SimpleNamespace(render=lambda width: [f"width: {width}"])
    tui = create_interactive_tui_reference(lambda: renderer)
    original_render = tui.render
    tui.render = lambda width: original_render(width)

    assert tui.render(80) == ["width: 80"]


def test_routes_a_captured_method_to_a_replacement_renderer():
    calls: list[str] = []
    state = {"renderer": SimpleNamespace(request_render=lambda: calls.append("regular"))}
    tui = create_interactive_tui_reference(lambda: state["renderer"])
    request_render = tui.request_render

    request_render()
    state["renderer"] = SimpleNamespace(request_render=lambda: calls.append("fullscreen"))
    request_render()

    assert calls == ["regular", "fullscreen"]
