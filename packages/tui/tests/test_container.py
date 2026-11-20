"""Mirror of pi tui test/container.test.ts."""

from pidrei_tui.components.box import Box
from pidrei_tui.components.spacer import Spacer
from pidrei_tui.components.text import Text
from pidrei_tui.tui import Container, TuiMouseEvent, TuiMouseEventResult


def wheel_at(y: int, width: int, height: int) -> TuiMouseEvent:
    return TuiMouseEvent(
        type="wheel",
        button="none",
        x=0,
        y=y,
        screen_x=0,
        screen_y=y,
        width=width,
        height=height,
        shift=False,
        alt=False,
        ctrl=False,
        wheel_delta=1,
    )


# #9549: copying every transcript line on every frame made long fullscreen transcripts expensive.


def test_returns_the_same_lines_while_children_are_unchanged():
    text = Text("hello", 0, 0)
    inner = Container()
    inner.add_child(text)
    inner.add_child(Spacer(1))
    outer = Container()
    outer.add_child(inner)

    first = outer.render(20)
    assert outer.render(20) is first

    text.set_text("changed")
    changed = outer.render(20)
    assert changed is not first
    assert [line.rstrip() for line in changed] == ["changed", ""]
    assert [line.rstrip() for line in first] == ["hello", ""]


def test_re_renders_when_children_are_added_or_the_width_changes():
    container = Container()
    container.add_child(Text("a", 0, 0))
    first = container.render(20)
    container.add_child(Text("b", 0, 0))
    assert [line.rstrip() for line in container.render(20)] == ["a", "b"]
    assert container.render(10) is not first


def test_reuses_box_output_while_its_children_are_unchanged():
    text = Text("boxed", 0, 0)
    box = Box(1, 0)
    box.add_child(text)
    first = box.render(20)
    assert box.render(20) is first
    text.set_text("next")
    assert box.render(20) is not first


def test_dispatches_mouse_events_by_the_rendered_child_heights():
    hits: list[str] = []

    class Target:
        def __init__(self, name: str, lines: int) -> None:
            self._name = name
            self._lines = lines

        def render(self, width: int) -> list[str]:
            return [self._name for _ in range(self._lines)]

        def invalidate(self) -> None:
            pass

        def handle_mouse(self, event: TuiMouseEvent) -> TuiMouseEventResult:
            hits.append(self._name)
            return TuiMouseEventResult(handled=True)

    container = Container()
    container.add_child(Target("first", 2))
    container.add_child(Target("second", 1))
    container.render(20)
    container.handle_mouse(wheel_at(2, 20, 3))
    container.handle_mouse(wheel_at(1, 20, 3))
    assert hits == ["second", "first"]
