"""Mirror of pi tui test/text.test.ts."""

from pidrei_tui.components.text import Text


def test_keeps_its_wrapped_lines_when_the_same_text_is_set_again():
    text = Text("hello world", 0, 0)
    lines = text.render(5)
    text.set_text("hello world")
    assert text.render(5) is lines
