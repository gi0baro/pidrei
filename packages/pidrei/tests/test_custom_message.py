"""Mirror of pi coding-agent test/custom-message.test.ts."""

import time

import pytest

from pidrei.core.messages import CustomMessage
from pidrei.modes.interactive.components.custom_message import CustomMessageComponent
from pidrei.modes.interactive.theme import init_theme_sync
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import Text


@pytest.fixture(autouse=True)
def _setup():
    init_theme_sync("dark")


def test_provides_output_padding_to_custom_renderers_and_updates_it():
    options_seen = []

    def renderer(_message, options, _theme):
        options_seen.append(options)
        return Text("custom", options["outputPad"], 0)

    message = CustomMessage(
        custom_type="test",
        content="custom",
        display=True,
        timestamp=int(time.time() * 1000),
    )
    component = CustomMessageComponent(message, renderer, None, 1)

    assert options_seen == [{"expanded": False, "outputPad": 1}]
    assert any(line.startswith(" custom") for line in map(strip_ansi, component.render(40)))

    component.set_output_pad(0)

    assert options_seen[-1] == {"expanded": False, "outputPad": 0}
    assert any(line.startswith("custom") for line in map(strip_ansi, component.render(40)))
