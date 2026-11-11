"""Mirror of pi coding-agent test/custom-message.test.ts."""

import time

import pytest

from pidrei.core.messages import CustomMessage
from pidrei.modes.interactive.components.custom_message import CustomMessageComponent
from pidrei.modes.interactive.theme import init_theme
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import Text


@pytest.fixture(autouse=True)
async def _setup():
    await init_theme("dark")


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

    # pi's toMatchObject: the options also carry the host's keybinding hints.
    assert len(options_seen) == 1
    assert {"expanded": False, "outputPad": 1}.items() <= options_seen[0].items()
    assert any(line.startswith(" custom") for line in map(strip_ansi, component.render(40)))

    component.set_output_pad(0)

    assert {"expanded": False, "outputPad": 0}.items() <= options_seen[-1].items()
    assert any(line.startswith("custom") for line in map(strip_ansi, component.render(40)))


def test_runs_markdown_transformers_on_default_rendered_content():
    contexts = []

    def transformer(markdown, context):
        contexts.append(context)
        return markdown.replace("original", "transformed")

    message = CustomMessage(
        custom_type="test",
        content="original",
        display=True,
        timestamp=int(time.time() * 1000),
    )
    component = CustomMessageComponent(message, None, None, 1, [transformer], "entry-1", "2025-01-15T10:30:00.000Z")

    rendered = "\n".join(map(strip_ansi, component.render(40)))
    assert "transformed" in rendered
    assert "original" not in rendered
    # pi's toMatchObject: a subset check.
    assert {
        "messageType": "custom",
        "isStreaming": False,
        "messageId": "entry-1",
        "timestamp": "2025-01-15T10:30:00.000Z",
    }.items() <= contexts[0].items()
