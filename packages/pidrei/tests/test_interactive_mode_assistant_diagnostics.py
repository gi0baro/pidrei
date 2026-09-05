"""Mirror of pi coding-agent test/interactive-mode-assistant-diagnostics.test.ts.

pi grabs the private method off InteractiveMode.prototype and calls it on a
fake `this`; the Python function is called the same way on a stub object.
"""

from types import SimpleNamespace

from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei.modes.interactive.theme import init_theme_sync
from pidrei.utils.ansi import strip_ansi
from pidrei_ai.types import AssistantMessage, AssistantMessageDiagnostic, TextContent, Usage, UsageCost
from pidrei_tui import Container


MESSAGE = AssistantMessage(
    content=[TextContent(text="survived")],
    api="anthropic-messages",
    provider="anthropic",
    model="claude-fable-5-1",
    usage=Usage(input=1, output=1, total_tokens=2, cost=UsageCost()),
    stop_reason="stop",
    timestamp=1,
    diagnostics=[
        AssistantMessageDiagnostic(
            type="anthropic_input_transformations",
            timestamp=1,
            details={
                "transformations": [
                    {"type": "thinking_dropped", "path": "messages.2.content.0", "reason": "prefix_binding_mismatch"}
                ]
            },
        )
    ],
)


def test_shows_anthropic_thinking_drops_when_cache_miss_notices_are_enabled():
    init_theme_sync("dark")
    enabled = SimpleNamespace(
        _chat_container=Container(),
        settings_manager=SimpleNamespace(get_show_cache_miss_notices=lambda: True),
    )
    InteractiveMode._maybe_show_assistant_diagnostics(enabled, MESSAGE)
    output = strip_ansi("\n".join(enabled._chat_container.render(120)))
    assert "Anthropic dropped thinking block: prefix_binding_mismatch at messages.2.content.0" in output

    disabled = SimpleNamespace(
        _chat_container=Container(),
        settings_manager=SimpleNamespace(get_show_cache_miss_notices=lambda: False),
    )
    InteractiveMode._maybe_show_assistant_diagnostics(disabled, MESSAGE)
    assert len(disabled._chat_container.children) == 0
