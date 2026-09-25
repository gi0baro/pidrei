"""Mirror of pi coding-agent test/interactive-mode-assistant-diagnostics.test.ts.

pi grabs the private method off InteractiveMode.prototype and calls it on a
fake `this`; the Python function is called the same way on a stub object.
"""

from dataclasses import replace
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
                    {"type": "thinking_dropped", "path": "messages.2.content.0", "reason": "prefix_binding_mismatch"},
                    {"type": "thinking_dropped", "path": "messages.5.content.0", "reason": "prefix_binding_mismatch"},
                    {"type": "thinking_dropped", "path": "messages.8.content.0", "reason": "prefix_binding_mismatch"},
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
        session_manager=SimpleNamespace(get_branch=list),
    )
    InteractiveMode._maybe_show_thinking_drop_notice(enabled, MESSAGE)
    output = strip_ansi("\n".join(enabled._chat_container.render(120)))
    assert "Anthropic dropped 3 thinking blocks (details in session)" in output

    disabled = SimpleNamespace(
        _chat_container=Container(),
        settings_manager=SimpleNamespace(get_show_cache_miss_notices=lambda: False),
        session_manager=SimpleNamespace(get_branch=list),
    )
    InteractiveMode._maybe_show_thinking_drop_notice(disabled, MESSAGE)
    assert len(disabled._chat_container.children) == 0


def test_does_not_repeat_unchanged_anthropic_thinking_drops():
    init_theme_sync("dark")
    context = SimpleNamespace(
        _chat_container=Container(),
        settings_manager=SimpleNamespace(get_show_cache_miss_notices=lambda: True),
        session_manager=SimpleNamespace(get_branch=lambda: [{"type": "message", "message": MESSAGE}]),
    )

    InteractiveMode._maybe_show_thinking_drop_notice(context, replace(MESSAGE, timestamp=2))

    assert len(context._chat_container.children) == 0


def test_ignores_the_current_message_when_it_is_already_persisted():
    # pidrei-specific: the UI owner can handle message_end after the session persisted
    # the message, so the branch's last assistant entry may be the message itself.
    init_theme_sync("dark")
    context = SimpleNamespace(
        _chat_container=Container(),
        settings_manager=SimpleNamespace(get_show_cache_miss_notices=lambda: True),
        session_manager=SimpleNamespace(get_branch=lambda: [{"type": "message", "message": MESSAGE}]),
    )

    InteractiveMode._maybe_show_thinking_drop_notice(context, MESSAGE)

    output = strip_ansi("\n".join(context._chat_container.render(120)))
    assert "Anthropic dropped 3 thinking blocks (details in session)" in output
