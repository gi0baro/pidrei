"""Mirror of pi coding-agent test/interactive-mode-assistant-diagnostics.test.ts.

pi grabs the private method off InteractiveMode.prototype and calls it on a
fake `this`; the Python function is called the same way on a stub object.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei.modes.interactive.theme import init_theme
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


@pytest.mark.tonio
async def test_shows_anthropic_thinking_drops_when_cache_miss_notices_are_enabled():
    await init_theme("dark")
    enabled = SimpleNamespace(
        _chat_container=Container(),
        settings_manager=SimpleNamespace(get_show_cache_miss_notices=lambda: True),
        session_manager=SimpleNamespace(get_branch=list),
    )
    InteractiveMode._maybe_show_thinking_drop_notice(enabled, MESSAGE)
    output = strip_ansi("\n".join(enabled._chat_container.render(120)))
    assert "Anthropic dropped 3 thinking blocks: prefix_binding_mismatch (details in session)" in output

    disabled = SimpleNamespace(
        _chat_container=Container(),
        settings_manager=SimpleNamespace(get_show_cache_miss_notices=lambda: False),
        session_manager=SimpleNamespace(get_branch=list),
    )
    InteractiveMode._maybe_show_thinking_drop_notice(disabled, MESSAGE)
    assert len(disabled._chat_container.children) == 0


@pytest.mark.tonio
async def test_does_not_repeat_unchanged_anthropic_thinking_drops():
    await init_theme("dark")
    context = SimpleNamespace(
        _chat_container=Container(),
        settings_manager=SimpleNamespace(get_show_cache_miss_notices=lambda: True),
        session_manager=SimpleNamespace(get_branch=lambda: [{"type": "message", "message": MESSAGE}]),
    )

    InteractiveMode._maybe_show_thinking_drop_notice(context, replace(MESSAGE, timestamp=2))

    assert len(context._chat_container.children) == 0


@pytest.mark.tonio
async def test_ignores_the_current_message_when_it_is_already_persisted():
    # pidrei-specific guard: when the notice runs after the session persisted the
    # message, the branch's last assistant entry is the message itself. (It dates
    # from when interactive mode handled message_end after persistence; the
    # listener now applies it inside the emit, but the guard keeps the notice
    # correct either way.)
    await init_theme("dark")
    context = SimpleNamespace(
        _chat_container=Container(),
        settings_manager=SimpleNamespace(get_show_cache_miss_notices=lambda: True),
        session_manager=SimpleNamespace(get_branch=lambda: [{"type": "message", "message": MESSAGE}]),
    )

    InteractiveMode._maybe_show_thinking_drop_notice(context, MESSAGE)

    output = strip_ansi("\n".join(context._chat_container.render(120)))
    assert "Anthropic dropped 3 thinking blocks: prefix_binding_mismatch (details in session)" in output


def openai_message(ids: list[str], reasons: list[str]) -> AssistantMessage:
    return replace(
        MESSAGE,
        api="openai-responses",
        provider="openai",
        model="gpt-6.2-sol",
        timestamp=3,
        diagnostics=[
            AssistantMessageDiagnostic(
                type="openai_input_transformations",
                timestamp=3,
                details={
                    "transformations": [
                        {
                            "type": "reasoning",
                            "id": id,
                            "action": "removed",
                            "reason": reasons[min(index, len(reasons) - 1)],
                        }
                        for index, id in enumerate(ids)
                    ]
                },
            )
        ],
    )


@pytest.mark.tonio
async def test_shows_openai_reasoning_removals_with_their_reasons():
    await init_theme("dark")
    context = SimpleNamespace(
        _chat_container=Container(),
        settings_manager=SimpleNamespace(get_show_cache_miss_notices=lambda: True),
        session_manager=SimpleNamespace(get_branch=lambda: [{"type": "message", "message": MESSAGE}]),
    )

    # The previous Anthropic response dropped more blocks; that must not hide the OpenAI removals.
    InteractiveMode._maybe_show_thinking_drop_notice(
        context, openai_message(["rs_1", "rs_2"], ["prefix_mismatch", "organization_mismatch"])
    )

    output = strip_ansi("\n".join(context._chat_container.render(120)))
    assert "OpenAI dropped 2 reasoning items: prefix_mismatch, organization_mismatch (details in session)" in output


@pytest.mark.tonio
async def test_shows_openai_reasoning_removals_only_when_new_items_are_removed():
    await init_theme("dark")
    # OpenAI reports a removed item again on every later request that replays it.
    previous = openai_message(["rs_1", "rs_2"], ["prefix_mismatch"])
    context = SimpleNamespace(
        _chat_container=Container(),
        settings_manager=SimpleNamespace(get_show_cache_miss_notices=lambda: True),
        session_manager=SimpleNamespace(get_branch=lambda: [{"type": "message", "message": previous}]),
    )

    InteractiveMode._maybe_show_thinking_drop_notice(context, openai_message(["rs_1", "rs_2"], ["prefix_mismatch"]))
    assert len(context._chat_container.children) == 0

    InteractiveMode._maybe_show_thinking_drop_notice(
        context, openai_message(["rs_1", "rs_2", "rs_3"], ["prefix_mismatch"])
    )
    output = strip_ansi("\n".join(context._chat_container.render(120)))
    assert "OpenAI dropped 3 reasoning items: prefix_mismatch (details in session)" in output
