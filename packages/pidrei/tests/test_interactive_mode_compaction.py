"""Mirror of pi coding-agent test/interactive-mode-compaction.test.ts."""

from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.core.agent_session import CompactionEndEvent, PromptOptions
from pidrei.modes.interactive.interactive_mode import InteractiveMode


@pytest.mark.tonio
async def test_rebuilds_chat_and_appends_a_synthetic_compaction_summary_at_the_bottom():
    flush_calls: list = []

    async def flush_compaction_queue(options=None):
        flush_calls.append(options)

    fake = SimpleNamespace(
        _is_initialized=True,
        invalidate_calls=[],
        _auto_compaction_escape_handler=None,
        _default_editor=SimpleNamespace(),
        clear_status_calls=[],
        chat_clear_calls=[],
        rebuild_calls=[],
        added_messages=[],
        show_error_calls=[],
        show_status_calls=[],
        request_render_calls=[],
        _flush_compaction_queue=flush_compaction_queue,
        settings_manager=SimpleNamespace(get_show_terminal_progress=lambda: False),
    )
    fake._footer = SimpleNamespace(invalidate=lambda: fake.invalidate_calls.append(True))
    fake._clear_status_indicator = lambda kind=None: fake.clear_status_calls.append(kind)
    fake._chat_container = SimpleNamespace(clear=lambda: fake.chat_clear_calls.append(True))
    fake._rebuild_chat_from_messages = lambda: fake.rebuild_calls.append(True)
    fake._add_message_to_chat = fake.added_messages.append
    fake.show_error = fake.show_error_calls.append
    fake.show_status = fake.show_status_calls.append
    fake.ui = SimpleNamespace(
        request_render=lambda force=False: fake.request_render_calls.append(force),
        terminal=SimpleNamespace(set_progress=lambda active: None),
    )

    InteractiveMode._handle_event(
        fake,
        CompactionEndEvent(
            reason="manual",
            result=SimpleNamespace(tokens_before=123, summary="summary"),
            aborted=False,
            will_retry=False,
        ),
    )
    # The compaction queue flush is spawned; let it run.
    await tonio.time.sleep(0.01)

    assert fake.chat_clear_calls == [True]
    assert fake.rebuild_calls == [True]
    assert len(fake.added_messages) == 1
    message = fake.added_messages[0]
    assert message.role == "compactionSummary"
    assert message.tokens_before == 123
    assert message.summary == "summary"
    assert flush_calls == [{"willRetry": False}]


@pytest.mark.tonio
async def test_preserves_steering_behavior_when_flushing_into_an_active_agent_run():
    prompt_calls: list = []

    async def prompt(text, options=None):
        prompt_calls.append((text, options))

    async def steer(text):
        raise AssertionError("steer should not be used for the first prompt")

    async def follow_up(text):
        raise AssertionError("followUp should not be used for steer messages")

    fake = SimpleNamespace(
        _compaction_queued_messages=[{"text": "change direction", "mode": "steer"}],
        session=SimpleNamespace(
            clear_queue=lambda: {"steering": [], "followUp": []},
            prompt=prompt,
            steer=steer,
            follow_up=follow_up,
        ),
        _is_extension_command=lambda text: False,
        update_display_calls=[],
        show_error_calls=[],
    )
    fake._update_pending_messages_display = lambda: fake.update_display_calls.append(True)
    fake.show_error = fake.show_error_calls.append

    await InteractiveMode._flush_compaction_queue(fake, {"willRetry": False})
    # The first prompt is dispatched on a spawned task; let it run.
    await tonio.time.sleep(0.01)

    assert prompt_calls == [("change direction", PromptOptions(streaming_behavior="steer"))]
    assert fake._compaction_queued_messages == []
    assert fake.show_error_calls == []
