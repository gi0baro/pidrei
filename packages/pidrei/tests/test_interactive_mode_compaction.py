"""Mirror of pi coding-agent test/interactive-mode-compaction.test.ts."""

from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.core.agent_session import CompactionEndEvent, PromptOptions
from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei.modes.interactive.theme import init_theme_sync
from pidrei.utils.ansi import strip_ansi
from pidrei_ai.types import Usage, UsageCost
from pidrei_tui import Container


def _usage(total_cost: float) -> Usage:
    return Usage(
        input=10,
        output=20,
        cache_read=30,
        cache_write=40,
        total_tokens=100,
        cost=UsageCost(input=0.01, output=0.02, cache_read=0.03, cache_write=total_cost - 0.06, total=total_cost),
    )


def _compaction_entry(entry_id: str, parent_id: str | None, summary: str, tokens_before: int, usage: Usage) -> dict:
    return {
        "type": "compaction",
        "id": entry_id,
        "parentId": parent_id,
        "timestamp": "2025-01-02T00:00:00Z",
        "summary": summary,
        "firstKeptEntryId": "kept",
        "tokensBefore": tokens_before,
        "usage": usage,
    }


def test_uses_the_cache_miss_notice_setting_for_compaction_and_branch_summary_costs():
    usage = _usage(0.125)
    init_theme_sync("dark")

    enabled = SimpleNamespace(
        _chat_container=Container(),
        settings_manager=SimpleNamespace(get_show_cache_miss_notices=lambda: True),
    )
    InteractiveMode._add_compaction_cost_notice(
        enabled, {"type": "compaction_cost", "kind": "compaction", "usage": usage}
    )
    InteractiveMode._add_compaction_cost_notice(
        enabled, {"type": "compaction_cost", "kind": "branch_summary", "usage": usage}
    )
    output = strip_ansi("\n".join(enabled._chat_container.render(120)))
    # pi's fixture cost of 0.125 is an exact binary tie: JS `toFixed(2)` rounds it
    # half-up to 0.13, Python's `:.2f` rounds half-even to 0.12. The pre-existing
    # cache-miss notice below formats the same way, so both notices stay consistent.
    assert "Compaction: 100 tokens billed (~$0.12)" in output
    assert "Branch summary: 100 tokens billed (~$0.12)" in output

    disabled = SimpleNamespace(
        _chat_container=Container(),
        settings_manager=SimpleNamespace(get_show_cache_miss_notices=lambda: False),
    )
    InteractiveMode._add_compaction_cost_notice(
        disabled, {"type": "compaction_cost", "kind": "compaction", "usage": usage}
    )
    assert disabled._chat_container.children == []


def test_renders_each_compaction_cost_after_its_summary():
    current_usage = _usage(0.1)
    previous_usage = Usage(
        input=1,
        output=2,
        cache_read=3,
        cache_write=4,
        total_tokens=10,
        cost=UsageCost(input=0.001, output=0.002, cache_read=0.003, cache_write=0.004, total=0.01),
    )
    entries = [
        _compaction_entry("current", "previous", "current summary", 200, current_usage),
        _compaction_entry("previous", None, "previous summary", 100, previous_usage),
    ]
    render_calls: list = []
    fake = SimpleNamespace()
    fake._render_session_items = lambda items, options=None: render_calls.append((items, options))

    InteractiveMode._render_session_entries(fake, entries)

    assert len(render_calls) == 1
    items, options = render_calls[0]
    assert options is None
    assert items[0].role == "compactionSummary"
    assert items[0].summary == "current summary"
    assert items[1] == {"type": "compaction_cost", "kind": "compaction", "usage": current_usage}
    assert items[2].role == "compactionSummary"
    assert items[2].summary == "previous summary"
    assert items[3] == {"type": "compaction_cost", "kind": "compaction", "usage": previous_usage}
    assert len(items) == 4


@pytest.mark.tonio
async def test_renders_retained_entries_and_appends_the_latest_summary_cost_at_the_bottom():
    usage = _usage(0.125)
    latest_compaction = _compaction_entry("latest", "previous", "summary", 123, usage)
    previous_compaction = _compaction_entry("previous", None, "previous summary", 100, usage)
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
        render_entries_calls=[],
        cost_notices=[],
        added_messages=[],
        session_manager=SimpleNamespace(build_context_entries=lambda: [latest_compaction, previous_compaction]),
        show_error_calls=[],
        show_status_calls=[],
        request_render_calls=[],
        _flush_compaction_queue=flush_compaction_queue,
        settings_manager=SimpleNamespace(get_show_terminal_progress=lambda: False),
    )
    fake._footer = SimpleNamespace(invalidate=lambda: fake.invalidate_calls.append(True))
    fake._clear_status_indicator = lambda kind=None: fake.clear_status_calls.append(kind)
    fake._chat_container = SimpleNamespace(clear=lambda: fake.chat_clear_calls.append(True))
    fake._render_session_entries = lambda entries: fake.render_entries_calls.append(entries)
    fake._add_message_to_chat = fake.added_messages.append
    fake._add_compaction_cost_notice = fake.cost_notices.append
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
            result=SimpleNamespace(tokens_before=123, summary="summary", usage=usage),
            aborted=False,
            will_retry=False,
        ),
    )
    # The compaction queue flush is spawned; let it run.
    await tonio.time.sleep(0.01)

    assert fake.chat_clear_calls == [True]
    assert fake.render_entries_calls == [[previous_compaction]]
    assert len(fake.added_messages) == 1
    message = fake.added_messages[0]
    assert message.role == "compactionSummary"
    assert message.tokens_before == 123
    assert message.summary == "summary"
    assert fake.cost_notices == [{"type": "compaction_cost", "kind": "compaction", "usage": usage}]
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
