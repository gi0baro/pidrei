"""Mirror of pi coding-agent test/footer-width.test.ts."""

from types import SimpleNamespace

import pytest

from pidrei.core.session_manager import SessionManager
from pidrei.modes.interactive.components import FooterComponent, format_cwd_for_footer
from pidrei.modes.interactive.theme import init_theme_sync
from pidrei.utils.ansi import strip_ansi
from pidrei_ai.types import AssistantMessage, Usage, UsageCost
from pidrei_tui import visible_width


def _usage(*, input=0, output=0, cache_read=0, cache_write=0, total=0.0) -> Usage:
    return Usage(
        input=input,
        output=output,
        cache_read=cache_read,
        cache_write=cache_write,
        cost=UsageCost(total=total),
    )


def create_session(
    *,
    session_name: str,
    model_id: str = "test-model",
    provider: str = "test",
    reasoning: bool = False,
    thinking_level: str = "off",
    usage: Usage | None = None,
    branch_usage: Usage | None = None,
    compaction_usage: Usage | None = None,
    tool_usage: Usage | None = None,
    using_subscription: bool = False,
):
    entries: list = []

    if usage is not None:
        entries.append({"type": "message", "message": SimpleNamespace(role="assistant", usage=usage)})

    if branch_usage is not None:
        entries.append({"type": "branch_summary", "usage": branch_usage})

    if compaction_usage is not None:
        entries.append({"type": "compaction", "usage": compaction_usage})

    if tool_usage is not None:
        entries.append({"type": "message", "message": SimpleNamespace(role="toolResult", usage=tool_usage)})

    return SimpleNamespace(
        state=SimpleNamespace(
            model=SimpleNamespace(id=model_id, provider=provider, context_window=200_000, reasoning=reasoning),
            thinking_level=thinking_level,
        ),
        session_manager=SimpleNamespace(
            get_entries=lambda: entries,
            get_entries_revision=lambda: len(entries),
            get_session_name=lambda: session_name,
            get_cwd=lambda: "/tmp/project",
        ),
        # AgentSession.get_context_usage is a method (pi calls it too) — the
        # fake must be callable, not a data attribute.
        get_context_usage=lambda: SimpleNamespace(context_window=200_000, percent=12.3),
        model_runtime=SimpleNamespace(is_using_subscription=lambda provider_id: using_subscription),
    )


def create_footer_data(provider_count: int):
    return SimpleNamespace(
        get_git_branch=lambda: "main",
        get_extension_statuses=dict,
        get_available_provider_count=lambda: provider_count,
        on_branch_change=lambda callback: lambda: None,
    )


class TestFormatCwdForFooter:
    def test_does_not_abbreviate_sibling_paths_that_share_the_home_prefix(self):
        assert format_cwd_for_footer("/home/user2", "/home/user") == "/home/user2"

    def test_abbreviates_the_home_directory_and_descendants(self):
        assert format_cwd_for_footer("/home/user", "/home/user") == "~"
        assert format_cwd_for_footer("/home/user/project", "/home/user") == "~/project"


class TestFooterComponentWidthHandling:
    @pytest.fixture(autouse=True)
    def _theme(self):
        init_theme_sync(None, False)

    def test_keeps_all_lines_within_width_for_wide_session_names(self):
        width = 93
        session = create_session(session_name="한글" * 30)
        footer = FooterComponent(session, create_footer_data(1))

        lines = footer.render(width)
        for line in lines:
            assert visible_width(line) <= width

    def test_keeps_stats_line_within_width_for_wide_model_and_provider_names(self):
        width = 60
        session = create_session(
            session_name="",
            model_id="模" * 30,
            provider="공급자",
            reasoning=True,
            thinking_level="high",
            usage=_usage(input=12_345, output=6_789, total=1.234),
        )
        footer = FooterComponent(session, create_footer_data(2))

        lines = footer.render(width)
        for line in lines:
            assert visible_width(line) <= width

    def test_includes_summary_and_tool_result_usage_in_the_total_cost(self):
        session = create_session(
            session_name="",
            usage=_usage(input=100, output=10, total=0.5),
            branch_usage=_usage(input=20, output=5, total=0.25),
            compaction_usage=_usage(input=5, output=2, total=0.125),
            tool_usage=_usage(input=15, output=3, total=0.375),
        )
        footer = FooterComponent(session, create_footer_data(1))

        stats_line = strip_ansi(footer.render(120)[1])
        assert "$1.250" in stats_line

    def test_shows_the_latest_cache_hit_rate_when_cache_usage_is_present(self):
        session = create_session(
            session_name="",
            usage=_usage(input=100, output=10, cache_read=50, cache_write=50, total=0.001),
        )
        footer = FooterComponent(session, create_footer_data(1))

        stats_line = strip_ansi(footer.render(120)[1])
        assert "CH25.0%" in stats_line

    def test_marks_explicitly_identified_subscription_auth(self):
        session = create_session(session_name="", provider="anthropic", using_subscription=True)
        footer = FooterComponent(session, create_footer_data(1))

        assert "$0.000 (sub)" in strip_ansi(footer.render(120)[1])

    def test_does_not_mark_generic_oauth_sign_in_as_a_subscription(self):
        session = create_session(
            session_name="",
            provider="openrouter",
            usage=_usage(input=100, output=10, total=1.234),
        )
        footer = FooterComponent(session, create_footer_data(1))

        line = strip_ansi(footer.render(120)[1])
        assert "$1.234" in line
        assert "(sub)" not in line

    def test_marks_kimi_coding_costs_as_subscription_estimates(self):
        session = create_session(
            session_name="",
            provider="kimi-coding",
            usage=_usage(input=100, output=10, total=1.234),
        )
        footer = FooterComponent(session, create_footer_data(1))

        assert "$1.234 (sub)" in strip_ansi(footer.render(120)[1])

    @pytest.mark.tonio
    async def test_usage_totals_follow_entries_appended_to_a_real_session(self):
        # The totals are cached per frame on the session manager's entries
        # revision; an append must show up on the next render.
        session_manager = SessionManager.in_memory()
        session = create_session(session_name="")
        session.session_manager = session_manager
        footer = FooterComponent(session, create_footer_data(1))

        assert "↑" not in strip_ansi(footer.render(120)[1])

        await session_manager.append_message(
            AssistantMessage(
                content=[],
                api="anthropic-messages",
                provider="anthropic",
                model="test-model",
                usage=_usage(input=1500, output=10, total=0.5),
                stop_reason="stop",
                timestamp=0,
            )
        )
        assert "↑1.5k" in strip_ansi(footer.render(120)[1])
