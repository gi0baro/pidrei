"""Mirror of pi coding-agent test/session-selector-search.test.ts."""

from datetime import UTC, datetime

from pidrei.core.session_manager import SessionInfo
from pidrei.modes.interactive.components.session_selector_search import filter_and_sort_sessions


def make_session(
    *, id, modified, all_messages_text, cwd="", name=None, created=None, message_count=1, first_message="(no messages)"
):
    return SessionInfo(
        path=f"/tmp/{id}.jsonl",
        id=id,
        cwd=cwd,
        name=name,
        created=created if created is not None else datetime.fromtimestamp(0, tz=UTC),
        modified=modified,
        message_count=message_count,
        first_message=first_message,
        all_messages_text=all_messages_text,
    )


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


class TestSessionSelectorSearch:
    def test_filters_by_quoted_phrase_with_whitespace_normalization(self):
        sessions = [
            make_session(
                id="a", modified=_dt("2026-01-01T00:00:00.000Z"), all_messages_text="node\n\n   cve was discussed"
            ),
            make_session(id="b", modified=_dt("2026-01-02T00:00:00.000Z"), all_messages_text="node something else"),
        ]

        result = filter_and_sort_sessions(sessions, '"node cve"', "recent")
        assert [s.id for s in result] == ["a"]

    def test_filters_by_regex_re_and_is_case_insensitive(self):
        sessions = [
            make_session(id="a", modified=_dt("2026-01-02T00:00:00.000Z"), all_messages_text="Brave is great"),
            make_session(id="b", modified=_dt("2026-01-03T00:00:00.000Z"), all_messages_text="bravery is not the same"),
        ]

        result = filter_and_sort_sessions(sessions, "re:\\bbrave\\b", "recent")
        assert [s.id for s in result] == ["a"]

    def test_recent_sort_preserves_input_order(self):
        sessions = [
            make_session(id="newer", modified=_dt("2026-01-03T00:00:00.000Z"), all_messages_text="brave"),
            make_session(id="older", modified=_dt("2026-01-01T00:00:00.000Z"), all_messages_text="brave"),
            make_session(id="nomatch", modified=_dt("2026-01-04T00:00:00.000Z"), all_messages_text="something else"),
        ]

        result = filter_and_sort_sessions(sessions, '"brave"', "recent")
        assert [s.id for s in result] == ["newer", "older"]

    def test_relevance_sort_orders_by_score_and_tie_breaks_by_modified_desc(self):
        sessions = [
            make_session(id="late", modified=_dt("2026-01-03T00:00:00.000Z"), all_messages_text="xxxx brave"),
            make_session(id="early", modified=_dt("2026-01-01T00:00:00.000Z"), all_messages_text="brave xxxx"),
        ]

        result1 = filter_and_sort_sessions(sessions, '"brave"', "relevance")
        assert [s.id for s in result1] == ["early", "late"]

        tie_sessions = [
            make_session(id="newer", modified=_dt("2026-01-03T00:00:00.000Z"), all_messages_text="brave"),
            make_session(id="older", modified=_dt("2026-01-01T00:00:00.000Z"), all_messages_text="brave"),
        ]

        result2 = filter_and_sort_sessions(tie_sessions, '"brave"', "relevance")
        assert [s.id for s in result2] == ["newer", "older"]

    def test_returns_empty_list_for_invalid_regex(self):
        sessions = [
            make_session(id="a", modified=_dt("2026-01-01T00:00:00.000Z"), all_messages_text="brave"),
        ]

        result = filter_and_sort_sessions(sessions, "re:(", "recent")
        assert result == []


def _name_filter_sessions() -> list:
    return [
        make_session(
            id="named1", name="My Project", modified=_dt("2026-01-03T00:00:00.000Z"), all_messages_text="blueberry"
        ),
        make_session(
            id="named2", name="Another Named", modified=_dt("2026-01-02T00:00:00.000Z"), all_messages_text="blueberry"
        ),
        make_session(id="other1", modified=_dt("2026-01-04T00:00:00.000Z"), all_messages_text="blueberry"),
        make_session(id="other2", modified=_dt("2026-01-01T00:00:00.000Z"), all_messages_text="blueberry"),
    ]


class TestNameFilter:
    def test_returns_all_sessions_when_name_filter_is_all(self):
        result = filter_and_sort_sessions(_name_filter_sessions(), "", "recent", "all")
        assert [session.id for session in result] == ["named1", "named2", "other1", "other2"]

    def test_returns_only_named_sessions_when_name_filter_is_named(self):
        result = filter_and_sort_sessions(_name_filter_sessions(), "", "recent", "named")
        assert [session.id for session in result] == ["named1", "named2"]

    def test_applies_name_filter_before_search_query(self):
        result = filter_and_sort_sessions(_name_filter_sessions(), "blueberry", "recent", "named")
        assert [session.id for session in result] == ["named1", "named2"]

    def test_excludes_whitespace_only_names_from_named_filter(self):
        sessions_with_whitespace = [
            make_session(
                id="whitespace", name="   ", modified=_dt("2026-01-01T00:00:00.000Z"), all_messages_text="test"
            ),
            make_session(id="empty", name="", modified=_dt("2026-01-02T00:00:00.000Z"), all_messages_text="test"),
            make_session(
                id="named", name="Real Name", modified=_dt("2026-01-03T00:00:00.000Z"), all_messages_text="test"
            ),
        ]

        result = filter_and_sort_sessions(sessions_with_whitespace, "", "recent", "named")
        assert [session.id for session in result] == ["named"]
