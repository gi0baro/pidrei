"""Mirror of pi coding-agent test/session-selector-search.test.ts."""

from datetime import UTC, datetime

from pidrei.core.session_manager import SessionInfo
from pidrei.modes.interactive.components.session_selector_search import MessageTextCache, filter_and_sort_sessions


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
            make_session(id="message", modified=_dt("2026-01-03T00:00:00.000Z"), all_messages_text="brave"),
            make_session(
                id="name", name="Brave work", modified=_dt("2026-01-01T00:00:00.000Z"), all_messages_text="xxxx"
            ),
        ]

        result1 = filter_and_sort_sessions(sessions, '"brave"', "relevance")
        assert [s.id for s in result1] == ["name", "message"]

        tie_sessions = [
            make_session(id="newer", modified=_dt("2026-01-03T00:00:00.000Z"), all_messages_text="brave"),
            make_session(id="older", modified=_dt("2026-01-01T00:00:00.000Z"), all_messages_text="brave"),
        ]

        result2 = filter_and_sort_sessions(tie_sessions, '"brave"', "relevance")
        assert [s.id for s in result2] == ["newer", "older"]

    # #11220
    def test_ignores_the_position_of_a_match_in_message_text(self):
        filler = "lorem ipsum dolor sit amet " * 400
        sessions = [
            make_session(id="early", modified=_dt("2026-01-01T00:00:00.000Z"), all_messages_text=f"opus {filler}"),
            make_session(
                id="deep",
                modified=_dt("2026-01-02T00:00:00.000Z"),
                all_messages_text=f"{filler} we discussed the opus model",
            ),
            make_session(id="scattered", modified=_dt("2026-01-03T00:00:00.000Z"), all_messages_text=filler),
        ]

        result = filter_and_sort_sessions(sessions, "opus", "relevance")
        assert [s.id for s in result] == ["deep", "early"]

    # #11024
    def test_does_not_match_scattered_characters_in_session_ids(self):
        sessions = [
            make_session(
                id="019e9024-aaaa-7bbb-8ccc-dddddddddddd",
                modified=_dt("2026-01-03T00:00:00.000Z"),
                all_messages_text="unrelated",
            ),
            make_session(
                id="01a1ffff-aaaa-7bbb-8ccc-dddddddddddd",
                name="#1024 TDD bare env",
                modified=_dt("2026-01-01T00:00:00.000Z"),
                all_messages_text="unrelated",
            ),
        ]

        assert [s.name for s in filter_and_sort_sessions(sessions, "1024", "relevance")] == ["#1024 TDD bare env"]
        assert [s.id for s in filter_and_sort_sessions(sessions, "9024-aa", "relevance")] == [
            "019e9024-aaaa-7bbb-8ccc-dddddddddddd"
        ]

    def test_ranks_name_text_then_message_text_then_scattered_name_matches(self):
        sessions = [
            make_session(
                id="scattered-name",
                name="c-a-c-h-e notes",
                modified=_dt("2026-01-04T00:00:00.000Z"),
                all_messages_text="unrelated",
            ),
            make_session(
                id="message",
                name="Other work",
                modified=_dt("2026-01-03T00:00:00.000Z"),
                all_messages_text="A discussion mentioning the cache.",
            ),
            make_session(
                id="name", name="Cache work", modified=_dt("2026-01-01T00:00:00.000Z"), all_messages_text="Summary"
            ),
        ]

        assert [s.id for s in filter_and_sort_sessions(sessions, "cache", "relevance")] == [
            "name",
            "message",
            "scattered-name",
        ]

    def test_matches_each_token_within_a_single_field(self):
        sessions = [
            make_session(
                id="split", name="Fuzzy", modified=_dt("2026-01-01T00:00:00.000Z"), all_messages_text="search ranking"
            ),
        ]

        assert [s.id for s in filter_and_sort_sessions(sessions, "fuzzy ranking", "relevance")] == ["split"]
        assert filter_and_sort_sessions(sessions, "zzysea", "relevance") == []
        assert filter_and_sort_sessions(sessions, "re:fuzzy search", "relevance") == []

    def test_ranks_regex_name_matches_first(self):
        sessions = [
            make_session(id="message", modified=_dt("2026-01-02T00:00:00.000Z"), all_messages_text="brave"),
            make_session(id="name", name="Brave", modified=_dt("2026-01-01T00:00:00.000Z"), all_messages_text="xxxx"),
        ]

        assert [s.id for s in filter_and_sort_sessions(sessions, "re:\\bbrave\\b", "relevance")] == [
            "name",
            "message",
        ]

    def test_matches_a_reloaded_session_against_its_new_message_text(self):
        # PiDrei-only: the lowercased text is cached by session file, not by session object.
        messages = MessageTextCache()
        before = make_session(id="a", modified=_dt("2026-01-01T00:00:00.000Z"), all_messages_text="first topic")
        assert filter_and_sort_sessions([before], "first", "relevance", "all", messages) == [before]

        after = make_session(id="a", modified=_dt("2026-01-02T00:00:00.000Z"), all_messages_text="second topic")
        assert filter_and_sort_sessions([after], "second", "relevance", "all", messages) == [after]
        assert filter_and_sort_sessions([after], "first", "relevance", "all", messages) == []

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
