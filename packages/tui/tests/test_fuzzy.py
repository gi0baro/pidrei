"""Mirror of pi tui test/fuzzy.test.ts."""

from pidrei_tui.fuzzy import fuzzy_filter, fuzzy_match


# fuzzyMatch


def test_empty_query_matches_everything_with_score_0():
    result = fuzzy_match("", "anything")
    assert result["matches"] is True
    assert result["score"] == 0


def test_query_longer_than_text_does_not_match():
    result = fuzzy_match("longquery", "short")
    assert result["matches"] is False


def test_exact_match_has_good_score():
    result = fuzzy_match("test", "test")
    assert result["matches"] is True
    assert result["score"] < 0  # Should be negative due to consecutive bonuses


def test_characters_must_appear_in_order():
    match_in_order = fuzzy_match("abc", "aXbXc")
    assert match_in_order["matches"] is True

    match_out_of_order = fuzzy_match("abc", "cba")
    assert match_out_of_order["matches"] is False


def test_case_insensitive_matching():
    result = fuzzy_match("ABC", "abc")
    assert result["matches"] is True

    result2 = fuzzy_match("abc", "ABC")
    assert result2["matches"] is True


def test_consecutive_matches_score_better_than_scattered_matches():
    consecutive = fuzzy_match("foo", "foobar")
    scattered = fuzzy_match("foo", "f_o_o_bar")

    assert consecutive["matches"] is True
    assert scattered["matches"] is True
    assert consecutive["score"] < scattered["score"]


def test_word_boundary_matches_score_better():
    at_boundary = fuzzy_match("fb", "foo-bar")
    not_at_boundary = fuzzy_match("fb", "afbx")

    assert at_boundary["matches"] is True
    assert not_at_boundary["matches"] is True
    assert at_boundary["score"] < not_at_boundary["score"]


def test_matches_swapped_alpha_numeric_tokens():
    result = fuzzy_match("codex52", "gpt-5.2-codex")
    assert result["matches"] is True


# fuzzyFilter


def test_empty_query_returns_all_items_unchanged():
    items = ["apple", "banana", "cherry"]
    result = fuzzy_filter(items, "", lambda x: x)
    assert result == items


def test_filters_out_non_matching_items():
    items = ["apple", "banana", "cherry"]
    result = fuzzy_filter(items, "an", lambda x: x)
    assert "banana" in result
    assert "apple" not in result
    assert "cherry" not in result


def test_sorts_results_by_match_quality():
    items = ["a_p_p", "app", "application"]
    result = fuzzy_filter(items, "app", lambda x: x)

    # "app" should be first (exact consecutive match at start)
    assert result[0] == "app"


def test_prioritizes_exact_matches_over_longer_prefix_matches():
    items = ["clone", "cl"]
    result = fuzzy_filter(items, "cl", lambda x: x)

    assert result == ["cl", "clone"]


def test_works_with_custom_get_text_function():
    items = [
        {"name": "foo", "id": 1},
        {"name": "bar", "id": 2},
        {"name": "foobar", "id": 3},
    ]
    result = fuzzy_filter(items, "foo", lambda item: item["name"])

    assert len(result) == 2
    assert "foo" in [r["name"] for r in result]
    assert "foobar" in [r["name"] for r in result]


def test_matches_slash_separated_provider_model_queries_against_reordered_text():
    item = {"id": "gpt-5.5", "provider": "openai-codex"}
    result = fuzzy_filter([item], "openai-codex/gpt-5.5", lambda model: f"{model['id']} {model['provider']}")

    assert result == [item]
