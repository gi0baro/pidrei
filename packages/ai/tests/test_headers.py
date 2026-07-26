from pidrei_ai.utils.headers import merge_headers, provider_headers_to_record


def test_provider_headers_to_record_drops_nulls():
    assert provider_headers_to_record({"a": "1", "b": None}) == {"a": "1"}


def test_provider_headers_to_record_empty_results_are_none():
    assert provider_headers_to_record(None) is None
    assert provider_headers_to_record({}) is None
    assert provider_headers_to_record({"a": None}) is None


def test_merge_headers_none_inputs():
    assert merge_headers(None, None) is None
    # Mirrors the JS behavior: an empty object is not "absent".
    assert merge_headers({}, None) == {}
    assert merge_headers(None, {}) == {}


def test_merge_headers_override_is_case_insensitive():
    merged = merge_headers({"X-Api-Key": "old", "Accept": "json"}, {"x-api-key": "new"})

    assert merged == {"Accept": "json", "x-api-key": "new"}
    # Override casing wins and the entry moves to the end.
    assert list(merged) == ["Accept", "x-api-key"]


def test_merge_headers_preserves_null_values():
    # Null values pass through the merge; dropping them is providerHeadersToRecord's job.
    merged = merge_headers({"a": "1"}, {"a": None, "b": "2"})

    assert merged == {"a": None, "b": "2"}
