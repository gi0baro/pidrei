"""Mirror of pi coding-agent test/provider-matching.test.ts.

pi matches objects with `id` and `name`; the PiDrei function takes an accessor
for the two names, which the mirror passes for its dict records.
"""

from pidrei.core.provider_matching import find_matching_providers


PROVIDERS = [
    {"id": "first", "name": "Shared Name", "authType": "oauth"},
    {"id": "first", "name": "Shared Name", "authType": "api_key"},
    {"id": "other", "name": "FIRST", "authType": "api_key"},
    {"id": "second", "name": "Second Provider", "authType": "oauth"},
]


def names(provider: dict) -> tuple[str, str]:
    return provider["id"], provider["name"]


def test_matches_ids_and_display_names_case_insensitively_with_surrounding_whitespace():
    assert find_matching_providers(PROVIDERS, "  sEcOnD  ", names) == [PROVIDERS[3]]
    assert find_matching_providers(PROVIDERS, "\tSECOND PROVIDER\n", names) == [PROVIDERS[3]]


def test_keeps_all_matching_providers_and_auth_methods_in_input_order():
    matched = find_matching_providers(PROVIDERS, "first", names)
    assert matched == PROVIDERS[0:3]
    assert matched[0] is PROVIDERS[0]
    assert matched[1]["authType"] == "api_key"
    assert find_matching_providers(PROVIDERS, "shared name", names) == PROVIDERS[0:2]


def test_does_not_fuzzy_match_unknown_or_partial_references():
    assert find_matching_providers(PROVIDERS, "unknown", names) == []
    assert find_matching_providers(PROVIDERS, "sec", names) == []
    assert find_matching_providers([], "first", names) == []


def test_never_matches_an_empty_reference_even_against_empty_display_names():
    assert find_matching_providers([{"id": "unnamed", "name": ""}], "", names) == []
    assert find_matching_providers(PROVIDERS, " \t ", names) == []
