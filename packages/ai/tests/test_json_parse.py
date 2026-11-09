"""Tests of `pidrei_ai.utils.json_parse`, with the mirror of pi's json-parse.test.ts.

pi's `parseStreamingJson` cases have no target: PiDrei does not port that
function (streamed arguments go through `streaming_json`). The behaviour they
pin, invalid `\\u` escapes kept literal, valid ones decoded and one cut off
mid-stream omitted, is `test_streaming_json.py`'s #10868 case.
"""

import pytest

from pidrei_ai.utils.json_parse import parse_json_with_repair, parse_tool_call_arguments, repair_json


def test_repair_doubles_invalid_escapes():
    assert repair_json('{"path":"A\\H"}') == '{"path":"A\\\\H"}'


def test_repair_keeps_valid_escapes():
    text = '{"a":"line\\nbreak \\t \\u00e9 \\\\ \\" end"}'
    assert repair_json(text) == text


def test_repair_escapes_raw_control_characters_in_strings():
    assert repair_json('{"a":"col1\tcol2"}') == '{"a":"col1\\tcol2"}'
    assert repair_json('{"a":"x\x01y"}') == '{"a":"x\\u0001y"}'


def test_repair_doubles_incomplete_unicode_escape():
    # A \u without four hex digits is an invalid escape like any other, so it
    # is kept as literal text (pi #10868).
    assert repair_json('{"a":"\\u12"}') == '{"a":"\\\\u12"}'


def test_repair_trailing_backslash():
    assert repair_json('{"a":"x\\') == '{"a":"x\\\\'


def test_repair_leaves_structure_outside_strings_alone():
    assert repair_json('{\n\t"a": 1\n}') == '{\n\t"a": 1\n}'


def test_parse_with_repair_fixes_pi_malformed_tool_json():
    # The malformed payload from pi's anthropic-sse-parsing.test.ts: an invalid
    # \H escape and a raw tab inside a string literal.
    parsed = parse_json_with_repair('{"path":"A\\H","text":"col1\tcol2"}')
    assert parsed == {"path": "A\\H", "text": "col1\tcol2"}


def test_parse_with_repair_raises_on_unrepairable_input():
    with pytest.raises(ValueError):
        parse_json_with_repair("not json")


# Mirror of pi's json-parse.test.ts (parseToolCallArguments, pi #10897).


def test_tool_call_arguments_parses_complete_json_and_repairs_invalid_escapes():
    assert parse_tool_call_arguments(r'{"kind":"seek","note":"\url"}') == ({"kind": "seek", "note": "\\url"}, None)


def test_tool_call_arguments_uses_empty_arguments_for_empty_text():
    assert parse_tool_call_arguments("") == ({}, None)


def test_tool_call_arguments_does_not_keep_the_valid_prefix_of_malformed_json():
    raw = '{"action":"apply","actions":[{"kind":"seek","time":.5}]}'
    arguments, arguments_error = parse_tool_call_arguments(raw)
    assert arguments == {}
    assert "\n\nReceived arguments:\n" in arguments_error
    assert arguments_error.endswith(raw)


def test_tool_call_arguments_does_not_complete_cut_off_json():
    raw = '{"kind":"seek","clipId":"clip-fl'
    arguments, arguments_error = parse_tool_call_arguments(raw)
    assert arguments == {}
    assert arguments_error.endswith(raw)


def test_tool_call_arguments_rejects_json_that_is_not_an_object():
    assert parse_tool_call_arguments("[1,2]") == (
        {},
        "Tool arguments must be a JSON object\n\nReceived arguments:\n[1,2]",
    )
