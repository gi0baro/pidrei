import pytest

from pidrei_ai.utils.json_parse import parse_json_with_repair, parse_streaming_json, repair_json


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


def test_streaming_empty_input_returns_empty_dict():
    assert parse_streaming_json(None) == {}
    assert parse_streaming_json("") == {}
    assert parse_streaming_json("   ") == {}


def test_streaming_partial_object():
    assert parse_streaming_json('{"a": "he') == {"a": "he"}
    assert parse_streaming_json('{"a": 1, "b"') == {"a": 1}


def test_streaming_partial_with_repair():
    assert parse_streaming_json('{"path":"A\\H","text":"col1\tcol2"}') == {"path": "A\\H", "text": "col1\tcol2"}


def test_streaming_prefix_with_raw_control_character_still_repairs():
    # A streamed delta is both truncated and (pi-style) malformed; the repair
    # gate must not skip the repair just because the text is a prefix.
    assert parse_streaming_json('{"path":"A\\H","text":"col1\tcol') == {"path": "A\\H", "text": "col1\tcol"}


def test_streaming_garbage_returns_empty_dict():
    assert parse_streaming_json("not json at all") == {}


# Mirror of pi's json-parse.test.ts (parseStreamingJson).


def test_streaming_keeps_invalid_unicode_escapes_as_literal_text_without_dropping_fields():
    # pi #10868
    assert parse_streaming_json(r'{"text":"\url{x}"}') == {"text": "\\url{x}"}
    assert parse_streaming_json(r'{"path":"a.tex","content":"\usepackage{x}","mode":"w"}') == {
        "path": "a.tex",
        "content": "\\usepackage{x}",
        "mode": "w",
    }


def test_streaming_decodes_valid_unicode_escapes():
    # The \u escape for "é", spelled in two pieces.
    assert parse_streaming_json(r'{"text":"caf\u' + r'00e9 \url"}') == {"text": "café \\url"}


def test_streaming_omits_a_unicode_escape_cut_off_mid_stream():
    assert parse_streaming_json(r'{"text":"ab\u00') == {"text": "ab"}
