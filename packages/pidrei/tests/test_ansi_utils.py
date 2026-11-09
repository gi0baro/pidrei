"""Mirror of pi coding-agent test/ansi-utils.test.ts.

The non-string values are Python's (`None`, an int, a dict), and the TypeError
names the Python type, as `strip_ansi` does.
"""

import re

import pytest

from pidrei.utils.ansi import split_incomplete_ansi_suffix, strip_ansi


def _reference_ansi_regex() -> re.Pattern[str]:
    st = r"(?:\x07|\x1b\\|\x9c)"
    osc = rf"(?:\x1b\][\s\S]*?{st})"
    csi = r"[\x1b\x9b][\[\]()#;?]*(?:\d{1,4}(?:[;:]\d{0,4})*)?[\dA-PR-TZcf-nq-uy=><~]"
    return re.compile(f"{osc}|{csi}")


_REFERENCE_RE = _reference_ansi_regex()


def _reference_strip_ansi(value: str) -> str:
    if "\x1b" not in value and "\x9b" not in value:
        return value
    return _REFERENCE_RE.sub("", value)


def _compatibility_inputs() -> list[str]:
    inputs = [
        "plain",
        "a\x1b[31mred\x1b[0mz",
        "a\x1b]8;;https://example.com\x07link\x1b]8;;\x07z",
        "a\x1b]unterminated",
        "a\x1b]funterminated",
        "a\x1bPabc\x1b\\z",
        "a\x1b^abc\x07z",
        "a\x1b_abc\x9cz",
        "a\x90abc\x9cz",
        "a\x9dabc\x9cz",
        "a\x9b31mred",
        "a\x1b(0x",
        "a\x1b*0x",
        "a\x1b+c",
        "a\x1b/0x",
        "a\x1bcok",
        "a\x1b\\ok",
    ]
    chars = [
        "a", "f", "0", "1", ";", ":", "[", "]", "(", ")", "#", "?", "m", "P", "_", "\\",
        "\x07", "\x1b", "\x9b", "\x9c", "\x90", "\x9d",
    ]  # fmt: skip
    for char in chars:
        inputs.append(f"x\x1b{char}y")
        inputs.append(f"x\x9b{char}y")
        for index in range(0, len(chars), 3):
            inputs.append(f"x\x1b{char}{chars[index]}y")
    return inputs


class TestStripAnsi:
    def test_matches_chalk_strip_ansi_for_generated_compatibility_inputs(self):
        for value in _compatibility_inputs():
            assert strip_ansi(value) == _reference_strip_ansi(value), repr(value)

    def test_throws_the_same_type_error_as_chalk_strip_ansi_for_non_string_values(self):
        for value in (None, 123, {}):
            with pytest.raises(TypeError, match=f"Expected a `string`, got `{type(value).__name__}`"):
                strip_ansi(value)

    def test_strips_ris_without_leaking_the_final_byte(self):
        assert strip_ansi("\x1bcdone") == "done"

    def test_strips_single_byte_esc_sequences_without_leaking_final_bytes(self):
        for code in [*range(ord("g"), ord("m") + 1), *range(ord("r"), ord("t") + 1)]:
            assert strip_ansi(f"\x1b{chr(code)}ok") == "ok"

    def test_strips_common_ansi_sequences_used_in_tool_output(self):
        value = "a\x1b[31mred\x1b[0m\x1b]8;;https://example.com\x07link\x1b]8;;\x07z"
        assert strip_ansi(value) == "aredlinkz"

    # https://github.com/earendil-works/pi/issues/11377
    def test_strips_private_parameter_csi_sequences_with_their_parameters(self):
        assert strip_ansi("\x1b[<35;59;1Mstep 1") == "step 1"
        assert strip_ansi("a\x1b[<0;10;20mb") == "ab"
        assert strip_ansi("a\x1b[>0;276;0cb") == "ab"
        assert strip_ansi("a\x1b[>cb") == "ab"
        assert strip_ansi("a\x1b[=5hb") == "ab"
        assert strip_ansi("a\x9b<35;59;1Mb") == "ab"

    # https://github.com/earendil-works/pi/issues/11377
    def test_keeps_digits_after_bare_esc_keypad_mode_sequences(self):
        assert strip_ansi("\x1b[?1h\x1b=123 tests passed") == "123 tests passed"
        assert strip_ansi("\x1b[?1l\x1b>42 files") == "42 files"
        assert strip_ansi("\x1b<10 items") == "10 items"


class TestSplitIncompleteAnsiSuffix:
    # https://github.com/earendil-works/pi/issues/11377
    def test_holds_back_unfinished_private_parameter_csi_sequences(self):
        assert split_incomplete_ansi_suffix("x\x1b[<35;59") == ("x", "\x1b[<35;59")
        assert split_incomplete_ansi_suffix("x\x1b=") == ("x\x1b=", "")
