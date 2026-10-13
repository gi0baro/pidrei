"""Mirror of pi codemode test/source.test.ts, with `#` options lines."""

import re

import pytest

from pidrei_codemode.source import (
    CodemodeSourceError,
    CodemodeSourceOptions,
    ParsedCodemodeSource,
    parse_codemode_source,
)


def test_returns_plain_code_unchanged():
    assert parse_codemode_source("text('hi')") == ParsedCodemodeSource("text('hi')", CodemodeSourceOptions())
    assert parse_codemode_source("# just a comment\nreturn 1") == ParsedCodemodeSource(
        "# just a comment\nreturn 1", CodemodeSourceOptions()
    )


def test_parses_the_options_line_and_keeps_line_numbers():
    assert parse_codemode_source('# @options: {"timeout_ms": 10}\na = 1\ntext(a)') == ParsedCodemodeSource(
        "\na = 1\ntext(a)", CodemodeSourceOptions(timeout_ms=10)
    )
    assert parse_codemode_source('  # @options:{"max_output_tokens":0,"timeout_ms":1500}\r\ntext(1)').options == (
        CodemodeSourceOptions(max_output_tokens=0, timeout_ms=1500)
    )
    assert parse_codemode_source("# @options: {}\ntext(1)") == ParsedCodemodeSource(
        "\ntext(1)", CodemodeSourceOptions()
    )


def test_only_treats_the_first_line_as_an_options_line():
    assert parse_codemode_source("# @optionsx {}\ntext(1)").options == CodemodeSourceOptions()
    assert parse_codemode_source('text("# @options: {}")').options == CodemodeSourceOptions()


# pi #10631: an options line after the first line used to be ignored, so its timeout never applied.
@pytest.mark.parametrize(
    ("source", "line"),
    [
        ('# count things\n# @options: {"timeout_ms": 1000}\nwhile True: pass', 2),
        ('\n  # @options: {"timeout_ms": 1000}\nwhile True: pass', 2),
        ('text(1)\r\n# @options: {"timeout_ms": 1}', 2),
        ('# @options: {"timeout_ms": 1}\ntext(1)\n# @options: {"timeout_ms": 2}', 3),
    ],
)
def test_rejects_an_options_line_that_is_not_the_first_line(source, line):
    with pytest.raises(
        CodemodeSourceError,
        match=f"^The # @options: line must be the first line of the input; found one on line {line}$",
    ):
        parse_codemode_source(source)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("", re.escape("Expected Python source text (non-empty)")),
        ("  \n", re.escape("Expected Python source text (non-empty)")),
        ("# @options:\ntext(1)", "@options must be a JSON object with supported fields"),
        ("# @options: {timeout_ms: 1}\ntext(1)", "@options must be valid JSON with supported fields"),
        ("# @options: [1]\ntext(1)", "@options must be a JSON object with supported fields"),
        (
            '# @options: {"yield": 1}\ntext(1)',
            re.escape("@options only supports `max_output_tokens` and `timeout_ms`; got `yield`"),
        ),
        (
            '# @options: {"max_output_tokens": 1.5}\ntext(1)',
            re.escape("@options field `max_output_tokens` must be a non-negative safe integer"),
        ),
        ('# @options: {"timeout_ms": 0}\ntext(1)', re.escape("@options field `timeout_ms` must be a positive integer")),
        (
            '# @options: {"timeout_ms": 1}',
            re.escape("The @options line must be followed by Python source on subsequent lines"),
        ),
        (
            '# @options: {"timeout_ms": 1}\n  \n',
            re.escape("The @options line must be followed by Python source on subsequent lines"),
        ),
    ],
)
def test_rejects_empty_input_and_invalid_options(source, message):
    with pytest.raises(CodemodeSourceError, match=message):
        parse_codemode_source(source)
