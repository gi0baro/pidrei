"""Mirror of pi's streaming-json.test.ts (#9265: streamed tool-call arguments
were re-parsed in full on every delta).

Where it differs:

- "matches today's parseStreamingJson at every prefix of compact documents"
  is not mirrored: PiDrei no longer has `parse_streaming_json` (no caller
  remained once every adapter switched, so it went with `partial-json-parser`).
  The reference reading below checks the same rules at every prefix. The
  repair case compares complete text with `parse_json_with_repair`, which is
  what `parseStreamingJson` returns for complete text.
- pi's cost tests measure V8 heap allocation (its `memory-helpers.ts`); these
  measure the thread's CPU time, best of three runs, for a document ten times
  larger than the other: linear growth is about 10x, quadratic about 100x.
- Not mirrored, JavaScript-specific: `__proto__` as an own member (a dict has
  no prototype) and "is collected with its snapshots once the stream is
  dropped" (dicts cannot be weakly referenced; CPython frees them by reference
  count, and the stream holds no registry).
- Comparisons tell `True` from `1`, as Vitest's `toEqual` does
  (`assert_json_equal`).
"""

import base64
import copy
import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from pidrei_ai.utils.json_parse import parse_json_with_repair, parse_tool_call_arguments
from pidrei_ai.utils.streaming_json import (
    NO_VALUE,
    JsonAppend,
    JsonChange,
    JsonSet,
    StreamingJsonParser,
    ToolCallArgumentsStream,
)


type Random = Callable[[], float]


def random(seed: int) -> Random:
    """Seeded random numbers, so every run checks the same documents."""
    state = seed

    def next_random() -> float:
        nonlocal state
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF

    return next_random


def pick(rand: Random, items: list) -> Any:
    return items[min(int(rand() * len(items)), len(items) - 1)]


STRING_PIECES = ["a", "b", "x y", " ", "\n", "\t", "\\", '"', "/", "é", "😀", "\u0001", "\u001f", " ", "{", "]", ","]


def random_string(rand: Random) -> str:
    return "".join(pick(rand, STRING_PIECES) for _ in range(int(rand() * 6)))


def random_scalar(rand: Random) -> Any:
    return pick(
        rand,
        [
            lambda: random_string(rand),
            lambda: int(rand() * 2000) - 1000,
            lambda: rand() * 100,
            lambda: 1e21,
            lambda: -5e-8,
            lambda: -0.0,
            lambda: 0,
            lambda: True,
            lambda: False,
            lambda: None,
        ],
    )()


def random_value(rand: Random, depth: int) -> Any:
    roll = rand()
    if depth > 3 or roll < 0.3:
        return random_scalar(rand)
    if roll < 0.65:
        members = {}
        for _ in range(int(rand() * 4)):
            members[random_string(rand)] = random_value(rand, depth + 1)
        return members
    return [random_value(rand, depth + 1) for _ in range(int(rand() * 4))]


def render(value: Any, rand: Random, spaced: bool) -> str:
    """JSON text of `value`, with random insignificant whitespace when `spaced`."""

    def space() -> str:
        return pick(rand, ["", "", " ", "\n  ", "\t", "\r\n"]) if spaced else ""

    if isinstance(value, list):
        return f"[{space()}{f'{space()},{space()}'.join(render(item, rand, spaced) for item in value)}{space()}]"
    if isinstance(value, dict):
        members = [
            f"{json.dumps(key, ensure_ascii=False)}{space()}:{space()}{render(item, rand, spaced)}"
            for key, item in value.items()
        ]
        return f"{{{space()}{f'{space()},{space()}'.join(members)}{space()}}}"
    return json.dumps(value, ensure_ascii=False)


def random_documents(seed: int, count: int, spaced: bool) -> list[str]:
    rand = random(seed)
    documents = []
    for _ in range(count):
        value = {"path": random_string(rand), "args": random_value(rand, 0)} if rand() < 0.8 else random_value(rand, 0)
        documents.append(render(value, rand, spaced))
    return documents


def _tagged(value: Any) -> Any:
    """`value` with booleans tagged, so `==` tells `True` from `1` as Vitest's `toEqual` does."""
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, dict):
        return {key: _tagged(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tagged(item) for item in value]
    return value


def assert_json_equal(actual: Any, expected: Any, label: str = "") -> None:
    assert _tagged(actual) == _tagged(expected), label


# --- the reference ------------------------------------------------------------

# Whether the reference shows unterminated numbers and literals, as snapshots
# do; changes leave them out.
_pending_scalars = True


def reference_snapshot(text: str, *, pending_scalars: bool = True) -> Any:
    """Reference for `StreamingJsonParser` on a prefix of valid JSON (strings may
    also hold raw control characters and invalid escapes): a plain
    recursive-descent reading of the whole prefix that follows the documented
    rules. It is written independently of the incremental parser so the two
    check each other."""
    global _pending_scalars
    _pending_scalars = pending_scalars
    read = _read_value(text, _skip_space(text, 0))
    return NO_VALUE if read is None else read.value


@dataclass(frozen=True)
class Read:
    value: Any
    complete: bool
    end: int


_SPACE = " \t\n\r"
_SIMPLE = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
_HEX = re.compile(r"[0-9a-fA-F]")


def _skip_space(text: str, start: int) -> int:
    index = start
    while index < len(text) and text[index] in _SPACE:
        index += 1
    return index


def _read_value(text: str, start: int) -> Read | None:
    if start >= len(text):
        return None
    char = text[start]
    if char == '"':
        return _read_string(text, start)
    if char == "{":
        return _read_object(text, start)
    if char == "[":
        return _read_array(text, start)
    return _read_scalar(text, start)


def _join_surrogates(text: str) -> str:
    # Decoded `\uXXXX` escapes are UTF-16 units: a pair is one character, a lone half stays.
    return text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")


def _read_string(text: str, start: int) -> Read:
    value: list[str] = []
    index = start + 1

    def read(complete: bool, end: int) -> Read:
        return Read(_join_surrogates("".join(value)), complete, end)

    while index < len(text):
        char = text[index]
        if char == '"':
            return read(True, index + 1)
        if char != "\\":
            value.append(char)
            index += 1
            continue
        if index + 1 >= len(text):
            return read(False, len(text))
        following = text[index + 1]
        if following in _SIMPLE:
            value.append(_SIMPLE[following])
            index += 2
            continue
        if following == "u":
            end = index + 2
            while end < index + 6 and end < len(text) and _HEX.fullmatch(text[end]):
                end += 1
            if end == index + 6:
                value.append(chr(int(text[index + 2 : end], 16)))
                index = end
                continue
            if end == len(text):
                return read(False, end)
            # Not four hex digits: the escape stays literal and the next character is read normally.
            value.append(text[index:end])
            index = end
            continue
        # Invalid escape: the backslash stays literal and the next character is read normally.
        value.append("\\")
        index += 1
    return read(False, len(text))


def _read_object(text: str, start: int) -> Read:
    members: dict = {}
    partial = Read(members, False, len(text))
    index = _skip_space(text, start + 1)
    if index >= len(text):
        return partial
    if text[index] == "}":
        return Read(members, True, index + 1)
    while True:
        if text[index] != '"':
            raise AssertionError(f"reference reads valid JSON only: {text!r}")
        key = _read_string(text, index)
        if not key.complete:
            return partial
        index = _skip_space(text, key.end)
        if index >= len(text):
            return partial
        index = _skip_space(text, index + 1)
        member = _read_value(text, index)
        if member is None:
            return partial
        members[key.value] = member.value
        if not member.complete:
            return partial
        index = _skip_space(text, member.end)
        if index >= len(text):
            return partial
        if text[index] == "}":
            return Read(members, True, index + 1)
        index = _skip_space(text, index + 1)
        if index >= len(text):
            return partial


def _read_array(text: str, start: int) -> Read:
    elements: list = []
    partial = Read(elements, False, len(text))
    index = _skip_space(text, start + 1)
    if index >= len(text):
        return partial
    if text[index] == "]":
        return Read(elements, True, index + 1)
    while True:
        element = _read_value(text, index)
        if element is None:
            return partial
        elements.append(element.value)
        if not element.complete:
            return partial
        index = _skip_space(text, element.end)
        if index >= len(text):
            return partial
        if text[index] == "]":
            return Read(elements, True, index + 1)
        index = _skip_space(text, index + 1)
        if index >= len(text):
            return partial


def _read_scalar(text: str, start: int) -> Read | None:
    end = start
    while end < len(text) and text[end] not in f'{_SPACE},:{{}}[]"':
        end += 1
    token = text[start:end]
    if end < len(text):
        return Read(json.loads(token), True, end)
    if not _pending_scalars:
        return None
    # At the end of the text: a literal prefix, a number, or a number whose exponent has no digits yet.
    for literal, value in (("true", True), ("false", False), ("null", None)):
        if literal.startswith(token):
            return Read(value, False, end)
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", token):
        return Read(json.loads(token), False, end)
    mantissa = re.fullmatch(r"(-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?)[eE][+-]?", token)
    return None if mantissa is None else Read(json.loads(mantissa[1]), False, end)


# --- streaming helpers --------------------------------------------------------


@dataclass(frozen=True)
class Step:
    read: str
    snapshot: Any
    copy: Any


def stream(text: str, chunk_size: int | Random, *, copies: bool = False) -> list[Step]:
    """Feed `text` in pieces; returns the snapshot after each piece, a copy taken
    right then when `copies`, and the text read by then."""
    parser = StreamingJsonParser()
    steps: list[Step] = []
    index = 0
    while index < len(text):
        size = chunk_size if isinstance(chunk_size, int) else 1 + int(chunk_size() * 12)
        parser.append(text[index : index + size])
        index += size
        snapshot = parser.snapshot()
        steps.append(Step(text[:index], snapshot, copy.deepcopy(snapshot) if copies else None))
    return steps


def final_snapshot(text: str, chunk_size: int | None = None) -> Any:
    parser = StreamingJsonParser()
    size = chunk_size or len(text) or 1
    for index in range(0, len(text), size):
        parser.append(text[index : index + size])
    return parser.snapshot()


CHUNK_SIZES: list[int | str] = [1, 2, 3, 7, "random"]


def read32(text: str) -> int:
    """A stable seed from the text, so random chunk sizes differ per document."""
    value = 7
    for char in text:
        value = (value * 31 + ord(char)) & 0xFFFFFFFF
    return value + 1


def chunks_for(text: str, size: int | str) -> int | Random:
    return random(read32(text)) if size == "random" else size  # type: ignore[return-value]


# --- differential fuzzing -----------------------------------------------------


def test_matches_the_reference_after_every_piece_at_every_chunk_size_with_insignificant_whitespace():
    for text in random_documents(1, 250, True):
        for size in CHUNK_SIZES:
            for step in stream(text, chunks_for(text, size)):
                assert_json_equal(step.snapshot, reference_snapshot(step.read), f"{step.read!r} (chunk size {size})")


def test_reads_complete_documents_exactly_like_json_loads():
    for text in [*random_documents(3, 300, True), *random_documents(4, 300, False)]:
        expected = json.loads(text)
        for size in CHUNK_SIZES:
            assert_json_equal(final_snapshot(text, 5 if size == "random" else size), expected, text)


def test_never_changes_a_snapshot_once_it_is_returned():
    for text in random_documents(5, 150, True):
        for step in stream(text, random(read32(text)), copies=True):
            assert_json_equal(step.snapshot, step.copy)


def test_repairs_raw_control_characters_and_invalid_escapes_and_shows_them_while_streaming():
    rand = random(6)
    malformed = ["\n", "\t", "\u0000", "\\d", "\\x41", "\\u12", "\\uZZZZ", "\\u00e9", "\\\\", "\\q"]
    for round_ in range(300):
        content = "".join(pick(rand, [*malformed, "a", " ", "é"]) for _ in range(6))
        text = f'{{"path":"p","content":"{content}","n":{round_}}}'
        # Complete text: the same repaired value as the finished parse.
        assert_json_equal(final_snapshot(text, 1), parse_json_with_repair(text), repr(text))
        for step in stream(text, 1):
            assert_json_equal(step.snapshot, reference_snapshot(step.read), repr(step.read))


# --- strings ------------------------------------------------------------------


def test_holds_an_escape_back_until_it_is_complete_at_every_split_position():
    for text, value in [
        (r'{"a":"xéy"}', "xéy"),
        (r'{"a":"x\ny"}', "x\ny"),
        (r'{"a":"x\\y"}', "x\\y"),
        (r'{"a":"x\"y"}', 'x"y'),
        (r'{"a":"x\/y"}', "x/y"),
    ]:
        for split in range(len(text) + 1):
            parser = StreamingJsonParser()
            parser.append(text[:split])
            assert_json_equal(parser.snapshot(), reference_snapshot(text[:split]), text[:split])
            parser.append(text[split:])
            assert parser.snapshot() == {"a": value}


def test_joins_surrogate_pairs_split_across_pieces_escaped_or_raw():
    escaped = r'{"a":"😀!"}'
    raw = '{"a":"😀!"}'
    for text in (escaped, raw):
        for split in range(len(text) + 1):
            parser = StreamingJsonParser()
            parser.append(text[:split])
            parser.snapshot()
            parser.append(text[split:])
            assert parser.snapshot() == {"a": "😀!"}
    # A high surrogate whose low half has not arrived shows alone, like json.loads of the escape.
    parser = StreamingJsonParser()
    parser.append(r'{"a":"\ud83d')
    assert parser.snapshot() == {"a": "\ud83d"}


def test_keeps_trailing_whitespace_of_an_unterminated_string():
    parser = StreamingJsonParser()
    parser.append('{"code":"x\t\t  ')
    assert parser.snapshot() == {"code": "x\t\t  "}


def test_keeps_raw_control_characters_and_invalid_escapes_visible_while_streaming():
    parser = StreamingJsonParser()
    parser.append('{"a":"x\ny \\d \\u12"')
    assert parser.snapshot() == {"a": "x\ny \\d \\u12"}


def test_keeps_invalid_u_escapes_literal_without_dropping_fields():
    # #10868
    assert final_snapshot(r'{"path":"a.tex","content":"\usepackage{x}","mode":"w"}', 1) == {
        "path": "a.tex",
        "content": "\\usepackage{x}",
        "mode": "w",
    }
    assert final_snapshot(r'{"text":"café \url"}', 3) == {"text": "café \\url"}
    parser = StreamingJsonParser()
    parser.append(r'{"text":"ab\u00')
    assert parser.snapshot() == {"text": "ab"}


def test_keeps_the_last_value_of_a_duplicate_key_in_the_first_keys_position_as_json_loads_does():
    text = '{"a":1,"b":2,"a":"three"}'
    assert list(final_snapshot(text, 1)) == list(json.loads(text))
    assert final_snapshot(text, 1) == json.loads(text)


# --- ported test vectors ------------------------------------------------------

# partial-json (MIT, https://github.com/promplate/partial-json-parser-js/blob/main/tests/examples.test.js), with
# every partial type allowed, as parseStreamingJson uses it.
PARTIAL_JSON_VECTORS: list[tuple[str, Any]] = [
    ('"', ""),
    ('["', [""]),
    ('{"": "', {"": ""}),
    ("n", None),
    ("t", True),
    ("f", False),
    ("0", 0),
    ("-1.25e+4", -1.25e4),
    ("-1.25e+", -1.25),
    ("-1.25e", -1.25),
    # Differences, documented in streaming_json.py: an invalid escape stays visible (partial-json cuts the string
    # at the backslash), and `Infinity`/`NaN` words are not JSON and are dropped.
    ('" \\x12', " \\x12"),
    ("I", NO_VALUE),
    ("-I", NO_VALUE),
    ("N", NO_VALUE),
]

# Vercel AI SDK fixJson (Apache-2.0, https://github.com/vercel/ai/blob/main/packages/ai/src/util/fix-json.test.ts):
# the expected values are JSON.parse() of its repaired texts.
FIX_JSON_VECTORS: list[tuple[str, Any]] = [
    ("", NO_VALUE),
    ("nul", None),
    ("t", True),
    ("fals", False),
    ("12.2", 12.2),
    ("-12", -12),
    ("-", NO_VALUE),
    ("2.5e", 2.5),
    ("2.5e-", 2.5),
    ("2.5e3", 2.5e3),
    ("-2.5e3", -2.5e3),
    ("2.5E", 2.5),
    ("2.5E-", 2.5),
    ("2.5E3", 2.5e3),
    ("-2.5E3", -2.5e3),
    ("12.34e", 12.34),
    ("5e", 5),
    ('"abc', "abc"),
    ('"value with \\"quoted\\" text and \\\\ escape', 'value with "quoted" text and \\ escape'),
    ('"value with \\', "value with "),
    ('"\\u', ""),
    ('"\\u12', ""),
    ('"text \\u00', "text "),
    ('{"a":"\\u12', {"a": ""}),
    ('"value with unicode <"', "value with unicode <"),
    ("[", []),
    ("[[1], [2", [[1], [2]]),
    ('[["1"], ["2', [["1"], ["2"]]),
    ("[[false], [nu", [[False], [None]]),
    ("[[[]], [[]", [[[]], [[]]]),
    ("[[{}], [{", [[{}], [{}]]),
    ("[1, ", [1]),
    ("[[], 123", [[], 123]),
    ('{"key":', {}),
    ('{"a": {"b": 1}, "c": {"d": 2', {"a": {"b": 1}, "c": {"d": 2}}),
    ('{"a": {"b": "1"}, "c": {"d": 2', {"a": {"b": "1"}, "c": {"d": 2}}),
    ('{"a": {"b": false}, "c": {"d": 2', {"a": {"b": False}, "c": {"d": 2}}),
    ('{"a": {"b": []}, "c": {"d": 2', {"a": {"b": []}, "c": {"d": 2}}),
    ('{"a": {"b": {}}, "c": {"d": 2', {"a": {"b": {}}, "c": {"d": 2}}),
    ('{"ke', {}),
    ('{"k1": 1, "k2', {"k1": 1}),
    ('{"k1": 1, "k2":', {"k1": 1}),
    ('{"key": "value"  ', {"key": "value"}),
    ('{"a": {"b": {}', {"a": {"b": {}}}),
    ("[1, [2, 3, [", [1, [2, 3, []]]),
    ("[false, [true, [", [False, [True, []]]),
    ('{"key": {"subKey":', {"key": {}}),
    ('{"key": 123, "key2": {"subKey":', {"key": 123, "key2": {}}),
    ('{"key": null, "key2": {"subKey":', {"key": None, "key2": {}}),
    ('{"key": [1, 2, {', {"key": [1, 2, {}]}),
    ('[1, 2, {"key": "value",', [1, 2, {"key": "value"}]),
    ('{"a": {"b": ["c", {"d": "e",', {"a": {"b": ["c", {"d": "e"}]}}),
    ('{"a": {"b": {"c": {"d":', {"a": {"b": {"c": {}}}}),
    ('{"a": 1, "b": [', {"a": 1, "b": []}),
    ('{"a": 1, "b": {', {"a": 1, "b": {}}),
    ('{"a": 1, "b": "', {"a": 1, "b": ""}),
    (
        (
            '{\n  "a": [\n    {\n      "a1": "v1",\n      "a2": "v2",\n      "a3": "v3"\n    }\n  ],\n'
            '  "b": [\n    {\n      "b1": "n'
        ),
        {"a": [{"a1": "v1", "a2": "v2", "a3": "v3"}], "b": [{"b1": "n"}]},
    ),
    (
        '{"type":"div","children":[{"type":"Card","props":{}',
        {"type": "div", "children": [{"type": "Card", "props": {}}]},
    ),
    # Difference: fixJson completes `12.` to 12; like parseStreamingJson, a number cut after its dot is omitted.
    ("12.", NO_VALUE),
    ("12.e", NO_VALUE),
]


@pytest.mark.parametrize(("text", "expected"), [*PARTIAL_JSON_VECTORS, *FIX_JSON_VECTORS])
def test_reads_vector(text, expected):
    for size in (len(text) or 1, 1, 2):
        assert_json_equal(final_snapshot(text, size), expected)


def _json_parse(text: str) -> Any:
    """`JSON.parse`: rejects what `json.loads` accepts beyond JSON (`NaN`, `Infinity`)."""

    def reject(constant: str) -> Any:
        raise ValueError(constant)

    return json.loads(text, parse_constant=reject)


def expect_json(value: Any, label: str) -> None:
    """Asserts `value` is plain, finite JSON data."""
    pending = [value]
    while pending:
        item = pending.pop()
        if item is NO_VALUE or item is None or isinstance(item, str | bool | int):
            continue
        if isinstance(item, float):
            assert math.isfinite(item), label
        elif isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        else:
            pytest.fail(f"{label}: not JSON: {item!r}")


def test_reads_every_json_test_suite_document_json_loads_accepts_exactly_like_it_and_never_raises_on_the_rest():
    suite = json.loads((Path(__file__).parent / "json-test-suite.json").read_text())
    for name, encoded in suite["files"].items():
        text = base64.b64decode(encoded).decode("utf-8", errors="replace")
        try:
            expected = _json_parse(text)
        except ValueError, RecursionError:
            expected = NO_VALUE
        for size in (len(text) or 1, 1, 3):
            snapshot = final_snapshot(text, size)
            if expected is not NO_VALUE:
                assert_json_equal(snapshot, expected, name)
            else:
                expect_json(snapshot, name)


# --- malformed text -----------------------------------------------------------

MALFORMED = [
    '{"a":1 "b":2}',
    '{"a" 1, "b": 2}',
    "{a:1, 'b': 2}",
    '{"a":1,,"b":2,}',
    "[1 2, , 3,]",
    '{"a":[1,2}, "b":3]',
    "]]]}}}{",
    '{"a":NaN,"b":Infinity,"c":-Infinity,"d":undefined,"e":1}',
    '{"a":01,"b":1.,"c":.5,"d":+1,"e":0x10,"f":1e}',
    '{"a":"x"} trailing garbage {"b":2}',
    '{"a": tru, "b": nulll, "c": falsey}',
    '{"a":{"b":{"c":',
    '{"a\\": 1}',
    '"unterminated \\',
    '// comment\n{"a":1}',
    '\u0000{\u0001"a"\u0002:\u00031}',
    '{"a":"x"}}}]]]',
]


def test_never_raises_keeps_snapshots_plain_json_and_unchanged_at_every_chunk_size():
    for text in MALFORMED:
        for size in CHUNK_SIZES:
            steps = stream(text, chunks_for(text, size), copies=True)
            for step in steps:
                expect_json(step.snapshot, text)
                assert_json_equal(step.snapshot, step.copy)
            # Every chunk size reads malformed text the same way.
            assert_json_equal(steps[-1].snapshot, final_snapshot(text), text)


def test_tolerates_missing_commas_and_colons_and_drops_values_in_invalid_positions():
    assert final_snapshot('{"a":1 "b":2}') == {"a": 1, "b": 2}
    assert final_snapshot('{"a" 1, "b": 2}') == {"a": 1, "b": 2}
    assert final_snapshot("[1 2, , 3,]") == [1, 2, 3]
    assert final_snapshot('{"a":NaN,"b":Infinity,"e":1}') == {"e": 1}
    assert final_snapshot('{"a":"x"} trailing garbage {"b":2}') == {"a": "x"}


def test_survives_random_corruption_of_valid_documents():
    rand = random(7)
    noise = ['"', "\\", "{", "}", "[", "]", ",", ":", "x", "1", "-", "\n", "t"]
    for text in random_documents(8, 300, True):
        characters = list(text)
        for _ in range(3):
            at = int(rand() * (len(characters) + 1))
            if rand() < 0.5:
                del characters[at : at + 1]
            else:
                characters.insert(at, pick(rand, noise))
        corrupted = "".join(characters)
        for step in stream(corrupted, random(read32(corrupted))):
            expect_json(step.snapshot, corrupted)
        # The changes keep a copy equal to the parser's own value, malformed or not.
        parser = StreamingJsonParser()
        mirror = Mirror()
        for index in range(0, len(corrupted), 4):
            mirror.apply(parser.append(corrupted[index : index + 4]))
        # A space ends a number or literal at the end, which snapshots show and changes do not report yet.
        mirror.apply(parser.append(" "))
        assert_json_equal(mirror.value, parser.snapshot(), corrupted)


# --- changes ------------------------------------------------------------------


class Mirror:
    """A copy of a value kept current by applying the parser's changes, as a writer keeps one."""

    def __init__(self) -> None:
        self.value: Any = NO_VALUE

    def apply(self, changes: list[JsonChange]) -> None:
        for change in changes:
            parent = self.value
            for segment in change.path[:-1]:
                parent = parent[segment]
            key = change.path[-1] if change.path else None
            if isinstance(change, JsonSet):
                value = copy.deepcopy(change.value)
                if key is None:
                    self.value = value
                elif isinstance(parent, list) and key == len(parent):
                    parent.append(value)
                else:
                    parent[key] = value
            elif key is None:
                self.value = f"{self.value}{change.text}"
            else:
                parent[key] = f"{parent[key]}{change.text}"


def mirror_stream(text: str, size: int | Random) -> Mirror:
    """Feed `text` in pieces into a parser and a mirror of its changes; checks the mirror after each piece."""
    parser = StreamingJsonParser()
    mirror = Mirror()
    index = 0
    while index < len(text):
        length = size if isinstance(size, int) else 1 + int(size() * 12)
        mirror.apply(parser.append(text[index : index + length]))
        index += length
        # Changes leave out unterminated numbers and literals, which snapshots show.
        assert_json_equal(mirror.value, reference_snapshot(text[:index], pending_scalars=False), repr(text[:index]))
    # A space ends a root number or literal, which changes do not report while it is unterminated, as in the
    # corruption test.
    mirror.apply(parser.append(" "))
    return mirror


def test_changes_keep_a_copy_equal_to_the_value_read_so_far_at_every_chunk_size_and_to_json_loads_at_the_end():
    for text in random_documents(9, 200, True):
        for size in CHUNK_SIZES:
            mirror = mirror_stream(text, chunks_for(text, size))
            assert_json_equal(mirror.value, json.loads(text), text)


def test_changes_report_a_number_that_ends_and_a_string_that_starts_in_one_piece_in_order():
    parser = StreamingJsonParser()
    assert parser.append('{"n":1') == [JsonSet((), {})]
    assert parser.append('2,"s":"ab') == [JsonSet(("n",), 12), JsonSet(("s",), "ab")]
    assert parser.append('c","t":tru') == [JsonAppend(("s",), "c")]
    assert parser.append("e}") == [JsonSet(("t",), True)]
    assert parser.append(' {"ignored":1}') == []


def test_changes_report_nested_objects_and_arrays_as_empty_containers_then_their_members_and_items():
    parser = StreamingJsonParser()
    assert parser.append('{"a":[1,{"b":["x"') == [
        JsonSet((), {}),
        JsonSet(("a",), []),
        JsonSet(("a", 0), 1),
        JsonSet(("a", 1), {}),
        JsonSet(("a", 1, "b"), []),
        JsonSet(("a", 1, "b", 0), "x"),
    ]
    assert parser.append(',"y') == [JsonSet(("a", 1, "b", 1), "y")]
    assert parser.append('z"]}],"a":"again"}') == [
        JsonAppend(("a", 1, "b", 1), "z"),
        JsonSet(("a",), "again"),
    ]


def test_changes_hold_split_escapes_back_and_report_their_decoded_text_once_complete():
    text = r'{"a":"xé\n\\\"😀\q"}'
    expected = json.loads(r'{"a":"xé\n\\\"😀\\q"}')["a"]
    for first in range(1, len(text)):
        for second in range(first, len(text)):
            parser = StreamingJsonParser()
            mirror = Mirror()
            for piece in (text[:first], text[first:second], text[second:]):
                mirror.apply(parser.append(piece))
            assert mirror.value == {"a": expected}
    parser = StreamingJsonParser()
    parser.append('{"a":"x')
    assert parser.append("\\u00") == []
    assert parser.append("e9") == [JsonAppend(("a",), "é")]


def test_changes_report_a_container_nested_deeper_than_32_levels_once_whole_when_it_closes():
    depth = 40
    text = f'{'{"a":' * depth}"deep"{"}" * depth}'
    parser = StreamingJsonParser()
    mirror = Mirror()
    for index in range(0, len(text), 3):
        mirror.apply(parser.append(text[index : index + 3]))
    assert mirror.value == json.loads(text)
    changes = StreamingJsonParser().append(text)
    # The containers down to path length 32, then the one at length 32 again, whole, as it closes.
    assert len(changes) == 34
    last = changes[-1]
    assert isinstance(last, JsonSet)
    assert len(last.path) == 32
    assert last.value == json.loads(text[32 * 5 : -32])


def test_changes_report_nothing_for_values_in_invalid_positions_and_text_after_the_root():
    parser = StreamingJsonParser()
    assert parser.append('{"a":1 2, "b" "x"}') == [JsonSet((), {}), JsonSet(("a",), 1), JsonSet(("b",), "x")]
    assert parser.append('{"c":3}') == []


# --- cost ---------------------------------------------------------------------

CONTENT = "".join(f'const line{line} = "value {line}"; // \\ done\n' for line in range(20_000))


def write_arguments(size: int) -> str:
    """`write`-style arguments with `size` characters of file content."""
    return json.dumps({"path": "/src/file.ts", "content": CONTENT[:size]})


def cpu_seconds(run: Callable[[], object]) -> float:
    """The calling thread's CPU time for `run`, best of three."""
    best = math.inf
    for _ in range(3):
        started = time.thread_time()
        run()
        best = min(best, time.thread_time() - started)
    return best


def stream_arguments(text: str, delta: int) -> tuple[dict, str | None]:
    args = ToolCallArgumentsStream()
    for index in range(0, len(text), delta):
        args.append(text[index : index + delta])
    return args.finish()


def test_takes_time_linear_in_the_streamed_text_never_per_accumulated_text():
    small = write_arguments(50_000)
    large = write_arguments(500_000)
    assert stream_arguments(large, 32)[0] == json.loads(large)
    small_seconds = cpu_seconds(lambda: stream_arguments(small, 32))
    large_seconds = cpu_seconds(lambda: stream_arguments(large, 32))
    # Ten times the text: linear growth is about 10x, re-reading the accumulated text about 100x.
    assert large_seconds / small_seconds < 20


def test_keeps_very_large_open_arrays_linear_by_letting_their_preview_lag():
    def elements(count: int) -> str:
        return json.dumps({"items": list(range(count))})

    small = elements(10_000)
    large = elements(100_000)
    small_seconds = cpu_seconds(lambda: stream_arguments(small, 16))
    large_seconds = cpu_seconds(lambda: stream_arguments(large, 16))
    assert large_seconds / small_seconds < 20

    parser = StreamingJsonParser()
    shown = 0
    for index in range(0, len(large), 16):
        parser.append(large[index : index + 16])
        snapshot = parser.snapshot()
        items = snapshot.get("items", []) if isinstance(snapshot, dict) else []
        # The preview only moves forward.
        assert len(items) >= shown
        shown = len(items)
    assert parser.snapshot() == json.loads(large)


def test_keeps_a_very_large_open_string_linear_by_letting_its_preview_lag():
    # PiDrei-only: Python strings are not ropes, so showing an open string builds it whole.
    text = write_arguments(500_000)
    content = json.loads(text)["content"]
    parser = StreamingJsonParser()
    shown = 0
    for index in range(0, len(text), 32):
        parser.append(text[index : index + 32])
        snapshot = parser.snapshot()
        partial = snapshot.get("content", "") if isinstance(snapshot, dict) else ""
        assert content.startswith(partial)
        assert len(partial) >= shown
        shown = len(partial)
    assert parser.snapshot() == json.loads(text)


def test_keeps_a_very_long_open_number_linear_by_not_showing_it():
    # PiDrei-only: Python strings are not ropes, so an oversized number is measured before it is joined.
    def stream_number(digits: int) -> Any:
        text = f'{{"x":0.{"7" * digits}}}'
        parser = StreamingJsonParser()
        for index in range(0, len(text), 32):
            parser.append(text[index : index + 32])
            parser.snapshot()
        return parser.snapshot()

    assert stream_number(400_000) == {"x": 0.7777777777777778}
    small_seconds = cpu_seconds(lambda: stream_number(40_000))
    large_seconds = cpu_seconds(lambda: stream_number(400_000))
    assert large_seconds / small_seconds < 20


def test_reads_deep_nesting_without_recursion_and_in_linear_time():
    depth = 200_000
    text = f"{'[' * depth}1{']' * depth}"
    started = time.perf_counter()
    parser = StreamingJsonParser()
    for index in range(0, len(text), 7):
        parser.append(text[index : index + 7])
        parser.snapshot()
    value = parser.snapshot()
    for _ in range(depth):
        value = value[0]
    assert value == 1
    assert time.perf_counter() - started < 5


def test_reads_one_huge_piece_such_as_a_providers_complete_arguments_in_linear_time():
    small = write_arguments(50_000)
    large = write_arguments(500_000)

    def read(text: str) -> Any:
        parser = StreamingJsonParser()
        parser.append(text)
        return parser.snapshot()

    assert read(large) == json.loads(large)
    assert cpu_seconds(lambda: read(large)) / cpu_seconds(lambda: read(small)) < 20


# --- ToolCallArgumentsStream --------------------------------------------------


def test_shows_partial_arguments_as_an_object_and_finishes_with_one_full_parse():
    args = ToolCallArgumentsStream()
    assert args.append("  ") == {}
    assert args.append('{"path":"a.ts","content":"x') == {"path": "a.ts", "content": "x"}
    before = args.append("y")
    assert args.append("") is before
    assert args.text == '  {"path":"a.ts","content":"xy'
    assert args.finish() == parse_tool_call_arguments('  {"path":"a.ts","content":"xy')
    args.append('"}')
    assert args.finish() == ({"path": "a.ts", "content": "xy"}, None)


def test_shows_non_object_roots_as_no_arguments_yet():
    args = ToolCallArgumentsStream()
    assert args.append("[1,2") == {}
    args.append("]")
    _, error = args.finish()
    assert error is not None
    assert error.startswith("Tool arguments must be a JSON object")


def test_replaces_the_text_with_the_providers_complete_text_returning_the_missing_piece():
    args = ToolCallArgumentsStream()
    args.append('{"a":"x')
    assert args.replace('{"a":"xyz"}') == ('yz"}', {"a": "xyz"})
    assert args.replace('{"b":2}') == (None, {"b": 2})
    assert args.text == '{"b":2}'
