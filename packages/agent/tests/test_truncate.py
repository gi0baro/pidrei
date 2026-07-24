"""Mirror of pi agent/test/harness/truncate.test.ts."""

from pidrei_agent.harness.utils.truncate import truncate_head, truncate_tail


def _is_high_surrogate(code: int) -> bool:
    return 0xD800 <= code <= 0xDBFF


def _is_low_surrogate(code: int) -> bool:
    return 0xDC00 <= code <= 0xDFFF


def buffer_utf8(content: str) -> bytes:
    """Mirror of JS `Buffer.from(content, "utf8")`.

    Adjacent surrogate chars encode as one code point; unpaired surrogates
    encode as U+FFFD.
    """
    output = bytearray()
    index = 0
    size = len(content)
    while index < size:
        code = ord(content[index])
        if _is_high_surrogate(code) and index + 1 < size and _is_low_surrogate(ord(content[index + 1])):
            combined = 0x10000 + ((code - 0xD800) << 10) + (ord(content[index + 1]) - 0xDC00)
            output += chr(combined).encode("utf-8")
            index += 2
            continue
        if 0xD800 <= code <= 0xDFFF:
            output += "�".encode()
        else:
            output += content[index].encode("utf-8")
        index += 1
    return bytes(output)


def utf16_canonical(content: str) -> str:
    """Combine adjacent surrogate chars into astral chars (JS string equality)."""
    output: list[str] = []
    index = 0
    size = len(content)
    while index < size:
        code = ord(content[index])
        if _is_high_surrogate(code) and index + 1 < size and _is_low_surrogate(ord(content[index + 1])):
            output.append(chr(0x10000 + ((code - 0xD800) << 10) + (ord(content[index + 1]) - 0xDC00)))
            index += 2
            continue
        output.append(content[index])
        index += 1
    return "".join(output)


def buffer_tail(content: str, max_bytes: int) -> str:
    """Mirror of the TS test's Buffer-based tail truncation."""
    encoded = buffer_utf8(content)
    if len(encoded) <= max_bytes:
        return utf16_canonical(content)
    start = len(encoded) - max_bytes
    while start < len(encoded) and (encoded[start] & 0xC0) == 0x80:
        start += 1
    return encoded[start:].decode("utf-8")


def assert_matches_buffer_tail(content: str, max_byte_values: list[int] | None = None) -> None:
    total_bytes = len(buffer_utf8(content))
    values = max_byte_values if max_byte_values is not None else list(range(total_bytes + 5))
    for max_bytes in values:
        result = truncate_tail(content, max_bytes=max_bytes, max_lines=10)
        expected = buffer_tail(content, max_bytes)
        actual = utf16_canonical(result.content)
        assert actual == expected, f"tail mismatch input={content!r} max_bytes={max_bytes} {expected!r} != {actual!r}"
        output_bytes = len(buffer_utf8(result.content))
        assert output_bytes <= max(max_bytes, len(buffer_utf8(content))), (
            f"tail output exceeded byte limit input={content!r} max_bytes={max_bytes} output_bytes={output_bytes}"
        )


def sampled_byte_limits(content: str) -> list[int]:
    total_bytes = len(buffer_utf8(content))
    candidates = [
        0,
        1,
        2,
        3,
        4,
        5,
        8,
        total_bytes // 2 - 1,
        total_bytes // 2,
        total_bytes // 2 + 1,
        total_bytes - 8,
        total_bytes - 5,
        total_bytes - 4,
        total_bytes - 3,
        total_bytes - 2,
        total_bytes - 1,
        total_bytes,
        total_bytes + 1,
        total_bytes + 4,
    ]
    return sorted({value for value in candidates if value >= 0})


def test_counts_utf8_bytes_without_node_buffer():
    content = "aé🙂\nb"
    result = truncate_head(content, max_bytes=100, max_lines=10)

    assert result.truncated is False
    assert result.total_bytes == len(buffer_utf8(content))
    assert result.output_bytes == len(buffer_utf8(content))
    assert result.total_bytes == 9


def test_does_not_count_trailing_newline_as_extra_line():
    content = "\n".join("line" for _ in range(3)) + "\n"
    head = truncate_head(content, max_bytes=100, max_lines=3)
    tail = truncate_tail(content, max_bytes=100, max_lines=3)

    assert (head.truncated, head.total_lines, head.output_lines) == (False, 3, 3)
    assert (tail.truncated, tail.total_lines, tail.output_lines) == (False, 3, 3)


def test_truncates_head_on_utf8_byte_limits_without_partial_lines():
    result = truncate_head("éé\nabc", max_bytes=4, max_lines=10)

    assert result.content == "éé"
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert result.output_bytes == 4
    assert result.first_line_exceeds_limit is False


def test_reports_head_truncation_when_first_line_exceeds_byte_limit():
    result = truncate_head("éé\nabc", max_bytes=3, max_lines=10)

    assert result.content == ""
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert result.first_line_exceeds_limit is True


def test_truncates_tail_on_utf8_boundaries_when_only_partial_last_line_fits():
    result = truncate_tail("aé🙂b", max_bytes=5, max_lines=10)

    assert result.content == "🙂b"
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert result.last_line_partial is True
    assert result.output_bytes == 5


def test_truncates_oversized_single_line_with_trailing_newline():
    content = "X" * 300_000 + "\n"
    result = truncate_tail(content, max_bytes=1024, max_lines=100)

    assert result.content == "X" * 1024
    assert result.output_bytes == 1024
    assert result.output_lines == 1
    assert result.last_line_partial is True
    assert result.truncated_by == "bytes"


def test_drops_oversized_trailing_character_when_it_cannot_fit_in_tail_byte_limit():
    result = truncate_tail("abc🙂", max_bytes=3, max_lines=10)

    assert result.content == ""
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert result.last_line_partial is True
    assert result.output_bytes == 0


def test_matches_buffer_tail_semantics_for_surrogate_edge_cases():
    inputs = ["a\ud83d", "\ude42b", "a\ude42b", "\ud83d🙂", "🙂\ude42", "👩‍💻"]
    for content in inputs:
        assert_matches_buffer_tail(content)


def test_matches_buffer_tail_semantics_across_deterministic_fuzz_cases():
    alphabet = [
        "a",
        "",
        "",
        "é",
        "߿",
        "ࠀ",
        "中",
        "퟿",
        "\ud800",
        "\ud83d",
        "\udc00",
        "\ude42",
        "🙂",
        "",
        "￿",
    ]

    def check_exhaustive(prefix: str, depth: int) -> None:
        assert_matches_buffer_tail(prefix, sampled_byte_limits(prefix))
        if depth == 0:
            return
        for character in alphabet:
            check_exhaustive(prefix + character, depth - 1)

    check_exhaustive("", 3)

    seed = 0x12345678

    def rand() -> float:
        nonlocal seed
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        return seed / 0x100000000

    for _ in range(1_000):
        length = int(rand() * 80)
        content = "".join(alphabet[int(rand() * len(alphabet))] for _ in range(length))
        assert_matches_buffer_tail(content, sampled_byte_limits(content))
