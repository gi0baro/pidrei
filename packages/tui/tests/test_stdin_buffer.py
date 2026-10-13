"""Mirror of pi tui test/stdin-buffer.test.ts.

pi's upstream note: based on code from OpenTUI
(https://github.com/anomalyco/opentui), MIT License, Copyright (c) 2025
opentui.

pi drives the 10ms flush timeout with mocked timers; here the buffer has no
timer, only a deadline its reader honours with `expire()`. The cases that
wait on it (`_make_manual_buffer`) run the buffer on a `_ManualClock` whose
`tick` advances the time and expires the buffer, as pi's
`vi.advanceTimersByTime` fires the due timer.
"""

import pytest

from pidrei_tui.keys import matches_key
from pidrei_tui.stdin_buffer import StdinBuffer


class _ManualClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.buffer: StdinBuffer | None = None

    def __call__(self) -> float:
        return self.now

    def tick(self, ms: float) -> None:
        self.now += ms / 1000
        self.buffer.expire()


def _make_buffer():
    buffer = StdinBuffer(timeout=10)
    emitted_sequences = []
    buffer.on_data(emitted_sequences.append)
    return buffer, emitted_sequences


def _make_manual_buffer(**options):
    """A buffer whose flush happens only on `timers.tick` (pi's mocked
    timers), recording what it emits."""
    timers = _ManualClock()
    buffer = StdinBuffer(clock=timers, **options)
    timers.buffer = buffer
    emitted_sequences = []
    buffer.on_data(emitted_sequences.append)
    return buffer, emitted_sequences, timers


# Regular Characters


@pytest.mark.tonio
async def test_passes_through_regular_characters_immediately():
    buffer, emitted = _make_buffer()
    buffer.process("a")
    assert emitted == ["a"]


@pytest.mark.tonio
async def test_passes_through_multiple_regular_characters():
    buffer, emitted = _make_buffer()
    buffer.process("abc")
    assert emitted == ["a", "b", "c"]


@pytest.mark.tonio
async def test_handles_unicode_characters():
    buffer, emitted = _make_buffer()
    buffer.process("hello 世界")
    assert emitted == ["h", "e", "l", "l", "o", " ", "世", "界"]


# Complete Escape Sequences


@pytest.mark.tonio
async def test_passes_through_complete_mouse_sgr_sequences():
    buffer, emitted = _make_buffer()
    mouse_seq = "\x1b[<35;20;5m"
    buffer.process(mouse_seq)
    assert emitted == [mouse_seq]


@pytest.mark.tonio
async def test_passes_through_complete_arrow_key_sequences():
    buffer, emitted = _make_buffer()
    up_arrow = "\x1b[A"
    buffer.process(up_arrow)
    assert emitted == [up_arrow]


@pytest.mark.tonio
async def test_passes_through_complete_function_key_sequences():
    buffer, emitted = _make_buffer()
    f1 = "\x1b[11~"
    buffer.process(f1)
    assert emitted == [f1]


@pytest.mark.tonio
async def test_passes_through_meta_key_sequences():
    buffer, emitted = _make_buffer()
    meta_a = "\x1ba"
    buffer.process(meta_a)
    assert emitted == [meta_a]


@pytest.mark.tonio
async def test_passes_through_ss3_sequences():
    buffer, emitted = _make_buffer()
    ss3 = "\x1bOA"
    buffer.process(ss3)
    assert emitted == [ss3]


# Partial Escape Sequences


@pytest.mark.tonio
async def test_buffers_incomplete_mouse_sgr_sequence():
    buffer, emitted = _make_buffer()
    buffer.process("\x1b")
    assert emitted == []
    assert buffer.get_buffer() == "\x1b"

    buffer.process("[<35")
    assert emitted == []
    assert buffer.get_buffer() == "\x1b[<35"

    buffer.process(";20;5m")
    assert emitted == ["\x1b[<35;20;5m"]
    assert buffer.get_buffer() == ""


@pytest.mark.tonio
async def test_buffers_incomplete_csi_sequence():
    buffer, emitted = _make_buffer()
    buffer.process("\x1b[")
    assert emitted == []

    buffer.process("1;")
    assert emitted == []

    buffer.process("5H")
    assert emitted == ["\x1b[1;5H"]


@pytest.mark.tonio
async def test_buffers_split_across_many_chunks():
    buffer, emitted = _make_buffer()
    for chunk in ["\x1b", "[", "<", "3", "5", ";", "2", "0", ";", "5", "m"]:
        buffer.process(chunk)

    assert emitted == ["\x1b[<35;20;5m"]


@pytest.mark.tonio
async def test_flushes_incomplete_sequence_after_timeout():
    buffer, emitted, timers = _make_manual_buffer(timeout=10)
    buffer.process("\x1b[<35")
    assert emitted == []

    # Wait for timeout
    timers.tick(15)

    assert emitted == ["\x1b[<35"]


@pytest.mark.tonio
async def test_flushes_a_lone_esc_as_escape_when_cr_arrives_after_the_timeout():
    # Legacy-mode Alt+Enter is ESC + CR; when the terminal/transport splits
    # the bytes further apart than the timeout, ESC is flushed alone and the
    # host sees Escape (interrupt) instead of Alt+Enter. This locks in the
    # behavior so the configurable timeout in ProcessTerminal stays honest.
    buffer, emitted, timers = _make_manual_buffer(timeout=10)
    buffer.process("\x1b")
    timers.tick(15)
    buffer.process("\r")

    assert emitted == ["\x1b", "\r"]
    assert matches_key(emitted[0], "escape") is True


@pytest.mark.tonio
async def test_merges_esc_plus_cr_split_across_chunks_within_a_larger_escape_timeout():
    buffer, emitted, timers = _make_manual_buffer(escape_timeout=100)

    buffer.process("\x1b")
    timers.tick(20)  # > 10ms default escape timeout, < 100ms configured
    buffer.process("\r")

    assert emitted == ["\x1b\r"]
    assert matches_key(emitted[0], "alt+enter") is True
    buffer.destroy()


@pytest.mark.tonio
async def test_does_not_apply_the_sequence_timeout_to_a_lone_esc():
    buffer, emitted, timers = _make_manual_buffer(timeout=100)

    buffer.process("\x1b")
    timers.tick(20)
    buffer.process("\r")

    assert emitted == ["\x1b", "\r"]
    assert matches_key(emitted[0], "escape") is True
    buffer.destroy()


@pytest.mark.tonio
async def test_keeps_fragmented_mouse_sequences_buffered_across_delayed_chunks_by_default():
    delayed_buffer, delayed_sequences, timers = _make_manual_buffer()

    delayed_buffer.process("\x1b[")
    timers.tick(20)
    assert delayed_sequences == []
    delayed_buffer.process("<65;48;39M")
    assert delayed_sequences == ["\x1b[<65;48;39M"]
    delayed_buffer.destroy()


# Terminal Reply Strings

_REPLIES = [
    ("\x1b]4;1;", "\x1b]4;1;rgb:eaea/6c6c/6d6d\x07"),  # Ghostty's Ayu Light palette color 1.
    ("\x1b]10;", "\x1b]10;rgb:ffff/ffff/ffff\x1b\\"),
    ("\x1b]11;", "\x1b]11;rgb:0000/0000/0000\x07"),
    ("\x1b]4;1;", "\x1b]4;1;rgb:6464/6464/4d4d\x07"),  # Another reported color-reply tail.
    ("\x1bP>|", "\x1bP>|Ghostty 1.3.1\x1b\\"),
    ("\x1b_Gi=1;", "\x1b_Gi=1;OK\x1b\\"),
    ("\x1b]7501;?", "\x1b]7501;?\x1b\\"),
]


@pytest.mark.tonio
@pytest.mark.parametrize(("header", "reply"), _REPLIES)
async def test_keeps_a_delayed_reply_intact_at_every_split_after_its_reply_header(header, reply):
    for split in range(len(header), len(reply)):
        buffer, sequences, timers = _make_manual_buffer()
        buffer.process(reply[:split])
        timers.tick(5000)
        buffer.process(f"{reply[split:]}x")
        assert not any(matches_key(sequence, "ctrl+g") for sequence in sequences), split
        assert sequences == [reply, "x"], split


@pytest.mark.tonio
async def test_keeps_an_osc_reply_buffered_across_multiple_expired_sequence_timeouts():
    buffer, emitted, timers = _make_manual_buffer(timeout=10)
    buffer.process("\x1b]4;1;rgb:eaea/6c")
    timers.tick(100)
    assert emitted == []
    buffer.process("6c/6d6d")
    timers.tick(100)
    assert emitted == []
    buffer.process("\x07")
    assert emitted == [_REPLIES[0][1]]


@pytest.mark.tonio
async def test_still_times_out_bare_string_introducers_as_alt_keys():
    buffer, emitted, timers = _make_manual_buffer(timeout=10)
    for prefix in ("\x1b]", "\x1bP", "\x1b_"):
        buffer.process(prefix)
        timers.tick(10)
    assert emitted == ["\x1b]", "\x1bP", "\x1b_"]


@pytest.mark.tonio
@pytest.mark.parametrize(
    "ambiguous",
    [
        "\x1b]a",
        "\x1b]1",
        "\x1b]10",
        "\x1b]0;title",
        "\x1b]4;",
        "\x1b]4;a;",
        "\x1bPa",
        "\x1bP>",
        "\x1b_a",
        "\x1b_G",
        "\x1b_Gi=1",
    ],
)
async def test_recovers_batched_legacy_alt_input_within_the_sequence_timeout(ambiguous):
    buffer, sequences, timers = _make_manual_buffer()
    buffer.process(ambiguous)
    timers.tick(20)
    buffer.process("x\r\x03\x04")
    timers.tick(49)
    assert sequences == []
    timers.tick(1)
    assert sequences == [ambiguous[:2], *ambiguous[2:], "x", "\r", "\x03", "\x04"]
    assert buffer.get_buffer() == ""

    buffer.process("y\r\x03\x04")
    assert sequences[-4:] == ["y", "\r", "\x03", "\x04"]


@pytest.mark.tonio
async def test_recognizes_a_reply_header_assembled_within_the_ambiguous_input_timeout():
    buffer, emitted, timers = _make_manual_buffer(timeout=10)
    buffer.process("\x1b]1")
    timers.tick(5)
    buffer.process("0;")
    timers.tick(5000)
    assert emitted == []
    buffer.process("rgb:ffff/ffff/ffff\x1b\\")
    assert emitted == [_REPLIES[1][1]]


@pytest.mark.tonio
async def test_recovers_from_an_unterminated_reply_when_a_new_escape_sequence_arrives():
    buffer, emitted, timers = _make_manual_buffer(timeout=10)
    buffer.process("\x1b]4;1;rgb:eaea/6c")
    timers.tick(100)
    buffer.process("\x1b")
    timers.tick(100)
    buffer.process("[Ahello")
    assert emitted == ["\x1b[A", "h", "e", "l", "l", "o"]


@pytest.mark.tonio
async def test_discards_a_canceled_reply_without_swallowing_subsequent_input():
    buffer, emitted = _make_buffer()
    for cancel in ("\x18", "\x1a"):
        buffer.process(f"\x1b]11;rgb:0000{cancel}x")
    assert emitted == ["x", "x"]


# Mixed Content


@pytest.mark.tonio
async def test_handles_characters_followed_by_escape_sequence():
    buffer, emitted = _make_buffer()
    buffer.process("abc\x1b[A")
    assert emitted == ["a", "b", "c", "\x1b[A"]


@pytest.mark.tonio
async def test_handles_escape_sequence_followed_by_characters():
    buffer, emitted = _make_buffer()
    buffer.process("\x1b[Aabc")
    assert emitted == ["\x1b[A", "a", "b", "c"]


@pytest.mark.tonio
async def test_handles_multiple_complete_sequences():
    buffer, emitted = _make_buffer()
    buffer.process("\x1b[A\x1b[B\x1b[C")
    assert emitted == ["\x1b[A", "\x1b[B", "\x1b[C"]


@pytest.mark.tonio
async def test_handles_partial_sequence_with_preceding_characters():
    buffer, emitted = _make_buffer()
    buffer.process("abc\x1b[<35")
    assert emitted == ["a", "b", "c"]
    assert buffer.get_buffer() == "\x1b[<35"

    buffer.process(";20;5m")
    assert emitted == ["a", "b", "c", "\x1b[<35;20;5m"]


# Kitty Keyboard Protocol


@pytest.mark.tonio
async def test_handles_kitty_csi_u_press_events():
    buffer, emitted = _make_buffer()
    buffer.process("\x1b[97u")
    assert emitted == ["\x1b[97u"]


@pytest.mark.tonio
async def test_handles_kitty_csi_u_release_events():
    buffer, emitted = _make_buffer()
    buffer.process("\x1b[97;1:3u")
    assert emitted == ["\x1b[97;1:3u"]


@pytest.mark.tonio
async def test_handles_batched_kitty_press_and_release():
    # Press 'a', release 'a' batched together (common over SSH)
    buffer, emitted = _make_buffer()
    buffer.process("\x1b[97u\x1b[97;1:3u")
    assert emitted == ["\x1b[97u", "\x1b[97;1:3u"]


@pytest.mark.tonio
async def test_handles_multiple_batched_kitty_events():
    # Press 'a', release 'a', press 'b', release 'b'
    buffer, emitted = _make_buffer()
    buffer.process("\x1b[97u\x1b[97;1:3u\x1b[98u\x1b[98;1:3u")
    assert emitted == ["\x1b[97u", "\x1b[97;1:3u", "\x1b[98u", "\x1b[98;1:3u"]


@pytest.mark.tonio
async def test_handles_kitty_arrow_keys_with_event_type():
    # Up arrow press with event type
    buffer, emitted = _make_buffer()
    buffer.process("\x1b[1;1:1A")
    assert emitted == ["\x1b[1;1:1A"]


@pytest.mark.tonio
async def test_handles_kitty_functional_keys_with_event_type():
    # Delete key release
    buffer, emitted = _make_buffer()
    buffer.process("\x1b[3;1:3~")
    assert emitted == ["\x1b[3;1:3~"]


@pytest.mark.tonio
async def test_splits_esc_esc_csi_into_standalone_esc_and_csi_sequence():
    # WezTerm with enable_kitty_keyboard sends Escape key press as raw \x1b
    # and the release as a full Kitty CSI-u sequence, concatenated.
    # The buffer must not treat \x1b\x1b as a complete meta-key when the
    # following byte starts a new escape sequence.
    buffer, emitted = _make_buffer()
    buffer.process("\x1b\x1b[27;129:3u")
    assert emitted == ["\x1b", "\x1b[27;129:3u"]


@pytest.mark.tonio
async def test_splits_esc_esc_csi_with_no_modifier():
    buffer, emitted = _make_buffer()
    buffer.process("\x1b\x1b[27;1:3u")
    assert emitted == ["\x1b", "\x1b[27;1:3u"]


@pytest.mark.tonio
async def test_still_emits_esc_esc_as_single_sequence_when_not_followed_by_new_escape():
    # \x1b\x1b alone (no following CSI) stays as-is — e.g. ctrl+alt+[
    buffer, emitted = _make_buffer()
    buffer.process("\x1b\x1b")
    assert emitted == ["\x1b\x1b"]


@pytest.mark.tonio
async def test_handles_plain_characters_mixed_with_kitty_sequences():
    # Plain 'a' followed by Kitty release
    buffer, emitted = _make_buffer()
    buffer.process("a\x1b[97;1:3u")
    assert emitted == ["a", "\x1b[97;1:3u"]


@pytest.mark.tonio
async def test_drops_raw_duplicate_character_after_matching_kitty_printable_sequence():
    buffer, emitted = _make_buffer()
    buffer.process("\x1b[224uà")
    assert emitted == ["\x1b[224u"]


@pytest.mark.tonio
async def test_drops_raw_duplicate_character_after_kitty_printable_across_chunks():
    buffer, emitted = _make_buffer()
    buffer.process("\x1b[64u")
    buffer.process("@")
    assert emitted == ["\x1b[64u"]


@pytest.mark.tonio
async def test_keeps_non_matching_plain_character_after_kitty_printable_sequence():
    buffer, emitted = _make_buffer()
    buffer.process("\x1b[97ub")
    assert emitted == ["\x1b[97u", "b"]


@pytest.mark.tonio
async def test_keeps_raw_character_after_modified_kitty_printable_sequence():
    buffer, emitted = _make_buffer()
    buffer.process("\x1b[64;3u@")
    assert emitted == ["\x1b[64;3u", "@"]


@pytest.mark.tonio
async def test_handles_rapid_typing_simulation_with_kitty_protocol():
    # Simulates typing "hi" quickly with releases interleaved
    buffer, emitted = _make_buffer()
    buffer.process("\x1b[104u\x1b[104;1:3u\x1b[105u\x1b[105;1:3u")
    assert emitted == ["\x1b[104u", "\x1b[104;1:3u", "\x1b[105u", "\x1b[105;1:3u"]


# Mouse Events


@pytest.mark.tonio
async def test_handles_mouse_press_event():
    buffer, emitted = _make_buffer()
    buffer.process("\x1b[<0;10;5M")
    assert emitted == ["\x1b[<0;10;5M"]


@pytest.mark.tonio
async def test_handles_mouse_release_event():
    buffer, emitted = _make_buffer()
    buffer.process("\x1b[<0;10;5m")
    assert emitted == ["\x1b[<0;10;5m"]


@pytest.mark.tonio
async def test_handles_mouse_move_event():
    buffer, emitted = _make_buffer()
    buffer.process("\x1b[<35;20;5m")
    assert emitted == ["\x1b[<35;20;5m"]


@pytest.mark.tonio
async def test_handles_split_mouse_events():
    buffer, emitted = _make_buffer()
    buffer.process("\x1b[<3")
    buffer.process("5;1")
    buffer.process("5;")
    buffer.process("10m")
    assert emitted == ["\x1b[<35;15;10m"]


@pytest.mark.tonio
async def test_handles_multiple_mouse_events():
    buffer, emitted = _make_buffer()
    buffer.process("\x1b[<35;1;1m\x1b[<35;2;2m\x1b[<35;3;3m")
    assert emitted == ["\x1b[<35;1;1m", "\x1b[<35;2;2m", "\x1b[<35;3;3m"]


@pytest.mark.tonio
async def test_handles_old_style_mouse_sequence():
    buffer, emitted = _make_buffer()
    buffer.process("\x1b[M abc")
    assert emitted == ["\x1b[M ab", "c"]


@pytest.mark.tonio
async def test_buffers_incomplete_old_style_mouse_sequence():
    buffer, emitted = _make_buffer()
    buffer.process("\x1b[M")
    assert buffer.get_buffer() == "\x1b[M"

    buffer.process(" a")
    assert buffer.get_buffer() == "\x1b[M a"

    buffer.process("b")
    assert emitted == ["\x1b[M ab"]


# Edge Cases


@pytest.mark.tonio
async def test_handles_empty_input():
    buffer, emitted = _make_buffer()
    buffer.process("")
    # Empty string emits an empty data event
    assert emitted == [""]


@pytest.mark.tonio
async def test_handles_lone_escape_character_with_timeout():
    buffer, emitted, timers = _make_manual_buffer(timeout=10)
    buffer.process("\x1b")
    assert emitted == []

    # After timeout, should emit
    timers.tick(15)
    assert emitted == ["\x1b"]


@pytest.mark.tonio
async def test_flushes_a_lone_escape_promptly_with_the_longer_default_sequence_timeout():
    default_buffer, default_sequences, timers = _make_manual_buffer()

    default_buffer.process("\x1b")
    timers.tick(20)
    assert default_sequences == ["\x1b"]
    default_buffer.destroy()


@pytest.mark.tonio
async def test_handles_lone_escape_character_with_explicit_flush():
    buffer, emitted = _make_buffer()
    buffer.process("\x1b")
    assert emitted == []

    flushed = buffer.flush()
    assert flushed == ["\x1b"]


@pytest.mark.tonio
async def test_handles_buffer_input():
    buffer, emitted = _make_buffer()
    buffer.process(b"\x1b[A")
    assert emitted == ["\x1b[A"]


@pytest.mark.tonio
async def test_handles_very_long_sequences():
    buffer, emitted = _make_buffer()
    long_seq = "\x1b[" + "1;" * 50 + "H"
    buffer.process(long_seq)
    assert emitted == [long_seq]


# Flush


@pytest.mark.tonio
async def test_flushes_incomplete_sequences():
    buffer, _emitted = _make_buffer()
    buffer.process("\x1b[<35")
    flushed = buffer.flush()
    assert flushed == ["\x1b[<35"]
    assert buffer.get_buffer() == ""


@pytest.mark.tonio
async def test_returns_empty_list_if_nothing_to_flush():
    buffer, _emitted = _make_buffer()
    flushed = buffer.flush()
    assert flushed == []


@pytest.mark.tonio
async def test_emits_flushed_data_via_timeout():
    buffer, emitted, timers = _make_manual_buffer(timeout=10)
    buffer.process("\x1b[<35")
    assert emitted == []

    # Wait for timeout to flush
    timers.tick(15)

    assert emitted == ["\x1b[<35"]


# Clear


@pytest.mark.tonio
async def test_clears_buffered_content_without_emitting():
    buffer, emitted = _make_buffer()
    buffer.process("\x1b[<35")
    assert buffer.get_buffer() == "\x1b[<35"

    buffer.clear()
    assert buffer.get_buffer() == ""
    assert emitted == []


# Bracketed Paste


def _make_paste_buffer():
    buffer, emitted = _make_buffer()
    emitted_paste = []
    buffer.on_paste(emitted_paste.append)
    return buffer, emitted, emitted_paste


@pytest.mark.tonio
async def test_emits_paste_event_for_complete_bracketed_paste():
    buffer, emitted, emitted_paste = _make_paste_buffer()
    paste_start = "\x1b[200~"
    paste_end = "\x1b[201~"
    content = "hello world"

    buffer.process(paste_start + content + paste_end)

    assert emitted_paste == ["hello world"]
    assert emitted == []  # No data events during paste


@pytest.mark.tonio
async def test_handles_paste_arriving_in_chunks():
    buffer, emitted, emitted_paste = _make_paste_buffer()
    buffer.process("\x1b[200~")
    assert emitted_paste == []

    buffer.process("hello ")
    assert emitted_paste == []

    buffer.process("world\x1b[201~")
    assert emitted_paste == ["hello world"]
    assert emitted == []


@pytest.mark.tonio
async def test_handles_paste_with_input_before_and_after():
    buffer, emitted, emitted_paste = _make_paste_buffer()
    buffer.process("a")
    buffer.process("\x1b[200~pasted\x1b[201~")
    buffer.process("b")

    assert emitted == ["a", "b"]
    assert emitted_paste == ["pasted"]


@pytest.mark.tonio
async def test_handles_paste_with_newlines():
    buffer, emitted, emitted_paste = _make_paste_buffer()
    buffer.process("\x1b[200~line1\nline2\nline3\x1b[201~")

    assert emitted_paste == ["line1\nline2\nline3"]
    assert emitted == []


@pytest.mark.tonio
async def test_handles_paste_with_unicode():
    buffer, emitted, emitted_paste = _make_paste_buffer()
    buffer.process("\x1b[200~Hello 世界 🎉\x1b[201~")

    assert emitted_paste == ["Hello 世界 🎉"]
    assert emitted == []


# Destroy


@pytest.mark.tonio
async def test_clears_buffer_on_destroy():
    buffer, _emitted = _make_buffer()
    buffer.process("\x1b[<35")
    assert buffer.get_buffer() == "\x1b[<35"

    buffer.destroy()
    assert buffer.get_buffer() == ""


@pytest.mark.tonio
async def test_clears_pending_timeouts_on_destroy():
    buffer, emitted, timers = _make_manual_buffer(timeout=10)
    buffer.process("\x1b[<35")
    buffer.destroy()

    # Wait longer than timeout
    timers.tick(15)

    # Should not have emitted anything
    assert emitted == []
