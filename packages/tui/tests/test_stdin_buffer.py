"""Mirror of pi tui test/stdin-buffer.test.ts.

pi's upstream note: based on code from OpenTUI
(https://github.com/anomalyco/opentui), MIT License, Copyright (c) 2025
opentui.

pi drives the 10ms flush timeout with mocked timers; here the timers are real
tonio tasks, so the ticks become short real-time sleeps.
"""

import pytest
import tonio.colored as tonio

from pidrei_tui.stdin_buffer import StdinBuffer


FLUSH_WAIT = 0.05  # pi waits 15ms for the 10ms timer; leave real-time margin


def _make_buffer():
    buffer = StdinBuffer(timeout=10)
    emitted_sequences = []

    async def record(sequence):
        emitted_sequences.append(sequence)

    buffer.on_data(record)
    return buffer, emitted_sequences


# Regular Characters


@pytest.mark.tonio
async def test_passes_through_regular_characters_immediately():
    buffer, emitted = _make_buffer()
    await buffer.process("a")
    assert emitted == ["a"]


@pytest.mark.tonio
async def test_passes_through_multiple_regular_characters():
    buffer, emitted = _make_buffer()
    await buffer.process("abc")
    assert emitted == ["a", "b", "c"]


@pytest.mark.tonio
async def test_handles_unicode_characters():
    buffer, emitted = _make_buffer()
    await buffer.process("hello 世界")
    assert emitted == ["h", "e", "l", "l", "o", " ", "世", "界"]


# Complete Escape Sequences


@pytest.mark.tonio
async def test_passes_through_complete_mouse_sgr_sequences():
    buffer, emitted = _make_buffer()
    mouse_seq = "\x1b[<35;20;5m"
    await buffer.process(mouse_seq)
    assert emitted == [mouse_seq]


@pytest.mark.tonio
async def test_passes_through_complete_arrow_key_sequences():
    buffer, emitted = _make_buffer()
    up_arrow = "\x1b[A"
    await buffer.process(up_arrow)
    assert emitted == [up_arrow]


@pytest.mark.tonio
async def test_passes_through_complete_function_key_sequences():
    buffer, emitted = _make_buffer()
    f1 = "\x1b[11~"
    await buffer.process(f1)
    assert emitted == [f1]


@pytest.mark.tonio
async def test_passes_through_meta_key_sequences():
    buffer, emitted = _make_buffer()
    meta_a = "\x1ba"
    await buffer.process(meta_a)
    assert emitted == [meta_a]


@pytest.mark.tonio
async def test_passes_through_ss3_sequences():
    buffer, emitted = _make_buffer()
    ss3 = "\x1bOA"
    await buffer.process(ss3)
    assert emitted == [ss3]


# Partial Escape Sequences


@pytest.mark.tonio
async def test_buffers_incomplete_mouse_sgr_sequence():
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b")
    assert emitted == []
    assert buffer.get_buffer() == "\x1b"

    await buffer.process("[<35")
    assert emitted == []
    assert buffer.get_buffer() == "\x1b[<35"

    await buffer.process(";20;5m")
    assert emitted == ["\x1b[<35;20;5m"]
    assert buffer.get_buffer() == ""


@pytest.mark.tonio
async def test_buffers_incomplete_csi_sequence():
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[")
    assert emitted == []

    await buffer.process("1;")
    assert emitted == []

    await buffer.process("5H")
    assert emitted == ["\x1b[1;5H"]


@pytest.mark.tonio
async def test_buffers_split_across_many_chunks():
    buffer, emitted = _make_buffer()
    for chunk in ["\x1b", "[", "<", "3", "5", ";", "2", "0", ";", "5", "m"]:
        await buffer.process(chunk)

    assert emitted == ["\x1b[<35;20;5m"]


@pytest.mark.tonio
async def test_flushes_incomplete_sequence_after_timeout():
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[<35")
    assert emitted == []

    # Wait for timeout
    await tonio.sleep(FLUSH_WAIT)

    assert emitted == ["\x1b[<35"]


# Mixed Content


@pytest.mark.tonio
async def test_handles_characters_followed_by_escape_sequence():
    buffer, emitted = _make_buffer()
    await buffer.process("abc\x1b[A")
    assert emitted == ["a", "b", "c", "\x1b[A"]


@pytest.mark.tonio
async def test_handles_escape_sequence_followed_by_characters():
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[Aabc")
    assert emitted == ["\x1b[A", "a", "b", "c"]


@pytest.mark.tonio
async def test_handles_multiple_complete_sequences():
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[A\x1b[B\x1b[C")
    assert emitted == ["\x1b[A", "\x1b[B", "\x1b[C"]


@pytest.mark.tonio
async def test_handles_partial_sequence_with_preceding_characters():
    buffer, emitted = _make_buffer()
    await buffer.process("abc\x1b[<35")
    assert emitted == ["a", "b", "c"]
    assert buffer.get_buffer() == "\x1b[<35"

    await buffer.process(";20;5m")
    assert emitted == ["a", "b", "c", "\x1b[<35;20;5m"]


# Kitty Keyboard Protocol


@pytest.mark.tonio
async def test_handles_kitty_csi_u_press_events():
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[97u")
    assert emitted == ["\x1b[97u"]


@pytest.mark.tonio
async def test_handles_kitty_csi_u_release_events():
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[97;1:3u")
    assert emitted == ["\x1b[97;1:3u"]


@pytest.mark.tonio
async def test_handles_batched_kitty_press_and_release():
    # Press 'a', release 'a' batched together (common over SSH)
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[97u\x1b[97;1:3u")
    assert emitted == ["\x1b[97u", "\x1b[97;1:3u"]


@pytest.mark.tonio
async def test_handles_multiple_batched_kitty_events():
    # Press 'a', release 'a', press 'b', release 'b'
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[97u\x1b[97;1:3u\x1b[98u\x1b[98;1:3u")
    assert emitted == ["\x1b[97u", "\x1b[97;1:3u", "\x1b[98u", "\x1b[98;1:3u"]


@pytest.mark.tonio
async def test_handles_kitty_arrow_keys_with_event_type():
    # Up arrow press with event type
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[1;1:1A")
    assert emitted == ["\x1b[1;1:1A"]


@pytest.mark.tonio
async def test_handles_kitty_functional_keys_with_event_type():
    # Delete key release
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[3;1:3~")
    assert emitted == ["\x1b[3;1:3~"]


@pytest.mark.tonio
async def test_splits_esc_esc_csi_into_standalone_esc_and_csi_sequence():
    # WezTerm with enable_kitty_keyboard sends Escape key press as raw \x1b
    # and the release as a full Kitty CSI-u sequence, concatenated.
    # The buffer must not treat \x1b\x1b as a complete meta-key when the
    # following byte starts a new escape sequence.
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b\x1b[27;129:3u")
    assert emitted == ["\x1b", "\x1b[27;129:3u"]


@pytest.mark.tonio
async def test_splits_esc_esc_csi_with_no_modifier():
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b\x1b[27;1:3u")
    assert emitted == ["\x1b", "\x1b[27;1:3u"]


@pytest.mark.tonio
async def test_still_emits_esc_esc_as_single_sequence_when_not_followed_by_new_escape():
    # \x1b\x1b alone (no following CSI) stays as-is — e.g. ctrl+alt+[
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b\x1b")
    assert emitted == ["\x1b\x1b"]


@pytest.mark.tonio
async def test_handles_plain_characters_mixed_with_kitty_sequences():
    # Plain 'a' followed by Kitty release
    buffer, emitted = _make_buffer()
    await buffer.process("a\x1b[97;1:3u")
    assert emitted == ["a", "\x1b[97;1:3u"]


@pytest.mark.tonio
async def test_drops_raw_duplicate_character_after_matching_kitty_printable_sequence():
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[224uà")
    assert emitted == ["\x1b[224u"]


@pytest.mark.tonio
async def test_drops_raw_duplicate_character_after_kitty_printable_across_chunks():
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[64u")
    await buffer.process("@")
    assert emitted == ["\x1b[64u"]


@pytest.mark.tonio
async def test_keeps_non_matching_plain_character_after_kitty_printable_sequence():
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[97ub")
    assert emitted == ["\x1b[97u", "b"]


@pytest.mark.tonio
async def test_keeps_raw_character_after_modified_kitty_printable_sequence():
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[64;3u@")
    assert emitted == ["\x1b[64;3u", "@"]


@pytest.mark.tonio
async def test_handles_rapid_typing_simulation_with_kitty_protocol():
    # Simulates typing "hi" quickly with releases interleaved
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[104u\x1b[104;1:3u\x1b[105u\x1b[105;1:3u")
    assert emitted == ["\x1b[104u", "\x1b[104;1:3u", "\x1b[105u", "\x1b[105;1:3u"]


# Mouse Events


@pytest.mark.tonio
async def test_handles_mouse_press_event():
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[<0;10;5M")
    assert emitted == ["\x1b[<0;10;5M"]


@pytest.mark.tonio
async def test_handles_mouse_release_event():
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[<0;10;5m")
    assert emitted == ["\x1b[<0;10;5m"]


@pytest.mark.tonio
async def test_handles_mouse_move_event():
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[<35;20;5m")
    assert emitted == ["\x1b[<35;20;5m"]


@pytest.mark.tonio
async def test_handles_split_mouse_events():
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[<3")
    await buffer.process("5;1")
    await buffer.process("5;")
    await buffer.process("10m")
    assert emitted == ["\x1b[<35;15;10m"]


@pytest.mark.tonio
async def test_handles_multiple_mouse_events():
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[<35;1;1m\x1b[<35;2;2m\x1b[<35;3;3m")
    assert emitted == ["\x1b[<35;1;1m", "\x1b[<35;2;2m", "\x1b[<35;3;3m"]


@pytest.mark.tonio
async def test_handles_old_style_mouse_sequence():
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[M abc")
    assert emitted == ["\x1b[M ab", "c"]


@pytest.mark.tonio
async def test_buffers_incomplete_old_style_mouse_sequence():
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[M")
    assert buffer.get_buffer() == "\x1b[M"

    await buffer.process(" a")
    assert buffer.get_buffer() == "\x1b[M a"

    await buffer.process("b")
    assert emitted == ["\x1b[M ab"]


# Edge Cases


@pytest.mark.tonio
async def test_handles_empty_input():
    buffer, emitted = _make_buffer()
    await buffer.process("")
    # Empty string emits an empty data event
    assert emitted == [""]


@pytest.mark.tonio
async def test_handles_lone_escape_character_with_timeout():
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b")
    assert emitted == []

    # After timeout, should emit
    await tonio.sleep(FLUSH_WAIT)
    assert emitted == ["\x1b"]


@pytest.mark.tonio
async def test_handles_lone_escape_character_with_explicit_flush():
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b")
    assert emitted == []

    flushed = buffer.flush()
    assert flushed == ["\x1b"]


@pytest.mark.tonio
async def test_handles_buffer_input():
    buffer, emitted = _make_buffer()
    await buffer.process(b"\x1b[A")
    assert emitted == ["\x1b[A"]


@pytest.mark.tonio
async def test_handles_very_long_sequences():
    buffer, emitted = _make_buffer()
    long_seq = "\x1b[" + "1;" * 50 + "H"
    await buffer.process(long_seq)
    assert emitted == [long_seq]


# Flush


@pytest.mark.tonio
async def test_flushes_incomplete_sequences():
    buffer, _emitted = _make_buffer()
    await buffer.process("\x1b[<35")
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
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[<35")
    assert emitted == []

    # Wait for timeout to flush
    await tonio.sleep(FLUSH_WAIT)

    assert emitted == ["\x1b[<35"]


# Clear


@pytest.mark.tonio
async def test_clears_buffered_content_without_emitting():
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[<35")
    assert buffer.get_buffer() == "\x1b[<35"

    buffer.clear()
    assert buffer.get_buffer() == ""
    assert emitted == []


# Bracketed Paste


def _make_paste_buffer():
    buffer, emitted = _make_buffer()
    emitted_paste = []

    async def record(content):
        emitted_paste.append(content)

    buffer.on_paste(record)
    return buffer, emitted, emitted_paste


@pytest.mark.tonio
async def test_emits_paste_event_for_complete_bracketed_paste():
    buffer, emitted, emitted_paste = _make_paste_buffer()
    paste_start = "\x1b[200~"
    paste_end = "\x1b[201~"
    content = "hello world"

    await buffer.process(paste_start + content + paste_end)

    assert emitted_paste == ["hello world"]
    assert emitted == []  # No data events during paste


@pytest.mark.tonio
async def test_handles_paste_arriving_in_chunks():
    buffer, emitted, emitted_paste = _make_paste_buffer()
    await buffer.process("\x1b[200~")
    assert emitted_paste == []

    await buffer.process("hello ")
    assert emitted_paste == []

    await buffer.process("world\x1b[201~")
    assert emitted_paste == ["hello world"]
    assert emitted == []


@pytest.mark.tonio
async def test_handles_paste_with_input_before_and_after():
    buffer, emitted, emitted_paste = _make_paste_buffer()
    await buffer.process("a")
    await buffer.process("\x1b[200~pasted\x1b[201~")
    await buffer.process("b")

    assert emitted == ["a", "b"]
    assert emitted_paste == ["pasted"]


@pytest.mark.tonio
async def test_handles_paste_with_newlines():
    buffer, emitted, emitted_paste = _make_paste_buffer()
    await buffer.process("\x1b[200~line1\nline2\nline3\x1b[201~")

    assert emitted_paste == ["line1\nline2\nline3"]
    assert emitted == []


@pytest.mark.tonio
async def test_handles_paste_with_unicode():
    buffer, emitted, emitted_paste = _make_paste_buffer()
    await buffer.process("\x1b[200~Hello 世界 🎉\x1b[201~")

    assert emitted_paste == ["Hello 世界 🎉"]
    assert emitted == []


# Destroy


@pytest.mark.tonio
async def test_clears_buffer_on_destroy():
    buffer, _emitted = _make_buffer()
    await buffer.process("\x1b[<35")
    assert buffer.get_buffer() == "\x1b[<35"

    buffer.destroy()
    assert buffer.get_buffer() == ""


@pytest.mark.tonio
async def test_clears_pending_timeouts_on_destroy():
    buffer, emitted = _make_buffer()
    await buffer.process("\x1b[<35")
    buffer.destroy()

    # Wait longer than timeout
    await tonio.sleep(FLUSH_WAIT)

    # Should not have emitted anything
    assert emitted == []
