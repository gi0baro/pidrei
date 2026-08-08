"""Port of pi protocol test/framing.test.ts."""

import pytest

from pidrei_protocol.framing import (
    DEFAULT_MAX_FRAME_LENGTH,
    FrameDecoder,
    FrameDecoderOptions,
    FrameError,
    assert_complete_frame,
    encode_frame,
)


def test_prefixes_payloads_with_a_four_byte_big_endian_length():
    assert encode_frame(b"\xaa\xbb\xcc") == b"\x00\x00\x00\x03\xaa\xbb\xcc"
    assert encode_frame(b"") == b"\x00\x00\x00\x00"


def test_validates_one_complete_bounded_frame_without_accepting_trailing_or_partial_bytes():
    assert_complete_frame(b"\x00\x00\x00\x02\x01\x02", FrameDecoderOptions(max_frame_length=2))
    with pytest.raises(FrameError, match="complete"):
        assert_complete_frame(b"\x00\x00\x00\x02\x01")
    with pytest.raises(FrameError, match="exactly"):
        assert_complete_frame(b"\x00\x00\x00\x01\x01\x02")
    with pytest.raises(FrameError, match="limit"):
        assert_complete_frame(b"\x00\x00\x00\x03\x01\x02\x03", FrameDecoderOptions(max_frame_length=2))


def test_decodes_fragmented_coalesced_and_empty_frames_in_order():
    wire = encode_frame(b"\x01\x02\x03") + encode_frame(b"") + encode_frame(b"\x04")
    decoder = FrameDecoder()
    frames: list[bytes] = []
    for index in range(len(wire)):
        frames.extend(decoder.push(wire[index : index + 1]))
    decoder.end()
    assert frames == [b"\x01\x02\x03", b"", b"\x04"]

    coalesced = FrameDecoder()
    assert coalesced.push(wire) == frames
    coalesced.end()


def test_assembles_payloads_spanning_multiple_internal_blocks():
    payload = bytes(index % 251 for index in range(70_000))
    wire = encode_frame(payload)
    decoder = FrameDecoder()
    frames = [
        *decoder.push(wire[:101]),
        *decoder.push(wire[101:65_541]),
        *decoder.push(wire[65_541:]),
    ]
    decoder.end()
    assert frames == [payload]


def test_handles_every_split_point_across_a_frame():
    wire = encode_frame(bytes([10, 20, 30, 40]))
    for split in range(len(wire) + 1):
        decoder = FrameDecoder()
        frames = [*decoder.push(wire[:split]), *decoder.push(wire[split:])]
        decoder.end()
        assert frames == [bytes([10, 20, 30, 40])]


def test_copies_payload_bytes_instead_of_retaining_or_aliasing_input_chunks():
    chunk = bytearray(encode_frame(b"\x01\x02\x03"))
    decoder = FrameDecoder()
    frames = decoder.push(chunk)
    for index in range(len(chunk)):
        chunk[index] = 9
    assert frames == [b"\x01\x02\x03"]


def test_accepts_empty_chunks_and_a_clean_empty_stream():
    decoder = FrameDecoder()
    assert decoder.push(b"") == []
    decoder.end()


@pytest.mark.parametrize(
    ("label", "wire"),
    [
        ("partial header", b"\x00\x00\x00"),
        ("partial payload", b"\x00\x00\x00\x02\x01"),
    ],
)
def test_rejects_a_truncated_stream_at_end(label, wire):
    decoder = FrameDecoder()
    assert decoder.push(wire) == []
    with pytest.raises(FrameError):
        decoder.end()


def test_rejects_an_oversized_declared_length_as_soon_as_its_header_is_complete():
    decoder = FrameDecoder(FrameDecoderOptions(max_frame_length=3))
    with pytest.raises(FrameError, match="limit"):
        decoder.push(b"\x00\x00\x00\x04")
    with pytest.raises(FrameError, match="failed"):
        decoder.push(b"\x01")


def test_accepts_a_frame_exactly_at_the_configured_maximum():
    decoder = FrameDecoder(FrameDecoderOptions(max_frame_length=3))
    assert decoder.push(encode_frame(b"\x01\x02\x03")) == [b"\x01\x02\x03"]
    decoder.end()


def test_cannot_be_pushed_after_end():
    decoder = FrameDecoder()
    decoder.end()
    with pytest.raises(FrameError, match="ended"):
        decoder.push(b"")
    with pytest.raises(FrameError, match="ended"):
        decoder.end()


@pytest.mark.parametrize("max_frame_length", [-1, 1.5, float("nan"), DEFAULT_MAX_FRAME_LENGTH * 1_000])
def test_rejects_invalid_maximum_frame_length(max_frame_length):
    with pytest.raises(ValueError):
        FrameDecoder(FrameDecoderOptions(max_frame_length=max_frame_length))
