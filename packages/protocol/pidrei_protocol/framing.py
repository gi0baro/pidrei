"""Length-prefixed binary framing (port of pi protocol `framing.ts`).

Payloads and chunks are `bytes` (bytearray accepted on input, always copied);
pi's fixed 64 KiB payload blocks are an allocation strategy, mirrored here as
a chunk list joined on completion — the observable contract (no preallocation
from a declared length, no aliasing of input chunks) is the same.
"""

from dataclasses import dataclass


_FRAME_HEADER_LENGTH = 4
_MAX_UINT32 = 0xFFFF_FFFF

# Default upper bound for one framed CBOR payload.
DEFAULT_MAX_FRAME_LENGTH = 16 * 1024 * 1024


@dataclass(slots=True, kw_only=True)
class FrameDecoderOptions:
    max_frame_length: int | None = None


class FrameError(Exception):
    pass


def _resolve_max_frame_length(options: FrameDecoderOptions | None) -> int:
    value = (
        options.max_frame_length
        if options is not None and options.max_frame_length is not None
        else DEFAULT_MAX_FRAME_LENGTH
    )
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > _MAX_UINT32:
        raise ValueError(f"maxFrameLength must be an integer between 0 and {_MAX_UINT32}")
    return value


def encode_frame(payload: bytes) -> bytes:
    """Prefixes a payload with its unsigned 32-bit big-endian byte length."""
    if not isinstance(payload, bytes | bytearray):
        raise TypeError("Frame payload must be bytes")
    if len(payload) > _MAX_UINT32:
        raise ValueError("Frame payload exceeds the unsigned 32-bit length limit")
    return len(payload).to_bytes(_FRAME_HEADER_LENGTH, "big") + payload


class FrameDecoder:
    """Incrementally splits arbitrary byte chunks into length-prefixed payloads."""

    def __init__(self, options: FrameDecoderOptions | None = None):
        self._max_frame_length = _resolve_max_frame_length(options)
        self._header = bytearray()
        self._payload_chunks: list[bytes] = []
        self._expected_payload_length: int | None = None
        self._payload_length = 0
        self._state = "open"

    def push(self, chunk: bytes) -> list[bytes]:
        if self._state == "ended":
            raise FrameError("Frame decoder has ended")
        if self._state == "failed":
            raise FrameError("Frame decoder has failed")
        if not isinstance(chunk, bytes | bytearray):
            raise TypeError("Frame chunk must be bytes")

        frames: list[bytes] = []
        chunk_offset = 0
        chunk_length = len(chunk)
        while chunk_offset < chunk_length:
            if self._expected_payload_length is None:
                header_bytes = min(_FRAME_HEADER_LENGTH - len(self._header), chunk_length - chunk_offset)
                self._header += chunk[chunk_offset : chunk_offset + header_bytes]
                chunk_offset += header_bytes
                if len(self._header) < _FRAME_HEADER_LENGTH:
                    continue

                frame_length = int.from_bytes(self._header, "big")
                self._header = bytearray()
                if frame_length > self._max_frame_length:
                    self._fail(f"Frame length {frame_length} exceeds configured limit of {self._max_frame_length}")
                if frame_length == 0:
                    frames.append(b"")
                    continue
                self._expected_payload_length = frame_length
                self._payload_chunks = []
                self._payload_length = 0

            expected_payload_length = self._expected_payload_length
            payload_bytes = min(expected_payload_length - self._payload_length, chunk_length - chunk_offset)
            if payload_bytes > 0:
                self._payload_chunks.append(bytes(chunk[chunk_offset : chunk_offset + payload_bytes]))
                self._payload_length += payload_bytes
                chunk_offset += payload_bytes
            if self._payload_length == expected_payload_length:
                frames.append(b"".join(self._payload_chunks))
                self._payload_chunks = []
                self._expected_payload_length = None
                self._payload_length = 0
        return frames

    def end(self) -> None:
        if self._state == "ended":
            raise FrameError("Frame decoder has ended")
        if self._state == "failed":
            raise FrameError("Frame decoder has failed")
        if len(self._header) != 0 or self._expected_payload_length is not None:
            self._fail("Truncated frame at end of stream")
        self._state = "ended"

    def _fail(self, message: str) -> None:
        self._state = "failed"
        self._header = bytearray()
        self._payload_chunks = []
        self._expected_payload_length = None
        self._payload_length = 0
        raise FrameError(message)
