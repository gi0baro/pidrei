"""Transport layer of pi's protocol package: CBOR and length-prefixed framing.

The message schemas and codec (the remote-session wire vocabulary) are not
ported — UPSTREAM_EXPERIMENTAL_RULING.md.
"""

from .cbor import (
    DEFAULT_MAX_CBOR_BYTE_LENGTH,
    DEFAULT_MAX_CBOR_CONTAINER_LENGTH,
    DEFAULT_MAX_CBOR_DEPTH,
    CborError,
    CborOptions,
    decode_cbor,
    encode_cbor,
)
from .framing import (
    DEFAULT_MAX_FRAME_LENGTH,
    FrameDecoder,
    FrameDecoderOptions,
    FrameError,
    assert_complete_frame,
    encode_frame,
)


__all__ = [
    "DEFAULT_MAX_CBOR_BYTE_LENGTH",
    "DEFAULT_MAX_CBOR_CONTAINER_LENGTH",
    "DEFAULT_MAX_CBOR_DEPTH",
    "DEFAULT_MAX_FRAME_LENGTH",
    "CborError",
    "CborOptions",
    "FrameDecoder",
    "FrameDecoderOptions",
    "FrameError",
    "assert_complete_frame",
    "decode_cbor",
    "encode_cbor",
    "encode_frame",
]
