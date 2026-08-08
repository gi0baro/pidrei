"""Shared CBOR limits, error type and option resolution (port of pi protocol `cbor/options.ts`)."""

from dataclasses import dataclass


UINT32_BASE = 0x1_0000_0000
MAX_UINT32 = 0xFFFF_FFFF
_MAX_CONFIGURED_DEPTH = 512

# The wire-level integer range is pi's Number.isSafeInteger: values outside
# ±(2^53 - 1) are rejected on both encode and decode even though Python ints
# could carry them, so every peer sees the same numeric domain.
MAX_SAFE_INTEGER = 2**53 - 1
MIN_SAFE_INTEGER = -(2**53 - 1)

# Safe defaults for untrusted protocol payloads.
DEFAULT_MAX_CBOR_BYTE_LENGTH = 16 * 1024 * 1024
DEFAULT_MAX_CBOR_CONTAINER_LENGTH = 1_000_000
DEFAULT_MAX_CBOR_DEPTH = 64


@dataclass(slots=True, kw_only=True)
class CborOptions:
    # Maximum encoded input/output bytes and maximum byte/text string length.
    max_byte_length: int | None = None
    # Maximum number of elements in an array or entries in a map.
    max_container_length: int | None = None
    # Maximum recursive item depth.
    max_depth: int | None = None


@dataclass(slots=True, kw_only=True)
class ResolvedCborOptions:
    max_byte_length: int
    max_container_length: int
    max_depth: int


class CborError(Exception):
    pass


def _resolve_limit(name: str, value: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > maximum:
        raise ValueError(f"{name} must be an integer between 0 and {maximum}")
    return value


def resolve_options(options: CborOptions | None) -> ResolvedCborOptions:
    if options is None:
        options = CborOptions()
    return ResolvedCborOptions(
        max_byte_length=_resolve_limit(
            "maxByteLength",
            options.max_byte_length if options.max_byte_length is not None else DEFAULT_MAX_CBOR_BYTE_LENGTH,
            MAX_UINT32,
        ),
        max_container_length=_resolve_limit(
            "maxContainerLength",
            options.max_container_length
            if options.max_container_length is not None
            else DEFAULT_MAX_CBOR_CONTAINER_LENGTH,
            MAX_UINT32,
        ),
        max_depth=_resolve_limit(
            "maxDepth",
            options.max_depth if options.max_depth is not None else DEFAULT_MAX_CBOR_DEPTH,
            _MAX_CONFIGURED_DEPTH,
        ),
    )
