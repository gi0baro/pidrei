"""Strict definite-length CBOR encoder (port of pi protocol `cbor/encoder.ts`).

Type mapping from pi's JS values: `None`/`bool`/`str`/`list`/plain `dict` map
directly; `bytes` stands in for Uint8Array; Python `int` maps to CBOR integers
(major types 0/1, safe-integer bounded) and Python `float` always encodes as
float64 — including integral floats, a distinction JS numbers cannot carry.
There is no `undefined`, so pi's hole/undefined rejections have no analog.
"""

import math
import struct

from .options import (
    MAX_SAFE_INTEGER,
    MAX_UINT32,
    MIN_SAFE_INTEGER,
    CborError,
    CborOptions,
    ResolvedCborOptions,
    resolve_options,
)


class _CborWriter:
    __slots__ = ("_buffer", "_max_byte_length")

    def __init__(self, max_byte_length: int):
        self._buffer = bytearray()
        self._max_byte_length = max_byte_length

    def _ensure_capacity(self, additional_bytes: int) -> None:
        if len(self._buffer) + additional_bytes > self._max_byte_length:
            raise CborError(f"CBOR byte length exceeds configured limit of {self._max_byte_length}")

    def write_byte(self, value: int) -> None:
        self._ensure_capacity(1)
        self._buffer.append(value)

    def write_bytes(self, data: bytes) -> None:
        self._ensure_capacity(len(data))
        self._buffer += data

    def write_uint16(self, value: int) -> None:
        self._ensure_capacity(2)
        self._buffer += value.to_bytes(2, "big")

    def write_uint32(self, value: int) -> None:
        self._ensure_capacity(4)
        self._buffer += value.to_bytes(4, "big")

    def write_uint64(self, value: int) -> None:
        self._ensure_capacity(8)
        self._buffer += value.to_bytes(8, "big")

    def write_float64(self, value: float) -> None:
        self._ensure_capacity(9)
        self._buffer.append(0xFB)
        self._buffer += struct.pack(">d", value)

    def finish(self) -> bytes:
        return bytes(self._buffer)


def _write_argument(writer: _CborWriter, major_type: int, value: int) -> None:
    prefix = major_type << 5
    if value < 24:
        writer.write_byte(prefix | value)
    elif value <= 0xFF:
        writer.write_byte(prefix | 24)
        writer.write_byte(value)
    elif value <= 0xFFFF:
        writer.write_byte(prefix | 25)
        writer.write_uint16(value)
    elif value <= MAX_UINT32:
        writer.write_byte(prefix | 26)
        writer.write_uint32(value)
    else:
        writer.write_byte(prefix | 27)
        writer.write_uint64(value)


def _encode_text(writer: _CborWriter, value: str, options: ResolvedCborOptions) -> None:
    try:
        data = value.encode("utf-8")
    except UnicodeEncodeError:
        raise CborError("CBOR text strings must contain valid Unicode scalar values")
    if len(data) > options.max_byte_length:
        raise CborError(f"CBOR text string length exceeds configured limit of {options.max_byte_length}")
    _write_argument(writer, 3, len(data))
    writer.write_bytes(data)


def _encode_value(
    writer: _CborWriter,
    value: object,
    options: ResolvedCborOptions,
    depth: int,
    ancestors: set[int],
) -> None:
    if depth > options.max_depth:
        raise CborError(f"CBOR nesting depth exceeds configured limit of {options.max_depth}")

    if value is None:
        writer.write_byte(0xF6)
        return
    if isinstance(value, bool):
        writer.write_byte(0xF5 if value else 0xF4)
        return
    if isinstance(value, int):
        if value < MIN_SAFE_INTEGER or value > MAX_SAFE_INTEGER:
            raise CborError("CBOR integers must be safe JavaScript integers")
        if value >= 0:
            _write_argument(writer, 0, value)
        else:
            _write_argument(writer, 1, -1 - value)
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CborError("CBOR numbers must be finite")
        writer.write_float64(value)
        return
    if isinstance(value, str):
        _encode_text(writer, value, options)
        return
    if isinstance(value, bytes):
        if len(value) > options.max_byte_length:
            raise CborError(f"CBOR byte string length exceeds configured limit of {options.max_byte_length}")
        _write_argument(writer, 2, len(value))
        writer.write_bytes(value)
        return
    if isinstance(value, list):
        if id(value) in ancestors:
            raise CborError("CBOR values must not contain cycles")
        if len(value) > options.max_container_length:
            raise CborError(f"CBOR array length exceeds configured limit of {options.max_container_length}")
        ancestors.add(id(value))
        try:
            _write_argument(writer, 4, len(value))
            for item in value:
                _encode_value(writer, item, options, depth + 1, ancestors)
        finally:
            ancestors.discard(id(value))
        return
    if type(value) is dict:
        if id(value) in ancestors:
            raise CborError("CBOR values must not contain cycles")
        if len(value) > options.max_container_length:
            raise CborError(f"CBOR map length exceeds configured limit of {options.max_container_length}")
        ancestors.add(id(value))
        try:
            _write_argument(writer, 5, len(value))
            for key, entry_value in value.items():
                if not isinstance(key, str):
                    raise CborError("CBOR map keys must be strings")
                _encode_text(writer, key, options)
                _encode_value(writer, entry_value, options, depth + 1, ancestors)
        finally:
            ancestors.discard(id(value))
        return

    raise CborError(f"Unsupported CBOR value type: {type(value).__name__}")


def encode_cbor(value: object, options: CborOptions | None = None) -> bytes:
    """Encodes the protocol's strict, definite-length RFC 8949 subset."""
    resolved = resolve_options(options)
    writer = _CborWriter(resolved.max_byte_length)
    _encode_value(writer, value, resolved, 0, set())
    return writer.finish()
