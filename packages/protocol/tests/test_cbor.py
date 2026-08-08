"""Port of pi protocol test/cbor/cbor.test.ts.

JS-only encoder rejections (undefined, array holes, bigint, symbol) have no
Python analog; the substituted invalid payloads (tuple, set, object, datetime,
complex, non-string keys) exercise the same "only the strict subset encodes"
guarantee. Unsafe integers are real Python ints beyond pi's safe range.
"""

import datetime
import math

import pytest

from pidrei_protocol.cbor import (
    DEFAULT_MAX_CBOR_BYTE_LENGTH,
    DEFAULT_MAX_CBOR_CONTAINER_LENGTH,
    DEFAULT_MAX_CBOR_DEPTH,
    CborError,
    CborOptions,
    decode_cbor,
    encode_cbor,
)


MAX_SAFE_INTEGER = 2**53 - 1
MIN_SAFE_INTEGER = -(2**53 - 1)

KNOWN_VECTORS = [
    (None, "f6"),
    (False, "f4"),
    (True, "f5"),
    (0, "00"),
    (1, "01"),
    (10, "0a"),
    (23, "17"),
    (24, "1818"),
    (25, "1819"),
    (100, "1864"),
    (1000, "1903e8"),
    (1_000_000, "1a000f4240"),
    (1_000_000_000_000, "1b000000e8d4a51000"),
    (MAX_SAFE_INTEGER, "1b001fffffffffffff"),
    (-1, "20"),
    (-10, "29"),
    (-24, "37"),
    (-25, "3818"),
    (-100, "3863"),
    (-1000, "3903e7"),
    (-1_000_000, "3a000f423f"),
    (MIN_SAFE_INTEGER, "3b001ffffffffffffe"),
    (1.1, "fb3ff199999999999a"),
    (-0.0, "fb8000000000000000"),
    (b"\x01\x02\x03\x04", "4401020304"),
    ("", "60"),
    ("IETF", "6449455446"),
    ("ü", "62c3bc"),
    ("水", "63e6b0b4"),
    ("𐅑", "64f0908591"),
    ([], "80"),
    ([1, 2, 3], "83010203"),
    ([1, [2, 3], [4, 5]], "8301820203820405"),
    ({"a": 1, "b": [2, 3]}, "a26161016162820203"),
]


@pytest.mark.parametrize(("value", "wire"), KNOWN_VECTORS)
def test_encodes_and_decodes_rfc_8949_vector(value, wire):
    encoded = encode_cbor(value)
    assert encoded.hex() == wire
    decoded = decode_cbor(bytes.fromhex(wire))
    if isinstance(value, float) and value == 0.0 and math.copysign(1.0, value) < 0:
        assert isinstance(decoded, float) and decoded == 0.0 and math.copysign(1.0, decoded) < 0
    else:
        assert decoded == value


def test_preserves_falsey_object_values():
    value = {"zero": 0, "empty": "", "no": False, "nil": None}
    assert decode_cbor(encode_cbor(value)) == value


def test_preserves_a_leading_unicode_bom():
    assert decode_cbor(bytes.fromhex("63efbbbf")) == "﻿"


@pytest.mark.parametrize(
    ("label", "value"),
    [
        ("NaN", float("nan")),
        ("positive infinity", float("inf")),
        ("negative infinity", float("-inf")),
        ("unsafe positive integer", MAX_SAFE_INTEGER + 1),
        ("unsafe negative integer", MIN_SAFE_INTEGER - 1),
        ("tuple", (1, 2)),
        ("set", {1, 2}),
        ("function", lambda: None),
        ("datetime", datetime.datetime(1970, 1, 1, tzinfo=datetime.UTC)),
        ("complex", 1j),
        ("object", object()),
    ],
)
def test_rejects_unsupported_encoder_value(label, value):
    with pytest.raises(CborError):
        encode_cbor(value)


def test_rejects_maps_with_non_string_keys():
    with pytest.raises(CborError, match="strings"):
        encode_cbor({1: True})


def test_rejects_lossy_strings_cycles_and_excessive_encoder_depth():
    with pytest.raises(CborError, match="Unicode"):
        encode_cbor("\ud800")

    cyclic: list = []
    cyclic.append(cyclic)
    with pytest.raises(CborError, match="cycles"):
        encode_cbor(cyclic)

    too_deep: object = None
    for _ in range(DEFAULT_MAX_CBOR_DEPTH + 1):
        too_deep = [too_deep]
    with pytest.raises(CborError, match="depth"):
        encode_cbor(too_deep)


@pytest.mark.parametrize(
    ("label", "wire"),
    [
        ("empty input", ""),
        ("truncated integer", "18"),
        ("reserved additional information", "1c"),
        ("indefinite byte string", "5f"),
        ("indefinite text string", "7f"),
        ("indefinite array", "9f"),
        ("indefinite map", "bf"),
        ("tag", "c000"),
        ("undefined", "f7"),
        ("unsupported simple value", "e0"),
        ("break outside an indefinite item", "ff"),
        ("float16", "f93c00"),
        ("float32", "fa3f800000"),
        ("positive infinity", "fb7ff0000000000000"),
        ("NaN", "fb7ff8000000000000"),
        ("truncated float64", "fb3ff00000"),
        ("truncated byte string", "44010203"),
        ("truncated text string", "636162"),
        ("truncated array", "8201"),
        ("truncated map", "a16161"),
        ("trailing data", "0000"),
        ("non-string map key", "a10102"),
        ("duplicate map key", "a2616101616102"),
        ("invalid UTF-8 byte", "61ff"),
        ("overlong UTF-8", "62c080"),
        ("UTF-8 surrogate", "63eda080"),
        ("unsafe positive integer", "1b0020000000000000"),
        ("unsafe negative integer", "3b001fffffffffffff"),
        ("unsafe integer encoded as float64", "fb4340000000000000"),
    ],
)
def test_rejects_invalid_decoder_input(label, wire):
    with pytest.raises(CborError):
        decode_cbor(bytes.fromhex(wire))


def test_enforces_depth_and_declared_length_limits_before_traversing_values():
    too_deep = bytearray(b"\x81" * (DEFAULT_MAX_CBOR_DEPTH + 1)) + b"\xf6"
    with pytest.raises(CborError, match="depth"):
        decode_cbor(bytes(too_deep))

    oversized_bytes = bytes.fromhex(f"5a{DEFAULT_MAX_CBOR_BYTE_LENGTH + 1:08x}")
    oversized_text = bytes.fromhex(f"7a{DEFAULT_MAX_CBOR_BYTE_LENGTH + 1:08x}")
    oversized_array = bytes.fromhex(f"9a{DEFAULT_MAX_CBOR_CONTAINER_LENGTH + 1:08x}")
    oversized_map = bytes.fromhex(f"ba{DEFAULT_MAX_CBOR_CONTAINER_LENGTH + 1:08x}")
    for wire in (oversized_bytes, oversized_text, oversized_array, oversized_map):
        with pytest.raises(CborError, match="limit"):
            decode_cbor(wire)


def test_supports_stricter_caller_provided_limits():
    with pytest.raises(CborError, match="limit"):
        decode_cbor(bytes.fromhex("83010203"), CborOptions(max_container_length=2))
    with pytest.raises(CborError, match="limit"):
        decode_cbor(bytes.fromhex("626162"), CborOptions(max_byte_length=2))
    with pytest.raises(CborError, match="limit"):
        encode_cbor([1, 2, 3], CborOptions(max_container_length=2))
    with pytest.raises(CborError, match="limit"):
        encode_cbor("ab", CborOptions(max_byte_length=2))
