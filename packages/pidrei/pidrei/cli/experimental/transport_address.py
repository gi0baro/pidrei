"""Transport address parsing (port of pi `cli/experimental/transport-address.ts`).

pi validates with the WHATWG `URL` parser; Python has no equivalent, so the
port mirrors each observable check explicitly:

- `new URL(value)` throwing → the value has no `scheme:` prefix (or an
  authority that fails to parse).
- `url.protocol` → the scheme, lowercased, formatted with a trailing colon.
- `url.href !== value` → WHATWG would re-encode part of the value. For a
  non-special scheme that happens exactly when the value contains a character
  in the path percent-encode set (C0 controls, space, `"`, `<`, `>`, `` ` ``,
  `{`, `}`), a non-ASCII character, or leading/trailing C0/space that the URL
  constructor strips. `?`/`#` are rejected separately like pi does.
- `decodeURIComponent` → strict percent-decoding that rejects malformed
  escapes and invalid UTF-8 (Python's `unquote` silently passes both).
"""

import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit


@dataclass(slots=True, frozen=True)
class UnixTransportAddress:
    path: str
    transport: Literal["unix"] = "unix"


type TransportAddress = UnixTransportAddress


@dataclass(slots=True, frozen=True)
class TransportAddressResult:
    address: TransportAddress | None = None
    error: str | None = None


_URL_SCHEME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*):")
_PERCENT_ESCAPE_RE = re.compile(r"%[0-9A-Fa-f]{2}")
# Characters the WHATWG path percent-encode set would re-encode (making
# `url.href !== value`); `?` and `#` are checked separately.
_REENCODED_CHARS = set(' "<>`{}')


def _decode_uri_component(text: str) -> str:
    """JS `decodeURIComponent`: throws on malformed escapes and invalid UTF-8."""
    decoded = bytearray()
    index = 0
    while index < len(text):
        char = text[index]
        if char == "%":
            if _PERCENT_ESCAPE_RE.match(text, index) is None:
                raise ValueError(f"Malformed percent escape at {index}")
            decoded.append(int(text[index + 1 : index + 3], 16))
            index += 3
        else:
            decoded.extend(char.encode("utf-8"))
            index += 1
    return decoded.decode("utf-8")


def _would_reencode(value: str) -> bool:
    """Mirror of pi's `url.href !== value` for an authority-free unix: URL."""
    return any(ord(char) < 0x21 or ord(char) > 0x7E or char in _REENCODED_CHARS for char in value)


def parse_transport_address(value: str, option: Literal["--listen", "--connect"]) -> TransportAddressResult:
    invalid = TransportAddressResult(error=f'Invalid {option} address "{value}"')
    scheme_match = _URL_SCHEME_RE.match(value)
    if scheme_match is None:
        return invalid
    scheme = scheme_match.group(1).lower()
    if scheme != "unix":
        return TransportAddressResult(error=f'Unsupported {option} transport "{scheme}:"')
    try:
        url = urlsplit(value)
        has_authority = bool(url.hostname) or url.port is not None or bool(url.username) or bool(url.password)
    except ValueError:
        return invalid
    if has_authority:
        return TransportAddressResult(error="Unix transport address must not include an authority")
    if (
        not value.startswith("unix:///")
        or value.startswith("unix:////")
        or "?" in value
        or "#" in value
        or _would_reencode(value)
    ):
        return invalid
    try:
        path = _decode_uri_component(url.path)
    except ValueError, UnicodeDecodeError:
        return invalid
    if "\0" in path:
        return invalid
    if not path.startswith("/"):
        return TransportAddressResult(error="Unix transport address requires an absolute path")
    return TransportAddressResult(address=UnixTransportAddress(path=path))
