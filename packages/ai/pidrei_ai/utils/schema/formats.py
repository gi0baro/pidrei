"""The format checks of tool-argument validation: pi-schema's `formats`
(packages/schema/src/formats/index.ts) over `jsonschema`'s checkers.

pi checks every format JSON Schema 2020-12 defines plus `url` and
`json-pointer-uri-fragment`, with checks adapted from TypeBox. Here the
standard formats are `jsonschema`'s draft 2020-12 checkers, so how strict
each one is follows the Python libraries behind them, not TypeBox's regular
expressions. `url`, `json-pointer-uri-fragment`, `iri` and `iri-reference`
are checked here: `jsonschema` checks IRIs only with `rfc3987-syntax`, whose
grammars would add ~0.65 s to every launch. Any other format is an
annotation, as in pi.
"""

import re
from urllib.parse import quote, urlsplit

from jsonschema import Draft202012Validator, FormatChecker
from rfc3986_validator import validate_rfc3986


_STANDARD = (
    "date",
    "date-time",
    "duration",
    "email",
    "hostname",
    "idn-email",
    "idn-hostname",
    "ipv4",
    "ipv6",
    "json-pointer",
    "regex",
    "relative-json-pointer",
    "time",
    "uri",
    "uri-reference",
    "uri-template",
    "uuid",
)

# The WHATWG URL parser (pi's `URL.canParse`) strips leading and trailing C0
# controls and spaces, needs a scheme, and needs a host for the special schemes.
_URL_TRIM = "".join(map(chr, range(0x21)))
_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*:")
_SPECIAL_SCHEMES = frozenset({"ftp", "http", "https", "ws", "wss"})
_FORBIDDEN_HOST = re.compile(r"[\x00-\x20#%/:<>?@\[\\\]^|]")

_JSON_POINTER_URI_FRAGMENT = re.compile(r"#(?:/(?:[a-z0-9_\-.!$&'()*+,;:=@]|%[0-9a-f]{2}|~0|~1)*)*", re.IGNORECASE)


def is_url(value: object) -> bool:
    """Whether `value` is a URL: a scheme, then, for http(s), ws(s) and ftp, a
    host. Formats apply to strings only, as `jsonschema`'s own checkers."""
    if not isinstance(value, str):
        return True
    text = value.strip(_URL_TRIM)
    scheme = _SCHEME.match(text)
    if scheme is None:
        return False
    if scheme[0][:-1].lower() not in _SPECIAL_SCHEMES:
        return True
    try:
        parts = urlsplit(text)
        host = parts.hostname
        _ = parts.port
    except ValueError:
        return False
    if not host:
        return False
    if host.startswith("[") or ":" in host:
        return True
    return _FORBIDDEN_HOST.search(host) is None


def is_json_pointer_uri_fragment(value: object) -> bool:
    """Whether `value` is a JSON Pointer in a URI fragment (RFC 6901)."""
    return not isinstance(value, str) or _JSON_POINTER_URI_FRAGMENT.fullmatch(value) is not None


def _as_uri(value: str) -> str | None:
    """An IRI mapped to a URI as RFC 3987 section 3.1 does: every non-ASCII
    character percent-encoded as UTF-8. None for a lone surrogate, which no IRI
    holds."""
    try:
        return "".join(c if c.isascii() else quote(c, safe="") for c in value)
    except UnicodeEncodeError:
        return None


def is_iri(value: object) -> bool:
    """Whether `value` is an IRI (RFC 3987)."""
    if not isinstance(value, str):
        return True
    uri = _as_uri(value)
    return uri is not None and bool(validate_rfc3986(uri, rule="URI"))


def is_iri_reference(value: object) -> bool:
    """Whether `value` is an IRI reference (RFC 3987)."""
    if not isinstance(value, str):
        return True
    uri = _as_uri(value)
    return uri is not None and bool(validate_rfc3986(uri, rule="URI_reference"))


def _formats() -> FormatChecker:
    standard = Draft202012Validator.FORMAT_CHECKER
    checker = FormatChecker(())
    for name in _STANDARD:
        if name in standard.checkers:
            checker.checkers[name] = standard.checkers[name]
    checker.checks("iri")(is_iri)
    checker.checks("iri-reference")(is_iri_reference)
    checker.checks("url")(is_url)
    checker.checks("json-pointer-uri-fragment")(is_json_pointer_uri_fragment)
    return checker


#: Every format pi checks, by name; built once at import and never changed.
FORMATS = _formats()
