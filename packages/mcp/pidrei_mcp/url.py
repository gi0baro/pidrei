"""The parts of the WHATWG `URL` the MCP code relies on.

pi parses and rebuilds URLs with `new URL(...)`, and discovery compares the
results, so the normalization is observable: the scheme and host are
lower-cased, a scheme's default port is dropped, an http(s) URL without a
path gets `/` (`String(new URL("http://a"))` is `http://a/`), dot segments
are resolved, and `URLSearchParams` serializes as form encoding. `urlsplit`
does none of that on its own. International domain names are not converted
to punycode.

No pi counterpart (pi has the platform's `URL`). Public, since the MCP
extension builds redirect URIs and credential keys the same way.
"""

import re
from dataclasses import dataclass, replace
from urllib.parse import parse_qsl, quote_plus, urljoin, urlsplit


_SPECIAL_DEFAULT_PORTS = {"http": "80", "https": "443", "ws": "80", "wss": "443", "ftp": "21"}
# The URL spec's path percent-encode set, minus the delimiters `urlsplit` has
# already peeled off (`#`, `?`).
_PATH_ESCAPES = frozenset('" <>`{}')
_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*")


@dataclass(frozen=True, slots=True)
class Url:
    scheme: str
    userinfo: str
    hostname: str
    port: str
    pathname: str
    query: str
    fragment: str
    # Non-special URLs (`javascript:`, `data:`) are kept as written: only
    # their protocol is ever looked at.
    opaque: str | None = None

    @property
    def protocol(self) -> str:
        return f"{self.scheme}:"

    @property
    def host(self) -> str:
        return f"{self.hostname}:{self.port}" if self.port else self.hostname

    @property
    def origin(self) -> str:
        return f"{self.scheme}://{self.host}"

    @property
    def search(self) -> str:
        return f"?{self.query}" if self.query else ""

    @property
    def href(self) -> str:
        if self.opaque is not None:
            return self.opaque
        userinfo = f"{self.userinfo}@" if self.userinfo else ""
        fragment = f"#{self.fragment}" if self.fragment else ""
        return f"{self.scheme}://{userinfo}{self.host}{self.pathname}{self.search}{fragment}"

    def __str__(self) -> str:
        return self.href

    def without_fragment(self) -> Url:
        return replace(self, fragment="")

    def search_param(self, name: str) -> str | None:
        """`url.searchParams.get(name)`."""
        for key, value in parse_qsl(self.query, keep_blank_values=True):
            if key == name:
                return value
        return None

    def with_search_param(self, name: str, value: str) -> Url:
        """`url.searchParams.set(name, value)`: replaces the first `name`,
        drops the others, and reserializes the whole query."""
        pairs = parse_qsl(self.query, keep_blank_values=True)
        updated: list[tuple[str, str]] = []
        found = False
        for key, item in pairs:
            if key == name:
                if not found:
                    updated.append((name, value))
                    found = True
                continue
            updated.append((key, item))
        if not found:
            updated.append((name, value))
        return replace(self, query=form_encode(updated))


def form_encode(pairs: list[tuple[str, str]]) -> str:
    """`URLSearchParams.toString()`: spaces become `+`; only ASCII
    alphanumerics and `*-._` stay unescaped (`~` is escaped, unlike
    `urlencode`)."""
    return "&".join(f"{_form_escape(key)}={_form_escape(value)}" for key, value in pairs)


def _form_escape(value: str) -> str:
    return quote_plus(value, safe="*").replace("~", "%7E")


def _encode_path(path: str) -> str:
    encoded: list[str] = []
    for character in path:
        if character in _PATH_ESCAPES or ord(character) <= 0x20 or ord(character) == 0x7F:
            encoded.extend(f"%{byte:02X}" for byte in character.encode("utf-8"))
        else:
            encoded.append(character)
    return "".join(encoded)


def _remove_dot_segments(path: str) -> str:
    """RFC 3986 §5.2.4, as the URL parser applies it to special URLs."""
    output: list[str] = []
    segments = path.split("/")
    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        if segment in (".", "%2e", "%2E"):
            if last:
                output.append("")
            continue
        if segment.lower() in ("..", ".%2e", "%2e.", "%2e%2e"):
            if len(output) > 1:
                output.pop()
            if last:
                output.append("")
            continue
        output.append(segment)
    result = "/".join(output)
    return result if result.startswith("/") else f"/{result}"


def parse_url(value: str, base: str | Url | None = None) -> Url:
    """`new URL(value, base)`; raises `ValueError` where it throws."""
    text = value.strip()
    if base is not None:
        text = urljoin(str(base), text)
    split = urlsplit(text)
    scheme = split.scheme.lower()
    if not scheme or not _SCHEME.fullmatch(split.scheme) or not text[len(split.scheme) :].startswith(":"):
        raise ValueError(f"Invalid URL: {value}")
    if scheme not in _SPECIAL_DEFAULT_PORTS:
        return Url(scheme, "", "", "", "", "", "", opaque=text)
    try:
        port_number = split.port
    except ValueError:
        raise ValueError(f"Invalid URL: {value}") from None
    hostname = split.hostname or ""
    if not hostname:
        raise ValueError(f"Invalid URL: {value}")
    if ":" in hostname:
        hostname = f"[{hostname}]"
    port = "" if port_number is None else str(port_number)
    if port == _SPECIAL_DEFAULT_PORTS[scheme]:
        port = ""
    userinfo = split.netloc.rpartition("@")[0] if "@" in split.netloc else ""
    return Url(
        scheme=scheme,
        userinfo=userinfo,
        hostname=hostname.lower(),
        port=port,
        pathname=_remove_dot_segments(_encode_path(split.path)) if split.path else "/",
        query=split.query,
        fragment=split.fragment,
    )


def can_parse(value: str) -> bool:
    """`URL.canParse(value)`."""
    try:
        parse_url(value)
    except ValueError:
        return False
    return True
