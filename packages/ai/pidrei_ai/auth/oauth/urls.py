"""`new URL(value).href` for the URLs the login flows hand to a browser.

Port of pi's `trustedHttpUrl` (packages/ai/src/auth/oauth/trusted-url.ts),
which pi 1.5.0 consolidated from three flows' copies; pidrei had consolidated
them here before. It is the check that stops a hostile authorization server
from getting `open`/`xdg-open` to launch something that is not a web page. The
normalization is the fiddly half: `urlsplit` is far more permissive than the
URL constructor and encodes nothing, so a URL carrying terminal escapes would
otherwise reach the user's terminal and browser verbatim. As the URL
constructor does, control characters are percent-encoded in the userinfo,
path, query and fragment, a host containing one or an invalid port does not
parse, and the host is lowercased and a default port dropped. The rest of the URL
spec's host parsing (IDNA, forbidden host code points other than controls and
spaces) is not replicated.
"""

from urllib.parse import urlsplit, urlunsplit


# The URL spec's percent-encode sets, minus the delimiters `urlsplit` has
# already peeled off (`#`, `?`). Each also holds the C0 control percent-encode
# set: code points up to U+001F and above U+007E.
_PATH_ESCAPES = frozenset('" <>`{}')
_QUERY_ESCAPES = frozenset("\" #<>'")  # the special-query set
_FRAGMENT_ESCAPES = frozenset('" <>`')
# The userinfo set, minus the `:` and `@` delimiters, which are peeled off below.
_USERINFO_ESCAPES = frozenset('" #<>?`{}/;=[\\]^|')
# Schemes for which the URL constructor requires a host, with their default ports.
_DEFAULT_PORTS = {"ftp": 21, "http": 80, "https": 443, "ws": 80, "wss": 443}
_SPECIAL_SCHEMES = frozenset(_DEFAULT_PORTS)


def _percent_encode(text: str, escapes: frozenset[str]) -> str:
    encoded: list[str] = []
    for character in text:
        if character in escapes or ord(character) <= 0x1F or ord(character) > 0x7E:
            encoded.extend(f"%{byte:02X}" for byte in character.encode("utf-8"))
        else:
            encoded.append(character)
    return "".join(encoded)


def _authority(netloc: str, scheme: str, port: int | None) -> str:
    """The URL constructor's serialization of `netloc`: the username and password
    percent-encoded and kept as written (an empty password, or both empty, is
    dropped), the host lowercased, and the port dropped when it is empty or the
    scheme's default."""
    userinfo, at, host_port = netloc.rpartition("@")
    username, _, password = userinfo.partition(":")
    credentials = _percent_encode(username, _USERINFO_ESCAPES)
    if password:
        credentials += ":" + _percent_encode(password, _USERINFO_ESCAPES)
    # The port starts after the `]` of an IPv6 literal, else at the first `:`.
    if host_port.startswith("["):
        host = host_port[: host_port.find("]") + 1]
    else:
        host = host_port.partition(":")[0]
    authority = host.lower()
    if port is not None and port != _DEFAULT_PORTS.get(scheme):
        authority += f":{port}"
    return f"{credentials}@{authority}" if at and credentials else authority


def normalized_url(value: object) -> tuple[str, str] | None:
    """`(scheme, href)` for a parseable absolute URL, else None.

    The scheme is returned alongside the href so callers can apply their own
    protocol policy, which is the only part that differs between pi's copies.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        split = urlsplit(value.strip())
        # Out of range or not a number: the URL does not parse.
        port = split.port
    except ValueError:
        return None
    if not split.scheme:
        return None
    scheme = split.scheme.lower()
    if scheme in _SPECIAL_SCHEMES and not split.netloc:
        return None
    # A control character or space is a forbidden host code point: the URL does not parse.
    if any(ord(character) <= 0x20 or ord(character) == 0x7F for character in split.hostname or ""):
        return None
    href = urlunsplit(
        (
            scheme,
            _authority(split.netloc, scheme, port),
            _percent_encode(split.path, _PATH_ESCAPES) or ("/" if split.netloc else ""),
            _percent_encode(split.query, _QUERY_ESCAPES),
            _percent_encode(split.fragment, _FRAGMENT_ESCAPES),
        )
    )
    return scheme, href


def https_url(value: object) -> str | None:
    """The href of an https URL, else None."""
    normalized = normalized_url(value)
    return normalized[1] if normalized is not None and normalized[0] == "https" else None


def trusted_http_url(value: object) -> str | None:
    """The `http:` or `https:` URL `value` names, normalized, or None. Login flows open server-supplied URLs with the
    platform URL handler, which would also launch a `file:` URL or a custom protocol handler. Normalizing
    percent-encodes control characters, so the URL can also be embedded in terminal hyperlinks."""
    normalized = normalized_url(value)
    return normalized[1] if normalized is not None and normalized[0] in ("http", "https") else None
