"""`new URL(value).href` for the URLs the login flows hand to a browser.

Port of pi's `trustedHttpUrl` (packages/ai/src/auth/oauth/trusted-url.ts),
which pi 1.5.0 consolidated from three flows' copies; pidrei had consolidated
them here before. It is the check that stops a hostile authorization server
from getting `open`/`xdg-open` to launch something that is not a web page. The
normalization is the fiddly half: `urlsplit` is far more permissive than the
URL constructor and encodes nothing, so a URL carrying terminal escapes would
otherwise reach the user's terminal and browser verbatim. As the URL
constructor does, control characters are percent-encoded in the path, query
and fragment, and a host containing one does not parse. The rest of the URL
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
# Schemes for which the URL constructor requires a host.
_SPECIAL_SCHEMES = frozenset({"ftp", "http", "https", "ws", "wss"})


def _percent_encode(text: str, escapes: frozenset[str]) -> str:
    encoded: list[str] = []
    for character in text:
        if character in escapes or ord(character) <= 0x1F or ord(character) > 0x7E:
            encoded.extend(f"%{byte:02X}" for byte in character.encode("utf-8"))
        else:
            encoded.append(character)
    return "".join(encoded)


def normalized_url(value: object) -> tuple[str, str] | None:
    """`(scheme, href)` for a parseable absolute URL, else None.

    The scheme is returned alongside the href so callers can apply their own
    protocol policy, which is the only part that differs between pi's copies.
    """
    if not isinstance(value, str) or not value:
        return None
    split = urlsplit(value.strip())
    if not split.scheme:
        return None
    if split.scheme in _SPECIAL_SCHEMES and not split.netloc:
        return None
    # A control character or space is a forbidden host code point: the URL does not parse.
    if any(ord(character) <= 0x20 or ord(character) == 0x7F for character in split.hostname or ""):
        return None
    href = urlunsplit(
        (
            split.scheme.lower(),
            split.netloc.lower(),
            _percent_encode(split.path, _PATH_ESCAPES) or ("/" if split.netloc else ""),
            _percent_encode(split.query, _QUERY_ESCAPES),
            _percent_encode(split.fragment, _FRAGMENT_ESCAPES),
        )
    )
    return split.scheme.lower(), href


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
