"""`new URL(value).href` for the verification URIs the flows hand to a browser.

Three of pi's flows carry their own copy of "parse it, insist on http(s), return
`url.href`" — the check that stops a hostile authorization server from getting
`open`/`xdg-open` to launch something that is not a web page. They are
consolidated here because the normalization is the fiddly half: `urlsplit` is
far more permissive than the URL constructor and does not encode anything, so a
`verification_uri` carrying terminal escapes would otherwise reach the user's
terminal and browser verbatim.
"""

from urllib.parse import urlsplit, urlunsplit


# The URL spec's path percent-encode set, minus the delimiters `urlsplit` has
# already peeled off (`#`, `?`).
_PATH_ESCAPES = frozenset('" <>`{}')
# Schemes for which the URL constructor requires a host.
_SPECIAL_SCHEMES = frozenset({"ftp", "http", "https", "ws", "wss"})


def _encode_path(path: str) -> str:
    encoded: list[str] = []
    for character in path:
        if character in _PATH_ESCAPES or ord(character) <= 0x20 or ord(character) == 0x7F:
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
    href = urlunsplit(
        (
            split.scheme.lower(),
            split.netloc.lower(),
            _encode_path(split.path) or ("/" if split.netloc else ""),
            split.query,
            split.fragment,
        )
    )
    return split.scheme.lower(), href


def https_url(value: object) -> str | None:
    """The href of an https URL, else None."""
    normalized = normalized_url(value)
    return normalized[1] if normalized is not None and normalized[0] == "https" else None


def http_or_https_url(value: object) -> str | None:
    """The href of an http(s) URL, else None."""
    normalized = normalized_url(value)
    return normalized[1] if normalized is not None and normalized[0] in ("http", "https") else None
