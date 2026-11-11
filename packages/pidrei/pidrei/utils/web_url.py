"""Port of pi's utils/web-url.ts.

pi's `webUrl` is `pidrei_ai.auth.oauth.urls.trusted_http_url`: the `http:` or
`https:` URL a value names, normalized, or None. The shared launcher also
opens files and custom protocols, so login only opens web URLs. Normalizing
percent-encodes control characters, which would otherwise end a terminal
hyperlink early. Only `display_url` lives here.
"""

from pidrei_ai.auth.oauth.urls import trusted_http_url
from pidrei_tui import replace_control_characters


def display_url(value: str) -> str:
    """`value` normalized as a web URL, else with control characters replaced: an untrusted string printed as is
    could end a hyperlink early or send its own escape sequences to the terminal."""
    return trusted_http_url(value) or replace_control_characters(value)
