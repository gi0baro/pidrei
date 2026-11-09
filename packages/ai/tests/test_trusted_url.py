"""Tests for pidrei_ai.auth.oauth.urls.trusted_http_url (pi's trusted-url.ts has no test file).

Expected values are what `new URL(value).href` gives in pi; pi 1.5.0 relies on the normalization
percent-encoding control characters, so the URL can be embedded in terminal hyperlinks.
"""

from pidrei_ai.auth.oauth.urls import trusted_http_url


def test_accepts_only_http_and_https_urls():
    assert trusted_http_url("https://example.test/device") == "https://example.test/device"
    assert trusted_http_url("http://example.test") == "http://example.test/"
    assert trusted_http_url("file:///Users/victim/Downloads/payload.app") is None
    assert trusted_http_url("search-ms:query=x") is None
    assert trusted_http_url(42) is None
    assert trusted_http_url("") is None


def test_percent_encodes_control_characters_in_the_path_query_and_fragment():
    assert (
        trusted_http_url("https://x.test/a\x1b[31m?q=\x1b]8;;evil\x07#f\x1b")
        == "https://x.test/a%1B[31m?q=%1B]8;;evil%07#f%1B"
    )
    assert trusted_http_url("https://x.test/é?q='é'#é") == "https://x.test/%C3%A9?q=%27%C3%A9%27#%C3%A9"


def test_rejects_a_host_containing_a_control_character():
    assert trusted_http_url("https://x\x1b.test/") is None
