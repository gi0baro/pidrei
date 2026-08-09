"""Update check against the GitHub releases API.

Adapted from pi's version-check.test.ts, which was never mirrored. pi stubs
global fetch; pidrei routes through the `utils/http.py` seam, so these swap
`shared_client` on the module the checker imports from.

The comparison cases are pi's, plus the pidrei-only ones: our version scheme is
pi's version with our own segment appended (`0.82.0.N`) and dev builds carry
`.devN`, neither of which pi's three-segment semver pattern parses.
"""

import json
import os

import pytest

import pidrei_ai.utils.http as http_module
from pidrei.utils.version_check import (
    RELEASES_URL,
    check_for_new_version,
    compare_package_versions,
    get_latest_release,
    get_latest_version,
    is_newer_package_version,
)


class _Response:
    def __init__(self, payload, status_code: int = 200):
        self.status_code = status_code
        self._payload = payload

    async def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class _Client:
    def __init__(self, response):
        self._response = response
        self.calls: list[tuple[str, dict]] = []

    async def get(self, url, *, headers=None, timeout=None):
        self.calls.append((url, headers or {}))
        return self._response


@pytest.fixture
def stub_http(request):
    """Install a fake shared_client and hand back the recorder.

    Not `monkeypatch`: that is a yield fixture, which the tonio plugin cannot
    wrap (Rust abort). Same reason as the server suite's `tmp_dir`.
    """
    original = http_module.shared_client
    request.addfinalizer(lambda: setattr(http_module, "shared_client", original))

    for name in ("PIDREI_OFFLINE", "PIDREI_SKIP_VERSION_CHECK"):
        previous = os.environ.pop(name, None)
        request.addfinalizer(
            lambda n=name, v=previous: os.environ.__setitem__(n, v) if v is not None else os.environ.pop(n, None)
        )

    def install(payload, status_code: int = 200) -> _Client:
        client = _Client(_Response(payload, status_code))
        http_module.shared_client = lambda: client
        return client

    return install


class TestComparison:
    def test_compares_package_versions(self):
        assert compare_package_versions("0.70.6", "0.70.5") > 0
        assert compare_package_versions("0.70.5", "0.70.5") == 0
        assert compare_package_versions("0.70.4", "0.70.5") < 0
        assert compare_package_versions("5.0.0-beta.20", "5.0.0-beta.9") > 0
        assert is_newer_package_version("0.70.5", "0.70.5") is False
        assert is_newer_package_version("0.70.6", "0.70.5") is True

    def test_compares_the_pidrei_four_segment_scheme(self):
        # pi 0.82.0 + our Nth build. A three-segment-only parser returns None
        # for both sides here and falls back to string inequality, which
        # reports an "update" for every difference including downgrades.
        assert compare_package_versions("0.82.0.1", "0.82.0.0") > 0
        assert compare_package_versions("0.82.0.0", "0.82.0.1") < 0
        assert compare_package_versions("0.82.0.0", "0.82.0") == 0
        assert compare_package_versions("0.83.0.0", "0.82.0.9") > 0
        assert is_newer_package_version("0.82.0.0", "0.82.0.1") is False

    def test_orders_dev_and_prerelease_builds_before_releases(self):
        assert compare_package_versions("0.1.0", "0.1.0.dev0") > 0
        assert compare_package_versions("0.1.0.dev1", "0.1.0.dev0") > 0
        assert compare_package_versions("1.0.0rc1", "1.0.0.dev9") > 0
        assert compare_package_versions("1.0.0", "1.0.0rc1") > 0
        assert is_newer_package_version("0.1.0", "0.1.0.dev0") is True

    def test_unparseable_versions_fall_back_to_string_inequality(self):
        assert compare_package_versions("nightly", "1.0.0") is None
        assert is_newer_package_version("nightly", "1.0.0") is True
        assert is_newer_package_version("nightly", "nightly") is False


@pytest.mark.tonio
class TestApi:
    @pytest.mark.tonio
    async def test_returns_only_newer_versions(self, stub_http):
        stub_http({"tag_name": "1.2.3", "html_url": "https://example.test/r/1.2.3"})
        assert await check_for_new_version("1.2.3") is None

        stub_http({"tag_name": "1.2.3", "html_url": "https://example.test/r/1.2.3"})
        assert await check_for_new_version("1.2.2") == {
            "version": "1.2.3",
            "url": "https://example.test/r/1.2.3",
        }

    @pytest.mark.tonio
    async def test_uses_the_github_releases_api_with_a_pidrei_user_agent(self, stub_http):
        client = stub_http({"tag_name": "1.2.4"})

        assert await get_latest_version("1.2.3") == "1.2.4"

        url, headers = client.calls[0]
        assert url == "https://api.github.com/repos/gi0baro/pidrei/releases/latest"
        assert headers["User-Agent"].startswith("pidrei/1.2.3 ")
        assert headers["accept"] == "application/vnd.github+json"
        assert headers["x-github-api-version"] == "2022-11-28"

    @pytest.mark.tonio
    async def test_returns_release_notes_and_url(self, stub_http):
        stub_http(
            {
                "tag_name": "1.2.4",
                "html_url": "https://example.test/r/1.2.4",
                "body": " **Read this** ",
            }
        )
        assert await get_latest_release("1.2.3") == {
            "version": "1.2.4",
            "url": "https://example.test/r/1.2.4",
            "note": "**Read this**",
        }

    @pytest.mark.tonio
    async def test_tolerates_a_v_prefixed_tag(self, stub_http):
        stub_http({"tag_name": "v1.2.4"})
        assert await get_latest_version("1.2.3") == "1.2.4"

    @pytest.mark.tonio
    async def test_falls_back_to_the_releases_page_without_an_html_url(self, stub_http):
        stub_http({"tag_name": "1.2.4"})
        release = await get_latest_release("1.2.3")
        assert release == {"version": "1.2.4", "url": "https://github.com/gi0baro/pidrei/releases"}

    @pytest.mark.tonio
    async def test_returns_none_without_a_tag(self, stub_http):
        stub_http({"html_url": "https://example.test/r"})
        assert await get_latest_release("1.2.3") is None

    @pytest.mark.tonio
    async def test_returns_none_on_an_error_response(self, stub_http):
        stub_http({"message": "Not Found"}, status_code=404)
        assert await get_latest_release("1.2.3") is None

    @pytest.mark.tonio
    async def test_skips_automatic_api_calls_when_version_checks_are_disabled(self, stub_http):
        client = stub_http({"tag_name": "1.2.4"})
        os.environ["PIDREI_SKIP_VERSION_CHECK"] = "1"

        assert await check_for_new_version("1.2.3") is None
        assert client.calls == []

    @pytest.mark.tonio
    async def test_allows_direct_api_calls_when_automatic_checks_are_disabled(self, stub_http):
        client = stub_http({"tag_name": "1.2.4"})
        os.environ["PIDREI_SKIP_VERSION_CHECK"] = "1"

        assert await get_latest_version("1.2.3") == "1.2.4"
        assert len(client.calls) == 1

    @pytest.mark.tonio
    async def test_makes_no_request_when_offline(self, stub_http):
        client = stub_http({"tag_name": "1.2.4"})
        os.environ["PIDREI_OFFLINE"] = "1"

        assert await get_latest_release("1.2.3") is None
        assert client.calls == []


@pytest.mark.tonio
async def test_check_swallows_transport_errors(stub_http):
    stub_http({"tag_name": "1.2.4"})  # registers the teardown for shared_client

    class _Failing:
        async def get(self, *_args, **_kwargs):
            raise OSError("connection refused")

    http_module.shared_client = lambda: _Failing()
    assert await check_for_new_version("1.2.3") is None


@pytest.mark.tonio
async def test_no_pi_dev_host_is_contacted(stub_http):
    client = stub_http({"tag_name": "1.2.4"})
    await get_latest_release("1.2.3")
    assert all("pi.dev" not in url for url, _headers in client.calls)


def test_release_url_constant_points_at_our_repo():
    assert RELEASES_URL == "https://github.com/gi0baro/pidrei/releases"


@pytest.mark.tonio
async def test_retries_a_transient_version_request_when_explicitly_requested(stub_http):
    stub_http({"tag_name": "1.2.4"})  # registers the teardown for shared_client

    class _FlakyClient:
        def __init__(self):
            self.calls = 0

        async def get(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls < 3:
                raise OSError("connection refused")
            return _Response({"tag_name": "1.2.4"})

    client = _FlakyClient()
    http_module.shared_client = lambda: client

    release = await get_latest_release("1.2.3", {"retry": True})

    assert release["version"] == "1.2.4"
    assert client.calls == 3


@pytest.mark.tonio
async def test_keeps_automatic_version_checks_to_one_request(stub_http):
    stub_http({"tag_name": "1.2.4"})  # registers the teardown for shared_client

    class _FailingClient:
        def __init__(self):
            self.calls = 0

        async def get(self, *_args, **_kwargs):
            self.calls += 1
            raise OSError("connection refused")

    client = _FailingClient()
    http_module.shared_client = lambda: client

    assert await check_for_new_version("1.2.3") is None
    assert client.calls == 1
