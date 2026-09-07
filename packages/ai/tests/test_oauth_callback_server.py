"""pidrei-only: the OAuth loopback callback server against the real HTTP stack.

No pi counterpart (pi's is node's `http.createServer`). The flows' unit tests
stub the server, so this is the one place `auth/oauth/callback_server.py`'s
httpunk `H1Server` path — head parsing, the request record, `respond`, and
keep-alive — is exercised end to end, with punkreq as the browser. Added with
the httpunk 0.3.0 bump (0.85.1.1), whose exception refactor made the gap
visible.
"""

import pytest

from pidrei_ai.auth.oauth.callback_server import CallbackResponse, start_callback_server
from pidrei_ai.utils import http


@pytest.mark.tonio
async def test_serves_a_browser_redirect_over_a_real_socket_and_keeps_the_connection_alive():
    seen = []

    async def handle(request):
        seen.append(request)
        if request.path == "/boom":
            raise RuntimeError("handler crashed")
        return CallbackResponse(status=200, html=f"<p>code={request.get('code')}</p>")

    server = await start_callback_server(host="127.0.0.1", port=0, handle=handle)
    client = http.create_client(timeout=http.oneshot_timeout(5_000), trust_env=False)
    try:
        base = f"http://127.0.0.1:{server.port}"
        first = await client.get(f"{base}/callback?code=abc&state=xyz&empty=")
        assert first.status_code == 200
        assert first.headers["content-type"].startswith("text/html")
        assert await first.read() == b"<p>code=abc</p>"

        # A second request reuses the pooled keep-alive connection.
        second = await client.get(f"{base}/callback?code=def")
        assert second.status_code == 200
        assert await second.read() == b"<p>code=def</p>"

        assert [(r.method, r.path, r.query) for r in seen] == [
            ("GET", "/callback", {"code": "abc", "state": "xyz", "empty": ""}),
            ("GET", "/callback", {"code": "def"}),
        ]

        # A handler crash drops that connection; the server keeps accepting.
        with pytest.raises(Exception):
            await (await client.get(f"{base}/boom")).read()
        third = await client.get(f"{base}/callback?code=ghi")
        assert third.status_code == 200
        assert await third.read() == b"<p>code=ghi</p>"
    finally:
        await client.close()
        server.close()
