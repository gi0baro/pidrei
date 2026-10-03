"""The generic callback-server layer, pidrei-only (pi's lower half is node's
`http.createServer`).

The first test is the one place the httpunk `H1Server` path — head parsing,
the request record, `respond`, and keep-alive — is exercised end to end, with
punkreq as the browser. Added with the httpunk 0.3.0 bump (0.85.1.1), whose
exception refactor made the gap visible.
"""

import pytest

from pidrei_http import http
from pidrei_http.callback_server import CallbackResponse, start_callback_server


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


@pytest.mark.tonio
async def test_close_all_connections_drops_kept_alive_connections():
    """pidrei-only: the ChatGPT flow's `closeAllConnections()`. After close()
    alone a kept-alive connection still reaches the old handler (node's
    behaviour too); after close_all_connections() it does not."""
    seen = []

    async def handle(request):
        seen.append(request.get("code"))
        return CallbackResponse(status=200, html="ok")

    server = await start_callback_server(host="127.0.0.1", port=0, handle=handle)
    client = http.create_client(timeout=http.oneshot_timeout(5_000), trust_env=False)
    base = f"http://127.0.0.1:{server.port}/callback"
    try:
        await (await client.get(f"{base}?code=first")).read()
        server.close()
        await (await client.get(f"{base}?code=after-close")).read()
        assert seen == ["first", "after-close"]

        server.close_all_connections()
        with pytest.raises(Exception):
            await (await client.get(f"{base}?code=after-close-all")).read()
        assert seen == ["first", "after-close"]
    finally:
        await client.close()
        server.close()


@pytest.mark.tonio
async def test_after_sent_runs_once_the_page_is_written():
    """pidrei-only: the ChatGPT flow settles its result from `after_sent`, and
    the login it wakes drops every connection. The browser still gets the page."""
    servers = []

    async def handle(_request):
        return CallbackResponse(status=200, html="signed in", after_sent=servers[0].close_all_connections)

    servers.append(await start_callback_server(host="127.0.0.1", port=0, handle=handle))
    client = http.create_client(timeout=http.oneshot_timeout(5_000), trust_env=False)
    try:
        page = await client.get(f"http://127.0.0.1:{servers[0].port}/callback")
        assert page.status_code == 200
        assert await page.read() == b"signed in"
    finally:
        await client.close()
        servers[0].close()
