"""Fake Mistral transport for the native-transport test mirrors.

pi's tests inject a `fetch` returning a `Response`; pidrei's seam is
`MistralOptions.client`. The fake records the request the adapter would have
sent (URL, wire payload, headers) and replays canned SSE bytes.
"""

from typing import Any


def sse_body(data_lines: list[str], *, separator: str = "\r\n\r\n", done: bool = True) -> bytes:
    text = separator.join(f"data: {line}" for line in data_lines)
    if done:
        text = f"{text}{separator}data: [DONE]{separator}"
    return text.encode()


class FakeMistralResponse:
    def __init__(
        self,
        body: bytes = b"",
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        *,
        chunks: list[bytes] | None = None,
        stall_forever: bool = False,
    ):
        self.status_code = status_code
        self.headers = {"content-type": "text/event-stream", **(headers or {})}
        self._chunks = chunks if chunks is not None else ([body] if body else [])
        self._stall_forever = stall_forever

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk
        if self._stall_forever:
            import tonio.colored as tonio

            await tonio.Event().wait()

    async def read(self) -> bytes:
        return b"".join(self._chunks)


class FakeMistralClient:
    def __init__(self, response: FakeMistralResponse | None = None, *, body: bytes | None = None):
        self.response = response if response is not None else FakeMistralResponse(body or b"")
        self.requests: list[dict[str, Any]] = []

    async def post_chat_completions(self, url, wire_payload, *, headers, timeout_ms, cancel):
        self.requests.append({"url": url, "payload": wire_payload, "headers": headers, "timeout_ms": timeout_ms})
        return self.response
