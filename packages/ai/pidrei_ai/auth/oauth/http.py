"""The one request function every OAuth flow goes through.

pi's flows call the global `fetch` and its tests replace it with
`vi.stubGlobal("fetch", ...)`. There is no global to replace here, so the flows
call `oauth_http.request(...)` module-qualified and the mirrors substitute this
one attribute — the same single point of interception, and the punkreq seam
stays in `utils/http.py`.

Cancellation raises `AbortError` rather than a flow's wording: each flow turns
it into its own message ("Login cancelled", "…refresh aborted"), exactly as
pi's `if (signal?.aborted)` branches do.
"""

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

from pidrei_ai.utils import http
from pidrei_ai.utils.abort import run_cancellable
from pidrei_ai.utils.cancel import CancelToken


@dataclass(slots=True)
class OAuthHttpResponse:
    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """`Response.ok`: 2xx."""
        return 200 <= self.status < 300

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", "replace")

    def json(self) -> Any:
        """`await response.json()`; raises on a body that is not JSON."""
        return json.loads(self.text)

    def json_object(self) -> dict[str, Any] | None:
        """The body as a JSON object, or None when it is not one.

        Mirrors pi's repeated `parsed && typeof parsed === "object" &&
        !Array.isArray(parsed)` guard, including its tolerance of a body that
        does not parse at all.
        """
        try:
            parsed = self.json()
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None


async def request(
    url: str,
    *,
    method: str = "POST",
    headers: dict[str, str] | None = None,
    json_body: Any = None,
    form: dict[str, str] | None = None,
    timeout_ms: float | None = None,
    cancel: CancelToken | None = None,
) -> OAuthHttpResponse:
    """Perform one OAuth request and read the whole body.

    `form` is sent as `application/x-www-form-urlencoded`; the caller still sets
    its own `Content-Type` header so the mirrors can assert pi's exact casing.
    """
    content = urlencode(form).encode("utf-8") if form is not None else None

    async def _send() -> OAuthHttpResponse:
        client = http.shared_client()
        response = await client.request(
            method,
            url,
            headers=headers,
            json=json_body,
            content=content,
            timeout=http.oneshot_timeout(timeout_ms),
        )
        body = await response.read()
        return OAuthHttpResponse(
            status=response.status_code,
            body=body if isinstance(body, bytes) else str(body).encode("utf-8"),
            headers={name.lower(): value for name, value in response.headers.items()},
        )

    # The request is unwound at its current await when the token fires.
    return await run_cancellable(_send(), cancel)
