"""Mirror of pi mcp src/oauth/errors.ts."""

import json


class OAuthError(Exception):
    def __init__(self, code: str, message: str, error_uri: str | None = None) -> None:
        super().__init__(message or code)
        self.code = code
        self.error_uri = error_uri


class OAuthIssuerMismatchError(Exception):
    def __init__(self, expected: str, received: str | None) -> None:
        # `received` is None when an authorization response lacks the `iss`
        # parameter its server promised (RFC 9207).
        shown = "none" if received is None else json.dumps(received, ensure_ascii=False)
        super().__init__(
            f"OAuth issuer mismatch: expected {json.dumps(expected, ensure_ascii=False)}, received {shown}"
        )
        self.expected = expected
        self.received = received


class OAuthInsecureEndpointError(Exception):
    def __init__(self, endpoint: str) -> None:
        super().__init__(f"Refusing to send OAuth credentials to non-HTTPS endpoint {endpoint}")
        self.endpoint = endpoint


class OAuthRegistrationError(Exception):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"OAuth dynamic client registration failed with status {status}: {body}")
        self.status = status
        self.body = body


class McpOAuthAuthorizationRequiredError(Exception):
    def __init__(self) -> None:
        super().__init__("MCP OAuth authorization requires user interaction")
