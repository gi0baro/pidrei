"""Port of pi's PKCE helper (packages/ai/src/auth/oauth/pkce.ts).

pi's `generatePKCE` is async only because it reaches for `crypto.subtle.digest`;
`hashlib` is synchronous, so this one is too — every call site awaited it purely
to satisfy the Web Crypto API.
"""

import base64
import hashlib
import secrets
from dataclasses import dataclass


def base64url_encode(data: bytes) -> str:
    """Unpadded base64url, matching pi's `base64urlEncode`."""
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


@dataclass(slots=True)
class Pkce:
    verifier: str
    challenge: str


def generate_pkce() -> Pkce:
    """A PKCE verifier and its S256 challenge."""
    verifier = base64url_encode(secrets.token_bytes(32))
    challenge = base64url_encode(hashlib.sha256(verifier.encode("utf-8")).digest())
    return Pkce(verifier=verifier, challenge=challenge)
