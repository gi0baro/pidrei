"""Mirror of pi mcp src/oauth/types.ts.

Adapted from modelcontextprotocol/typescript-sdk v1.29.0.
Copyright (c) 2024 Anthropic, PBC. Licensed under MIT; see LICENSES/.
Modified to use dependency-free structural validation.
"""

import math
from typing import Any, NotRequired, TypedDict

from ..protocol.jsonrpc import is_number, is_object
from ..url import parse_url


# Both metadata documents keep the fields they do not declare (pi's index signature).
class OAuthProtectedResourceMetadata(TypedDict):
    resource: str
    authorization_servers: NotRequired[list[str]]
    scopes_supported: NotRequired[list[str]]


class AuthorizationServerMetadata(TypedDict):
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: NotRequired[str]
    scopes_supported: NotRequired[list[str]]
    response_types_supported: list[str]
    grant_types_supported: NotRequired[list[str]]
    token_endpoint_auth_methods_supported: NotRequired[list[str]]
    code_challenge_methods_supported: NotRequired[list[str]]
    client_id_metadata_document_supported: NotRequired[bool]
    # Whether authorization responses carry an `iss` parameter (RFC 9207).
    authorization_response_iss_parameter_supported: NotRequired[bool]


class OAuthTokens(TypedDict):
    access_token: str
    token_type: str
    expires_in: NotRequired[float]
    scope: NotRequired[str]
    refresh_token: NotRequired[str]
    id_token: NotRequired[str]


class OAuthClientMetadata(TypedDict, total=False):
    redirect_uris: list[str]
    token_endpoint_auth_method: str
    grant_types: list[str]
    response_types: list[str]
    # OpenID Connect client type, `native` or `web`. Dynamic client registration
    # derives it from `redirect_uris` when absent (MCP SEP-837).
    application_type: str
    client_name: str
    client_uri: str
    logo_uri: str
    scope: str
    contacts: list[str]
    tos_uri: str
    policy_uri: str
    jwks_uri: str
    jwks: Any
    software_id: str
    software_version: str
    software_statement: str


class OAuthClientInformation(TypedDict):
    client_id: str
    client_secret: NotRequired[str]
    client_id_issued_at: NotRequired[float]
    client_secret_expires_at: NotRequired[float]


class OAuthClientInformationFull(OAuthClientInformation, OAuthClientMetadata):
    pass


type OAuthClientInformationMixed = OAuthClientInformation | OAuthClientInformationFull


class OAuthDiscoveryState(TypedDict):
    authorizationServerUrl: str
    authorizationServerMetadata: NotRequired[AuthorizationServerMetadata]
    resourceMetadata: NotRequired[OAuthProtectedResourceMetadata]
    resourceMetadataUrl: NotRequired[str]


class OAuthServerInfo(TypedDict):
    authorizationServerUrl: str
    authorizationServerMetadata: NotRequired[AuthorizationServerMetadata]
    resourceMetadata: NotRequired[OAuthProtectedResourceMetadata]


class OAuthChallenge(TypedDict, total=False):
    resourceMetadataUrl: str
    scope: str
    error: str
    errorDescription: str


def _object(value: Any, name: str) -> dict[str, Any]:
    if not is_object(value):
        raise ValueError(f"Invalid {name}")
    return value


def _compact(value: dict[str, Any]) -> dict[str, Any]:
    """Drops absent values so optional fields are missing rather than present-but-undefined."""
    return {key: item for key, item in value.items() if item is not _ABSENT}


class _Absent:
    __slots__ = ()


# pi's `undefined` in the objects `compact` filters: a parsed field that came
# out empty. JSON `null` is a value of its own there, so None cannot stand in.
_ABSENT: Any = _Absent()


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid {name}")
    return value


def _absent(value: Any) -> bool:
    """`null` and `""` count as absent: servers send them for fields they
    have no value for, like `scope: ""`."""
    return value is _ABSENT or value is None or value == ""


def _optional_string(value: Any, name: str) -> Any:
    if _absent(value):
        return _ABSENT
    return _required_string(value, name)


def _optional_strings(value: Any, name: str) -> Any:
    if value is _ABSENT or value is None:
        return _ABSENT
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Invalid {name}")
    return list(value)


def _safe_url(value: Any, name: str) -> str:
    text = _required_string(value, name)
    try:
        url = parse_url(text)
    except ValueError:
        raise ValueError(f"Invalid {name}") from None
    if url.protocol in ("javascript:", "data:", "vbscript:"):
        raise ValueError(f"Invalid {name}")
    return text


def _optional_url(value: Any, name: str) -> Any:
    return _ABSENT if _absent(value) else _safe_url(value, name)


def _field(input: dict[str, Any], key: str) -> Any:
    return input.get(key, _ABSENT)


def _js_number(value: Any) -> float:
    """`Number(value)` for a parsed JSON value."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if is_number(value):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0.0
        if "_" in text:
            # `float` and `int` accept digit separators; `Number` does not.
            return math.nan
        try:
            return float(int(text, 0)) if text.lower().startswith(("0x", "0o", "0b")) else float(text)
        except ValueError:
            return math.nan
    return math.nan


def parse_protected_resource_metadata(value: Any) -> OAuthProtectedResourceMetadata:
    input = _object(value, "OAuth protected resource metadata")
    authorization_servers = _optional_strings(_field(input, "authorization_servers"), "authorization_servers")
    if authorization_servers is not _ABSENT:
        authorization_servers = [_safe_url(url, "authorization server URL") for url in authorization_servers]
    return _compact(
        {
            **input,
            "resource": _safe_url(_field(input, "resource"), "OAuth protected resource metadata resource"),
            "authorization_servers": authorization_servers,
            "scopes_supported": _optional_strings(_field(input, "scopes_supported"), "scopes_supported"),
        }
    )  # type: ignore[return-value]


def parse_authorization_server_metadata(value: Any) -> AuthorizationServerMetadata:
    input = _object(value, "authorization server metadata")
    response_types = _optional_strings(_field(input, "response_types_supported"), "response_types_supported")
    if response_types is _ABSENT:
        raise ValueError("Invalid response_types_supported")

    def boolean(key: str) -> Any:
        item = input.get(key)
        return item if isinstance(item, bool) else _ABSENT

    return _compact(
        {
            **input,
            "issuer": _safe_url(_field(input, "issuer"), "authorization server issuer"),
            "authorization_endpoint": _safe_url(_field(input, "authorization_endpoint"), "authorization endpoint"),
            "token_endpoint": _safe_url(_field(input, "token_endpoint"), "token endpoint"),
            "registration_endpoint": _optional_url(_field(input, "registration_endpoint"), "registration endpoint"),
            "scopes_supported": _optional_strings(_field(input, "scopes_supported"), "scopes_supported"),
            "response_types_supported": response_types,
            "grant_types_supported": _optional_strings(_field(input, "grant_types_supported"), "grant_types_supported"),
            "token_endpoint_auth_methods_supported": _optional_strings(
                _field(input, "token_endpoint_auth_methods_supported"), "token_endpoint_auth_methods_supported"
            ),
            "code_challenge_methods_supported": _optional_strings(
                _field(input, "code_challenge_methods_supported"), "code_challenge_methods_supported"
            ),
            "client_id_metadata_document_supported": boolean("client_id_metadata_document_supported"),
            "authorization_response_iss_parameter_supported": boolean("authorization_response_iss_parameter_supported"),
        }
    )  # type: ignore[return-value]


def parse_oauth_tokens(value: Any) -> OAuthTokens:
    input = _object(value, "OAuth token response")
    # `Number(null)` is 0, which would mark the token as expired at once.
    raw_expires = _field(input, "expires_in")
    expires: Any = _ABSENT if _absent(raw_expires) else _js_number(raw_expires)
    if expires is not _ABSENT and not math.isfinite(expires):
        raise ValueError("Invalid expires_in")
    return _compact(
        {
            "access_token": _required_string(_field(input, "access_token"), "access_token"),
            "token_type": _required_string(_field(input, "token_type"), "token_type"),
            "expires_in": expires,
            "scope": _optional_string(_field(input, "scope"), "scope"),
            "refresh_token": _optional_string(_field(input, "refresh_token"), "refresh_token"),
            "id_token": _optional_string(_field(input, "id_token"), "id_token"),
        }
    )  # type: ignore[return-value]


def parse_client_information(value: Any) -> OAuthClientInformationFull:
    input = _object(value, "OAuth client registration response")
    redirect_uris = _optional_strings(_field(input, "redirect_uris"), "redirect_uris")

    def number(key: str) -> Any:
        item = input.get(key)
        return item if is_number(item) else _ABSENT

    return _compact(
        {
            **input,
            "client_id": _required_string(_field(input, "client_id"), "client_id"),
            "client_secret": _optional_string(_field(input, "client_secret"), "client_secret"),
            "client_id_issued_at": number("client_id_issued_at"),
            "client_secret_expires_at": number("client_secret_expires_at"),
            "redirect_uris": [] if redirect_uris is _ABSENT else redirect_uris,
        }
    )  # type: ignore[return-value]
