"""Mirror of pi mcp src/oauth/discovery.ts.

Adapted from modelcontextprotocol/typescript-sdk v1.29.0 src/client/auth.ts.
Copyright (c) 2024 Anthropic, PBC. Licensed under MIT; see LICENSES/.
Modified to remove Zod/CORS shims and enforce authorization-server issuer validation.

A network failure (fetch's `TypeError` in pi) is `pidrei_http.http.TransportError`
here, or an `OSError` from an injected fetch.
"""

import re
from typing import Any, Literal

from pidrei_http import http

from ..fetch import McpFetch, McpResponse, default_fetch
from ..protocol.jsonrpc import parse_json
from ..protocol.types import LATEST_PROTOCOL_VERSION
from ..url import Url, parse_url
from .errors import OAuthIssuerMismatchError
from .types import (
    AuthorizationServerMetadata,
    OAuthChallenge,
    OAuthProtectedResourceMetadata,
    OAuthServerInfo,
    parse_authorization_server_metadata,
    parse_protected_resource_metadata,
)


def _discard(response: McpResponse) -> None:
    """pi cancels the body without waiting for it."""
    http.abandon_response(response)


def _ok(response: McpResponse) -> bool:
    return 200 <= response.status < 300


async def _json(response: McpResponse) -> Any:
    return parse_json(await response.read())


def is_network_failure(error: BaseException) -> bool:
    return isinstance(error, http.TransportError | OSError)


def _is_discovery_miss(status: int) -> bool:
    """4xx and 502 mean "not here", so discovery tries the next candidate URL."""
    return 400 <= status < 500 or status == 502


def _path_suffix(pathname: str) -> str:
    """Path suffix for `/.well-known/<kind><path>`; empty for the root path."""
    return pathname.removesuffix("/")


def _field(header: str, name: str) -> str | None:
    match = re.search(rf'(?:^|[,\s]){name}=(?:"([^"]*)"|([^\s,]+))', header, re.IGNORECASE)
    # An empty value (`scope=""`) carries no information, so it counts as absent.
    if match is None:
        return None
    return match.group(1) or match.group(2) or None


def parse_www_authenticate(header: str | None) -> OAuthChallenge:
    if not header:
        return {}
    words = header.lstrip().split(None, 1)
    scheme = words[0].lower() if words else ""
    if scheme not in ("bearer", "dpop"):
        return {}
    challenge: OAuthChallenge = {}
    resource_metadata = _field(header, "resource_metadata")
    if resource_metadata:
        try:
            challenge["resourceMetadataUrl"] = parse_url(resource_metadata).href
        except ValueError:
            pass
    for key, name in (("scope", "scope"), ("error", "error"), ("errorDescription", "error_description")):
        value = _field(header, name)
        if value is not None:
            challenge[key] = value  # type: ignore[literal-required]
    return challenge


def _fetch_metadata(url: Url, fetch: McpFetch, protocol_version: str) -> Any:
    return fetch(url.href, headers={"Accept": "application/json", "MCP-Protocol-Version": protocol_version})


async def discover_protected_resource_metadata(
    server_url: str,
    *,
    resource_metadata_url: str | None = None,
    protocol_version: str | None = None,
    fetch: McpFetch | None = None,
) -> OAuthProtectedResourceMetadata:
    server = parse_url(server_url)
    fetch = fetch if fetch is not None else default_fetch
    version = protocol_version or LATEST_PROTOCOL_VERSION
    if resource_metadata_url:
        url = parse_url(resource_metadata_url)
    else:
        url = parse_url(f"/.well-known/oauth-protected-resource{_path_suffix(server.pathname)}", server.origin)
    response = await _fetch_metadata(url, fetch, version)
    if not resource_metadata_url and server.pathname != "/" and _is_discovery_miss(response.status):
        _discard(response)
        response = await _fetch_metadata(
            parse_url("/.well-known/oauth-protected-resource", server.origin), fetch, version
        )
    if not _ok(response):
        _discard(response)
        raise RuntimeError(f"HTTP {response.status} loading OAuth protected resource metadata")
    return parse_protected_resource_metadata(await _json(response))


def build_authorization_server_discovery_urls(
    authorization_server_url: str,
) -> list[tuple[Url, Literal["oauth", "oidc"]]]:
    issuer = parse_url(authorization_server_url)
    path = _path_suffix(issuer.pathname)
    urls: list[tuple[Url, Literal["oauth", "oidc"]]] = [
        (parse_url(f"/.well-known/oauth-authorization-server{path}", issuer.origin), "oauth"),
        (parse_url(f"/.well-known/openid-configuration{path}", issuer.origin), "oidc"),
    ]
    if path:
        urls.append((parse_url(f"{path}/.well-known/openid-configuration", issuer.origin), "oidc"))
    return urls


async def discover_authorization_server_metadata(
    authorization_server_url: str,
    *,
    fetch: McpFetch | None = None,
    protocol_version: str | None = None,
    skip_issuer_validation: bool = False,
) -> AuthorizationServerMetadata | None:
    fetch = fetch if fetch is not None else default_fetch
    for url, _type in build_authorization_server_discovery_urls(authorization_server_url):
        response = await _fetch_metadata(url, fetch, protocol_version or LATEST_PROTOCOL_VERSION)
        if not _ok(response):
            _discard(response)
            if _is_discovery_miss(response.status):
                continue
            raise RuntimeError(f"HTTP {response.status} loading authorization server metadata from {url.href}")
        metadata = parse_authorization_server_metadata(await _json(response))
        if not skip_issuer_validation:
            expected = authorization_server_url
            # URL parsing adds a trailing slash to bare origins, so compare without one on either side.
            if metadata["issuer"].removesuffix("/") != expected.removesuffix("/"):
                raise OAuthIssuerMismatchError(expected, metadata["issuer"])
        return metadata
    return None


async def discover_oauth_server_info(
    server_url: str,
    *,
    resource_metadata_url: str | None = None,
    authorization_server_metadata_url: str | None = None,
    fetch: McpFetch | None = None,
    skip_issuer_validation: bool = False,
) -> OAuthServerInfo:
    """`authorization_server_metadata_url` is a metadata document to use
    instead of discovery. It is trusted as configured, so its issuer is not
    checked."""
    resource_metadata: OAuthProtectedResourceMetadata | None = None
    try:
        resource_metadata = await discover_protected_resource_metadata(
            server_url, resource_metadata_url=resource_metadata_url, fetch=fetch
        )
    except Exception as error:
        if is_network_failure(error):
            raise
    if authorization_server_metadata_url:
        url = parse_url(authorization_server_metadata_url)
        response = await _fetch_metadata(url, fetch if fetch is not None else default_fetch, LATEST_PROTOCOL_VERSION)
        if not _ok(response):
            _discard(response)
            raise RuntimeError(f"HTTP {response.status} loading authorization server metadata from {url.href}")
        metadata = parse_authorization_server_metadata(await _json(response))
        info: OAuthServerInfo = {
            "authorizationServerUrl": metadata["issuer"],
            "authorizationServerMetadata": metadata,
        }
        if resource_metadata is not None:
            info["resourceMetadata"] = resource_metadata
        return info
    servers = resource_metadata.get("authorization_servers") if resource_metadata is not None else None
    authorization_server_url = servers[0] if servers else parse_url("/", server_url).href
    info = {"authorizationServerUrl": authorization_server_url}
    discovered = await discover_authorization_server_metadata(
        authorization_server_url, fetch=fetch, skip_issuer_validation=skip_issuer_validation
    )
    if discovered is not None:
        info["authorizationServerMetadata"] = discovered
    if resource_metadata is not None:
        info["resourceMetadata"] = resource_metadata
    return info


def resource_url_from_server_url(value: str) -> Url:
    return parse_url(value).without_fragment()


def select_resource(server_url: str, metadata: OAuthProtectedResourceMetadata | None = None) -> str | None:
    if metadata is None:
        return None
    requested = resource_url_from_server_url(server_url)
    configured = parse_url(metadata["resource"])
    if requested.origin != configured.origin:
        raise RuntimeError(f"Protected resource {metadata['resource']} does not match MCP server {requested.href}")
    requested_path = requested.pathname if requested.pathname.endswith("/") else f"{requested.pathname}/"
    configured_path = configured.pathname if configured.pathname.endswith("/") else f"{configured.pathname}/"
    if not requested_path.startswith(configured_path):
        raise RuntimeError(f"Protected resource {metadata['resource']} does not match MCP server {requested.href}")
    return metadata["resource"]
