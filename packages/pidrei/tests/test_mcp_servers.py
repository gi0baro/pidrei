"""pidrei-only: MCP server configs and the registry behind `pi.register_mcp_server()`.

pi tests config validation through its MCP extension and `pi mcp` CLI, which
port later with codemode; the validation cases are here.

The registry step once an extension is loaded: pi checks the owner and the
namespace, then registers, as separate statements on its one thread; here
extensions register in parallel, so it is one step."""

import pytest

from pidrei.core.mcp_servers import McpServerRegistry, RegisteredMcpServer, validate_mcp_server_config


def test_claim_leaves_a_name_another_extension_owns():
    registry = McpServerRegistry()
    changes = []
    registry.set_change_listener(lambda: changes.append(1))
    first = RegisteredMcpServer(name="docs", config={"command": "one"}, extension_path="/ext/a.py")

    assert registry.claim(first) is None
    assert registry.claim(RegisteredMcpServer(name="docs", config={"command": "two"}, extension_path="/ext/b.py")) == (
        first
    )
    assert registry.get("docs") == first
    assert len(changes) == 1

    again = RegisteredMcpServer(name="docs", config={"command": "three"}, extension_path="/ext/a.py")
    assert registry.claim(again) is None
    assert registry.get("docs") == again
    assert len(changes) == 2


def test_claim_refuses_a_name_that_differs_only_in_dash_and_underscore():
    registry = McpServerRegistry()
    first = RegisteredMcpServer(name="dev-docs", config={"command": "one"}, extension_path="/ext/a.py")
    assert registry.claim(first) is None

    clash = RegisteredMcpServer(name="dev_docs", config={"command": "two"}, extension_path="/ext/a.py")
    assert registry.conflict_of(clash) == first
    assert registry.claim(clash) == first
    assert registry.get("dev_docs") is None
    other = RegisteredMcpServer(name="dev-docs-2", config={"command": "three"}, extension_path="/ext/b.py")
    assert registry.claim(other) is None


def test_resolves_the_codemode_deferred_alias_without_changing_the_input():
    raw = {"command": "srv", "exposure": "codemode-deferred", "toolExposure": {"a": "codemode-deferred", "b": "direct"}}

    validated = validate_mcp_server_config("docs", raw)

    assert validated == {"command": "srv", "exposure": "codemode", "toolExposure": {"a": "codemode", "b": "direct"}}
    assert raw["exposure"] == "codemode-deferred"
    assert raw["toolExposure"]["a"] == "codemode-deferred"
    assert validate_mcp_server_config("docs", {"command": "srv", "exposure": "lazy"}) == (
        'server "docs": exposure must be one of "codemode", "deferred", "direct", "hidden"'
    )


def test_validates_the_description():
    assert validate_mcp_server_config("docs", {"command": "srv", "description": "Product docs"}) == {
        "command": "srv",
        "description": "Product docs",
    }
    assert validate_mcp_server_config("docs", {"command": "srv", "description": 1}) == (
        'server "docs": description must be a string'
    )


@pytest.mark.parametrize("client_name", ["", "  ", 1])
def test_rejects_an_empty_or_non_string_oauth_client_name(client_name):
    config = {"url": "https://mcp.example.com", "oauth": {"clientName": client_name}}
    assert validate_mcp_server_config("docs", config) == 'server "docs": oauth.clientName must be a non-empty string'


def test_rejects_a_null_oauth():
    # pi skips only an absent `oauth` (`undefined`); null is not an object.
    config = {"url": "https://mcp.example.com", "oauth": None}
    assert validate_mcp_server_config("docs", config) == 'server "docs": oauth must be an object'


@pytest.mark.parametrize(
    ("callback_url", "valid"),
    [
        # Port 80 is http's default, so the URL names no port (`new URL().port` is "").
        ("http://localhost:80/cb", True),
        ("http://localhost/cb", True),
        ("http://localhost:9000/cb", True),
        ("http://localhost:8080/cb", False),
    ],
)
def test_compares_the_callback_url_port_as_url_parsing_reports_it(callback_url, valid):
    config = {"url": "https://mcp.example.com", "oauth": {"callbackUrl": callback_url, "callbackPort": 9000}}
    expected = config if valid else 'server "docs": oauth.callbackUrl and oauth.callbackPort name different ports'
    assert validate_mcp_server_config("docs", config) == expected


def test_accepts_an_oauth_client_name():
    config = {"url": "https://mcp.example.com", "oauth": {"clientName": "Claude"}}
    assert validate_mcp_server_config("docs", config) == config


@pytest.mark.parametrize(
    ("metadata_url", "valid"),
    [
        ("https://auth.example.com/.well-known/oauth-authorization-server", True),
        ("http://localhost:9000/.well-known/openid-configuration", True),
        ("http://[::1]:9000/metadata", True),
        ("http://auth.example.com/metadata", False),
        ("not a url", False),
        (1, False),
    ],
)
def test_validates_the_oauth_auth_server_metadata_url(metadata_url, valid):
    config = {"url": "https://mcp.example.com", "oauth": {"authServerMetadataUrl": metadata_url}}
    expected = (
        config
        if valid
        else 'server "docs": oauth.authServerMetadataUrl must be an https URL, or http on localhost, 127.0.0.1, or [::1]'
    )
    assert validate_mcp_server_config("docs", config) == expected


@pytest.mark.parametrize("auth", [None, "anthropic", {}, {"provider": ""}, {"provider": 1}])
def test_rejects_auth_without_a_provider_name(auth):
    config = {"url": "https://mcp.example.com", "auth": auth}
    assert validate_mcp_server_config("docs", config) == 'server "docs": auth.provider must be a provider name'


@pytest.mark.parametrize(
    ("url", "valid"),
    [
        ("https://mcp.example.com", True),
        ("http://127.0.0.1:8080/mcp", True),
        ("http://[::1]/mcp", True),
        ("http://mcp.example.com", False),
    ],
)
def test_requires_https_or_loopback_for_provider_auth(url, valid):
    config = {"url": url, "auth": {"provider": "anthropic"}}
    expected = (
        config if valid else 'server "docs": auth requires an https URL, or http on localhost, 127.0.0.1, or [::1]'
    )
    assert validate_mcp_server_config("docs", config) == expected
