"""Mirror of pi mcp test/config.test.ts."""

from pidrei_mcp import (
    McpServerConfigOptions,
    McpServerConfigResult,
    is_mcp_server_override,
    validate_mcp_server_config,
)


def _error_of(result: McpServerConfigResult) -> str:
    """The error of an invalid entry, or the empty string."""
    return result["error"] if result["kind"] == "invalid" else ""


# validate_mcp_server_config


def test_accepts_stdio_and_http_entries_and_keeps_them_as_given():
    stdio = {"command": "npx", "args": ["-y", "server"], "env": {"A": "1"}, "cwd": "sub", "enabled": False}
    assert validate_mcp_server_config("files", stdio) == {"kind": "stdio", "config": stdio}
    assert validate_mcp_server_config("files", {"type": "stdio", "command": "x"})["kind"] == "stdio"

    http = {
        "url": "https://example.com/mcp",
        "headers": {"Authorization": "Bearer token"},
        "oauth": {"clientId": "id", "callbackPort": 8080, "scope": "read"},
        "timeout": 30,
        "description": "Docs",
    }
    result = validate_mcp_server_config("docs", http)
    assert result == {"kind": "http", "config": http}
    assert result["config"] is http
    for server_type in ("http", "streamable-http"):
        entry = {"type": server_type, "url": "http://localhost:3000/mcp"}
        assert validate_mcp_server_config("docs", entry)["kind"] == "http"


def test_picks_the_transport_from_type_url_and_command():
    # With both, url wins unless the type says stdio.
    assert validate_mcp_server_config("s", {"url": "https://a.example", "command": "x"})["kind"] == "http"
    assert (
        validate_mcp_server_config("s", {"type": "stdio", "url": "https://a.example", "command": "x"})["kind"]
        == "stdio"
    )
    assert _error_of(validate_mcp_server_config("s", {"type": "http", "command": "x"})) == (
        'server "s" needs either "command" (stdio) or "url" (streamable HTTP)'
    )
    assert _error_of(validate_mcp_server_config("s", {"type": "sse", "url": "https://a.example/sse"})) == (
        'server "s": legacy SSE transport is not supported; use the streamable HTTP URL'
    )


def test_mirrors_javascript_on_a_trailing_newline_and_a_null_type():
    # pidrei-only: where Python's idiom would differ from pi's `/^...$/` and `type === undefined`.
    assert "invalid server name" in _error_of(validate_mcp_server_config("files\n", {"command": "x"}))
    needs_transport = 'needs either "command" (stdio) or "url" (streamable HTTP)'
    assert needs_transport in _error_of(validate_mcp_server_config("files", {"type": None, "command": "x"}))
    assert needs_transport in _error_of(validate_mcp_server_config("docs", {"type": None, "url": "https://e.com/mcp"}))


def test_reports_invalid_names_and_fields():
    cases = [
        ("bad name", {"command": "x"}, 'invalid server name "bad name" (use letters, digits, "_" and "-")'),
        ("s", "npx", 'server "s" must be an object'),
        ("s", [], 'server "s" must be an object'),
        ("s", {"command": "x", "enabled": "yes"}, 'server "s": enabled must be a boolean'),
        ("s", {"command": "x", "description": 1}, 'server "s": description must be a string'),
        ("s", {"command": "x", "timeout": 0}, 'server "s": timeout must be a positive number of seconds'),
        ("s", {"command": "x", "args": "a b"}, 'server "s": args must be an array of strings'),
        ("s", {"command": "x", "env": {"A": 1}}, 'server "s": env must map names to strings'),
        ("s", {"command": "x", "cwd": 1}, 'server "s": cwd must be a string'),
        ("s", {"url": "file:///x"}, 'server "s": url must be an http or https URL'),
        ("s", {"url": "not a url"}, 'server "s": url must be an http or https URL'),
        ("s", {"url": "https://a.example", "headers": {"A": 1}}, 'server "s": headers must map names to strings'),
        ("s", {}, 'server "s" needs either "command" (stdio) or "url" (streamable HTTP)'),
    ]
    for name, raw, error in cases:
        assert _error_of(validate_mcp_server_config(name, raw)) == error, (name, raw)


def test_checks_oauth_settings():
    def oauth(settings) -> str:
        return _error_of(validate_mcp_server_config("s", {"url": "https://a.example", "oauth": settings}))

    assert oauth("x") == 'server "s": oauth must be an object'
    assert oauth({"clientId": 1}) == 'server "s": oauth.clientId must be a string'
    assert oauth({"clientSecret": 1}) == 'server "s": oauth.clientSecret must be a string'
    assert oauth({"callbackPort": 70000}) == 'server "s": oauth.callbackPort must be a port number'
    assert "oauth.callbackUrl must be an http URI" in oauth({"callbackUrl": "https://localhost/callback"})
    assert "oauth.callbackUrl must be an http URI" in oauth({"callbackUrl": "http://example.com/callback"})
    assert "oauth.callbackUrl must be an http URI" in oauth({"callbackUrl": "http://localhost/callback?x=1"})
    assert oauth({"callbackUrl": "http://[::1]:8080/cb", "callbackPort": 8080}) == ""
    assert oauth({"callbackUrl": "http://localhost:8080/cb", "callbackPort": 9090}) == (
        'server "s": oauth.callbackUrl and oauth.callbackPort name different ports'
    )
    assert oauth({"scope": 1}) == 'server "s": oauth.scope must be a string'
    assert oauth({"clientName": " "}) == 'server "s": oauth.clientName must be a non-empty string'
    assert oauth({"clientRegistration": "x"}) == 'server "s": oauth.clientRegistration must be "auto", "dcr", or "cimd"'
    assert oauth({"clientRegistration": "dcr", "clientName": "app"}) == ""
    assert "oauth.authServerMetadataUrl must be an https URL" in oauth(
        {"authServerMetadataUrl": "http://auth.example/.well-known/x"}
    )
    assert oauth({"authServerMetadataUrl": "http://127.0.0.1/.well-known/x"}) == ""


def test_allows_cimd_only_with_the_applications_check_of_the_callback():
    def entry(oauth: dict) -> dict:
        return {"url": "https://a.example", "oauth": oauth}

    assert _error_of(validate_mcp_server_config("s", entry({"clientRegistration": "cimd"}))) == (
        'server "s": oauth.clientRegistration "cimd" is not supported: there are no Client ID Metadata Documents'
    )
    checked: list[str | None] = []

    def check(callback_url: str | None) -> str | None:
        checked.append(callback_url)
        return "the documents do not list it" if callback_url and callback_url.endswith("/other") else None

    options = McpServerConfigOptions(check_metadata_document_callback=check)
    assert validate_mcp_server_config("s", entry({"clientRegistration": "cimd"}), options)["kind"] == "http"
    assert _error_of(
        validate_mcp_server_config(
            "s", entry({"clientRegistration": "cimd", "callbackUrl": "http://localhost/other"}), options
        )
    ) == ('server "s": the documents do not list it')
    assert checked == [None, "http://localhost/other"]
    assert _error_of(
        validate_mcp_server_config("s", entry({"clientRegistration": "cimd", "clientId": "id"}), options)
    ) == ('server "s": oauth.clientRegistration "cimd" cannot be combined with oauth.clientId or oauth.clientName')


# extending validate_mcp_server_config


def _validate_app_config(name: str, raw) -> str:
    """An application's own fields, like p2's `shared` and `runOn`, checked on the result."""
    result = validate_mcp_server_config(name, raw)
    if result["kind"] == "invalid":
        return result["error"]
    config = result["config"]
    shared = config.get("shared")
    if "shared" in config and not isinstance(shared, bool):
        return f'server "{name}": shared must be a boolean'
    if result["kind"] == "http" and "runOn" in config:
        return f'server "{name}": runOn is for stdio'
    # pi's `String(shared)`.
    return f"{result['kind']} shared={'undefined' if 'shared' not in config else str(shared).lower()}"


def test_keeps_unknown_fields_for_the_caller_after_the_shared_fields_are_valid():
    assert _validate_app_config("s", {"command": "x", "shared": True, "runOn": "host"}) == "stdio shared=true"
    assert _validate_app_config("s", {"url": "https://a.example", "runOn": "host"}) == 'server "s": runOn is for stdio'
    assert _validate_app_config("s", {"command": "x", "shared": 1}) == 'server "s": shared must be a boolean'
    # The shared fields are reported first.
    assert _validate_app_config("s", {"command": "x", "shared": 1, "enabled": "no"}) == (
        'server "s": enabled must be a boolean'
    )


# is_mcp_server_override


def test_is_true_for_entries_without_command_url_or_type():
    assert is_mcp_server_override({"enabled": False}) is True
    assert is_mcp_server_override({}) is True
    assert is_mcp_server_override({"command": "x"}) is False
    assert is_mcp_server_override({"url": "https://a.example"}) is False
    assert is_mcp_server_override({"type": "stdio", "enabled": False}) is False
