"""Mirror of pi coding-agent test/auth-check.test.ts."""

import json
import time

import pytest

from pidrei.cli.args import parse_args
from pidrei.cli.auth_check import (
    AuthCheckResult,
    check_provider_auth,
    create_auth_check_model_runtime,
    get_provider_credential,
)
from pidrei.cli.auth_command import parse_auth_command
from pidrei.core.auth_storage import AuthStorage, ReadOnlyAuthStorage
from pidrei.core.model_runtime import ModelRuntime
from pidrei.core.models_store import InMemoryCodingAgentModelsStore
from pidrei_ai.auth.types import ApiKeyCredential, OAuthCredential


async def create_runtime(credentials) -> ModelRuntime:
    return await ModelRuntime.create(
        credentials=credentials,
        models_path=None,
        models_store=InMemoryCodingAgentModelsStore(),
        allow_model_network=False,
        refresh_on_create=False,
    )


@pytest.mark.tonio
async def test_reports_a_configured_provider_as_ready():
    runtime = await create_runtime(AuthStorage.in_memory({"openai": ApiKeyCredential(key="test-key")}))

    assert await check_provider_auth(parse_args(["--provider", "openai"]), runtime) == AuthCheckResult(
        status="ready", provider="openai", auth_type="api_key"
    )


@pytest.mark.tonio
async def test_resolves_the_provider_from_model():
    runtime = await create_runtime(AuthStorage.in_memory({"openai": ApiKeyCredential(key="test-key")}))

    assert await check_provider_auth(parse_args(["--model", "openai/gpt-5.5"]), runtime) == AuthCheckResult(
        status="ready", provider="openai", auth_type="api_key"
    )
    result = await check_provider_auth(parse_args(["--provider", "openai", "--model", "gpt-5.5"]), runtime)
    assert result.status == "ready"
    assert result.provider == "openai"


@pytest.mark.tonio
async def test_reads_credentials_without_refreshing_oauth_when_requested():
    api_credentials = AuthStorage.in_memory({"openai": ApiKeyCredential(key="test-key")})
    api_runtime = await create_runtime(api_credentials)
    assert await get_provider_credential("openai", api_runtime, api_credentials, refresh=False) == "test-key"

    credentials = AuthStorage.in_memory(
        {"openai-codex": OAuthCredential(access="old-token", refresh="refresh-token", expires=0)}
    )
    oauth_runtime = await create_runtime(credentials)
    oauth = oauth_runtime.get_provider("openai-codex").auth.oauth
    assert oauth is not None
    refresh_calls = {"count": 0}
    original_refresh = oauth.refresh

    async def counting_refresh(credential, cancel):
        refresh_calls["count"] += 1
        return await original_refresh(credential, cancel)

    oauth.refresh = counting_refresh

    assert await get_provider_credential("openai-codex", oauth_runtime, credentials, refresh=False) == "old-token"
    assert refresh_calls["count"] == 0


@pytest.mark.tonio
async def test_refreshes_oauth_by_default():
    credentials = AuthStorage.in_memory(
        {"openai-codex": OAuthCredential(access="old-token", refresh="refresh-token", expires=0)}
    )
    runtime = await create_runtime(credentials)
    oauth = runtime.get_provider("openai-codex").auth.oauth
    assert oauth is not None
    refresh_calls = {"count": 0}

    async def fresh_refresh(_credential, _cancel):
        refresh_calls["count"] += 1
        return OAuthCredential(
            access="fresh-token", refresh="refresh-token", expires=int(time.time() * 1000) + 60 * 60 * 1000
        )

    oauth.refresh = fresh_refresh

    result = await check_provider_auth(parse_args(["--provider", "openai-codex"]), runtime, refresh=True)
    assert result.status == "ready"
    assert refresh_calls["count"] == 1


@pytest.mark.tonio
async def test_reports_an_unknown_provider_as_not_ready():
    runtime = await create_runtime(AuthStorage.in_memory())

    assert await check_provider_auth(parse_args(["--provider", "not-installed"]), runtime) == AuthCheckResult(
        status="not_ready", provider="not-installed", reason="provider_not_found"
    )


@pytest.mark.tonio
async def test_does_not_treat_an_unresolved_stored_environment_reference_as_configured(tmp_path):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps({"openai": {"type": "api_key", "key": "$MISSING_AUTH_CHECK_KEY"}}), encoding="utf-8"
    )
    runtime = await create_runtime(ReadOnlyAuthStorage(str(auth_path)))

    assert await check_provider_auth(parse_args(["--provider", "openai"]), runtime) == AuthCheckResult(
        status="not_ready", provider="openai", reason="credentials_not_configured"
    )


@pytest.mark.tonio
async def test_reports_malformed_auth_state_as_invalid(tmp_path):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text("{invalid-json", encoding="utf-8")
    runtime = await create_runtime(ReadOnlyAuthStorage(str(auth_path)))

    assert await check_provider_auth(parse_args(["--provider", "openai"]), runtime) == AuthCheckResult(
        status="invalid", provider="openai", reason="invalid_state"
    )


@pytest.mark.tonio
async def test_does_not_create_an_auth_file_or_its_parent_directory(tmp_path):
    auth_path = tmp_path / "agent" / "auth.json"
    runtime = await create_runtime(ReadOnlyAuthStorage(str(auth_path)))

    result = await check_provider_auth(parse_args(["--provider", "openai"]), runtime)
    assert result.status == "not_ready"
    assert result.reason == "credentials_not_configured"
    assert not auth_path.exists()
    assert not (tmp_path / "agent").exists()


def test_accepts_optional_json_output_credential_output_and_no_refresh():
    command = parse_auth_command(["auth", "check", "--provider", "openai"])
    assert (command.kind, command.args, command.json, command.credentials, command.no_refresh) == (
        "check",
        ["--provider", "openai"],
        False,
        False,
        False,
    )
    command = parse_auth_command(["auth", "check", "--json", "--credentials", "--no-refresh", "--provider", "openai"])
    assert (command.kind, command.args, command.json, command.credentials, command.no_refresh) == (
        "check",
        ["--provider", "openai"],
        True,
        True,
        True,
    )


@pytest.mark.tonio
async def test_creates_an_auth_check_runtime_without_catalog_storage():
    runtime = await create_auth_check_model_runtime(AuthStorage.in_memory())
    assert runtime.get_provider("openai") is not None
