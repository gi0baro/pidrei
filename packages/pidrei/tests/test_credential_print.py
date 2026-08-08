"""Mirror of pi's credential-print.test.ts."""

import time

import pytest

from pidrei.cli.args import parse_args
from pidrei.cli.auth_command import AuthCommandError, is_auth_command_help, parse_auth_command
from pidrei.cli.credential_print import resolve_credential_for_print
from pidrei.core.auth_storage import AuthStorage
from pidrei.core.model_runtime import ModelRuntime
from pidrei_ai.auth.types import ApiKeyCredential, OAuthCredential
from pidrei_ai.models_store import InMemoryModelsStore


def now_ms() -> int:
    return int(time.time() * 1000)


async def create_runtime(credentials: AuthStorage) -> ModelRuntime:
    return await ModelRuntime.create(
        credentials=credentials,
        models_path=None,
        models_store=InMemoryModelsStore(),
        allow_model_network=False,
    )


@pytest.mark.tonio
async def test_prints_a_resolved_api_key():
    runtime = await create_runtime(AuthStorage.in_memory({"openai": ApiKeyCredential(key="test-api-key")}))
    args = parse_args(["--provider", "openai"])

    assert await resolve_credential_for_print(args, runtime, "api_key") == "test-api-key"


@pytest.mark.tonio
async def test_prints_bearer_tokens_resolved_from_an_authorization_header():
    runtime = await create_runtime(
        AuthStorage.in_memory(
            {
                "kimi-coding": OAuthCredential(
                    access="header-test-token",
                    refresh="test-refresh-token",
                    expires=now_ms() + 60 * 60 * 1000,
                )
            }
        )
    )
    args = parse_args(["--provider", "kimi-coding"])

    assert await resolve_credential_for_print(args, runtime, "bearer_token") == "header-test-token"


@pytest.mark.tonio
async def test_refreshes_an_expired_oauth_token_before_printing_it():
    storage = AuthStorage.in_memory(
        {
            "openai-codex": OAuthCredential(
                access="old-test-token",
                refresh="test-refresh-token",
                expires=0,
            )
        }
    )
    runtime = await create_runtime(storage)
    refresh_calls: list[str] = []

    async def refresh(credential, _cancel):
        refresh_calls.append(credential.access)
        return OAuthCredential(
            access="fresh-test-token",
            refresh="test-refresh-token",
            expires=now_ms() + 60 * 60 * 1000,
        )

    provider = runtime.get_provider("openai-codex")
    assert provider is not None and provider.auth.oauth is not None
    oauth = provider.auth.oauth
    original_refresh = oauth.refresh
    # The provider object is shared across the suite's single runtime; swap
    # manually and restore (monkeypatch cannot run under the tonio mark).
    oauth.refresh = refresh
    try:
        args = parse_args(["--provider", "openai-codex"])

        assert await resolve_credential_for_print(args, runtime, "bearer_token") == "fresh-test-token"
        assert len(refresh_calls) == 1
        stored = await storage.read("openai-codex")
        assert stored is not None and stored.access == "fresh-test-token"
    finally:
        oauth.refresh = original_refresh


@pytest.mark.tonio
async def test_reports_unknown_auth_options_like_package_commands():
    """pi drives `main()`; pidrei calls the extracted `_run_auth_command`."""
    import contextlib
    import io

    from pidrei.main import _run_auth_command
    from pidrei.utils.ansi import strip_ansi

    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        exit_code = await _run_auth_command(["auth", "check", "--provider", "openai-codex", "--credentails"])

    stderr = strip_ansi(buffer.getvalue())
    assert 'Unknown option --credentails for "auth check".' in stderr
    assert (
        'Use "pidrei --help" or "pidrei auth check --provider <provider> [--json] [--credentials] [--no-refresh]".'
    ) in stderr
    assert exit_code == 1


@pytest.mark.tonio
async def test_parses_credential_commands_and_rejects_invalid_arguments_or_credential_types():
    runtime = await create_runtime(
        AuthStorage.in_memory(
            {
                "openai-codex": OAuthCredential(
                    access="test-token-not-to-be-printed",
                    refresh="test-refresh-token",
                    expires=now_ms() + 60 * 60 * 1000,
                )
            }
        )
    )

    command = parse_auth_command(["auth", "print-api-key", "--provider", "openai"])
    assert command is not None
    assert command.kind == "api_key"
    assert command.args == ["--provider", "openai"]
    assert command.min_expiry_ms is None

    command = parse_auth_command(["auth", "print-bearer-token"])
    assert command is not None and command.kind == "bearer_token"

    command = parse_auth_command(["auth", "print-bearer-token", "--min-expiry", "30m"])
    assert command is not None
    assert command.kind == "bearer_token"
    assert command.args == []
    assert command.min_expiry_ms == 30 * 60_000

    with pytest.raises(AuthCommandError, match="only supported by print-bearer-token"):
        parse_auth_command(["auth", "print-api-key", "--min-expiry", "30m"])

    assert is_auth_command_help(["auth", "--help"]) is True

    with pytest.raises(AuthCommandError):
        parse_auth_command(["auth", "unknown"])

    with pytest.raises(AuthCommandError, match=r"requires --provider <provider> or --model <model>"):
        await resolve_credential_for_print(parse_args([]), runtime, "api_key")

    with pytest.raises(AuthCommandError, match="configured with OAuth"):
        await resolve_credential_for_print(parse_args(["--provider", "openai-codex"]), runtime, "api_key")
