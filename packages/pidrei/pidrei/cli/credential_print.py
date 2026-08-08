"""Port of pi's cli/credential-print.ts: the `auth print-*` command surface."""

from ..core.model_resolver import resolve_cli_model
from .auth_command import AuthCommandError, get_auth_credential, validate_auth_command_args


DEFAULT_BEARER_TOKEN_MIN_EXPIRY_MS = 30 * 60_000

type CredentialPrintKind = str  # "api_key" | "bearer_token"


async def resolve_credential_for_print(
    args,
    model_runtime,
    kind: CredentialPrintKind,
    min_expiry_ms: int | None = None,
    cancel=None,
) -> str:
    """Resolve one configured provider credential.

    This intentionally calls ModelRuntime.get_auth(), which refreshes and
    persists OAuth credentials with less than five minutes remaining through
    the normal request-auth path.
    """
    from pidrei_ai.auth.types import AuthOperationOptions

    from ..core.model_runtime import ModelRuntimeAuthOverrides

    cli_provider, cli_model = validate_auth_command_args(args, kind)
    credential_types = {
        credential.provider_id: credential.type
        for credential in await model_runtime.list_credentials(AuthOperationOptions(cancel=cancel))
    }
    providers: list[tuple[str, object | None]] = []
    if cli_provider:
        provider = model_runtime.get_provider(cli_provider)
        if provider is None:
            raise AuthCommandError(f'Unknown provider "{cli_provider}". Use --list-models to see available providers.')
        if cli_model:
            resolved = resolve_cli_model(cli_provider=provider.id, cli_model=cli_model, model_runtime=model_runtime)
            if resolved.error or resolved.model is None:
                raise AuthCommandError(resolved.error or "Unable to resolve the requested provider/model")
            providers.append((provider.id, resolved.model))
        else:
            providers.append((provider.id, None))
    else:
        for provider in model_runtime.get_providers():
            if provider.id not in credential_types:
                continue
            resolved = resolve_cli_model(cli_provider=provider.id, cli_model=cli_model, model_runtime=model_runtime)
            if (
                resolved.model is not None
                and not resolved.error
                and not (resolved.warning and "Using custom model id" in resolved.warning)
            ):
                providers.append((provider.id, resolved.model))
        if not providers:
            raise AuthCommandError(f'Model "{cli_model}" not found. Use --list-models to see available models.')

    credentials: list[tuple[str, str]] = []
    for provider_id, model in providers:
        credential_type = credential_types.get(provider_id)
        if kind == "api_key" and credential_type == "oauth":
            continue
        if kind == "bearer_token" and credential_type != "oauth":
            continue
        auth_options = ModelRuntimeAuthOverrides(
            min_oauth_validity_ms=(min_expiry_ms if min_expiry_ms is not None else DEFAULT_BEARER_TOKEN_MIN_EXPIRY_MS)
            if kind == "bearer_token"
            else None,
            cancel=cancel,
        )
        auth = await model_runtime.get_auth(model if model is not None else provider_id, auth_options)
        value = get_auth_credential(auth)
        if value:
            credentials.append((provider_id, value))

    if len(credentials) == 1:
        return credentials[0][1]
    if not credentials:
        provider_id = providers[0][0] if providers else None
        credential_type = credential_types.get(provider_id) if provider_id else None
        if cli_provider and kind == "api_key" and credential_type == "oauth":
            raise AuthCommandError(f'Provider "{provider_id}" is configured with OAuth, not an API key')
        if cli_provider and kind == "bearer_token" and credential_type != "oauth":
            raise AuthCommandError(f'Provider "{provider_id}" is not configured with an OAuth bearer token')
        raise AuthCommandError(f"No usable {'API key' if kind == 'api_key' else 'OAuth bearer token'} is configured")
    provider_list = ", ".join(provider_id for provider_id, _value in credentials)
    raise AuthCommandError(f"Multiple configured providers matched ({provider_list}). Specify --provider.")
