"""Port of pi's cli/auth-check.ts: the `auth check` preflight."""

from dataclasses import dataclass

from ..core.model_resolver import resolve_cli_model
from ..core.model_runtime import ModelRuntime
from ..core.models_store import InMemoryCodingAgentModelsStore
from .auth_command import AuthCommandError, get_auth_credential, validate_auth_command_args


type AuthCheckStatus = str  # "ready" | "not_ready" | "invalid"
type AuthCheckReason = (
    str  # "provider_not_found" | "credentials_not_configured" | "credential_not_available" | "invalid_state"
)


@dataclass(slots=True)
class AuthCheckResult:
    status: AuthCheckStatus
    provider: str
    reason: AuthCheckReason | None = None
    auth_type: str | None = None  # "api_key" | "oauth"


async def check_provider_auth(args, model_runtime: ModelRuntime, *, refresh: bool = False) -> AuthCheckResult:
    cli_provider, cli_model = validate_auth_command_args(args, "check")
    provider = cli_provider
    if cli_model:
        resolved = resolve_cli_model(cli_provider=cli_provider, cli_model=cli_model, model_runtime=model_runtime)
        if resolved.error or resolved.model is None:
            raise AuthCommandError(resolved.error or f'Unable to resolve model "{cli_model}"')
        provider = resolved.model.provider
    if not provider:
        raise AuthCommandError("Unable to resolve an auth provider")
    if model_runtime.get_error():
        return AuthCheckResult(status="invalid", provider=provider, reason="invalid_state")
    if model_runtime.get_provider(provider) is None:
        return AuthCheckResult(status="not_ready", provider=provider, reason="provider_not_found")
    try:
        auth = await model_runtime.check_auth(provider)
        if auth is None:
            return AuthCheckResult(status="not_ready", provider=provider, reason="credentials_not_configured")
        if refresh and await model_runtime.get_auth(provider) is None:
            return AuthCheckResult(status="not_ready", provider=provider, reason="credentials_not_configured")
        return AuthCheckResult(status="ready", provider=provider, auth_type=auth.type)
    except Exception:
        return AuthCheckResult(status="invalid", provider=provider, reason="invalid_state")


async def get_provider_credential(
    provider_id: str, model_runtime: ModelRuntime, credentials, *, refresh: bool
) -> str | None:
    credential = await credentials.read(provider_id)
    if not refresh and credential is not None and credential.type == "oauth":
        return credential.access
    return get_auth_credential(await model_runtime.get_auth(provider_id))


async def create_auth_check_model_runtime(credentials) -> ModelRuntime:
    return await ModelRuntime.create(
        credentials=credentials,
        models_store=InMemoryCodingAgentModelsStore(),
        allow_model_network=False,
        refresh_on_create=False,
    )
