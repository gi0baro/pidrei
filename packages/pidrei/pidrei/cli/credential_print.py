"""Port of pi's cli/credential-print.ts: the `auth print-*` command surface."""

import re

from ..config import APP_NAME
from ..core.model_resolver import resolve_cli_model


DEFAULT_BEARER_TOKEN_MIN_EXPIRY_MS = 30 * 60_000

type CredentialPrintKind = str  # "api_key" | "bearer_token"


class CredentialPrintError(Exception):
    pass


class CredentialPrintCommand:
    __slots__ = ("args", "kind", "min_expiry_ms")

    def __init__(self, kind: CredentialPrintKind, args: list[str], min_expiry_ms: int | None = None):
        self.kind = kind
        self.args = args
        self.min_expiry_ms = min_expiry_ms


def is_credential_print_help(args: list[str]) -> bool:
    if not args or args[0] != "auth":
        return False
    return len(args) == 1 or args[1] in ("help", "--help", "-h")


def print_credential_print_help() -> None:
    print(f"""Usage:
  {APP_NAME} auth print-api-key --model <model> [--provider <provider>]
  {APP_NAME} auth print-bearer-token --model <model> [--provider <provider>] [--min-expiry <duration>]

Prints the configured credential alone on stdout. Provider inference uses configured credentials; specify --provider to select explicitly. Bearer tokens have a 30-minute minimum expiry by default. --min-expiry accepts ms, s, m, or h (for example, 30m).""")


_MIN_EXPIRY_RE = re.compile(r"^(\d+)(ms|s|m|h)$", re.IGNORECASE)
_UNIT_MS = {"ms": 1, "s": 1_000, "m": 60_000, "h": 3_600_000}


def parse_credential_print_command(args: list[str]) -> CredentialPrintCommand | None:
    """Parse the small, extensible `auth` command surface before normal startup."""
    if not args or args[0] != "auth":
        return None

    sub = args[1] if len(args) > 1 else None
    kind = "api_key" if sub == "print-api-key" else "bearer_token" if sub == "print-bearer-token" else None
    if kind is None:
        raise CredentialPrintError(
            f'Unknown auth command "{sub if sub is not None else ""}". '
            f'Use "{APP_NAME} auth print-api-key" or "{APP_NAME} auth print-bearer-token".'
        )

    command_args: list[str] = []
    min_expiry_ms: int | None = None
    index = 2
    while index < len(args):
        if args[index] != "--min-expiry":
            command_args.append(args[index])
            index += 1
            continue
        if kind != "bearer_token":
            raise CredentialPrintError("--min-expiry is only supported by print-bearer-token")
        index += 1
        value = args[index] if index < len(args) else None
        match = _MIN_EXPIRY_RE.match(value) if value else None
        if match is None:
            raise CredentialPrintError("--min-expiry must use a duration such as 30m or 1h")
        min_expiry_ms = int(match.group(1)) * _UNIT_MS[match.group(2).lower()]
        index += 1

    return CredentialPrintCommand(kind, command_args, min_expiry_ms)


def validate_credential_print_args(args) -> None:
    if not (args.model or "").strip():
        raise CredentialPrintError("Credential printing requires --model <model>")
    if args.api_key is not None:
        raise CredentialPrintError("Credential printing reads configured credentials; --api-key is not supported")
    if args.messages or args.file_args or args.unknown_flags:
        raise CredentialPrintError("Credential printing only accepts --provider and --model")


_BEARER_RE = re.compile(r"^Bearer\s+(.+)$", re.IGNORECASE)


async def resolve_credential_for_print(
    args,
    model_runtime,
    kind: CredentialPrintKind,
    min_expiry_ms: int | None = None,
) -> str:
    """Resolve one request credential for a specific provider/model pair.

    This intentionally calls ModelRuntime.get_auth(), which refreshes and
    persists OAuth credentials with less than five minutes remaining through
    the normal request-auth path.
    """
    from ..core.model_runtime import ModelRuntimeAuthOverrides

    validate_credential_print_args(args)

    credential_types = {
        credential.provider_id: credential.type for credential in await model_runtime.list_credentials()
    }
    models = []
    if args.provider:
        resolved = resolve_cli_model(cli_provider=args.provider, cli_model=args.model, model_runtime=model_runtime)
        if resolved.error or resolved.model is None:
            raise CredentialPrintError(resolved.error or "Unable to resolve the requested provider/model")
        models.append(resolved.model)
    else:
        for provider in model_runtime.get_providers():
            if provider.id not in credential_types:
                continue
            resolved = resolve_cli_model(cli_provider=provider.id, cli_model=args.model, model_runtime=model_runtime)
            if (
                resolved.model is not None
                and not resolved.error
                and not (resolved.warning and "Using custom model id" in resolved.warning)
            ):
                models.append(resolved.model)
        if not models:
            raise CredentialPrintError(f'Model "{args.model}" not found. Use --list-models to see available models.')

    credentials: list[tuple[str, str]] = []
    for model in models:
        credential_type = credential_types.get(model.provider)
        if kind == "api_key" and credential_type == "oauth":
            continue
        if kind == "bearer_token" and credential_type != "oauth":
            continue

        auth = await model_runtime.get_auth(
            model,
            ModelRuntimeAuthOverrides(
                min_oauth_validity_ms=min_expiry_ms if min_expiry_ms is not None else DEFAULT_BEARER_TOKEN_MIN_EXPIRY_MS
            )
            if kind == "bearer_token"
            else None,
        )
        authorization = None
        if auth is not None and auth.auth.headers:
            for name, header_value in auth.auth.headers.items():
                if name.lower() == "authorization":
                    authorization = header_value
                    break
        bearer_match = _BEARER_RE.match(authorization) if isinstance(authorization, str) else None
        bearer_token = bearer_match.group(1) if bearer_match else None
        api_key = auth.auth.api_key if auth is not None else None
        value = (api_key or bearer_token) if kind == "bearer_token" else api_key
        if value:
            credentials.append((model.provider, value))

    if len(credentials) == 1:
        return credentials[0][1]
    if not credentials:
        provider_id = models[0].provider if models else None
        credential_type = credential_types.get(provider_id) if provider_id else None
        if args.provider and kind == "api_key" and credential_type == "oauth":
            raise CredentialPrintError(f'Provider "{provider_id}" is configured with OAuth, not an API key')
        if args.provider and kind == "bearer_token" and credential_type != "oauth":
            raise CredentialPrintError(f'Provider "{provider_id}" is not configured with an OAuth bearer token')
        raise CredentialPrintError(
            f"No usable {'API key' if kind == 'api_key' else 'OAuth bearer token'} is configured"
        )
    provider_list = ", ".join(provider_id for provider_id, _value in credentials)
    raise CredentialPrintError(
        f'Model "{args.model}" has multiple configured providers ({provider_list}). Specify --provider.'
    )
