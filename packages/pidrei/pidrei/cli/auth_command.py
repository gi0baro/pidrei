"""Port of pi's cli/auth-command.ts: the shared `auth` command surface."""

import re

from ..config import APP_NAME


type AuthCommandKind = str  # "check" | "api_key" | "bearer_token"


class AuthCommand:
    __slots__ = ("args", "credentials", "json", "kind", "min_expiry_ms", "no_refresh")

    def __init__(
        self,
        kind: AuthCommandKind,
        args: list[str],
        *,
        json: bool = False,
        credentials: bool = False,
        no_refresh: bool = False,
        min_expiry_ms: int | None = None,
    ):
        self.kind = kind
        self.args = args
        self.json = json
        self.credentials = credentials
        self.no_refresh = no_refresh
        self.min_expiry_ms = min_expiry_ms


class AuthCommandError(Exception):
    pass


_AUTH_COMMAND_USAGE = {
    "check": f"{APP_NAME} auth check --provider <provider> [--json] [--credentials] [--no-refresh]",
    "api_key": f"{APP_NAME} auth print-api-key --provider <provider> [--model <model>]",
    "bearer_token": f"{APP_NAME} auth print-bearer-token --provider <provider> [--model <model>] [--min-expiry <duration>]",
}


def get_auth_command_name(kind: AuthCommandKind) -> str:
    if kind == "check":
        return "auth check"
    return "auth print-api-key" if kind == "api_key" else "auth print-bearer-token"


def get_auth_command_usage(kind: AuthCommandKind) -> str:
    return _AUTH_COMMAND_USAGE[kind]


def is_auth_command_help(args: list[str]) -> bool:
    if not args or args[0] != "auth":
        return False
    return len(args) == 1 or args[1] == "help" or "--help" in args or "-h" in args


def print_auth_command_help() -> None:
    print(f"""Usage:
  {APP_NAME} auth print-api-key [--provider <provider>] [--model <model>]
  {APP_NAME} auth print-bearer-token [--provider <provider>] [--model <model>] [--min-expiry <duration>]
  {APP_NAME} auth check [--provider <provider>] [--model <model>] [--json] [--credentials] [--no-refresh]

Auth commands require at least one of --provider or --model. Checks refresh expired OAuth credentials by default; --no-refresh prevents this. --credentials emits the credential, or includes it in JSON output.""")


_MIN_EXPIRY_RE = re.compile(r"^(\d+)(ms|s|m|h)$", re.IGNORECASE)
_UNIT_MS = {"ms": 1, "s": 1_000, "m": 60_000, "h": 3_600_000}


def parse_auth_command(args: list[str]) -> AuthCommand | None:
    if not args or args[0] != "auth":
        return None

    sub = args[1] if len(args) > 1 else None
    kind = (
        "check"
        if sub == "check"
        else "api_key"
        if sub == "print-api-key"
        else "bearer_token"
        if sub == "print-bearer-token"
        else None
    )
    if kind is None:
        raise AuthCommandError(
            f'Unknown auth command "{sub if sub is not None else ""}". '
            f'Use "{APP_NAME} auth print-api-key", "{APP_NAME} auth print-bearer-token", or "{APP_NAME} auth check".'
        )

    command_args: list[str] = []
    json_output = False
    credentials = False
    no_refresh = False
    min_expiry_ms: int | None = None
    index = 2
    while index < len(args):
        arg = args[index]
        if arg == "--min-expiry":
            if kind != "bearer_token":
                raise AuthCommandError("--min-expiry is only supported by print-bearer-token")
            index += 1
            value = args[index] if index < len(args) else None
            match = _MIN_EXPIRY_RE.match(value) if value else None
            if match is None:
                raise AuthCommandError("--min-expiry must use a duration such as 30m or 1h")
            min_expiry_ms = int(match.group(1)) * _UNIT_MS[match.group(2).lower()]
            index += 1
            continue
        if arg in ("--json", "--credentials", "--no-refresh"):
            if kind != "check":
                raise AuthCommandError(f"{arg} is only supported by auth check")
            if arg == "--json":
                json_output = True
            elif arg == "--credentials":
                credentials = True
            else:
                no_refresh = True
            index += 1
            continue
        command_args.append(arg)
        index += 1

    return AuthCommand(
        kind,
        command_args,
        json=json_output,
        credentials=credentials,
        no_refresh=no_refresh,
        min_expiry_ms=min_expiry_ms,
    )


def validate_auth_command_args(args, kind: AuthCommandKind) -> tuple[str | None, str | None]:
    """Returns `(provider, model)`."""
    provider = (args.provider or "").strip() or None
    model = (args.model or "").strip() or None
    if args.unknown_flags:
        option = next(iter(args.unknown_flags))
        raise AuthCommandError(f'Unknown option --{option} for "{get_auth_command_name(kind)}".')
    if args.api_key is not None or args.messages or args.file_args:
        raise AuthCommandError("Auth commands only accept --provider and --model")
    if kind == "check":
        if not provider and not model:
            raise AuthCommandError("Auth checks require --provider <provider> or --model <model>")
        return provider, model
    if not provider and not model:
        raise AuthCommandError("Credential printing requires --provider <provider> or --model <model>")
    return provider, model


_BEARER_RE = re.compile(r"^Bearer\s+(.+)$", re.IGNORECASE)


def get_auth_credential(auth) -> str | None:
    if auth is not None and auth.auth.api_key:
        return auth.auth.api_key
    authorization = None
    if auth is not None and auth.auth.headers:
        for name, header_value in auth.auth.headers.items():
            if name.lower() == "authorization":
                authorization = header_value
                break
    match = _BEARER_RE.match(authorization) if isinstance(authorization, str) else None
    return match.group(1) if match else None
