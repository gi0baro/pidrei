"""Experimental CLI auth inputs (port of pi `cli/experimental/auth.ts`)."""

from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True, frozen=True)
class TokenAuthInput:
    token: str
    type: Literal["token"] = "token"


@dataclass(slots=True, frozen=True)
class FileAuthInput:
    path: str
    type: Literal["file"] = "file"


type AuthInput = TokenAuthInput | FileAuthInput


@dataclass(slots=True, frozen=True)
class RawAuthOptions:
    auth_token: str | None = None
    auth_token_file: str | None = None


@dataclass(slots=True, frozen=True)
class ParsedAuthInput:
    auth: AuthInput | None
    errors: tuple[str, ...]


def parse_auth_input(options: RawAuthOptions) -> ParsedAuthInput:
    if options.auth_token is not None and options.auth_token_file is not None:
        return ParsedAuthInput(auth=None, errors=("--auth-token and --auth-token-file are mutually exclusive",))
    if options.auth_token is not None:
        return ParsedAuthInput(auth=TokenAuthInput(token=options.auth_token), errors=())
    if options.auth_token_file is not None:
        return ParsedAuthInput(auth=FileAuthInput(path=options.auth_token_file), errors=())
    return ParsedAuthInput(auth=None, errors=())
