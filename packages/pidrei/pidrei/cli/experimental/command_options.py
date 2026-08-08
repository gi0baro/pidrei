"""Shared experimental command options (port of pi `cli/experimental/command-options.ts`)."""

from dataclasses import dataclass
from typing import Literal

from ..args import Args, parse_args
from .auth import ParsedAuthInput, RawAuthOptions, parse_auth_input
from .command import CommandOption, CommandOptionParseResult, ParsedCommandInput, string_option, value_option
from .transport_address import parse_transport_address


auth_token_option = string_option("--auth-token")
auth_token_file_option = string_option("--auth-token-file")


def transport_option(name: Literal["--listen", "--connect"]) -> CommandOption:
    def parse(value: str) -> CommandOptionParseResult:
        result = parse_transport_address(value, name)
        if result.address is not None:
            return CommandOptionParseResult(ok=True, value=result.address)
        error = result.error if result.error is not None else f'Invalid {name} address "{value}"'
        return CommandOptionParseResult(ok=False, error=error)

    return value_option(name, parse)


def parse_auth(input: ParsedCommandInput) -> ParsedAuthInput:
    return parse_auth_input(
        RawAuthOptions(
            auth_token=input.value(auth_token_option),
            auth_token_file=input.value(auth_token_file_option),
        )
    )


@dataclass(slots=True, frozen=True)
class ParsedLegacyOptions:
    options: Args
    errors: tuple[str, ...]


def parse_legacy_options(input: ParsedCommandInput) -> ParsedLegacyOptions:
    options = parse_args(list(input.remaining_args))
    return ParsedLegacyOptions(
        options=options,
        errors=tuple(diagnostic["message"] for diagnostic in options.diagnostics if diagnostic["type"] == "error"),
    )


def unsupported_legacy_options(command: str, input: ParsedCommandInput) -> list[str]:
    if len(input.remaining_args) == 0:
        return []
    return [f"The experimental {command} command does not support existing CLI options yet"]
