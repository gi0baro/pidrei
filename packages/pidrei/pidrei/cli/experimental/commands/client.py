"""Experimental `client` command (port of pi `cli/experimental/commands/client.ts`)."""

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Literal, Protocol

from ..auth import AuthInput
from ..command import Command, CommandBuildResult, CommandParseResult, ParsedCommandInput
from ..command_options import (
    auth_token_file_option,
    auth_token_option,
    parse_auth,
    parse_legacy_options,
    transport_option,
    unsupported_legacy_options,
)
from ..transport_address import TransportAddress


@dataclass(slots=True, frozen=True)
class ClientCommand:
    auth: AuthInput | None = None
    connect: TransportAddress | None = None
    command: Literal["client"] = "client"


class ClientCommandContext(Protocol):
    def run_client(self, command: ClientCommand) -> Awaitable[None]: ...


_connect_option = transport_option("--connect")


def _build(input: ParsedCommandInput) -> CommandBuildResult:
    parsed_auth = parse_auth(input)
    connect = input.value(_connect_option)
    legacy = parse_legacy_options(input)
    errors = [*parsed_auth.errors, *legacy.errors, *unsupported_legacy_options("client", input)]
    if errors:
        return CommandParseResult(ok=False, errors=tuple(errors))
    return CommandParseResult(
        ok=True,
        command=ClientCommand(auth=parsed_auth.auth, connect=connect),
    )


client_command = (
    Command("client")
    .option(_connect_option)
    .option(auth_token_option)
    .option(auth_token_file_option)
    .build(_build)
    .action(lambda command, context: context.run_client(command))
)
