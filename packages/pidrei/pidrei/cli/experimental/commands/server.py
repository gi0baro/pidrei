"""Experimental `server` command (port of pi `cli/experimental/commands/server.ts`)."""

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
class ServerCommand:
    auth: AuthInput | None = None
    listen: tuple[TransportAddress, ...] | None = None
    command: Literal["server"] = "server"


class ServerCommandContext(Protocol):
    def run_server(self, command: ServerCommand) -> Awaitable[None]: ...


_listen_option = transport_option("--listen")


def _build(input: ParsedCommandInput) -> CommandBuildResult:
    parsed_auth = parse_auth(input)
    listen = input.values(_listen_option)
    legacy = parse_legacy_options(input)
    errors = [*parsed_auth.errors, *legacy.errors, *unsupported_legacy_options("server", input)]
    if errors:
        return CommandParseResult(ok=False, errors=tuple(errors))
    return CommandParseResult(
        ok=True,
        command=ServerCommand(auth=parsed_auth.auth, listen=listen if listen else None),
    )


server_command = (
    Command("server")
    .option(_listen_option)
    .option(auth_token_option)
    .option(auth_token_file_option)
    .build(_build)
    .action(lambda command, context: context.run_server(command))
)
