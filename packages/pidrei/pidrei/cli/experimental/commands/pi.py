"""Experimental `pi` command (port of pi `cli/experimental/commands/pi.ts`)."""

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Literal, Protocol

from ...args import Args
from ..auth import AuthInput
from ..command import Command, CommandBuildResult, CommandParseResult, ParsedCommandInput
from ..command_options import (
    auth_token_file_option,
    auth_token_option,
    parse_auth,
    parse_legacy_options,
    transport_option,
)
from ..transport_address import TransportAddress


@dataclass(slots=True, frozen=True)
class PiCommand:
    options: Args
    auth: AuthInput | None = None
    listen: tuple[TransportAddress, ...] | None = None
    command: Literal["pi"] = "pi"


class PiCommandContext(Protocol):
    def run_pi(self, command: PiCommand) -> Awaitable[None]: ...


_listen_option = transport_option("--listen")


def _build(input: ParsedCommandInput) -> CommandBuildResult:
    parsed_auth = parse_auth(input)
    listen = input.values(_listen_option)
    legacy = parse_legacy_options(input)
    errors = [*parsed_auth.errors, *legacy.errors]
    if "connect" in legacy.options.unknown_flags:
        errors.append("--connect is only valid for client mode")
    if errors:
        return CommandParseResult(ok=False, errors=tuple(errors))
    return CommandParseResult(
        ok=True,
        command=PiCommand(
            options=legacy.options,
            auth=parsed_auth.auth,
            listen=listen if listen else None,
        ),
    )


pi_command = (
    Command("pi")
    .option(_listen_option)
    .option(auth_token_option)
    .option(auth_token_file_option)
    .build(_build)
    .action(lambda command, context: context.run_pi(command))
)
