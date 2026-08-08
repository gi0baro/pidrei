"""Experimental CLI composition (port of pi `cli/experimental/cli.ts`)."""

from typing import Protocol

from .commands.client import ClientCommandContext, client_command
from .commands.pi import PiCommandContext, pi_command
from .commands.server import ServerCommandContext, server_command


class ExperimentalCliContext(PiCommandContext, ServerCommandContext, ClientCommandContext, Protocol): ...


experimental_cli = pi_command.command(server_command).command(client_command)
