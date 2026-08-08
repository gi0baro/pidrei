"""Experimental command modeling (port of pi `cli/experimental/command.ts`).

A tiny composable argv parser: registered `--option`s are consumed until the
first unregistered argument, everything from there on is handed to the
command's builder as `remaining_args` (the existing CLI parser consumes it).
pi's `{ ok: true, ... } | { ok: false, errors }` unions map to one result
dataclass with `ok` discriminating; TS generics are dropped (invocations are
plain dataclasses).

Per the async-only callback policy, command actions return awaitables (pi
types them `void | Promise<void>`).
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class CommandOptionParseResult:
    ok: bool
    value: Any = None
    error: str | None = None


@dataclass(slots=True, frozen=True)
class CommandOption:
    name: str
    parse: Callable[[str], CommandOptionParseResult]


def value_option(name: str, parse: Callable[[str], CommandOptionParseResult]) -> CommandOption:
    return CommandOption(name=name, parse=parse)


def string_option(name: str) -> CommandOption:
    return value_option(name, lambda value: CommandOptionParseResult(ok=True, value=value))


@dataclass(slots=True, frozen=True)
class CommandParseResult:
    """pi's CommandParseResult / CommandExecutionResult / CommandBuildResult."""

    ok: bool
    command: Any = None
    errors: tuple[str, ...] = ()


type CommandExecutionResult = CommandParseResult
type CommandBuildResult = CommandParseResult


@dataclass(slots=True, frozen=True)
class ParsedCommandInput:
    remaining_args: tuple[str, ...]
    parsed_values: dict[str, list[Any]] = field(default_factory=dict)

    def value(self, option: CommandOption) -> Any:
        values = self.parsed_values.get(option.name)
        return values[0] if values else None

    def values(self, option: CommandOption) -> tuple[Any, ...]:
        return tuple(self.parsed_values.get(option.name, ()))


type CommandBuilder = Callable[[ParsedCommandInput], CommandBuildResult]
type CommandAction = Callable[[Any, Any], Awaitable[None]]


class Command:
    def __init__(self, name: str) -> None:
        self.name = name
        self._options: dict[str, CommandOption] = {}
        self._subcommands: dict[str, Command] = {}
        self._builder: CommandBuilder | None = None
        self._command_action: CommandAction | None = None

    def option(self, option: CommandOption) -> Command:
        if option.name in self._options:
            raise Exception(f"Option {option.name} is already registered for {self.name}")
        self._options[option.name] = option
        return self

    def build(self, builder: CommandBuilder) -> Command:
        self._builder = builder
        return self

    def action(self, action: CommandAction) -> Command:
        self._command_action = action
        return self

    def command(self, command: Command) -> Command:
        if command.name in self._subcommands:
            raise Exception(f"Command {command.name} is already registered")
        self._subcommands[command.name] = command
        return self

    def parse(self, argv: list[str]) -> CommandParseResult:
        selected = self._select(argv)
        if selected is not None:
            command, rest = selected
            return command.parse(rest)
        return self._parse_own(argv)

    async def execute(self, argv: list[str], context: Any) -> CommandExecutionResult:
        selected = self._select(argv)
        if selected is not None:
            command, rest = selected
            return await command.execute(rest, context)

        parsed = self._parse_own(argv)
        if not parsed.ok:
            return parsed
        if self._command_action is None:
            raise Exception(f"Command {self.name} does not define an action")
        await self._command_action(parsed.command, context)
        return CommandParseResult(ok=True, command=parsed.command)

    def _select(self, argv: list[str]) -> tuple[Command, list[str]] | None:
        candidate = argv[0] if argv else None
        if candidate is None:
            return None
        command = self._subcommands.get(candidate)
        return (command, argv[1:]) if command is not None else None

    def _parse_own(self, argv: list[str]) -> CommandParseResult:
        if self._builder is None:
            raise Exception(f"Command {self.name} does not define a builder")
        parsed_input, parse_errors = self._parse_options(argv)
        built = self._builder(parsed_input)
        errors = list(parse_errors)
        if not built.ok:
            errors.extend(built.errors)
        if errors:
            return CommandParseResult(ok=False, errors=tuple(errors))
        if not built.ok:
            raise Exception(f"Command {self.name} failed without an error")
        return CommandParseResult(ok=True, command=built.command)

    def _parse_options(self, argv: list[str]) -> tuple[ParsedCommandInput, list[str]]:
        values: dict[str, list[Any]] = {}
        remaining_args: list[str] = []
        errors: list[str] = []
        index = 0
        while index < len(argv):
            argument = argv[index]
            if argument == "--":
                remaining_args.extend(argv[index:])
                break

            equals = argument.find("=")
            name = argument if equals == -1 else argument[:equals]
            option = self._options.get(name)
            if option is None:
                remaining_args.extend(argv[index:])
                break

            value = None if equals == -1 else argument[equals + 1 :]
            if value is None:
                nxt = argv[index + 1] if index + 1 < len(argv) else None
                if nxt is not None and not nxt.startswith("-"):
                    value = nxt
                    index += 1
            if value is None or value == "":
                errors.append(f"{name} requires a value")
                index += 1
                continue

            existing = values.get(name, [])
            if existing:
                errors.append(f"{name} may only be specified once")
                index += 1
                continue
            result = option.parse(value)
            if not result.ok:
                errors.append(result.error or "")
                index += 1
                continue
            existing.append(result.value)
            values[name] = existing
            index += 1
        return ParsedCommandInput(remaining_args=tuple(remaining_args), parsed_values=values), errors
