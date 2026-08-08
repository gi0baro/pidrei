"""Mirror of pi coding-agent test/experimental-cli-resolution.test.ts.

Upstream's two `--ui-mode` cases exercise a flag pidrei has not ported yet
(it lands with U10's ui-mode commits); until then the same behavior — legacy
parser output and diagnostics composing with the capability error — is
exercised through `--mode` and an unknown short option. Align these cases
with upstream when `--tui-mode` lands.
"""

import pytest

from pidrei.cli.experimental.auth import TokenAuthInput
from pidrei.cli.experimental.cli import experimental_cli
from pidrei.cli.experimental.transport_address import UnixTransportAddress


UNSUPPORTED_SERVER_OPTIONS = "The experimental server command does not support existing CLI options yet"
UNSUPPORTED_CLIENT_OPTIONS = "The experimental client command does not support existing CLI options yet"


def test_composes_pi_command_options_with_the_existing_parser():
    result = experimental_cli.parse(
        [
            "--listen",
            "unix:///tmp/pi.sock",
            "--auth-token",
            "secret",
            "--provider",
            "anthropic",
            "--model",
            "claude-sonnet",
            "--thinking",
            "high",
            "inspect",
        ]
    )

    assert result.ok is True
    assert result.command.command == "pi"
    assert result.command.listen == (UnixTransportAddress(path="/tmp/pi.sock"),)
    assert result.command.auth == TokenAuthInput(token="secret")
    assert result.command.options.provider == "anthropic"
    assert result.command.options.model == "claude-sonnet"
    assert result.command.options.thinking == "high"
    assert result.command.options.messages == ["inspect"]


@pytest.mark.parametrize("option", ["--help", "--version"])
def test_keeps_pi_help_and_version_handling_in_existing_cli_options(option):
    result = experimental_cli.parse([option])

    assert result.ok is True
    assert result.command.command == "pi"
    if option == "--help":
        assert result.command.options.help is True
    else:
        assert result.command.options.version is True


@pytest.mark.parametrize(
    ("command", "option", "error"),
    [
        ("server", "--help", UNSUPPORTED_SERVER_OPTIONS),
        ("server", "--version", UNSUPPORTED_SERVER_OPTIONS),
        ("client", "--help", UNSUPPORTED_CLIENT_OPTIONS),
        ("client", "--version", UNSUPPORTED_CLIENT_OPTIONS),
    ],
)
def test_rejects_deferred_help_and_version_handling(command, option, error):
    result = experimental_cli.parse([command, option])

    assert result.ok is False
    assert result.errors == (error,)


def test_rejects_existing_options_that_the_server_command_does_not_support_yet():
    result = experimental_cli.parse(["server", "--model", "claude-sonnet", "prompt"])

    assert result.ok is False
    assert result.errors == (UNSUPPORTED_SERVER_OPTIONS,)


def test_rejects_existing_options_that_the_client_command_does_not_support_yet():
    # Upstream: ["client", "--ui-mode", "fullscreen", "@prompt.md"].
    result = experimental_cli.parse(["client", "--mode", "json", "@prompt.md"])

    assert result.ok is False
    assert result.errors == (UNSUPPORTED_CLIENT_OPTIONS,)


def test_reports_existing_parser_errors_before_capability_errors():
    # Upstream: ["client", "--ui-mode", "wrong", "--model", "claude-sonnet"]
    # with the invalid-UI-mode diagnostic; pidrei has no --ui-mode yet, so an
    # unknown short option produces the parser-error diagnostic instead.
    result = experimental_cli.parse(["client", "-x", "--model", "claude-sonnet"])

    assert result.ok is False
    assert result.errors == ("Unknown option: -x", UNSUPPORTED_CLIENT_OPTIONS)


def test_parses_an_empty_server_command():
    result = experimental_cli.parse(["server"])

    assert result.ok is True
    assert result.command.command == "server"
    assert result.command.auth is None
    assert result.command.listen is None


class _RecordingContext:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run_pi(self, command) -> None:
        self.calls.append("pi")

    async def run_server(self, command) -> None:
        self.calls.append("server")

    async def run_client(self, command) -> None:
        self.calls.append("client")


@pytest.mark.tonio
@pytest.mark.parametrize("name", ["pi", "server", "client"])
async def test_executes_the_parsed_command(name):
    context = _RecordingContext()
    result = await experimental_cli.execute([] if name == "pi" else [name], context)

    assert result.ok is True
    assert result.command.command == name
    assert context.calls == [name]
