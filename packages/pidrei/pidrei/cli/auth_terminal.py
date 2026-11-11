"""Port of pi's cli/auth-terminal.ts: terminal I/O for `pidrei auth login` and
`pidrei auth logout`.

pi's `createProcessLoginTerminal(onInterrupt)` is also wired to the process's
SIGINT. PiDrei handles no SIGINT, so Ctrl+C at a line question ends the
process through CPython's default handling, as for every other PiDrei shell
command. `on_interrupt` runs at end of input in a question, and on Ctrl+C in
a secret, which reads the terminal in raw mode where Ctrl+C is a key.
"""

import codecs
import sys
import termios
from collections.abc import Callable
from typing import Protocol

from pidrei_utils.cancel import CancelToken

from ..core.output_guard import drain_output, stdout_isatty, write_stderr, write_stdout
from ..utils.fd_io import FdReader
from ..utils.open_browser import open_browser
from ..utils.terminal_question import question_terminal, read_stdin


class AuthOutput(Protocol):
    """Output shared by the terminal authentication commands."""

    def write(self, line: str) -> None: ...

    def error(self, line: str) -> None: ...


class LoginTerminal(AuthOutput, Protocol):
    """Line-based terminal I/O for `pidrei auth login`. Prompts raise when `cancel` fires."""

    # Whether stdin and stdout are a terminal; login prompts need one.
    interactive: bool

    async def question(self, message: str, cancel: CancelToken | None) -> str: ...

    async def secret(self, message: str, cancel: CancelToken | None) -> str:
        """Read a line without echoing it."""
        ...

    def open_url(self, url: str) -> None: ...


class ProcessAuthOutput:
    """`AuthOutput` on stdout and stderr."""

    def write(self, line: str) -> None:
        write_stdout(f"{line}\n")

    def error(self, line: str) -> None:
        write_stderr(f"{line}\n")


class ProcessLoginTerminal(ProcessAuthOutput):
    """`LoginTerminal` on stdin and stdout."""

    def __init__(self, on_interrupt: Callable[[], None]) -> None:
        self.interactive = sys.stdin.isatty() and stdout_isatty()
        self._on_interrupt = on_interrupt

    def question(self, message: str, cancel: CancelToken | None):
        return question_terminal(message, output=write_stdout, cancel=cancel, on_interrupt=self._on_interrupt)

    def secret(self, message: str, cancel: CancelToken | None):
        return read_secret(message, cancel, self._on_interrupt)

    def open_url(self, url: str) -> None:
        open_browser(url)


def create_process_login_terminal(on_interrupt: Callable[[], None]) -> LoginTerminal:
    """Terminal I/O on stdin and stdout. `on_interrupt` runs at end of input in a question, or on Ctrl+C in a
    secret."""
    return ProcessLoginTerminal(on_interrupt)


async def read_secret(message: str, cancel: CancelToken | None, on_interrupt: Callable[[], None]) -> str:
    """Read a line from the terminal in raw mode without echoing it. Raises `Login cancelled` when `cancel`
    fires; the terminal mode is restored on every exit, synchronously."""
    if cancel is not None and cancel.cancelled:
        raise Exception("Login cancelled")
    fd = sys.stdin.fileno()
    write_stdout(message)
    await drain_output()
    saved = termios.tcgetattr(fd)
    termios.tcsetattr(fd, termios.TCSANOW, _raw_attributes(saved))
    try:
        return await read_stdin(lambda reader: _read_secret_keys(reader, on_interrupt), cancel)
    except Exception:
        if cancel is not None and cancel.cancelled:
            raise Exception("Login cancelled") from None
        raise
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, saved)
        write_stdout("\n")


def _raw_attributes(attributes: list) -> list:
    """Node's `setRawMode(true)` (libuv's raw mode): no echo, no line editing, no
    signals from keys, output processing kept."""
    iflag, oflag, cflag, lflag, ispeed, ospeed, cc = attributes
    iflag &= ~(termios.BRKINT | termios.ICRNL | termios.INPCK | termios.ISTRIP | termios.IXON)
    oflag |= termios.ONLCR
    cflag |= termios.CS8
    lflag &= ~(termios.ECHO | termios.ICANON | termios.IEXTEN | termios.ISIG)
    cc = list(cc)
    cc[termios.VMIN] = 1
    cc[termios.VTIME] = 0
    return [iflag, oflag, cflag, lflag, ispeed, ospeed, cc]


async def _read_secret_keys(reader: FdReader, on_interrupt: Callable[[], None]) -> str:
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    value = ""
    while True:
        data = await reader.read()
        if not data:
            # pi's raw read has no end-of-input case and would wait for an
            # abort; a closed terminal ends the login as Ctrl+C does.
            on_interrupt()
            raise Exception("Login cancelled")
        chunk = decoder.decode(data)
        # Ignore escape sequences such as arrow keys.
        if chunk.startswith("\x1b"):
            continue
        for char in chunk:
            if char in ("\r", "\n"):
                return value
            if char == "\x03":
                on_interrupt()
                break
            if char in ("\x7f", "\b"):
                value = value[:-1]
            elif char >= " ":
                value += char
