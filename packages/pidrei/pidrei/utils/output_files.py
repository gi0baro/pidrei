"""Output files: files pidrei writes so the model can use output it was not
shown in full, such as the full text of truncated tool output, binary MCP
resources, and images shown by codemode scripts. Every output file is created
here, so where they are stored can change in one place. Today they go to the OS
temp directory.

Each file is created exclusively and readable only by the user
(`create_private_file_blocking`).
"""

import secrets

import tonio.colored as tonio

from ..config import TEMP_DIR
from .temp_file_writer import TempFileWriter, create_private_file_blocking


def _create_output_file_path(prefix: str, extension: str) -> str:
    """A new, unused path: `<dir>/<prefix>-<random hex><extension>`. `extension` includes the dot."""
    return str(TEMP_DIR / f"{prefix}-{secrets.token_hex(8)}{extension}")


def _write_output_file_blocking(path: str, data: str | bytes) -> None:
    with create_private_file_blocking(path) as file:
        file.write(data.encode("utf-8") if isinstance(data, str) else data)


async def write_output_file(prefix: str, extension: str, data: str | bytes) -> str:
    """Write `data` to a new output file and return its path."""
    path = _create_output_file_path(prefix, extension)
    await tonio.spawn_blocking(_write_output_file_blocking, path, data)
    return path


def create_output_file_stream(prefix: str, extension: str) -> tuple[str, TempFileWriter]:
    """Open a new output file for streamed output."""
    path = _create_output_file_path(prefix, extension)
    return path, TempFileWriter(path)
