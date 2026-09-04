"""Mirror of pi server `transports/unix/address.ts`."""

import re
from posixpath import join


_SERVER_ID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")


def get_unix_socket_path(server_id: str, server_directory: str) -> str:
    """Derive the local Unix socket path for one logical server identity."""
    if not _SERVER_ID_RE.fullmatch(server_id):
        raise TypeError("Unix serverId must be a canonical lowercase UUIDv4")
    return join(server_directory, f"{server_id}.sock")
