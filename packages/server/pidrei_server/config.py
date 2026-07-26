"""Mirror of pi server src/config.ts.

Not ported: the Bun-compiled-binary detection (`isBunBinary`) and the
package.json walk for the version — pidrei is distributed as a Python
package, so the version comes from package metadata and the RPC child is
always spawned as `python -m pidrei` (see rpc_process.py).
"""

import importlib.metadata
import os


CONFIG_DIR_NAME = ".pidrei"
ENV_SERVER_DIR = "PIDREI_SERVER_DIR"

try:
    VERSION = importlib.metadata.version("pidrei-server")
except importlib.metadata.PackageNotFoundError:
    VERSION = "0.0.0"


def get_server_dir() -> str:
    env_dir = os.environ.get(ENV_SERVER_DIR)
    if env_dir:
        return env_dir

    pidrei_dir = os.environ.get("PIDREI_CONFIG_DIR") or os.path.join(os.path.expanduser("~"), CONFIG_DIR_NAME)
    return os.path.join(pidrei_dir, "server")


def get_auth_path() -> str:
    return os.path.join(get_server_dir(), "auth.json")


def get_machine_path() -> str:
    return os.path.join(get_server_dir(), "machine.json")


def get_instances_path() -> str:
    return os.path.join(get_server_dir(), "instances.json")


def get_socket_path() -> str:
    return os.path.join(get_server_dir(), "server.sock")
