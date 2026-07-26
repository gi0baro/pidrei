"""Port of pi's proxy-env resolution (packages/ai/src/utils/node-http-proxy.ts).

Named `http_proxy` rather than `node_http_proxy`: pi's name refers to the Node
`http` module it configures an agent for, and nothing in pidrei is Node.

punkreq reads `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` from `os.environ` itself
(`trust_env=True`), so this module exists for the part it cannot see —
provider-scoped `options.env` overrides, which take precedence over the process
environment — plus pi's explicit refusal of SOCKS/PAC proxy URLs.
"""

import re
from urllib.parse import urlsplit

from pidrei_ai.types import ProviderEnv
from pidrei_ai.utils.provider_env import get_provider_env_value


DEFAULT_PROXY_PORTS = {
    "ftp": 21,
    "gopher": 70,
    "http": 80,
    "https": 443,
    "ws": 80,
    "wss": 443,
}

UNSUPPORTED_PROXY_PROTOCOL_MESSAGE = (
    "Unsupported proxy protocol. SOCKS and PAC proxy URLs are not supported; use an HTTP or HTTPS proxy URL."
)

_HOST_PORT_RE = re.compile(r"^(.+):(\d+)$")


def _get_proxy_env(key: str, env: ProviderEnv | None = None) -> str:
    lowercase_key = key.lower()
    uppercase_key = key.upper()
    scoped = env or {}
    return (
        scoped.get(lowercase_key)
        or scoped.get(uppercase_key)
        # pi passes no `env` here: the scoped values were already checked above.
        or get_provider_env_value(lowercase_key)
        or get_provider_env_value(uppercase_key)
        or ""
    )


def _should_proxy_hostname(hostname: str, port: int, env: ProviderEnv | None = None) -> bool:
    no_proxy = _get_proxy_env("no_proxy", env).lower()
    if not no_proxy:
        return True
    if no_proxy == "*":
        return False

    def allows(entry: str) -> bool:
        if not entry:
            return True
        match = _HOST_PORT_RE.match(entry)
        proxy_hostname = match.group(1) if match else entry
        proxy_port = int(match.group(2)) if match else 0
        if proxy_port and proxy_port != port:
            return True
        if not proxy_hostname.startswith((".", "*")):
            return hostname != proxy_hostname
        return not hostname.endswith(proxy_hostname.removeprefix("*"))

    return all(allows(entry) for entry in re.split(r"[,\s]", no_proxy))


def _get_proxy_for_url(target_url: str, env: ProviderEnv | None = None) -> str:
    parsed = urlsplit(target_url)
    if not parsed.scheme or not parsed.netloc:
        return ""

    protocol = parsed.scheme
    # `netloc` carries userinfo too; pi reads `URL.host`, which does not.
    host = parsed.netloc.rpartition("@")[2]
    hostname = re.sub(r":\d*$", "", host)
    try:
        port = int(parsed.port or 0)
    except ValueError:
        port = 0
    port = port or DEFAULT_PROXY_PORTS.get(protocol, 0)
    if not _should_proxy_hostname(hostname, port, env):
        return ""

    proxy = _get_proxy_env(f"{protocol}_proxy", env) or _get_proxy_env("all_proxy", env)
    if proxy and "://" not in proxy:
        proxy = f"{protocol}://{proxy}"
    return proxy


def resolve_http_proxy_url_for_target(target_url: str, env: ProviderEnv | None = None) -> str | None:
    """The proxy URL to use for `target_url`, or None when it must not be proxied.

    pi returns a `URL`; the string form is what punkreq's `proxy=` takes. The
    trailing-slash normalization of `str(new URL(...))` is preserved so the
    values are comparable to pi's.
    """
    proxy = _get_proxy_for_url(target_url, env)
    if not proxy:
        return None

    parsed = urlsplit(proxy)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid proxy URL {proxy!r}: no host")

    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"{UNSUPPORTED_PROXY_PROTOCOL_MESSAGE} Got {parsed.scheme}:")

    return proxy if parsed.path else f"{proxy}/"
