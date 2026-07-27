"""Mirror of pi coding-agent src/core/resolve-config-value.ts (POSIX-only).

Resolve configuration values that may be shell commands, environment
variables, or literals. Used by auth storage and the model registry.

pi's win32 configured-shell path (getShellConfig + stdin transport) is not
ported; POSIX always executes commands through the default shell, exactly
like pi's `executeWithDefaultShell`.
"""

import os
import re
import subprocess
import threading
from dataclasses import dataclass


# Cache for shell command results (persists for process lifetime)
_command_result_cache: dict[str, str | None] = {}
_command_result_cache_lock = threading.Lock()

_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_VAR_NAME_PREFIX_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*")


@dataclass(slots=True)
class _LiteralPart:
    value: str


@dataclass(slots=True)
class _EnvPart:
    name: str


type _TemplatePart = _LiteralPart | _EnvPart


def _append_literal(parts: list[_TemplatePart], value: str) -> None:
    if not value:
        return
    if parts and isinstance(parts[-1], _LiteralPart):
        parts[-1].value += value
        return
    parts.append(_LiteralPart(value))


def _parse_config_value_template(config: str) -> list[_TemplatePart]:
    parts: list[_TemplatePart] = []
    index = 0

    while index < len(config):
        dollar_index = config.find("$", index)
        if dollar_index < 0:
            _append_literal(parts, config[index:])
            break

        _append_literal(parts, config[index:dollar_index])
        next_char = config[dollar_index + 1] if dollar_index + 1 < len(config) else None

        if next_char in ("$", "!"):
            _append_literal(parts, next_char)
            index = dollar_index + 2
            continue

        if next_char == "{":
            end_index = config.find("}", dollar_index + 2)
            if end_index < 0:
                _append_literal(parts, "$")
                index = dollar_index + 1
                continue

            name = config[dollar_index + 2 : end_index]
            if _ENV_VAR_NAME_RE.match(name):
                parts.append(_EnvPart(name))
            else:
                _append_literal(parts, config[dollar_index : end_index + 1])
            index = end_index + 1
            continue

        match = _ENV_VAR_NAME_PREFIX_RE.match(config[dollar_index + 1 :])
        if match:
            parts.append(_EnvPart(match.group(0)))
            index = dollar_index + 1 + len(match.group(0))
            continue

        _append_literal(parts, "$")
        index = dollar_index + 1

    return parts


def _is_command_reference(config: str) -> bool:
    return config.startswith("!")


def _resolve_env_config_value(name: str, env: dict[str, str] | None) -> str | None:
    return (env or {}).get(name) or os.environ.get(name) or None


def _get_template_env_var_names(parts: list[_TemplatePart]) -> list[str]:
    names: list[str] = []
    for part in parts:
        if not isinstance(part, _EnvPart) or part.name in names:
            continue
        names.append(part.name)
    return names


def _resolve_template(parts: list[_TemplatePart], env: dict[str, str] | None) -> str | None:
    resolved = ""
    for part in parts:
        if isinstance(part, _LiteralPart):
            resolved += part.value
            continue
        env_value = _resolve_env_config_value(part.name, env)
        if env_value is None:
            return None
        resolved += env_value
    return resolved


def get_config_value_env_var_name(config: str) -> str | None:
    if _is_command_reference(config):
        return None
    parts = _parse_config_value_template(config)
    if len(parts) == 1 and isinstance(parts[0], _EnvPart):
        return parts[0].name
    return None


def get_config_value_env_var_names(config: str) -> list[str]:
    if _is_command_reference(config):
        return []
    return _get_template_env_var_names(_parse_config_value_template(config))


def get_missing_config_value_env_var_names(config: str, env: dict[str, str] | None = None) -> list[str]:
    return [name for name in get_config_value_env_var_names(config) if _resolve_env_config_value(name, env) is None]


def is_command_config_value(config: str) -> bool:
    return _is_command_reference(config)


def is_config_value_configured(config: str, env: dict[str, str] | None = None) -> bool:
    return len(get_missing_config_value_env_var_names(config, env)) == 0


def resolve_config_value(config: str, env: dict[str, str] | None = None) -> str | None:
    """Resolve a config value (API key, header value, etc.) to an actual value.

    - If starts with "!", executes the rest as a shell command and uses stdout (cached)
    - Interpolates "$ENV_VAR" or "${ENV_VAR}" references with the named environment variable
    - In non-command values, "$$" escapes a literal "$" and "$!" escapes a literal "!"
    - Otherwise treats the value as a literal
    """
    if _is_command_reference(config):
        return _execute_command(config)
    return _resolve_template(_parse_config_value_template(config), env)


# Sync by design: every runtime-reachable caller offloads this —
# `AuthStorage.read`, `provider_composer`'s two header resolutions, and
# `ModelRuntime.get_auth` all go through `spawn_blocking`. A new caller
# must do the same: this runs an arbitrary shell command (`!cmd`).
def _execute_with_default_shell(command: str) -> str | None:
    try:
        result = subprocess.run(  # noqa: S602
            command,
            shell=True,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            text=True,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _execute_command_uncached(command_config: str) -> str | None:
    return _execute_with_default_shell(command_config[1:])


def _execute_command(command_config: str) -> str | None:
    with _command_result_cache_lock:
        if command_config in _command_result_cache:
            return _command_result_cache[command_config]

    result = _execute_command_uncached(command_config)
    with _command_result_cache_lock:
        _command_result_cache[command_config] = result
    return result


def resolve_config_value_uncached(config: str, env: dict[str, str] | None = None) -> str | None:
    if _is_command_reference(config):
        return _execute_command_uncached(config)
    return _resolve_template(_parse_config_value_template(config), env)


def resolve_config_value_or_throw(config: str, description: str, env: dict[str, str] | None = None) -> str:
    resolved_value = resolve_config_value_uncached(config, env)
    if resolved_value is not None:
        return resolved_value

    if _is_command_reference(config):
        raise Exception(f"Failed to resolve {description} from shell command: {config[1:]}")

    missing_env_vars = get_missing_config_value_env_var_names(config, env)
    if len(missing_env_vars) == 1:
        raise Exception(f"Failed to resolve {description} from environment variable: {missing_env_vars[0]}")
    if len(missing_env_vars) > 1:
        raise Exception(f"Failed to resolve {description} from environment variables: {', '.join(missing_env_vars)}")

    raise Exception(f"Failed to resolve {description}")


def resolve_headers(headers: dict[str, str] | None, env: dict[str, str] | None = None) -> dict[str, str] | None:
    """Resolve all header values using the same resolution logic as API keys."""
    if headers is None:
        return None
    resolved: dict[str, str] = {}
    for key, value in headers.items():
        resolved_value = resolve_config_value(value, env)
        if resolved_value:
            resolved[key] = resolved_value
    return resolved if resolved else None


def resolve_headers_or_throw(
    headers: dict[str, str] | None, description: str, env: dict[str, str] | None = None
) -> dict[str, str] | None:
    if headers is None:
        return None
    resolved: dict[str, str] = {}
    for key, value in headers.items():
        resolved[key] = resolve_config_value_or_throw(value, f'{description} header "{key}"', env)
    return resolved if resolved else None


def clear_config_value_cache() -> None:
    """Clear the config value command cache. Exported for testing."""
    with _command_result_cache_lock:
        _command_result_cache.clear()
