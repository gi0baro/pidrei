"""Mirror of pi coding-agent test/resolve-config-value.test.ts (POSIX-only).

pi's final test ("uses stdin when the configured Windows shell requires it")
exercises the win32 configured-shell path, which is not ported.

Resolution is async now (a `!cmd` runs through `utils.process.run_command`
rather than a blocking-pool thread), so these are tonio tests. Env
vars and temp directories are managed by hand (predates tonio 0.9.14;
`monkeypatch`/`tmp_path` work in tonio tests now).
"""

import os
import tempfile

import pytest

from pidrei.core.resolve_config_value import (
    clear_config_value_cache,
    resolve_config_value,
    resolve_config_value_uncached,
)


def _set_env(**values: str | None) -> dict[str, str | None]:
    """Set env vars, returning what they were so a `finally` can restore them."""
    previous = {name: os.environ.get(name) for name in values}
    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _counter_file(name: str) -> str:
    path = os.path.join(tempfile.mkdtemp(), name)
    with open(path, "w") as handle:
        handle.write("0")
    return path


def _read(path: str) -> str:
    with open(path) as handle:
        return handle.read().strip()


def _counting_command(path: str, tail: str) -> str:
    escaped = path.replace("\\", "/").replace('"', '\\"')
    return f'!sh -c \'count=$(cat "{escaped}"); echo $((count + 1)) > "{escaped}"; {tail}\''


@pytest.mark.tonio
async def test_resolves_literals_environment_templates_and_escapes():
    previous = _set_env(TEST_CONFIG_LEFT="left", TEST_CONFIG_RIGHT="right")
    try:
        assert await resolve_config_value("literal-key") == "literal-key"
        assert await resolve_config_value("$TEST_CONFIG_LEFT") == "left"
        assert await resolve_config_value("${TEST_CONFIG_LEFT}_$TEST_CONFIG_RIGHT") == "left_right"
        assert await resolve_config_value("$$TEST_CONFIG_LEFT") == "$TEST_CONFIG_LEFT"
        assert await resolve_config_value("$!literal-$TEST_CONFIG_RIGHT") == "!literal-right"
    finally:
        _restore_env(previous)


@pytest.mark.tonio
async def test_uses_credential_scoped_environment_before_process_env():
    previous = _set_env(TEST_CONFIG_SCOPED="process")
    try:
        resolved = await resolve_config_value("$TEST_CONFIG_SCOPED", {"TEST_CONFIG_SCOPED": "credential"})
        assert resolved == "credential"
    finally:
        _restore_env(previous)


@pytest.mark.tonio
async def test_executes_shell_commands_and_trims_their_output():
    clear_config_value_cache()
    try:
        assert await resolve_config_value("!echo '  spaced-key  '") == "spaced-key"
        assert await resolve_config_value("!printf 'line1\\nline2'") == "line1\nline2"
        assert await resolve_config_value("!echo 'hello world' | tr ' ' '-'") == "hello-world"
    finally:
        clear_config_value_cache()


@pytest.mark.tonio
@pytest.mark.parametrize("command", ["!exit 1", "!nonexistent-command-12345", "!printf ''"])
async def test_returns_undefined_when_command_resolution_fails(command):
    clear_config_value_cache()
    try:
        assert await resolve_config_value(command) is None
    finally:
        clear_config_value_cache()


@pytest.mark.tonio
async def test_caches_successful_and_failed_commands_until_explicitly_cleared():
    clear_config_value_cache()
    counter = _counter_file("counter")
    try:
        success = _counting_command(counter, "echo value")
        assert await resolve_config_value(success) == "value"
        assert await resolve_config_value(success) == "value"
        assert _read(counter) == "1"

        clear_config_value_cache()
        assert await resolve_config_value(success) == "value"
        assert _read(counter) == "2"

        failure = _counting_command(counter, "exit 1")
        assert await resolve_config_value(failure) is None
        assert await resolve_config_value(failure) is None
        assert _read(counter) == "3"
    finally:
        clear_config_value_cache()


@pytest.mark.tonio
async def test_does_not_cache_environment_values():
    previous = _set_env(TEST_CONFIG_DYNAMIC="first")
    try:
        assert await resolve_config_value("$TEST_CONFIG_DYNAMIC") == "first"
        os.environ["TEST_CONFIG_DYNAMIC"] = "second"
        assert await resolve_config_value("$TEST_CONFIG_DYNAMIC") == "second"
    finally:
        _restore_env(previous)


@pytest.mark.tonio
async def test_uncached_resolution_executes_a_command_on_every_call():
    clear_config_value_cache()
    counter = _counter_file("uncached-counter")
    try:
        command = _counting_command(counter, "echo value")
        assert await resolve_config_value_uncached(command) == "value"
        assert await resolve_config_value_uncached(command) == "value"
        assert _read(counter) == "2"
    finally:
        clear_config_value_cache()
