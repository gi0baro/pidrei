"""Mirror of pi coding-agent test/resolve-config-value.test.ts (POSIX-only).

pi's final test ("uses stdin when the configured Windows shell requires it")
exercises the win32 configured-shell path, which is not ported.
"""

import pytest

from pidrei.core.resolve_config_value import (
    clear_config_value_cache,
    resolve_config_value,
    resolve_config_value_uncached,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_config_value_cache()
    yield
    clear_config_value_cache()


def test_resolves_literals_environment_templates_and_escapes(monkeypatch):
    monkeypatch.setenv("TEST_CONFIG_LEFT", "left")
    monkeypatch.setenv("TEST_CONFIG_RIGHT", "right")
    assert resolve_config_value("literal-key") == "literal-key"
    assert resolve_config_value("$TEST_CONFIG_LEFT") == "left"
    assert resolve_config_value("${TEST_CONFIG_LEFT}_$TEST_CONFIG_RIGHT") == "left_right"
    assert resolve_config_value("$$TEST_CONFIG_LEFT") == "$TEST_CONFIG_LEFT"
    assert resolve_config_value("$!literal-$TEST_CONFIG_RIGHT") == "!literal-right"


def test_uses_credential_scoped_environment_before_process_env(monkeypatch):
    monkeypatch.setenv("TEST_CONFIG_SCOPED", "process")
    assert resolve_config_value("$TEST_CONFIG_SCOPED", {"TEST_CONFIG_SCOPED": "credential"}) == "credential"


def test_executes_shell_commands_and_trims_their_output():
    assert resolve_config_value("!echo '  spaced-key  '") == "spaced-key"
    assert resolve_config_value("!printf 'line1\\nline2'") == "line1\nline2"
    assert resolve_config_value("!echo 'hello world' | tr ' ' '-'") == "hello-world"


@pytest.mark.parametrize("command", ["!exit 1", "!nonexistent-command-12345", "!printf ''"])
def test_returns_undefined_when_command_resolution_fails(command):
    assert resolve_config_value(command) is None


def test_caches_successful_and_failed_commands_until_explicitly_cleared(tmp_path):
    counter_file = tmp_path / "counter"
    counter_file.write_text("0")
    escaped_path = str(counter_file).replace("\\", "/").replace('"', '\\"')
    success = f'!sh -c \'count=$(cat "{escaped_path}"); echo $((count + 1)) > "{escaped_path}"; echo value\''

    assert resolve_config_value(success) == "value"
    assert resolve_config_value(success) == "value"
    assert counter_file.read_text().strip() == "1"

    clear_config_value_cache()
    assert resolve_config_value(success) == "value"
    assert counter_file.read_text().strip() == "2"

    failure = f'!sh -c \'count=$(cat "{escaped_path}"); echo $((count + 1)) > "{escaped_path}"; exit 1\''
    assert resolve_config_value(failure) is None
    assert resolve_config_value(failure) is None
    assert counter_file.read_text().strip() == "3"


def test_does_not_cache_environment_values(monkeypatch):
    monkeypatch.setenv("TEST_CONFIG_DYNAMIC", "first")
    assert resolve_config_value("$TEST_CONFIG_DYNAMIC") == "first"
    monkeypatch.setenv("TEST_CONFIG_DYNAMIC", "second")
    assert resolve_config_value("$TEST_CONFIG_DYNAMIC") == "second"


def test_uncached_resolution_executes_a_command_on_every_call(tmp_path):
    counter_file = tmp_path / "uncached-counter"
    counter_file.write_text("0")
    escaped_path = str(counter_file).replace("\\", "/").replace('"', '\\"')
    command = f'!sh -c \'count=$(cat "{escaped_path}"); echo $((count + 1)) > "{escaped_path}"; echo value\''
    assert resolve_config_value_uncached(command) == "value"
    assert resolve_config_value_uncached(command) == "value"
    assert counter_file.read_text().strip() == "2"
