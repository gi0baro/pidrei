"""Mirror of pi coding-agent test/experimental.test.ts."""

from pidrei.core.experimental import are_experimental_features_enabled


def test_returns_false_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("PIDREI_EXPERIMENTAL", raising=False)
    assert are_experimental_features_enabled() is False


def test_returns_false_when_env_is_empty(monkeypatch):
    monkeypatch.setenv("PIDREI_EXPERIMENTAL", "")
    assert are_experimental_features_enabled() is False


def test_returns_true_when_env_is_set_to_1(monkeypatch):
    monkeypatch.setenv("PIDREI_EXPERIMENTAL", "1")
    assert are_experimental_features_enabled() is True


def test_returns_false_when_env_is_set_to_0(monkeypatch):
    monkeypatch.setenv("PIDREI_EXPERIMENTAL", "0")
    assert are_experimental_features_enabled() is False


def test_returns_false_when_env_is_set_to_a_non_1_value(monkeypatch):
    monkeypatch.setenv("PIDREI_EXPERIMENTAL", "true")
    assert are_experimental_features_enabled() is False
