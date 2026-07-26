"""Mirror of pi coding-agent test/interactive-mode-anthropic-warning.test.ts."""

from types import SimpleNamespace

import pytest

from pidrei.modes.interactive.interactive_mode import InteractiveMode


def _create_model_runtime(credential, api_key=None):
    runtime = SimpleNamespace(check_auth_calls=[], get_auth_calls=[])

    async def check_auth(provider_id):
        runtime.check_auth_calls.append(provider_id)
        return credential

    async def get_auth(provider_id):
        runtime.get_auth_calls.append(provider_id)
        return SimpleNamespace(auth=SimpleNamespace(api_key=api_key)) if api_key else None

    runtime.check_auth = check_auth
    runtime.get_auth = get_auth
    return runtime


def _create_fake(model_runtime, warnings=None):
    fake = SimpleNamespace(
        _anthropic_subscription_warning_shown=False,
        settings_manager=SimpleNamespace(get_warnings=lambda: warnings or {}),
        session=SimpleNamespace(model_runtime=model_runtime),
        warnings_shown=[],
    )
    fake.show_warning = fake.warnings_shown.append
    return fake


@pytest.mark.tonio
async def test_warns_once_when_anthropic_subscription_auth_is_detected():
    model_runtime = _create_model_runtime(None, "sk-ant-oat01-test")
    fake = _create_fake(model_runtime)

    await InteractiveMode._maybe_warn_about_anthropic_subscription_auth(
        fake, SimpleNamespace(provider="anthropic")
    )
    await InteractiveMode._maybe_warn_about_anthropic_subscription_auth(
        fake, SimpleNamespace(provider="anthropic")
    )

    assert len(fake.warnings_shown) == 1
    assert len(model_runtime.get_auth_calls) == 1


@pytest.mark.tonio
async def test_warns_when_anthropic_oauth_is_stored_even_if_token_refresh_lookup_would_fail():
    model_runtime = _create_model_runtime(SimpleNamespace(type="oauth"))
    fake = _create_fake(model_runtime)

    await InteractiveMode._maybe_warn_about_anthropic_subscription_auth(
        fake, SimpleNamespace(provider="anthropic")
    )

    assert len(fake.warnings_shown) == 1
    assert model_runtime.get_auth_calls == []


@pytest.mark.tonio
async def test_does_not_warn_for_non_anthropic_models():
    model_runtime = _create_model_runtime(None)
    fake = _create_fake(model_runtime)

    await InteractiveMode._maybe_warn_about_anthropic_subscription_auth(fake, SimpleNamespace(provider="openai"))

    assert fake.warnings_shown == []
    assert model_runtime.get_auth_calls == []


@pytest.mark.tonio
async def test_does_not_warn_when_anthropic_extra_usage_warning_is_disabled():
    model_runtime = _create_model_runtime(None)
    fake = _create_fake(model_runtime, {"anthropicExtraUsage": False})

    await InteractiveMode._maybe_warn_about_anthropic_subscription_auth(
        fake, SimpleNamespace(provider="anthropic")
    )

    assert fake.warnings_shown == []
    assert model_runtime.check_auth_calls == []
    assert model_runtime.get_auth_calls == []
