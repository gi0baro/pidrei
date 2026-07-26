"""Mirror of pi coding-agent test/oauth-selector.test.ts."""

from types import SimpleNamespace

import pytest

from pidrei.core.keybindings import KeybindingsManager
from pidrei.modes.interactive.components import OAuthSelectorComponent
from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei.modes.interactive.theme import init_theme
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import set_keybindings


@pytest.fixture(autouse=True)
def _setup():
    init_theme("dark")
    set_keybindings(KeybindingsManager())


class TestOAuthSelectorComponent:
    def test_projects_provider_owned_auth_options_without_provider_specific_filtering(self):
        async def login(*_args):
            return {}

        providers = [
            SimpleNamespace(
                id="anthropic",
                name="Anthropic",
                auth=SimpleNamespace(
                    oauth=SimpleNamespace(name="Anthropic (Claude Pro/Max)", login=login),
                    api_key=SimpleNamespace(name="Anthropic API key", login=login),
                ),
            ),
            SimpleNamespace(
                id="google-vertex",
                name="Google Vertex AI",
                auth=SimpleNamespace(oauth=None, api_key=SimpleNamespace(name="Google Cloud credentials")),
            ),
        ]
        fake = SimpleNamespace(
            session=SimpleNamespace(
                model_runtime=SimpleNamespace(
                    get_providers=lambda: providers,
                    get_provider_auth_status=lambda provider_id: SimpleNamespace(
                        configured=False, label=None, source=None
                    ),
                    is_using_oauth=lambda provider_id: False,
                )
            )
        )

        api_key_options = InteractiveMode.get_login_provider_options(fake, "api_key")
        assert [
            (option["id"], option["name"], option["authType"], option["method"].name) for option in api_key_options
        ] == [
            ("anthropic", "Anthropic", "api_key", "Anthropic API key"),
            ("google-vertex", "Google Vertex AI", "api_key", "Google Cloud credentials"),
        ]

        oauth_options = InteractiveMode.get_login_provider_options(fake, "oauth")
        assert [(option["id"], option["name"], option["authType"]) for option in oauth_options] == [
            ("anthropic", "Anthropic", "oauth")
        ]

    def test_renders_an_option_without_compiled_auth_status_as_unconfigured(self):
        selector = OAuthSelectorComponent(
            "login",
            [{"id": "google", "name": "Google", "authType": "api_key", "status": None}],
            lambda provider_id, auth_type: None,
            lambda: None,
        )

        output = strip_ansi("\n".join(selector.render(120)))
        assert "unconfigured" in output
        assert "✓ configured" not in output

    def test_shows_oauth_auth_distinctly_in_the_api_key_selector(self):
        selector = OAuthSelectorComponent(
            "login",
            [
                {
                    "id": "anthropic",
                    "name": "Anthropic",
                    "authType": "api_key",
                    "status": {"type": "oauth", "source": "OAuth"},
                }
            ],
            lambda provider_id, auth_type: None,
            lambda: None,
        )

        output = strip_ansi("\n".join(selector.render(120)))
        assert "subscription configured" in output

    def test_shows_environment_api_key_auth_as_configured(self):
        selector = OAuthSelectorComponent(
            "login",
            [
                {
                    "id": "openai",
                    "name": "OpenAI",
                    "authType": "api_key",
                    "status": {"type": "api_key", "source": "OPENAI_API_KEY"},
                }
            ],
            lambda provider_id, auth_type: None,
            lambda: None,
        )

        output = strip_ansi("\n".join(selector.render(120)))
        assert "✓ env: OPENAI_API_KEY" in output
        assert "unconfigured" not in output

    def test_shows_models_json_api_key_auth_as_configured(self):
        selector = OAuthSelectorComponent(
            "login",
            [
                {
                    "id": "local-proxy",
                    "name": "local-proxy",
                    "authType": "api_key",
                    "status": {"type": "api_key", "source": "key in models.json"},
                }
            ],
            lambda provider_id, auth_type: None,
            lambda: None,
        )

        assert "✓ key in models.json" in strip_ansi("\n".join(selector.render(120)))

    def test_shows_models_json_command_auth_as_configured(self):
        selector = OAuthSelectorComponent(
            "login",
            [
                {
                    "id": "op-proxy",
                    "name": "op-proxy",
                    "authType": "api_key",
                    "status": {"type": "api_key", "source": "command in models.json"},
                }
            ],
            lambda provider_id, auth_type: None,
            lambda: None,
        )

        assert "✓ command in models.json" in strip_ansi("\n".join(selector.render(120)))
