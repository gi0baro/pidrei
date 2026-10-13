"""Mirror of pi coding-agent test/config-schemas.test.ts.

pi checks documents with TypeBox's compiled validator; here the rendered
schemas run through `jsonschema`. The schemas live at PiDrei's own URLs and
describe PiDrei's settings (see `pidrei/core/settings_schema.py`), so the
theme title and the schema URLs follow the app-name rename.
"""

import importlib.util
import json
from pathlib import Path

import jsonschema
import pytest

from pidrei.core.keybindings import KeybindingsManager
from pidrei.core.model_config import ModelConfig
from pidrei.core.settings_manager import SettingsManager
from pidrei.core.settings_schema import SETTINGS_SCHEMA
from pidrei.modes.interactive.theme import validate_theme_json


SCHEMA_BASE_URL = "https://raw.githubusercontent.com/gi0baro/pidrei/master/packages/pidrei/pidrei/schemas"
PACKAGE_DIRECTORY = Path(__file__).parents[1] / "pidrei"


def _load_generate_schemas():
    """Import the generator script (not an installed module)."""
    path = Path(__file__).parents[1] / "scripts" / "generate_schemas.py"
    spec = importlib.util.spec_from_file_location("pidrei_scripts_generate_schemas", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


render_config_schemas = _load_generate_schemas().render_config_schemas


def _schema_url(name: str) -> str:
    return f"{SCHEMA_BASE_URL}/{name}.schema.json"


def _rendered(relative_path: str) -> dict:
    return json.loads(render_config_schemas()[relative_path])


def _validator(schema: dict) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(schema)


def test_matches_the_four_committed_artifacts():
    rendered = render_config_schemas()
    assert list(rendered) == [
        "schemas/models.schema.json",
        "schemas/settings.schema.json",
        "schemas/keybindings.schema.json",
        "schemas/theme.schema.json",
    ]
    for relative_path, expected in rendered.items():
        committed = (PACKAGE_DIRECTORY / relative_path).read_text(encoding="utf-8")
        assert committed == expected, f"{relative_path} is stale; run make config-schemas"


def test_preserves_editor_guidance_in_the_generated_theme_schema():
    schema = _rendered("schemas/theme.schema.json")
    assert schema["title"] == "PiDrei Coding Agent Theme"
    assert schema["description"] == "Theme schema for PiDrei coding agent"
    colors = schema["properties"]["colors"]
    assert "use compatible fallbacks" in colors["description"]
    assert "falls back to muted" in colors["properties"]["scrollbarTrack"]["description"]
    accent = colors["properties"]["accent"]
    assert accent["description"] == "Primary accent color (logo, selected items, cursor)"
    assert "defaults derived from userMessageBg" in schema["properties"]["export"]["description"]
    color_value = schema["$defs"]["ColorValue"] if "$ref" in accent else accent
    assert len(color_value["anyOf"]) == 2
    assert "Hex color" in color_value["anyOf"][0]["description"]


def test_preserves_moved_model_and_settings_guidance():
    models = _rendered("schemas/models.schema.json")
    cost = models["$defs"]["ModelCost"]["properties"]
    assert "USD per million tokens" in cost["input"]["description"]
    assert "highest matching input threshold" in cost["tiers"]["description"]
    input_limits = models["$defs"]["ModelInputLimits"]["properties"]
    assert "serialized provider request size" in input_limits["maxRequestBytes"]["description"]

    settings = _rendered("schemas/settings.schema.json")["properties"]
    assert "hide all startup output" in settings["quietStartup"]["description"]
    assert "+name and -name" in settings["defaultTools"]["description"]
    assert "1 to 100" in settings["fullscreenWheelScrollLines"]["description"]


@pytest.mark.parametrize(
    ("artifact", "valid", "invalid"),
    [
        pytest.param(
            "schemas/models.schema.json",
            {
                "providers": {
                    "local": {
                        "compat": {
                            "zaiToolStream": True,
                            "thinkingTokenBudgetField": "thinking_budget",
                            "chatTemplateKwargs": {"budget": {"$var": "thinking.budget"}},
                            "supportsMidConvoSystemMessages": True,
                        },
                        "models": [
                            {
                                "id": "model",
                                "inputLimits": {"maxRequestBytes": 1024, "images": {"maxPerRequest": 2}},
                                "promptCache": {"short": 300},
                            }
                        ],
                    }
                }
            },
            {"providers": {"local": {"models": [{"id": ""}]}}},
            id="models.json",
        ),
        pytest.param(
            "schemas/settings.schema.json",
            {
                "theme": "dark",
                "cacheWarming": "idle",
                "extensionSetting": {"enabled": True},
                "queueMode": "all",
                "websockets": True,
                "skills": {"enableSkillCommands": True, "customDirectories": ["./skills"]},
                "retry": {"maxDelayMs": 1000},
            },
            {"cacheWarming": "always"},
            id="settings.json",
        ),
        pytest.param(
            "schemas/keybindings.schema.json",
            {"app.session.new": "ctrl+n", "extension.action": ["alt+x"]},
            {"app.session.new": 42},
            id="keybindings.json",
        ),
    ],
)
def test_validates_representative_documents(artifact, valid, invalid):
    validator = _validator(_rendered(artifact))
    assert validator.is_valid(valid)
    assert not validator.is_valid(invalid)


def test_rejects_non_positive_model_token_limits():
    validator = _validator(_rendered("schemas/models.schema.json"))
    for field in ("contextWindow", "maxTokens"):
        for value in (0, -1):
            assert not validator.is_valid({"providers": {"local": {"models": [{"id": "model", field: value}]}}}), (
                f"models[].{field}={value}"
            )
            assert not validator.is_valid({"providers": {"local": {"modelOverrides": {"model": {field: value}}}}}), (
                f"modelOverrides.{field}={value}"
            )
    assert validator.is_valid(
        {
            "providers": {
                "local": {
                    "models": [{"id": "model", "contextWindow": 1, "maxTokens": 1}],
                    "modelOverrides": {"model": {"contextWindow": 1, "maxTokens": 1}},
                }
            }
        }
    )


def test_validates_keybinding_syntax():
    validator = _validator(_rendered("schemas/keybindings.schema.json"))
    for binding in ("a", "9", "pageUp", "+", "ctrl+shift+x", "alt+ctrl+?", "ctrl+shift+alt+super+f12"):
        assert validator.is_valid({"extension.action": binding}), binding
    for binding in ("", "Ctrl+x", "control+x", "ctrl+not-a-key", "ctrl+ctrl+x", "ctrl+shift+alt+super+ctrl+x"):
        assert not validator.is_valid({"extension.action": binding}), binding


def test_rejects_settings_values_that_runtime_accessors_reject():
    validator = _validator(SETTINGS_SCHEMA)
    max_safe_integer = 2**53 - 1
    for invalid in (
        {"compaction": {"reserveTokens": -1}},
        {"compaction": {"keepRecentTokens": 1.5}},
        {"compaction": {"modelOverrides": {"provider/model": {"reserveTokens": max_safe_integer + 1}}}},
        {"httpIdleTimeoutMs": -1},
        {"websocketConnectTimeoutMs": -1},
        {"httpIdleTimeoutMs": "bogus"},
        {"websocketConnectTimeoutMs": "30000"},
        {"codemode": {"inlineBudget": -1}},
    ):
        assert not validator.is_valid(invalid), invalid
    assert validator.is_valid(
        {
            "compaction": {
                "reserveTokens": 0,
                "keepRecentTokens": max_safe_integer,
                "modelOverrides": {"provider/model": {"reserveTokens": 0}},
            },
            "httpIdleTimeoutMs": 0,
            "websocketConnectTimeoutMs": "disabled",
            "codemode": {"inlineBudget": 0},
        }
    )


def test_describes_legacy_settings_migrations_accurately():
    schema = _rendered("schemas/settings.schema.json")
    assert "provider.maxRetryDelayMs" in schema["properties"]["retry"]["properties"]["maxDelayMs"]["description"]
    assert schema["properties"]["queueMode"]["description"] == "Legacy setting migrated to steeringMode."


@pytest.mark.tonio
async def test_accepts_custom_compatibility_settings_and_rejects_malformed_known_fields(tmp_path):
    path = tmp_path / "models.json"
    compat = {"customOption": "value", "supportsStore": True, "supportsAdditionalTools": True}
    path.write_text(json.dumps({"providers": {"demo": {"api": "custom-api", "compat": compat}}}), encoding="utf-8")

    config = await ModelConfig.load(str(path))
    assert config.get_error() is None
    assert config.get_provider("demo")["compat"] == compat

    path.write_text(
        json.dumps({"providers": {"demo": {"compat": {"supportsLongCacheRetention": "yes"}}}}), encoding="utf-8"
    )
    assert "Invalid models.json schema" in (await ModelConfig.load(str(path))).get_error()


@pytest.mark.tonio
async def test_validates_schema_in_models_json_at_runtime(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "$schema": _schema_url("models"),
                "providers": {"demo": {"baseUrl": "http://localhost:8080/v1", "models": [{"id": "demo"}]}},
            }
        ),
        encoding="utf-8",
    )

    config = await ModelConfig.load(str(path))
    assert config.get_error() is None
    assert config.get_provider("demo")["models"][0]["id"] == "demo"

    path.write_text(
        json.dumps({"$schema": _schema_url("models"), "providers": {"demo": {"models": [{"id": ""}]}}}),
        encoding="utf-8",
    )
    assert "Invalid models.json schema" in (await ModelConfig.load(str(path))).get_error()

    path.write_text(json.dumps({"$schema": 42, "providers": {}}), encoding="utf-8")
    assert "$schema" in (await ModelConfig.load(str(path))).get_error()


@pytest.mark.tonio
async def test_accepts_schema_in_settings_json_and_preserves_unknown_settings(tmp_path):
    (tmp_path / "settings.json").write_text(
        json.dumps({"$schema": _schema_url("settings"), "theme": "light", "extensionSetting": {"enabled": True}}),
        encoding="utf-8",
    )

    manager = await SettingsManager(str(tmp_path), str(tmp_path))
    assert manager.get_theme() == "light"
    global_settings = manager.get_global_settings()
    assert global_settings["$schema"] == _schema_url("settings")
    assert global_settings["extensionSetting"] == {"enabled": True}


@pytest.mark.tonio
async def test_accepts_schema_in_keybindings_json_without_treating_it_as_an_action(tmp_path):
    (tmp_path / "keybindings.json").write_text(
        json.dumps(
            {
                "$schema": _schema_url("keybindings"),
                "app.session.new": "ctrl+n",
                "extension.invalid": "ctrl+not-a-key",
            }
        ),
        encoding="utf-8",
    )

    manager = await KeybindingsManager(str(tmp_path))
    assert manager.get_user_bindings() == {"app.session.new": "ctrl+n"}
    assert manager.get_effective_config()["app.session.new"] == "ctrl+n"


def test_accepts_schema_in_themes_and_rejects_malformed_themes():
    built_in = json.loads(
        (PACKAGE_DIRECTORY / "modes" / "interactive" / "theme" / "dark.json").read_text(encoding="utf-8")
    )
    assert validate_theme_json("custom", {**built_in, "$schema": _schema_url("theme"), "name": "custom"})["name"] == (
        "custom"
    )
    with pytest.raises(ValueError, match="theme names cannot contain"):
        validate_theme_json("invalid", {**built_in, "name": "invalid/name"})
    with pytest.raises(ValueError, match="Missing required color tokens"):
        validate_theme_json("invalid", {**built_in, "colors": {}})
    for invalid in (
        {**built_in, "unexpected": True},
        {**built_in, "colors": {**built_in["colors"], "scrollbarThmb": ""}},
        {**built_in, "export": {"unexpected": ""}},
    ):
        with pytest.raises(ValueError, match="(?i)additional properties"):
            validate_theme_json("invalid", invalid)
