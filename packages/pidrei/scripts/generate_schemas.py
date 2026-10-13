"""Configuration schema generator — Python port of pi's scripts/generate-schemas.ts.

Renders the JSON Schemas of models.json, settings.json, keybindings.json and
theme files from the schemas the runtime uses, into
`packages/pidrei/pidrei/schemas/<name>.schema.json` (shipped in the wheel, and
published at their GitHub raw URLs for editors). Selected repeated schemas
become `$defs` entries referenced with `$ref`, keeping each use's annotations.

Run: make config-schemas
"""

import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pidrei.core.keybindings_schema import KEYBINDING_VALUE_SCHEMA, KEYBINDINGS_SCHEMA
from pidrei.core.model_config import MODELS_CONFIG_SCHEMA
from pidrei.core.settings_schema import SETTINGS_SCHEMA
from pidrei.modes.interactive.theme.theme_schema import COLOR_VALUE_SCHEMA, THEME_JSON_SCHEMA
from pidrei_ai.providers.compat_schema import PROVIDER_COMPAT_SCHEMA
from pidrei_ai.providers.model_schema import (
    MODEL_COST_SCHEMA,
    MODEL_INPUT_LIMITS_SCHEMA,
    MODEL_PROMPT_CACHE_SCHEMA,
    MODEL_THINKING_LEVEL_SCHEMA,
    THINKING_LEVEL_MAP_SCHEMA,
)


PACKAGE_DIRECTORY = Path(__file__).resolve().parent.parent / "pidrei"
SCHEMA_BASE_URL = "https://raw.githubusercontent.com/gi0baro/pidrei/master/packages/pidrei/pidrei/schemas"
SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"
GENERATED_COMMENT = "This file is generated from Python source. Do not edit it manually; run make config-schemas."

# (filename, schema, definitions)
SCHEMA_ARTIFACTS: tuple[tuple[str, Mapping[str, Any], Mapping[str, Mapping[str, Any]] | None], ...] = (
    (
        "models.schema.json",
        MODELS_CONFIG_SCHEMA,
        {
            "ModelCost": MODEL_COST_SCHEMA,
            "ModelInputLimits": MODEL_INPUT_LIMITS_SCHEMA,
            "ModelPromptCache": MODEL_PROMPT_CACHE_SCHEMA,
            "ProviderCompat": PROVIDER_COMPAT_SCHEMA,
            "ThinkingLevelMap": THINKING_LEVEL_MAP_SCHEMA,
        },
    ),
    ("settings.schema.json", SETTINGS_SCHEMA, {"ModelThinkingLevel": MODEL_THINKING_LEVEL_SCHEMA}),
    ("keybindings.schema.json", KEYBINDINGS_SCHEMA, {"KeybindingValue": KEYBINDING_VALUE_SCHEMA}),
    ("theme.schema.json", THEME_JSON_SCHEMA, {"ColorValue": COLOR_VALUE_SCHEMA}),
)

SCHEMA_ANNOTATION_KEYS = frozenset(
    ("default", "deprecated", "description", "examples", "readOnly", "title", "writeOnly")
)

_DEFINITION_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def _split_annotations(schema: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    shape: dict[str, Any] = {}
    annotations: dict[str, Any] = {}
    for key, value in schema.items():
        (annotations if key in SCHEMA_ANNOTATION_KEYS else shape)[key] = value
    return shape, annotations


def _shape_key(shape: Mapping[str, Any]) -> str:
    # Sorted keys: shapes match whatever order their dicts were built in; JSON keeps `true` apart from `1`.
    return json.dumps(shape, sort_keys=True)


def _add_definitions(schema: Mapping[str, Any], definitions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Replace selected repeated schemas with local references while retaining per-use annotations."""
    prepared_definitions = []
    for name, definition in definitions.items():
        if not _DEFINITION_NAME_RE.match(name):
            raise ValueError(f"Invalid schema definition name: {name}")
        shape, _ = _split_annotations(definition)
        prepared_definitions.append((name, shape, _shape_key(shape)))

    referenced_definitions: set[str] = set()

    def replace_definitions(value: Any, excluded_name: str | None = None) -> Any:
        if isinstance(value, list):
            return [replace_definitions(item, excluded_name) for item in value]
        if not isinstance(value, Mapping):
            return value

        shape, annotations = _split_annotations(value)
        shape_key = _shape_key(shape)
        for name, _, definition_key in prepared_definitions:
            if name != excluded_name and definition_key == shape_key:
                referenced_definitions.add(name)
                return {"$ref": f"#/$defs/{name}", **annotations}

        return {
            key: child if key in SCHEMA_ANNOTATION_KEYS else replace_definitions(child, excluded_name)
            for key, child in value.items()
        }

    compact_schema = replace_definitions(schema)
    compact_definitions = {name: replace_definitions(shape, name) for name, shape, _ in prepared_definitions}
    unused_definitions = [name for name, _, _ in prepared_definitions if name not in referenced_definitions]
    if unused_definitions:
        raise ValueError(f"Unused schema definitions: {', '.join(unused_definitions)}")
    return {**compact_schema, "$defs": compact_definitions}


def _serialize_schema(
    filename: str, schema: Mapping[str, Any], definitions: Mapping[str, Mapping[str, Any]] | None
) -> str:
    rendered = _add_definitions(schema, definitions) if definitions else dict(schema)
    document = {"$schema": SCHEMA_DRAFT, "$id": f"{SCHEMA_BASE_URL}/{filename}", "$comment": GENERATED_COMMENT}
    document.update(rendered)
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def render_config_schemas() -> dict[str, str]:
    """`schemas/<filename>` (relative to the pidrei package) -> file content."""
    return {
        f"schemas/{filename}": _serialize_schema(filename, schema, definitions)
        for filename, schema, definitions in SCHEMA_ARTIFACTS
    }


def generate_schemas() -> None:
    for relative_path, content in render_config_schemas().items():
        path = PACKAGE_DIRECTORY / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        sys.stdout.write(f"Generated packages/pidrei/pidrei/{relative_path}\n")


if __name__ == "__main__":
    generate_schemas()
