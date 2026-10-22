"""Reference page generator — Python port of pi's scripts/generate-references.ts.

Renders `packages/pidrei/pidrei/docs/models-configuration.md` (shipped in the
wheel) from the models.json schemas the runtime validates with, so the field
reference cannot drift from what `models.json` accepts.

Run: make references
"""

import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pidrei.core.model_config import MODELS_JSON_SCHEMAS
from pidrei_ai.providers.compat_schema import COMPAT_SCHEMAS_BY_API
from pidrei_ai.providers.model_schema import MODEL_INPUT_LIMITS_SCHEMA


PACKAGE_DIRECTORY = Path(__file__).resolve().parent.parent / "pidrei"
SCHEMA_BASE_URL = "https://raw.githubusercontent.com/gi0baro/pidrei/master/packages/pidrei/pidrei/schemas"
GENERATED_COMMENT = "This file is generated from Python source. Do not edit it manually; run make references."


@dataclass(slots=True, frozen=True)
class ReferenceSection:
    heading: str
    intro: str | None = None
    # Object schema whose properties become table rows.
    schema: Mapping[str, Any] | None = None
    # Top-level fields whose type links to another section instead of listing nested fields.
    links: Mapping[str, str] = field(default_factory=dict)
    # Top-level fields listed without their nested fields.
    collapsed: Sequence[str] = ()


@dataclass(slots=True, frozen=True)
class ReferenceArtifact:
    path: str
    title: str
    intro: str
    sections: Sequence[ReferenceSection]


def _compat_schema_apis() -> list[tuple[Mapping[str, Any], list[str]]]:
    """APIs grouped by the compatibility schema they share, in first-seen order."""
    groups: dict[int, tuple[Mapping[str, Any], list[str]]] = {}
    for api, schema in COMPAT_SCHEMAS_BY_API.items():
        groups.setdefault(id(schema), (schema, []))[1].append(api)
    return list(groups.values())


def _format_list(items: Sequence[str]) -> str:
    return f"{', '.join(items[:-1])} and {items[-1]}" if len(items) > 1 else (items[0] if items else "")


def _compat_section(schema: Mapping[str, Any], apis: list[str]) -> ReferenceSection:
    api, *aliases = apis
    return ReferenceSection(
        heading=f"### `{api}`",
        intro=f"Also applies to {_format_list([f'`{alias}`' for alias in aliases])}." if aliases else None,
        schema=schema,
        collapsed=("openRouterRouting", "vercelGatewayRouting"),
    )


REFERENCE_ARTIFACTS: tuple[ReferenceArtifact, ...] = (
    ReferenceArtifact(
        path="docs/models-configuration.md",
        title="Models Configuration",
        intro="\n\n".join(
            (
                (
                    "This reference lists the fields of `models.json`. See [Custom models](models.md) for examples "
                    "and [Configuration](configuration.md#agent-directory) for the file location."
                ),
                (
                    f'Add `"$schema": "{SCHEMA_BASE_URL}/models.schema.json"` to `models.json` for editor validation '
                    "and completion."
                ),
            )
        ),
        sections=(
            ReferenceSection(
                heading="Providers",
                intro=(
                    "Each key under `providers` is a provider ID. A built-in provider ID changes that provider. Any "
                    "other ID adds a provider, which needs `baseUrl` and an `api` at provider or model level."
                ),
                schema=MODELS_JSON_SCHEMAS["provider"],
                links={"models": "#models", "modelOverrides": "#model-overrides", "compat": "#compatibility"},
            ),
            ReferenceSection(
                heading="Models",
                intro=(
                    "Each `models` entry adds a model to the provider. An entry whose `id` matches an existing model "
                    "replaces it."
                ),
                schema=MODELS_JSON_SCHEMAS["model"],
                links={"inputLimits": "#input-limits", "compat": "#compatibility"},
            ),
            ReferenceSection(
                heading="Model overrides",
                intro=(
                    "Each `modelOverrides` key is the ID of a built-in or extension model. Unknown IDs are ignored. "
                    "Object fields, such as `cost`, merge with the model's value field by field; other fields "
                    "replace it."
                ),
                schema=MODELS_JSON_SCHEMAS["modelOverride"],
                links={"inputLimits": "#input-limits", "compat": "#compatibility"},
            ),
            ReferenceSection(
                heading="Input limits",
                intro="See [Image input limits](models.md#image-input-limits).",
                schema=MODEL_INPUT_LIMITS_SCHEMA,
            ),
            ReferenceSection(
                heading="Compatibility",
                intro=(
                    "`compat` describes verified differences from an API's standard request and response behavior. "
                    "Model values override provider values per field. The fields that apply depend on the model's "
                    "`api`."
                ),
            ),
            *(_compat_section(schema, apis) for schema, apis in _compat_schema_apis()),
        ),
    ),
)


def _json(value: Any) -> str:
    """JS `JSON.stringify`: no spaces after separators."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _reference_type(schema: Mapping[str, Any]) -> str:
    if "const" in schema:
        return _json(schema["const"])
    if "enum" in schema:
        return " | ".join(_json(value) for value in schema["enum"])
    if "anyOf" in schema:
        return " | ".join(dict.fromkeys(_reference_type(branch) for branch in schema["anyOf"]))
    if schema.get("type") == "array":
        items = schema.get("items")
        item = _reference_type(items) if items is not None else "unknown"
        return f"({item})[]" if " | " in item else f"{item}[]"
    return schema.get("type", "unknown")


def _documented_object(schema: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]] | None:
    """Nested object whose fields get their own rows, if any of them carry documentation."""
    if schema.get("type") == "array":
        items = schema.get("items")
        item = _documented_object(items) if items is not None else None
        return (f"[]{item[0]}", item[1]) if item is not None else None
    if "properties" in schema:
        found: Mapping[str, Any] | None = schema
    else:
        found = next((branch for branch in schema.get("anyOf", ()) if "properties" in branch), None)
    if found is None:
        return None
    documented = any("description" in entry or "default" in entry for entry in found["properties"].values())
    return ("", found) if documented else None


def _escape_table_text(text: str) -> str:
    return text.replace("|", "\\|").replace("<", "&lt;")


_PLAIN_TYPE_RE = re.compile(r"[a-z]+")


def _reference_rows(schema: Mapping[str, Any], section: ReferenceSection, prefix: str = "") -> list[str]:
    rows: list[str] = []
    for name, field_schema in schema.get("properties", {}).items():
        path = f"{prefix}{name}"
        link = section.links.get(name) if prefix == "" else None
        type_ = _escape_table_text(_reference_type(field_schema))
        type_text = type_ if _PLAIN_TYPE_RE.fullmatch(type_) else f"`{type_}`"
        default_text = f"`{_escape_table_text(_json(field_schema['default']))}`" if "default" in field_schema else ""
        description = " ".join(
            text
            for text in ("Required." if name in schema.get("required", ()) else None, field_schema.get("description"))
            if text is not None
        )
        rows.append(
            f"| `{path}` | {f'[{type_text}]({link})' if link else type_text} | {default_text} | "
            f"{_escape_table_text(description)} |"
        )
        if link or (prefix == "" and name in section.collapsed):
            continue
        nested = _documented_object(field_schema)
        if nested is not None:
            suffix, nested_schema = nested
            rows.extend(_reference_rows(nested_schema, section, f"{path}{suffix}."))
    return rows


def _render_reference(artifact: ReferenceArtifact) -> str:
    blocks = [f"# {artifact.title}", f"<!-- {GENERATED_COMMENT} -->", artifact.intro]
    for section in artifact.sections:
        blocks.append(section.heading if section.heading.startswith("#") else f"## {section.heading}")
        if section.intro:
            blocks.append(section.intro)
        if section.schema is not None:
            rows = _reference_rows(section.schema, section)
            blocks.append("\n".join(("| Field | Type | Default | Description |", "|---|---|---|---|", *rows)))
    return "\n\n".join(blocks) + "\n"


def render_references() -> dict[str, str]:
    """`docs/<page>.md` (relative to the pidrei package) -> page content."""
    return {artifact.path: _render_reference(artifact) for artifact in REFERENCE_ARTIFACTS}


def generate_references() -> None:
    for relative_path, content in render_references().items():
        path = PACKAGE_DIRECTORY / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        sys.stdout.write(f"Generated packages/pidrei/pidrei/{relative_path}\n")


if __name__ == "__main__":
    generate_references()
