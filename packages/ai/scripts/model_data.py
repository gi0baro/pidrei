"""Port of pi's scripts/model-data.ts — integrity checks for the vendored catalog.

Two deliberate structural differences from pi, both forced by the layout:

- pi derives the expected *structure* (`{provider: {"type:id": api}}`) from its
  committed TypeScript shards (`providers/<id>.models.ts`) and the
  `models.generated.ts` aggregator, then validates the JSON against it. pidrei
  loads the JSON directly and has no shards, so the second representation is the
  `structure` field the generator records in the manifest — the invariant that
  actually survives the translation is "the data files match what the generator
  recorded", and that is what `read_model_data_structure` reads.
- the manifest is `_manifest.json`, not pi's `.manifest.json`: the catalog loader
  skips `_`-prefixed stems, which is how it keeps the manifest out of `MODELS`.
"""

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any


MODEL_DATA_SCHEMA_VERSION = 6
MODEL_DATA_MANIFEST_FILE = "_manifest.json"

type ModelDataStructure = dict[str, dict[str, str]]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sorted_record[T](entries: dict[str, T]) -> dict[str, T]:
    return {key: entries[key] for key in sorted(entries)}


def _describe_set_difference(expected: list[str], actual: list[str]) -> str:
    missing = [value for value in expected if value not in set(actual)]
    extra = [value for value in actual if value not in set(expected)]
    parts = []
    if missing:
        parts.append(f"missing: {', '.join(missing)}")
    if extra:
        parts.append(f"extra: {', '.join(extra)}")
    return "; ".join(parts)


def assert_exact_model_ids(label: str, expected: Iterable[str], actual: Iterable[str]) -> None:
    expected_ids = sorted(set(expected))
    actual_ids = sorted(set(actual))
    if expected_ids == actual_ids:
        return
    raise RuntimeError(f"{label} model IDs do not match ({_describe_set_difference(expected_ids, actual_ids)})")


def _read_json_object(path: Path, description: str, errors: list[str]) -> dict[str, Any] | None:
    try:
        parsed = json.loads(path.read_text())
    except Exception as error:
        errors.append(f"{description} is not valid JSON: {error}")
        return None
    if not isinstance(parsed, dict):
        errors.append(f"{description} must contain a JSON object")
        return None
    return parsed


def read_model_data_structure(data_dir: Path) -> ModelDataStructure:
    """The structure the manifest recorded at generation time.

    pi reads this from its TS shards; see the module docstring for why pidrei
    reads the manifest instead.
    """
    errors: list[str] = []
    manifest = _read_json_object(data_dir / MODEL_DATA_MANIFEST_FILE, "model data manifest", errors)
    if manifest is None:
        raise ValueError("\n".join(errors))
    structure = manifest.get("structure")
    if not isinstance(structure, dict) or not structure:
        raise ValueError("model data manifest has no recorded structure")
    return _sorted_record({provider_id: _sorted_record(models) for provider_id, models in structure.items()})


def model_data_structure_hash(structure: ModelDataStructure) -> str:
    normalized = _sorted_record({provider_id: _sorted_record(models) for provider_id, models in structure.items()})
    # separators match JS `JSON.stringify` so the hash is comparable to pi's.
    return _sha256(json.dumps(normalized, separators=(",", ":")))


def create_model_data_manifest(
    structure: ModelDataStructure,
    file_contents: dict[str, str],
    generated_at: str,
    *,
    source: str | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schemaVersion": MODEL_DATA_SCHEMA_VERSION,
        "generatedAt": generated_at,
        "structureHash": model_data_structure_hash(structure),
        "files": _sorted_record({file: _sha256(content) for file, content in file_contents.items()}),
        "structure": _sorted_record({provider: _sorted_record(models) for provider, models in structure.items()}),
    }
    if source is not None:
        manifest["source"] = source
    return manifest


_VALID_MODALITIES = {"text", "image"}


def _is_modality_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(entry in _VALID_MODALITIES for entry in value)


def _is_positive_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and value > 0


def _validate_model_value(
    value: Any,
    provider_id: str,
    model_id: str,
    expected_api: str,
    errors: list[str],
) -> None:
    label = f"{provider_id}/{model_id}"
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return
    if value.get("id") != model_id:
        errors.append(f"{label} has id {json.dumps(value.get('id'))}, expected {json.dumps(model_id)}")
    if value.get("provider") != provider_id:
        errors.append(f"{label} has provider {json.dumps(value.get('provider'))}, expected {json.dumps(provider_id)}")
    if value.get("api") != expected_api:
        errors.append(f"{label} has api {json.dumps(value.get('api'))}, expected {json.dumps(expected_api)}")
    name = value.get("name")
    if not isinstance(name, str) or not name:
        errors.append(f"{label} has no model name")
    if not isinstance(value.get("baseUrl"), str):
        errors.append(f"{label} has no baseUrl string")
    if not _is_modality_list(value.get("input")):
        errors.append(f"{label} has invalid input modalities")
    model_type = value.get("type")
    if model_type == "image":
        output = value.get("output")
        if not _is_modality_list(output) or "image" not in output:
            errors.append(f"{label} has invalid output modalities")
    elif "output" in value:
        errors.append(f"{label} has unsupported output modalities")
    if model_type == "chat":
        if not isinstance(value.get("reasoning"), bool):
            errors.append(f"{label} has no reasoning boolean")
        for field in ("contextWindow", "maxTokens"):
            if not _is_positive_number(value.get(field)):
                errors.append(f"{label} has invalid {field}")
    elif model_type == "classifier":
        if not _is_positive_number(value.get("contextWindow")):
            errors.append(f"{label} has invalid contextWindow")
    elif model_type != "image":
        type_json = json.dumps(model_type) if "type" in value else "undefined"
        errors.append(f'{label} has type {type_json}, expected "chat", "image", or "classifier"')
    cost = value.get("cost")
    if not isinstance(cost, dict):
        errors.append(f"{label} has invalid cost metadata")
    else:
        for field in ("input", "output", "cacheRead", "cacheWrite"):
            number = cost.get(field)
            if isinstance(number, bool) or not isinstance(number, int | float):
                errors.append(f"{label} has invalid cost.{field}")


def _raise_validation_errors(errors: list[str]) -> None:
    visible = errors[:30]
    suffix = f"\n  ... and {len(errors) - len(visible)} more" if len(errors) > len(visible) else ""
    listing = "\n".join(f"  - {error}" for error in visible)
    raise ValueError(f"Invalid generated model data:\n{listing}{suffix}")


def validate_model_data_directory(structure: ModelDataStructure, data_dir: Path) -> None:
    if not data_dir.is_dir():
        raise ValueError(f"Generated model data directory does not exist: {data_dir}")

    errors: list[str] = []
    expected_files = sorted(f"{provider_id}.json" for provider_id in structure)
    actual_files = sorted(path.name for path in data_dir.glob("*.json") if path.name != MODEL_DATA_MANIFEST_FILE)
    if expected_files != actual_files:
        errors.append(
            "provider data files do not match the generated catalog "
            f"({_describe_set_difference(expected_files, actual_files)})"
        )

    manifest = _read_json_object(data_dir / MODEL_DATA_MANIFEST_FILE, "model data manifest", errors)
    if manifest is None or manifest.get("schemaVersion") != MODEL_DATA_SCHEMA_VERSION:
        found = json.dumps(manifest.get("schemaVersion")) if manifest is not None else "null"
        errors.append(f"model data schema is {found}, expected {MODEL_DATA_SCHEMA_VERSION}")
    generated_at = manifest.get("generatedAt") if manifest is not None else None
    if not isinstance(generated_at, str) or not _parses_as_timestamp(generated_at):
        errors.append("model data manifest has an invalid generation timestamp")
    expected_structure_hash = model_data_structure_hash(structure)
    if manifest is None or manifest.get("structureHash") != expected_structure_hash:
        errors.append("model data generation stamp does not match the generated catalog")
    manifest_files = manifest.get("files") if manifest is not None else None
    if not isinstance(manifest_files, dict):
        manifest_files = None
        errors.append("model data manifest has no file hashes")
    elif expected_files != sorted(manifest_files):
        errors.append(
            "manifest file hashes do not match provider data files "
            f"({_describe_set_difference(expected_files, sorted(manifest_files))})"
        )

    for provider_id, expected_models in structure.items():
        filename = f"{provider_id}.json"
        path = data_dir / filename
        if not path.exists():
            continue
        content = path.read_text()
        if manifest_files is not None and manifest_files.get(filename) != _sha256(content):
            errors.append(f"{filename} does not match its manifest hash")
        groups = _read_json_object(path, filename, errors)
        if groups is None:
            continue

        actual_models: dict[str, str] = {}
        for api, value in groups.items():
            if not isinstance(value, dict):
                errors.append(f"{filename} API group {json.dumps(api)} must be an object")
                continue
            for model_key, model in value.items():
                if model_key in actual_models:
                    errors.append(f"{provider_id}/{model_key} appears in more than one API group")
                    continue
                actual_models[model_key] = api
                model_id = model_key.split(":", 1)[1] if ":" in model_key else model_key
                _validate_model_value(model, provider_id, model_id, api, errors)
                if isinstance(model, dict) and model_key != f"{_js_string(model, 'type')}:{_js_string(model, 'id')}":
                    errors.append(f"{provider_id}/{model_key} has mismatched type/id identity")

        if sorted(expected_models) != sorted(actual_models):
            errors.append(
                f"{filename} model IDs do not match the generated catalog "
                f"({_describe_set_difference(sorted(expected_models), sorted(actual_models))})"
            )
        for model_id, expected_api in expected_models.items():
            actual_api = actual_models.get(model_id)
            if actual_api is not None and actual_api != expected_api:
                errors.append(
                    f"{provider_id}/{model_id} is grouped under API {json.dumps(actual_api)}, "
                    f"expected {json.dumps(expected_api)}"
                )

    if errors:
        _raise_validation_errors(errors)


def _js_string(record: dict[str, Any], key: str) -> str:
    """JS `String(record[key])` for the identity check (a missing key is "undefined")."""
    if key not in record:
        return "undefined"
    value = record[key]
    return value if isinstance(value, str) else json.dumps(value)


def _parses_as_timestamp(value: str) -> bool:
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


def validate_generated_model_data(data_dir: Path) -> None:
    validate_model_data_directory(read_model_data_structure(data_dir), data_dir)
