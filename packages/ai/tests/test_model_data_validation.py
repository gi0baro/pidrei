"""Mirror of pi's model-data-validation.test.ts.

pi's fixture writes a TS shard plus the aggregator so `readModelDataStructure`
can derive the expected structure from them; pidrei records the structure in the
manifest instead (see scripts/model_data.py), so the fixture writes only the
data file and its manifest. pi's last case — "rejects missing provider shards
imported by the aggregator" — has no analogue for the same reason; its stand-in
here is that a manifest with no recorded structure is rejected.
"""

import importlib.util
import json
from pathlib import Path

import pytest


def _load_model_data():
    """Import the sibling generator script (not an installed module)."""
    path = Path(__file__).parents[1] / "scripts" / "model_data.py"
    spec = importlib.util.spec_from_file_location("pidrei_ai_scripts_model_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model_data = _load_model_data()

GENERATED_AT = "2026-07-23T10:00:00+00:00"

MODEL_A = {
    "id": "model-a",
    "name": "Model A",
    "api": "openai-completions",
    "provider": "test-provider",
    "baseUrl": "https://example.test/v1",
    "reasoning": False,
    "input": ["text"],
    "cost": {"input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0},
    "contextWindow": 1000,
    "maxTokens": 100,
}


def write_fixture_data(
    data_dir: Path,
    structure: dict,
    values: dict,
    schema_version: int | None = None,
    api_group: str = "openai-completions",
) -> None:
    filename = "test-provider.json"
    content = json.dumps({api_group: values}) + "\n"
    (data_dir / filename).write_text(content)
    manifest = model_data.create_model_data_manifest(structure, {filename: content}, GENERATED_AT)
    if schema_version is not None:
        manifest["schemaVersion"] = schema_version
    (data_dir / model_data.MODEL_DATA_MANIFEST_FILE).write_text(json.dumps(manifest) + "\n")


def create_fixture(tmp_path: Path) -> tuple[Path, dict, dict]:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    structure = {"test-provider": {"model-a": "openai-completions"}}
    values = {"model-a": dict(MODEL_A)}
    write_fixture_data(data_dir, structure, values)
    return data_dir, structure, values


def read_manifest(data_dir: Path) -> dict:
    return json.loads((data_dir / model_data.MODEL_DATA_MANIFEST_FILE).read_text())


def write_manifest(data_dir: Path, manifest: dict) -> None:
    (data_dir / model_data.MODEL_DATA_MANIFEST_FILE).write_text(json.dumps(manifest) + "\n")


def test_rejects_a_missing_upstream_model_from_an_exact_generated_allowlist():
    with pytest.raises(RuntimeError) as excinfo:
        model_data.assert_exact_model_ids("qwen-token-plan-individual", ["model-a", "model-b"], ["model-a"])
    assert "qwen-token-plan-individual model IDs do not match (missing: model-b)" in str(excinfo.value)


def test_rejects_an_unexpected_model_from_an_exact_generated_allowlist():
    with pytest.raises(RuntimeError) as excinfo:
        model_data.assert_exact_model_ids("test-provider", ["model-a"], ["model-a", "model-b"])
    assert "test-provider model IDs do not match (extra: model-b)" in str(excinfo.value)


def test_reads_and_validates_api_grouped_model_data(tmp_path):
    data_dir, structure, _ = create_fixture(tmp_path)

    assert model_data.read_model_data_structure(data_dir) == structure
    model_data.validate_model_data_directory(structure, data_dir)


def test_rejects_a_missing_model_data_directory(tmp_path):
    data_dir, structure, _ = create_fixture(tmp_path)
    for path in data_dir.iterdir():
        path.unlink()
    data_dir.rmdir()

    with pytest.raises(ValueError, match="does not exist"):
        model_data.validate_model_data_directory(structure, data_dir)


@pytest.mark.parametrize(
    ("field", "value", "expected_message"),
    [
        ("id", "wrong-id", "has id"),
        ("provider", "wrong-provider", "has provider"),
        ("api", "anthropic-messages", "has api"),
    ],
)
def test_rejects_a_wrong_model_field(tmp_path, field, value, expected_message):
    data_dir, structure, values = create_fixture(tmp_path)
    values["model-a"][field] = value
    write_fixture_data(data_dir, structure, values)

    with pytest.raises(ValueError, match=expected_message):
        model_data.validate_model_data_directory(structure, data_dir)


def test_rejects_a_model_in_the_wrong_api_group(tmp_path):
    data_dir, structure, values = create_fixture(tmp_path)
    write_fixture_data(data_dir, structure, values, api_group="anthropic-messages")

    with pytest.raises(ValueError, match="grouped under API"):
        model_data.validate_model_data_directory(structure, data_dir)


def test_rejects_duplicate_model_ids_across_api_groups(tmp_path):
    data_dir, structure, values = create_fixture(tmp_path)
    filename = "test-provider.json"
    content = json.dumps({"openai-completions": values, "anthropic-messages": values}) + "\n"
    (data_dir / filename).write_text(content)
    write_manifest(data_dir, model_data.create_model_data_manifest(structure, {filename: content}, GENERATED_AT))

    with pytest.raises(ValueError, match="more than one API group"):
        model_data.validate_model_data_directory(structure, data_dir)


def test_rejects_missing_model_ids_and_stale_file_hashes(tmp_path):
    data_dir, structure, _ = create_fixture(tmp_path)
    (data_dir / "test-provider.json").write_text("{}\n")

    with pytest.raises(ValueError, match="manifest hash|model IDs"):
        model_data.validate_model_data_directory(structure, data_dir)


def test_rejects_incompatible_schema_and_generation_stamps(tmp_path):
    data_dir, structure, values = create_fixture(tmp_path)
    write_fixture_data(data_dir, structure, values, schema_version=model_data.MODEL_DATA_SCHEMA_VERSION + 1)

    with pytest.raises(ValueError, match="model data schema"):
        model_data.validate_model_data_directory(structure, data_dir)

    manifest = read_manifest(data_dir)
    manifest["schemaVersion"] = model_data.MODEL_DATA_SCHEMA_VERSION
    manifest["structureHash"] = "stale"
    write_manifest(data_dir, manifest)

    with pytest.raises(ValueError, match="generation stamp"):
        model_data.validate_model_data_directory(structure, data_dir)


def test_rejects_an_invalid_generation_timestamp(tmp_path):
    data_dir, structure, _ = create_fixture(tmp_path)
    manifest = read_manifest(data_dir)
    manifest["generatedAt"] = "invalid"
    write_manifest(data_dir, manifest)

    with pytest.raises(ValueError, match="generation timestamp"):
        model_data.validate_model_data_directory(structure, data_dir)


def test_rejects_a_manifest_with_no_recorded_structure(tmp_path):
    data_dir, _, _ = create_fixture(tmp_path)
    manifest = read_manifest(data_dir)
    del manifest["structure"]
    write_manifest(data_dir, manifest)

    with pytest.raises(ValueError, match="no recorded structure"):
        model_data.read_model_data_structure(data_dir)


def test_validates_the_committed_catalog(tmp_path):
    """The shipped data dir passes its own integrity checks."""
    data_dir = Path(model_data.__file__).parents[1] / "pidrei_ai" / "providers" / "data"

    model_data.validate_generated_model_data(data_dir)
