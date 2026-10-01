"""Mirror of pi's edit-tool-legacy-input.test.ts."""

import json
import os

import pytest

from pidrei.core.extensions.types import ExtensionContext
from pidrei.core.tools.edit import create_edit_tool_definition
from pidrei_ai.types import TextContent


class TestEditToolPrepareArguments:
    def test_keeps_legacy_fields_out_of_the_public_schema(self):
        definition = create_edit_tool_definition(os.getcwd())
        assert "oldText" not in definition.parameters["properties"]
        assert "newText" not in definition.parameters["properties"]

    def test_folds_top_level_old_text_new_text_into_edits(self):
        definition = create_edit_tool_definition(os.getcwd())
        prepared = definition.prepare_arguments({"path": "file.txt", "oldText": "before", "newText": "after"})
        assert prepared == {"path": "file.txt", "edits": [{"oldText": "before", "newText": "after"}]}

    def test_appends_legacy_replacement_to_existing_edits(self):
        definition = create_edit_tool_definition(os.getcwd())
        prepared = definition.prepare_arguments(
            {"path": "file.txt", "edits": [{"oldText": "a", "newText": "b"}], "oldText": "c", "newText": "d"}
        )
        assert prepared == {
            "path": "file.txt",
            "edits": [{"oldText": "a", "newText": "b"}, {"oldText": "c", "newText": "d"}],
        }

    def test_passes_through_valid_input_unchanged(self):
        definition = create_edit_tool_definition(os.getcwd())
        payload = {"path": "file.txt", "edits": [{"oldText": "a", "newText": "b"}]}
        assert definition.prepare_arguments(payload) is payload

    def test_passes_through_non_object_input_unchanged(self):
        definition = create_edit_tool_definition(os.getcwd())
        assert definition.prepare_arguments(None) is None
        assert definition.prepare_arguments("garbage") == "garbage"

    @pytest.mark.tonio
    async def test_prepared_args_execute_correctly(self, tmp_path):
        file_path = tmp_path / "legacy.txt"
        file_path.write_text("before\n", encoding="utf-8")
        definition = create_edit_tool_definition(str(tmp_path))
        prepared = definition.prepare_arguments({"path": "legacy.txt", "oldText": "before", "newText": "after"})

        result = await definition.execute("tool-1", prepared, None, None, ExtensionContext())

        assert result.content == [TextContent(text="Successfully replaced 1 block(s) in legacy.txt.")]
        assert file_path.read_text(encoding="utf-8") == "after\n"


class TestEditToolStringifiedEdits:
    def test_parses_edits_from_a_json_string(self):
        definition = create_edit_tool_definition(os.getcwd())
        prepared = definition.prepare_arguments(
            {"path": "file.txt", "edits": json.dumps([{"oldText": "a", "newText": "b"}])}
        )
        assert prepared == {"path": "file.txt", "edits": [{"oldText": "a", "newText": "b"}]}

    def test_leaves_edits_alone_when_the_string_is_not_valid_json(self):
        definition = create_edit_tool_definition(os.getcwd())
        prepared = definition.prepare_arguments({"path": "file.txt", "edits": "not json"})
        assert prepared == {"path": "file.txt", "edits": "not json"}
