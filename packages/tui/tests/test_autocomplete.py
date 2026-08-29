"""Mirror of pi tui test/autocomplete.test.ts."""

import os
import shutil

import pytest

from pidrei_tui.autocomplete import CombinedAutocompleteProvider
from pidrei_tui.components.cancellable_loader import CancelToken


def _resolve_fd_path():
    return shutil.which("fd")


def _setup_folder(base_dir, dirs=None, files=None):
    for directory in dirs or []:
        os.makedirs(os.path.join(base_dir, directory), exist_ok=True)
    for file_path, contents in (files or {}).items():
        full_path = os.path.join(base_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as handle:
            handle.write(contents)


_FD_PATH = _resolve_fd_path()

requires_fd = pytest.mark.skipif(_FD_PATH is None, reason="fd is not available")


def _require_fd_path():
    if not _FD_PATH:
        raise Exception("fd is not available")
    return _FD_PATH


async def get_suggestions(provider, lines, cursor_line, cursor_col, force=False):
    return await provider.get_suggestions(lines, cursor_line, cursor_col, {"signal": CancelToken(), "force": force})


@pytest.fixture
def fd_dirs(tmp_path):
    """Root/cwd/outside layout used by the fd @ suggestion tests."""
    base_dir = tmp_path / "cwd"
    outside_dir = tmp_path / "outside"
    base_dir.mkdir()
    outside_dir.mkdir()
    return {"root": str(tmp_path), "base": str(base_dir), "outside": str(outside_dir)}


# extractPathPrefix


@pytest.mark.tonio
async def test_extracts_slash_from_hey_slash_when_forced():
    provider = CombinedAutocompleteProvider([], "/tmp")
    lines = ["hey /"]
    cursor_line = 0
    cursor_col = 5  # After the "/"

    result = await get_suggestions(provider, lines, cursor_line, cursor_col, True)

    assert result is not None, "Should return suggestions for root directory"
    assert result["prefix"] == "/", "Prefix should be '/'"


@pytest.mark.tonio
async def test_extracts_slash_a_from_slash_a_when_forced():
    provider = CombinedAutocompleteProvider([], "/tmp")
    lines = ["/A"]
    cursor_line = 0
    cursor_col = 2  # After the "A"

    result = await get_suggestions(provider, lines, cursor_line, cursor_col, True)

    # This might return None if /A doesn't match anything, which is fine
    # We're mainly testing that the prefix extraction works
    if result is not None:
        assert result["prefix"] == "/A", "Prefix should be '/A'"


@pytest.mark.tonio
async def test_does_not_trigger_for_slash_commands():
    provider = CombinedAutocompleteProvider([], "/tmp")
    lines = ["/model"]
    cursor_line = 0
    cursor_col = 6  # After "model"

    result = await get_suggestions(provider, lines, cursor_line, cursor_col, True)

    assert result is None, "Should not trigger for slash commands"


@pytest.mark.tonio
async def test_triggers_for_absolute_paths_after_slash_command_argument():
    provider = CombinedAutocompleteProvider([], "/tmp")
    lines = ["/command /"]
    cursor_line = 0
    cursor_col = 10  # After the second "/"

    result = await get_suggestions(provider, lines, cursor_line, cursor_col, True)

    assert result is not None, "Should trigger for absolute paths in command arguments"
    assert result["prefix"] == "/", "Prefix should be '/'"


# fd @ file suggestions


@requires_fd
@pytest.mark.tonio
async def test_returns_all_files_and_folders_for_empty_at_query(fd_dirs):
    _setup_folder(fd_dirs["base"], dirs=["src"], files={"README.md": "readme"})

    provider = CombinedAutocompleteProvider([], fd_dirs["base"], _require_fd_path())
    line = "@"
    result = await get_suggestions(provider, [line], 0, len(line))

    values = sorted(item["value"] for item in result["items"])
    assert values == sorted(["@README.md", "@src/"])


@requires_fd
@pytest.mark.tonio
async def test_matches_file_with_extension_in_query(fd_dirs):
    _setup_folder(fd_dirs["base"], files={"file.txt": "content"})

    provider = CombinedAutocompleteProvider([], fd_dirs["base"], _require_fd_path())
    line = "@file.txt"
    result = await get_suggestions(provider, [line], 0, len(line))

    values = [item["value"] for item in result["items"]]
    assert "@file.txt" in values


@requires_fd
@pytest.mark.tonio
async def test_filters_are_case_insensitive(fd_dirs):
    _setup_folder(fd_dirs["base"], dirs=["src"], files={"README.md": "readme"})

    provider = CombinedAutocompleteProvider([], fd_dirs["base"], _require_fd_path())
    line = "@re"
    result = await get_suggestions(provider, [line], 0, len(line))

    values = sorted(item["value"] for item in result["items"])
    assert values == ["@README.md"]


@requires_fd
@pytest.mark.tonio
async def test_ranks_directories_before_files(fd_dirs):
    _setup_folder(fd_dirs["base"], dirs=["src"], files={"src.txt": "text"})

    provider = CombinedAutocompleteProvider([], fd_dirs["base"], _require_fd_path())
    line = "@src"
    result = await get_suggestions(provider, [line], 0, len(line))

    first_value = result["items"][0]["value"]
    has_src_file = any(item["value"] == "@src.txt" for item in result["items"])
    assert first_value == "@src/"
    assert has_src_file


@requires_fd
@pytest.mark.tonio
async def test_returns_nested_file_paths(fd_dirs):
    _setup_folder(fd_dirs["base"], files={"src/index.ts": "export {};\n"})

    provider = CombinedAutocompleteProvider([], fd_dirs["base"], _require_fd_path())
    line = "@index"
    result = await get_suggestions(provider, [line], 0, len(line))

    values = [item["value"] for item in result["items"]]
    assert "@src/index.ts" in values


@requires_fd
@pytest.mark.tonio
async def test_matches_deeply_nested_paths(fd_dirs):
    _setup_folder(
        fd_dirs["base"],
        files={
            "packages/tui/src/autocomplete.ts": "export {};",
            "packages/ai/src/autocomplete.ts": "export {};",
        },
    )

    provider = CombinedAutocompleteProvider([], fd_dirs["base"], _require_fd_path())
    line = "@tui/src/auto"
    result = await get_suggestions(provider, [line], 0, len(line))

    values = [item["value"] for item in result["items"]]
    assert "@packages/tui/src/autocomplete.ts" in values
    assert "@packages/ai/src/autocomplete.ts" not in values


@requires_fd
@pytest.mark.tonio
async def test_matches_directory_in_middle_of_path_with_full_path(fd_dirs):
    _setup_folder(
        fd_dirs["base"],
        files={
            "src/components/Button.tsx": "export {};",
            "src/utils/helpers.ts": "export {};",
        },
    )

    provider = CombinedAutocompleteProvider([], fd_dirs["base"], _require_fd_path())
    line = "@components/"
    result = await get_suggestions(provider, [line], 0, len(line))

    values = [item["value"] for item in result["items"]]
    assert "@src/components/Button.tsx" in values
    assert "@src/utils/helpers.ts" not in values


@requires_fd
@pytest.mark.tonio
async def test_scopes_fuzzy_search_to_relative_directories_and_searches_recursively(fd_dirs):
    _setup_folder(
        fd_dirs["outside"],
        files={
            "nested/alpha.ts": "export {};",
            "nested/deeper/also-alpha.ts": "export {};",
            "nested/deeper/zzz.ts": "export {};",
        },
    )

    provider = CombinedAutocompleteProvider([], fd_dirs["base"], _require_fd_path())
    line = "@../outside/a"
    result = await get_suggestions(provider, [line], 0, len(line))

    values = [item["value"] for item in result["items"]]
    assert "@../outside/nested/alpha.ts" in values
    assert "@../outside/nested/deeper/also-alpha.ts" in values
    assert "@../outside/nested/deeper/zzz.ts" not in values


@requires_fd
@pytest.mark.tonio
async def test_quotes_paths_with_spaces_for_at_suggestions(fd_dirs):
    _setup_folder(fd_dirs["base"], dirs=["my folder"], files={"my folder/test.txt": "content"})

    provider = CombinedAutocompleteProvider([], fd_dirs["base"], _require_fd_path())
    line = "@my"
    result = await get_suggestions(provider, [line], 0, len(line))

    values = [item["value"] for item in result["items"]]
    assert '@"my folder/"' in values


@requires_fd
@pytest.mark.tonio
async def test_includes_hidden_paths_but_excludes_dot_git(fd_dirs):
    _setup_folder(
        fd_dirs["base"],
        dirs=[".pi", ".github", ".git"],
        files={
            ".pi/config.json": "{}",
            ".github/workflows/ci.yml": "name: ci",
            ".git/config": "[core]",
        },
    )

    provider = CombinedAutocompleteProvider([], fd_dirs["base"], _require_fd_path())
    line = "@"
    result = await get_suggestions(provider, [line], 0, len(line))

    values = [item["value"] for item in result["items"]] if result else []
    assert "@.pi/" in values
    assert "@.github/" in values
    assert not any(value == "@.git" or value.startswith("@.git/") for value in values)


@requires_fd
@pytest.mark.tonio
async def test_follows_symlinked_directories_for_fuzzy_at_search(fd_dirs):
    _setup_folder(fd_dirs["base"], files={"dir/some_file.txt": "real"})
    _setup_folder(fd_dirs["outside"], files={"some_file.txt": "symlinked"})
    os.symlink("../outside", os.path.join(fd_dirs["base"], "symlinked_dir"))

    provider = CombinedAutocompleteProvider([], fd_dirs["base"], _require_fd_path())
    line = "@some"
    result = await get_suggestions(provider, [line], 0, len(line))

    values = [item["value"] for item in result["items"]] if result else []
    assert "@dir/some_file.txt" in values
    assert "@symlinked_dir/some_file.txt" in values


@requires_fd
@pytest.mark.tonio
async def test_returns_symlinked_directories_when_matching_their_name(fd_dirs):
    _setup_folder(fd_dirs["outside"], files={"nested/file.txt": "symlinked"})
    os.symlink("../outside", os.path.join(fd_dirs["base"], "symlinked_dir"))

    provider = CombinedAutocompleteProvider([], fd_dirs["base"], _require_fd_path())
    line = "@symlinked"
    result = await get_suggestions(provider, [line], 0, len(line))

    values = [item["value"] for item in result["items"]] if result else []
    assert "@symlinked_dir/" in values


@requires_fd
@pytest.mark.tonio
async def test_returns_symlinked_files_without_requiring_type_l(fd_dirs):
    _setup_folder(fd_dirs["base"], files={"original.txt": "content"})
    link_path = os.path.join(fd_dirs["base"], "link.txt")
    os.symlink("original.txt", link_path)

    provider = CombinedAutocompleteProvider([], fd_dirs["base"], _require_fd_path())
    line = "@link"
    result = await get_suggestions(provider, [line], 0, len(line))

    values = [item["value"] for item in result["items"]] if result else []
    assert "@link.txt" in values


@requires_fd
@pytest.mark.tonio
async def test_returns_the_same_at_suggestions_when_the_cwd_path_contains_the_query(fd_dirs):
    normal_base_dir = os.path.join(fd_dirs["root"], "cwd-normal")
    query_in_path_base_dir = os.path.join(fd_dirs["root"], "cwd-plan-repro")
    os.makedirs(normal_base_dir, exist_ok=True)
    os.makedirs(query_in_path_base_dir, exist_ok=True)

    structure = {
        "dirs": ["packages/coding-agent/examples/extensions/plan-mode"],
        "files": {
            "packages/coding-agent/examples/extensions/plan-mode/README.md": "readme",
            "packages/tui/docs/plan.md": "plan",
        },
    }
    _setup_folder(normal_base_dir, **structure)
    _setup_folder(query_in_path_base_dir, **structure)

    query = "@plan"
    normal_provider = CombinedAutocompleteProvider([], normal_base_dir, _require_fd_path())
    query_in_path_provider = CombinedAutocompleteProvider([], query_in_path_base_dir, _require_fd_path())

    normal_result = await get_suggestions(normal_provider, [query], 0, len(query))
    query_in_path_result = await get_suggestions(query_in_path_provider, [query], 0, len(query))

    def normalize(result):
        return sorted(
            f"{item['label']} :: {item.get('description', '')}" for item in (result["items"] if result else [])
        )

    assert normalize(query_in_path_result) == normalize(normal_result)
    assert "plan-mode/ :: packages/coding-agent/examples/extensions/plan-mode" in normalize(normal_result)
    assert "plan.md :: packages/tui/docs/plan.md" in normalize(normal_result)


@requires_fd
@pytest.mark.tonio
async def test_continues_autocomplete_inside_quoted_at_paths(fd_dirs):
    _setup_folder(
        fd_dirs["base"],
        files={
            "my folder/test.txt": "content",
            "my folder/other.txt": "content",
        },
    )

    provider = CombinedAutocompleteProvider([], fd_dirs["base"], _require_fd_path())
    line = '@"my folder/"'
    result = await get_suggestions(provider, [line], 0, len(line) - 1)

    assert result is not None, "Should return suggestions for quoted folder path"
    values = [item["value"] for item in result["items"]]
    assert '@"my folder/test.txt"' in values
    assert '@"my folder/other.txt"' in values


@requires_fd
@pytest.mark.tonio
async def test_applies_quoted_at_completion_without_duplicating_closing_quote(fd_dirs):
    _setup_folder(fd_dirs["base"], files={"my folder/test.txt": "content"})

    provider = CombinedAutocompleteProvider([], fd_dirs["base"], _require_fd_path())
    line = '@"my folder/te"'
    cursor_col = len(line) - 1
    result = await get_suggestions(provider, [line], 0, cursor_col)

    assert result is not None, "Should return suggestions for quoted @ path"
    item = next((entry for entry in result["items"] if entry["value"] == '@"my folder/test.txt"'), None)
    assert item, "Should find test.txt suggestion"

    applied = provider.apply_completion([line], 0, cursor_col, item, result["prefix"])
    assert applied["lines"][0] == '@"my folder/test.txt" '


# dot-slash path completion


@pytest.mark.tonio
async def test_preserves_dot_slash_prefix_when_completing_paths(tmp_path):
    _setup_folder(str(tmp_path), files={"update.sh": "#!/bin/bash", "utils.ts": "export {};"})

    provider = CombinedAutocompleteProvider([], str(tmp_path))
    line = "./up"
    result = await get_suggestions(provider, [line], 0, len(line), True)

    assert result is not None, "Should return suggestions for ./ path"
    values = [item["value"] for item in result["items"]]
    assert "./update.sh" in values, f"Expected ./update.sh in {values}"


@pytest.mark.tonio
async def test_preserves_dot_slash_prefix_for_directory_completions(tmp_path):
    _setup_folder(str(tmp_path), dirs=["src"], files={"src/index.ts": "export {};"})

    provider = CombinedAutocompleteProvider([], str(tmp_path))
    line = "./sr"
    result = await get_suggestions(provider, [line], 0, len(line), True)

    assert result is not None, "Should return suggestions for ./ directory path"
    values = [item["value"] for item in result["items"]]
    assert "./src/" in values, f"Expected ./src/ in {values}"


# quoted path completion


@pytest.mark.tonio
async def test_quotes_paths_with_spaces_for_direct_completion(tmp_path):
    _setup_folder(str(tmp_path), dirs=["my folder"], files={"my folder/test.txt": "content"})

    provider = CombinedAutocompleteProvider([], str(tmp_path))
    line = "my"
    result = await get_suggestions(provider, [line], 0, len(line), True)

    assert result is not None, "Should return suggestions for path completion"
    values = [item["value"] for item in result["items"]]
    assert '"my folder/"' in values


@pytest.mark.tonio
async def test_continues_completion_inside_quoted_paths(tmp_path):
    _setup_folder(
        str(tmp_path),
        files={
            "my folder/test.txt": "content",
            "my folder/other.txt": "content",
        },
    )

    provider = CombinedAutocompleteProvider([], str(tmp_path))
    line = '"my folder/"'
    result = await get_suggestions(provider, [line], 0, len(line) - 1, True)

    assert result is not None, "Should return suggestions for quoted folder path"
    values = [item["value"] for item in result["items"]]
    assert '"my folder/test.txt"' in values
    assert '"my folder/other.txt"' in values


@pytest.mark.tonio
async def test_applies_quoted_completion_without_duplicating_closing_quote(tmp_path):
    _setup_folder(str(tmp_path), files={"my folder/test.txt": "content"})

    provider = CombinedAutocompleteProvider([], str(tmp_path))
    line = '"my folder/te"'
    cursor_col = len(line) - 1
    result = await get_suggestions(provider, [line], 0, cursor_col, True)

    assert result is not None, "Should return suggestions for quoted path"
    item = next((entry for entry in result["items"] if entry["value"] == '"my folder/test.txt"'), None)
    assert item, "Should find test.txt suggestion"

    applied = provider.apply_completion([line], 0, cursor_col, item, result["prefix"])
    assert applied["lines"][0] == '"my folder/test.txt"'
