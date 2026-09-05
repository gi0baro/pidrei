"""Mirror of pi's suite/regressions/6104-find-root-relativization.test.ts.

pi parametrizes the helper over `posix` and `win32` path modules; pidrei is
POSIX-only (see `find.py`), so only the POSIX half is mirrored — the Windows
drive-root cases have no code path here.
"""

import pytest

from pidrei.core.tools.find import create_find_tool_definition, relativize_find_result_path


class TestPosixRoot:
    def test_preserves_the_first_segment_and_one_trailing_slash_for_directories_under_root(self):
        assert relativize_find_result_path("/home/user/project/", "/") == "home/user/project/"

    def test_preserves_backslashes_in_posix_filenames(self):
        assert relativize_find_result_path("/home/user/file\\", "/home/user") == "file\\"


class TestAbsoluteResultsOutsideTheSearchPath:
    def test_falls_back_to_relpath_when_the_absolute_paths_do_not_share_a_prefix(self):
        assert relativize_find_result_path("/tmp/results/file.txt", "/workspace/project") == (
            "../../tmp/results/file.txt"
        )

    def test_keeps_a_trailing_slash_on_directories_resolved_through_relpath(self):
        assert relativize_find_result_path("/tmp/results/dir/", "/workspace/project") == "../../tmp/results/dir/"

    def test_does_not_relativize_a_sibling_directory_that_shares_a_name_prefix(self):
        assert relativize_find_result_path("/ai/Models2/file.txt", "/ai/Models") == "../Models2/file.txt"

    def test_normalizes_relative_custom_glob_results_without_corrupting_them(self):
        assert relativize_find_result_path("ai/models/textgen/gemma4/", "/") == "ai/models/textgen/gemma4/"


@pytest.mark.tonio
async def test_relativizes_custom_glob_results_against_a_root_search_path():
    class _Operations:
        async def exists(self, _path: str) -> bool:
            return True

        async def glob(self, _pattern, _path, **_options):
            return ["/home/user/project/", "/home/user/project/file.txt"]

    definition = create_find_tool_definition("/", operations=_Operations())

    # pi passes `{}` as a minimal ctx stub; since 62835ea8 the tool reads `ctx?.cwd`,
    # which is `undefined` on that stub. The Python "no cwd override" spelling is None.
    result = await definition.execute("call-1", {"pattern": "**"}, None, None, None)

    assert result.content[0].text == "home/user/project/\nhome/user/project/file.txt"
