"""Mirror of pi's suite/regressions/12066-find-ancestor-glob.test.ts.

Regression test for https://github.com/earendil-works/pi/issues/12066

fd matches --full-path globs against the absolute path, and `find` prefixed path patterns with
`**/`. A pattern could therefore match through folders above the search path: with the project
in `.../src/app`, `src/**/*.ts` returned every `.ts` file in the project. Path patterns are now
anchored at the search path, like grep's glob.

The glob characters case uses pi's POSIX folder name; the symlink case is a plain symlink (pi's
junction is its Windows spelling).
"""

import os
import shutil
from pathlib import Path

import pytest

from pidrei.core.tools.find import create_find_tool_definition


pytestmark = pytest.mark.skipif(shutil.which("fd") is None, reason="fd not installed")


@pytest.fixture
def temp_root(tmp_path) -> Path:
    # macOS temp paths go through a symlink; pi resolves its temp root too.
    return Path(os.path.realpath(tmp_path))


def _create_project(root: Path) -> None:
    for directory in ("src", "lib", "test", "pkg/src"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "src" / "a.ts").write_text("")
    (root / "lib" / "b.ts").write_text("")
    (root / "test" / "c.ts").write_text("")
    (root / "pkg" / "src" / "d.ts").write_text("")


async def _run_find(search_path: Path, pattern: str) -> list[str]:
    definition = create_find_tool_definition(str(search_path))
    result = await definition.execute("call-1", {"pattern": pattern}, None, None, None)
    text = result.content[0].text if result.content and result.content[0].type == "text" else ""
    if text == "No files found matching pattern":
        return []
    return sorted(text.split("\n"))


@pytest.mark.tonio
async def test_does_not_match_through_an_ancestor_folder_with_the_same_name(temp_root):
    project = temp_root / "src" / "app"
    _create_project(project)

    assert await _run_find(project, "src/**/*.ts") == ["src/a.ts"]
    assert await _run_find(project, "app/**/*.ts") == []


@pytest.mark.tonio
async def test_anchors_path_patterns_at_the_search_path_and_a_leading_globstar_matches_at_any_depth(temp_root):
    project = temp_root / "src" / "app"
    _create_project(project)

    assert await _run_find(project, "src/*.ts") == ["src/a.ts"]
    assert await _run_find(project, "./src/*.ts") == ["src/a.ts"]
    assert await _run_find(project, "pkg/src/*.ts") == ["pkg/src/d.ts"]
    assert await _run_find(project, "**/src/*.ts") == ["pkg/src/d.ts", "src/a.ts"]
    assert await _run_find(project, "**/src/**/*.ts") == ["pkg/src/d.ts", "src/a.ts"]
    # Basename and absolute patterns are unchanged.
    assert await _run_find(project, "*.ts") == ["lib/b.ts", "pkg/src/d.ts", "src/a.ts", "test/c.ts"]
    assert await _run_find(project, f"{project}/lib/*.ts") == ["lib/b.ts"]


@pytest.mark.tonio
async def test_treats_glob_characters_in_the_search_path_literally(temp_root):
    project = temp_root / "w*?[1]{x,y}" / "src"
    _create_project(project)

    assert await _run_find(project, "src/*.ts") == ["src/a.ts"]


@pytest.mark.tonio
async def test_keeps_smart_case_when_the_search_path_has_upper_case_letters(temp_root):
    project = temp_root / "Upper"
    (project / "src").mkdir(parents=True)
    (project / "src" / "A.ts").write_text("")
    (project / "src" / "b.ts").write_text("")

    assert await _run_find(project, "src/a.ts") == ["src/A.ts"]
    assert await _run_find(project, "src/B.ts") == []


@pytest.mark.tonio
async def test_works_through_a_symlinked_search_path(temp_root):
    project = temp_root / "src" / "app"
    _create_project(project)
    link = temp_root / "src" / "link"
    link.symlink_to(project, target_is_directory=True)

    assert await _run_find(link, "src/**/*.ts") == ["src/a.ts"]
