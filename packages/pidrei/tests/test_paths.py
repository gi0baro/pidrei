"""Mirror of pi coding-agent test/paths.test.ts (POSIX-only tests)."""

import os
from pathlib import Path

import pytest

from pidrei.utils.paths import (
    canonicalize_path,
    get_cwd_relative_path,
    is_local_path,
    normalize_path,
    resolve_path,
)


HOME = os.path.expanduser("~")


class TestCanonicalizePath:
    def test_returns_the_real_path_for_a_regular_file(self, tmp_path):
        file = tmp_path / "file.txt"
        file.write_text("hello")
        assert canonicalize_path(str(file)) == str(file.resolve(strict=True))

    def test_resolves_symlinks_to_their_targets(self, tmp_path):
        target = tmp_path / "target.txt"
        link = tmp_path / "link.txt"
        target.write_text("hello")
        link.symlink_to(target)
        assert canonicalize_path(str(link)) == str(target.resolve(strict=True))

    def test_resolves_directory_symlinks(self, tmp_path):
        target_dir = tmp_path / "target-dir"
        link_dir = tmp_path / "link-dir"
        target_dir.mkdir()
        link_dir.symlink_to(target_dir, target_is_directory=True)
        assert canonicalize_path(str(link_dir)) == str(target_dir.resolve(strict=True))

    def test_falls_back_to_the_raw_path_when_the_target_does_not_exist(self, tmp_path):
        nonexistent = str(tmp_path / "no-such-file")
        assert canonicalize_path(nonexistent) == nonexistent

    def test_falls_back_to_the_raw_path_for_a_dangling_symlink(self, tmp_path):
        target = tmp_path / "target.txt"
        link = tmp_path / "link.txt"
        # Create a symlink whose target does not exist.
        link.symlink_to(target)
        # Strict resolution would fail, so canonicalize_path returns the link path.
        assert canonicalize_path(str(link)) == str(link)


class TestGetCwdRelativePath:
    def test_keeps_cwd_relative_names_that_start_with_dots(self, tmp_path):
        cwd = str(tmp_path / "pidrei-paths-cwd")
        assert get_cwd_relative_path(os.path.join(cwd, "..config", "AGENTS.md"), cwd) == os.path.join(
            "..config", "AGENTS.md"
        )

    def test_rejects_parent_directory_traversals(self, tmp_path):
        cwd = str(tmp_path / "pidrei-paths-cwd")
        assert get_cwd_relative_path(os.path.join(cwd, "..", "AGENTS.md"), cwd) is None


class TestResolvePath:
    def test_expands_only_home_tilde_shortcuts(self, tmp_path):
        cwd = str(tmp_path / "pidrei-paths-cwd")
        assert normalize_path("~") == HOME
        assert normalize_path("~/file.txt") == os.path.join(HOME, "file.txt")
        assert resolve_path("~draft.md", cwd) == os.path.join(cwd, "~draft.md")
        assert normalize_path("~draft.md") == "~draft.md"

    def test_resolves_relative_paths_against_the_base_directory(self, tmp_path):
        cwd = tmp_path / "pidrei-paths-cwd"
        expected = os.path.join(str(cwd), "subdir", "file.txt")
        assert resolve_path("subdir/file.txt", str(cwd)) == expected
        assert resolve_path("subdir/file.txt", cwd.as_uri()) == expected

    def test_accepts_file_urls(self, tmp_path):
        file_path = tmp_path / "file with spaces.txt"
        assert resolve_path(file_path.as_uri(), str(tmp_path / "base")) == str(file_path)

    def test_throws_for_invalid_file_urls(self):
        with pytest.raises(Exception):
            resolve_path("file:///%E0%A4%A")

    def test_preserves_posix_absolute_paths_with_literal_percent_sequences(self, tmp_path):
        for name in ("report%2026.md", "foo%2Fbar", "malformed%A.md"):
            file_path = str(tmp_path / name)
            assert resolve_path(file_path, str(tmp_path / "base")) == file_path


class TestIsLocalPath:
    def test_returns_true_for_bare_names(self):
        assert is_local_path("my-package") is True

    def test_returns_true_for_relative_paths(self):
        assert is_local_path("./foo") is True

    def test_returns_true_for_file_urls(self):
        assert is_local_path("file:///tmp/foo") is True

    def test_returns_false_for_npm_protocol(self):
        assert is_local_path("npm:package") is False

    def test_returns_false_for_git_protocol(self):
        assert is_local_path("git://repo") is False

    def test_returns_false_for_https_protocol(self):
        assert is_local_path("https://example.com") is False


class TestFileUrlBase:
    def test_file_url_base_dir_with_percent_name(self, tmp_path):
        # Base dirs passed as file URLs go through the same normalization.
        base = tmp_path / "base"
        assert resolve_path("x.txt", base.as_uri()) == str(Path(base, "x.txt"))
