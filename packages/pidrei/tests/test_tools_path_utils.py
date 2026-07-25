"""Mirror of pi coding-agent test/path-utils.test.ts."""

import os

from pidrei.core.tools.path_utils import expand_path, resolve_read_path, resolve_to_cwd


class TestExpandPath:
    def test_expands_tilde_to_home_directory(self):
        assert "~" not in expand_path("~")

    def test_expands_tilde_path_to_home_directory(self):
        assert "~/" not in expand_path("~/Documents/file.txt")

    def test_keeps_tilde_prefixed_filenames_literal(self):
        assert expand_path("~draft.md") == "~draft.md"
        assert expand_path("@~draft.md") == "~draft.md"

    def test_normalizes_unicode_spaces(self):
        # Non-breaking space (U+00A0) should become regular space
        assert expand_path("file name.txt") == "file name.txt"


class TestResolveToCwd:
    def test_resolves_absolute_paths_as_is(self, tmp_path):
        absolute_path = str(tmp_path / "absolute" / "path" / "file.txt")
        assert resolve_to_cwd(absolute_path, str(tmp_path / "some" / "cwd")) == absolute_path

    def test_resolves_relative_paths_against_cwd(self):
        assert resolve_to_cwd("relative/file.txt", "/some/cwd") == "/some/cwd/relative/file.txt"

    def test_resolves_tilde_prefixed_filenames_against_cwd(self, tmp_path):
        cwd = str(tmp_path / "pidrei-path-utils-cwd")
        assert resolve_to_cwd("~draft.md", cwd) == os.path.join(cwd, "~draft.md")
        assert resolve_to_cwd("@~draft.md", cwd) == os.path.join(cwd, "~draft.md")


class TestResolveReadPath:
    def test_resolves_existing_file_path(self, tmp_path):
        (tmp_path / "test-file.txt").write_text("content")
        assert resolve_read_path("test-file.txt", str(tmp_path)) == str(tmp_path / "test-file.txt")

    def test_handles_nfc_vs_nfd_unicode_normalization(self, tmp_path):
        import re

        nfd_file_name = "fileé.txt"  # e + combining acute accent
        nfc_file_name = "file\u00e9.txt"  # precomposed e-acute

        assert nfd_file_name != nfc_file_name

        (tmp_path / nfd_file_name).write_text("content")

        result = resolve_read_path(nfc_file_name, str(tmp_path))
        assert str(tmp_path) in result
        assert re.search(r"file.+\.txt$", result)

    def test_handles_curly_quotes_vs_straight_quotes(self, tmp_path):
        curly_quote_name = "Capture d\u2019cran.txt"
        straight_quote_name = "Capture d'cran.txt"

        assert curly_quote_name != straight_quote_name

        (tmp_path / curly_quote_name).write_text("content")

        assert resolve_read_path(straight_quote_name, str(tmp_path)) == str(tmp_path / curly_quote_name)

    def test_handles_combined_nfc_and_curly_quote(self, tmp_path):
        nfc_curly_name = "Capture d\u2019\u00e9cran.txt"
        nfc_straight_name = "Capture d'\u00e9cran.txt"

        assert nfc_curly_name != nfc_straight_name

        (tmp_path / nfc_curly_name).write_text("content")

        assert resolve_read_path(nfc_straight_name, str(tmp_path)) == str(tmp_path / nfc_curly_name)

    def test_handles_macos_screenshot_am_pm_variant_with_narrow_no_break_space(self, tmp_path):
        macos_name = "Screenshot 2024-01-01 at 10.00.00 AM.png"
        user_name = "Screenshot 2024-01-01 at 10.00.00 AM.png"

        (tmp_path / macos_name).write_text("content")

        assert resolve_read_path(user_name, str(tmp_path)) == str(tmp_path / macos_name)

    def test_handles_macos_screenshot_lowercase_am_pm_variant(self, tmp_path):
        macos_name = "Screenshot 2024-01-01 at 10.00.00 am.png"
        user_name = "Screenshot 2024-01-01 at 10.00.00 am.png"

        (tmp_path / macos_name).write_text("content")

        assert resolve_read_path(user_name, str(tmp_path)) == str(tmp_path / macos_name)
