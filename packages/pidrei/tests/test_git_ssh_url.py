"""Mirror of pi's git-ssh-url.test.ts.

`parse_git_url` returns a plain dict here, so pi's `toMatchObject` becomes a
subset check on the returned keys.
"""

import pytest

from pidrei.utils.git import parse_git_url


def assert_matches(result: dict | None, expected: dict) -> None:
    assert result is not None
    for key, value in expected.items():
        assert result.get(key) == value, key


# -- protocol URLs (accepted without a git: prefix) ------------------------------


def test_parses_an_https_url():
    assert_matches(
        parse_git_url("https://github.com/user/repo"),
        {"host": "github.com", "path": "user/repo", "repo": "https://github.com/user/repo"},
    )


def test_parses_an_ssh_url():
    assert_matches(
        parse_git_url("ssh://git@github.com/user/repo"),
        {"host": "github.com", "path": "user/repo", "repo": "ssh://git@github.com/user/repo"},
    )


def test_parses_a_protocol_url_with_a_ref():
    assert_matches(
        parse_git_url("https://github.com/user/repo@v1.0.0"),
        {"host": "github.com", "path": "user/repo", "ref": "v1.0.0", "repo": "https://github.com/user/repo"},
    )


# -- shorthand URLs (accepted only with a git: prefix) ---------------------------


def test_parses_git_at_host_colon_path_with_a_git_prefix():
    assert_matches(
        parse_git_url("git:git@github.com:user/repo"),
        {"host": "github.com", "path": "user/repo", "repo": "git@github.com:user/repo"},
    )


def test_parses_host_slash_path_shorthand_with_a_git_prefix():
    assert_matches(
        parse_git_url("git:github.com/user/repo"),
        {"host": "github.com", "path": "user/repo", "repo": "https://github.com/user/repo"},
    )


def test_parses_shorthand_with_a_ref_and_a_git_prefix():
    assert_matches(
        parse_git_url("git:git@github.com:user/repo@v1.0.0"),
        {"host": "github.com", "path": "user/repo", "ref": "v1.0.0", "repo": "git@github.com:user/repo"},
    )


# -- rejections ------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "git:git@evil.example:../../victim/repo",
        "https://evil.example/..%2F..%2Fvictim/repo",
        "https://evil.example/..%2F..%2Fvictim/repo%",
        "git:git@evil.example:/absolute/repo",
        "git:git@evil.example:user\\repo/name",
        "git:git@evil.example:user/repo\0name",
    ],
)
def test_rejects_unsafe_git_install_path_inputs(source):
    assert parse_git_url(source) is None


def test_rejects_git_at_host_colon_path_without_a_git_prefix():
    assert parse_git_url("git@github.com:user/repo") is None


def test_rejects_host_slash_path_shorthand_without_a_git_prefix():
    assert parse_git_url("github.com/user/repo") is None


def test_rejects_user_slash_repo_shorthand():
    assert parse_git_url("user/repo") is None
