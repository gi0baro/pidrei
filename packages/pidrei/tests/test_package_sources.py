"""Mirror of pi's package-manager-ssh.test.ts plus the git and source-parsing
cases from package-manager.test.ts.

pi's `npm:` cases become one case here: the source is refused by name. See
`package_manager.py`'s docstring for why package sources are git and local
only.

Nothing here reaches the network. Where pi lets a clone of a nonexistent repo
fail for real, this stubs `_run_command` — the assertion is about which
commands the manager decides to run, which is exactly what the stub records.
"""

import os
import shutil
import tempfile

import pytest

from pidrei.core.package_manager import (
    DefaultPackageManager,
    GitSource,
    LocalSource,
    UnsupportedSourceError,
    get_extension_temp_folder,
)
from pidrei.core.settings_manager import SettingsManager


class _Dirs:
    def __init__(self) -> None:
        self.previous_offline = os.environ.pop("PIDREI_OFFLINE", None)
        self.root = tempfile.mkdtemp(prefix="pidrei-pkg-src-")
        self.agent_dir = os.path.join(self.root, "agent")
        os.makedirs(self.agent_dir)
        self.settings = SettingsManager.in_memory()
        self.manager = DefaultPackageManager(cwd=self.root, agent_dir=self.agent_dir, settings_manager=self.settings)
        self.commands: list[tuple[str, list[str], str | None]] = []

    def stub_run_command(self, *, fail: bool = False) -> None:
        async def run_command(command: str, args: list[str], *, cwd: str | None = None) -> str:
            self.commands.append((command, args, cwd))
            if fail:
                raise Exception("git failed")
            if args and args[0] == "clone":
                os.makedirs(args[-1], exist_ok=True)
            return ""

        self.manager._run_command = run_command

    def cleanup(self) -> None:
        if self.previous_offline is None:
            os.environ.pop("PIDREI_OFFLINE", None)
        else:
            os.environ["PIDREI_OFFLINE"] = self.previous_offline
        shutil.rmtree(self.root, ignore_errors=True)


@pytest.fixture
def dirs(request):
    holder = _Dirs()
    request.addfinalizer(holder.cleanup)
    return holder


# -- protocol URLs (accepted without a git: prefix) ------------------------------


def test_parses_an_https_url(dirs):
    parsed = dirs.manager.parse_source("https://github.com/user/repo")

    assert isinstance(parsed, GitSource)
    assert parsed.host == "github.com"
    assert parsed.path == "user/repo"


def test_parses_an_ssh_url(dirs):
    parsed = dirs.manager.parse_source("ssh://git@github.com/user/repo")

    assert isinstance(parsed, GitSource)
    assert parsed.host == "github.com"
    assert parsed.path == "user/repo"
    assert parsed.repo == "ssh://git@github.com/user/repo"


# -- shorthand URLs (only with a git: prefix) ------------------------------------


def test_parses_the_git_at_host_colon_path_format(dirs):
    parsed = dirs.manager.parse_source("git:git@github.com:user/repo")

    assert isinstance(parsed, GitSource)
    assert parsed.host == "github.com"
    assert parsed.path == "user/repo"
    assert parsed.repo == "git@github.com:user/repo"
    assert parsed.pinned is False


def test_parses_host_slash_path_shorthand(dirs):
    parsed = dirs.manager.parse_source("git:github.com/user/repo")

    assert isinstance(parsed, GitSource)
    assert parsed.host == "github.com"
    assert parsed.path == "user/repo"


def test_parses_shorthand_with_a_ref(dirs):
    parsed = dirs.manager.parse_source("git:git@github.com:user/repo@v1.0.0")

    assert isinstance(parsed, GitSource)
    assert parsed.ref == "v1.0.0"
    assert parsed.pinned is True


# -- unsupported without a git: prefix -------------------------------------------


def test_treats_git_at_host_colon_path_as_local_without_a_prefix(dirs):
    assert isinstance(dirs.manager.parse_source("git@github.com:user/repo"), LocalSource)


def test_treats_host_slash_path_shorthand_as_local_without_a_prefix(dirs):
    assert isinstance(dirs.manager.parse_source("github.com/user/repo"), LocalSource)


def test_normalizes_protocol_and_shorthand_urls_to_one_identity(dirs):
    prefixed = dirs.manager._get_package_identity("git:git@github.com:user/repo")
    https = dirs.manager._get_package_identity("https://github.com/user/repo")
    ssh = dirs.manager._get_package_identity("ssh://git@github.com/user/repo")

    assert prefixed == "git:github.com/user/repo"
    assert prefixed == https == ssh


# -- source parsing --------------------------------------------------------------


def test_parses_the_source_types_from_the_docs_examples(dirs):
    for source in (
        "git:github.com/user/repo@v1",
        "https://github.com/user/repo@v1",
        "git:git@github.com:user/repo@v1",
        "ssh://git@github.com/user/repo@v1",
    ):
        assert isinstance(dirs.manager.parse_source(source), GitSource), source

    for source in ("/absolute/path/to/package", "./relative/path/to/package", "../relative/path/to/package"):
        assert isinstance(dirs.manager.parse_source(source), LocalSource), source


def test_never_parses_dot_relative_paths_as_git(dirs):
    for source in ("./packages/agent-timers", "../packages/agent-timers"):
        parsed = dirs.manager.parse_source(source)
        assert isinstance(parsed, LocalSource)
        assert parsed.path == source


def test_refuses_npm_sources_by_name(dirs):
    """pi installs these; pidrei is git-and-local only, and says so rather than
    resolving the entry to nothing."""
    with pytest.raises(UnsupportedSourceError) as excinfo:
        dirs.manager.parse_source("npm:@scope/pkg@1.2.3")

    assert "npm package sources are not supported" in str(excinfo.value)
    assert "git source" in str(excinfo.value)


# -- install paths ----------------------------------------------------------------


def test_rejects_paths_outside_the_git_install_roots(dirs):
    traversal = GitSource(repo="git@evil.example:../../victim/repo", host="evil.example", path="../../victim/repo")

    for scope in ("user", "project", "temporary"):
        with pytest.raises(Exception, match="outside package install root"):
            dirs.manager._get_git_install_path(traversal, scope)


def test_places_temporary_git_packages_under_the_agent_temp_folder(dirs):
    source = dirs.manager.parse_source("git:github.com/user/repo")

    install_path = dirs.manager._get_git_install_path(source, "temporary")
    temp_root = os.path.join(dirs.agent_dir, "tmp", "extensions")

    assert not os.path.relpath(install_path, temp_root).startswith("..")
    assert install_path.endswith(os.path.join("user", "repo"))
    assert os.stat(get_extension_temp_folder(dirs.agent_dir)).st_mode & 0o777 == 0o700


def test_user_and_project_scopes_get_separate_git_roots(dirs):
    source = dirs.manager.parse_source("git:github.com/user/repo")

    user_path = dirs.manager._get_git_install_path(source, "user")
    project_path = dirs.manager._get_git_install_path(source, "project")

    assert user_path == os.path.join(dirs.agent_dir, "git", "github.com", "user", "repo")
    assert project_path == os.path.join(dirs.root, ".pidrei", "git", "github.com", "user", "repo")


def test_project_scope_requires_project_trust(dirs):
    dirs.settings.set_project_trusted(False)
    source = dirs.manager.parse_source("git:github.com/user/repo")

    with pytest.raises(Exception, match="Project is not trusted"):
        dirs.manager._get_git_install_path(source, "project")


# -- install / update -------------------------------------------------------------


@pytest.mark.tonio
async def test_installs_a_git_source_by_cloning_into_the_scope_root(dirs):
    dirs.stub_run_command()
    events = []
    dirs.manager.set_progress_callback(events.append)

    await dirs.manager.install("git:github.com/user/repo")

    target = os.path.join(dirs.agent_dir, "git", "github.com", "user", "repo")
    assert dirs.commands == [("git", ["clone", "https://github.com/user/repo", target], None)]
    assert [(event.type, event.action) for event in events] == [("start", "install"), ("complete", "install")]
    # The managed root is marked ignored so a checkout inside a repo stays invisible.
    with open(os.path.join(dirs.agent_dir, "git", ".gitignore"), encoding="utf-8") as handle:
        assert handle.read() == "*\n"


@pytest.mark.tonio
async def test_checks_out_a_pinned_ref_after_cloning(dirs):
    dirs.stub_run_command()

    await dirs.manager.install("git:github.com/user/repo@v1.0.0")

    assert [args[0] for _command, args, _cwd in dirs.commands] == ["clone", "checkout"]
    assert dirs.commands[1][1] == ["checkout", "v1.0.0"]


@pytest.mark.tonio
async def test_reconciles_an_existing_checkout_to_a_pinned_ref_instead_of_recloning(dirs):
    target = os.path.join(dirs.agent_dir, "git", "github.com", "user", "repo")
    os.makedirs(target)
    dirs.stub_run_command()

    await dirs.manager.install("git:github.com/user/repo@v2.0.0")

    assert [args[0] for _command, args, _cwd in dirs.commands] == ["fetch", "checkout"]
    assert dirs.commands[0][1] == ["fetch", "origin", "v2.0.0"]
    assert dirs.commands[1][1] == ["checkout", "--force", "FETCH_HEAD"]


@pytest.mark.tonio
async def test_emits_start_and_error_progress_events_when_a_clone_fails(dirs):
    dirs.stub_run_command(fail=True)
    events = []
    dirs.manager.set_progress_callback(events.append)

    with pytest.raises(Exception, match="git failed"):
        await dirs.manager.install("https://github.com/nonexistent/repo")

    assert [(event.type, event.action) for event in events] == [("start", "install"), ("error", "install")]


@pytest.mark.tonio
async def test_a_local_source_install_only_checks_that_the_path_exists(dirs):
    dirs.stub_run_command()
    package_dir = os.path.join(dirs.root, "local-pkg")
    os.makedirs(package_dir)

    await dirs.manager.install(package_dir)

    assert dirs.commands == []

    with pytest.raises(Exception, match="Path does not exist"):
        await dirs.manager.install(os.path.join(dirs.root, "missing-pkg"))


@pytest.mark.tonio
async def test_removing_a_git_source_deletes_the_checkout_and_prunes_empty_parents(dirs):
    target = os.path.join(dirs.agent_dir, "git", "github.com", "user", "repo")
    os.makedirs(target)

    await dirs.manager.remove("git:github.com/user/repo")

    assert not os.path.exists(target)
    assert not os.path.exists(os.path.join(dirs.agent_dir, "git", "github.com"))


@pytest.mark.tonio
async def test_update_skips_everything_when_offline(dirs):
    os.environ["PIDREI_OFFLINE"] = "1"
    dirs.stub_run_command()
    dirs.settings.set_packages(["git:github.com/user/repo"])

    await dirs.manager.update()

    assert dirs.commands == []


@pytest.mark.tonio
async def test_update_reconciles_each_configured_git_package(dirs):
    target = os.path.join(dirs.agent_dir, "git", "github.com", "user", "repo")
    os.makedirs(target)
    dirs.stub_run_command()
    dirs.settings.set_packages(["git:github.com/user/repo@v1.0.0"])

    await dirs.manager.update()

    assert [args[0] for _command, args, _cwd in dirs.commands] == ["fetch", "checkout"]


@pytest.mark.tonio
async def test_update_reports_an_unmatched_source(dirs):
    dirs.settings.set_packages(["git:github.com/user/repo"])

    with pytest.raises(Exception, match="No matching package found"):
        await dirs.manager.update("git:github.com/other/repo")


@pytest.mark.tonio
async def test_resolve_skips_installing_missing_sources_when_offline(dirs):
    os.environ["PIDREI_OFFLINE"] = "1"
    dirs.stub_run_command()
    dirs.settings.set_packages(["git:github.com/user/repo"])

    result = await dirs.manager.resolve()

    assert dirs.commands == []
    assert result.extensions == []


# -- settings source normalization -------------------------------------------------


def test_stores_global_local_packages_relative_to_the_agent_settings_base(dirs):
    package_dir = os.path.join(dirs.root, "packages", "local-global-pkg")
    os.makedirs(os.path.join(package_dir, "extensions"))

    assert dirs.manager.add_source_to_settings("./packages/local-global-pkg") is True

    expected = os.path.relpath(package_dir, dirs.agent_dir)
    assert dirs.settings.get_global_settings()["packages"] == [expected]


def test_stores_project_local_packages_relative_to_the_project_settings_base(dirs):
    package_dir = os.path.join(dirs.root, "project-local-pkg")
    os.makedirs(os.path.join(package_dir, "extensions"))

    assert dirs.manager.add_source_to_settings("./project-local-pkg", local=True) is True

    expected = os.path.relpath(package_dir, os.path.join(dirs.root, ".pidrei"))
    assert dirs.settings.get_project_settings()["packages"] == [expected]


def test_removes_local_package_entries_using_equivalent_path_forms(dirs):
    package_dir = os.path.join(dirs.root, "remove-local-pkg")
    os.makedirs(os.path.join(package_dir, "extensions"))
    dirs.manager.add_source_to_settings("./remove-local-pkg")

    assert dirs.manager.remove_source_from_settings(f"{package_dir}/") is True
    assert dirs.settings.get_global_settings().get("packages") == []


def test_returns_false_when_adding_the_same_git_source_with_the_same_ref(dirs):
    assert dirs.manager.add_source_to_settings("git:github.com/user/repo@v1") is True
    assert dirs.manager.add_source_to_settings("git:github.com/user/repo@v1") is False
    assert dirs.settings.get_global_settings()["packages"] == ["git:github.com/user/repo@v1"]


def test_updates_the_ref_when_adding_the_same_git_source_with_a_different_ref(dirs):
    dirs.manager.add_source_to_settings("git:github.com/user/repo@v1")

    assert dirs.manager.add_source_to_settings("git:github.com/user/repo@v2") is True
    assert dirs.settings.get_global_settings()["packages"] == ["git:github.com/user/repo@v2"]


def test_preserves_package_filters_when_replacing_a_package_source_ref(dirs):
    dirs.settings.set_packages(
        [
            {
                "source": "git:github.com/user/repo@v1",
                "extensions": ["extensions/main.py"],
                "skills": [],
                "prompts": ["prompts/review.md"],
                "themes": ["themes/dark.json"],
            }
        ]
    )

    assert dirs.manager.add_source_to_settings("git:github.com/user/repo@v2") is True
    assert dirs.settings.get_global_settings()["packages"] == [
        {
            "source": "git:github.com/user/repo@v2",
            "extensions": ["extensions/main.py"],
            "skills": [],
            "prompts": ["prompts/review.md"],
            "themes": ["themes/dark.json"],
        }
    ]


def test_lists_configured_packages_with_their_installed_paths(dirs):
    installed = os.path.join(dirs.agent_dir, "git", "github.com", "user", "repo")
    os.makedirs(installed)
    dirs.settings.set_packages(["git:github.com/user/repo"])
    dirs.settings.set_project_packages(["git:github.com/other/repo"])

    configured = dirs.manager.list_configured_packages()

    assert [(package.source, package.scope) for package in configured] == [
        ("git:github.com/user/repo", "user"),
        ("git:github.com/other/repo", "project"),
    ]
    assert configured[0].installed_path == installed
    assert configured[1].installed_path is None
