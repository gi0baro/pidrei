"""Mirror of pi's package-command-paths.test.ts.

Ported except for the self-update half. pi's suite spends 8 of its 26 cases on
npm mechanics that pidrei has no equivalent for — the package name coming back
from the version API, the pnpm metadata hint, reinstalling a renamed npm
package, `--force`, "suggests the configured source when update input omits the
npm prefix". pidrei does not self-update at all (Phase 7 step 6), so those are
replaced by cases pinning the refusal and the reinterpreted update targets.

pi drives everything through `main([...])`; these call `handle_package_command`
directly, which is the same entry point `main` dispatches to and avoids booting
a session per case.
"""

import contextlib
import io
import json
import os
from dataclasses import dataclass

import pytest

from pidrei.cli.package_commands import (
    SELF_UPDATE_HINT,
    handle_config_command,
    handle_package_command,
    parse_package_command,
)


@pytest.fixture
def workspace(request, tmp_path):
    """agent dir + project dir + a local package, with cwd and env restored.

    `realpath`, not the raw temp dir: on macOS `/var` is a symlink to
    `/private/var`, so `os.chdir()` + `os.getcwd()` hands back the resolved path
    while an env-provided agent dir keeps the symlinked one. A relative package
    path stored against that mismatch needs `..` hops across the symlink, and
    `os.path.realpath` resolves symlinks *before* collapsing `..` — which
    doubled the prefix into `/private/private/var/...` and failed only on macOS
    CI. Same precedent as `test_serve_e2e.py`.
    """
    root = os.path.realpath(str(tmp_path))
    agent_dir = os.path.join(root, "agent")
    project_dir = os.path.join(root, "project")
    package_dir = os.path.join(root, "package")
    os.makedirs(os.path.join(package_dir, "extensions"), exist_ok=True)
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(agent_dir, exist_ok=True)
    with open(os.path.join(package_dir, "extensions", "hello.py"), "w", encoding="utf-8") as handle:
        handle.write("def extension(pi):\n    pass\n")

    previous_cwd = os.getcwd()
    previous_agent = os.environ.get("PIDREI_CODING_AGENT_DIR")
    os.chdir(project_dir)
    os.environ["PIDREI_CODING_AGENT_DIR"] = agent_dir

    def restore() -> None:
        os.chdir(previous_cwd)
        if previous_agent is None:
            os.environ.pop("PIDREI_CODING_AGENT_DIR", None)
        else:
            os.environ["PIDREI_CODING_AGENT_DIR"] = previous_agent

    request.addfinalizer(restore)
    return {"root": root, "agent_dir": agent_dir, "project_dir": project_dir, "package_dir": package_dir}


@dataclass(slots=True)
class Captured:
    out: str = ""
    err: str = ""


@contextlib.contextmanager
def capture():
    """Redirect inside the test body (predates tonio 0.9.14, which made
    yield fixtures like `capsys` usable in tonio tests)."""
    result = Captured()
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            yield result
    finally:
        result.out, result.err = out.getvalue(), err.getvalue()


def read_settings(agent_dir: str) -> dict:
    path = os.path.join(agent_dir, "settings.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class TestDispatch:
    def test_non_package_arguments_are_not_claimed(self):
        for args in ([], ["--version"], ["--help"], ["-p", "hello"], ["installer"]):
            assert parse_package_command(args) is None, args

    def test_uninstall_is_an_alias_for_remove(self):
        assert parse_package_command(["uninstall", "./x"]).command == "remove"

    @pytest.mark.tonio
    async def test_config_command_is_only_claimed_by_config(self):
        assert await handle_config_command(["list"]) is False


class TestErrors:
    @pytest.mark.tonio
    async def test_unknown_option_for_a_command(self, workspace):
        with capture() as captured:
            assert await handle_package_command(["list", "-l"]) == 1
        assert 'Unknown option -l for "list".' in captured.err

    @pytest.mark.tonio
    async def test_missing_install_source(self, workspace):
        with capture() as captured:
            assert await handle_package_command(["install"]) == 1
        assert "Missing install source." in captured.err

    @pytest.mark.tonio
    async def test_missing_option_value(self, workspace):
        with capture() as captured:
            assert await handle_package_command(["update", "--extension"]) == 1
        assert "Missing value for --extension." in captured.err

    @pytest.mark.tonio
    async def test_unexpected_argument(self, workspace):
        # The *second* positional is the error; pi accepts a stray first one
        # for every command, so `list one` is silently fine there too.
        with capture() as captured:
            assert await handle_package_command(["list", "one", "two"]) == 1
        assert "Unexpected argument two." in captured.err

    @pytest.mark.tonio
    async def test_conflicting_update_targets(self, workspace):
        with capture() as captured:
            assert await handle_package_command(["update", "--models", "--all"]) == 1
        assert "cannot be combined" in captured.err

    @pytest.mark.tonio
    async def test_shows_install_subcommand_help(self, workspace):
        with capture() as captured:
            assert await handle_package_command(["install", "--help"]) == 0
        assert "pidrei install <source>" in captured.out


class TestInstallRemoveList:
    @pytest.mark.tonio
    async def test_installs_and_lists_a_local_package(self, workspace):
        with capture() as captured:
            assert await handle_package_command(["install", workspace["package_dir"]]) == 0
        assert "Installed" in captured.out

        with capture() as captured:
            assert await handle_package_command(["list"]) == 0
        assert "User packages:" in captured.out

    @pytest.mark.tonio
    async def test_persists_global_relative_paths_relative_to_settings(self, workspace):
        relative = os.path.join(workspace["project_dir"], "packages", "local-package")
        os.makedirs(relative, exist_ok=True)

        assert await handle_package_command(["install", "./packages/local-package"]) == 0

        stored = read_settings(workspace["agent_dir"]).get("packages") or []
        assert len(stored) == 1
        resolved = os.path.realpath(os.path.join(workspace["agent_dir"], stored[0]))
        assert resolved == os.path.realpath(relative)

    @pytest.mark.tonio
    async def test_removes_a_package_given_a_trailing_slash(self, workspace):
        assert await handle_package_command(["install", workspace["package_dir"] + "/"]) == 0
        assert len(read_settings(workspace["agent_dir"]).get("packages") or []) == 1

        assert await handle_package_command(["remove", workspace["package_dir"] + "/"]) == 0
        assert (read_settings(workspace["agent_dir"]).get("packages") or []) == []

    @pytest.mark.tonio
    async def test_removing_an_unknown_package_fails(self, workspace):
        with capture() as captured:
            assert await handle_package_command(["remove", "./nope"]) == 1
        assert "No matching package found" in captured.err

    @pytest.mark.tonio
    async def test_lists_nothing_when_no_packages_are_configured(self, workspace):
        with capture() as captured:
            assert await handle_package_command(["list"]) == 0
        assert "No packages installed." in captured.out

    @pytest.mark.tonio
    async def test_blocks_local_package_changes_when_project_is_untrusted(self, workspace):
        os.makedirs(os.path.join(workspace["project_dir"], ".pidrei"), exist_ok=True)
        with capture() as captured:
            result = await handle_package_command(["install", workspace["package_dir"], "-l", "--no-approve"])
        assert result == 1
        assert "Project is not trusted" in captured.err

    @pytest.mark.tonio
    async def test_allows_local_install_with_approve(self, workspace):
        assert await handle_package_command(["install", workspace["package_dir"], "-l", "--approve"]) == 0
        project_settings = os.path.join(workspace["project_dir"], ".pidrei", "settings.json")
        assert os.path.exists(project_settings)
        with open(project_settings, encoding="utf-8") as handle:
            assert len(json.load(handle).get("packages") or []) == 1

    @pytest.mark.tonio
    async def test_skips_untrusted_project_package_settings_when_listing(self, workspace):
        os.makedirs(os.path.join(workspace["project_dir"], ".pidrei"), exist_ok=True)
        with open(os.path.join(workspace["project_dir"], ".pidrei", "settings.json"), "w", encoding="utf-8") as handle:
            json.dump({"packages": ["./project-only"]}, handle)

        with capture() as captured:
            assert await handle_package_command(["list", "--no-approve"]) == 0
        assert "project-only" not in captured.out


class TestSelfUpdateIsRefused:
    """pidrei has no self-update; a habit carried over from pi gets an answer."""

    @pytest.mark.parametrize("args", [["update", "--self"], ["update", "self"], ["update", "pi"], ["update", "pidrei"]])
    def test_parsed_as_a_self_update_request(self, args):
        assert parse_package_command(args).self_update_requested is True

    @pytest.mark.tonio
    @pytest.mark.parametrize("args", [["update", "--self"], ["update", "self"], ["update", "--force"]])
    async def test_refused_with_the_install_command(self, workspace, args):
        with capture() as captured:
            assert await handle_package_command(args) == 1
        err = captured.err
        assert "does not support self-update" in err
        assert "uv tool install" in err

    def test_the_hint_names_both_install_channels(self):
        assert "uv tool install" in SELF_UPDATE_HINT
        assert "brew upgrade" in SELF_UPDATE_HINT


class TestUpdateTargets:
    """pi's targets, reinterpreted now that `self` is gone."""

    def test_bare_update_means_extensions(self):
        options = parse_package_command(["update"])
        assert options.update_target == "extensions"
        assert options.update_source is None
        assert options.self_update_requested is False

    def test_positional_source_updates_one_package(self):
        assert parse_package_command(["update", "./pkg"]).update_source == "./pkg"

    def test_extension_flag_updates_one_package(self):
        options = parse_package_command(["update", "--extension", "./pkg"])
        assert (options.update_target, options.update_source) == ("extensions", "./pkg")

    def test_models_and_all_are_distinct_targets(self):
        assert parse_package_command(["update", "--models"]).update_target == "models"
        assert parse_package_command(["update", "--all"]).update_target == "all"

    @pytest.mark.tonio
    async def test_update_with_no_packages_reports_success(self, workspace):
        with capture() as captured:
            assert await handle_package_command(["update"]) == 0
        assert "Updated packages" in captured.out

    @pytest.mark.tonio
    async def test_update_of_an_unknown_source_fails_even_offline(self, workspace):
        """Caught a parity bug: the port checked offline *before* validating the
        source, so an unknown source silently succeeded. pi validates first."""
        with capture() as captured:
            assert await handle_package_command(["update", "./nope"]) == 1
        assert "No matching package found" in captured.err
