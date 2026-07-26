"""Mirror of pi's package-manager.test.ts.

`.ts` extensions become `.py`, `package.json`'s `pi` key becomes
`pyproject.toml`'s `[tool.pidrei]`, and `.pi` becomes `.pidrei`.

**pi's npm cases are not mirrored** — package sources here are git and local
only (see `package_manager.py`'s docstring). That drops pi's `npmCommand`,
pnpm-global, npm-root, npm-version-range, npm-update-check and
legacy-migration cases; `test_package_sources.py` covers the git and
`npm:`-refusal behaviour that replaces them. Everything else in pi's file —
resolution, metadata, `.agents` discovery, ignore files, pattern filtering,
package dedupe and multi-file discovery — is mirrored here.
"""

import json
import os
import shutil
import tempfile

import pytest

from pidrei.core.package_manager import DefaultPackageManager
from pidrei.core.settings_manager import SettingsManager


EXTENSION_SOURCE = "def extension(pi):\n    pass\n"


def write(path: str, content: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return path


def write_skill(path: str, name: str, description: str = "A skill") -> str:
    return write(path, f"---\nname: {name}\ndescription: {description}\n---\nContent")


def write_manifest(package_dir: str, table: dict) -> str:
    lines = ["[tool.pidrei]"]
    for key, values in table.items():
        lines.append(f"{key} = {json.dumps(values)}")
    return write(os.path.join(package_dir, "pyproject.toml"), "\n".join(lines) + "\n")


class _Dirs:
    def __init__(self) -> None:
        self.previous_offline = os.environ.pop("PIDREI_OFFLINE", None)
        self.previous_home = os.environ.get("HOME")
        self.root = tempfile.mkdtemp(prefix="pidrei-pm-test-")
        self.agent_dir = os.path.join(self.root, "agent")
        os.makedirs(self.agent_dir)
        self.settings = SettingsManager.in_memory()
        self.manager = DefaultPackageManager(cwd=self.root, agent_dir=self.agent_dir, settings_manager=self.settings)

    def set_home(self, home: str) -> None:
        os.environ["HOME"] = home

    def cleanup(self) -> None:
        if self.previous_offline is None:
            os.environ.pop("PIDREI_OFFLINE", None)
        else:
            os.environ["PIDREI_OFFLINE"] = self.previous_offline
        if self.previous_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = self.previous_home
        shutil.rmtree(self.root, ignore_errors=True)


@pytest.fixture
def dirs(request):
    holder = _Dirs()
    request.addfinalizer(holder.cleanup)
    return holder


def paths_of(resources) -> list[str]:
    return [resource.path for resource in resources]


def find(resources, path: str):
    return next((resource for resource in resources if resource.path == path), None)


# -- resolve ---------------------------------------------------------------------


@pytest.mark.tonio
async def test_returns_no_package_sourced_paths_when_no_sources_configured(dirs):
    result = await dirs.manager.resolve()

    assert result.extensions == []
    assert result.prompts == []
    assert result.themes == []
    assert all(
        resource.metadata.source == "auto" and resource.metadata.origin == "top-level" for resource in result.skills
    )


@pytest.mark.tonio
async def test_resolves_local_extension_paths_from_settings(dirs):
    extension_path = write(os.path.join(dirs.agent_dir, "extensions", "my-extension.py"), EXTENSION_SOURCE)
    dirs.settings.set_extension_paths(["extensions/my-extension.py"])

    result = await dirs.manager.resolve()

    resource = find(result.extensions, extension_path)
    assert resource is not None and resource.enabled


@pytest.mark.tonio
async def test_resolves_skill_paths_from_settings(dirs):
    skill_file = write_skill(
        os.path.join(dirs.agent_dir, "skills", "my-skill", "SKILL.md"), "test-skill", "A test skill"
    )
    dirs.settings.set_skill_paths(["skills"])

    result = await dirs.manager.resolve()

    resource = find(result.skills, skill_file)
    assert resource is not None and resource.enabled


@pytest.mark.tonio
async def test_auto_discovers_root_markdown_skills_from_config_skill_dirs(dirs):
    skill_file = write_skill(os.path.join(dirs.agent_dir, "skills", "single-file.md"), "single-file")

    result = await dirs.manager.resolve()

    resource = find(result.skills, skill_file)
    assert resource is not None and resource.enabled


@pytest.mark.tonio
async def test_resolves_project_paths_relative_to_the_config_dir(dirs):
    extension_path = write(os.path.join(dirs.root, ".pidrei", "extensions", "project-ext.py"), EXTENSION_SOURCE)
    dirs.settings.set_project_extension_paths(["extensions/project-ext.py"])

    result = await dirs.manager.resolve()

    resource = find(result.extensions, extension_path)
    assert resource is not None and resource.enabled


@pytest.mark.tonio
async def test_auto_discovers_user_prompts_with_overrides(dirs):
    prompt_path = write(os.path.join(dirs.agent_dir, "prompts", "auto.md"), "Auto prompt")
    dirs.settings.set_prompt_template_paths(["!prompts/auto.md"])

    result = await dirs.manager.resolve()

    resource = find(result.prompts, prompt_path)
    assert resource is not None and not resource.enabled


@pytest.mark.tonio
async def test_auto_discovers_project_prompts_with_overrides(dirs):
    prompt_path = write(os.path.join(dirs.root, ".pidrei", "prompts", "is.md"), "Is prompt")
    dirs.settings.set_project_prompt_template_paths(["!prompts/is.md"])

    result = await dirs.manager.resolve()

    resource = find(result.prompts, prompt_path)
    assert resource is not None and not resource.enabled


@pytest.mark.tonio
async def test_resolves_symlinked_user_and_project_resources_once(dirs):
    dirs.set_home(dirs.root)
    shared = os.path.join(dirs.root, "shared-resources")
    write(os.path.join(shared, "extensions", "shared.py"), EXTENSION_SOURCE)
    write_skill(os.path.join(shared, "skills", "shared-skill", "SKILL.md"), "shared-skill")
    write(os.path.join(shared, "prompts", "shared.md"), "Shared prompt")
    write(os.path.join(shared, "themes", "shared.json"), json.dumps({"name": "shared-theme"}))

    os.makedirs(os.path.join(dirs.root, ".pidrei"), exist_ok=True)
    for resource_type in ("extensions", "skills", "prompts", "themes"):
        os.symlink(os.path.join(shared, resource_type), os.path.join(dirs.agent_dir, resource_type))
        os.symlink(os.path.join(shared, resource_type), os.path.join(dirs.root, ".pidrei", resource_type))

    result = await dirs.manager.resolve()

    assert [len(result.extensions), len(result.skills), len(result.prompts), len(result.themes)] == [1, 1, 1, 1]
    # Project auto-discovery outranks user auto-discovery, so the survivor is project-scoped.
    assert result.extensions[0].metadata.scope == "project"
    assert result.skills[0].metadata.scope == "project"
    assert result.prompts[0].metadata.scope == "project"
    assert result.themes[0].metadata.scope == "project"


@pytest.mark.tonio
async def test_resolves_a_directory_with_a_manifest_in_the_extensions_setting(dirs):
    package_dir = os.path.join(dirs.root, "my-extensions-pkg")
    write(os.path.join(package_dir, "extensions", "clip.py"), EXTENSION_SOURCE)
    write(os.path.join(package_dir, "extensions", "cost.py"), EXTENSION_SOURCE)
    # Not in the manifest, so it must not be loaded.
    write(os.path.join(package_dir, "extensions", "helper.py"), "X = 1\n")
    write_manifest(package_dir, {"extensions": ["./extensions/clip.py", "./extensions/cost.py"]})

    dirs.settings.set_extension_paths([package_dir])

    result = await dirs.manager.resolve()

    for name in ("clip.py", "cost.py"):
        resource = find(result.extensions, os.path.join(package_dir, "extensions", name))
        assert resource is not None and resource.enabled
    assert not any(path.endswith("helper.py") for path in paths_of(result.extensions))


# -- auto-discovered skill metadata ----------------------------------------------


@pytest.mark.tonio
async def test_uses_the_agent_dir_as_base_dir_for_user_config_skills(dirs):
    skill_path = write_skill(os.path.join(dirs.agent_dir, "skills", "user-pi", "SKILL.md"), "user-pi")

    result = await dirs.manager.resolve()

    skill = find(result.skills, skill_path)
    assert skill.metadata.source == "auto"
    assert skill.metadata.scope == "user"
    assert skill.metadata.base_dir == dirs.agent_dir


@pytest.mark.tonio
async def test_uses_the_project_config_dir_as_base_dir_for_project_skills(dirs):
    project_base_dir = os.path.join(dirs.root, ".pidrei")
    skill_path = write_skill(os.path.join(project_base_dir, "skills", "project-pi", "SKILL.md"), "project-pi")

    result = await dirs.manager.resolve()

    skill = find(result.skills, skill_path)
    assert skill.metadata.source == "auto"
    assert skill.metadata.scope == "project"
    assert skill.metadata.base_dir == project_base_dir


@pytest.mark.tonio
async def test_uses_home_agents_as_base_dir_for_user_agents_skills(dirs):
    dirs.set_home(dirs.root)
    agents_base_dir = os.path.join(dirs.root, ".agents")
    skill_path = write_skill(os.path.join(agents_base_dir, "skills", "user-agents", "SKILL.md"), "user-agents")

    result = await dirs.manager.resolve()

    skill = find(result.skills, skill_path)
    assert skill.metadata.source == "auto"
    assert skill.metadata.scope == "user"
    assert skill.metadata.base_dir == agents_base_dir


@pytest.mark.tonio
async def test_uses_each_project_agents_dir_as_base_dir(dirs):
    repo_root = os.path.join(dirs.root, "repo")
    nested_cwd = os.path.join(repo_root, "packages", "feature")
    os.makedirs(nested_cwd)
    os.makedirs(os.path.join(repo_root, ".git"))

    repo_agents_base = os.path.join(repo_root, ".agents")
    repo_skill = write_skill(os.path.join(repo_agents_base, "skills", "repo", "SKILL.md"), "repo")
    package_agents_base = os.path.join(repo_root, "packages", ".agents")
    package_skill = write_skill(os.path.join(package_agents_base, "skills", "package", "SKILL.md"), "package")

    manager = DefaultPackageManager(cwd=nested_cwd, agent_dir=dirs.agent_dir, settings_manager=dirs.settings)
    result = await manager.resolve()

    resolved_repo = find(result.skills, repo_skill)
    resolved_package = find(result.skills, package_skill)
    assert (resolved_repo.metadata.source, resolved_repo.metadata.scope) == ("auto", "project")
    assert resolved_repo.metadata.base_dir == repo_agents_base
    assert (resolved_package.metadata.source, resolved_package.metadata.scope) == ("auto", "project")
    assert resolved_package.metadata.base_dir == package_agents_base


# -- .agents/skills auto-discovery -----------------------------------------------


@pytest.mark.tonio
async def test_scans_agents_skills_from_cwd_up_to_the_git_repo_root(dirs):
    repo_root = os.path.join(dirs.root, "repo")
    nested_cwd = os.path.join(repo_root, "packages", "feature")
    os.makedirs(nested_cwd)
    os.makedirs(os.path.join(repo_root, ".git"))

    above_repo_skill = write_skill(os.path.join(dirs.root, ".agents", "skills", "above-repo", "SKILL.md"), "above-repo")
    repo_root_skill = write_skill(os.path.join(repo_root, ".agents", "skills", "repo-root", "SKILL.md"), "repo-root")
    nested_skill = write_skill(os.path.join(repo_root, "packages", ".agents", "skills", "nested", "SKILL.md"), "nested")

    manager = DefaultPackageManager(cwd=nested_cwd, agent_dir=dirs.agent_dir, settings_manager=dirs.settings)
    result = await manager.resolve()

    assert find(result.skills, repo_root_skill).enabled
    assert find(result.skills, nested_skill).enabled
    assert find(result.skills, above_repo_skill) is None


@pytest.mark.tonio
async def test_scans_agents_skills_up_to_the_filesystem_root_outside_a_git_repo(dirs):
    non_repo_root = os.path.join(dirs.root, "non-repo")
    nested_cwd = os.path.join(non_repo_root, "a", "b")
    os.makedirs(nested_cwd)

    root_skill = write_skill(os.path.join(non_repo_root, ".agents", "skills", "root", "SKILL.md"), "root")
    middle_skill = write_skill(os.path.join(non_repo_root, "a", ".agents", "skills", "middle", "SKILL.md"), "middle")

    manager = DefaultPackageManager(cwd=nested_cwd, agent_dir=dirs.agent_dir, settings_manager=dirs.settings)
    result = await manager.resolve()

    assert find(result.skills, root_skill).enabled
    assert find(result.skills, middle_skill).enabled


@pytest.mark.tonio
async def test_ignores_root_markdown_files_in_agents_skills(dirs):
    agents_skills_dir = os.path.join(dirs.root, ".agents", "skills")
    root_skill = write_skill(os.path.join(agents_skills_dir, "root-file.md"), "root-file")
    nested_skill = write_skill(os.path.join(agents_skills_dir, "nested-skill", "SKILL.md"), "nested-skill")
    work_dir = os.path.join(dirs.root, "work")
    os.makedirs(work_dir)

    manager = DefaultPackageManager(cwd=work_dir, agent_dir=dirs.agent_dir, settings_manager=dirs.settings)
    result = await manager.resolve()

    assert find(result.skills, root_skill) is None
    assert find(result.skills, nested_skill).enabled


@pytest.mark.tonio
async def test_keeps_home_agents_skills_user_scoped_when_cwd_is_under_home(dirs):
    dirs.set_home(dirs.root)
    cwd = os.path.join(dirs.root, "scratch", "nested")
    local_agent_dir = os.path.join(dirs.root, ".pidrei", "agent")
    os.makedirs(cwd)
    os.makedirs(local_agent_dir)
    home_skill = write_skill(os.path.join(dirs.root, ".agents", "skills", "home-skill", "SKILL.md"), "home-skill")

    manager = DefaultPackageManager(cwd=cwd, agent_dir=local_agent_dir, settings_manager=SettingsManager.in_memory())
    result = await manager.resolve()

    matching = [resource for resource in result.skills if resource.path == home_skill]
    assert len(matching) == 1
    assert matching[0].enabled
    assert matching[0].metadata.scope == "user"
    assert matching[0].metadata.source == "auto"


@pytest.mark.tonio
async def test_dedupes_user_skills_when_the_agent_skills_dir_is_a_symlink(dirs):
    dirs.set_home(dirs.root)
    agents_skills_dir = os.path.join(dirs.root, ".agents", "skills")
    os.makedirs(agents_skills_dir)
    os.symlink(agents_skills_dir, os.path.join(dirs.agent_dir, "skills"))
    write_skill(os.path.join(agents_skills_dir, "foo", "SKILL.md"), "foo")

    result = await dirs.manager.resolve()

    assert len([path for path in paths_of(result.skills) if path.endswith(os.path.join("foo", "SKILL.md"))]) == 1


# -- ignore files -----------------------------------------------------------------


@pytest.mark.tonio
async def test_respects_gitignore_in_skill_directories(dirs):
    skills_dir = os.path.join(dirs.agent_dir, "skills")
    write(os.path.join(skills_dir, ".gitignore"), "venv\n__pycache__\n")
    write_skill(os.path.join(skills_dir, "good-skill", "SKILL.md"), "good-skill", "Good")
    write_skill(os.path.join(skills_dir, "venv", "bad-skill", "SKILL.md"), "bad-skill", "Bad")
    dirs.settings.set_skill_paths(["skills"])

    result = await dirs.manager.resolve()

    assert any("good-skill" in resource.path and resource.enabled for resource in result.skills)
    assert not any("venv" in resource.path and resource.enabled for resource in result.skills)


@pytest.mark.tonio
async def test_does_not_apply_a_parent_gitignore_to_config_auto_discovery(dirs):
    write(os.path.join(dirs.root, ".gitignore"), ".pidrei\n")
    skill_path = write_skill(os.path.join(dirs.root, ".pidrei", "skills", "auto-skill", "SKILL.md"), "auto-skill")

    result = await dirs.manager.resolve()

    assert find(result.skills, skill_path).enabled


# -- resolve_extension_sources ----------------------------------------------------


@pytest.mark.tonio
async def test_resolves_local_paths(dirs):
    extension_path = write(os.path.join(dirs.root, "ext.py"), EXTENSION_SOURCE)

    result = await dirs.manager.resolve_extension_sources([extension_path])

    assert find(result.extensions, extension_path).enabled


@pytest.mark.tonio
async def test_handles_directories_with_a_manifest(dirs):
    package_dir = os.path.join(dirs.root, "my-package")
    write(os.path.join(package_dir, "src", "__init__.py"), EXTENSION_SOURCE)
    write_skill(os.path.join(package_dir, "skills", "my-skill", "SKILL.md"), "my-skill", "Test")
    write_manifest(package_dir, {"extensions": ["./src/__init__.py"], "skills": ["./skills"]})

    result = await dirs.manager.resolve_extension_sources([package_dir])

    assert find(result.extensions, os.path.join(package_dir, "src", "__init__.py")).enabled
    assert find(result.skills, os.path.join(package_dir, "skills", "my-skill", "SKILL.md")).enabled


@pytest.mark.tonio
async def test_keeps_manifest_entries_with_leading_tilde_package_relative(dirs):
    package_dir = os.path.join(dirs.root, "tilde-manifest-package")
    direct_extension = write(os.path.join(package_dir, "~extensions", "main.py"), EXTENSION_SOURCE)
    slash_extension = write(os.path.join(package_dir, "~", "extensions", "alt.py"), EXTENSION_SOURCE)
    direct_skill = write_skill(os.path.join(package_dir, "~skills", "direct-skill", "SKILL.md"), "direct-skill")
    slash_skill = write_skill(os.path.join(package_dir, "~", "skills", "slash-skill", "SKILL.md"), "slash-skill")
    write_manifest(
        package_dir,
        {"extensions": ["~extensions/main.py", "~/extensions/alt.py"], "skills": ["~skills", "~/skills"]},
    )

    result = await dirs.manager.resolve_extension_sources([package_dir])

    assert find(result.extensions, direct_extension).enabled
    assert find(result.extensions, slash_extension).enabled
    assert find(result.skills, direct_skill).enabled
    assert find(result.skills, slash_skill).enabled


@pytest.mark.tonio
async def test_handles_directories_with_an_auto_discovery_layout(dirs):
    package_dir = os.path.join(dirs.root, "auto-pkg")
    write(os.path.join(package_dir, "extensions", "main.py"), EXTENSION_SOURCE)
    write(os.path.join(package_dir, "themes", "dark.json"), "{}")

    result = await dirs.manager.resolve_extension_sources([package_dir])

    assert any(path.endswith("main.py") for path in paths_of(result.extensions))
    assert any(path.endswith("dark.json") for path in paths_of(result.themes))


@pytest.mark.tonio
async def test_stops_recursing_when_a_package_skill_directory_contains_skill_md(dirs):
    package_dir = os.path.join(dirs.root, "skill-root-pkg")
    root_skill = write_skill(os.path.join(package_dir, "skills", "root-skill", "SKILL.md"), "root-skill")
    nested_skill = write_skill(
        os.path.join(package_dir, "skills", "root-skill", "nested-skill", "SKILL.md"), "nested-skill"
    )

    result = await dirs.manager.resolve_extension_sources([package_dir])

    assert find(result.skills, root_skill).enabled
    assert find(result.skills, nested_skill) is None


@pytest.mark.tonio
async def test_emits_no_progress_events_for_local_sources(dirs):
    events = []
    dirs.manager.set_progress_callback(events.append)
    extension_path = write(os.path.join(dirs.root, "ext.py"), EXTENSION_SOURCE)

    await dirs.manager.resolve_extension_sources([extension_path])

    assert events == []


# -- pattern filtering in top-level arrays ----------------------------------------


def enabled_names(resources) -> set[str]:
    return {os.path.basename(resource.path) for resource in resources if resource.enabled}


def disabled_names(resources) -> set[str]:
    return {os.path.basename(resource.path) for resource in resources if not resource.enabled}


@pytest.mark.tonio
async def test_excludes_extensions_with_a_bang_pattern(dirs):
    extension_dir = os.path.join(dirs.agent_dir, "extensions")
    write(os.path.join(extension_dir, "keep.py"), EXTENSION_SOURCE)
    write(os.path.join(extension_dir, "remove.py"), EXTENSION_SOURCE)
    dirs.settings.set_extension_paths(["extensions", "!**/remove.py"])

    result = await dirs.manager.resolve()

    assert "keep.py" in enabled_names(result.extensions)
    assert "remove.py" in disabled_names(result.extensions)


@pytest.mark.tonio
async def test_filters_themes_with_glob_patterns(dirs):
    themes_dir = os.path.join(dirs.agent_dir, "themes")
    for name in ("dark.json", "light.json", "funky.json"):
        write(os.path.join(themes_dir, name), "{}")
    dirs.settings.set_theme_paths(["themes", "!funky.json"])

    result = await dirs.manager.resolve()

    assert {"dark.json", "light.json"} <= enabled_names(result.themes)
    assert "funky.json" in disabled_names(result.themes)


@pytest.mark.tonio
async def test_filters_prompts_with_an_exclusion_pattern(dirs):
    prompts_dir = os.path.join(dirs.agent_dir, "prompts")
    write(os.path.join(prompts_dir, "review.md"), "Review code")
    write(os.path.join(prompts_dir, "explain.md"), "Explain code")
    dirs.settings.set_prompt_template_paths(["prompts", "!explain.md"])

    result = await dirs.manager.resolve()

    assert "review.md" in enabled_names(result.prompts)
    assert "explain.md" in disabled_names(result.prompts)


@pytest.mark.tonio
async def test_filters_skills_with_an_exclusion_pattern(dirs):
    skills_dir = os.path.join(dirs.agent_dir, "skills")
    write_skill(os.path.join(skills_dir, "good-skill", "SKILL.md"), "good-skill", "Good")
    write_skill(os.path.join(skills_dir, "bad-skill", "SKILL.md"), "bad-skill", "Bad")
    dirs.settings.set_skill_paths(["skills", "!**/bad-skill"])

    result = await dirs.manager.resolve()

    assert any("good-skill" in r.path and r.enabled for r in result.skills)
    assert any("bad-skill" in r.path and not r.enabled for r in result.skills)


@pytest.mark.tonio
async def test_works_without_patterns(dirs):
    extension_path = write(os.path.join(dirs.agent_dir, "extensions", "my-ext.py"), EXTENSION_SOURCE)
    dirs.settings.set_extension_paths(["extensions/my-ext.py"])

    result = await dirs.manager.resolve()

    assert find(result.extensions, extension_path).enabled


# -- pattern filtering in the manifest ---------------------------------------------


@pytest.mark.tonio
async def test_supports_glob_patterns_in_manifest_extensions(dirs):
    package_dir = os.path.join(dirs.root, "manifest-pkg")
    write(os.path.join(package_dir, "extensions", "local.py"), EXTENSION_SOURCE)
    write(os.path.join(package_dir, "vendor", "dep", "extensions", "remote.py"), EXTENSION_SOURCE)
    write(os.path.join(package_dir, "vendor", "dep", "extensions", "skip.py"), EXTENSION_SOURCE)
    write_manifest(package_dir, {"extensions": ["extensions", "vendor/dep/extensions", "!**/skip.py"]})

    result = await dirs.manager.resolve_extension_sources([package_dir])

    assert {"local.py", "remote.py"} <= enabled_names(result.extensions)
    assert not any(path.endswith("skip.py") for path in paths_of(result.extensions))


@pytest.mark.tonio
async def test_supports_glob_patterns_in_manifest_skills(dirs):
    package_dir = os.path.join(dirs.root, "skill-manifest-pkg")
    write_skill(os.path.join(package_dir, "skills", "good-skill", "SKILL.md"), "good-skill", "Good")
    write_skill(os.path.join(package_dir, "skills", "bad-skill", "SKILL.md"), "bad-skill", "Bad")
    write_manifest(package_dir, {"skills": ["skills", "!**/bad-skill"]})

    result = await dirs.manager.resolve_extension_sources([package_dir])

    assert any("good-skill" in r.path and r.enabled for r in result.skills)
    assert not any("bad-skill" in path for path in paths_of(result.skills))


@pytest.mark.tonio
async def test_expands_positive_glob_manifest_entries_before_collecting_skills(dirs):
    package_dir = os.path.join(dirs.root, "skill-manifest-glob-pkg")
    write_skill(
        os.path.join(package_dir, "plugins", "pdf-to-markdown", "skills", "pdf-to-markdown", "SKILL.md"),
        "pdf-to-markdown",
    )
    write_skill(
        os.path.join(package_dir, "plugins", "nutrient-dws", "skills", "document-processor-api", "SKILL.md"),
        "document-processor-api",
    )
    write_manifest(package_dir, {"skills": ["./plugins/*/skills"]})

    result = await dirs.manager.resolve_extension_sources([package_dir])

    assert any("pdf-to-markdown" in r.path and r.enabled for r in result.skills)
    assert any("document-processor-api" in r.path and r.enabled for r in result.skills)


# -- pattern filtering in package filters ------------------------------------------


@pytest.mark.tonio
async def test_applies_user_filters_on_top_of_manifest_filters(dirs):
    package_dir = os.path.join(dirs.root, "layered-pkg")
    for name in ("foo.py", "bar.py", "baz.py"):
        write(os.path.join(package_dir, "extensions", name), EXTENSION_SOURCE)
    write_manifest(package_dir, {"extensions": ["extensions", "!**/baz.py"]})

    dirs.settings.set_packages(
        [{"source": package_dir, "extensions": ["!**/bar.py"], "skills": [], "prompts": [], "themes": []}]
    )

    result = await dirs.manager.resolve()

    assert "foo.py" in enabled_names(result.extensions)
    assert "bar.py" in disabled_names(result.extensions)
    assert not any(path.endswith("baz.py") for path in paths_of(result.extensions))


@pytest.mark.tonio
async def test_excludes_extensions_from_a_package_with_a_bang_pattern(dirs):
    package_dir = os.path.join(dirs.root, "pattern-pkg")
    for name in ("foo.py", "bar.py", "baz.py"):
        write(os.path.join(package_dir, "extensions", name), EXTENSION_SOURCE)
    dirs.settings.set_packages(
        [{"source": package_dir, "extensions": ["!**/baz.py"], "skills": [], "prompts": [], "themes": []}]
    )

    result = await dirs.manager.resolve()

    assert {"foo.py", "bar.py"} <= enabled_names(result.extensions)
    assert "baz.py" in disabled_names(result.extensions)


@pytest.mark.tonio
async def test_filters_themes_from_a_package(dirs):
    package_dir = os.path.join(dirs.root, "theme-pkg")
    write(os.path.join(package_dir, "themes", "nice.json"), "{}")
    write(os.path.join(package_dir, "themes", "ugly.json"), "{}")
    dirs.settings.set_packages(
        [{"source": package_dir, "extensions": [], "skills": [], "prompts": [], "themes": ["!ugly.json"]}]
    )

    result = await dirs.manager.resolve()

    assert "nice.json" in enabled_names(result.themes)
    assert "ugly.json" in disabled_names(result.themes)


@pytest.mark.tonio
async def test_combines_include_and_exclude_patterns(dirs):
    package_dir = os.path.join(dirs.root, "combo-pkg")
    for name in ("alpha.py", "beta.py", "gamma.py"):
        write(os.path.join(package_dir, "extensions", name), EXTENSION_SOURCE)
    dirs.settings.set_packages(
        [
            {
                "source": package_dir,
                "extensions": ["**/alpha.py", "**/beta.py", "!**/beta.py"],
                "skills": [],
                "prompts": [],
                "themes": [],
            }
        ]
    )

    result = await dirs.manager.resolve()

    assert "alpha.py" in enabled_names(result.extensions)
    assert {"beta.py", "gamma.py"} <= disabled_names(result.extensions)


@pytest.mark.tonio
async def test_works_with_direct_paths_in_package_filters(dirs):
    package_dir = os.path.join(dirs.root, "direct-pkg")
    write(os.path.join(package_dir, "extensions", "one.py"), EXTENSION_SOURCE)
    write(os.path.join(package_dir, "extensions", "two.py"), EXTENSION_SOURCE)
    dirs.settings.set_packages(
        [{"source": package_dir, "extensions": ["extensions/one.py"], "skills": [], "prompts": [], "themes": []}]
    )

    result = await dirs.manager.resolve()

    assert "one.py" in enabled_names(result.extensions)
    assert "two.py" in disabled_names(result.extensions)


@pytest.mark.tonio
async def test_resolves_autoload_disabled_project_entries_as_deltas_over_global_packages(dirs):
    """pi uses an npm package here; a local package exercises the same delta path."""
    package_dir = os.path.join(dirs.root, "shared-tools")
    write(os.path.join(package_dir, "extensions", "foo.py"), EXTENSION_SOURCE)
    write(os.path.join(package_dir, "extensions", "bar.py"), EXTENSION_SOURCE)
    dirs.settings.set_packages([package_dir])
    dirs.settings.set_project_packages(
        [{"source": package_dir, "autoload": False, "extensions": ["-extensions/foo.py"]}]
    )

    result = await dirs.manager.resolve()

    states = {resource.path: (resource.enabled, resource.metadata.scope) for resource in result.extensions}
    assert states[os.path.join(package_dir, "extensions", "foo.py")] == (False, "project")
    assert states[os.path.join(package_dir, "extensions", "bar.py")] == (True, "user")


@pytest.mark.tonio
async def test_resolves_autoload_disabled_entries_as_positive_only_without_a_global_package(dirs):
    package_dir = os.path.join(dirs.root, "positive-only-pkg")
    write(os.path.join(package_dir, "extensions", "foo.py"), EXTENSION_SOURCE)
    write(os.path.join(package_dir, "extensions", "bar.py"), EXTENSION_SOURCE)
    write(os.path.join(package_dir, "skills", "foo", "SKILL.md"), "# Foo\n")
    relative_source = os.path.relpath(package_dir, os.path.join(dirs.root, ".pidrei"))
    dirs.settings.set_project_packages(
        [{"source": relative_source, "autoload": False, "extensions": ["+extensions/foo.py"]}]
    )

    result = await dirs.manager.resolve()

    assert paths_of(result.extensions) == [os.path.join(package_dir, "extensions", "foo.py")]
    assert result.skills == []


# -- force-include / force-exclude -------------------------------------------------


@pytest.mark.tonio
async def test_force_includes_extensions_after_an_exclusion(dirs):
    extension_dir = os.path.join(dirs.agent_dir, "extensions")
    for name in ("keep.py", "excluded.py", "force-back.py"):
        write(os.path.join(extension_dir, name), EXTENSION_SOURCE)
    dirs.settings.set_extension_paths(["extensions", "!extensions/*.py", "+extensions/force-back.py"])

    result = await dirs.manager.resolve()

    assert {"keep.py", "excluded.py"} <= disabled_names(result.extensions)
    assert "force-back.py" in enabled_names(result.extensions)


@pytest.mark.tonio
async def test_force_include_overrides_exclude_in_package_filters(dirs):
    package_dir = os.path.join(dirs.root, "force-pkg")
    for name in ("alpha.py", "beta.py", "gamma.py"):
        write(os.path.join(package_dir, "extensions", name), EXTENSION_SOURCE)
    dirs.settings.set_packages(
        [
            {
                "source": package_dir,
                "extensions": ["!**/*.py", "+extensions/beta.py"],
                "skills": [],
                "prompts": [],
                "themes": [],
            }
        ]
    )

    result = await dirs.manager.resolve()

    assert {"alpha.py", "gamma.py"} <= disabled_names(result.extensions)
    assert "beta.py" in enabled_names(result.extensions)


@pytest.mark.tonio
async def test_force_includes_multiple_resources(dirs):
    package_dir = os.path.join(dirs.root, "multi-force-pkg")
    for name in ("skill-a", "skill-b", "skill-c"):
        write_skill(os.path.join(package_dir, "skills", name, "SKILL.md"), name)
    dirs.settings.set_packages(
        [
            {
                "source": package_dir,
                "extensions": [],
                "skills": ["!**/*", "+skills/skill-a", "+skills/skill-c"],
                "prompts": [],
                "themes": [],
            }
        ]
    )

    result = await dirs.manager.resolve()

    assert any("skill-a" in r.path and r.enabled for r in result.skills)
    assert any("skill-b" in r.path and not r.enabled for r in result.skills)
    assert any("skill-c" in r.path and r.enabled for r in result.skills)


@pytest.mark.tonio
async def test_force_includes_after_a_specific_exclusion(dirs):
    extension_dir = os.path.join(dirs.agent_dir, "extensions")
    write(os.path.join(extension_dir, "a.py"), EXTENSION_SOURCE)
    write(os.path.join(extension_dir, "b.py"), EXTENSION_SOURCE)
    dirs.settings.set_extension_paths(["extensions", "!extensions/b.py", "+extensions/b.py"])

    result = await dirs.manager.resolve()

    assert {"a.py", "b.py"} <= enabled_names(result.extensions)


@pytest.mark.tonio
async def test_handles_force_include_in_manifest_patterns(dirs):
    package_dir = os.path.join(dirs.root, "manifest-force-pkg")
    for name in ("one.py", "two.py", "three.py"):
        write(os.path.join(package_dir, "extensions", name), EXTENSION_SOURCE)
    write_manifest(package_dir, {"extensions": ["extensions", "!**/two.py", "+extensions/two.py"]})

    result = await dirs.manager.resolve_extension_sources([package_dir])

    assert {"one.py", "two.py", "three.py"} <= enabled_names(result.extensions)


@pytest.mark.tonio
async def test_force_includes_themes(dirs):
    themes_dir = os.path.join(dirs.agent_dir, "themes")
    for name in ("dark.json", "light.json", "special.json"):
        write(os.path.join(themes_dir, name), "{}")
    dirs.settings.set_theme_paths(["themes", "!themes/*.json", "+themes/special.json"])

    result = await dirs.manager.resolve()

    assert {"dark.json", "light.json"} <= disabled_names(result.themes)
    assert "special.json" in enabled_names(result.themes)


@pytest.mark.tonio
async def test_force_includes_prompts(dirs):
    prompts_dir = os.path.join(dirs.agent_dir, "prompts")
    for name in ("review.md", "explain.md", "debug.md"):
        write(os.path.join(prompts_dir, name), "Text")
    dirs.settings.set_prompt_template_paths(["prompts", "!prompts/*.md", "+prompts/debug.md"])

    result = await dirs.manager.resolve()

    assert {"review.md", "explain.md"} <= disabled_names(result.prompts)
    assert "debug.md" in enabled_names(result.prompts)


@pytest.mark.tonio
async def test_force_excludes_top_level_resources(dirs):
    extension_dir = os.path.join(dirs.agent_dir, "extensions")
    write(os.path.join(extension_dir, "alpha.py"), EXTENSION_SOURCE)
    write(os.path.join(extension_dir, "beta.py"), EXTENSION_SOURCE)
    dirs.settings.set_extension_paths(["extensions", "+extensions/alpha.py", "-extensions/alpha.py"])

    result = await dirs.manager.resolve()

    assert "alpha.py" in disabled_names(result.extensions)
    assert "beta.py" in enabled_names(result.extensions)


@pytest.mark.tonio
async def test_force_excludes_in_package_filters(dirs):
    package_dir = os.path.join(dirs.root, "force-exclude-pkg")
    write(os.path.join(package_dir, "extensions", "alpha.py"), EXTENSION_SOURCE)
    write(os.path.join(package_dir, "extensions", "beta.py"), EXTENSION_SOURCE)
    dirs.settings.set_packages(
        [
            {
                "source": package_dir,
                "extensions": ["extensions/*.py", "+extensions/alpha.py", "-extensions/alpha.py"],
                "skills": [],
                "prompts": [],
                "themes": [],
            }
        ]
    )

    result = await dirs.manager.resolve()

    assert "alpha.py" in disabled_names(result.extensions)
    assert "beta.py" in enabled_names(result.extensions)


# -- package deduplication ---------------------------------------------------------


@pytest.mark.tonio
async def test_dedupes_the_same_local_package_in_global_and_project(dirs):
    package_dir = os.path.join(dirs.root, "shared-pkg")
    write(os.path.join(package_dir, "extensions", "shared.py"), EXTENSION_SOURCE)
    dirs.settings.set_packages([package_dir])
    dirs.settings.set_project_packages([package_dir])

    assert dirs.settings.get_global_settings()["packages"] == [package_dir]
    assert dirs.settings.get_project_settings()["packages"] == [package_dir]

    result = await dirs.manager.resolve()

    shared = [resource for resource in result.extensions if "shared-pkg" in resource.path]
    assert len(shared) == 1
    assert shared[0].metadata.scope == "project"


@pytest.mark.tonio
async def test_keeps_both_when_the_packages_differ(dirs):
    first = os.path.join(dirs.root, "pkg1")
    second = os.path.join(dirs.root, "pkg2")
    write(os.path.join(first, "extensions", "from-pkg1.py"), EXTENSION_SOURCE)
    write(os.path.join(second, "extensions", "from-pkg2.py"), EXTENSION_SOURCE)
    dirs.settings.set_packages([first])
    dirs.settings.set_project_packages([second])

    result = await dirs.manager.resolve()

    assert any("pkg1" in path for path in paths_of(result.extensions))
    assert any("pkg2" in path for path in paths_of(result.extensions))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("https://github.com/user/repo", "git:git@github.com:user/repo"),
        ("https://github.com/user/repo@v1.0.0", "git:git@github.com:user/repo@v1.0.0"),
        ("ssh://git@github.com/user/repo", "git:git@github.com:user/repo"),
    ],
)
def test_dedupes_equivalent_git_urls(dirs, left, right):
    assert dirs.manager._get_package_identity(left) == "git:github.com/user/repo"
    assert dirs.manager._get_package_identity(right) == "git:github.com/user/repo"


def test_dedupes_all_supported_url_formats_for_the_same_repo(dirs):
    urls = [
        "https://github.com/user/repo",
        "https://github.com/user/repo.git",
        "ssh://git@github.com/user/repo",
        "git:https://github.com/user/repo",
        "git:github.com/user/repo",
        "git:git@github.com:user/repo",
        "git:git@github.com:user/repo.git",
    ]

    identities = {dirs.manager._get_package_identity(url) for url in urls}

    assert identities == {"git:github.com/user/repo"}


def test_keeps_different_repos_separate(dirs):
    assert dirs.manager._get_package_identity("https://github.com/user/repo1") == "git:github.com/user/repo1"
    assert dirs.manager._get_package_identity("git:git@github.com:user/repo2") == "git:github.com/user/repo2"


# -- multi-file extension discovery ------------------------------------------------


@pytest.mark.tonio
async def test_only_loads_the_package_entry_point_not_helper_modules(dirs):
    package_dir = os.path.join(dirs.root, "multifile-pkg")
    write(os.path.join(package_dir, "extensions", "subagent", "__init__.py"), EXTENSION_SOURCE)
    write(os.path.join(package_dir, "extensions", "subagent", "agents.py"), "def helper():\n    return 1\n")
    write(os.path.join(package_dir, "extensions", "standalone.py"), EXTENSION_SOURCE)

    result = await dirs.manager.resolve_extension_sources([package_dir])

    assert any(path.endswith(os.path.join("subagent", "__init__.py")) for path in paths_of(result.extensions))
    assert any(path.endswith("standalone.py") for path in paths_of(result.extensions))
    assert not any(path.endswith("agents.py") for path in paths_of(result.extensions))


@pytest.mark.tonio
async def test_respects_a_manifest_in_a_subdirectory(dirs):
    package_dir = os.path.join(dirs.root, "manifest-subdir-pkg")
    custom = os.path.join(package_dir, "extensions", "custom")
    write(os.path.join(custom, "main.py"), EXTENSION_SOURCE)
    write(os.path.join(custom, "utils.py"), "UTIL = 1\n")
    write_manifest(custom, {"extensions": ["./main.py"]})

    result = await dirs.manager.resolve_extension_sources([package_dir])

    assert any(path.endswith(os.path.join("custom", "main.py")) for path in paths_of(result.extensions))
    assert not any(path.endswith("utils.py") for path in paths_of(result.extensions))


@pytest.mark.tonio
async def test_handles_mixed_top_level_files_and_subdirectories(dirs):
    package_dir = os.path.join(dirs.root, "mixed-pkg")
    write(os.path.join(package_dir, "extensions", "simple.py"), EXTENSION_SOURCE)
    complex_dir = os.path.join(package_dir, "extensions", "complex")
    write(os.path.join(complex_dir, "__init__.py"), EXTENSION_SOURCE)
    write(os.path.join(complex_dir, "a.py"), "A = 1\n")
    write(os.path.join(complex_dir, "b.py"), "B = 2\n")

    result = await dirs.manager.resolve_extension_sources([package_dir])

    paths = paths_of(result.extensions)
    assert any(path.endswith("simple.py") for path in paths)
    assert any(path.endswith(os.path.join("complex", "__init__.py")) for path in paths)
    assert not any(path.endswith(os.path.join("complex", "a.py")) for path in paths)
    assert not any(path.endswith(os.path.join("complex", "b.py")) for path in paths)
    assert len([resource for resource in result.extensions if resource.enabled]) == 2


@pytest.mark.tonio
async def test_skips_subdirectories_without_an_entry_point_or_manifest(dirs):
    package_dir = os.path.join(dirs.root, "no-entry-pkg")
    broken = os.path.join(package_dir, "extensions", "broken")
    write(os.path.join(broken, "helper.py"), "X = 1\n")
    write(os.path.join(broken, "another.py"), "Y = 2\n")
    write(os.path.join(package_dir, "extensions", "valid.py"), EXTENSION_SOURCE)

    result = await dirs.manager.resolve_extension_sources([package_dir])

    assert any(path.endswith("valid.py") for path in paths_of(result.extensions))
    assert len([resource for resource in result.extensions if resource.enabled]) == 1


# -- gaps pi's suite leaves open ---------------------------------------------------


@pytest.mark.tonio
async def test_an_explicitly_empty_filter_list_disables_that_resource_type(dirs):
    """pi passes `skills: []` throughout its package-filter cases but never
    asserts the consequence, so nothing pins that an empty list means *off*
    rather than *unfiltered*."""
    package_dir = os.path.join(dirs.root, "empty-filter-pkg")
    write(os.path.join(package_dir, "extensions", "one.py"), EXTENSION_SOURCE)
    write_skill(os.path.join(package_dir, "skills", "a-skill", "SKILL.md"), "a-skill")
    write(os.path.join(package_dir, "themes", "dark.json"), "{}")
    dirs.settings.set_packages(
        [{"source": package_dir, "extensions": ["extensions/one.py"], "skills": [], "themes": []}]
    )

    result = await dirs.manager.resolve()

    assert "one.py" in enabled_names(result.extensions)
    assert result.skills != [] and not any(resource.enabled for resource in result.skills)
    assert result.themes != [] and not any(resource.enabled for resource in result.themes)


@pytest.mark.tonio
async def test_an_autoload_disabled_package_contributes_nothing_for_unlisted_types(dirs):
    """The delta filter decides only what a pattern names; a resource type with
    no patterns must not be collected at all, not collected-and-enabled."""
    package_dir = os.path.join(dirs.root, "delta-pkg")
    write(os.path.join(package_dir, "extensions", "foo.py"), EXTENSION_SOURCE)
    write_skill(os.path.join(package_dir, "skills", "untouched", "SKILL.md"), "untouched")
    dirs.settings.set_project_packages(
        [{"source": package_dir, "autoload": False, "extensions": ["+extensions/foo.py"]}]
    )

    result = await dirs.manager.resolve()

    assert paths_of(result.extensions) == [os.path.join(package_dir, "extensions", "foo.py")]
    assert result.skills == []
