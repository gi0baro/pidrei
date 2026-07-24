"""Mirror of pi agent/test/harness/prompt-templates.test.ts."""

import os

import pytest

from pidrei_agent.harness.env.local import LocalExecutionEnv
from pidrei_agent.harness.prompt_templates import (
    PromptTemplate,
    SourcedPromptTemplate,
    format_prompt_template_invocation,
    load_prompt_templates,
    load_sourced_prompt_templates,
)
from pidrei_agent.harness.types import get_or_throw
from tests.session_helpers import create_temp_dir


@pytest.mark.tonio
async def test_loads_markdown_templates_non_recursively_from_one_or_more_dirs():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    get_or_throw(await env.create_dir("a/nested", recursive=True))
    get_or_throw(await env.create_dir("b", recursive=True))
    get_or_throw(await env.write_file("a/one.md", "---\ndescription: One template\n---\nHello $1"))
    get_or_throw(await env.write_file("a/nested/ignored.md", "Ignored"))
    get_or_throw(await env.write_file("b/two.md", "First line description\nBody"))

    result = await load_prompt_templates(env, ["a", "b"])

    assert result.diagnostics == []
    assert result.prompt_templates == [
        PromptTemplate(name="one", description="One template", content="Hello $1"),
        PromptTemplate(name="two", description="First line description", content="First line description\nBody"),
    ]


@pytest.mark.tonio
async def test_preserves_source_info_for_sourced_prompt_templates():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    get_or_throw(await env.create_dir("prompts", recursive=True))
    get_or_throw(await env.write_file("prompts/example.md", "---\ndescription: Example\n---\nExample body"))

    result = await load_sourced_prompt_templates(env, [{"path": "prompts", "source": {"type": "project"}}])

    assert result.diagnostics == []
    assert result.prompt_templates == [
        SourcedPromptTemplate(
            prompt_template=PromptTemplate(name="example", description="Example", content="Example body"),
            source={"type": "project"},
        )
    ]


@pytest.mark.tonio
async def test_attaches_source_info_to_diagnostics():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    get_or_throw(await env.write_file("broken.md", "---\ndescription: [unterminated\n---\nBody"))

    result = await load_sourced_prompt_templates(env, [{"path": "broken.md", "source": {"type": "user"}}])

    assert result.prompt_templates == []
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.type == "warning"
    assert diagnostic.path == os.path.join(root, "broken.md")
    assert diagnostic.source == {"type": "user"}


@pytest.mark.tonio
async def test_loads_explicit_markdown_files_and_symlinked_files():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    get_or_throw(await env.write_file("target.md", "---\ndescription: Target\n---\nTarget body"))
    os.symlink(os.path.join(root, "target.md"), os.path.join(root, "link.md"))

    result = await load_prompt_templates(env, ["target.md", "link.md"])

    assert result.prompt_templates == [
        PromptTemplate(name="target", description="Target", content="Target body"),
        PromptTemplate(name="link", description="Target", content="Target body"),
    ]


@pytest.mark.tonio
async def test_substitutes_command_arguments():
    content = "$1 $" + "{@:2} $ARGUMENTS"
    assert format_prompt_template_invocation(PromptTemplate(name="one", content=content), ["hello world", "test"]) == (
        "hello world test hello world test"
    )
