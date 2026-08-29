"""Mirror of pi's extensions-discovery.test.ts.

Translated to the pidrei ABI: `.ts`/`.js` files become `.py` modules whose
factory is a module-level `extension(pi)`, `index.ts` becomes `__init__.py`,
and `package.json`'s `pi.extensions` becomes `pyproject.toml`'s
`[tool.pidrei] extensions`.

Three of pi's cases have no analogue and are not mirrored: the two that assert
jiti's alias table still resolves `@earendil-works/pi-coding-agent` and
`@earendil-works/pi-ai/oauth` (a pidrei extension imports `pidrei`/`pidrei_ai`
from the same interpreter — there is no alias table to break), and the one
loading an extension with its own bundled `node_modules`.
"""

import json
import os
import shutil
import tempfile

import pytest

from pidrei.core.extensions.loader import discover_and_load_extensions, load_extensions


EXTENSION_CODE = """
def extension(pi):
    pi.register_command("test", handler=lambda args, ctx: None)
"""


def extension_code_with_tool(tool_name: str) -> str:
    return f"""
from pidrei.core.extensions import ToolDefinition


def extension(pi):
    pi.register_tool(
        ToolDefinition(
            name={tool_name!r},
            label={tool_name!r},
            description="Test tool",
            parameters={{"type": "object", "properties": {{}}}},
            execute=lambda *args: {{"content": [{{"type": "text", "text": "ok"}}]}},
        )
    )
"""


def write(path: str, content: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(content)
    return path


class _Temp:
    """Suite-managed temp dir (predates tonio 0.9.14 yield-fixture support)."""

    def __init__(self) -> None:
        self.root = tempfile.mkdtemp(prefix="pidrei-ext-test-")
        self.extensions = os.path.join(self.root, "extensions")
        os.makedirs(self.extensions)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


@pytest.fixture
def temp(request):
    holder = _Temp()
    request.addfinalizer(holder.cleanup)
    return holder


@pytest.mark.tonio
async def test_discovers_direct_py_files_in_extensions(temp):
    write(os.path.join(temp.extensions, "foo.py"), EXTENSION_CODE)
    write(os.path.join(temp.extensions, "bar.py"), EXTENSION_CODE)

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert len(result.extensions) == 2
    assert sorted(os.path.basename(e.path) for e in result.extensions) == ["bar.py", "foo.py"]


@pytest.mark.tonio
async def test_discovers_subdirectory_with_package_init(temp):
    subdir = os.path.join(temp.extensions, "my-extension")
    write(os.path.join(subdir, "__init__.py"), EXTENSION_CODE)

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert len(result.extensions) == 1
    assert "my-extension" in result.extensions[0].path
    assert "__init__.py" in result.extensions[0].path


@pytest.mark.tonio
async def test_discovers_subdirectory_with_pyproject_manifest(temp):
    subdir = os.path.join(temp.extensions, "my-package")
    write(os.path.join(subdir, "src", "main.py"), EXTENSION_CODE)
    write(
        os.path.join(subdir, "pyproject.toml"),
        '[project]\nname = "my-package"\n\n[tool.pidrei]\nextensions = ["./src/main.py"]\n',
    )

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert len(result.extensions) == 1
    assert "src" in result.extensions[0].path
    assert "main.py" in result.extensions[0].path


@pytest.mark.tonio
async def test_keeps_manifest_entries_with_leading_tilde_package_relative(temp):
    subdir = os.path.join(temp.extensions, "tilde-package")
    direct = write(os.path.join(subdir, "~entry.py"), EXTENSION_CODE)
    slashed = write(os.path.join(subdir, "~", "entry.py"), EXTENSION_CODE)
    write(
        os.path.join(subdir, "pyproject.toml"),
        '[tool.pidrei]\nextensions = ["~entry.py", "~/entry.py"]\n',
    )

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert sorted(e.path for e in result.extensions) == sorted([direct, slashed])


@pytest.mark.tonio
async def test_manifest_can_declare_multiple_extensions(temp):
    subdir = os.path.join(temp.extensions, "my-package")
    write(os.path.join(subdir, "ext1.py"), EXTENSION_CODE)
    write(os.path.join(subdir, "ext2.py"), EXTENSION_CODE)
    write(os.path.join(subdir, "pyproject.toml"), '[tool.pidrei]\nextensions = ["./ext1.py", "./ext2.py"]\n')

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert len(result.extensions) == 2


@pytest.mark.tonio
async def test_manifest_takes_precedence_over_package_init(temp):
    subdir = os.path.join(temp.extensions, "my-package")
    write(os.path.join(subdir, "__init__.py"), extension_code_with_tool("from-index"))
    write(os.path.join(subdir, "custom.py"), extension_code_with_tool("from-custom"))
    write(os.path.join(subdir, "pyproject.toml"), '[tool.pidrei]\nextensions = ["./custom.py"]\n')

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert len(result.extensions) == 1
    assert "custom.py" in result.extensions[0].path
    assert "from-custom" in result.extensions[0].tools
    assert "from-index" not in result.extensions[0].tools


@pytest.mark.tonio
async def test_ignores_pyproject_without_pidrei_table_falls_back_to_package_init(temp):
    subdir = os.path.join(temp.extensions, "my-package")
    write(os.path.join(subdir, "__init__.py"), EXTENSION_CODE)
    write(os.path.join(subdir, "pyproject.toml"), '[project]\nname = "my-package"\nversion = "1.0.0"\n')

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert len(result.extensions) == 1
    assert "__init__.py" in result.extensions[0].path


@pytest.mark.tonio
async def test_ignores_subdirectory_without_package_init_or_manifest(temp):
    subdir = os.path.join(temp.extensions, "not-an-extension")
    write(os.path.join(subdir, "helper.py"), EXTENSION_CODE)
    write(os.path.join(subdir, "utils.py"), EXTENSION_CODE)

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert result.extensions == []


@pytest.mark.tonio
async def test_does_not_recurse_beyond_one_level(temp):
    nested = os.path.join(temp.extensions, "container", "nested")
    write(os.path.join(nested, "__init__.py"), EXTENSION_CODE)
    # No __init__.py or pyproject.toml in container/

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert result.extensions == []


@pytest.mark.tonio
async def test_handles_mixed_direct_files_and_subdirectories(temp):
    write(os.path.join(temp.extensions, "direct.py"), EXTENSION_CODE)
    write(os.path.join(temp.extensions, "with-init", "__init__.py"), EXTENSION_CODE)
    manifest_dir = os.path.join(temp.extensions, "with-manifest")
    write(os.path.join(manifest_dir, "entry.py"), EXTENSION_CODE)
    write(os.path.join(manifest_dir, "pyproject.toml"), '[tool.pidrei]\nextensions = ["./entry.py"]\n')

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert len(result.extensions) == 3


@pytest.mark.tonio
async def test_skips_non_existent_paths_declared_in_the_manifest(temp):
    subdir = os.path.join(temp.extensions, "my-package")
    write(os.path.join(subdir, "exists.py"), EXTENSION_CODE)
    write(
        os.path.join(subdir, "pyproject.toml"),
        '[tool.pidrei]\nextensions = ["./exists.py", "./missing.py"]\n',
    )

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert len(result.extensions) == 1
    assert "exists.py" in result.extensions[0].path


@pytest.mark.tonio
async def test_loads_extensions_and_registers_commands(temp):
    write(os.path.join(temp.extensions, "with-command.py"), EXTENSION_CODE)

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert len(result.extensions) == 1
    assert "test" in result.extensions[0].commands


@pytest.mark.tonio
async def test_loads_extensions_and_registers_tools(temp):
    write(os.path.join(temp.extensions, "with-tool.py"), extension_code_with_tool("my-tool"))

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert len(result.extensions) == 1
    assert "my-tool" in result.extensions[0].tools


@pytest.mark.tonio
async def test_reports_errors_for_invalid_extension_code(temp):
    write(os.path.join(temp.extensions, "invalid.py"), "this is not valid python def")

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert len(result.errors) == 1
    assert "invalid.py" in result.errors[0].path
    assert result.extensions == []


@pytest.mark.tonio
async def test_handles_explicitly_configured_paths(temp):
    custom = write(os.path.join(temp.root, "custom-location", "my-ext.py"), EXTENSION_CODE)

    result = await discover_and_load_extensions([custom], temp.root, temp.root)

    assert result.errors == []
    assert len(result.extensions) == 1
    assert "my-ext.py" in result.extensions[0].path


@pytest.mark.tonio
async def test_registers_message_and_entry_renderers(temp):
    write(
        os.path.join(temp.extensions, "with-renderer.py"),
        """
def extension(pi):
    pi.register_markdown_transformer(lambda markdown, context: markdown)
    pi.register_message_renderer("my-custom-type", lambda *args: None)
    pi.register_entry_renderer("my-entry-type", lambda *args: None)
""",
    )

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert len(result.extensions) == 1
    assert result.extensions[0].markdown_transformer is not None
    assert "my-custom-type" in result.extensions[0].message_renderers
    assert "my-entry-type" in result.extensions[0].entry_renderers


@pytest.mark.tonio
async def test_reports_error_when_extension_raises_during_initialization(temp):
    write(
        os.path.join(temp.extensions, "throws.py"),
        '\ndef extension(pi):\n    raise RuntimeError("Initialization failed!")\n',
    )

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert len(result.errors) == 1
    assert "Initialization failed!" in result.errors[0].error
    assert result.extensions == []


@pytest.mark.tonio
async def test_reports_error_when_the_module_defines_no_factory(temp):
    write(
        os.path.join(temp.extensions, "no-factory.py"),
        '\ndef not_the_factory(pi):\n    pi.register_command("test", handler=lambda args, ctx: None)\n',
    )

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert len(result.errors) == 1
    assert "does not define a valid extension() factory function" in result.errors[0].error
    assert result.extensions == []


@pytest.mark.tonio
async def test_allows_multiple_extensions_to_register_different_tools(temp):
    write(os.path.join(temp.extensions, "tool-a.py"), extension_code_with_tool("tool-a"))
    write(os.path.join(temp.extensions, "tool-b.py"), extension_code_with_tool("tool-b"))

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert len(result.extensions) == 2

    all_tools = {name for ext in result.extensions for name in ext.tools}
    assert "tool-a" in all_tools
    assert "tool-b" in all_tools


@pytest.mark.tonio
async def test_loads_extension_with_event_handlers(temp):
    write(
        os.path.join(temp.extensions, "with-handlers.py"),
        """
async def noop(event, ctx):
    return None


def extension(pi):
    pi.on("agent_start", noop)
    pi.on("tool_call", noop)
    pi.on("agent_end", noop)
""",
    )

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert len(result.extensions) == 1
    handlers = result.extensions[0].handlers
    assert "agent_start" in handlers
    assert "tool_call" in handlers
    assert "agent_end" in handlers


@pytest.mark.tonio
async def test_loads_extension_with_shortcuts(temp):
    write(
        os.path.join(temp.extensions, "with-shortcut.py"),
        """
def extension(pi):
    pi.register_shortcut("ctrl+t", description="Test shortcut", handler=lambda ctx: None)
""",
    )

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert "ctrl+t" in result.extensions[0].shortcuts


@pytest.mark.tonio
async def test_loads_extension_with_flags(temp):
    write(
        os.path.join(temp.extensions, "with-flag.py"),
        """
def extension(pi):
    pi.register_flag("my-flag", type="boolean", description="My custom flag")
""",
    )

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert "my-flag" in result.extensions[0].flags


@pytest.mark.tonio
async def test_load_extensions_only_loads_explicit_paths_without_discovery(temp):
    write(os.path.join(temp.extensions, "discovered.py"), extension_code_with_tool("discovered"))
    explicit = write(os.path.join(temp.root, "explicit.py"), extension_code_with_tool("explicit"))

    result = await load_extensions([explicit], temp.root)

    assert result.errors == []
    assert len(result.extensions) == 1
    assert "explicit" in result.extensions[0].tools
    assert "discovered" not in result.extensions[0].tools


@pytest.mark.tonio
async def test_load_extensions_with_no_paths_loads_nothing(temp):
    write(os.path.join(temp.extensions, "discovered.py"), EXTENSION_CODE)

    result = await load_extensions([], temp.root)

    assert result.errors == []
    assert result.extensions == []


# -- pidrei-only: rules that only exist because the ABI is Python ----------------


@pytest.mark.tonio
async def test_underscore_prefixed_modules_are_helpers_not_extensions(temp):
    """pi has no equivalent: `.ts` has no private-module convention, so a
    helper sitting next to an extension would be loaded and fail."""
    write(os.path.join(temp.extensions, "_helper.py"), "VALUE = 1\n")
    write(
        os.path.join(temp.extensions, "uses-helper.py"),
        """
from . import _helper


def extension(pi):
    pi.register_command(str(_helper.VALUE), handler=lambda args, ctx: None)
""",
    )

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert len(result.extensions) == 1
    # The relative import resolved against the extension's own directory.
    assert "1" in result.extensions[0].commands


@pytest.mark.tonio
async def test_a_package_extension_can_import_its_own_submodules(temp):
    subdir = os.path.join(temp.extensions, "pkg-ext")
    write(os.path.join(subdir, "inner.py"), "NAME = 'from-inner'\n")
    write(
        os.path.join(subdir, "__init__.py"),
        """
from .inner import NAME


def extension(pi):
    pi.register_command(NAME, handler=lambda args, ctx: None)
""",
    )

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert "from-inner" in result.extensions[0].commands


@pytest.mark.tonio
async def test_an_async_factory_is_awaited(temp):
    write(
        os.path.join(temp.extensions, "async-factory.py"),
        """
import tonio.colored as tonio


async def extension(pi):
    await tonio.sleep(0)
    pi.register_command("late", handler=lambda args, ctx: None)
""",
    )

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert "late" in result.extensions[0].commands


@pytest.mark.tonio
async def test_the_manifest_is_read_from_the_tool_pidrei_table_only(temp):
    """A `[tool.pi]` table is another tool's config, not ours."""
    subdir = os.path.join(temp.extensions, "other-tool")
    write(os.path.join(subdir, "entry.py"), EXTENSION_CODE)
    write(os.path.join(subdir, "pyproject.toml"), '[tool.pi]\nextensions = ["./entry.py"]\n')

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert result.extensions == []


@pytest.mark.tonio
async def test_a_malformed_pyproject_is_ignored_not_fatal(temp):
    subdir = os.path.join(temp.extensions, "broken-manifest")
    write(os.path.join(subdir, "__init__.py"), EXTENSION_CODE)
    write(os.path.join(subdir, "pyproject.toml"), "this is not toml [[[\n")

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert len(result.extensions) == 1
    assert "__init__.py" in result.extensions[0].path


@pytest.mark.tonio
async def test_json_manifests_are_not_read(temp):
    """Guard against reintroducing pi's package.json path by habit."""
    subdir = os.path.join(temp.extensions, "npm-style")
    write(os.path.join(subdir, "entry.py"), EXTENSION_CODE)
    write(
        os.path.join(subdir, "package.json"),
        json.dumps({"name": "npm-style", "pi": {"extensions": ["./entry.py"]}}),
    )

    result = await discover_and_load_extensions([], temp.root, temp.root)

    assert result.errors == []
    assert result.extensions == []
