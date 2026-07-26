"""Mirror of pi's regressions/extension-factory-cache.test.ts.

pi counts module evaluations and factory runs through a global on
`globalThis`. Here the extension writes to a file whose path it reads from the
environment, because each load gets a fresh module object *by design* — a
module-level counter would be reset by the very thing under test.
"""

import os
import shutil
import tempfile

import pytest

from pidrei.core.extensions.loader import clear_extension_cache, load_extensions, load_extensions_cached
from pidrei.core.resource_loader import DefaultResourceLoader


COUNTING_EXTENSION = """
import os

with open(os.environ["PIDREI_CACHE_TEST_COUNTER"], "a") as handle:
    handle.write("module\\n")


def extension(pi):
    with open(os.environ["PIDREI_CACHE_TEST_COUNTER"], "a") as handle:
        handle.write("factory\\n")
"""


class _Fixture:
    def __init__(self) -> None:
        self.root = tempfile.mkdtemp(prefix="pidrei-extension-cache-")
        self.cwd = os.path.join(self.root, "project")
        self.agent_dir = os.path.join(self.root, "agent")
        os.makedirs(self.cwd)
        os.makedirs(self.agent_dir)
        self.counter = os.path.join(self.root, "counter.log")
        os.environ["PIDREI_CACHE_TEST_COUNTER"] = self.counter
        clear_extension_cache()

    def write_counting_extension(self, path: str) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            handle.write(COUNTING_EXTENSION)
        return path

    def counts(self) -> tuple[int, int]:
        if not os.path.exists(self.counter):
            return 0, 0
        with open(self.counter) as handle:
            lines = handle.read().split()
        return lines.count("module"), lines.count("factory")

    def cleanup(self) -> None:
        os.environ.pop("PIDREI_CACHE_TEST_COUNTER", None)
        shutil.rmtree(self.root, ignore_errors=True)
        clear_extension_cache()


@pytest.fixture
def fixture(request):
    holder = _Fixture()
    request.addfinalizer(holder.cleanup)
    return holder


@pytest.mark.tonio
async def test_caches_extension_modules_for_cached_same_cwd_loads_but_reruns_factories(fixture):
    path = fixture.write_counting_extension(os.path.join(fixture.root, "counting.py"))

    first = await load_extensions_cached([path], fixture.cwd)
    second = await load_extensions_cached([path], fixture.cwd)

    assert fixture.counts() == (1, 2)
    assert first.extensions[0] is not second.extensions[0]
    assert first.runtime is not second.runtime


@pytest.mark.tonio
async def test_does_not_cache_direct_load_extensions_calls(fixture):
    path = fixture.write_counting_extension(os.path.join(fixture.root, "counting.py"))

    await load_extensions([path], fixture.cwd)
    await load_extensions([path], fixture.cwd)

    assert fixture.counts() == (2, 2)


@pytest.mark.tonio
async def test_clears_the_cache_on_resource_loader_reload(fixture):
    fixture.write_counting_extension(os.path.join(fixture.agent_dir, "extensions", "counting.py"))
    loader = DefaultResourceLoader(
        cwd=fixture.cwd,
        agent_dir=fixture.agent_dir,
        no_skills=True,
        no_prompt_templates=True,
        no_themes=True,
    )

    await loader.reload()
    await loader.reload()

    assert fixture.counts() == (2, 2)


@pytest.mark.tonio
async def test_keeps_the_cache_scoped_to_one_cwd(fixture):
    first_cwd = os.path.join(fixture.root, "first")
    second_cwd = os.path.join(fixture.root, "second")
    os.makedirs(first_cwd)
    os.makedirs(second_cwd)
    path = fixture.write_counting_extension(os.path.join(fixture.root, "counting.py"))

    await load_extensions_cached([path], first_cwd)
    await load_extensions_cached([path], second_cwd)
    await load_extensions_cached([path], second_cwd)

    assert fixture.counts() == (2, 3)
