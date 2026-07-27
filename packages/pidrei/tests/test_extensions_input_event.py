"""Mirror of pi's extensions-input-event.test.ts.

pi's extensions stash observations on `globalThis`; here they append to a
file, because each pidrei extension load gets a fresh module by design.
"""

import os
import shutil
import tempfile

import pytest

from pidrei.core.auth_storage import AuthStorage
from pidrei.core.extensions.loader import discover_and_load_extensions
from pidrei.core.extensions.runner import ExtensionRunner
from pidrei.core.session_manager import SessionManager

from .model_runtime_helpers import create_in_memory_model_registry


RECORD = """
import os


def record(value):
    with open(os.environ["PIDREI_INPUT_EVENT_LOG"], "a") as handle:
        handle.write(f"{value}\\n")
"""


class _Fixture:
    def __init__(self) -> None:
        self.root = tempfile.mkdtemp(prefix="pidrei-input-test-")
        self.extensions = os.path.join(self.root, "extensions")
        os.makedirs(self.extensions)
        self.log = os.path.join(self.root, "observed.log")
        os.environ["PIDREI_INPUT_EVENT_LOG"] = self.log

    def observed(self) -> list[str]:
        if not os.path.exists(self.log):
            return []
        with open(self.log) as handle:
            return handle.read().split()

    def cleanup(self) -> None:
        os.environ.pop("PIDREI_INPUT_EVENT_LOG", None)
        shutil.rmtree(self.root, ignore_errors=True)


@pytest.fixture
def fx(request):
    holder = _Fixture()
    request.addfinalizer(holder.cleanup)
    return holder


async def create_runner(fx, *extensions: str) -> ExtensionRunner:
    shutil.rmtree(fx.extensions, ignore_errors=True)
    os.makedirs(fx.extensions)
    # The recorder is a private helper module, so discovery skips it.
    with open(os.path.join(fx.extensions, "_record.py"), "w") as handle:
        handle.write(RECORD)
    for index, source in enumerate(extensions):
        with open(os.path.join(fx.extensions, f"e{index}.py"), "w") as handle:
            handle.write(source)

    result = await discover_and_load_extensions([], fx.root, fx.root)
    assert result.errors == []
    session_manager = SessionManager.in_memory()
    model_registry = await create_in_memory_model_registry(await AuthStorage.create(os.path.join(fx.root, "auth.json")))
    return ExtensionRunner(result.extensions, result.runtime, fx.root, session_manager, model_registry)


def handler_extension(body: str) -> str:
    return f"""
from ._record import record


async def handler(event, ctx):
{body}


def extension(pi):
    pi.on("input", handler)
"""


@pytest.mark.tonio
async def test_returns_continue_with_no_handlers_none_return_or_explicit_continue(fx):
    runner = await create_runner(fx)
    assert (await runner.emit_input("x", None, "interactive")).action == "continue"

    runner = await create_runner(fx, handler_extension("    return None"))
    assert (await runner.emit_input("x", None, "interactive")).action == "continue"

    runner = await create_runner(fx, handler_extension('    return {"action": "continue"}'))
    assert (await runner.emit_input("x", None, "interactive")).action == "continue"


@pytest.mark.tonio
async def test_transforms_text_and_preserves_images_when_omitted(fx):
    runner = await create_runner(
        fx, handler_extension('    return {"action": "transform", "text": "T:" + event["text"]}')
    )
    images = [{"type": "image", "data": "orig", "mimeType": "image/png"}]

    result = await runner.emit_input("hi", images, "interactive")

    assert result.action == "transform"
    assert result.text == "T:hi"
    assert result.images == images


@pytest.mark.tonio
async def test_transforms_and_replaces_images_when_provided(fx):
    runner = await create_runner(
        fx,
        handler_extension(
            '    return {"action": "transform", "text": "X",'
            ' "images": [{"type": "image", "data": "new", "mimeType": "image/jpeg"}]}'
        ),
    )

    result = await runner.emit_input("hi", [{"type": "image", "data": "orig", "mimeType": "image/png"}], "interactive")

    assert result.action == "transform"
    assert result.text == "X"
    assert result.images == [{"type": "image", "data": "new", "mimeType": "image/jpeg"}]


@pytest.mark.tonio
async def test_chains_transforms_across_multiple_handlers(fx):
    runner = await create_runner(
        fx,
        handler_extension('    return {"action": "transform", "text": event["text"] + "[1]"}'),
        handler_extension('    return {"action": "transform", "text": event["text"] + "[2]"}'),
    )

    result = await runner.emit_input("X", None, "interactive")

    assert result.action == "transform"
    assert result.text == "X[1][2]"
    assert result.images is None


@pytest.mark.tonio
async def test_short_circuits_on_handled_and_skips_subsequent_handlers(fx):
    runner = await create_runner(
        fx,
        handler_extension('    return {"action": "handled"}'),
        handler_extension('    record("reached")\n    return None'),
    )

    result = await runner.emit_input("X", None, "interactive")

    assert result.action == "handled"
    assert result.text is None
    assert fx.observed() == []


@pytest.mark.tonio
async def test_passes_source_correctly_for_all_source_types(fx):
    runner = await create_runner(
        fx, handler_extension('    record(event["source"])\n    return {"action": "continue"}')
    )

    for source in ("interactive", "rpc", "extension"):
        await runner.emit_input("x", None, source)

    assert fx.observed() == ["interactive", "rpc", "extension"]


@pytest.mark.tonio
async def test_passes_streaming_behavior_correctly(fx):
    runner = await create_runner(
        fx, handler_extension('    record(event["streamingBehavior"])\n    return {"action": "continue"}')
    )

    await runner.emit_input("x", None, "interactive", "steer")
    await runner.emit_input("x", None, "interactive", "followUp")
    await runner.emit_input("x", None, "interactive")

    assert fx.observed() == ["steer", "followUp", "None"]


@pytest.mark.tonio
async def test_catches_handler_errors_and_continues(fx):
    runner = await create_runner(fx, handler_extension('    raise RuntimeError("boom")'))
    errors: list[str] = []
    runner.on_error(lambda error: errors.append(error.error))

    result = await runner.emit_input("x", None, "interactive")

    assert result.action == "continue"
    assert "boom" in errors


@pytest.mark.tonio
async def test_has_handlers_returns_the_correct_value(fx):
    runner = await create_runner(fx)
    assert runner.has_handlers("input") is False

    runner = await create_runner(fx, handler_extension("    return None"))
    assert runner.has_handlers("input") is True
