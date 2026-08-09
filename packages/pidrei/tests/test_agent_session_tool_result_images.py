"""Mirror of pi's suite/agent-session-tool-result-images.test.ts.

The fixture PNG is built with Pillow rather than hand-encoded (see
`test_tool_result_images.py`), and the tool is a `ToolDefinition` because
pidrei's harness registers extension-shaped tools.
"""

import base64
import io
import struct

import pytest
from PIL import Image

from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call
from pidrei_ai.types import ImageContent, TextContent

from .harness import create_harness


def _create_png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (width, height)).save(buffer, format="PNG")
    return buffer.getvalue()


def _read_png_dimensions(base64_data: str) -> tuple[int, int]:
    buffer = base64.b64decode(base64_data)
    return struct.unpack(">I", buffer[16:20])[0], struct.unpack(">I", buffer[20:24])[0]


OVERSIZED_PNG_BASE64 = base64.b64encode(_create_png(2400, 4800)).decode("ascii")


def _screenshot_tool() -> ToolDefinition:
    """Stands in for extension, MCP bridge, or screenshot tools returning images."""

    async def execute(*_args):
        return AgentToolResult(
            content=[
                TextContent(text="captured"),
                ImageContent(data=OVERSIZED_PNG_BASE64, mime_type="image/png"),
            ],
            details={},
        )

    return ToolDefinition(
        name="screenshot",
        label="Screenshot",
        description="Return an oversized screenshot",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )


def _tool_result_images(harness) -> list:
    return [
        block
        for message in harness.session.messages
        if message.role == "toolResult"
        for block in message.content
        if block.type == "image"
    ]


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


@pytest.mark.tonio
async def test_resizes_oversized_tool_result_images_before_they_enter_history(harnesses):
    harness = await create_harness(tools=[_screenshot_tool()])
    harnesses.append(harness)
    harness.set_responses(
        [
            faux_assistant_message([faux_tool_call("screenshot", {})], stop_reason="toolUse"),
            faux_assistant_message("done"),
        ]
    )

    await harness.session.prompt("take a screenshot")

    images = _tool_result_images(harness)
    assert len(images) == 1
    width, height = _read_png_dimensions(images[0].data)
    assert width <= 2000
    assert height <= 2000


@pytest.mark.tonio
async def test_honors_image_auto_resize_being_disabled(harnesses):
    harness = await create_harness(tools=[_screenshot_tool()], settings={"images": {"autoResize": False}})
    harnesses.append(harness)
    harness.set_responses(
        [
            faux_assistant_message([faux_tool_call("screenshot", {})], stop_reason="toolUse"),
            faux_assistant_message("done"),
        ]
    )

    await harness.session.prompt("take a screenshot")

    images = _tool_result_images(harness)
    assert len(images) == 1
    assert images[0].data == OVERSIZED_PNG_BASE64
