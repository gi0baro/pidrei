"""Mirror of pi's codemode-renderer.test.ts."""

import pytest

from pidrei.extensions.codemode import CodemodeNestedCall, CodemodeToolDetails
from pidrei.extensions.codemode.renderer import CODEMODE_RENDERERS
from pidrei.modes.interactive.theme import init_theme, theme
from pidrei.utils.ansi import strip_ansi
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent


HEADER = TextContent(text="Script completed\nWall time 0.1 seconds\nOutput:\n")


@pytest.fixture(autouse=True)
async def _theme():
    await init_theme("dark")


def render(result, *, is_error=False, expanded=True, width=200) -> str:
    context = {
        "args": {"code": ""},
        "toolCallId": "call",
        "invalidate": lambda: None,
        "lastComponent": None,
        "state": {},
        "cwd": "/",
        "executionStarted": True,
        "argsComplete": True,
        "isPartial": False,
        "expanded": expanded,
        "showImages": False,
        "isError": is_error,
        "outputPad": 1,
    }
    component = CODEMODE_RENDERERS.render_result(result, {"expanded": expanded, "isPartial": False}, theme, context)
    lines = [line.rstrip() for line in strip_ansi("\n".join(component.render(width))).split("\n")]
    return "\n".join(lines).strip()


def test_hides_the_script_header_and_shows_the_output():
    text = render(
        AgentToolResult(
            content=[HEADER, TextContent(text="hello")],
            details=CodemodeToolDetails(
                calls=[CodemodeNestedCall(id="call/1", name="read", args='{"path":"a"}', status="ok", duration_ms=5)]
            ),
        )
    )
    assert text == '✓ read {"path":"a"} 5ms\n\nhello'


def test_shows_the_cost_of_model_calls_and_their_total():
    def call(number: int, cost: float | None) -> CodemodeNestedCall:
        return CodemodeNestedCall(
            id=f"call/models.classify/{number}",
            name="models.classify",
            args="scorer/judge",
            status="ok",
            duration_ms=5,
            cost=cost,
        )

    text = render(
        AgentToolResult(
            content=[HEADER], details=CodemodeToolDetails(calls=[call(1, 0.000012936), call(2, 0.02), call(3, None)])
        )
    )
    assert text.split("\n") == [
        "✓ models.classify scorer/judge 5ms $0.000013",
        "✓ models.classify scorer/judge 5ms $0.02",
        "✓ models.classify scorer/judge 5ms",
        "Model calls: $0.02",
    ]


def test_shows_results_without_a_header_such_as_rejected_options():
    text = render(
        AgentToolResult(content=[TextContent(text="The @options line must be followed by Python source")]),
        is_error=True,
    )
    assert text == "The @options line must be followed by Python source"


def test_limits_collapsed_output_to_wrapped_lines_not_logical_lines():
    text = render(
        AgentToolResult(
            content=[HEADER, TextContent(text="x" * 1000)],
            details=CodemodeToolDetails(calls=[], full_output_path="/tmp/out.txt"),
        ),
        expanded=False,
        width=50,
    )
    lines = text.split("\n")
    assert len(lines) == 7
    assert lines[:5] == ["x" * 50] * 5
    assert lines[5].startswith("... (15 more lines,")
    assert lines[6] == "Full output: /tmp/out.txt"


def test_renders_details_read_back_from_a_session_file():
    """pidrei-only: a resumed session hands the renderer the details' wire form
    (camelCase keys), which renders like the live dataclass."""
    text = render(
        {
            "content": [HEADER, TextContent(text="hello")],
            "details": {
                "calls": [
                    {"id": "call/1", "name": "read", "args": '{"path":"a"}', "status": "error", "durationMs": 1500}
                ]
            },
        }
    )
    assert text == '✗ read {"path":"a"} 1.5s\n\nhello'
