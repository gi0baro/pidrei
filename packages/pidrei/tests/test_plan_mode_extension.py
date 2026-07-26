"""Mirror of pi's plan-mode-extension.test.ts."""

from types import SimpleNamespace

import pytest

from .example_extensions import load_example


def create_assistant_message(text: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "api": "anthropic-messages",
        "provider": "anthropic",
        "model": "mock",
        "stopReason": "stop",
    }


class _Setup:
    def __init__(self, *, active_tools=None, select_choice=None, editor_text=None):
        self.active_tools = list(active_tools or ["read", "bash", "edit", "write"])
        self.set_active_tools_calls: list[list[str]] = []
        self.messages: list[tuple] = []
        self.user_messages: list[tuple] = []
        self.entries: list[tuple] = []
        self.select_calls: list = []
        self._commands: dict = {}
        self._handlers: dict = {}

        def set_active_tools(tool_names):
            self.active_tools = list(tool_names)
            self.set_active_tools_calls.append(list(tool_names))

        api = SimpleNamespace(
            register_flag=lambda *_args, **_kwargs: None,
            register_command=lambda name, *, handler, **_kwargs: self._commands.__setitem__(name, handler),
            register_shortcut=lambda *_args, **_kwargs: None,
            on=lambda event, handler: self._handlers.__setitem__(event, handler),
            get_flag=lambda _name: False,
            get_active_tools=lambda: list(self.active_tools),
            set_active_tools=set_active_tools,
            send_message=lambda message, options=None: self.messages.append((message, options)),
            send_user_message=lambda content, options=None: self.user_messages.append((content, options)),
            append_entry=lambda custom_type, data=None: self.entries.append((custom_type, data)),
        )
        load_example("plan_mode").extension(api)

        async def select(title, options, **_kwargs):
            self.select_calls.append((title, options))
            return select_choice

        async def editor(_title, _prefill=None):
            return editor_text

        theme = SimpleNamespace(fg=lambda _name, text: text, strikethrough=lambda text: text)
        self.ctx = SimpleNamespace(
            has_ui=True,
            ui=SimpleNamespace(
                notify=lambda *_args: None,
                select=select,
                editor=editor,
                set_status=lambda *_args: None,
                set_widget=lambda *_args: None,
                theme=theme,
            ),
            session_manager=SimpleNamespace(get_entries=list),
            is_idle=lambda: False,
            has_pending_messages=lambda: False,
        )

    async def run_command(self, name: str) -> None:
        await self._commands[name]("", self.ctx)

    async def trigger_agent_end(self, text: str) -> None:
        await self._handlers["agent_end"]({"type": "agent_end", "messages": [create_assistant_message(text)]}, self.ctx)


@pytest.mark.tonio
async def test_preserves_custom_active_tools_while_toggling_plan_mode():
    setup = _Setup(active_tools=["read", "bash", "edit", "write", "echo_tool"])

    await setup.run_command("plan")

    expected_plan = ["read", "bash", "echo_tool", "grep", "find", "ls", "questionnaire"]
    assert setup.active_tools == expected_plan
    assert setup.set_active_tools_calls[-1] == expected_plan

    await setup.run_command("plan")

    assert setup.active_tools == ["read", "bash", "edit", "write", "echo_tool"]
    assert setup.set_active_tools_calls[-1] == ["read", "bash", "edit", "write", "echo_tool"]


@pytest.mark.tonio
async def test_does_not_prompt_when_the_response_contains_no_plan():
    setup = _Setup()

    await setup.run_command("plan")
    await setup.trigger_agent_end("This file defines the command-line argument parser.")

    assert setup.select_calls == []
    assert setup.messages == []


@pytest.mark.tonio
async def test_queues_plan_refinement_as_a_follow_up_user_message():
    setup = _Setup(select_choice="Refine the plan", editor_text="Add a regression test.")

    await setup.run_command("plan")
    await setup.trigger_agent_end("Plan:\n1. Inspect the current implementation\n2. Add a regression test")

    assert setup.user_messages == [("Add a regression test.", {"deliverAs": "followUp"})]


@pytest.mark.tonio
async def test_queues_plan_execution_as_a_follow_up_custom_message():
    setup = _Setup(
        active_tools=["read", "bash", "edit", "write", "echo_tool"],
        select_choice="Execute the plan (track progress)",
    )

    await setup.run_command("plan")
    await setup.trigger_agent_end("Plan:\n1. Inspect the current implementation\n2. Add a regression test")

    assert setup.active_tools == ["read", "bash", "edit", "write", "echo_tool"]
    execute = [message for message, _options in setup.messages if message["customType"] == "plan-mode-execute"]
    assert len(execute) == 1
    options = next(options for message, options in setup.messages if message["customType"] == "plan-mode-execute")
    assert options == {"triggerTurn": True, "deliverAs": "followUp"}
