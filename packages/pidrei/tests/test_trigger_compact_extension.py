"""Mirror of pi's trigger-compact-extension.test.ts."""

from types import SimpleNamespace

from .example_extensions import load_example


def create_context(tokens: int | None, compact) -> SimpleNamespace:
    return SimpleNamespace(
        mode="print",
        has_ui=False,
        ui=SimpleNamespace(),
        cwd=".",
        session_manager=SimpleNamespace(),
        model_registry=SimpleNamespace(),
        model=None,
        is_idle=lambda: True,
        is_project_trusted=lambda: True,
        signal=None,
        abort=lambda: None,
        has_pending_messages=lambda: False,
        shutdown=lambda: None,
        get_context_usage=lambda: SimpleNamespace(
            tokens=tokens,
            context_window=200_000,
            percent=None if tokens is None else tokens / 2000,
        ),
        compact=compact,
        get_system_prompt=lambda: "",
    )


def test_only_auto_compacts_when_context_usage_crosses_the_threshold():
    handlers: dict = {}
    commands: list[str] = []

    api = SimpleNamespace(
        on=lambda event, handler: handlers.__setitem__(event, handler),
        register_command=lambda name, **_options: commands.append(name),
    )
    load_example("trigger_compact").extension(api)

    assert "turn_end" in handlers
    assert commands == ["trigger-compact"]

    calls: list = []
    compact = calls.append
    event = {"type": "turn_end"}

    handlers["turn_end"](event, create_context(110_000, compact))
    assert calls == []

    handlers["turn_end"](event, create_context(120_000, compact))
    assert calls == []

    handlers["turn_end"](event, create_context(95_000, compact))
    assert calls == []

    handlers["turn_end"](event, create_context(105_000, compact))
    assert len(calls) == 1
