"""Mirror of pi tui test/program-status.test.ts (pi #10607)."""

import base64
import re

from pidrei_tui.program_status import ProgramStatus, format_program_status, is_program_status_reply


def decode_message(sequence: str) -> str | None:
    match = re.search(r":msg=([A-Za-z0-9+/=]*)", sequence)
    return None if match is None else base64.b64decode(match.group(1)).decode("utf-8")


def test_encodes_state_app_kind_and_a_base64_message():
    message = base64.b64encode(b"Allow bash?").decode()
    assert (
        format_program_status(ProgramStatus(state="blocked", app="pi", kind="permission", message="Allow bash?"))
        == f"\x1b]7501;state=blocked:app=pi:kind=permission:msg={message}\x1b\\"
    )
    assert format_program_status(ProgramStatus(state="clear")) == "\x1b]7501;state=clear\x1b\\"


def test_omits_kind_outside_blocked_invalid_app_names_and_empty_messages():
    assert (
        format_program_status(ProgramStatus(state="working", app="my app", kind="auth", message=" \n "))
        == "\x1b]7501;state=working\x1b\\"
    )
    assert re.search(r":app=a{32}\x1b", format_program_status(ProgramStatus(state="idle", app="a" * 32)))
    assert "app=" not in format_program_status(ProgramStatus(state="idle", app="a" * 33))


def test_replaces_control_characters_which_make_terminals_discard_the_report():
    sequence = format_program_status(ProgramStatus(state="error", message="first\nsecond\x1b[31m\u009bthird\t"))
    assert decode_message(sequence) == "first second [31m third"


def test_cuts_long_messages_at_a_utf8_boundary_within_the_spec_limits():
    sequence = format_program_status(ProgramStatus(state="working", app="pi", message="é" * 2000))
    message = decode_message(sequence)
    assert message == "é" * 1024
    assert len(message.encode()) <= 2048
    assert len(sequence) <= 4096


def test_accepts_the_query_echo_with_either_terminator_and_future_pairs():
    assert is_program_status_reply("\x1b]7501;?\x1b\\") is True
    assert is_program_status_reply("\x1b]7501;?\x07") is True
    assert is_program_status_reply("\x1b]7501;?version=2\x1b\\") is True
    assert is_program_status_reply("\x1b]7501;state=idle\x1b\\") is False
    assert is_program_status_reply("\x1b]11;rgb:0000/0000/0000\x07") is False
