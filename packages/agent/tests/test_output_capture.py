"""Mirror of pi agent/test/harness/output-capture.test.ts."""

from pidrei_agent.harness.types import ShellOutputCaptureOptions, ShellOutputLimits, ShellOutputUpdate, ShellOutputView
from pidrei_agent.harness.utils.output_capture import OutputCapture, apply_shell_output_update, sanitize_shell_output

from .fake_timers import fake_timers


def create_capture(*, max_bytes: int = 50, max_lines: int = 100, retain: str = "tail"):
    updates: list[ShellOutputUpdate] = []
    errors: list[Exception] = []
    capture = OutputCapture(
        ShellOutputCaptureOptions(limits=ShellOutputLimits(max_bytes=max_bytes, max_lines=max_lines, retain=retain)),
        None,
        on_update=lambda update, _cancel: updates.append(update),
        on_error=errors.append,
    )
    return capture, updates, errors


def fold(updates: list[ShellOutputUpdate]) -> ShellOutputView | None:
    output: ShellOutputView | None = None
    for update in updates:
        output = apply_shell_output_update(output, update)
    return output


def test_removes_invalid_control_characters_without_changing_text_or_line_boundaries():
    with fake_timers():
        text = "a\0b\tc\nd\ref￹g￻h😀"
        assert sanitize_shell_output(text) == "ab\tc\ndefgh😀"
        capture, _updates, _errors = create_capture()
        capture.push(text)
        assert capture.snapshot().text == "ab\tc\ndefgh😀"


def test_decodes_utf8_split_across_raw_process_chunks():
    with fake_timers():
        capture, _updates, _errors = create_capture()
        encoded = "😀".encode()
        capture.push(encoded[:2])
        assert capture.snapshot().text == ""
        capture.push(encoded[2:])
        capture.finish()
        assert capture.snapshot().text == "😀"


def test_publishes_the_first_bounded_view_immediately_and_trickling_appends_responsively():
    with fake_timers() as timers:
        capture, updates, _errors = create_capture()

        capture.push("one")
        assert len(updates) == 1
        assert updates[0].kind == "replace"

        timers.advance(150)
        capture.push(" two")
        assert len(updates) == 2
        assert updates[1].kind == "append"
        assert updates[1].text == " two"
        assert fold(updates).text == "one two"


def test_collapses_a_burst_into_one_trailing_update():
    with fake_timers() as timers:
        capture, updates, _errors = create_capture()
        capture.push("a")
        capture.push("b")
        capture.push("c")
        assert len(updates) == 1

        timers.advance(100)
        assert len(updates) == 2
        assert updates[1].kind == "append"
        assert updates[1].text == "bc"
        assert fold(updates).text == "abc"


def test_publishes_a_small_slide_for_post_cap_trickle():
    with fake_timers() as timers:
        capture, updates, _errors = create_capture(max_bytes=10)
        capture.push("abcdefghij")
        timers.advance(150)
        capture.push("k")

        assert updates[1].kind == "slide"
        assert updates[1].drop == 1
        assert updates[1].text == "k"
        assert fold(updates).text == "bcdefghijk"
        assert fold(updates).truncation.total_bytes == 11


def test_keeps_the_exact_byte_count_for_a_single_line_larger_than_its_working_buffer():
    with fake_timers():
        capture, _updates, _errors = create_capture(max_bytes=10)
        capture.push("x" * 100)
        snapshot = capture.snapshot()
        assert snapshot.text == "x" * 10
        assert snapshot.last_line_bytes == 100
        assert snapshot.truncation.last_line_partial is True


def test_uses_a_cap_bounded_replacement_after_complete_turnover():
    with fake_timers() as timers:
        capture, updates, _errors = create_capture(max_bytes=10)
        capture.push("abcdefghij")
        capture.push("x" * 100)
        timers.advance(100)

        assert updates[1].kind == "replace"
        assert len(fold(updates).text) == 10
        assert fold(updates).truncation.total_bytes == 110


def test_forces_held_state_and_cancels_its_trailing_timer_on_dispose():
    with fake_timers() as timers:
        capture, updates, _errors = create_capture()
        capture.push("a")
        capture.push("b")
        capture.flush()
        assert fold(updates).text == "ab"
        capture.dispose()
        timers.advance(1_000)
        assert len(updates) == 2


def test_preserves_the_original_head_after_its_raw_guard_is_crossed():
    with fake_timers():
        capture, _updates, _errors = create_capture(max_bytes=100, max_lines=2, retain="head")
        capture.push(f"first\nsecond\n{'tail' * 100}")
        assert capture.snapshot().text == "first\nsecond"


def test_publishes_spill_metadata_without_resending_text():
    with fake_timers():
        capture, updates, errors = create_capture()
        capture.push("output")
        capture.set_spill_path("/tmp/output.log")
        assert updates[-1].kind == "metadata"
        assert updates[-1].metadata.spill_path == "/tmp/output.log"
        assert fold(updates).spill_path == "/tmp/output.log"
        assert errors == []
