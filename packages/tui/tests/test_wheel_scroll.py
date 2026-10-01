"""Mirror of pi tui test/wheel-scroll.test.ts."""

from pidrei_tui.wheel_scroll import WheelScrollAccelerator


def scroll(accelerator: WheelScrollAccelerator, times: list[float], direction: int = 1) -> list[int]:
    return [accelerator.next(direction, time) for time in times]


# #9758: fullscreen wheel scrolling was one line per notch on terminals that do not accelerate wheels.


def test_uses_fixed_line_counts_regardless_of_timing():
    accelerator = WheelScrollAccelerator(3, True)
    assert scroll(accelerator, [0, 10, 20, 1000]) == [3, 3, 3, 3]
    accelerator.set_lines(0.5)
    assert accelerator.next(1, 2000) == 1


def test_keeps_one_line_per_event_in_auto_mode_when_the_terminal_already_accelerates():
    accelerator = WheelScrollAccelerator("auto", False)
    assert scroll(accelerator, [0, 10, 20, 30]) == [1, 1, 1, 1]


def test_scales_auto_mode_with_wheel_velocity():
    accelerator = WheelScrollAccelerator("auto", True)
    assert scroll(accelerator, [0, 150, 300, 450]) == [1, 1, 1, 1]
    assert scroll(accelerator, [1000, 1050, 1100, 1150]) == [1, 2, 2, 2]
    assert scroll(accelerator, [2000, 2020, 2040, 2060]) == [1, 5, 5, 5]
    assert scroll(accelerator, [3000, 3010, 3020, 3030]) == [1, 6, 6, 6]


def test_does_not_accelerate_bursts_of_events_for_a_single_notch():
    accelerator = WheelScrollAccelerator("auto", True)
    assert scroll(accelerator, [0, 3, 6, 9]) == [1, 1, 1, 1]


def test_resets_acceleration_on_direction_changes_and_pauses():
    accelerator = WheelScrollAccelerator("auto", True)
    assert scroll(accelerator, [0, 20, 40]) == [1, 5, 5]
    assert accelerator.next(-1, 60) == 1
    assert scroll(accelerator, [500, 520]) == [1, 5]


def test_carries_fractional_lines_between_events():
    accelerator = WheelScrollAccelerator("auto", True)
    assert scroll(accelerator, [0, 40, 80, 120, 160]) == [1, 2, 3, 2, 3]
