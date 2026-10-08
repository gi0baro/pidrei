"""Mirror of pi tui test/terminal-image.test.ts and
test/bug-regression-isimageline-startswith-bug.test.ts.

The Image-component cases ("caps Image component height...", "places image
sequence on first line...", tool-output integration) land with the
components slice.
"""

import base64
import contextlib
import os
import re
import threading

import pytest
import tonio.colored as tonio

from pidrei_tui import ImageConversions, terminal_image
from pidrei_tui.components.image import Image
from pidrei_tui.terminal_image import (
    crop_kitty_image_line,
    delete_all_kitty_images,
    delete_all_kitty_placements,
    delete_kitty_image,
    detect_capabilities,
    encode_iterm2,
    encode_kitty,
    get_capabilities,
    get_kitty_image_metadata,
    get_kitty_image_placement,
    get_kitty_image_placement_rows,
    hyperlink,
    image_fallback,
    is_image_line,
    prime_capabilities,
    register_kitty_image_metadata,
    render_image,
    reset_capabilities_cache,
    set_capabilities,
    set_capability_overrides,
    set_cell_dimensions,
)
from pidrei_tui.utils import visible_width

from .tui_helpers import env_var


class _FakeTui:
    """What `Image` uses of its TUI. `request_render` counts the frames the
    conversions asked for; `wait_for_renders(n)` waits until `n` were."""

    def __init__(self) -> None:
        self.state_lock = threading.RLock()
        self.image_conversions = ImageConversions()
        self.errors: list[BaseException] = []
        self._renders = 0
        self._target: tuple[int, tonio.Event] | None = None

    def apply(self, fn):
        with self.state_lock:
            return fn()

    def request_render(self, force: bool = False) -> None:
        with self.state_lock:
            self._renders += 1
            if self._target is not None and self._renders >= self._target[0]:
                self._target[1].set()

    def report_error(self, error: BaseException) -> None:
        self.errors.append(error)

    def render(self, component, width: int) -> list[str]:
        """Render under the state lock, as the TUI's frames do: a conversion
        stores its result through `apply` from another thread meanwhile."""
        with self.state_lock:
            return component.render(width)

    async def wait_for_renders(self, count: int) -> None:
        event = tonio.Event()
        with self.state_lock:
            if self._renders >= count:
                return
            self._target = (count, event)
        await event.wait(5)
        assert event.is_set(), f"expected {count} render requests"


_TUI = _FakeTui()


_ENV_KEYS = (
    "TERM",
    "TERM_PROGRAM",
    "TERMINAL_EMULATOR",
    "COLORTERM",
    "TMUX",
    "KITTY_WINDOW_ID",
    "GHOSTTY_RESOURCES_DIR",
    "WEZTERM_PANE",
    "ITERM_SESSION_ID",
    "WT_SESSION",
    "CMUX_WORKSPACE_ID",
    "WARP_SESSION_ID",
    "WARP_TERMINAL_SESSION_UUID",
    "PIDREI_HYPERLINKS",
    "PIDREI_IMAGE_PROTOCOL",
    "PIDREI_TRUE_COLOR",
)


@contextlib.contextmanager
def clean_env(overrides: dict | None = None):
    """Clear all detection-relevant env vars, then apply overrides."""
    with contextlib.ExitStack() as stack:
        for key in _ENV_KEYS:
            stack.enter_context(env_var(key, None))
        for key, value in (overrides or {}).items():
            stack.enter_context(env_var(key, value))
        yield


# isImageLine — iTerm2 image protocol


def test_detects_iterm2_image_escape_sequence_at_start_of_line():
    iterm2_image_line = "\x1b]1337;File=size=100,100;inline=1:base64encodeddata==\x07"
    assert is_image_line(iterm2_image_line) is True


def test_detects_iterm2_image_escape_sequence_with_text_before_it():
    line_with_text_and_image = "Some text \x1b]1337;File=size=100,100;inline=1:base64data==\x07 more text"
    assert is_image_line(line_with_text_and_image) is True


def test_detects_iterm2_image_escape_sequence_in_middle_of_long_line():
    long_line_with_image = "Text before image..." + "\x1b]1337;File=inline=1:verylongbase64data==" + "...text after"
    assert is_image_line(long_line_with_image) is True


def test_detects_iterm2_image_escape_sequence_at_end_of_line():
    line_with_image_at_end = "Regular text ending with \x1b]1337;File=inline=1:base64data==\x07"
    assert is_image_line(line_with_image_at_end) is True


def test_detects_minimal_iterm2_image_escape_sequence():
    minimal_image_line = "\x1b]1337;File=:\x07"
    assert is_image_line(minimal_image_line) is True


# isImageLine — Kitty image protocol


def test_detects_kitty_image_escape_sequence_at_start_of_line():
    kitty_image_line = "\x1b_Ga=T,f=100,t=f,d=base64data...\x1b\\\x1b_Gm=i=1;\x1b\\"
    assert is_image_line(kitty_image_line) is True


def test_detects_kitty_image_escape_sequence_with_text_before_it():
    line_with_text_and_kitty_image = "Output: \x1b_Ga=T,f=100;data...\x1b\\\x1b_Gm=i=1;\x1b\\"
    assert is_image_line(line_with_text_and_kitty_image) is True


def test_detects_kitty_image_escape_sequence_with_padding():
    kitty_with_padding = "  \x1b_Ga=T,f=100...\x1b\\\x1b_Gm=i=1;\x1b\\  "
    assert is_image_line(kitty_with_padding) is True


# isImageLine — bug regression


def test_detects_image_sequences_in_very_long_lines():
    base64_char = "A" * 100
    image_sequence = "\x1b]1337;File=size=800,600;inline=1:"

    long_line = "Text prefix " + image_sequence + base64_char * 3000 + " suffix"

    assert (len(long_line) > 300000) is True
    assert is_image_line(long_line) is True


def test_detects_image_sequences_when_terminal_does_not_support_images():
    line_with_image = "Read image file [image/jpeg]\x1b]1337;File=inline=1:base64data==\x07"
    assert is_image_line(line_with_image) is True


def test_detects_image_sequences_with_ansi_codes_before_them():
    line_with_ansi_and_image = "\x1b[31mError output \x1b]1337;File=inline=1:image==\x07"
    assert is_image_line(line_with_ansi_and_image) is True


def test_detects_image_sequences_with_ansi_codes_after_them():
    line_with_image_and_ansi = "\x1b_Ga=T,f=100:data...\x1b\\\x1b_Gm=i=1;\x1b\\\x1b[0m reset"
    assert is_image_line(line_with_image_and_ansi) is True


# isImageLine — negative cases


def test_does_not_detect_images_in_plain_text_lines():
    assert is_image_line("This is just a regular text line without any escape sequences") is False


def test_does_not_detect_images_in_lines_with_only_ansi_codes():
    assert is_image_line("\x1b[31mRed text\x1b[0m and \x1b[32mgreen text\x1b[0m") is False


def test_does_not_detect_images_in_lines_with_cursor_movement_codes():
    assert is_image_line("\x1b[1A\x1b[2KLine cleared and moved up") is False


def test_does_not_detect_images_in_lines_with_partial_iterm2_sequences():
    assert is_image_line("Some text with ]1337;File but missing ESC at start") is False


def test_does_not_detect_images_in_lines_with_partial_kitty_sequences():
    assert is_image_line("Some text with _G but missing ESC at start") is False


def test_does_not_detect_images_in_empty_lines():
    assert is_image_line("") is False


def test_does_not_detect_images_in_lines_with_newlines_only():
    assert is_image_line("\n") is False
    assert is_image_line("\n\n") is False


# isImageLine — mixed content


def test_detects_images_when_line_has_both_kitty_and_iterm2_sequences():
    mixed_line = "Kitty: \x1b_Ga=T...\x1b\\\x1b_Gm=i=1;\x1b\\ iTerm2: \x1b]1337;File=inline=1:data==\x07"
    assert is_image_line(mixed_line) is True


def test_detects_image_in_line_with_multiple_text_and_image_segments():
    complex_line = "Start \x1b]1337;File=img1==\x07 middle \x1b]1337;File=img2==\x07 end"
    assert is_image_line(complex_line) is True


def test_does_not_falsely_detect_image_in_line_with_file_path_containing_keywords():
    assert is_image_line("/path/to/File_1337_backup/image.jpg") is False


def test_detects_kitty_sequences_in_any_position():
    scenarios = [
        "At start: \x1b_Ga=T,f=100,data...\x1b\\",
        "Prefix \x1b_Ga=T,data...\x1b\\",
        "Suffix text \x1b_Ga=T,data...\x1b\\ suffix",
        "Middle \x1b_Ga=T,data...\x1b\\ more text",
        # Very long line (simulating 300KB+ crash scenario)
        "Text before \x1b_Ga=T,f=100" + "A" * 300000 + " text after",
    ]
    for line in scenarios:
        assert is_image_line(line) is True, f"Should detect Kitty sequence in: {line[:50]}..."


def test_detects_iterm2_sequences_in_any_position():
    scenarios = [
        "At start: \x1b]1337;File=size=100,100:base64...\x07",
        "Prefix \x1b]1337;File=inline=1:data==\x07",
        "Suffix text \x1b]1337;File=inline=1:data==\x07 suffix",
        "Middle \x1b]1337;File=inline=1:data==\x07 more text",
        # Very long line (simulating 304KB crash scenario)
        "Text before \x1b]1337;File=size=800,600;inline=1:" + "B" * 300000 + " text after",
    ]
    for line in scenarios:
        assert is_image_line(line) is True, f"Should detect iTerm2 sequence in: {line[:50]}..."


def test_does_not_crash_on_very_long_lines_with_image_sequences():
    base64_char = "A" * 100
    iterm2_sequence = "\x1b]1337;File=size=800,600;inline=1:"

    crash_line = "Output: " + iterm2_sequence + base64_char * 3040 + " end of output"

    assert len(crash_line) > 300000, "Test line should be > 300KB"
    assert is_image_line(crash_line) is True


def test_handles_lines_exactly_matching_crash_log_dimensions():
    target_width = 58649
    prefix = "Text"
    sequence = "\x1b_Ga=T,f=100"
    suffix = "End"
    padding = "A" * (target_width - len(prefix) - len(sequence) - len(suffix))
    line = f"{prefix}{sequence}{padding}{suffix}"

    assert len(line) == 58649
    assert is_image_line(line) is True


def test_does_not_detect_images_in_regular_long_text():
    assert is_image_line("A" * 100000) is False


def test_does_not_detect_images_in_lines_with_file_paths():
    file_paths = [
        "/path/to/1337/image.jpg",
        "/usr/local/bin/File_converter",
        "~/Documents/1337File_backup.png",
        "./_G_test_file.txt",
    ]
    for path in file_paths:
        assert is_image_line(path) is False, f"Should not falsely detect image sequence in path: {path}"


# detectCapabilities


def test_defaults_to_hyperlinks_false_for_unknown_terminals():
    with clean_env():
        caps = detect_capabilities()
        assert caps["hyperlinks"] is False
        assert caps["images"] is None


def test_applies_environment_overrides():
    with clean_env({"PIDREI_HYPERLINKS": "1", "PIDREI_IMAGE_PROTOCOL": "kitty", "PIDREI_TRUE_COLOR": "1"}):
        assert detect_capabilities() == {"images": "kitty", "trueColor": True, "hyperlinks": True}
    with clean_env(
        {
            "TERM_PROGRAM": "iTerm.app",
            "PIDREI_HYPERLINKS": "0",
            "PIDREI_IMAGE_PROTOCOL": "none",
            "PIDREI_TRUE_COLOR": "0",
        }
    ):
        assert detect_capabilities() == {"images": None, "trueColor": False, "hyperlinks": False}


def test_preserves_auto_detection_for_auto_environment_overrides():
    with clean_env(
        {
            "TERM_PROGRAM": "ghostty",
            "PIDREI_HYPERLINKS": "auto",
            "PIDREI_IMAGE_PROTOCOL": "auto",
            "PIDREI_TRUE_COLOR": "auto",
        }
    ):
        assert detect_capabilities() == {"images": "kitty", "trueColor": True, "hyperlinks": True}


def test_applies_and_clears_programmatic_overrides():
    with clean_env({"PIDREI_HYPERLINKS": "1", "PIDREI_IMAGE_PROTOCOL": "kitty", "PIDREI_TRUE_COLOR": "1"}):
        set_capability_overrides({"images": None, "trueColor": False, "hyperlinks": False})
        try:
            assert get_capabilities() == {"images": None, "trueColor": False, "hyperlinks": False}
            set_capability_overrides({})
            assert get_capabilities() == {"images": "kitty", "trueColor": True, "hyperlinks": True}
        finally:
            set_capability_overrides({})
            reset_capabilities_cache()


def test_bypasses_the_tmux_probe_when_hyperlinks_are_overridden():
    probed = []

    def probe() -> bool:
        probed.append(True)
        return False

    with clean_env(
        {"TMUX": "/tmp/tmux-1000/default,1234,0", "PIDREI_HYPERLINKS": "1", "PIDREI_IMAGE_PROTOCOL": "kitty"}
    ):
        caps = detect_capabilities(probe)

    assert probed == []
    assert caps["hyperlinks"] is True
    assert caps["images"] == "kitty"


def test_enables_hyperlinks_under_tmux_when_the_client_forwards_them():
    with clean_env({"TMUX": "/tmp/tmux-1000/default,1234,0", "TERM_PROGRAM": "ghostty"}):
        caps = detect_capabilities(lambda: True)
        assert caps["hyperlinks"] is True
        assert caps["images"] is None


def test_disables_hyperlinks_under_tmux_when_the_client_does_not_forward_them():
    with clean_env({"TMUX": "/tmp/tmux-1000/default,1234,0", "TERM_PROGRAM": "ghostty"}):
        caps = detect_capabilities(lambda: False)
        assert caps["hyperlinks"] is False
        assert caps["images"] is None


def test_checks_tmux_capability_when_term_starts_with_tmux():
    with clean_env({"TERM": "tmux-256color", "TERM_PROGRAM": "iterm.app"}):
        caps = detect_capabilities(lambda: True)
        assert caps["hyperlinks"] is True
        assert caps["images"] is None

        caps2 = detect_capabilities(lambda: False)
        assert caps2["hyperlinks"] is False


def test_forces_hyperlinks_false_when_term_starts_with_screen():
    with clean_env({"TERM": "screen-256color"}):
        caps = detect_capabilities()
        assert caps["hyperlinks"] is False
        assert caps["images"] is None


# pi #10573
def test_enables_hyperlinks_without_images_for_herdr():
    with clean_env(
        {"TERM_PROGRAM": "herdr", "TERM": "xterm-256color", "COLORTERM": "truecolor", "KITTY_WINDOW_ID": "1"}
    ):
        caps = detect_capabilities()
        assert caps["hyperlinks"] is True
        assert caps["images"] is None
        assert caps["trueColor"] is True
    with clean_env({"TERM_PROGRAM": "herdr", "PIDREI_HYPERLINKS": "0"}):
        assert detect_capabilities()["hyperlinks"] is False


def test_enables_hyperlinks_for_ghostty():
    with clean_env({"TERM_PROGRAM": "ghostty"}):
        assert detect_capabilities()["hyperlinks"] is True


def test_does_not_disable_ghostty_images_solely_because_cmux_is_present():
    with clean_env({"TERM_PROGRAM": "ghostty", "CMUX_WORKSPACE_ID": "workspace"}):
        caps = detect_capabilities()
        assert caps["images"] == "kitty"
        assert caps["hyperlinks"] is True


def test_enables_hyperlinks_for_kitty():
    with clean_env({"KITTY_WINDOW_ID": "1"}):
        assert detect_capabilities()["hyperlinks"] is True


def test_enables_hyperlinks_for_wezterm():
    with clean_env({"WEZTERM_PANE": "0"}):
        assert detect_capabilities()["hyperlinks"] is True


def test_enables_images_and_hyperlinks_for_warp_via_term_program():
    with clean_env({"TERM_PROGRAM": "WarpTerminal"}):
        caps = detect_capabilities()
        assert caps["images"] == "kitty"
        assert caps["trueColor"] is True
        assert caps["hyperlinks"] is True


def test_enables_images_and_hyperlinks_for_warp_via_warp_session_id():
    with clean_env({"WARP_SESSION_ID": "some-session-id"}):
        caps = detect_capabilities()
        assert caps["images"] == "kitty"
        assert caps["trueColor"] is True
        assert caps["hyperlinks"] is True


def test_enables_images_and_hyperlinks_for_warp_via_warp_terminal_session_uuid():
    with clean_env({"WARP_TERMINAL_SESSION_UUID": "d0e1a2e5-7ca7-44cd-9037-ac7222011161"}):
        caps = detect_capabilities()
        assert caps["images"] == "kitty"
        assert caps["trueColor"] is True
        assert caps["hyperlinks"] is True


def test_disables_images_for_warp_inside_tmux():
    with clean_env(
        {
            "TERM_PROGRAM": "WarpTerminal",
            "TMUX": "/tmp/tmux-1000/default,1234,0",
            "TERM": "tmux-256color",
        }
    ):
        caps = detect_capabilities(lambda: True)
        assert caps["images"] is None
        assert caps["hyperlinks"] is True


def test_enables_hyperlinks_for_iterm2():
    with clean_env({"TERM_PROGRAM": "iterm.app"}):
        assert detect_capabilities()["hyperlinks"] is True


def test_enables_hyperlinks_for_vscode():
    with clean_env({"TERM_PROGRAM": "vscode"}):
        assert detect_capabilities()["hyperlinks"] is True


def test_enables_alacritty_capabilities_for_zed():
    with clean_env({"TERM_PROGRAM": "zed"}):
        assert detect_capabilities() == {"images": None, "trueColor": True, "hyperlinks": True}


def test_enables_truecolor_and_hyperlinks_for_windows_terminal_outside_multiplexers():
    with clean_env({"WT_SESSION": "session", "TERM": "xterm-256color"}):
        caps = detect_capabilities()
        assert caps["trueColor"] is True
        assert caps["hyperlinks"] is True
        assert caps["images"] is None


def test_enables_truecolor_without_hyperlinks_for_jetbrains_terminal():
    with clean_env({"TERMINAL_EMULATOR": "JetBrains-JediTerm", "TERM": "xterm-256color"}):
        caps = detect_capabilities()
        assert caps["trueColor"] is True
        assert caps["hyperlinks"] is False
        assert caps["images"] is None


def test_does_not_inherit_windows_terminal_truecolor_through_tmux():
    with clean_env({"WT_SESSION": "session", "TMUX": "/tmp/tmux-1000/default,1234,0", "TERM": "tmux-256color"}):
        caps = detect_capabilities(lambda: False)
        assert caps["trueColor"] is False
        assert caps["hyperlinks"] is False
        assert caps["images"] is None


def test_trusts_explicit_truecolor_hints_through_tmux():
    with clean_env({"COLORTERM": "truecolor", "TMUX": "/tmp/tmux-1000/default,1234,0", "TERM": "tmux-256color"}):
        caps = detect_capabilities(lambda: False)
        assert caps["trueColor"] is True
        assert caps["hyperlinks"] is False
        assert caps["images"] is None


def test_detects_truecolor_from_direct_color_term_values():
    with clean_env({"TERM": "xterm-direct"}):
        assert detect_capabilities(lambda: False)["trueColor"] is True


# Kitty image cursor movement


# iTerm2 image encoding


def test_includes_the_decoded_payload_size_in_osc_1337_metadata():
    sequence = encode_iterm2("AAAA", width=2, height="auto")
    assert sequence == "\x1b]1337;File=inline=1;size=3;width=2;height=auto:AAAA\x07"


# Kitty image cursor movement


def test_can_request_no_terminal_side_cursor_movement():
    sequence = encode_kitty("AAAA", columns=2, rows=2, move_cursor=False)
    assert sequence.startswith("\x1b_Ga=T,f=100,q=2,C=1,c=2,r=2;")


def test_reads_explicit_placement_rows_without_registered_metadata():
    sequence = encode_kitty("AAAA", columns=2, rows=3, move_cursor=False)
    assert get_kitty_image_placement_rows(sequence) == 3


def test_suppresses_kitty_replies_for_delete_commands():
    assert delete_kitty_image(42) == "\x1b_Ga=d,d=I,i=42,q=2\x1b\\"
    assert delete_all_kitty_images() == "\x1b_Ga=d,d=A,q=2\x1b\\"
    assert delete_all_kitty_placements() == "\x1b_Ga=d,d=a,q=2\x1b\\"


def test_preserves_render_image_default_terminal_side_cursor_movement():
    set_capabilities({"images": "kitty", "trueColor": True, "hyperlinks": True})
    set_cell_dimensions({"widthPx": 10, "heightPx": 10})
    try:
        result = render_image("AAAA", {"widthPx": 20, "heightPx": 20}, max_width_cells=2)
        assert result
        assert ",C=1," not in result["sequence"]
        assert result["rows"] == 2
    finally:
        reset_capabilities_cache()
        set_cell_dimensions({"widthPx": 9, "heightPx": 18})


def test_can_opt_render_image_into_no_terminal_side_cursor_movement():
    set_capabilities({"images": "kitty", "trueColor": True, "hyperlinks": True})
    set_cell_dimensions({"widthPx": 10, "heightPx": 10})
    try:
        result = render_image("AAAA", {"widthPx": 20, "heightPx": 20}, max_width_cells=2, move_cursor=False)
        assert result
        assert ",C=1," in result["sequence"]
        assert result["rows"] == 2
    finally:
        reset_capabilities_cache()
        set_cell_dimensions({"widthPx": 9, "heightPx": 18})


def test_registers_metadata_and_crops_a_partially_visible_placement():
    set_capabilities({"images": "kitty", "trueColor": True, "hyperlinks": True})
    set_cell_dimensions({"widthPx": 10, "heightPx": 10})
    try:
        result = render_image(
            "AAAA", {"widthPx": 100, "heightPx": 100}, max_width_cells=3, image_id=42, move_cursor=False
        )
        assert result
        assert get_kitty_image_metadata(result["sequence"]) == {
            "imageId": 42,
            "columns": 3,
            "rows": 3,
            "widthPx": 100,
            "heightPx": 100,
        }
        assert "y=66,h=34,r=1" in crop_kitty_image_line(result["sequence"], 2, 1)
    finally:
        reset_capabilities_cache()
        set_cell_dimensions({"widthPx": 9, "heightPx": 18})


def test_creates_placement_only_commands_for_uploaded_and_cropped_images():
    register_kitty_image_metadata({"imageId": 42, "columns": 3, "rows": 3, "widthPx": 100, "heightPx": 100})
    transmission = encode_kitty("A" * 8192, columns=3, rows=3, image_id=42, move_cursor=False)
    line = f"left {crop_kitty_image_line(transmission, 2, 1)} right"
    placement = get_kitty_image_placement(line)
    assert placement
    assert get_kitty_image_placement_rows(line) == 1
    assert placement["transmissionBytes"] == len(line) - len("left ") - len(" right")
    assert placement["estimatedDecodedBytes"] == 100 * 100 * 4
    assert placement["rows"] == 1
    assert placement["sequence"] == "\x1b_Ga=p,q=2,C=1,c=3,i=42,y=66,h=34,r=1\x1b\\"
    assert placement["replacementLine"] == f"left {placement['sequence']} right"
    assert "AAAA" not in placement["replacementLine"]


def test_honors_max_height_cells_by_reducing_rendered_width():
    set_capabilities({"images": "kitty", "trueColor": True, "hyperlinks": True})
    set_cell_dimensions({"widthPx": 10, "heightPx": 10})
    try:
        result = render_image("AAAA", {"widthPx": 10, "heightPx": 100}, max_width_cells=10, max_height_cells=5)
        assert result
        assert result["rows"] == 5
        assert ",c=1,r=5" in result["sequence"]
    finally:
        reset_capabilities_cache()
        set_cell_dimensions({"widthPx": 9, "heightPx": 18})


def test_caps_image_component_height_to_a_square_pixel_box_by_default():
    set_capabilities({"images": "kitty", "trueColor": True, "hyperlinks": True})
    set_cell_dimensions({"widthPx": 10, "heightPx": 20})
    try:
        image = Image(
            _TUI,
            "AAAA",
            "image/png",
            {"fallbackColor": lambda value: value},
            {"maxWidthCells": 10},
            {"widthPx": 10, "heightPx": 100},
        )
        lines = image.render(12)
        assert len(lines) == 5
        assert ",c=1,r=5" in lines[0]
    finally:
        reset_capabilities_cache()
        set_cell_dimensions({"widthPx": 9, "heightPx": 18})


def test_places_image_sequence_on_first_line_with_empty_padding_rows():
    set_capabilities({"images": "kitty", "trueColor": True, "hyperlinks": True})
    set_cell_dimensions({"widthPx": 10, "heightPx": 10})
    try:
        image = Image(
            _TUI,
            "AAAA",
            "image/png",
            {"fallbackColor": lambda value: value},
            {"maxWidthCells": 2},
            {"widthPx": 20, "heightPx": 20},
        )
        lines = image.render(4)
        image_id = image.get_image_id()
        assert isinstance(image_id, int)
        assert lines[0].startswith("\x1b_G")
        assert ",C=1," in lines[0]
        assert f",i={image_id}" in lines[0]
        assert lines[0].endswith("\x1b\\")
        assert lines[1:] == [""]
    finally:
        reset_capabilities_cache()
        set_cell_dimensions({"widthPx": 9, "heightPx": 18})


# image cell sizing — #8938: reduce Kitty placement distortion without
# shrinking iTerm2 reservations.


@contextlib.contextmanager
def _image_caps(images: str, cell_dimensions: dict):
    set_capabilities({"images": images, "trueColor": True, "hyperlinks": True})
    set_cell_dimensions(cell_dimensions)
    try:
        yield
    finally:
        reset_capabilities_cache()
        set_cell_dimensions({"widthPx": 9, "heightPx": 18})


def _sized_image(max_width_cells: int, image_id: int | None, dimensions: dict) -> Image:
    options: dict = {"maxWidthCells": max_width_cells}
    if image_id is not None:
        options["imageId"] = image_id
    return Image(_TUI, "AAAA", "image/png", {"fallbackColor": lambda value: value}, options, dimensions)


def test_reserves_at_least_one_kitty_row_for_thin_images():
    with _image_caps("kitty", {"widthPx": 9, "heightPx": 18}):
        result = render_image("AAAA", {"widthPx": 1200, "heightPx": 12}, max_width_cells=60)
        assert result
        assert result["rows"] == 1
        assert ",c=60,r=1;" in result["sequence"]


def test_keeps_kitty_placement_reserved_lines_and_cropping_metadata_consistent_across_width_changes():
    with _image_caps("kitty", {"widthPx": 9, "heightPx": 18}):
        image = _sized_image(60, 8938, {"widthPx": 615, "heightPx": 86})
        lines = image.render(62)
        assert len(lines) == 4
        assert lines[1:] == ["", "", ""]
        assert ",c=60,r=4,i=8938;" in lines[0]
        assert get_kitty_image_metadata(lines[0]) == {
            "imageId": 8938,
            "columns": 60,
            "rows": 4,
            "widthPx": 615,
            "heightPx": 86,
        }
        cropped = crop_kitty_image_line(lines[0], 1, 2)
        assert get_kitty_image_placement(cropped)["sequence"] == "\x1b_Ga=p,q=2,C=1,c=60,i=8938,y=21,h=44,r=2\x1b\\"

        narrower_lines = image.render(32)
        assert len(narrower_lines) == 2
        assert ",c=30,r=2,i=8938;" in narrower_lines[0]
        assert get_kitty_image_metadata(narrower_lines[0])["rows"] == 2


def test_keeps_the_ceiling_placement_when_rounding_down_would_increase_distortion():
    with _image_caps("kitty", {"widthPx": 15, "heightPx": 28}):
        result = render_image("AAAA", {"widthPx": 615, "heightPx": 86}, max_width_cells=60)
        assert result
        assert result["rows"] == 5
        assert ",c=60,r=5;" in result["sequence"]


def test_keeps_height_limited_kitty_columns_reservations_and_crop_metadata_consistent():
    with _image_caps("kitty", {"widthPx": 14, "heightPx": 28}):
        image = _sized_image(30, 8938, {"widthPx": 400, "heightPx": 900})
        lines = image.render(32)
        assert len(lines) == 15
        assert ",c=13,r=15,i=8938;" in lines[0]
        assert get_kitty_image_metadata(lines[0]) == {
            "imageId": 8938,
            "columns": 13,
            "rows": 15,
            "widthPx": 400,
            "heightPx": 900,
        }
        assert (
            get_kitty_image_placement(crop_kitty_image_line(lines[0], 1, 2))["sequence"]
            == "\x1b_Ga=p,q=2,C=1,c=13,i=8938,y=60,h=120,r=2\x1b\\"
        )
        narrower_lines = image.render(22)
        assert len(narrower_lines) == 10
        assert ",c=9,r=10,i=8938;" in narrower_lines[0]


def test_chooses_thin_kitty_widths_by_proportions_while_keeping_at_least_one_column():
    with _image_caps("kitty", {"widthPx": 1, "heightPx": 1}):
        for width_px, columns in [(1, 1), (140, 1), (149, 2)]:
            result = render_image(
                "AAAA", {"widthPx": width_px, "heightPx": 1000}, max_width_cells=30, max_height_cells=10
            )
            assert result
            assert result["columns"] == columns
            assert result["rows"] == 10
            assert f",c={columns},r=10;" in result["sequence"]


def test_keeps_iterm2_ceiling_width_when_height_limited():
    with _image_caps("iterm2", {"widthPx": 14, "heightPx": 28}):
        result = render_image("AAAA", {"widthPx": 400, "heightPx": 900}, max_width_cells=30, max_height_cells=15)
        assert result
        assert result["columns"] == 14
        assert result["rows"] == 15
        assert result["sequence"] == "\x1b]1337;File=inline=1;size=3;width=14;height=auto:AAAA\x07"


def test_keeps_iterm2_ceiling_based_reserved_lines_and_cursor_offset():
    with _image_caps("iterm2", {"widthPx": 9, "heightPx": 18}):
        image = _sized_image(60, None, {"widthPx": 615, "heightPx": 86})
        assert image.render(62) == [
            "",
            "",
            "",
            "",
            "\x1b[4A\x1b]1337;File=inline=1;size=3;width=60;height=auto:AAAA\x07",
        ]


# Image transcoding — Kitty only accepts PNG (f=100); non-PNG images must be
# transcoded (#10292). Pi's transcoder runs during render; here the TUI's
# converter is async, so the first render shows the fallback and each case
# waits for the conversion's render request before rendering again.

_JPEG = base64.b64encode(b"jpeg").decode()
# Minimal PNG header (signature + IHDR) for a 40x10 image. Enough for get_png_dimensions.
_PNG = base64.b64encode(bytes.fromhex("89504e470d0a1a0a0000000d49484452000000280000000a")).decode()


@pytest.fixture
def transcoding():
    set_capabilities({"images": "kitty", "trueColor": True, "hyperlinks": True})
    set_cell_dimensions({"widthPx": 10, "heightPx": 10})
    calls: list[str] = []

    async def transcode(data: str, _mime_type: str) -> str | None:
        calls.append(data)
        return _PNG if data == _JPEG else None

    try:
        yield calls, transcode
    finally:
        reset_capabilities_cache()
        set_cell_dimensions({"widthPx": 9, "heightPx": 18})


def _image(tui, data: str, mime_type: str, dimensions: dict | None = None) -> Image:
    return Image(tui, data, mime_type, {"fallbackColor": lambda value: value}, {}, dimensions)


def _is_fallback(line: str) -> bool:
    return re.match(r"^\[Image: \[image/jpeg\]", line) is not None


@pytest.mark.tonio
async def test_sends_converted_png_data_sized_from_the_png(transcoding):
    _calls, transcode = transcoding
    tui = _FakeTui()
    tui.image_conversions.set_converter(transcode)
    image = _image(tui, _JPEG, "image/jpeg", {"widthPx": 20, "heightPx": 20})
    assert _is_fallback(tui.render(image, 80)[0])
    await tui.wait_for_renders(1)
    lines = tui.render(image, 20)
    assert "f=100" in lines[0] and f";{_PNG}\x1b\\" in lines[0]
    # 40x10 PNG at 18 columns: 5 rows, not the 18 rows of the 20x20 source dimensions.
    assert len(lines) == 5


@pytest.mark.tonio
async def test_renders_a_text_fallback_until_a_working_converter_is_set(transcoding):
    _calls, transcode = transcoding
    tui = _FakeTui()
    image = _image(tui, _JPEG, "image/jpeg")
    assert _is_fallback(tui.render(image, 80)[0])

    async def failing(_data: str, _mime_type: str) -> None:
        return None

    tui.image_conversions.set_converter(failing)
    image.invalidate()
    assert _is_fallback(tui.render(image, 80)[0])
    await tui.wait_for_renders(1)
    assert _is_fallback(tui.render(image, 80)[0])

    tui.image_conversions.set_converter(transcode)
    image.invalidate()
    assert _is_fallback(tui.render(image, 80)[0])
    await tui.wait_for_renders(2)
    assert "\x1b_G" in tui.render(image, 80)[0]


@pytest.mark.tonio
async def test_converts_each_image_once(transcoding):
    calls, transcode = transcoding
    tui = _FakeTui()
    tui.image_conversions.set_converter(transcode)
    image = _image(tui, _JPEG, "image/jpeg")
    tui.render(image, 80)
    await tui.wait_for_renders(1)
    # A new instance hits the shared cache.
    assert "\x1b_G" in tui.render(_image(tui, _JPEG, "image/jpeg"), 80)[0]
    # Evicts the shared entry.
    for index in range(40):
        tui.render(_image(tui, f"other-{index}", "image/jpeg"), 80)
    await tui.wait_for_renders(41)
    image.invalidate()
    # The instance keeps its own PNG.
    assert "\x1b_G" in tui.render(image, 40)[0]
    assert calls.count(_JPEG) == 1


def test_does_not_convert_png_data_or_iterm2_output(transcoding):
    calls, transcode = transcoding
    tui = _FakeTui()
    tui.image_conversions.set_converter(transcode)
    assert f";{_PNG}\x1b\\" in tui.render(_image(tui, _PNG, "image/png", {"widthPx": 20, "heightPx": 20}), 20)[0]
    set_capabilities({"images": "iterm2", "trueColor": True, "hyperlinks": True})
    iterm2_lines = tui.render(_image(tui, _JPEG, "image/jpeg", {"widthPx": 20, "heightPx": 20}), 20)
    assert iterm2_lines[-1].endswith(f":{_JPEG}\x07")
    assert calls == []


@pytest.mark.tonio
async def test_reports_a_converter_that_raises_and_keeps_the_fallback(transcoding):
    # pidrei-only: pi's transcoder runs inside render; here its failure goes to the TUI's error handler.
    tui = _FakeTui()

    async def raising(_data: str, _mime_type: str) -> str:
        raise RuntimeError("converter exploded")

    tui.image_conversions.set_converter(raising)
    image = _image(tui, _JPEG, "image/jpeg")
    assert _is_fallback(tui.render(image, 80)[0])
    await tui.wait_for_renders(1)
    assert [str(error) for error in tui.errors] == ["converter exploded"]
    image.invalidate()
    assert _is_fallback(tui.render(image, 80)[0])


# hyperlink


def test_wraps_text_in_osc8_open_and_close_sequences():
    result = hyperlink("click me", "https://example.com")
    assert result == "\x1b]8;;https://example.com\x1b\\click me\x1b]8;;\x1b\\"


def test_preserves_ansi_styling_inside_the_hyperlink():
    styled = "\x1b[4m\x1b[34mclick me\x1b[0m"
    result = hyperlink(styled, "https://example.com")
    assert result.startswith("\x1b]8;;https://example.com\x1b\\")
    assert styled in result
    assert result.endswith("\x1b]8;;\x1b\\")


def test_works_with_empty_text():
    result = hyperlink("", "https://example.com")
    assert result == "\x1b]8;;https://example.com\x1b\\\x1b]8;;\x1b\\"


def test_works_with_file_uris():
    result = hyperlink("README.md", "file:///home/user/README.md")
    assert "file:///home/user/README.md" in result
    assert "README.md" in result


# --- image fallback (0.83.0: shorten paths, hyperlink, clamp width) -----------


def test_truncates_long_image_fallback_lines_to_render_width():
    set_capabilities({"images": None, "trueColor": False, "hyperlinks": False})
    try:
        long_path = os.path.join(
            os.path.expanduser("~"),
            "images",
            "generated-image-with-a-very-long-absolute-path" * 4 + ".png",
        )
        width = 40
        image = Image(
            _TUI,
            "AAAA",
            "image/png",
            {"fallbackColor": lambda value: f"\x1b[33m{value}\x1b[0m"},
            {"filename": long_path},
            {"widthPx": 1280, "heightPx": 720},
        )
        lines = image.render(width)
        assert len(lines) == 1
        assert visible_width(lines[0]) <= width
        assert "..." in lines[0]
        assert "~" in lines[0]
    finally:
        reset_capabilities_cache()


def test_shortens_home_prefixed_absolute_paths_without_hyperlinks():
    set_capabilities({"images": None, "trueColor": False, "hyperlinks": False})
    try:
        abs_path = os.path.join(os.path.expanduser("~"), ".pi", "agent", "shot.png")
        result = image_fallback("image/png", {"widthPx": 1280, "heightPx": 720}, abs_path)
        assert result == "[Image: ~/.pi/agent/shot.png [image/png] 1280x720]"
    finally:
        reset_capabilities_cache()


def test_wraps_shortened_absolute_paths_in_osc8_file_links_when_hyperlinks_are_enabled():
    set_capabilities({"images": None, "trueColor": False, "hyperlinks": True})
    try:
        abs_path = os.path.join(os.path.expanduser("~"), ".pi", "agent", "shot.png")
        result = image_fallback("image/png", {"widthPx": 10, "heightPx": 10}, abs_path)
        assert "\x1b]8;;file://" in result
        assert abs_path.replace("\\", "/") in result or abs_path in result
        # Visible text must use ~/... not the expanded home path.
        visible = re.sub(r"\x1b\]8;;.*?\x1b\\", "", result)
        assert visible == "[Image: ~/.pi/agent/shot.png [image/png] 10x10]"
    finally:
        reset_capabilities_cache()


def test_leaves_bare_basenames_unchanged_and_does_not_hyperlink_them():
    set_capabilities({"images": None, "trueColor": False, "hyperlinks": True})
    try:
        result = image_fallback("image/png", {"widthPx": 1, "heightPx": 1}, "clankolas.png")
        assert result == "[Image: clankolas.png [image/png] 1x1]"
        assert "\x1b]8;" not in result
    finally:
        reset_capabilities_cache()


def test_omits_filename_segment_when_not_provided():
    set_capabilities({"images": None, "trueColor": False, "hyperlinks": False})
    try:
        assert image_fallback("image/png", {"widthPx": 8, "heightPx": 6}) == "[Image: [image/png] 8x6]"
    finally:
        reset_capabilities_cache()


def test_an_override_change_during_detection_leaves_no_stale_cache(monkeypatch):
    """pidrei-only (spec/ui-island.md, "Terminal-level state"): overrides and cache are one
    snapshot, so capabilities computed from old overrides are not stored
    next to new ones."""
    detected = {"images": None, "trueColor": True, "hyperlinks": False}
    changed: list[bool] = []

    def detect(*_args):
        if not changed:
            changed.append(True)
            # Another task changes the overrides while this one detects.
            terminal_image.set_capability_overrides({"trueColor": False})
        return dict(detected)

    monkeypatch.setattr(terminal_image, "detect_capabilities", detect)
    terminal_image.reset_capabilities_cache()
    try:
        terminal_image.get_capabilities()
        assert terminal_image.get_capabilities()["trueColor"] is False
    finally:
        terminal_image.set_capability_overrides({})
        terminal_image.reset_capabilities_cache()


@pytest.mark.tonio
async def test_priming_under_tmux_awaits_the_probe_once_and_replaces_earlier_detection():
    """pidrei-only: detection never runs the tmux probe (pi's `execSync`);
    `prime_capabilities` awaits the caller's, and what was detected before
    the answer (the probe's failure answer) is dropped."""
    probes: list[bool] = []

    async def probe() -> bool:
        probes.append(True)
        return True

    reset_capabilities_cache()
    try:
        with clean_env({"TMUX": "/tmp/tmux-1000/default,1234,0", "TERM_PROGRAM": "ghostty"}):
            assert get_capabilities()["hyperlinks"] is False
            await prime_capabilities(probe)
            assert get_capabilities()["hyperlinks"] is True
            await prime_capabilities(probe)
        assert probes == [True]
    finally:
        reset_capabilities_cache()


@pytest.mark.tonio
async def test_priming_outside_tmux_never_probes():
    probes: list[bool] = []

    async def probe() -> bool:
        probes.append(True)
        return True

    reset_capabilities_cache()
    try:
        with clean_env({"TERM_PROGRAM": "ghostty"}):
            await prime_capabilities(probe)
        assert probes == []
    finally:
        reset_capabilities_cache()
