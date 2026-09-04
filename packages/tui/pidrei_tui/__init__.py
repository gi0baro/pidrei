"""Mirror of pi tui src/index.ts (re-exports grow as modules are ported)."""

from .autocomplete import CombinedAutocompleteProvider
from .components.box import Box
from .components.cancellable_loader import CancellableLoader
from .components.editor import Editor, word_wrap_line
from .components.h_stack import HStack
from .components.image import Image
from .components.input import Input
from .components.loader import Loader
from .components.markdown import Markdown
from .components.markdown_lexer import lex_markdown
from .components.mouse_region import MouseRegion
from .components.scroll_view import ScrollView
from .components.select_list import SelectList
from .components.settings_list import SettingsList
from .components.spacer import Spacer
from .components.text import Text
from .components.truncated_text import TruncatedText
from .components.v_stack import VStack
from .editor_component import EditorComponent
from .fuzzy import fuzzy_filter, fuzzy_match
from .keybindings import (
    TUI_KEYBINDINGS,
    KeybindingsManager,
    get_keybindings,
    set_keybindings,
)
from .keys import (
    Key,
    decode_kitty_printable,
    is_key_release,
    is_key_repeat,
    is_kitty_protocol_active,
    matches_key,
    parse_key,
    set_kitty_protocol_active,
)
from .latex import render_latex
from .stdin_buffer import StdinBuffer
from .terminal import (
    ProcessTerminal,
    Terminal,
    is_apple_terminal_session,
    normalize_apple_terminal_input,
    parse_keyboard_protocol_negotiation_sequence,
)
from .terminal_colors import (
    parse_osc11_background_color,
    parse_terminal_color_scheme_report,
)
from .terminal_image import (
    allocate_image_id,
    calculate_image_rows,
    delete_all_kitty_images,
    delete_kitty_image,
    detect_capabilities,
    encode_iterm2,
    encode_kitty,
    get_capabilities,
    get_cell_dimensions,
    get_gif_dimensions,
    get_image_dimensions,
    get_jpeg_dimensions,
    get_png_dimensions,
    get_webp_dimensions,
    hyperlink,
    image_fallback,
    render_image,
    reset_capabilities_cache,
    set_capabilities,
    set_capability_overrides,
    set_cell_dimensions,
)
from .tui import (
    CURSOR_MARKER,
    TUI,
    Component,
    Container,
    Focusable,
    OverlayHandle,
    TuiMouseDispatchResult,
    TuiMouseEvent,
    TuiMouseEventResult,
    composite_tui_line,
    dispatch_mouse_event,
    is_focusable,
    is_viewport_tui,
    retarget_mouse_event,
)
from .tui_alt_screen import TuiAltScreen
from .tui_main_screen import TuiMainScreen
from .utils import (
    get_osc8_link_at_column,
    slice_by_column,
    strip_terminal_sequences,
    truncate_to_width,
    visible_width,
    wrap_text_with_ansi,
)


__all__ = [
    "CURSOR_MARKER",
    "TUI",
    "TUI_KEYBINDINGS",
    "Box",
    "CancellableLoader",
    "CombinedAutocompleteProvider",
    "Component",
    "Container",
    "Editor",
    "EditorComponent",
    "Focusable",
    "HStack",
    "Image",
    "Input",
    "Key",
    "KeybindingsManager",
    "Loader",
    "Markdown",
    "MouseRegion",
    "OverlayHandle",
    "ProcessTerminal",
    "ScrollView",
    "SelectList",
    "SettingsList",
    "Spacer",
    "StdinBuffer",
    "Terminal",
    "Text",
    "TruncatedText",
    "TuiAltScreen",
    "TuiMainScreen",
    "TuiMouseDispatchResult",
    "TuiMouseEvent",
    "TuiMouseEventResult",
    "VStack",
    "allocate_image_id",
    "calculate_image_rows",
    "composite_tui_line",
    "decode_kitty_printable",
    "delete_all_kitty_images",
    "delete_kitty_image",
    "detect_capabilities",
    "dispatch_mouse_event",
    "encode_iterm2",
    "encode_kitty",
    "fuzzy_filter",
    "fuzzy_match",
    "get_capabilities",
    "get_cell_dimensions",
    "get_gif_dimensions",
    "get_image_dimensions",
    "get_jpeg_dimensions",
    "get_keybindings",
    "get_osc8_link_at_column",
    "get_png_dimensions",
    "get_webp_dimensions",
    "hyperlink",
    "image_fallback",
    "is_apple_terminal_session",
    "is_focusable",
    "is_key_release",
    "is_key_repeat",
    "is_kitty_protocol_active",
    "is_viewport_tui",
    "lex_markdown",
    "matches_key",
    "normalize_apple_terminal_input",
    "parse_key",
    "parse_keyboard_protocol_negotiation_sequence",
    "parse_osc11_background_color",
    "parse_terminal_color_scheme_report",
    "render_image",
    "render_latex",
    "reset_capabilities_cache",
    "retarget_mouse_event",
    "set_capabilities",
    "set_capability_overrides",
    "set_cell_dimensions",
    "set_keybindings",
    "set_kitty_protocol_active",
    "slice_by_column",
    "strip_terminal_sequences",
    "truncate_to_width",
    "visible_width",
    "word_wrap_line",
    "wrap_text_with_ansi",
]
