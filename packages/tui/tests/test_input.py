"""Mirror of pi tui test/input.test.ts."""

import pytest

from pidrei_tui.components.input import Input
from pidrei_tui.utils import strip_terminal_sequences, visible_width


@pytest.mark.tonio
async def test_supports_a_custom_prompt_and_styled_placeholder():
    input_component = Input(
        {"prompt": "", "placeholder": "Find transcript", "placeholderStyle": lambda text: f"\x1b[2m{text}\x1b[22m"}
    )
    input_component.focused = True

    empty = input_component.render(20)[0]
    assert "\x1b[2m" in empty
    assert strip_terminal_sequences(empty).rstrip() == "Find transcript"

    await input_component.handle_input("n")
    populated = input_component.render(20)[0]
    assert strip_terminal_sequences(populated).rstrip() == "n"


@pytest.mark.tonio
async def test_submits_value_including_backslash_on_enter():
    input_component = Input()
    submitted = None

    def on_submit(value):
        nonlocal submitted
        submitted = value

    input_component.on_submit = on_submit

    # Type hello, then backslash, then Enter
    await input_component.handle_input("h")
    await input_component.handle_input("e")
    await input_component.handle_input("l")
    await input_component.handle_input("l")
    await input_component.handle_input("o")
    await input_component.handle_input("\\")
    await input_component.handle_input("\r")

    # Input is single-line, no backslash+Enter workaround
    assert submitted == "hello\\"


@pytest.mark.tonio
async def test_inserts_backslash_as_regular_character():
    input_component = Input()

    await input_component.handle_input("\\")
    await input_component.handle_input("x")

    assert input_component.get_value() == "\\x"


# render


@pytest.mark.tonio
async def test_does_not_overflow_with_wide_cjk_and_fullwidth_text():
    width = 93
    cases = [
        "가나다라마바사아자차카타파하 한글 텍스트가 터미널 너비를 초과하면 크래시가 발생합니다 이것은 재현용 테스트입니다",
        "これはテスト文章です。日本語のテキストが正しく表示されるかどうかを確認するためのサンプルテキストです。あいうえお",
        "这是一段测试文本，用于验证中文字符在终端中的显示宽度是否被正确计算，如果不正确就会导致用户界面崩溃的问题",
        "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍ",
    ]

    async def move_start(_input):
        pass

    async def move_middle(input_component):
        for _ in range(10):
            await input_component.handle_input("\x1b[C")

    async def move_end(input_component):
        await input_component.handle_input("\x05")

    cursor_positions = [("start", move_start), ("middle", move_middle), ("end", move_end)]

    for text in cases:
        for label, move in cursor_positions:
            input_component = Input()
            input_component.set_value(text)
            input_component.focused = True
            await move(input_component)

            line = input_component.render(width)[0]
            assert line
            assert visible_width(line) <= width, f"rendered line overflowed for {text} at {label}"


@pytest.mark.tonio
async def test_keeps_the_cursor_visible_when_horizontally_scrolling_wide_text():
    input_component = Input()
    width = 20
    text = "가나다라마바사아자차카타파하"
    input_component.set_value(text)
    input_component.focused = True
    await input_component.handle_input("\x01")
    for _ in range(5):
        await input_component.handle_input("\x1b[C")

    line = input_component.render(width)[0]
    assert line
    assert visible_width(line) <= width


# Kill ring


@pytest.mark.tonio
async def test_ctrl_w_saves_deleted_text_to_kill_ring_and_ctrl_y_yanks_it():
    input_component = Input()

    input_component.set_value("foo bar baz")
    # Move cursor to end
    await input_component.handle_input("\x05")  # Ctrl+E

    await input_component.handle_input("\x17")  # Ctrl+W - deletes "baz"
    assert input_component.get_value() == "foo bar "

    # Move to beginning and yank
    await input_component.handle_input("\x01")  # Ctrl+A
    await input_component.handle_input("\x19")  # Ctrl+Y
    assert input_component.get_value() == "bazfoo bar "


@pytest.mark.tonio
async def test_ctrl_w_preserves_ascii_punctuation_boundaries():
    input_component = Input()

    input_component.set_value("foo.bar")
    await input_component.handle_input("\x05")  # Ctrl+E
    await input_component.handle_input("\x17")  # Ctrl+W - deletes "bar"
    assert input_component.get_value() == "foo."

    input_component.set_value("foo:bar")
    await input_component.handle_input("\x05")  # Ctrl+E
    await input_component.handle_input("\x17")  # Ctrl+W - deletes "bar"
    assert input_component.get_value() == "foo:"


@pytest.mark.tonio
async def test_ctrl_w_handles_unicode_word_boundaries():
    input_component = Input()

    # "你好世界。你好，世界" segments as: 你好|世界|。|你好|，|世界
    input_component.set_value("你好世界。你好，世界")
    await input_component.handle_input("\x05")  # Ctrl+E
    await input_component.handle_input("\x17")  # Ctrl+W - deletes "世界"
    assert input_component.get_value() == "你好世界。你好，"
    await input_component.handle_input("\x17")  # Ctrl+W - deletes "，"
    assert input_component.get_value() == "你好世界。你好"
    await input_component.handle_input("\x17")  # Ctrl+W - deletes "你好"
    assert input_component.get_value() == "你好世界。"
    await input_component.handle_input("\x17")  # Ctrl+W - deletes "。"
    assert input_component.get_value() == "你好世界"
    await input_component.handle_input("\x17")  # Ctrl+W - deletes "世界"
    assert input_component.get_value() == "你好"
    await input_component.handle_input("\x17")  # Ctrl+W - deletes "你好"
    assert input_component.get_value() == ""


@pytest.mark.tonio
async def test_ctrl_u_saves_deleted_text_to_kill_ring():
    input_component = Input()

    input_component.set_value("hello world")
    # Move cursor to after "hello "
    await input_component.handle_input("\x01")  # Ctrl+A
    for _ in range(6):
        await input_component.handle_input("\x1b[C")

    await input_component.handle_input("\x15")  # Ctrl+U - deletes "hello "
    assert input_component.get_value() == "world"

    await input_component.handle_input("\x19")  # Ctrl+Y
    assert input_component.get_value() == "hello world"


@pytest.mark.tonio
async def test_ctrl_k_saves_deleted_text_to_kill_ring():
    input_component = Input()

    input_component.set_value("hello world")
    await input_component.handle_input("\x01")  # Ctrl+A
    await input_component.handle_input("\x0b")  # Ctrl+K - deletes "hello world"

    assert input_component.get_value() == ""

    await input_component.handle_input("\x19")  # Ctrl+Y
    assert input_component.get_value() == "hello world"


@pytest.mark.tonio
async def test_ctrl_y_does_nothing_when_kill_ring_is_empty():
    input_component = Input()

    input_component.set_value("test")
    await input_component.handle_input("\x05")  # Ctrl+E
    await input_component.handle_input("\x19")  # Ctrl+Y
    assert input_component.get_value() == "test"


@pytest.mark.tonio
async def test_alt_y_cycles_through_kill_ring_after_ctrl_y():
    input_component = Input()

    # Create kill ring with multiple entries
    input_component.set_value("first")
    await input_component.handle_input("\x05")  # Ctrl+E
    await input_component.handle_input("\x17")  # Ctrl+W - deletes "first"
    input_component.set_value("second")
    await input_component.handle_input("\x05")  # Ctrl+E
    await input_component.handle_input("\x17")  # Ctrl+W - deletes "second"
    input_component.set_value("third")
    await input_component.handle_input("\x05")  # Ctrl+E
    await input_component.handle_input("\x17")  # Ctrl+W - deletes "third"

    assert input_component.get_value() == ""

    await input_component.handle_input("\x19")  # Ctrl+Y - yanks "third"
    assert input_component.get_value() == "third"

    await input_component.handle_input("\x1by")  # Alt+Y - cycles to "second"
    assert input_component.get_value() == "second"

    await input_component.handle_input("\x1by")  # Alt+Y - cycles to "first"
    assert input_component.get_value() == "first"

    await input_component.handle_input("\x1by")  # Alt+Y - cycles back to "third"
    assert input_component.get_value() == "third"


@pytest.mark.tonio
async def test_alt_y_does_nothing_if_not_preceded_by_yank():
    input_component = Input()

    input_component.set_value("test")
    await input_component.handle_input("\x05")  # Ctrl+E
    await input_component.handle_input("\x17")  # Ctrl+W - deletes "test"
    input_component.set_value("other")
    await input_component.handle_input("\x05")  # Ctrl+E

    # Type something to break the yank chain
    await input_component.handle_input("x")
    assert input_component.get_value() == "otherx"

    await input_component.handle_input("\x1by")  # Alt+Y - should do nothing
    assert input_component.get_value() == "otherx"


@pytest.mark.tonio
async def test_alt_y_does_nothing_if_kill_ring_has_one_entry():
    input_component = Input()

    input_component.set_value("only")
    await input_component.handle_input("\x05")  # Ctrl+E
    await input_component.handle_input("\x17")  # Ctrl+W - deletes "only"

    await input_component.handle_input("\x19")  # Ctrl+Y - yanks "only"
    assert input_component.get_value() == "only"

    await input_component.handle_input("\x1by")  # Alt+Y - should do nothing
    assert input_component.get_value() == "only"


@pytest.mark.tonio
async def test_consecutive_ctrl_w_accumulates_into_one_kill_ring_entry():
    input_component = Input()

    input_component.set_value("one two three")
    await input_component.handle_input("\x05")  # Ctrl+E
    await input_component.handle_input("\x17")  # Ctrl+W - deletes "three"
    await input_component.handle_input("\x17")  # Ctrl+W - deletes "two "
    await input_component.handle_input("\x17")  # Ctrl+W - deletes "one "

    assert input_component.get_value() == ""

    await input_component.handle_input("\x19")  # Ctrl+Y
    assert input_component.get_value() == "one two three"


@pytest.mark.tonio
async def test_non_delete_actions_break_kill_accumulation():
    input_component = Input()

    input_component.set_value("foo bar baz")
    await input_component.handle_input("\x05")  # Ctrl+E
    await input_component.handle_input("\x17")  # Ctrl+W - deletes "baz"
    assert input_component.get_value() == "foo bar "

    await input_component.handle_input("x")  # Typing breaks accumulation
    assert input_component.get_value() == "foo bar x"

    await input_component.handle_input("\x17")  # Ctrl+W - deletes "x" (separate entry)
    assert input_component.get_value() == "foo bar "

    await input_component.handle_input("\x19")  # Ctrl+Y - most recent is "x"
    assert input_component.get_value() == "foo bar x"

    await input_component.handle_input("\x1by")  # Alt+Y - cycle to "baz"
    assert input_component.get_value() == "foo bar baz"


@pytest.mark.tonio
async def test_non_yank_actions_break_alt_y_chain():
    input_component = Input()

    input_component.set_value("first")
    await input_component.handle_input("\x05")  # Ctrl+E
    await input_component.handle_input("\x17")  # Ctrl+W
    input_component.set_value("second")
    await input_component.handle_input("\x05")  # Ctrl+E
    await input_component.handle_input("\x17")  # Ctrl+W
    input_component.set_value("")

    await input_component.handle_input("\x19")  # Ctrl+Y - yanks "second"
    assert input_component.get_value() == "second"

    await input_component.handle_input("x")  # Breaks yank chain
    assert input_component.get_value() == "secondx"

    await input_component.handle_input("\x1by")  # Alt+Y - should do nothing
    assert input_component.get_value() == "secondx"


@pytest.mark.tonio
async def test_kill_ring_rotation_persists_after_cycling():
    input_component = Input()

    input_component.set_value("first")
    await input_component.handle_input("\x05")  # Ctrl+E
    await input_component.handle_input("\x17")  # deletes "first"
    input_component.set_value("second")
    await input_component.handle_input("\x05")  # Ctrl+E
    await input_component.handle_input("\x17")  # deletes "second"
    input_component.set_value("third")
    await input_component.handle_input("\x05")  # Ctrl+E
    await input_component.handle_input("\x17")  # deletes "third"
    input_component.set_value("")

    await input_component.handle_input("\x19")  # Ctrl+Y - yanks "third"
    await input_component.handle_input("\x1by")  # Alt+Y - cycles to "second"
    assert input_component.get_value() == "second"

    # Break chain and start fresh
    await input_component.handle_input("x")
    input_component.set_value("")

    # New yank should get "second" (now at end after rotation)
    await input_component.handle_input("\x19")  # Ctrl+Y
    assert input_component.get_value() == "second"


@pytest.mark.tonio
async def test_backward_deletions_prepend_forward_deletions_append_during_accumulation():
    input_component = Input()

    input_component.set_value("prefix|suffix")
    # Position cursor at "|"
    await input_component.handle_input("\x01")  # Ctrl+A
    for _ in range(6):
        await input_component.handle_input("\x1b[C")  # Move right 6

    await input_component.handle_input("\x0b")  # Ctrl+K - deletes "|suffix" (forward)
    assert input_component.get_value() == "prefix"

    await input_component.handle_input("\x19")  # Ctrl+Y
    assert input_component.get_value() == "prefix|suffix"


@pytest.mark.tonio
async def test_alt_d_deletes_word_forward_and_saves_to_kill_ring():
    input_component = Input()

    input_component.set_value("hello world test")
    await input_component.handle_input("\x01")  # Ctrl+A

    await input_component.handle_input("\x1bd")  # Alt+D - deletes "hello"
    assert input_component.get_value() == " world test"

    await input_component.handle_input("\x1bd")  # Alt+D - deletes " world"
    assert input_component.get_value() == " test"

    # Yank should get accumulated text
    await input_component.handle_input("\x19")  # Ctrl+Y
    assert input_component.get_value() == "hello world test"


@pytest.mark.tonio
async def test_alt_d_preserves_ascii_punctuation_boundaries():
    input_component = Input()

    input_component.set_value("foo.bar baz")
    await input_component.handle_input("\x01")  # Ctrl+A
    await input_component.handle_input("\x1bd")  # Alt+D - deletes "foo"
    assert input_component.get_value() == ".bar baz"
    await input_component.handle_input("\x1bd")  # Alt+D - deletes "."
    assert input_component.get_value() == "bar baz"
    await input_component.handle_input("\x1bd")  # Alt+D - deletes "bar"
    assert input_component.get_value() == " baz"


@pytest.mark.tonio
async def test_alt_d_handles_unicode_word_boundaries():
    input_component = Input()

    # "你好世界。你好，世界" segments as: 你好|世界|。|你好|，|世界
    input_component.set_value("你好世界。你好，世界")
    await input_component.handle_input("\x01")  # Ctrl+A
    await input_component.handle_input("\x1bd")  # Alt+D - deletes "你好"
    assert input_component.get_value() == "世界。你好，世界"
    await input_component.handle_input("\x1bd")  # Alt+D - deletes "世界"
    assert input_component.get_value() == "。你好，世界"
    await input_component.handle_input("\x1bd")  # Alt+D - deletes "。"
    assert input_component.get_value() == "你好，世界"
    await input_component.handle_input("\x1bd")  # Alt+D - deletes "你好"
    assert input_component.get_value() == "，世界"
    await input_component.handle_input("\x1bd")  # Alt+D - deletes "，"
    assert input_component.get_value() == "世界"
    await input_component.handle_input("\x1bd")  # Alt+D - deletes "世界"
    assert input_component.get_value() == ""


@pytest.mark.tonio
async def test_handles_yank_in_middle_of_text():
    input_component = Input()

    input_component.set_value("word")
    await input_component.handle_input("\x05")  # Ctrl+E
    await input_component.handle_input("\x17")  # Ctrl+W - deletes "word"
    input_component.set_value("hello world")
    # Move to middle (after "hello ")
    await input_component.handle_input("\x01")  # Ctrl+A
    for _ in range(6):
        await input_component.handle_input("\x1b[C")

    await input_component.handle_input("\x19")  # Ctrl+Y
    assert input_component.get_value() == "hello wordworld"


@pytest.mark.tonio
async def test_handles_yank_pop_in_middle_of_text():
    input_component = Input()

    # Create two kill ring entries
    input_component.set_value("FIRST")
    await input_component.handle_input("\x05")  # Ctrl+E
    await input_component.handle_input("\x17")  # Ctrl+W - deletes "FIRST"
    input_component.set_value("SECOND")
    await input_component.handle_input("\x05")  # Ctrl+E
    await input_component.handle_input("\x17")  # Ctrl+W - deletes "SECOND"

    # Set up "hello world" and position cursor after "hello "
    input_component.set_value("hello world")
    await input_component.handle_input("\x01")  # Ctrl+A
    for _ in range(6):
        await input_component.handle_input("\x1b[C")

    await input_component.handle_input("\x19")  # Ctrl+Y - yanks "SECOND"
    assert input_component.get_value() == "hello SECONDworld"

    await input_component.handle_input("\x1by")  # Alt+Y - replaces with "FIRST"
    assert input_component.get_value() == "hello FIRSTworld"


# Undo


@pytest.mark.tonio
async def test_does_nothing_when_undo_stack_is_empty():
    input_component = Input()

    await input_component.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert input_component.get_value() == ""


@pytest.mark.tonio
async def test_coalesces_consecutive_word_characters_into_one_undo_unit():
    input_component = Input()

    for char in "hello world":
        await input_component.handle_input(char)
    assert input_component.get_value() == "hello world"

    # Undo removes " world"
    await input_component.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert input_component.get_value() == "hello"

    # Undo removes "hello"
    await input_component.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert input_component.get_value() == ""


@pytest.mark.tonio
async def test_undoes_spaces_one_at_a_time():
    input_component = Input()

    for char in "hello  ":
        await input_component.handle_input(char)
    assert input_component.get_value() == "hello  "

    await input_component.handle_input("\x1b[45;5u")  # Ctrl+- (undo) - removes second " "
    assert input_component.get_value() == "hello "

    await input_component.handle_input("\x1b[45;5u")  # Ctrl+- (undo) - removes first " "
    assert input_component.get_value() == "hello"

    await input_component.handle_input("\x1b[45;5u")  # Ctrl+- (undo) - removes "hello"
    assert input_component.get_value() == ""


@pytest.mark.tonio
async def test_undoes_backspace():
    input_component = Input()

    for char in "hello":
        await input_component.handle_input(char)
    await input_component.handle_input("\x7f")  # Backspace
    assert input_component.get_value() == "hell"

    await input_component.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert input_component.get_value() == "hello"


@pytest.mark.tonio
async def test_undoes_forward_delete():
    input_component = Input()

    for char in "hello":
        await input_component.handle_input(char)
    await input_component.handle_input("\x01")  # Ctrl+A - go to start
    await input_component.handle_input("\x1b[C")  # Right arrow
    await input_component.handle_input("\x1b[3~")  # Delete key
    assert input_component.get_value() == "hllo"

    await input_component.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert input_component.get_value() == "hello"


@pytest.mark.tonio
async def test_undoes_ctrl_w_delete_word_backward():
    input_component = Input()

    for char in "hello world":
        await input_component.handle_input(char)
    assert input_component.get_value() == "hello world"

    await input_component.handle_input("\x17")  # Ctrl+W
    assert input_component.get_value() == "hello "

    await input_component.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert input_component.get_value() == "hello world"


@pytest.mark.tonio
async def test_undoes_ctrl_k_delete_to_line_end():
    input_component = Input()

    for char in "hello world":
        await input_component.handle_input(char)
    await input_component.handle_input("\x01")  # Ctrl+A
    for _ in range(6):
        await input_component.handle_input("\x1b[C")

    await input_component.handle_input("\x0b")  # Ctrl+K
    assert input_component.get_value() == "hello "

    await input_component.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert input_component.get_value() == "hello world"


@pytest.mark.tonio
async def test_undoes_ctrl_u_delete_to_line_start():
    input_component = Input()

    for char in "hello world":
        await input_component.handle_input(char)
    await input_component.handle_input("\x01")  # Ctrl+A
    for _ in range(6):
        await input_component.handle_input("\x1b[C")

    await input_component.handle_input("\x15")  # Ctrl+U
    assert input_component.get_value() == "world"

    await input_component.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert input_component.get_value() == "hello world"


@pytest.mark.tonio
async def test_undoes_yank():
    input_component = Input()

    for char in "hello ":
        await input_component.handle_input(char)
    await input_component.handle_input("\x17")  # Ctrl+W - delete "hello "
    await input_component.handle_input("\x19")  # Ctrl+Y - yank
    assert input_component.get_value() == "hello "

    await input_component.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert input_component.get_value() == ""


@pytest.mark.tonio
async def test_undoes_paste_atomically():
    input_component = Input()

    input_component.set_value("hello world")
    await input_component.handle_input("\x01")  # Ctrl+A
    for _ in range(5):
        await input_component.handle_input("\x1b[C")

    # Simulate bracketed paste
    await input_component.handle_input("\x1b[200~beep boop\x1b[201~")
    assert input_component.get_value() == "hellobeep boop world"

    # Single undo should restore entire pre-paste state
    await input_component.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert input_component.get_value() == "hello world"


@pytest.mark.tonio
async def test_undoes_alt_d_delete_word_forward():
    input_component = Input()

    input_component.set_value("hello world")
    await input_component.handle_input("\x01")  # Ctrl+A

    await input_component.handle_input("\x1bd")  # Alt+D - deletes "hello"
    assert input_component.get_value() == " world"

    await input_component.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert input_component.get_value() == "hello world"


@pytest.mark.tonio
async def test_cursor_movement_starts_new_undo_unit():
    input_component = Input()

    await input_component.handle_input("a")
    await input_component.handle_input("b")
    await input_component.handle_input("c")
    await input_component.handle_input("\x01")  # Ctrl+A - movement breaks coalescing
    await input_component.handle_input("\x05")  # Ctrl+E
    await input_component.handle_input("d")
    await input_component.handle_input("e")
    assert input_component.get_value() == "abcde"

    # Undo removes "de" (typed after movement)
    await input_component.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert input_component.get_value() == "abc"

    # Undo removes "abc"
    await input_component.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert input_component.get_value() == ""
