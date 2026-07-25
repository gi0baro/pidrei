"""Default themes for TUI tests (port of pi tui ``test/test-themes.ts``)."""

from .chalk_like import chalk


default_select_list_theme = {
    "selectedPrefix": chalk.blue,
    "selectedText": chalk.bold,
    "description": chalk.dim,
    "scrollInfo": chalk.dim,
    "noMatch": chalk.dim,
}

default_markdown_theme = {
    "heading": chalk.bold.cyan,
    "link": chalk.blue,
    "linkUrl": chalk.dim,
    "code": chalk.yellow,
    "codeBlock": chalk.green,
    "codeBlockBorder": chalk.dim,
    "quote": chalk.italic,
    "quoteBorder": chalk.dim,
    "hr": chalk.dim,
    "listBullet": chalk.cyan,
    "bold": chalk.bold,
    "italic": chalk.italic,
    "strikethrough": chalk.strikethrough,
    "underline": chalk.underline,
}

default_editor_theme = {
    "borderColor": chalk.dim,
    "selectList": default_select_list_theme,
}
