"""Autocomplete providers for the editor (port of pi tui ``autocomplete.ts``).

``CombinedAutocompleteProvider`` handles slash commands, ``@`` fuzzy file
search (via the external ``fd`` binary, spawned through tonio) and plain
path-prefix completion. Records are camelCase dicts mirroring pi's shapes:
items ``{"value", "label", "description"?}``, suggestions ``{"items",
"prefix"}``, apply results ``{"lines", "cursorLine", "cursorCol"}``.

Providers are duck-typed (pi's ``AutocompleteProvider`` interface): an object
with async ``get_suggestions(lines, cursor_line, cursor_col, options)`` and
sync ``apply_completion(lines, cursor_line, cursor_col, item, prefix)``, plus
optional ``trigger_characters`` and ``should_trigger_file_completion``.
``options["signal"]`` is a ``CancelToken`` (pi uses a DOM ``AbortSignal``).
"""

import os
import re
import subprocess

import tonio.colored as tonio

from .fuzzy import fuzzy_filter


__all__ = ["CombinedAutocompleteProvider"]

PATH_DELIMITERS = {" ", "\t", '"', "'", "="}

_ESCAPE_REGEX_RE = re.compile(r"[.*+?^${}()|[\]\\]")


def _to_display_path(value: str) -> str:
    return value.replace("\\", "/")


def _escape_regex(value: str) -> str:
    return _ESCAPE_REGEX_RE.sub(lambda m: "\\" + m.group(0), value)


def _build_fd_path_query(query: str) -> str:
    normalized = _to_display_path(query)
    if "/" not in normalized:
        return normalized

    has_trailing_separator = normalized.endswith("/")
    trimmed = re.sub(r"^/+|/+$", "", normalized)
    if not trimmed:
        return normalized

    separator_pattern = "[\\\\/]"
    segments = [_escape_regex(segment) for segment in trimmed.split("/") if segment]
    if not segments:
        return normalized

    pattern = separator_pattern.join(segments)
    if has_trailing_separator:
        pattern += separator_pattern
    return pattern


def _find_last_delimiter(text: str) -> int:
    for i in range(len(text) - 1, -1, -1):
        if text[i] in PATH_DELIMITERS:
            return i
    return -1


def _find_unclosed_quote_start(text: str) -> int | None:
    in_quotes = False
    quote_start = -1

    for i, char in enumerate(text):
        if char == '"':
            in_quotes = not in_quotes
            if in_quotes:
                quote_start = i

    return quote_start if in_quotes else None


def _is_token_start(text: str, index: int) -> bool:
    return index == 0 or text[index - 1] in PATH_DELIMITERS


def _extract_quoted_prefix(text: str) -> str | None:
    quote_start = _find_unclosed_quote_start(text)
    if quote_start is None:
        return None

    if quote_start > 0 and text[quote_start - 1] == "@":
        if not _is_token_start(text, quote_start - 1):
            return None
        return text[quote_start - 1 :]

    if not _is_token_start(text, quote_start):
        return None

    return text[quote_start:]


def _parse_path_prefix(prefix: str) -> dict:
    if prefix.startswith('@"'):
        return {"rawPrefix": prefix[2:], "isAtPrefix": True, "isQuotedPrefix": True}
    if prefix.startswith('"'):
        return {"rawPrefix": prefix[1:], "isAtPrefix": False, "isQuotedPrefix": True}
    if prefix.startswith("@"):
        return {"rawPrefix": prefix[1:], "isAtPrefix": True, "isQuotedPrefix": False}
    return {"rawPrefix": prefix, "isAtPrefix": False, "isQuotedPrefix": False}


def _build_completion_value(path: str, *, is_directory: bool, is_at_prefix: bool, is_quoted_prefix: bool) -> str:
    needs_quotes = is_quoted_prefix or " " in path
    prefix = "@" if is_at_prefix else ""

    if not needs_quotes:
        return f"{prefix}{path}"

    return f'{prefix}"{path}"'


async def _walk_directory_with_fd(base_dir: str, fd_path: str, query: str, max_results: int, signal) -> list[dict]:
    """Use fd to walk the directory tree (fast, respects .gitignore)."""
    args = [
        "--base-directory",
        base_dir,
        "--max-results",
        str(max_results),
        "--type",
        "f",
        "--type",
        "d",
        "--follow",
        "--hidden",
        "--exclude",
        ".git",
        "--exclude",
        ".git/*",
        "--exclude",
        ".git/**",
    ]

    if "/" in _to_display_path(query):
        args.append("--full-path")

    if query:
        args.append(_build_fd_path_query(query))

    if signal.cancelled:
        return []

    try:
        process = await tonio.open_process(
            [fd_path, *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []

    def _kill(_reason: BaseException) -> None:
        if process.returncode is None:
            process.kill()

    unsubscribe = signal.on_cancel(_kill)
    chunks: list[bytes] = []
    try:
        with process.stdout as stream:
            while True:
                chunk = await stream.receive_some()
                if not chunk:
                    break
                chunks.append(bytes(chunk))
        code = await process.wait()
    except Exception:
        unsubscribe()
        return []
    unsubscribe()

    stdout = b"".join(chunks).decode("utf-8", "replace")
    if signal.cancelled or code != 0 or not stdout:
        return []

    lines = [line for line in stdout.strip().split("\n") if line]
    results: list[dict] = []

    for line in lines:
        display_line = _to_display_path(line)
        has_trailing_separator = display_line.endswith("/")
        normalized_path = display_line[:-1] if has_trailing_separator else display_line
        if normalized_path == ".git" or normalized_path.startswith(".git/") or "/.git/" in normalized_path:
            continue

        results.append({"path": display_line, "isDirectory": has_trailing_separator})

    return results


class CombinedAutocompleteProvider:
    """Combined provider that handles both slash commands and file paths.

    ``commands`` entries are dicts: slash commands ``{"name", "description"?,
    "argumentHint"?, "getArgumentCompletions"?}`` or plain autocomplete items
    ``{"value", "label", "description"?}``.
    """

    def __init__(self, commands: list[dict], base_path: str, fd_path: str | None = None) -> None:
        self._commands = commands
        self._base_path = base_path
        self._fd_path = fd_path

    async def get_suggestions(self, lines: list[str], cursor_line: int, cursor_col: int, options: dict) -> dict | None:
        current_line = lines[cursor_line] if cursor_line < len(lines) else ""
        text_before_cursor = current_line[:cursor_col]

        at_prefix = self._extract_at_prefix(text_before_cursor)
        if at_prefix:
            parsed = _parse_path_prefix(at_prefix)
            suggestions = await self._get_fuzzy_file_suggestions(
                parsed["rawPrefix"],
                is_quoted_prefix=parsed["isQuotedPrefix"],
                signal=options["signal"],
            )
            if not suggestions:
                return None

            return {"items": suggestions, "prefix": at_prefix}

        if not options.get("force") and text_before_cursor.startswith("/"):
            space_index = text_before_cursor.find(" ")

            if space_index == -1:
                prefix = text_before_cursor[1:]
                command_items = []
                for cmd in self._commands:
                    name = cmd["name"] if "name" in cmd else cmd["value"]
                    hint = cmd["argumentHint"] if cmd.get("argumentHint") else None
                    desc = cmd.get("description") or ""
                    full_desc = ((f"{hint} — {desc}" if desc else hint) if hint else desc) or None
                    command_items.append({"name": name, "label": name, "description": full_desc})

                filtered = [
                    {
                        "value": item["name"],
                        "label": item["label"],
                        **({"description": item["description"]} if item["description"] else {}),
                    }
                    for item in fuzzy_filter(command_items, prefix, lambda item: item["name"])
                ]

                if not filtered:
                    return None

                return {"items": filtered, "prefix": text_before_cursor}

            command_name = text_before_cursor[1:space_index]
            argument_text = text_before_cursor[space_index + 1 :]

            command = None
            for cmd in self._commands:
                name = cmd["name"] if "name" in cmd else cmd["value"]
                if name == command_name:
                    command = cmd
                    break
            if command is None or not command.get("getArgumentCompletions"):
                return None

            argument_suggestions = await command["getArgumentCompletions"](argument_text)
            if not isinstance(argument_suggestions, list) or not argument_suggestions:
                return None

            return {"items": argument_suggestions, "prefix": argument_text}

        path_match = self._extract_path_prefix(text_before_cursor, options.get("force") or False)
        if path_match is None:
            return None

        # Directory scan + per-entry stat: pool-side, never on a worker.
        suggestions = await tonio.spawn_blocking(self._get_file_suggestions, path_match)
        if not suggestions:
            return None

        return {"items": suggestions, "prefix": path_match}

    def apply_completion(self, lines: list[str], cursor_line: int, cursor_col: int, item: dict, prefix: str) -> dict:
        current_line = lines[cursor_line] if cursor_line < len(lines) else ""
        before_prefix = current_line[: cursor_col - len(prefix)]
        after_cursor = current_line[cursor_col:]
        is_quoted_prefix = prefix.startswith(('"', '@"'))
        has_leading_quote_after_cursor = after_cursor.startswith('"')
        has_trailing_quote_in_item = item["value"].endswith('"')
        adjusted_after_cursor = (
            after_cursor[1:]
            if is_quoted_prefix and has_trailing_quote_in_item and has_leading_quote_after_cursor
            else after_cursor
        )

        # Check if we're completing a slash command (prefix starts with "/" but NOT a file path)
        # Slash commands are at the start of the line and don't contain path separators after the first /
        is_slash_command = prefix.startswith("/") and before_prefix.strip() == "" and "/" not in prefix[1:]
        if is_slash_command:
            # This is a command name completion
            new_line = f"{before_prefix}/{item['value']} {adjusted_after_cursor}"
            new_lines = [*lines]
            new_lines[cursor_line] = new_line

            return {
                "lines": new_lines,
                "cursorLine": cursor_line,
                # +2 for "/" and space
                "cursorCol": len(before_prefix) + len(item["value"]) + 2,
            }

        # Check if we're completing a file attachment (prefix starts with "@")
        if prefix.startswith("@"):
            # Don't add space after directories so user can continue autocompleting
            is_directory = item["label"].endswith("/")
            suffix = "" if is_directory else " "
            new_line = f"{before_prefix + item['value']}{suffix}{adjusted_after_cursor}"
            new_lines = [*lines]
            new_lines[cursor_line] = new_line

            has_trailing_quote = item["value"].endswith('"')
            cursor_offset = len(item["value"]) - 1 if is_directory and has_trailing_quote else len(item["value"])

            return {
                "lines": new_lines,
                "cursorLine": cursor_line,
                "cursorCol": len(before_prefix) + cursor_offset + len(suffix),
            }

        # Check if we're in a slash command context (beforePrefix contains "/command ")
        text_before_cursor = current_line[:cursor_col]
        if "/" in text_before_cursor and " " in text_before_cursor:
            # This is likely a command argument completion
            new_line = before_prefix + item["value"] + adjusted_after_cursor
            new_lines = [*lines]
            new_lines[cursor_line] = new_line

            is_directory = item["label"].endswith("/")
            has_trailing_quote = item["value"].endswith('"')
            cursor_offset = len(item["value"]) - 1 if is_directory and has_trailing_quote else len(item["value"])

            return {
                "lines": new_lines,
                "cursorLine": cursor_line,
                "cursorCol": len(before_prefix) + cursor_offset,
            }

        # For file paths, complete the path
        new_line = before_prefix + item["value"] + adjusted_after_cursor
        new_lines = [*lines]
        new_lines[cursor_line] = new_line

        is_directory = item["label"].endswith("/")
        has_trailing_quote = item["value"].endswith('"')
        cursor_offset = len(item["value"]) - 1 if is_directory and has_trailing_quote else len(item["value"])

        return {
            "lines": new_lines,
            "cursorLine": cursor_line,
            "cursorCol": len(before_prefix) + cursor_offset,
        }

    # Extract @ prefix for fuzzy file suggestions
    def _extract_at_prefix(self, text: str) -> str | None:
        quoted_prefix = _extract_quoted_prefix(text)
        if quoted_prefix is not None and quoted_prefix.startswith('@"'):
            return quoted_prefix

        last_delimiter_index = _find_last_delimiter(text)
        token_start = 0 if last_delimiter_index == -1 else last_delimiter_index + 1

        if token_start < len(text) and text[token_start] == "@":
            return text[token_start:]

        return None

    # Extract a path-like prefix from the text before cursor
    def _extract_path_prefix(self, text: str, force_extract: bool = False) -> str | None:
        quoted_prefix = _extract_quoted_prefix(text)
        if quoted_prefix is not None:
            return quoted_prefix

        last_delimiter_index = _find_last_delimiter(text)
        path_prefix = text if last_delimiter_index == -1 else text[last_delimiter_index + 1 :]

        # For forced extraction (Tab key), always return something
        if force_extract:
            return path_prefix

        # For natural triggers, return if it looks like a path, ends with /, starts with ~/, .
        # Only return empty string if the text looks like it's starting a path context
        if "/" in path_prefix or path_prefix.startswith((".", "~/")):
            return path_prefix

        # Return empty string only after a space (not for completely empty text)
        # Empty text should not trigger file suggestions - that's for forced Tab completion
        if path_prefix == "" and text.endswith(" "):
            return path_prefix

        return None

    # Expand home directory (~/) to actual home path
    def _expand_home_path(self, path: str) -> str:
        if path.startswith("~/"):
            expanded_path = os.path.join(os.path.expanduser("~"), path[2:])
            # Preserve trailing slash if original path had one
            if path.endswith("/") and not expanded_path.endswith("/"):
                return f"{expanded_path}/"
            return expanded_path
        elif path == "~":
            return os.path.expanduser("~")
        return path

    def _resolve_scoped_fuzzy_query(self, raw_query: str) -> dict | None:
        normalized_query = _to_display_path(raw_query)
        slash_index = normalized_query.rfind("/")
        if slash_index == -1:
            return None

        display_base = normalized_query[: slash_index + 1]
        query = normalized_query[slash_index + 1 :]

        if display_base.startswith("~/"):
            base_dir = self._expand_home_path(display_base)
        elif display_base.startswith("/"):
            base_dir = display_base
        else:
            base_dir = os.path.join(self._base_path, display_base)

        try:
            if not os.path.isdir(base_dir):
                return None
        except OSError:
            return None

        return {"baseDir": base_dir, "query": query, "displayBase": display_base}

    def _scoped_path_for_display(self, display_base: str, relative_path: str) -> str:
        normalized_relative_path = _to_display_path(relative_path)
        if display_base == "/":
            return f"/{normalized_relative_path}"
        return f"{_to_display_path(display_base)}{normalized_relative_path}"

    # Get file/directory suggestions for a given path prefix
    def _get_file_suggestions(self, prefix: str) -> list[dict]:
        try:
            parsed = _parse_path_prefix(prefix)
            raw_prefix = parsed["rawPrefix"]
            is_at_prefix = parsed["isAtPrefix"]
            is_quoted_prefix = parsed["isQuotedPrefix"]
            expanded_prefix = raw_prefix

            # Handle home directory expansion
            if expanded_prefix.startswith("~"):
                expanded_prefix = self._expand_home_path(expanded_prefix)

            is_root_prefix = (
                raw_prefix == ""
                or raw_prefix == "./"
                or raw_prefix == "../"
                or raw_prefix == "~"
                or raw_prefix == "~/"
                or raw_prefix == "/"
                or (is_at_prefix and raw_prefix == "")
            )

            if is_root_prefix:
                # Complete from specified position
                if raw_prefix.startswith("~") or expanded_prefix.startswith("/"):
                    search_dir = expanded_prefix
                else:
                    search_dir = os.path.join(self._base_path, expanded_prefix)
                search_prefix = ""
            elif raw_prefix.endswith("/"):
                # If prefix ends with /, show contents of that directory
                if raw_prefix.startswith("~") or expanded_prefix.startswith("/"):
                    search_dir = expanded_prefix
                else:
                    search_dir = os.path.join(self._base_path, expanded_prefix)
                search_prefix = ""
            else:
                # Split into directory and file prefix
                directory = os.path.dirname(expanded_prefix) or "."
                file = os.path.basename(expanded_prefix)
                if raw_prefix.startswith("~") or expanded_prefix.startswith("/"):
                    search_dir = directory
                else:
                    search_dir = os.path.join(self._base_path, directory)
                search_prefix = file

            with os.scandir(search_dir) as scanner:
                entries = list(scanner)
            suggestions: list[dict] = []

            for entry in entries:
                if not entry.name.lower().startswith(search_prefix.lower()):
                    continue

                # Check if entry is a directory (or a symlink pointing to a directory)
                is_directory = entry.is_dir(follow_symlinks=False)
                if not is_directory and entry.is_symlink():
                    try:
                        is_directory = os.path.isdir(os.path.join(search_dir, entry.name))
                    except OSError:
                        # Broken symlink or permission error - treat as file
                        pass

                name = entry.name
                display_prefix = raw_prefix

                if display_prefix.endswith("/"):
                    # If prefix ends with /, append entry to the prefix
                    relative_path = display_prefix + name
                elif "/" in display_prefix or "\\" in display_prefix:
                    # Preserve ~/ format for home directory paths
                    if display_prefix.startswith("~/"):
                        home_relative_dir = display_prefix[2:]
                        directory = os.path.dirname(home_relative_dir) or "."
                        relative_path = f"~/{name if directory == '.' else os.path.join(directory, name)}"
                    elif display_prefix.startswith("/"):
                        # Absolute path - construct properly
                        directory = os.path.dirname(display_prefix)
                        if directory == "/":
                            relative_path = f"/{name}"
                        else:
                            relative_path = f"{directory}/{name}"
                    else:
                        directory = os.path.dirname(display_prefix) or "."
                        relative_path = name if directory == "." else os.path.join(directory, name)
                        # node's path.join normalizes away ./ prefix, preserve it
                        if display_prefix.startswith("./") and not relative_path.startswith("./"):
                            relative_path = f"./{relative_path}"
                else:
                    # For standalone entries, preserve ~/ if original prefix was ~/
                    if display_prefix.startswith("~"):
                        relative_path = f"~/{name}"
                    else:
                        relative_path = name

                relative_path = _to_display_path(relative_path)
                path_value = f"{relative_path}/" if is_directory else relative_path
                value = _build_completion_value(
                    path_value,
                    is_directory=is_directory,
                    is_at_prefix=is_at_prefix,
                    is_quoted_prefix=is_quoted_prefix,
                )

                suggestions.append({"value": value, "label": name + ("/" if is_directory else "")})

            # Sort directories first, then alphabetically (approximates JS
            # localeCompare with a case-insensitive comparison).
            suggestions.sort(key=lambda s: (not s["value"].endswith("/"), s["label"].lower(), s["label"]))

            return suggestions
        except OSError:
            # Directory doesn't exist or not accessible
            return []

    # Score an entry against the query (higher = better match)
    # is_directory adds bonus to prioritize folders
    def _score_entry(self, file_path: str, query: str, is_directory: bool) -> int:
        # node's basename ignores trailing separators ("src/" -> "src");
        # os.path.basename returns "" instead.
        file_name = os.path.basename(file_path.rstrip("/")) if file_path != "/" else "/"
        lower_file_name = file_name.lower()
        lower_query = query.lower()

        score = 0

        # Exact filename match (highest)
        if lower_file_name == lower_query:
            score = 100
        # Filename starts with query
        elif lower_file_name.startswith(lower_query):
            score = 80
        # Substring match in filename
        elif lower_query in lower_file_name:
            score = 50
        # Substring match in full path
        elif lower_query in file_path.lower():
            score = 30

        # Directories get a bonus to appear first
        if is_directory and score > 0:
            score += 10

        return score

    # Fuzzy file search using fd (fast, respects .gitignore)
    async def _get_fuzzy_file_suggestions(self, query: str, *, is_quoted_prefix: bool, signal) -> list[dict]:
        if not self._fd_path or signal.cancelled:
            return []

        try:
            scoped_query = self._resolve_scoped_fuzzy_query(query)
            fd_base_dir = scoped_query["baseDir"] if scoped_query else self._base_path
            fd_query = scoped_query["query"] if scoped_query else query
            entries = await _walk_directory_with_fd(fd_base_dir, self._fd_path, fd_query, 100, signal)
            if signal.cancelled:
                return []

            scored_entries = [
                {**entry, "score": self._score_entry(entry["path"], fd_query, entry["isDirectory"]) if fd_query else 1}
                for entry in entries
            ]
            scored_entries = [entry for entry in scored_entries if entry["score"] > 0]

            scored_entries.sort(key=lambda entry: -entry["score"])
            top_entries = scored_entries[:20]

            suggestions: list[dict] = []
            for entry in top_entries:
                entry_path = entry["path"]
                is_directory = entry["isDirectory"]
                path_without_slash = entry_path[:-1] if is_directory else entry_path
                display_path = (
                    self._scoped_path_for_display(scoped_query["displayBase"], path_without_slash)
                    if scoped_query
                    else path_without_slash
                )
                entry_name = os.path.basename(path_without_slash)
                completion_path = f"{display_path}/" if is_directory else display_path
                value = _build_completion_value(
                    completion_path,
                    is_directory=is_directory,
                    is_at_prefix=True,
                    is_quoted_prefix=is_quoted_prefix,
                )

                suggestions.append(
                    {"value": value, "label": entry_name + ("/" if is_directory else ""), "description": display_path}
                )

            return suggestions
        except Exception:
            return []

    # Check if we should trigger file completion (called on Tab key)
    def should_trigger_file_completion(self, lines: list[str], cursor_line: int, cursor_col: int) -> bool:
        current_line = lines[cursor_line] if cursor_line < len(lines) else ""
        text_before_cursor = current_line[:cursor_col]

        # Don't trigger if we're typing a slash command at the start of the line
        return not (text_before_cursor.strip().startswith("/") and " " not in text_before_cursor.strip())
