"""Mirror of pi coding-agent src/modes/interactive/components/footer.ts."""

import os
import re

from pidrei_tui import truncate_to_width, visible_width

from ....core.experimental import are_experimental_features_enabled
from ....core.usage_totals import add_usage_to_totals, create_usage_totals
from ..theme import theme


_STATUS_WS_RE = re.compile(r"[\r\n\t]")
_MULTI_SPACE_RE = re.compile(r" +")


def _sanitize_status_text(text: str) -> str:
    """Sanitize text for display in a single-line status.

    Removes newlines, tabs, carriage returns, and other control characters.
    """
    return _MULTI_SPACE_RE.sub(" ", _STATUS_WS_RE.sub(" ", text)).strip()


def format_tokens(count: int) -> str:
    """Format token counts for compact footer display."""
    if count < 1000:
        return str(count)
    if count < 10000:
        return f"{count / 1000:.1f}k"
    if count < 1000000:
        return f"{round(count / 1000)}k"
    if count < 10000000:
        return f"{count / 1000000:.1f}M"
    return f"{round(count / 1000000)}M"


def format_cwd_for_footer(cwd: str, home: str | None) -> str:
    if not home:
        return cwd

    resolved_cwd = os.path.abspath(cwd)
    resolved_home = os.path.abspath(home)
    relative_to_home = os.path.relpath(resolved_cwd, resolved_home)
    if relative_to_home == ".":
        # node path.relative gives "" for identical paths
        relative_to_home = ""
    is_inside_home = relative_to_home == "" or (
        relative_to_home != ".."
        and not relative_to_home.startswith(".." + os.sep)
        and not os.path.isabs(relative_to_home)
    )

    if not is_inside_home:
        return cwd
    return "~" if relative_to_home == "" else f"~{os.sep}{relative_to_home}"


class FooterComponent:
    """Footer showing pwd, token stats, and context usage.

    Computes token/context stats from the session; gets git branch and
    extension statuses from the footer data provider.
    """

    def __init__(self, session, footer_data) -> None:
        self._auto_compact_enabled = True
        self._session = session
        self._footer_data = footer_data

    def set_session(self, session) -> None:
        self._session = session

    def set_auto_compact_enabled(self, enabled: bool) -> None:
        self._auto_compact_enabled = enabled

    def invalidate(self) -> None:
        # No-op: git branch is cached/invalidated by provider
        pass

    def dispose(self) -> None:
        # Git watcher cleanup handled by provider
        pass

    def render(self, width: int) -> list:
        state = self._session.state

        # Calculate cumulative usage from ALL session entries (not just
        # post-compaction messages)
        usage_totals = create_usage_totals()
        latest_cache_hit_rate: float | None = None

        for entry in self._session.session_manager.get_entries():
            message = entry.get("message")
            if entry.get("type") == "message" and getattr(message, "role", None) == "assistant":
                add_usage_to_totals(usage_totals, message.usage)

                latest_prompt_tokens = message.usage.input + message.usage.cache_read + message.usage.cache_write
                latest_cache_hit_rate = (
                    (message.usage.cache_read / latest_prompt_tokens) * 100 if latest_prompt_tokens > 0 else None
                )
            elif entry.get("type") == "message" and getattr(message, "role", None) == "toolResult" and message.usage:
                add_usage_to_totals(usage_totals, message.usage)
            elif entry.get("type") in ("branch_summary", "compaction") and entry.get("usage"):
                add_usage_to_totals(usage_totals, entry["usage"])

        # Calculate context usage from session (handles compaction
        # correctly). After compaction, tokens are unknown until the next LLM
        # response.
        context_usage = self._session.get_context_usage()
        if context_usage is not None:
            context_window = context_usage.context_window
        elif state.model is not None:
            context_window = state.model.context_window
        else:
            context_window = 0
        context_percent_value = (
            context_usage.percent if context_usage is not None and context_usage.percent is not None else 0
        )
        # JS: `contextUsage?.percent !== null` — "?" only for an explicit
        # null percent (post-compaction unknown); no usage at all shows 0.0
        if context_usage is not None and context_usage.percent is None:
            context_percent = "?"
        else:
            context_percent = f"{context_percent_value:.1f}"

        # Replace home directory with ~
        pwd = format_cwd_for_footer(self._session.session_manager.get_cwd(), os.environ.get("HOME"))

        # Add git branch if available
        branch = self._footer_data.get_git_branch()
        if branch:
            pwd = f"{pwd} ({branch})"

        # Add session name if set
        session_name = self._session.session_manager.get_session_name()
        if session_name:
            pwd = f"{pwd} • {session_name}"

        # Build stats line
        stats_parts = []
        if usage_totals.input:
            stats_parts.append(f"↑{format_tokens(usage_totals.input)}")
        if usage_totals.output:
            stats_parts.append(f"↓{format_tokens(usage_totals.output)}")
        if usage_totals.cache_read:
            stats_parts.append(f"R{format_tokens(usage_totals.cache_read)}")
        if usage_totals.cache_write:
            stats_parts.append(f"W{format_tokens(usage_totals.cache_write)}")
        if (usage_totals.cache_read > 0 or usage_totals.cache_write > 0) and latest_cache_hit_rate is not None:
            stats_parts.append(f"CH{latest_cache_hit_rate:.1f}%")

        # Kimi Coding is subscription-backed despite using API-key
        # authentication.
        using_subscription = (
            state.model.provider == "kimi-coding" or self._session.model_runtime.is_using_oauth(state.model.provider)
            if state.model is not None
            else False
        )
        if usage_totals.cost or using_subscription:
            cost_str = f"${usage_totals.cost:.3f}{' (sub)' if using_subscription else ''}"
            stats_parts.append(cost_str)

        # Colorize context percentage based on usage
        auto_indicator = " (auto)" if self._auto_compact_enabled else ""
        if context_percent == "?":
            context_percent_display = f"?/{format_tokens(context_window)}{auto_indicator}"
        else:
            context_percent_display = f"{context_percent}%/{format_tokens(context_window)}{auto_indicator}"
        if context_percent_value > 90:
            context_percent_str = theme.fg("error", context_percent_display)
        elif context_percent_value > 70:
            context_percent_str = theme.fg("warning", context_percent_display)
        else:
            context_percent_str = context_percent_display
        stats_parts.append(context_percent_str)
        if are_experimental_features_enabled():
            stats_parts.append(f"{theme.fg('dim', '•')} {theme.bold(theme.fg('warning', 'xp'))}")

        stats_left = " ".join(stats_parts)

        # Add model name on the right side, plus thinking level if model
        # supports it
        model_name = state.model.id if state.model is not None else "no-model"

        stats_left_width = visible_width(stats_left)

        # If stats_left is too wide, truncate it
        if stats_left_width > width:
            stats_left = truncate_to_width(stats_left, width, "...")
            stats_left_width = visible_width(stats_left)

        # Calculate available space for padding (minimum 2 spaces between
        # stats and model)
        min_padding = 2

        # Add thinking level indicator if model supports reasoning
        right_side_without_provider = model_name
        if state.model is not None and state.model.reasoning:
            thinking_level = state.thinking_level or "off"
            if thinking_level == "off":
                right_side_without_provider = f"{model_name} • thinking off"
            else:
                right_side_without_provider = f"{model_name} • {thinking_level}"

        # Prepend the provider in parentheses if there are multiple providers
        # and there's enough room
        right_side = right_side_without_provider
        if self._footer_data.get_available_provider_count() > 1 and state.model is not None:
            right_side = f"({state.model.provider}) {right_side_without_provider}"
            if stats_left_width + min_padding + visible_width(right_side) > width:
                # Too wide, fall back
                right_side = right_side_without_provider

        right_side_width = visible_width(right_side)
        total_needed = stats_left_width + min_padding + right_side_width

        if total_needed <= width:
            # Both fit - add padding to right-align model
            padding = " " * (width - stats_left_width - right_side_width)
            stats_line = stats_left + padding + right_side
        else:
            # Need to truncate right side
            available_for_right = width - stats_left_width - min_padding
            if available_for_right > 0:
                truncated_right = truncate_to_width(right_side, available_for_right, "")
                truncated_right_width = visible_width(truncated_right)
                padding = " " * max(0, width - stats_left_width - truncated_right_width)
                stats_line = stats_left + padding + truncated_right
            else:
                # Not enough space for right side at all
                stats_line = stats_left

        # Apply dim to each part separately. stats_left may contain color
        # codes (for context %) that end with a reset, which would clear an
        # outer dim wrapper. So we dim the parts before and after the colored
        # section independently.
        dim_stats_left = theme.fg("dim", stats_left)
        remainder = stats_line[len(stats_left) :]  # padding + right side
        dim_remainder = theme.fg("dim", remainder)

        pwd_line = truncate_to_width(theme.fg("dim", pwd), width, theme.fg("dim", "..."))
        lines = [pwd_line, dim_stats_left + dim_remainder]

        # Add extension statuses on a single line, sorted by key
        # alphabetically
        extension_statuses = self._footer_data.get_extension_statuses()
        if extension_statuses:
            sorted_statuses = [
                _sanitize_status_text(text)
                for _, text in sorted(extension_statuses.items(), key=lambda kv: (kv[0].lower(), kv[0]))
            ]
            status_line = " ".join(sorted_statuses)
            # Truncate to terminal width with dim ellipsis for consistency
            # with footer style
            lines.append(truncate_to_width(status_line, width, theme.fg("dim", "...")))

        return lines
