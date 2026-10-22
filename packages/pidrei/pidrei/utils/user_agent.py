"""Mirror of pi coding-agent src/utils/pi-user-agent.ts."""

from pidrei_ai.utils.identity import format_user_agent


def get_pidrei_user_agent(version: str) -> str:
    """pidrei's `User-Agent` for its own network requests; the same detailed form pidrei-ai sends to providers."""
    return format_user_agent({"user_agent": "detailed", "version": version})
