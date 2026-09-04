"""Shared test helpers for the agent harness helper layer."""

import tempfile


def create_temp_dir() -> str:
    return tempfile.mkdtemp(prefix="pidrei-agent-session-")
