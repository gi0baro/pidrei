"""Shared helpers for pidrei_tui tests."""

import contextlib
import os


@contextlib.contextmanager
def env_var(name, value):
    original = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original
