"""pidrei-specific: jieba loads in the background on the first Han run.

pi segments words with `Intl.Segmenter`, whose dictionary ships with the
engine. jieba's costs ~70MB and a third of a second of file I/O, so it loads
only once Han text is first segmented, off the runtime, and until then Han
runs keep the UAX #29 split (one ideograph per word).
"""

import threading

import pytest
import tonio.colored as tonio

from pidrei_tui import utils
from pidrei_tui.utils import get_word_segmenter


def _words(text: str) -> list[str]:
    return [segment["segment"] for segment in get_word_segmenter().segment(text)]


@pytest.mark.tonio
async def test_han_segmentation_loads_jieba_once_in_the_background(monkeypatch):
    loaded_cut = utils._jieba_cut
    assert loaded_cut is not None, "the conftest loads jieba for the session"
    release = threading.Event()
    published = threading.Event()
    second_load = threading.Event()
    loads = 0

    def initialize() -> None:
        nonlocal loads
        loads += 1
        if loads > 1:
            second_load.set()
        release.wait(5)
        utils._jieba_cut = loaded_cut
        published.set()

    monkeypatch.setattr(utils, "_jieba_cut", None)
    monkeypatch.setattr(utils, "_jieba_load_started", False)
    monkeypatch.setattr(utils, "_initialize_jieba", initialize)

    assert _words("hello world") == ["hello", " ", "world"]
    assert not utils._jieba_load_started, "text without Han must not start the load"

    # Not loaded yet: the UAX #29 split, while the load starts once.
    assert _words("你好世界") == ["你", "好", "世", "界"]
    assert _words("你好世界 test") == ["你", "好", "世", "界", " ", "test"]

    release.set()
    await tonio.spawn_blocking(published.wait, 5)
    assert published.is_set()
    assert _words("你好世界 test") == ["你好", "世界", " ", "test"]

    # Bounded: a second load would have been started by the second Han run.
    await tonio.spawn_blocking(second_load.wait, 0.2)
    assert not second_load.is_set(), "the load started more than once"
